"""
Simulation layer -- real execution engine.

The engine models `capacity` GPUs as `capacity` servers, each a worker thread
bound to a torch device. Tenants generate random matmul workloads; the
scheduler (Scheduling_control) decides which server each job runs on; the
server thread actually executes the workload on PyTorch and reports measured
metrics.

There is no virtual clock here -- unlike a discrete-event model, real PyTorch
work takes real wall-clock time, so "sim_time" is simply seconds elapsed since
the run started. State snapshots pushed onto event_q keep the same schema the
GUI already consumes:

    {"sim_time", "capacity", "gpu_owner", "tenant_stats", "device"}
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from Scheduling_control.scheduler import Scheduler
from Workload.workload import DEFAULT_TENANTS, build_experiment


@dataclass
class SimEvent:
    sim_time: float
    kind: str  # "job_start" | "job_end"
    payload: dict = field(default_factory=dict)


class _DefaultScheduler(Scheduler):
    """Fallback used when the caller supplies no scheduler: any free server.

    Keeps the engine runnable on its own (e.g. in tests) without depending on
    the schedulers/ plugin directory being present."""

    def select_server(self, job, servers):
        for s in servers:
            if s["free"]:
                return s["index"]
        return None


def _device_for(index: int) -> "torch.device":
    """Map a virtual server to a physical device. Each server gets its own
    CUDA device when several are present; otherwise everything runs on CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index % torch.cuda.device_count()}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Server -- one GPU slot. A worker thread that runs whatever job it is handed.
# ---------------------------------------------------------------------------

class Server(threading.Thread):
    def __init__(self, index, device, stop_flag, on_done):
        super().__init__(daemon=True)
        self.index = index
        self.device = device
        self.stop_flag = stop_flag
        self.on_done = on_done
        self.inbox = queue.Queue(maxsize=1)  # holds at most the current job

    def submit(self, job):
        self.inbox.put(job)

    def run(self):
        while not self.stop_flag.is_set():
            try:
                job = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            metrics = job.run(self.device)
            self.on_done(self.index, job, metrics)


# ---------------------------------------------------------------------------
# Main simulator -- connects the scheduler, runs each tenant's workloads, and
# assigns them to servers as they arrive.
# ---------------------------------------------------------------------------

class MultiTenantSimulation:
    def __init__(self, event_q, stop_flag, capacity, tenants, jobs, scheduler=None):
        self.event_q, self.stop_flag = event_q, stop_flag
        self.capacity = capacity
        self.tenants = list(tenants)
        # The pre-generated job trace: a finite, ordered list of jobs, each with
        # a planned arrival_time (seconds from run start). The run replays it and
        # then ends -- it is not an open-ended stream.
        self.trace = sorted(jobs, key=lambda j: j.arrival_time)
        self.total_jobs = len(self.trace)
        self.scheduler = scheduler or _DefaultScheduler()

        self._t0 = time.perf_counter()
        self._lock = threading.Lock()

        self.pending = deque()          # jobs waiting for a server
        self.free = set(range(capacity))
        self.owner = {}                 # server index -> tenant currently on it
        self.active_jobs = {}           # job_id -> live record (queued/running)
        self.completed = 0              # trace jobs finished so far
        self.stats = {
            t.tenant_id: {"queued": 0, "running": 0, "completed": 0}
            for t in self.tenants
        }

        self.servers = [
            Server(i, _device_for(i), stop_flag, self._on_job_done)
            for i in range(capacity)
        ]
        self.device_desc = str(self.servers[0].device) if self.servers else "cpu"

    def _now(self):
        return time.perf_counter() - self._t0

    # ---- replay the trace: release each job at its planned arrival time ----
    def _release_trace(self):
        for job in self.trace:
            if self._sleep_until(job.arrival_time):
                return  # stopped before this job's arrival
            with self._lock:
                self.stats[job.tenant_id]["queued"] += 1
                self.pending.append(job)
                self.active_jobs[job.job_id] = {
                    "job_id": job.job_id, "tenant": job.tenant_id,
                    "size": job.size, "iters": job.iters, "wtype": job.wtype,
                    "state": "queued", "gpu": None,
                }
            self._snapshot()

    def _sleep_until(self, target):
        """Sleep until `target` seconds after run start, in small chunks so a
        stop is honoured promptly. Return True if stopped, False on arrival."""
        while not self.stop_flag.is_set():
            remaining = target - self._now()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))
        return True

    # ---- placement: ask the scheduler where each pending job goes ----
    def _dispatch(self):
        assigned = False
        with self._lock:
            if not self.pending or not self.free:
                return
            # queue_len is 0 for an idle server and 1 for a busy one today
            # (each server runs a single job at a time); it becomes a real
            # per-server queue depth once server-side queuing lands.
            view = [
                {
                    "index": i,
                    "free": i in self.free,
                    "tenant": self.owner.get(i),
                    "queue_len": 0 if i in self.free else 1,
                }
                for i in range(self.capacity)
            ]
            still_pending = deque()
            while self.pending:
                job = self.pending.popleft()
                idx = self.scheduler.select_server(job, view)
                if idx is not None and idx in self.free:
                    self.free.discard(idx)
                    self.owner[idx] = job.tenant_id
                    self.stats[job.tenant_id]["queued"] -= 1
                    self.stats[job.tenant_id]["running"] += 1
                    view[idx]["free"] = False
                    view[idx]["tenant"] = job.tenant_id
                    view[idx]["queue_len"] = 1
                    rec = self.active_jobs.get(job.job_id)
                    if rec is not None:
                        rec["state"] = "running"
                        rec["gpu"] = idx
                    self.servers[idx].submit(job)
                    assigned = True
                else:
                    still_pending.append(job)
            self.pending = still_pending
        if assigned:
            self._snapshot()

    # ---- a server finished a job (called from the server thread) ----
    def _on_job_done(self, index, job, metrics):
        with self._lock:
            self.free.add(index)
            self.owner.pop(index, None)
            self.active_jobs.pop(job.job_id, None)
            self.stats[job.tenant_id]["running"] -= 1
            self.stats[job.tenant_id]["completed"] += 1
            self.completed += 1
        self._snapshot()

    def _is_complete(self):
        """The run is done once every trace job has finished. total_jobs is
        known up front, so a completed count reaching it means all jobs were
        released, dispatched, and executed."""
        with self._lock:
            return self.completed >= self.total_jobs

    def _snapshot(self, done=False):
        # Build the payload under the lock so the job list stays consistent
        # with owner/stats even while the release and server threads mutate them.
        with self._lock:
            payload = {
                "sim_time": self._now(),
                "capacity": self.capacity,
                "gpu_owner": dict(self.owner),
                "tenant_stats": {t: dict(s) for t, s in self.stats.items()},
                "device": self.device_desc,
                "jobs": [dict(r) for r in self.active_jobs.values()],
                "completed": self.completed,
                "total": self.total_jobs,
                "done": done,
            }
        self.event_q.put(payload)

    def run(self):
        for s in self.servers:
            s.start()
        releaser = threading.Thread(target=self._release_trace, daemon=True)
        releaser.start()

        # Run until the trace is exhausted and drained, or the user stops.
        while not self.stop_flag.is_set():
            self._dispatch()
            if self._is_complete():
                break
            time.sleep(0.02)

        # graceful shutdown: let in-flight jobs finish, then join workers
        releaser.join(timeout=1.0)
        for s in self.servers:
            s.join(timeout=2.0)

        # Final snapshot flags completion so the GUI can re-enable its controls.
        self._snapshot(done=True)


def simulation_thread_main(event_q, stop_flag, capacity, tenants=None, jobs=None,
                           scheduler=None):
    if tenants is None or jobs is None:
        # Standalone/default run: build a small reproducible experiment so the
        # engine stays runnable without the UI supplying a config.
        tenants, jobs = build_experiment(
            num_tenants=len(DEFAULT_TENANTS),
            counts={"light": 3, "medium": 2, "heavy": 1},
            seed=0,
        )
    sim = MultiTenantSimulation(event_q, stop_flag, capacity, tenants, jobs, scheduler)
    sim.run()
