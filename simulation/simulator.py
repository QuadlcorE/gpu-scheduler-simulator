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

import itertools
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from Scheduling_control.scheduler import PooledScheduler
from Workload.workload import DEFAULT_TENANTS


@dataclass
class SimEvent:
    sim_time: float
    kind: str  # "job_start" | "job_end"
    payload: dict = field(default_factory=dict)


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
    def __init__(self, event_q, stop_flag, capacity, tenants, scheduler=None):
        self.event_q, self.stop_flag = event_q, stop_flag
        self.capacity = capacity
        self.tenants = list(tenants)
        self.scheduler = scheduler or PooledScheduler()

        self._job_ids = itertools.count()
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()

        self.pending = deque()          # jobs waiting for a server
        self.free = set(range(capacity))
        self.owner = {}                 # server index -> tenant currently on it
        self.active_jobs = {}           # job_id -> live record (queued/running)
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

    # ---- each tenant produces random workloads over time ----
    def _produce(self, tenant):
        while not self.stop_flag.is_set():
            if self._interruptible_sleep(tenant.next_interarrival()):
                break
            job = tenant.make_job(next(self._job_ids), self._now())
            with self._lock:
                self.stats[tenant.tenant_id]["queued"] += 1
                self.pending.append(job)
                self.active_jobs[job.job_id] = {
                    "job_id": job.job_id, "tenant": job.tenant_id,
                    "size": job.size, "iters": job.iters,
                    "state": "queued", "gpu": None,
                }
            self._snapshot()

    def _interruptible_sleep(self, delay):
        """Sleep `delay` seconds in small chunks; return True if stopped."""
        end = time.perf_counter() + delay
        while time.perf_counter() < end:
            if self.stop_flag.is_set():
                return True
            time.sleep(min(0.05, end - time.perf_counter()))
        return False

    # ---- placement: ask the scheduler where each pending job goes ----
    def _dispatch(self):
        assigned = False
        with self._lock:
            if not self.pending or not self.free:
                return
            view = [
                {"index": i, "free": i in self.free, "tenant": self.owner.get(i)}
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
        self._snapshot()

    def _snapshot(self):
        # Build the payload under the lock so the job list stays consistent
        # with owner/stats even while producer and server threads mutate them.
        with self._lock:
            payload = {
                "sim_time": self._now(),
                "capacity": self.capacity,
                "gpu_owner": dict(self.owner),
                "tenant_stats": {t: dict(s) for t, s in self.stats.items()},
                "device": self.device_desc,
                "jobs": [dict(r) for r in self.active_jobs.values()],
            }
        self.event_q.put(payload)

    def run(self):
        for s in self.servers:
            s.start()
        producers = [
            threading.Thread(target=self._produce, args=(t,), daemon=True)
            for t in self.tenants
        ]
        for p in producers:
            p.start()

        while not self.stop_flag.is_set():
            self._dispatch()
            time.sleep(0.02)

        # graceful shutdown: let in-flight jobs finish, then join workers
        for s in self.servers:
            s.join(timeout=2.0)
        for p in producers:
            p.join(timeout=1.0)


def simulation_thread_main(event_q, stop_flag, capacity, tenants=None, scheduler=None):
    sim = MultiTenantSimulation(event_q, stop_flag, capacity,
                                tenants or DEFAULT_TENANTS, scheduler)
    sim.run()
