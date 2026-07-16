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

    {"sim_time", "capacity", "gpu_owner", "tenant_stats", "device", ...}

Queues and lifecycle timing
---------------------------
Each server owns an *unbounded* inbox, so the scheduler may place a job on a
busy server, where it waits in that server's queue behind the running job.
Every job passes through the same lifecycle, each stage timestamped so the two
kinds of waiting can be measured separately:

    submit_time -- released into the global queue (job arrives)
    assign_time -- scheduler places it on a server (leaves the global queue)
    start_time  -- that server begins executing it
    end_time    -- execution finishes

    global-queue wait = assign_time - submit_time   (waiting for placement)
    server-queue wait = start_time  - assign_time   (waiting behind other jobs)
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from Scheduling_control.scheduler import Scheduler
from Workload.workload import DEFAULT_TENANTS, build_experiment
from simulation.results import summarize


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
# Server -- one GPU slot. A worker thread that runs whatever jobs it is handed.
# Its inbox is an unbounded queue, so jobs assigned while it is busy wait here
# (FIFO) until it works its way to them.
# ---------------------------------------------------------------------------

class Server(threading.Thread):
    def __init__(self, index, device, stop_flag, on_start, on_done):
        super().__init__(daemon=True)
        self.index = index
        self.device = device
        self.stop_flag = stop_flag
        self.on_start = on_start
        self.on_done = on_done
        self.inbox = queue.Queue()  # unbounded: assigned jobs wait here in order

    def submit(self, job):
        self.inbox.put(job)

    def run(self):
        while not self.stop_flag.is_set():
            try:
                job = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            self.on_start(self.index, job)  # server-queue wait ends here
            metrics = job.run(self.device)
            self.on_done(self.index, job, metrics)


# ---------------------------------------------------------------------------
# Main simulator -- connects the scheduler, replays the job trace, and assigns
# each job to a server (possibly a busy one, where it queues) as it arrives.
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

        self.pending = deque()          # jobs in the global queue (not yet placed)
        # Per-server load: jobs assigned to a server that have not finished
        # (waiting in its inbox + the one running). 0 means the server is idle.
        self.server_load = [0] * capacity
        self.running_job = {}           # server index -> job_id currently executing
        self.owner = {}                 # server index -> tenant currently executing there
        self.active_jobs = {}           # job_id -> live record (queued/waiting/running)
        self.completed = 0              # trace jobs finished so far
        # Per-run results log: one finalized row per completed job (see
        # simulation.results). This is the record exported to CSV and folded
        # into the end-of-run summary -- what makes the run evaluable.
        self.results = []
        # Persistent per-tenant tallies. Live queued/waiting/running counts are
        # derived from active_jobs at snapshot time; completed jobs leave the
        # active set, so their finished wait times are accumulated here.
        self.stats = {
            t.tenant_id: {
                "completed": 0,
                "global_wait_total": 0.0,
                "server_wait_total": 0.0,
            }
            for t in self.tenants
        }

        self.servers = [
            Server(i, _device_for(i), stop_flag, self._on_job_start, self._on_job_done)
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
                self.pending.append(job)
                self.active_jobs[job.job_id] = {
                    "job_id": job.job_id, "tenant": job.tenant_id,
                    "size": job.size, "iters": job.iters, "wtype": job.wtype,
                    "state": "queued", "gpu": None,
                    "submit_time": self._now(),  # enters the global queue now
                    "assign_time": None,         # set when the scheduler places it
                    "start_time": None,          # set when a server begins it
                }
            self._snapshot()
            self._dispatch()  # place it now, so global-queue wait isn't just poll latency

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
            if not self.pending:
                return
            # A server is "free" when nothing is assigned to it (load 0);
            # queue_len is its real load -- jobs waiting in its inbox plus the
            # one running. The scheduler may pick a busy server to queue behind.
            view = [
                {
                    "index": i,
                    "free": self.server_load[i] == 0,
                    "tenant": self.owner.get(i),
                    "queue_len": self.server_load[i],
                }
                for i in range(self.capacity)
            ]
            still_pending = deque()
            while self.pending:
                job = self.pending.popleft()
                idx = self.scheduler.select_server(job, view)
                if idx is not None and 0 <= idx < self.capacity:
                    # Place the job on server idx; it queues there if busy.
                    self.server_load[idx] += 1
                    rec = self.active_jobs.get(job.job_id)
                    if rec is not None:
                        rec["state"] = "server-queue"
                        rec["gpu"] = idx
                        rec["assign_time"] = self._now()  # leaves the global queue
                    # Reflect the new load so later jobs in this pass see it.
                    view[idx]["free"] = False
                    view[idx]["queue_len"] = self.server_load[idx]
                    self.servers[idx].submit(job)
                    assigned = True
                else:
                    still_pending.append(job)  # scheduler holds it in the global queue
            self.pending = still_pending
        if assigned:
            self._snapshot()

    # ---- a server began executing a job (called from the server thread) ----
    def _on_job_start(self, index, job):
        with self._lock:
            rec = self.active_jobs.get(job.job_id)
            if rec is not None:
                rec["state"] = "running"
                rec["start_time"] = self._now()  # server-queue wait ends here
            self.owner[index] = job.tenant_id
            self.running_job[index] = job.job_id
        self._snapshot()

    # ---- a server finished a job (called from the server thread) ----
    def _on_job_done(self, index, job, metrics):
        with self._lock:
            self.server_load[index] -= 1
            if self.running_job.get(index) == job.job_id:
                self.running_job.pop(index, None)
                self.owner.pop(index, None)
            rec = self.active_jobs.pop(job.job_id, None)
            st = self.stats[job.tenant_id]
            st["completed"] += 1
            if rec is not None:
                submit = rec["submit_time"]
                assign = rec["assign_time"] if rec["assign_time"] is not None else submit
                start = rec["start_time"] if rec["start_time"] is not None else assign
                global_wait = assign - submit
                server_wait = start - assign
                st["global_wait_total"] += global_wait
                st["server_wait_total"] += server_wait
                # Append the finalized per-job row to the results log. arrival_s
                # is the planned trace arrival (deterministic from the seed);
                # the wait/run times are the measured outcomes for this run.
                self.results.append({
                    "job_id": job.job_id,
                    "tenant": job.tenant_id,
                    "wtype": job.wtype,
                    "size": job.size,
                    "iters": job.iters,
                    "arrival_s": job.arrival_time,
                    "global_wait_s": global_wait,
                    "server_wait_s": server_wait,
                    "run_s": metrics.get("wall_s", 0.0),
                    "gpu": index,
                    "start_s": start,
                    "end_s": self._now(),
                })
            self.completed += 1
        self._snapshot()
        # A slot just freed: offer any globally-queued jobs (schedulers that
        # decline busy servers) a server now rather than on the next heartbeat.
        self._dispatch()

    def build_summary(self):
        """Return the per-run summary over the results log (see
        simulation.results.summarize). Safe to call after run() completes."""
        with self._lock:
            results = list(self.results)
            capacity = self.capacity
        return summarize(results, capacity)

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
            now = self._now()

            # Per-job live records, with the two wait times computed at `now`:
            # a job still queued/waiting accrues wait up to the current instant,
            # while a job that has moved on carries its finalized wait.
            jobs = []
            for r in self.active_jobs.values():
                submit, assign, start = r["submit_time"], r["assign_time"], r["start_time"]
                if assign is None:
                    global_wait, server_wait = now - submit, 0.0
                else:
                    global_wait = assign - submit
                    server_wait = (now - assign) if start is None else (start - assign)
                d = dict(r)
                d["global_wait"] = global_wait
                d["server_wait"] = server_wait
                jobs.append(d)

            # Completed jobs stay in the table (the UI grays them out) instead
            # of vanishing: append their finalized rows from the results log.
            for r in self.results:
                jobs.append({
                    "job_id": r["job_id"], "tenant": r["tenant"],
                    "wtype": r["wtype"], "size": r["size"], "iters": r["iters"],
                    "state": "completed", "gpu": r["gpu"],
                    "global_wait": r["global_wait_s"],
                    "server_wait": r["server_wait_s"],
                })

            # Live per-tenant counts derived from the active records, merged with
            # the persistent completed/wait tallies (completed jobs are gone from
            # active_jobs, so their totals live in self.stats).
            live = {t: {"queued": 0, "waiting": 0, "running": 0} for t in self.stats}
            for r in self.active_jobs.values():
                bucket = {"queued": "queued", "server-queue": "waiting",
                          "running": "running"}.get(r["state"])
                if bucket:
                    live[r["tenant"]][bucket] += 1

            tenant_stats = {}
            for t, s in self.stats.items():
                done_n = s["completed"]
                tenant_stats[t] = {
                    "queued": live[t]["queued"],
                    "waiting": live[t]["waiting"],
                    "running": live[t]["running"],
                    "completed": done_n,
                    "global_wait_total": s["global_wait_total"],
                    "server_wait_total": s["server_wait_total"],
                    "avg_global_wait": s["global_wait_total"] / done_n if done_n else 0.0,
                    "avg_server_wait": s["server_wait_total"] / done_n if done_n else 0.0,
                }

            payload = {
                "sim_time": now,
                "capacity": self.capacity,
                "gpu_owner": dict(self.owner),
                "server_queue": list(self.server_load),
                "tenant_stats": tenant_stats,
                "device": self.device_desc,
                "jobs": jobs,
                "completed": self.completed,
                "total": self.total_jobs,
                "done": done,
            }
            # The final snapshot carries the whole results log and the folded
            # summary, so the UI can export the CSV and show the results view
            # without reaching back into the simulator thread.
            if done:
                payload["results"] = list(self.results)
                payload["summary"] = summarize(list(self.results), self.capacity)
        self.event_q.put(payload)

    def run(self):
        for s in self.servers:
            s.start()
        releaser = threading.Thread(target=self._release_trace, daemon=True)
        releaser.start()

        # Run until the trace is exhausted and drained, or the user stops. A
        # heartbeat snapshot each pass keeps waiting jobs' live wait times ticking
        # in the UI even when no job has started or finished this interval.
        while not self.stop_flag.is_set():
            self._dispatch()
            if self._is_complete():
                break
            self._snapshot()
            time.sleep(0.1)

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
