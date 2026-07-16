"""
Simulation layer -- SimPy hybrid engine (report §3.1-§3.2).

The engine is a hybrid of discrete-event simulation and real execution:
arrivals, queueing, dispatch, and admission decisions are modelled as SimPy
processes/events on a ``simpy.rt.RealtimeEnvironment`` (the event clock is
synchronized to the wall clock, factor = 1), while job execution happens on
real server threads running PyTorch. The two halves have to share one clock
because real GPU work takes real wall-clock time -- a purely virtual clock
would race ahead of the executions it is supposed to be timing.

Three actors drive a run:

    * the **arrival process** (SimPy) yields ``env.timeout()`` for each gap in
      the pre-generated trace and releases the next job into the global queue;
    * the **dispatch process** (SimPy) wakes on arrival/completion events and
      runs the scheduler -> admission -> placement pipeline;
    * the **server threads** (real) execute placed jobs on PyTorch and signal
      completion back to the dispatch process.

SimPy's event calendar is not thread-safe, so the server threads never touch
``env`` directly: a completion drops a token onto a thread-safe queue, and the
dispatch process -- which waits on its wake event with a short poll timeout --
drains that queue and re-enters dispatch. Arrivals happen inside the SimPy
thread, so they wake the dispatcher directly by succeeding its wake event.

The engine models `capacity` simulated MIG partitions (simulation.partition),
each served by a worker thread bound to a torch device -- on the single-GPU
setup all of them execute on the same physical cuda:0, and the partition
supplies the policy state: a VRAM reservation ledger (declared mem_mb of jobs
placed-and-not-finished) and a cooperative throttle flag that soft-enforces
its compute share. Tenants generate random matmul workloads; each dispatch
pass runs the scheduler -> admission -> placement pipeline: the scheduler
(Scheduling_control) proposes (job, partition) placements from the global
queue, the admission controller (the report's §3.6 four-step check) validates
each proposal, and admitted jobs are handed to the partition's server thread,
which actually executes the workload on PyTorch and reports measured metrics.
A proposal the controller holds (QUEUE / THROTTLE_RUNNING) stays in the
global queue for a later pass; a REJECTed job -- one whose declared demand
could never fit any partition -- is removed and recorded as rejected.

"sim_time" is seconds elapsed since the run started, which the realtime
environment keeps equal (to within scheduling jitter) to the SimPy ``env.now``.
Lifecycle timestamps are still *measured* with a monotonic clock -- they are
real observations of real executions, not simulated quantities. State
snapshots pushed onto event_q keep the same schema the GUI already consumes:

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

import simpy
import simpy.rt
import torch

from Scheduling_control.admission import ADMIT, REJECT, AdmissionController
from Scheduling_control.scheduler import Scheduler
from Workload.workload import DEFAULT_TENANTS, build_experiment
from simulation.partition import apply_compute_shares, build_partitions
from simulation.results import summarize


@dataclass
class SimEvent:
    sim_time: float
    kind: str  # "job_start" | "job_end"
    payload: dict = field(default_factory=dict)


# How long the dispatch process sleeps between wake checks. This bounds two
# latencies: how quickly a completion signalled from a server thread reaches
# the dispatcher, and how quickly a user stop is honoured. It is also the
# heartbeat-snapshot cadence that keeps live wait times ticking in the UI.
DISPATCH_POLL_S = 0.05


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
    def __init__(self, index, device, partition, stop_flag, on_start, on_done):
        super().__init__(daemon=True)
        self.index = index
        self.device = device
        # The logical partition this server executes for. On the single-GPU
        # setup every server's `device` is the same physical cuda:0; the
        # partition supplies the policy state (VRAM ledger, throttle flag).
        self.partition = partition
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
            # The partition's throttle flag enforces its compute share (and,
            # later, admission-controller throttling) cooperatively.
            metrics = job.run(self.device, throttle=self.partition.throttle)
            self.on_done(self.index, job, metrics)


# ---------------------------------------------------------------------------
# Main simulator -- connects the scheduler, replays the job trace, and assigns
# each job to a server (possibly a busy one, where it queues) as it arrives.
# ---------------------------------------------------------------------------

class MultiTenantSimulation:
    def __init__(self, event_q, stop_flag, capacity, tenants, jobs, scheduler=None,
                 partitions=None, admission=None):
        self.event_q, self.stop_flag = event_q, stop_flag
        # Each server slot is the execution arm of one simulated MIG partition
        # (simulation.partition). Default: `capacity` equal slices of the
        # physical VRAM with equal compute shares. Under-fair-share partitions
        # get their throttle flag raised here, once, for the whole run.
        self.partitions = partitions if partitions is not None else build_partitions(capacity)
        apply_compute_shares(self.partitions)
        capacity = len(self.partitions)
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

        # Bridge from the real server threads into the SimPy world: SimPy's
        # event calendar is not thread-safe, so a completing server thread
        # never touches env -- it puts a token here, and the dispatch process
        # drains the queue on its next wake. The wake event itself is created
        # fresh each iteration inside _dispatch_proc (no env exists yet here).
        self._completions = queue.Queue()
        self._wake = None

        # Admission controller: validates every placement the scheduler
        # proposes (§3.6). Callers may inject a configured one (custom
        # threshold/margin, NVML disabled for tests); by default it gates
        # against this run's partition set with log timestamps on sim_time.
        self.admission = admission if admission is not None else AdmissionController(
            partitions=self.partitions, clock=self._now)

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
                "rejected": 0,
                "global_wait_total": 0.0,
                "server_wait_total": 0.0,
            }
            for t in self.tenants
        }

        self.servers = [
            Server(i, _device_for(i), self.partitions[i], stop_flag,
                   self._on_job_start, self._on_job_done)
            for i in range(capacity)
        ]
        self.device_desc = str(self.servers[0].device) if self.servers else "cpu"

    def _now(self):
        return time.perf_counter() - self._t0

    # ---- SimPy processes ----------------------------------------------------
    def _arrival_proc(self, env):
        """SimPy arrival process: replay the trace, yielding ``env.timeout()``
        for each inter-arrival gap and releasing the job into the global
        queue at its planned arrival time. The realtime environment turns
        those timeouts into wall-clock waits, so arrivals land on schedule.

        A user stop simply abandons the remainder of the trace: the dispatch
        process is what ends the run, so this process just returns."""
        for job in self.trace:
            delay = job.arrival_time - env.now
            if delay > 0:
                yield env.timeout(delay)
            if self.stop_flag.is_set():
                return
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
            # Wake the dispatcher at this same sim-instant (same-thread, so
            # succeeding its event directly is safe), so global-queue wait
            # isn't just poll latency.
            self._notify_dispatch()

    def _notify_dispatch(self):
        """Succeed the dispatch process's wake event. Only callable from the
        SimPy thread (i.e. from other processes such as arrivals) -- foreign
        threads must go through the _completions queue instead."""
        if self._wake is not None and not self._wake.triggered:
            self._wake.succeed()

    def _dispatch_proc(self, env):
        """SimPy dispatch process -- the run's driver. Each iteration drains
        completion tokens signalled by the server threads, runs one dispatch
        pass (scheduler -> admission -> placement), and then waits for the
        next arrival wake / completion token / heartbeat tick. The run ends
        when every trace job has finished (or the user stops)."""
        while not self.stop_flag.is_set():
            # Absorb completion signals from the server threads. Bookkeeping
            # already happened in _on_job_done (lock-protected); the token is
            # purely a "something freed up, dispatch again" wake signal.
            while True:
                try:
                    self._completions.get_nowait()
                except queue.Empty:
                    break
            self._dispatch()
            if self._is_complete():
                return
            # Heartbeat snapshot keeps waiting jobs' live wait times ticking
            # in the UI even when nothing starts or finishes this interval.
            self._snapshot()
            # Sleep until an arrival wakes us or the poll interval elapses
            # (the poll bounds completion-signal latency and stop latency).
            self._wake = env.event()
            yield self._wake | env.timeout(DISPATCH_POLL_S)

    # ---- placement: scheduler proposes, admission validates, engine commits ----
    def _partition_view(self):
        """One dict per partition for the scheduler's `select` contract (see
        Scheduling_control/scheduler.py). Call with self._lock held."""
        view = []
        for i, p in enumerate(self.partitions):
            used = p.used_mb
            view.append({
                "index": i,
                # "free" = its server is idle; queue_len is its real load --
                # jobs waiting in its inbox plus the one running.
                "free": self.server_load[i] == 0,
                "tenant": self.owner.get(i),
                "queue_len": self.server_load[i],
                "cap_mb": p.vram_cap_mb,
                "used_mb": used,
                "headroom_mb": p.vram_cap_mb - used,
                # How much MORE declared demand admission would accept right
                # now: threshold-scaled cap, minus usage, minus safety margin.
                "admit_headroom_mb": (self.admission.limit_mb(p) - used
                                      - self.admission.safety_margin_mb),
                "throttled": p.throttle.is_set(),
                "share": p.compute_share,
            })
        return view

    def _remove_pending(self, job):
        """Drop `job` from the global queue by identity, preserving order of
        the rest. Call with self._lock held."""
        for k, j in enumerate(self.pending):
            if j is job:
                del self.pending[k]
                return

    def _commit(self, job, idx):
        """Commit an ADMITted placement: reserve the job's declared demand on
        the partition's ledger (held until completion) and hand it to the
        server, where it queues if the server is busy. Call with lock held."""
        self._remove_pending(job)
        self.server_load[idx] += 1
        self.partitions[idx].reserve(job)
        rec = self.active_jobs.get(job.job_id)
        if rec is not None:
            rec["state"] = "server-queue"
            rec["gpu"] = idx
            rec["assign_time"] = self._now()  # leaves the global queue
        self.servers[idx].submit(job)
        self.scheduler.job_placed(job, idx)

    def _reject(self, job):
        """Finalize a REJECTed job: its declared demand can never fit any
        partition, so it leaves the queue permanently and is recorded as a
        rejected outcome (feeds the OOM-avoidance narrative -- these are the
        placements that would otherwise have been guaranteed OOMs). Counts
        toward run completion. Call with self._lock held."""
        self._remove_pending(job)
        now = self._now()
        rec = self.active_jobs.pop(job.job_id, None)
        submit = rec["submit_time"] if rec is not None else now
        self.stats[job.tenant_id]["rejected"] += 1
        self.results.append({
            "job_id": job.job_id,
            "tenant": job.tenant_id,
            "wtype": job.wtype,
            "size": job.size,
            "iters": job.iters,
            "arrival_s": job.arrival_time,
            "global_wait_s": now - submit,
            "server_wait_s": 0.0,
            "run_s": 0.0,
            "gpu": None,
            "start_s": None,
            "end_s": now,
            "mem_mb": job.mem_mb,
            "priority": job.priority,
            "admission": REJECT,
            "oom": False,
            "throttled": False,
        })
        self.completed += 1

    def _dispatch(self):
        """One dispatch pass: repeatedly ask the scheduler for its next
        (job, partition) proposal and validate it with the admission
        controller. ADMIT commits the placement; REJECT removes the job for
        good; QUEUE / THROTTLE_RUNNING leave it in the global queue but stop
        offering it this pass (it is offered again on the next one)."""
        changed = False
        with self._lock:
            if not self.pending:
                return
            # Jobs still on offer this pass, in global-queue (arrival) order.
            # A job leaves this list once proposed, whatever the outcome, so
            # the pass always terminates.
            offered = list(self.pending)
            while offered:
                proposal = self.scheduler.select(list(offered),
                                                 self._partition_view())
                if proposal is None:
                    break  # scheduler has nothing more to place this pass
                job, idx = proposal
                pos = next((k for k, j in enumerate(offered) if j is job), None)
                if pos is None or not (0 <= idx < self.capacity):
                    break  # defensive: unknown job / bad index ends the pass
                del offered[pos]
                outcome = self.admission.decide(job, self.partitions[idx])
                if outcome == ADMIT:
                    self._commit(job, idx)
                    changed = True
                elif outcome == REJECT:
                    self._reject(job)
                    changed = True
                # QUEUE / THROTTLE_RUNNING: the job stays in self.pending and
                # is retried on a later pass (triggered by the next arrival,
                # completion, or heartbeat).
        if changed:
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
        self.partitions[index].release(job)  # frees its VRAM reservation
        # The partition may drop back under its admission threshold now: let
        # the controller release a throttle flag it raised (THROTTLE_RUNNING).
        self.admission.reconsider(self.partitions[index])
        # Completion feedback for stateful policies (e.g. fair-share tallies).
        self.scheduler.job_finished(job, index, metrics)
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
                    # Admission/execution outcomes (mirrored by _reject rows).
                    "mem_mb": job.mem_mb,
                    "priority": job.priority,
                    "admission": ADMIT,
                    "oom": bool(metrics.get("oom", False)),
                    "throttled": bool(metrics.get("throttled", False)),
                })
            self.completed += 1
        self._snapshot()
        # A slot just freed: signal the SimPy dispatch process so globally-
        # queued jobs are offered a server on its next wake (bounded by
        # DISPATCH_POLL_S). This thread must not touch env directly -- SimPy's
        # event calendar is not thread-safe -- hence the queue token.
        self._completions.put(job.job_id)

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

            # Finished jobs stay in the table (the UI grays them out) instead
            # of vanishing: append their finalized rows from the results log.
            # Rejected jobs (admission REJECT) are flagged as such.
            for r in self.results:
                jobs.append({
                    "job_id": r["job_id"], "tenant": r["tenant"],
                    "wtype": r["wtype"], "size": r["size"], "iters": r["iters"],
                    "state": ("rejected" if r.get("admission") == REJECT
                              else "completed"),
                    "gpu": r["gpu"],
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
                    "rejected": s["rejected"],
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
                # Per-partition memory/throttle state (ledger-based used/cap).
                "partitions": [p.snapshot() for p in self.partitions],
                # Admission-decision counters + currently throttled partitions.
                "admission": self.admission.snapshot(),
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
                # Full decision/throttle history, so the UI can export the
                # admission-log CSV without reaching into the simulator thread.
                payload["admission_log"] = list(self.admission.log)
                payload["throttle_log"] = list(self.admission.throttle_log)
        self.event_q.put(payload)

    def run(self):
        # Warm up CUDA before the clocks start: the first kernel launch of a
        # process pays ~1s of context initialization, which starves the GIL
        # and would freeze the arrival/dispatch processes mid-run (measured:
        # arrivals land ~1s late on a cold context vs ~10ms warm). Doing it
        # here, before _t0 / env.sync(), keeps it out of every measurement.
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda").item()
            torch.cuda.synchronize()

        for s in self.servers:
            s.start()

        # The SimPy half of the hybrid: a realtime environment pins the event
        # clock to the wall clock (factor=1), because job execution is real
        # PyTorch work taking real time. strict=False, so a slow step (e.g. a
        # CPU matmul hogging cores, or a long dispatch pass) degrades to
        # "catch up" instead of raising -- the lifecycle timestamps are
        # measured with a monotonic clock anyway, not derived from env.now.
        env = simpy.rt.RealtimeEnvironment(factor=1.0, strict=False)
        env.process(self._arrival_proc(env))
        dispatcher = env.process(self._dispatch_proc(env))
        # Anchor both clocks to the same instant: sim_time 0 (_now's epoch)
        # == env.now 0 (sync() resets the environment's real-time epoch).
        self._t0 = time.perf_counter()
        env.sync()
        # The dispatch process is the run's driver: it returns when the whole
        # trace has finished, or when the user stops. Arrival events still
        # pending at that point are simply never executed.
        env.run(until=dispatcher)

        # Graceful shutdown. On natural completion every job has finished
        # (dispatcher exits on completed == total), so raising stop_flag just
        # lets the idle server threads exit their poll loops and join.
        self.stop_flag.set()
        for s in self.servers:
            s.join(timeout=2.0)

        # Final snapshot flags completion so the GUI can re-enable its controls.
        self._snapshot(done=True)


def simulation_thread_main(event_q, stop_flag, capacity, tenants=None, jobs=None,
                           scheduler=None, partitions=None, admission=None):
    if tenants is None or jobs is None:
        # Standalone/default run: build a small reproducible experiment so the
        # engine stays runnable without the UI supplying a config.
        tenants, jobs = build_experiment(
            num_tenants=len(DEFAULT_TENANTS),
            counts={"light": 3, "medium": 2, "heavy": 1},
            seed=0,
        )
    sim = MultiTenantSimulation(event_q, stop_flag, capacity, tenants, jobs, scheduler,
                                partitions=partitions, admission=admission)
    sim.run()
