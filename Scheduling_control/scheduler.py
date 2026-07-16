"""
Scheduling and admission-control layer.

A scheduler decides WHICH pending job runs next and WHICH partition (server)
it runs on. Because the policy lives here, swapping the scheduler changes how
tenants share the GPU -- e.g. strict arrival order (FCFS) versus priority
order versus fair-share service balancing.

Writing a new algorithm
-----------------------
Drop a `.py` file into the top-level `schedulers/` directory. In it, subclass
`Scheduler` and implement `select`, and give the class a `NAME` attribute --
that string is what shows up in the dashboard's scheduler dropdown. The loader
(see `Scheduling_control.loader`) discovers every `Scheduler` subclass in that
directory at startup, and the dashboard's "Reload schedulers" button rescans
it without restarting the app. See `schedulers/fcfs.py`,
`schedulers/priority.py` and `schedulers/fair_share.py` for worked examples.

Contract (queue-aware selection)
--------------------------------
Each dispatch pass, the simulation engine repeatedly calls:

    scheduler.select(pending_jobs, partitions) -> (job, partition_index) | None

`pending_jobs` is the list of jobs currently in the global queue, in arrival
order, that have not yet been offered this pass. Each is a Workload.Job, so
`job.tenant_id`, `job.wtype` (light/medium/heavy), `job.size`, `job.iters`,
`job.mem_mb`, `job.priority` and `job.arrival_time` are all available to
inform the choice. `partitions` is a list of dicts, one per simulated MIG
partition (= server slot), ordered by index:

    {"index": int, "free": bool, "tenant": str | None, "queue_len": int,
     "cap_mb": int, "used_mb": int, "headroom_mb": int,
     "admit_headroom_mb": float, "throttled": bool, "share": float}

    index             -- the partition's slot number (= position in the list)
    free              -- True if its server is idle (nothing assigned to it)
    tenant            -- tenant_id of the job currently executing there, else None
    queue_len         -- jobs assigned to that server (waiting + running)
    cap_mb            -- the partition's VRAM cap (logical, policy-enforced)
    used_mb           -- declared mem_mb reserved by jobs placed-and-not-finished
    headroom_mb       -- cap_mb - used_mb
    admit_headroom_mb -- how much MORE declared demand the admission controller
                         would accept right now (threshold-scaled cap, minus
                         usage, minus the safety margin). The most useful
                         field for placement: a job with
                         mem_mb <= admit_headroom_mb is expected to be admitted.
    throttled         -- the partition's cooperative throttle flag is raised
    share             -- the partition's compute share (fraction of the GPU)

Return `(job, partition_index)` to propose placing that pending job on that
partition. The proposal is *validated by the admission controller* (the §3.6
four-step check) before it is committed:

    ADMIT  -- the job is placed (it queues at the server if one is busy);
    QUEUE / THROTTLE_RUNNING
           -- the job stays in the global queue; it is not offered again this
              pass, but is back in `pending_jobs` on the next dispatch pass;
    REJECT -- the job's demand can never fit any partition; it is removed
              from the queue permanently and recorded as rejected.

After each proposal the engine calls `select` again with the remaining jobs
and a refreshed partition view, until `select` returns None (nothing more to
place this pass) or the pending list is exhausted. Because a held job is only
excluded for the *current* pass, ordering policies are queue-order-level: a
job the admission controller holds does not block later jobs from being
offered, but it keeps its place and is offered first again on every
subsequent pass (no starvation, and -- per the report's limitation ii -- no
kernel-level preemption of anything already running).

Completion feedback
-------------------
Two optional no-op hooks let stateful policies (e.g. fair-share) track
consumed service without touching engine internals:

    scheduler.job_placed(job, partition_index)            -- placement committed
    scheduler.job_finished(job, partition_index, metrics) -- execution finished
                                                             (metrics = Job.run()'s
                                                             measured dict)

`job_finished` is called from a server thread, so a scheduler that mutates
shared state in it should lock. Both hooks must be fast and non-blocking.

Legacy contract
---------------
Older plugins that implement `select_server(job, servers) -> int | None`
still work: the base class's default `select` adapts them by offering the
pending jobs one at a time in arrival order and proposing the first placement
the plugin accepts. New schedulers should implement `select` directly.
"""


def admit_headroom(partition):
    """The partition's remaining admission capacity in MB, from the view dict.

    Falls back to raw ``headroom_mb`` when the engine did not attach an
    admission controller (e.g. bare unit tests)."""
    hr = partition.get("admit_headroom_mb")
    return hr if hr is not None else partition.get("headroom_mb", 0)


def first_fit(job, partitions):
    """Index of the first partition (in list order) whose current admission
    headroom can take the job's declared demand.

    When none currently fits, the roomiest partition's index is returned
    instead of None, so the admission controller still gets to rule on the
    placement -- QUEUE (retry on a later pass once reservations release) or
    REJECT (the demand can never fit anywhere). Never returns None for a
    non-empty partition list."""
    for p in partitions:
        if job.mem_mb <= admit_headroom(p):
            return p["index"]
    return max(partitions, key=admit_headroom)["index"]


class Scheduler:
    """Base class -- subclass and implement select (or legacy select_server).

    A subclass placed in the top-level ``schedulers/`` directory is picked up
    automatically by the loader; set a ``NAME`` class attribute to control its
    label in the UI (falls back to the class name)."""

    NAME = None

    def select(self, pending_jobs, partitions):
        """Propose the next placement: (job, partition_index), or None to end
        the dispatch pass.

        Default implementation adapts legacy ``select_server`` plugins:
        pending jobs are offered one at a time in the order given (arrival
        order), and the first job the plugin returns a valid index for
        becomes the proposal."""
        for job in pending_jobs:
            idx = self.select_server(job, partitions)
            if idx is not None and 0 <= idx < len(partitions):
                return job, idx
        return None

    def select_server(self, job, servers):
        """Legacy per-job contract: index of the server `job` should run on,
        or None to leave it in the global queue. Only called through the
        default ``select`` above."""
        raise NotImplementedError

    # ---- optional completion-feedback hooks (no-ops by default) ----------
    def job_placed(self, job, partition_index):
        """A proposal this scheduler made was admitted and committed."""

    def job_finished(self, job, partition_index, metrics):
        """A previously placed job finished executing. `metrics` is the
        measured dict from Job.run(). Called from a server thread."""
