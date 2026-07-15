"""
Scheduling and admission-control layer.

A scheduler decides WHICH server (GPU) a pending job runs on. Because the
placement policy lives here, swapping the scheduler changes how tenants map
to GPUs -- e.g. every job to any free GPU (pooled) versus each tenant pinned
to its own GPU (partitioned).

Writing a new algorithm
-----------------------
Drop a `.py` file into the top-level `schedulers/` directory. In it, subclass
`Scheduler` and implement `select_server`, and give the class a `NAME`
attribute -- that string is what shows up in the dashboard's scheduler
dropdown. The loader (see `Scheduling_control.loader`) discovers every
`Scheduler` subclass in that directory at startup, and the dashboard's
"Reload schedulers" button rescans it without restarting the app. See
`schedulers/pooled.py` and `schedulers/pinned.py` for worked examples.

Contract
--------
The simulation engine calls:

    scheduler.select_server(job, servers) -> int | None

`job` is the pending Workload.Job (so `job.tenant_id`, `job.wtype`
(light/medium/heavy), `job.size`, `job.iters` are all available to inform
placement). `servers` is a list of dicts, one per server slot, ordered by
index:

    {"index": int, "free": bool, "tenant": str | None, "queue_len": int}

    index     -- the server's slot number (also its position in the list)
    free      -- True if the server can take a job right now
    tenant    -- tenant_id of the job currently occupying the server, else None
    queue_len -- how many jobs are waiting on/running at that server

Return the index of the server the job should run on, or None to leave the
job queued (e.g. the server this tenant is pinned to is currently busy).
"""


class Scheduler:
    """Base class -- subclass and implement select_server.

    A subclass placed in the top-level ``schedulers/`` directory is picked up
    automatically by the loader; set a ``NAME`` class attribute to control its
    label in the UI (falls back to the class name)."""

    NAME = None

    def select_server(self, job, servers):
        raise NotImplementedError
