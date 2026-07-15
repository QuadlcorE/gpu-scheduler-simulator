"""
Scheduling and admission-control layer.

A scheduler decides WHICH server (GPU) a pending job runs on. Because the
placement policy lives here, swapping the scheduler changes how tenants map
to GPUs -- e.g. every job to any free GPU (pooled) versus each tenant pinned
to its own GPU (partitioned).

Contract
--------
The simulation engine calls:

    scheduler.select_server(job, servers) -> int | None

where `servers` is a list of dicts describing every server slot:

    {"index": int, "free": bool, "tenant": str | None}

Return the index of the server the job should run on, or None to leave the
job queued (e.g. the server this tenant is pinned to is currently busy).
"""


class Scheduler:
    """Base class -- subclass and implement select_server."""

    def select_server(self, job, servers):
        raise NotImplementedError


class PooledScheduler(Scheduler):
    """Any job runs on any free server (first-come-first-served placement)."""

    def select_server(self, job, servers):
        for s in servers:
            if s["free"]:
                return s["index"]
        return None


class PinnedScheduler(Scheduler):
    """Each tenant is assigned to a fixed server; its jobs only run there.

    Tenants are mapped to servers round-robin on first sight, so with more
    tenants than servers several tenants share a server and time-slice it."""

    def __init__(self):
        self._tenant_server = {}  # tenant_id -> server index

    def _server_for(self, tenant_id, num_servers):
        if tenant_id not in self._tenant_server:
            self._tenant_server[tenant_id] = len(self._tenant_server) % num_servers
        return self._tenant_server[tenant_id]

    def select_server(self, job, servers):
        idx = self._server_for(job.tenant_id, len(servers))
        return idx if servers[idx]["free"] else None
