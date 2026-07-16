"""Pinned placement: each tenant is bound to a fixed server.

Example scheduler plugin. See Scheduling_control/scheduler.py for the contract.
"""

from Scheduling_control.scheduler import Scheduler


class PinnedScheduler(Scheduler):
    """Each tenant is assigned to a fixed server; its jobs only run there.

    Tenants are mapped to servers round-robin on first sight, so with more
    tenants than servers several tenants share a server. A tenant's jobs always
    queue at its own server (waiting in that server's queue when it is busy)
    rather than sitting in the global queue."""

    NAME = "Pinned (tenant to GPU)"

    def __init__(self):
        self._tenant_server = {}  # tenant_id -> server index

    def _server_for(self, tenant_id, num_servers):
        if tenant_id not in self._tenant_server:
            self._tenant_server[tenant_id] = len(self._tenant_server) % num_servers
        return self._tenant_server[tenant_id]

    def select_server(self, job, servers):
        # Always return the pinned server: the job queues there if it is busy.
        return self._server_for(job.tenant_id, len(servers))
