"""Pooled placement: any job runs on any free server.

Example scheduler plugin. See Scheduling_control/scheduler.py for the contract.
"""

from Scheduling_control.scheduler import Scheduler


class PooledScheduler(Scheduler):
    """Any job runs on any free server (first-come-first-served placement)."""

    NAME = "Pooled (any free GPU)"

    def select_server(self, job, servers):
        for s in servers:
            if s["free"]:
                return s["index"]
        return None
