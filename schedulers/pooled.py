"""Pooled placement: any job runs on any free server.

Draws from the single global queue and places each job on an idle server,
holding it in the queue when every server is busy (rather than queuing behind a
running job). Among the idle servers it fills those not yet used this run first,
so a burst of jobs fans out across every GPU before any is reused instead of
piling onto the low-index ones.

Example scheduler plugin. See Scheduling_control/scheduler.py for the contract.
"""

from Scheduling_control.scheduler import Scheduler, ServerSpread


class PooledScheduler(Scheduler):
    """Any job runs on any free server, filling never-used GPUs first."""

    NAME = "Pooled (any free GPU)"

    def __init__(self):
        self._spread = ServerSpread()

    def job_placed(self, job, partition_index):
        self._spread.note_placed(partition_index)

    def select_server(self, job, servers):
        # A free server preferring never-used ones; None (hold in the global
        # queue) when all are busy.
        return self._spread.place_free(servers)
