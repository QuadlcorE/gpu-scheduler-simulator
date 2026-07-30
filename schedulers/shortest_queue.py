"""Shortest-queue placement: send each job to the least-loaded server.

Example scheduler plugin that makes use of the enriched server view -- it fills
never-used GPUs first (so a burst fans out across every GPU before any is
reused), then places on the server with the smallest ``queue_len`` (ties broken
by index), queuing behind the running job when every server is busy rather than
holding the job in the global queue. See Scheduling_control/scheduler.py for
the contract.
"""

from Scheduling_control.scheduler import Scheduler, ServerSpread


class ShortestQueueScheduler(Scheduler):
    """Place each job on a never-used GPU first, else the least-loaded one."""

    NAME = "Shortest queue (least-loaded GPU)"

    def __init__(self):
        self._spread = ServerSpread()

    def job_placed(self, job, partition_index):
        self._spread.note_placed(partition_index)

    def select_server(self, job, servers):
        if not servers:
            return None
        return self._spread.place(job, servers)
