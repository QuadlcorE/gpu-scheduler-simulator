"""Shortest-queue placement: send each job to the least-loaded server.

Example scheduler plugin that makes use of the enriched server view -- it
always places a job immediately on the server with the smallest ``queue_len``
(ties broken by index), queuing behind the running job when every server is
busy rather than holding the job in the global queue. See
Scheduling_control/scheduler.py for the contract.
"""

from Scheduling_control.scheduler import Scheduler


class ShortestQueueScheduler(Scheduler):
    """Place each job on whichever server currently carries the least work."""

    NAME = "Shortest queue (least-loaded GPU)"

    def select_server(self, job, servers):
        if not servers:
            return None
        best = min(servers, key=lambda s: (s.get("queue_len", 0), s["index"]))
        return best["index"]
