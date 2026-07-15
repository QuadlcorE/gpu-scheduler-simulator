"""Shortest-queue placement: send each job to the least-loaded server.

Example scheduler plugin that makes use of the enriched server view -- it
picks the free server with the smallest ``queue_len`` (ties broken by index).
See Scheduling_control/scheduler.py for the contract.
"""

from Scheduling_control.scheduler import Scheduler


class ShortestQueueScheduler(Scheduler):
    """Place each job on the free server carrying the least work."""

    NAME = "Shortest queue (least-loaded GPU)"

    def select_server(self, job, servers):
        free = [s for s in servers if s["free"]]
        if not free:
            return None
        best = min(free, key=lambda s: (s.get("queue_len", 0), s["index"]))
        return best["index"]
