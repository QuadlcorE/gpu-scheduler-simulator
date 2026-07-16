"""FCFS -- first-come-first-served, the report's §3.5.1 baseline policy.

Jobs are served strictly in arrival order: the oldest pending job is always
proposed first, on the first partition that can currently admit its declared
VRAM demand (first-fit by partition index). No other job property -- priority,
size, tenant -- influences the order, which is what makes FCFS the baseline
the other two policies are compared against.

When no partition can currently take the job, it is proposed on the roomiest
one anyway so the admission controller rules on it: QUEUE holds it (it stays
at the head of the queue and is retried first on every later pass), REJECT
removes a demand that could never fit. Jobs behind a held job may still be
offered later in the same pass -- ordering is queue-order-level only, with no
preemption of anything running.

See Scheduling_control/scheduler.py for the contract.
"""

from Scheduling_control.scheduler import Scheduler, first_fit


class FCFSScheduler(Scheduler):
    """Serve pending jobs strictly in arrival order (first-come-first-served)."""

    NAME = "FCFS (first-come-first-served)"

    def select(self, pending_jobs, partitions):
        if not pending_jobs or not partitions:
            return None
        # Oldest arrival first; job_id breaks the (rare) exact-time tie, since
        # trace ids are assigned in arrival order.
        job = min(pending_jobs, key=lambda j: (j.arrival_time, j.job_id))
        return job, first_fit(job, partitions)
