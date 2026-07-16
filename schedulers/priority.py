"""Priority scheduling -- the report's §3.5.2 policy.

The highest-priority pending job is always served first (each job's priority
is drawn from the seeded trace RNG; higher value = more urgent), with ties
broken by arrival order -- so among equal priorities the policy degrades to
FCFS. Placement is first-fit over the partitions, exactly as the FCFS
baseline, so the *only* behavioral difference between the two policies is the
serving order.

Higher priority preempts *queue order only*: an urgent arrival jumps ahead of
everything still waiting in the global queue, but never interrupts a job that
is already executing (the report's limitation ii -- no kernel-level
preemption).

See Scheduling_control/scheduler.py for the contract.
"""

from Scheduling_control.scheduler import Scheduler, first_fit


class PriorityScheduler(Scheduler):
    """Serve the highest-priority pending job first (ties by arrival order)."""

    NAME = "Priority (highest first)"

    def select(self, pending_jobs, partitions):
        if not pending_jobs or not partitions:
            return None
        # Highest priority first; equal priorities fall back to arrival order
        # (job_id is assigned in arrival order, breaking exact-time ties).
        job = min(pending_jobs,
                  key=lambda j: (-j.priority, j.arrival_time, j.job_id))
        return job, first_fit(job, partitions)
