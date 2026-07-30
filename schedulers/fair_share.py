"""Fair-share scheduling -- the report's §3.5.3 policy.

Tracks how much service each tenant has *consumed* and always serves the
pending job of the most under-served tenant, so no tenant can starve the
others by flooding the queue.

Consumed service is tallied per tenant from the completion hook, in the two
resource dimensions the report names:

    * GPU-seconds   -- measured execution wall time of the tenant's finished
                       jobs;
    * VRAM GB-seconds -- each finished job's declared memory footprint (GB)
                       multiplied by its execution time, so holding a large
                       slice for a long time costs as much fairness credit as
                       burning compute.

A tenant's service score is the sum of the two (both grow with runtime and
are of comparable magnitude on the ~GB-scale workloads, so neither dimension
drowns out the other). Selection order:

    1. lowest service score first (the most under-served tenant);
    2. ties by fewest in-flight (placed-and-not-finished) jobs -- this is
       what spreads the very first placements across tenants before any
       completions have accrued, instead of letting one tenant take every
       slot;
    3. then oldest arrival, so within one tenant jobs still run FCFS.

Placement spreads load across partitions: a never-used GPU is filled first (so
a tenant's jobs fan out across every slice before any is reused), then among
the partitions that can currently admit the job the least-loaded one (shortest
queue, then most admission headroom) is proposed, instead of piling onto a
single slice.

See Scheduling_control/scheduler.py for the contract.
"""

import threading
from collections import defaultdict

from Scheduling_control.scheduler import Scheduler, ServerSpread


class FairShareScheduler(Scheduler):
    """Serve the pending job of the tenant with the least consumed service."""

    NAME = "Fair share (per-tenant service)"

    def __init__(self):
        # All tallies are keyed by tenant_id. job_finished arrives from server
        # threads while select runs on the dispatch thread, hence the lock.
        self._lock = threading.Lock()
        self._gpu_s = defaultdict(float)      # completed GPU-seconds
        self._vram_gb_s = defaultdict(float)  # completed VRAM GB-seconds
        self._inflight = defaultdict(int)     # placed-and-not-finished jobs
        # Fills never-used GPUs first, then spreads by load (single queue).
        self._spread = ServerSpread()

    # ---- completion feedback ------------------------------------------------
    def job_placed(self, job, partition_index):
        self._spread.note_placed(partition_index)
        with self._lock:
            self._inflight[job.tenant_id] += 1

    def job_finished(self, job, partition_index, metrics):
        run_s = metrics.get("wall_s", 0.0)
        with self._lock:
            self._inflight[job.tenant_id] = max(
                0, self._inflight[job.tenant_id] - 1)
            self._gpu_s[job.tenant_id] += run_s
            self._vram_gb_s[job.tenant_id] += (job.mem_mb / 1024.0) * run_s

    def service_score(self, tenant_id):
        """Consumed service: GPU-seconds + VRAM GB-seconds. Lower = more
        under-served = served sooner."""
        with self._lock:
            return self._gpu_s[tenant_id] + self._vram_gb_s[tenant_id]

    # ---- selection -----------------------------------------------------------
    def select(self, pending_jobs, partitions):
        if not pending_jobs or not partitions:
            return None
        with self._lock:
            job = min(pending_jobs, key=lambda j: (
                self._gpu_s[j.tenant_id] + self._vram_gb_s[j.tenant_id],
                self._inflight[j.tenant_id],
                j.arrival_time, j.job_id,
            ))
        # Spread across partitions: fill a never-used GPU first, else the
        # least-loaded partition that can admit the job now (shortest queue,
        # then most admission headroom). If none currently fits, the roomiest
        # is proposed anyway so the admission controller rules on it (QUEUE to
        # retry later, or REJECT outright).
        return job, self._spread.place(job, partitions)
