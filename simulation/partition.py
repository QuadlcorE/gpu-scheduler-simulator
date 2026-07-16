"""
Simulated MIG partitions (report §3.4.1 / §3.4.3, Figure 3.2).

A :class:`Partition` is a *logical* slice of the physical GPU: a VRAM cap and a
compute share that scheduling and admission control reason about. There is no
hardware isolation (consumer cards have no MIG) -- the limits are policy-
enforced by the admission controller, exactly as the report describes:

    * **VRAM cap** -- each partition carries a *reservation ledger*: the sum of
      the declared ``mem_mb`` of every job admitted to it and not yet finished.
      ``used_mb`` / ``headroom_mb`` come from that ledger, and the admission
      controller refuses placements that would push the ledger past the cap.
    * **Compute share** -- enforced *softly* via the cooperative throttle hook
      in ``Job.run``: a partition whose share is below its fair share (1/N)
      runs its jobs with the throttle flag raised, so they yield the GPU
      between kernels. Queue-order-level control only -- no kernel preemption.

On the single-GPU setup every partition's jobs execute on the same physical
device (``cuda:0``); the partition set is what the scheduler sees as separate
"GPUs".

Resource constraints (§3.4.3): the builder enforces (i) per-partition caps and
(ii) a total-memory limit -- the sum of all caps must not exceed the physical
VRAM it is slicing.
"""

import threading
from dataclasses import dataclass, field

# Fallback physical VRAM when no NVML-visible GPU is present (CPU dev runs).
# Sized to the project's real hardware: the 4 GB single-GPU setup.
DEFAULT_TOTAL_VRAM_MB = 4096


def physical_vram_mb(device_index=0):
    """Total VRAM of the physical device in MB, via NVML; falls back to
    DEFAULT_TOTAL_VRAM_MB when no GPU (or no NVML) is available."""
    try:
        from GPUMonitoring.monitoring import GPU_AVAILABLE
        if GPU_AVAILABLE:
            import pynvml
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            return pynvml.nvmlDeviceGetMemoryInfo(h).total // (1024 * 1024)
    except Exception:
        pass
    return DEFAULT_TOTAL_VRAM_MB


@dataclass
class Partition:
    """One logical GPU slice: a VRAM cap, a compute share, and the reservation
    ledger of jobs currently admitted to it.

    The ledger maps job_id -> declared mem_mb for every job admitted and not
    yet finished; it is what the admission controller's "current usage" reads.
    All mutation is internally locked, so reserve/release may be called from
    the dispatch thread and the server threads without external locking."""

    index: int
    vram_cap_mb: int
    compute_share: float
    # job_id -> declared mem_mb, for jobs admitted-and-not-finished.
    _ledger: dict = field(default_factory=dict, repr=False)
    # Cooperative throttle flag, passed to Job.run() by this partition's
    # server thread. Raised statically when the partition's compute share is
    # under fair share (apply_compute_shares), and dynamically by the
    # admission controller's THROTTLE_RUNNING outcome.
    throttle: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- reservation ledger ------------------------------------------------
    def reserve(self, job):
        """Record `job` as admitted: its declared mem_mb counts against the cap
        until release(). Idempotent per job_id."""
        with self._lock:
            self._ledger[job.job_id] = job.mem_mb

    def release(self, job):
        """Remove a finished job's reservation. Tolerates jobs never reserved
        here (e.g. a run stopped mid-flight)."""
        with self._lock:
            self._ledger.pop(job.job_id, None)

    @property
    def used_mb(self):
        """Sum of declared demands of admitted-and-not-finished jobs."""
        with self._lock:
            return sum(self._ledger.values())

    @property
    def headroom_mb(self):
        """Cap minus ledger usage. Can go negative if something was placed
        past the cap (i.e. without admission control) -- a meaningful signal,
        so it is not clamped."""
        return self.vram_cap_mb - self.used_mb

    @property
    def admitted_jobs(self):
        """Number of jobs currently holding a reservation."""
        with self._lock:
            return len(self._ledger)

    def snapshot(self):
        """Plain-dict view for UI snapshots / logs."""
        with self._lock:
            used = sum(self._ledger.values())
            jobs = len(self._ledger)
        return {
            "index": self.index,
            "cap_mb": self.vram_cap_mb,
            "used_mb": used,
            "headroom_mb": self.vram_cap_mb - used,
            "share": self.compute_share,
            "admitted_jobs": jobs,
            "throttled": self.throttle.is_set(),
        }


def build_partitions(num_partitions=None, total_vram_mb=None, caps_mb=None,
                     shares=None):
    """Build the partition set from experiment config (§3.8 iii).

    Either give explicit per-partition ``caps_mb`` (list of MB values), or a
    ``num_partitions`` count -- the total VRAM is then split into N equal
    slices (e.g. 3 x 1365 MB on the 4 GB card). ``shares`` are the compute
    shares (default: equal, 1/N each). ``total_vram_mb`` defaults to the
    physical device's VRAM via NVML.

    Enforces the §3.4.3 resource constraints: every cap positive, and the sum
    of caps must not exceed the physical total it slices (raises ValueError).
    """
    if total_vram_mb is None:
        total_vram_mb = physical_vram_mb()

    if caps_mb is None:
        if not num_partitions or num_partitions < 1:
            raise ValueError("need num_partitions >= 1 or explicit caps_mb")
        # Equal slices, floored so the sum never exceeds the physical total.
        caps_mb = [total_vram_mb // num_partitions] * num_partitions
    else:
        caps_mb = list(caps_mb)
        if num_partitions is not None and num_partitions != len(caps_mb):
            raise ValueError(
                f"num_partitions={num_partitions} but {len(caps_mb)} caps given")

    if any(c <= 0 for c in caps_mb):
        raise ValueError(f"partition caps must be positive: {caps_mb}")
    if sum(caps_mb) > total_vram_mb:
        raise ValueError(
            f"sum of partition caps ({sum(caps_mb)} MB) exceeds physical "
            f"VRAM ({total_vram_mb} MB)")

    n = len(caps_mb)
    if shares is None:
        shares = [1.0 / n] * n
    else:
        shares = list(shares)
        if len(shares) != n:
            raise ValueError(f"{n} partitions but {len(shares)} shares given")
        if any(s <= 0 for s in shares):
            raise ValueError(f"compute shares must be positive: {shares}")

    return [Partition(index=i, vram_cap_mb=caps_mb[i], compute_share=shares[i])
            for i in range(n)]


def apply_compute_shares(partitions):
    """Soft compute-share enforcement: raise the throttle flag on every
    partition whose share is below its fair share (1/N), so its jobs run
    cooperatively throttled (share < fair share => throttled iterations)."""
    if not partitions:
        return
    fair = 1.0 / len(partitions)
    for p in partitions:
        if p.compute_share < fair - 1e-9:
            p.throttle.set()
