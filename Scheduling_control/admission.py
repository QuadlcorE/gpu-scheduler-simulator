"""
Admission controller -- the report's §3.6 four-step algorithm.

Before a job placement is committed, the admission controller decides whether
the target partition can safely take the job's declared VRAM demand. The
decision is a direct transcription of §3.6:

    1. current   = the partition's reservation-ledger usage, cross-checked
                   against a live NVML reading of the physical device;
    2. projected = current + job.mem_mb;
    3. add the configured safety margin;
    4. admit iff projected + margin <= vram_cap_mb * threshold_pct.

Both ``threshold_pct`` (§3.8 iv) and ``safety_margin_mb`` are experiment
parameters, passed in per run rather than hard-coded.

Outcomes
--------
``decide(job, partition)`` returns one of four module-level constants:

    ADMIT            -- the placement is safe; the caller commits it (reserves
                        the job on the partition's ledger and dispatches it).
    QUEUE            -- over threshold right now; hold the job in the global
                        queue and offer it again on a later dispatch pass.
    REJECT           -- the job's demand can never fit *any* partition, even
                        one that is completely empty. Recorded permanently --
                        these are the placements that would otherwise have
                        been guaranteed OOMs (feeds metric §3.9 v).
    THROTTLE_RUNNING -- the partition is already over its threshold with a
                        heavy job running: the job queues AND the partition's
                        cooperative throttle flag is raised, slowing the
                        running tenant until usage drops back under the
                        threshold ("throttled if already running", §3.6 --
                        queue-order-level control only, no kernel preemption).

``decide`` never mutates the ledger -- reservation stays the caller's move on
ADMIT -- but it does raise the partition's throttle flag on THROTTLE_RUNNING.
Call :meth:`AdmissionController.reconsider` after a job finishes so a flag the
controller raised is released once the partition is back under threshold
(flags raised statically by ``apply_compute_shares`` are left alone).

Every decision is appended to an admission log (job, partition, projected MB,
threshold, outcome, timestamp) -- exportable via :func:`write_admission_log_csv`
and the source of the UI's "scheduling decisions / throttling events" feed.
"""

import csv
import threading
import time

# ---------------------------------------------------------------------------
# Decision outcomes. Plain strings so they drop straight into log rows / CSVs.
# ---------------------------------------------------------------------------
ADMIT = "admit"
QUEUE = "queue"
REJECT = "reject"
THROTTLE_RUNNING = "throttle_running"

OUTCOMES = (ADMIT, QUEUE, REJECT, THROTTLE_RUNNING)

# Column order for the admission-log CSV export. Identity of the decision
# first, then the §3.6 arithmetic that produced it, then the outcome.
ADMISSION_LOG_FIELDS = [
    "timestamp", "job_id", "tenant", "wtype", "mem_mb", "priority",
    "partition", "current_mb", "projected_mb", "margin_mb",
    "threshold_pct", "cap_mb", "limit_mb", "nvml_free_mb", "outcome",
]

# A running job counts as "heavy" for the THROTTLE_RUNNING test when its
# reservation alone exceeds this fraction of the partition's cap. Cap-relative
# rather than tied to the workload presets, so resizing partitions or
# recalibrating the mem ranges keeps the test meaningful.
HEAVY_FRACTION = 0.5


def _nvml_free_mb(device_index=0):
    """Live free VRAM of the physical device in MB via NVML, or None when no
    GPU / no NVML is available (CPU dev runs). Never raises."""
    try:
        from GPUMonitoring.monitoring import GPU_AVAILABLE
        if GPU_AVAILABLE:
            import pynvml
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            return pynvml.nvmlDeviceGetMemoryInfo(h).free // (1024 * 1024)
    except Exception:
        pass
    return None


class AdmissionController:
    """Gatekeeper between the scheduler's proposed placement and the commit.

    Parameters
    ----------
    threshold_pct : float
        Admission threshold as a fraction of the partition cap (§3.8 iv).
        Values > 1 are treated as percentages (90 -> 0.9) so the UI's
        "threshold %" field can be passed through directly.
    safety_margin_mb : int
        Fixed headroom added on top of the projected usage before the
        threshold comparison (§3.6 step 3).
    partitions : list[Partition] | None
        The full partition set, used by the REJECT test ("can this demand
        ever fit *any* partition?"). When None, only the partition passed to
        each ``decide`` call is considered.
    use_nvml : bool
        Cross-check the ledger against a live NVML reading of the physical
        device (step 1). Disable for deterministic unit tests / CPU runs.
    clock : callable
        Zero-arg callable giving the log timestamp. Pass the engine's
        ``_now`` so admission-log times align with sim_time; defaults to
        ``time.time``.

    Thread safety: decisions are expected from the single dispatch thread,
    but the log/counters are lock-protected so the UI thread may snapshot
    them mid-run.
    """

    def __init__(self, threshold_pct=0.9, safety_margin_mb=64, partitions=None,
                 use_nvml=True, device_index=0, clock=None):
        if threshold_pct > 1.0:
            threshold_pct = threshold_pct / 100.0  # accept "90" for 90 %
        if not 0.0 < threshold_pct <= 1.0:
            raise ValueError(f"threshold_pct must be in (0, 1]: {threshold_pct}")
        if safety_margin_mb < 0:
            raise ValueError(f"safety_margin_mb must be >= 0: {safety_margin_mb}")
        self.threshold_pct = threshold_pct
        self.safety_margin_mb = int(safety_margin_mb)
        self.partitions = list(partitions) if partitions is not None else None
        self.use_nvml = use_nvml
        self.device_index = device_index
        self.clock = clock if clock is not None else time.time

        self._lock = threading.Lock()
        # Every decision, in order: one dict per ADMISSION_LOG_FIELDS row.
        self.log = []
        # Throttle raises/releases, for the UI's throttling-events feed. Rows:
        # {"timestamp", "partition", "event": "raised"|"released", "job_id"}.
        self.throttle_log = []
        self.counters = {o: 0 for o in OUTCOMES}
        # Partitions whose throttle flag *this controller* set (was clear when
        # we raised it). Only these are cleared by reconsider(); a flag set
        # statically by apply_compute_shares is never touched.
        self._throttled = set()

    # ---- §3.6 arithmetic ----------------------------------------------------
    def limit_mb(self, partition):
        """The admission ceiling for one partition: cap * threshold."""
        return partition.vram_cap_mb * self.threshold_pct

    def _fits_anywhere(self, job, fallback_partition):
        """Could this job ever be admitted -- i.e. does mem_mb + margin fit
        under at least one partition's threshold when that partition is empty?
        False means the job is structurally unplaceable => REJECT."""
        candidates = self.partitions if self.partitions else [fallback_partition]
        best = max(self.limit_mb(p) for p in candidates)
        return job.mem_mb + self.safety_margin_mb <= best

    # ---- the decision --------------------------------------------------------
    def decide(self, job, partition):
        """Run the four-step §3.6 check for placing `job` on `partition`.

        Returns one of ADMIT / QUEUE / REJECT / THROTTLE_RUNNING (see module
        docstring). Appends one row to the admission log either way. Does not
        reserve -- on ADMIT the caller commits with ``partition.reserve(job)``.
        """
        # Step 1: current usage from the reservation ledger...
        current = partition.used_mb
        # Step 2: project the job's declared demand on top.
        projected = current + job.mem_mb
        # Step 3: add the safety margin.
        needed = projected + self.safety_margin_mb
        # Step 4: compare against the threshold-scaled cap.
        limit = self.limit_mb(partition)

        nvml_free = None
        if not self._fits_anywhere(job, partition):
            # Demand can never fit any partition cap: admitting it anywhere
            # would be a guaranteed OOM, so it is rejected outright.
            outcome = REJECT
        elif needed <= limit:
            # ...cross-checked against the physical device (step 1): even if
            # the ledger says the slice has room, a live NVML reading may show
            # less actual free VRAM (allocator overhead, external processes).
            # Trust the more pessimistic of the two and queue instead.
            if self.use_nvml:
                nvml_free = _nvml_free_mb(self.device_index)
            if (nvml_free is not None
                    and job.mem_mb + self.safety_margin_mb > nvml_free):
                outcome = QUEUE
            else:
                outcome = ADMIT
        elif (current > limit
              and partition.largest_reservation_mb
                  > partition.vram_cap_mb * HEAVY_FRACTION):
            # The partition is already past its threshold with a heavy job
            # holding it: queue this job and throttle the running tenant so
            # the slice drains faster ("throttled if already running").
            self._raise_throttle(partition, job)
            outcome = THROTTLE_RUNNING
        else:
            # Over threshold right now, but a later pass can succeed once
            # running jobs release their reservations.
            outcome = QUEUE

        row = {
            "timestamp": self.clock(),
            "job_id": job.job_id,
            "tenant": job.tenant_id,
            "wtype": job.wtype,
            "mem_mb": job.mem_mb,
            "priority": job.priority,
            "partition": partition.index,
            "current_mb": current,
            "projected_mb": projected,
            "margin_mb": self.safety_margin_mb,
            "threshold_pct": self.threshold_pct,
            "cap_mb": partition.vram_cap_mb,
            "limit_mb": limit,
            "nvml_free_mb": nvml_free,
            "outcome": outcome,
        }
        with self._lock:
            self.log.append(row)
            self.counters[outcome] += 1
        return outcome

    # ---- throttle lifecycle ---------------------------------------------------
    def _raise_throttle(self, partition, job):
        """Raise the partition's cooperative throttle flag (if it was clear)
        and log the event. A flag that is already set -- e.g. statically by
        apply_compute_shares -- is left as-is and NOT recorded as ours, so
        reconsider() will never clear it."""
        with self._lock:
            if not partition.throttle.is_set():
                partition.throttle.set()
                self._throttled.add(partition.index)
                self.throttle_log.append({
                    "timestamp": self.clock(),
                    "partition": partition.index,
                    "event": "raised",
                    "job_id": job.job_id,
                })

    def reconsider(self, partition):
        """Release a throttle this controller raised, once the partition has
        drained back under its threshold. Call after a job completes (its
        reservation has been released). No-op for partitions we never
        throttled -- notably ones share-throttled at run start."""
        with self._lock:
            if partition.index not in self._throttled:
                return
            if partition.used_mb <= self.limit_mb(partition):
                partition.throttle.clear()
                self._throttled.discard(partition.index)
                self.throttle_log.append({
                    "timestamp": self.clock(),
                    "partition": partition.index,
                    "event": "released",
                    "job_id": None,
                })

    # ---- introspection for snapshots / UI ---------------------------------------
    def snapshot(self, recent=12):
        """Plain-dict view for the engine's state snapshots: outcome counters,
        the partitions currently admission-throttled, and the `recent` most
        recent decision / throttle-event rows (copies), so the UI can render a
        live "scheduling decisions / throttling events" feed without touching
        the full log mid-run."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "decisions": len(self.log),
                "throttled_partitions": sorted(self._throttled),
                "recent": [dict(r) for r in self.log[-recent:]],
                "recent_throttle": [dict(r) for r in self.throttle_log[-recent:]],
            }


def write_admission_log_csv(path, log):
    """Write an admission log (list of decision rows) to `path` as CSV, in
    decision order. Returns the number of rows written."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ADMISSION_LOG_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for r in log:
            writer.writerow({k: r.get(k) for k in ADMISSION_LOG_FIELDS})
    return len(log)
