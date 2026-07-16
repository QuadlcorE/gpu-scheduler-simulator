import threading
import time

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
except Exception:
    GPU_AVAILABLE = False
    GPU_COUNT = 0


def _safe(fn, default=None):
    """Run an NVML call and swallow errors from unsupported metrics on some cards."""
    try:
        return fn()
    except Exception:
        return default


def _gpu_processes(handle, with_names=True) -> list:
    """Per-process GPU memory usage and active PIDs (report §3.3.2 items ii/iv)
    via nvmlDeviceGetComputeRunningProcesses. Each entry:

        {"pid": int, "mem_mb": int | None, "name": str | None}

    mem_mb is None when the driver cannot attribute per-process memory (e.g.
    Windows WDDM mode reports N/A). ``with_names=False`` skips the per-PID
    name lookup -- the telemetry sampler uses that to keep samples cheap."""
    procs = _safe(lambda: pynvml.nvmlDeviceGetComputeRunningProcesses(handle),
                  default=[])
    out = []
    for p in procs or []:
        mem = getattr(p, "usedGpuMemory", None)
        name = _safe(lambda: pynvml.nvmlSystemGetProcessName(p.pid)) if with_names else None
        out.append({
            "pid": p.pid,
            # NVML reports bytes; a negative/None value means "not available".
            "mem_mb": mem // (1024 * 1024) if isinstance(mem, int) and mem >= 0 else None,
            "name": name.decode() if isinstance(name, bytes) else name,
        })
    return out


def _gpu_stats(index: int) -> dict:
    h = pynvml.nvmlDeviceGetHandleByIndex(index)

    # We are making use of safe to avoid exceptions from unsupported metrics on some cards. For example, the fan speed metric is not supported on some GPUs.
    util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
    mem = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
    temp = _safe(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    power = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(h))  # milliwatts
    power_limit = _safe(lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h))  # milliwatts
    fan = _safe(lambda: pynvml.nvmlDeviceGetFanSpeed(h))
    clock_gfx = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS))
    clock_mem = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
    name = _safe(lambda: pynvml.nvmlDeviceGetName(h))
    processes = _gpu_processes(h)

    return {
        "index": index,
        "name": name.decode() if isinstance(name, bytes) else name,
        "gpu_util": util.gpu if util else None,
        "mem_util": util.memory if util else None,
        "mem_used_mb": mem.used // (1024 * 1024) if mem else None,
        "mem_total_mb": mem.total // (1024 * 1024) if mem else None,
        "mem_percent": round(mem.used / mem.total * 100, 1) if mem else None,
        "temp_c": temp,
        "power_w": round(power / 1000, 1) if power is not None else None,
        "power_limit_w": round(power_limit / 1000, 1) if power_limit is not None else None,
        "fan_percent": fan,
        "clock_graphics_mhz": clock_gfx,
        "clock_mem_mhz": clock_mem,
        # Per-process memory usage / active PIDs (§3.3.2 ii and iv).
        "processes": processes,
        "proc_count": len(processes),
    }


def feed_hardware() -> dict:
    stats = {
        "cpu_total": psutil.cpu_percent(interval=None),
        "mem": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "gpus": [],
    }

    if GPU_AVAILABLE:
        for i in range(GPU_COUNT):
            gpu_stat = _safe(lambda: _gpu_stats(i))
            if gpu_stat:
                stats["gpus"].append(gpu_stat)

    return stats


# ---------------------------------------------------------------------------
# Run-scoped telemetry sampler (report §3.3.2).
#
# A background thread that, for the duration of one simulation run, samples
# GPU utilization + VRAM at a fixed interval into a telemetry log. This is
# what makes "average VRAM utilization" (§3.9 ii) and "partition-level
# resource stability" (§3.9 vii) computable from *measured* NVML data rather
# than the reservation ledger alone -- the ledger columns are sampled
# alongside so the two can be compared on the same timeline.
# ---------------------------------------------------------------------------

# Default sampling cadence. Runs last seconds-to-minutes, so 4 samples/s gives
# a usable timeline without NVML polling becoming measurable overhead.
TELEMETRY_INTERVAL_S = 0.25


class TelemetrySampler(threading.Thread):
    """Fixed-interval GPU telemetry sampler, scoped to one run.

    Usage: construct, ``start()`` when the run's clock starts, ``stop()`` when
    it ends; ``samples`` is then the run's telemetry log. Each sample row::

        {
            "t": float,              # seconds since run start (or `clock`)
            "gpu_util": int | None,  # NVML device utilization %
            "mem_used_mb": int | None,   # NVML device-wide VRAM used
            "mem_total_mb": int | None,  # NVML device-wide VRAM total
            "proc_count": int | None,    # active compute processes
            "proc_mem_mb": int | None,   # their summed attributed memory
            "part<i>_used_mb": int,      # per-partition ledger usage (one
        }                                #   column per partition supplied)

    The NVML fields are None on CPU-only machines (no GPU / no NVML) -- the
    partition ledger columns are still recorded, so partition-stability
    metrics stay computable in dev runs.

    Parameters
    ----------
    interval_s : float
        Sampling period.
    device_index : int
        Physical GPU to sample (the single-GPU setup uses 0).
    partitions : list[Partition] | None
        Simulated MIG partitions whose ledger usage is sampled per tick.
    clock : callable | None
        Zero-arg callable giving each sample's timestamp. Pass the engine's
        ``_now`` so telemetry times align with sim_time; defaults to seconds
        since the sampler started.
    """

    def __init__(self, interval_s=TELEMETRY_INTERVAL_S, device_index=0,
                 partitions=None, clock=None):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.device_index = device_index
        self.partitions = list(partitions) if partitions else []
        self._clock = clock
        self._t0 = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._samples = []

    def _now(self):
        if self._clock is not None:
            return self._clock()
        return time.perf_counter() - (self._t0 if self._t0 is not None else 0.0)

    def sample_once(self) -> dict:
        """Take one telemetry sample (also usable standalone, without the
        thread running)."""
        row = {
            "t": round(self._now(), 3),
            "gpu_util": None,
            "mem_used_mb": None,
            "mem_total_mb": None,
            "proc_count": None,
            "proc_mem_mb": None,
        }
        if GPU_AVAILABLE:
            h = _safe(lambda: pynvml.nvmlDeviceGetHandleByIndex(self.device_index))
            if h is not None:
                util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
                mem = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
                procs = _gpu_processes(h, with_names=False)
                if util is not None:
                    row["gpu_util"] = util.gpu
                if mem is not None:
                    row["mem_used_mb"] = mem.used // (1024 * 1024)
                    row["mem_total_mb"] = mem.total // (1024 * 1024)
                row["proc_count"] = len(procs)
                row["proc_mem_mb"] = sum(p["mem_mb"] for p in procs
                                         if p["mem_mb"] is not None)
        # Ledger view of the same instant: what admission control *believes*
        # each partition is using, for comparison against the measured columns.
        for p in self.partitions:
            row[f"part{p.index}_used_mb"] = p.used_mb
        return row

    def run(self):
        self._t0 = time.perf_counter()
        while True:
            row = self.sample_once()
            with self._lock:
                self._samples.append(row)
            # wait() doubles as the interval sleep and the stop signal, so a
            # stop lands within one interval instead of a full sleep.
            if self._stop_evt.wait(self.interval_s):
                return

    def stop(self, timeout=2.0):
        """Stop sampling and wait for the thread to exit."""
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)

    @property
    def samples(self) -> list:
        """Snapshot copy of the telemetry log (safe to call mid-run)."""
        with self._lock:
            return list(self._samples)