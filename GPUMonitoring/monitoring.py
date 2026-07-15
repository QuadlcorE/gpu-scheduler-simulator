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