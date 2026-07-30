"""
Workload generation layer.

    * Job    -- one unit of work: a real PyTorch matmul (GEMM) workload.
    * Tenant -- generates random workloads (jobs) for a single tenant.

A tenant does NOT run anything itself; it only decides *when* its next job
arrives and *what* that job looks like (matrix size + iteration count). The
simulation layer assigns the job to a server (GPU) and calls Job.run(device),
which actually executes on PyTorch and returns measured metrics.
"""

import random
import time
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Workload types -- three fixed presets so a run has a mix of job "sizes".
#
# Each type pins the matrix dimension (compute grows ~N^3), a batch dimension
# (the GEMM runs on (batch, N, N) tensors), the range the iteration count is
# drawn from, and the range its declared VRAM demand is drawn from. The
# compute sizes are chosen so "heavy" visibly takes seconds while "light" is
# near-instant; the memory ranges are sized for a 4 GB single-GPU setup so a
# handful of heavy jobs can genuinely exhaust a partition. Recalibrate per
# machine if needed.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkloadType:
    name: str            # "light" / "medium" / "heavy"
    label: str           # single-letter tag for compact UI ("L" / "M" / "H")
    size: int            # NxN matrix dimension for the GEMM
    batch: int           # batch dimension: the GEMM runs on (batch, N, N)
    iter_range: tuple    # (min, max) matmul iterations, inclusive
    mem_range_mb: tuple  # (min, max) declared VRAM demand in MB, inclusive


LIGHT = WorkloadType("light", "L", 128, 1, (100, 200), (100, 250))
MEDIUM = WorkloadType("medium", "M", 384, 2, (300, 450), (400, 700))
HEAVY = WorkloadType("heavy", "H", 768, 4, (600, 800), (900, 1400))

# Lookup by name; also the canonical light->medium->heavy ordering.
WORKLOAD_TYPES = {w.name: w for w in (LIGHT, MEDIUM, HEAVY)}

# Priority is drawn per job from this inclusive range (uniform, from the seeded
# trace RNG). Higher value = more urgent; the priority scheduler serves higher
# values first. The range is deliberately small so ties are common and the
# arrival-order tiebreak stays visible in results.
PRIORITY_RANGE = (0, 9)

# Global scale applied to every job's drawn iteration count (a whole-experiment
# compute-intensity knob, §3.8 i). 1 = the workload-type ranges as-is; N makes
# every job run N x its matmul iterations, so wall-clock work scales linearly
# without touching memory demand, arrival timing, or the seeded RNG stream (the
# multiplier is applied *after* the draw, so the same seed yields the same base
# counts at any multiplier -- runs stay comparable across settings).
DEFAULT_ITER_MULTIPLIER = 1

# How long one throttled iteration sleeps before doing its matmul. This is the
# cooperative compute-share/throttle knob: a throttled job still finishes, it
# just yields the GPU between kernels (queue-order-level control only -- there
# is no kernel-level preemption).
THROTTLE_SLEEP_S = 0.05


@dataclass
class Job:
    job_id: int
    tenant_id: str
    size: int                  # NxN matrix dimension for the GEMM
    iters: int                 # number of matmul iterations
    wtype: str = "medium"      # workload type name (key into WORKLOAD_TYPES)
    batch: int = 1             # GEMM batch dimension (from the workload type)
    mem_mb: int = 0            # declared VRAM demand in MB (drawn from the trace RNG)
    priority: int = 0          # drawn from PRIORITY_RANGE; higher = more urgent
    arrival_time: float = 0.0  # planned arrival, seconds from run start (from the trace)
    metrics: dict = field(default_factory=dict)  # filled in by run()

    def run(self, device, throttle=None):
        """Execute the GEMM workload on `device` and return measured metrics.

        Runs a chained batched matrix multiply so PyTorch cannot optimize the
        loop away, then forces evaluation so the wall-clock time reflects real
        compute (important once this runs on CUDA, where ops are async).

        The job first allocates a *ballast* tensor of its declared `mem_mb` on
        the device, so its real memory footprint matches the demand the
        admission controller reasoned about -- NVML genuinely sees it, and a
        bad admission decision can genuinely OOM. A CUDA OOM is caught and
        recorded in the metrics (``oom: True``) instead of crashing the server
        thread.

        `throttle` is an optional shared flag (anything with ``is_set()``,
        e.g. ``threading.Event``). It is checked before every iteration; while
        set, the job sleeps THROTTLE_SLEEP_S between kernels, cooperatively
        yielding the GPU. The metrics record whether (and how long) the job
        was throttled."""
        ballast = a = b = None
        throttled_iters = 0
        throttle_s = 0.0
        t0 = time.perf_counter()
        try:
            # Declared footprint: mem_mb of float32 on the device (1 MB = 2^20
            # bytes / 4 bytes per element), held for the job's whole lifetime.
            if self.mem_mb > 0:
                ballast = torch.empty(
                    int(self.mem_mb) * (1024 * 1024 // 4),
                    dtype=torch.float32, device=device,
                )
            a = torch.randn(self.batch, self.size, self.size, device=device)
            b = torch.randn(self.batch, self.size, self.size, device=device)

            for _ in range(self.iters):
                if throttle is not None and throttle.is_set():
                    throttled_iters += 1
                    ts = time.perf_counter()
                    time.sleep(THROTTLE_SLEEP_S)
                    throttle_s += time.perf_counter() - ts
                a = torch.matmul(a, b)
                a = a / (a.norm() + 1e-6)  # keep values bounded across iterations
            if device.type == "cuda":
                torch.cuda.synchronize()
            oom = False
        except torch.cuda.OutOfMemoryError:
            # The declared demand didn't fit -- the exact failure a bad
            # admission policy causes. Record it as a job outcome (feeds the
            # OOM-occurrence-rate metric) rather than killing the server.
            oom = True
        wall_s = time.perf_counter() - t0

        # Release this job's tensors and hand the freed VRAM back to the driver
        # regardless of outcome (not just on OOM). PyTorch's caching allocator
        # otherwise keeps the freed blocks in its own pool, so NVML keeps
        # reporting them as in use -- and the admission controller's live NVML
        # cross-check would then watch "free" VRAM shrink monotonically across
        # jobs until every placement is QUEUEd forever and the run stalls with
        # memory apparently used up. (The reservation ledger self-heals on
        # release; it's the raw NVML reading that needs the cache flushed.)
        ballast = a = b = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # A cheap flop estimate: each batched NxN @ NxN matmul is ~2*batch*N^3.
        flops = 2 * self.batch * (self.size ** 3) * self.iters
        self.metrics = {
            "wall_s": wall_s,
            "gflops": (flops / wall_s / 1e9) if (wall_s > 0 and not oom) else 0.0,
            "device": str(device),
            "mem_mb": self.mem_mb,
            "oom": oom,
            "throttled": throttled_iters > 0,
            "throttled_iters": throttled_iters,
            "throttle_s": throttle_s,
        }
        return self.metrics


class Tenant:
    def __init__(self, tenant_id, arrival_rate, color="#e0e0e0"):
        self.tenant_id = tenant_id
        self.arrival_rate = arrival_rate      # mean jobs per real second
        self.color = color

    def make_job(self, job_id, arrival_time, wtype, rng,
                 iter_multiplier=DEFAULT_ITER_MULTIPLIER):
        """Create one matmul workload for this tenant.

        `wtype` selects the workload type (a WorkloadType or its name); the
        type fixes the matrix size / batch and the ranges the iteration count
        and declared VRAM demand are drawn from. The job's priority is drawn
        uniformly from PRIORITY_RANGE. `rng` is a dedicated ``random.Random``
        instance so that a run is reproducible from its seed -- the global RNG
        is never used here.

        `iter_multiplier` scales the drawn iteration count (compute-intensity
        knob, §3.8 i). It is applied *after* the RNG draw, so it never perturbs
        the seeded stream -- the same seed yields the same base counts at any
        multiplier, just scaled."""
        if isinstance(wtype, str):
            wtype = WORKLOAD_TYPES[wtype]
        return Job(
            job_id=job_id,
            tenant_id=self.tenant_id,
            size=wtype.size,
            iters=rng.randint(*wtype.iter_range) * iter_multiplier,
            wtype=wtype.name,
            batch=wtype.batch,
            mem_mb=rng.randint(*wtype.mem_range_mb),
            priority=rng.randint(*PRIORITY_RANGE),
            arrival_time=arrival_time,
        )


# ---------------------------------------------------------------------------
# Experiment traces -- a run is a finite, pre-generated list of jobs.
#
# Given (number of tenants, per-tenant counts of each workload type, seed) we
# generate the entire job list up front: each tenant's jobs, their workload
# types, iteration counts, and arrival times are all drawn from one dedicated
# ``random.Random(seed)`` instance. Same seed + same config => byte-for-byte
# identical arrivals, sizes, and iteration counts on every run, which is what
# makes two schedulers comparable.
# ---------------------------------------------------------------------------

# Distinct, high-contrast swatches assigned to tenants in order. Cycled if a
# run asks for more tenants than there are colors.
TENANT_PALETTE = [
    "#4caf50", "#2196f3", "#ff9800", "#9c27b0", "#e91e63",
    "#00bcd4", "#8bc34a", "#ff5722", "#3f51b5", "#795548",
    "#009688", "#cddc39",
]

# Mean jobs-per-second used to space one tenant's arrivals. The counts (not the
# rate) decide how many jobs a run contains; the rate only sets their spacing.
DEFAULT_ARRIVAL_RATE = 1 / 2.0


def _tenant_name(index):
    """TenantA, TenantB, ... TenantZ, then Tenant27, Tenant28, ..."""
    if index < 26:
        return f"Tenant{chr(ord('A') + index)}"
    return f"Tenant{index + 1}"


def make_tenants(num_tenants, arrival_rate=DEFAULT_ARRIVAL_RATE):
    """Build `num_tenants` tenants with generated IDs and cycled palette colors."""
    return [
        Tenant(_tenant_name(i), arrival_rate, color=TENANT_PALETTE[i % len(TENANT_PALETTE)])
        for i in range(num_tenants)
    ]


def generate_trace(tenants, counts, seed, arrival_rate=None,
                   iter_multiplier=DEFAULT_ITER_MULTIPLIER):
    """Return the full, sorted list of jobs for one run.

    `counts` is ``{"light": n, "medium": n, "heavy": n}`` -- the number of jobs
    of each workload type *each* tenant contributes. `arrival_rate` (mean jobs
    per second per tenant) is an experiment parameter: when given it overrides
    every tenant's own rate, so arrival intensity is controlled per run rather
    than baked into the tenant objects; when None each tenant's configured rate
    is used. `iter_multiplier` scales every job's matmul iteration count (see
    Tenant.make_job) -- a whole-experiment compute-intensity knob applied after
    the RNG draw, so it leaves the trace otherwise identical. Every random
    choice (job ordering, iteration counts, memory demands, priorities, arrival
    gaps) comes from a single ``random.Random(seed)``, so the trace is fully
    determined by (tenants, counts, seed, arrival_rate, iter_multiplier). Jobs
    are returned sorted by arrival time with ids assigned in arrival order.
    """
    rng = random.Random(seed)
    jobs = []
    for tenant in tenants:
        rate = arrival_rate if arrival_rate is not None else tenant.arrival_rate
        # The multiset of workload types this tenant will submit, shuffled so
        # its light/medium/heavy jobs are interleaved rather than blocked.
        types = []
        for name in ("light", "medium", "heavy"):
            types += [name] * counts.get(name, 0)
        rng.shuffle(types)

        t = 0.0
        for wtype in types:
            t += rng.expovariate(rate)
            # job_id is a placeholder here; reassigned in arrival order below.
            jobs.append(tenant.make_job(0, arrival_time=t, wtype=wtype, rng=rng,
                                        iter_multiplier=iter_multiplier))

    jobs.sort(key=lambda j: j.arrival_time)
    for job_id, job in enumerate(jobs):
        job.job_id = job_id
    return jobs


def build_experiment(num_tenants, counts, seed, arrival_rate=DEFAULT_ARRIVAL_RATE,
                     iter_multiplier=DEFAULT_ITER_MULTIPLIER):
    """Convenience wrapper: build the tenants and their job trace together.

    `arrival_rate` and `iter_multiplier` are threaded through to the trace, so
    each is a single, citable config parameter for the whole experiment."""
    tenants = make_tenants(num_tenants, arrival_rate)
    jobs = generate_trace(tenants, counts, seed, arrival_rate=arrival_rate,
                          iter_multiplier=iter_multiplier)
    return tenants, jobs


# Default tenant set, used only for previews / standalone engine runs. A real
# experiment builds its tenants from the UI config via `make_tenants`.
DEFAULT_TENANTS = make_tenants(3)
