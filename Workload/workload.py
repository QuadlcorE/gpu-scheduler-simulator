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
# Each type pins the matrix dimension (compute grows ~N^3) and the range the
# iteration count is drawn from. The sizes are chosen so "heavy" visibly takes
# seconds while "light" is near-instant; recalibrate per machine if needed.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkloadType:
    name: str          # "light" / "medium" / "heavy"
    label: str         # single-letter tag for compact UI ("L" / "M" / "H")
    size: int          # NxN matrix dimension for the GEMM
    iter_range: tuple  # (min, max) matmul iterations, inclusive


LIGHT = WorkloadType("light", "L", 128, (10, 30))
MEDIUM = WorkloadType("medium", "M", 384, (10, 30))
HEAVY = WorkloadType("heavy", "H", 768, (5, 15))

# Lookup by name; also the canonical light->medium->heavy ordering.
WORKLOAD_TYPES = {w.name: w for w in (LIGHT, MEDIUM, HEAVY)}


@dataclass
class Job:
    job_id: int
    tenant_id: str
    size: int                  # NxN matrix dimension for the GEMM
    iters: int                 # number of matmul iterations
    wtype: str = "medium"      # workload type name (key into WORKLOAD_TYPES)
    arrival_time: float = 0.0  # planned arrival, seconds from run start (from the trace)
    metrics: dict = field(default_factory=dict)  # filled in by run()

    def run(self, device):
        """Execute the GEMM workload on `device` and return measured metrics.

        Runs a chained matrix multiply so PyTorch cannot optimize the loop
        away, then forces evaluation so the wall-clock time reflects real
        compute (important once this runs on CUDA, where ops are async)."""
        a = torch.randn(self.size, self.size, device=device)
        b = torch.randn(self.size, self.size, device=device)

        t0 = time.perf_counter()
        for _ in range(self.iters):
            a = torch.matmul(a, b)
            a = a / (a.norm() + 1e-6)  # keep values bounded across iterations
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0

        # A cheap flop estimate: each NxN @ NxN matmul is ~2*N^3 flops.
        flops = 2 * (self.size ** 3) * self.iters
        self.metrics = {
            "wall_s": wall_s,
            "gflops": flops / wall_s / 1e9 if wall_s > 0 else 0.0,
            "device": str(device),
        }
        return self.metrics


class Tenant:
    def __init__(self, tenant_id, arrival_rate, color="#e0e0e0"):
        self.tenant_id = tenant_id
        self.arrival_rate = arrival_rate      # mean jobs per real second
        self.color = color

    def make_job(self, job_id, arrival_time, wtype, rng):
        """Create one matmul workload for this tenant.

        `wtype` selects the workload type (a WorkloadType or its name); the
        type fixes the matrix size and the range the iteration count is drawn
        from. `rng` is a dedicated ``random.Random`` instance so that a run is
        reproducible from its seed -- the global RNG is never used here."""
        if isinstance(wtype, str):
            wtype = WORKLOAD_TYPES[wtype]
        return Job(
            job_id=job_id,
            tenant_id=self.tenant_id,
            size=wtype.size,
            iters=rng.randint(*wtype.iter_range),
            wtype=wtype.name,
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


def generate_trace(tenants, counts, seed):
    """Return the full, sorted list of jobs for one run.

    `counts` is ``{"light": n, "medium": n, "heavy": n}`` -- the number of jobs
    of each workload type *each* tenant contributes. Every random choice (job
    ordering, iteration counts, arrival gaps) comes from a single
    ``random.Random(seed)``, so the trace is fully determined by (tenants,
    counts, seed). Jobs are returned sorted by arrival time with ids assigned
    in arrival order.
    """
    rng = random.Random(seed)
    jobs = []
    for tenant in tenants:
        # The multiset of workload types this tenant will submit, shuffled so
        # its light/medium/heavy jobs are interleaved rather than blocked.
        types = []
        for name in ("light", "medium", "heavy"):
            types += [name] * counts.get(name, 0)
        rng.shuffle(types)

        t = 0.0
        for wtype in types:
            t += rng.expovariate(tenant.arrival_rate)
            # job_id is a placeholder here; reassigned in arrival order below.
            jobs.append(tenant.make_job(0, arrival_time=t, wtype=wtype, rng=rng))

    jobs.sort(key=lambda j: j.arrival_time)
    for job_id, job in enumerate(jobs):
        job.job_id = job_id
    return jobs


def build_experiment(num_tenants, counts, seed, arrival_rate=DEFAULT_ARRIVAL_RATE):
    """Convenience wrapper: build the tenants and their job trace together."""
    tenants = make_tenants(num_tenants, arrival_rate)
    jobs = generate_trace(tenants, counts, seed)
    return tenants, jobs


# Default tenant set, used only for previews / standalone engine runs. A real
# experiment builds its tenants from the UI config via `make_tenants`.
DEFAULT_TENANTS = make_tenants(3)
