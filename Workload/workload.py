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
    size: int                 # NxN matrix dimension for the GEMM
    iters: int                # number of matmul iterations
    wtype: str = "medium"     # workload type name (key into WORKLOAD_TYPES)
    submit_time: float = 0.0  # when the job arrived (real seconds)
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

    def next_interarrival(self):
        """Random gap (real seconds) until this tenant submits its next job."""
        return random.expovariate(self.arrival_rate)

    def make_job(self, job_id, submit_time, wtype=None):
        """Create the next matmul workload for this tenant.

        `wtype` selects the workload type (a WorkloadType, its name, or None
        to pick one at random); the type fixes the matrix size and the range
        the iteration count is drawn from."""
        if wtype is None:
            wtype = random.choice(list(WORKLOAD_TYPES.values()))
        elif isinstance(wtype, str):
            wtype = WORKLOAD_TYPES[wtype]
        return Job(
            job_id=job_id,
            tenant_id=self.tenant_id,
            size=wtype.size,
            iters=random.randint(*wtype.iter_range),
            wtype=wtype.name,
            submit_time=submit_time,
        )


# Default tenants used when the caller does not supply its own set.
DEFAULT_TENANTS = [
    Tenant("TenantA", 1 / 2.0, color="#4caf50"),
    Tenant("TenantB", 1 / 3.0, color="#2196f3"),
    Tenant("TenantC", 1 / 5.0, color="#ff9800"),
]
