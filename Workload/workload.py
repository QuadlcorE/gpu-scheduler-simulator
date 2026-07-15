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


@dataclass
class Job:
    job_id: int
    tenant_id: str
    size: int                 # NxN matrix dimension for the GEMM
    iters: int                # number of matmul iterations
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
    # The pool of matrix sizes a tenant's random jobs are drawn from.
    SIZES = (128, 256, 384, 512)

    def __init__(self, tenant_id, arrival_rate, color="#e0e0e0",
                 sizes=None, iter_range=(10, 50)):
        self.tenant_id = tenant_id
        self.arrival_rate = arrival_rate      # mean jobs per real second
        self.color = color
        self.sizes = sizes or self.SIZES
        self.iter_range = iter_range

    def next_interarrival(self):
        """Random gap (real seconds) until this tenant submits its next job."""
        return random.expovariate(self.arrival_rate)

    def make_job(self, job_id, submit_time):
        """Create the next random matmul workload for this tenant."""
        return Job(
            job_id=job_id,
            tenant_id=self.tenant_id,
            size=random.choice(self.sizes),
            iters=random.randint(*self.iter_range),
            submit_time=submit_time,
        )


# Default tenants used when the caller does not supply its own set.
DEFAULT_TENANTS = [
    Tenant("TenantA", 1 / 2.0, color="#4caf50"),
    Tenant("TenantB", 1 / 3.0, color="#2196f3"),
    Tenant("TenantC", 1 / 5.0, color="#ff9800"),
]
