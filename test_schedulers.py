"""
Unit + integration checks for the three report schedulers (§3.5) and the
queue-aware selection contract they implement.

  1. FCFS (§3.5.1): strictly oldest-arrival-first, first-fit partition choice,
     and the roomiest-partition fallback that lets admission QUEUE/REJECT.
  2. Priority (§3.5.2): higher priority jumps the queue order; ties fall back
     to arrival order (FCFS among equals).
  3. Fair-share (§3.5.3): the most under-served tenant is served first
     (consumed GPU-seconds + VRAM GB-seconds), in-flight placements spread the
     first picks across tenants, and load spreads across partitions.
  4. Legacy adapter: old-style ``select_server`` plugins (pooled) still work
     through the new ``select`` contract, and the loader discovers all six
     plugins.
  5. Engine integration: a full MultiTenantSimulation run on CPU with the
     scheduler -> admission -> placement pipeline -- every placeable job
     completes, the structurally unplaceable one is REJECTed and recorded,
     and the run terminates.

NVML is disabled in the integration run so the checks are deterministic on
any machine (GPU or not). Run it directly::

    python test_schedulers.py

It prints one line per check and exits non-zero if any fail.
"""

import queue
import sys
import threading

from Scheduling_control.admission import ADMIT, REJECT, AdmissionController
from Scheduling_control.loader import load_schedulers
from Workload.workload import Job, make_tenants
from simulation.partition import build_partitions
from simulation.simulator import MultiTenantSimulation

from schedulers.fcfs import FCFSScheduler
from schedulers.priority import PriorityScheduler
from schedulers.fair_share import FairShareScheduler
from schedulers.pooled import PooledScheduler

failures = 0


def check(name, got, want):
    global failures
    ok = got == want
    if not ok:
        failures += 1
    print(f"    {'OK  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return ok


def job(job_id, tenant="TenantA", mem_mb=100, priority=0, arrival=0.0):
    """A trace job with only the fields scheduling reads populated."""
    return Job(job_id=job_id, tenant_id=tenant, size=32, iters=1, wtype="light",
               mem_mb=mem_mb, priority=priority, arrival_time=arrival)


def pview(index, admit=400, queue_len=0, free=None, tenant=None, cap=512):
    """One partition-view dict as the engine's _partition_view builds it."""
    used = cap - admit  # only the relationship matters for these checks
    return {
        "index": index, "free": free if free is not None else queue_len == 0,
        "tenant": tenant, "queue_len": queue_len, "cap_mb": cap,
        "used_mb": used, "headroom_mb": admit, "admit_headroom_mb": admit,
        "throttled": False, "share": 0.5,
    }


def drain(sched, pending, partitions):
    """Repeatedly call select, removing each proposed job (as an all-ADMIT
    engine pass would). Returns the jobs in proposal order."""
    pending = list(pending)
    order = []
    while pending:
        prop = sched.select(list(pending), partitions)
        if prop is None:
            break
        j, _ = prop
        pending.remove(j)
        order.append(j.job_id)
    return order


def main():
    print("[1] FCFS: oldest arrival first, first-fit partitions")
    fcfs = FCFSScheduler()
    jobs = [job(0, arrival=0.3), job(1, arrival=0.1), job(2, arrival=0.2)]
    view = [pview(0), pview(1)]
    j, idx = fcfs.select(jobs, view)
    check("oldest pending job picked", j.job_id, 1)
    check("first partition that fits", idx, 0)
    j, idx = fcfs.select(jobs, [pview(0, admit=50), pview(1)])
    check("first-fit skips a full partition", idx, 1)
    j, idx = fcfs.select(jobs, [pview(0, admit=50), pview(1, admit=80)])
    check("nothing fits -> roomiest proposed (admission rules)", idx, 1)
    check("drain order is arrival order", drain(fcfs, jobs, view), [1, 2, 0])
    check("empty queue -> None", fcfs.select([], view), None)

    print("\n[1b] Single-queue spread: fill never-used GPUs before reusing")
    fcfs = FCFSScheduler()
    # Three idle, never-used partitions with equal room. Placing three jobs in
    # turn (committing each via job_placed, as the engine does) should land them
    # on three DISTINCT GPUs, not pile onto GPU 0.
    _, i0 = fcfs.select([job(0)], [pview(0), pview(1), pview(2)])
    fcfs.job_placed(job(0), i0)
    _, i1 = fcfs.select([job(1)], [pview(0, free=False, queue_len=1),
                                   pview(1), pview(2)])
    fcfs.job_placed(job(1), i1)
    _, i2 = fcfs.select([job(2)], [pview(0, free=False, queue_len=1),
                                   pview(1, free=False, queue_len=1), pview(2)])
    fcfs.job_placed(job(2), i2)
    check("spreads across distinct unused GPUs", sorted([i0, i1, i2]), [0, 1, 2])
    # Every GPU used now: a fourth job (all busy) goes to the least-loaded one.
    _, i3 = fcfs.select([job(3)], [pview(0, free=False, queue_len=1),
                                   pview(1, free=False, queue_len=2),
                                   pview(2, free=False, queue_len=3)])
    check("once all used, falls back to least-loaded", i3, 0)
    # A never-used GPU that can't admit the job is skipped for one that can.
    fresh = FCFSScheduler()
    _, i = fresh.select([job(0, mem_mb=100)],
                        [pview(0, admit=50), pview(1, admit=400)])
    check("unused-but-full GPU skipped for one that fits", i, 1)
    # Pooled fills a fresh GPU rather than reusing a busy one.
    pooled = PooledScheduler()
    _, pi0 = pooled.select([job(0)], [pview(0), pview(1), pview(2)])
    pooled.job_placed(job(0), pi0)
    _, pi1 = pooled.select([job(1)], [pview(0, free=False, queue_len=1),
                                      pview(1), pview(2)])
    check("pooled fills a fresh GPU, not the busy one", (pi0, pi1), (0, 1))

    print("\n[2] Priority: highest first, ties by arrival")
    prio = PriorityScheduler()
    jobs = [job(0, priority=2, arrival=0.0), job(1, priority=9, arrival=0.1),
            job(2, priority=9, arrival=0.2), job(3, priority=0, arrival=0.05)]
    check("priority preempts queue order", drain(prio, jobs, view), [1, 2, 0, 3])

    print("\n[3] Fair share: most under-served tenant first")
    fs = FairShareScheduler()
    a1 = job(0, "TenantA", arrival=0.0)
    a2 = job(1, "TenantA", arrival=0.1)
    b1 = job(2, "TenantB", arrival=0.2)
    j, idx = fs.select([a1, a2, b1], view)
    check("all tied -> oldest arrival wins", j.job_id, 0)
    fs.job_placed(a1, idx)
    j, _ = fs.select([a2, b1], view)
    check("in-flight placement steers next pick to the other tenant",
          j.job_id, 2)
    fs.job_finished(a1, 0, {"wall_s": 10.0})
    check("service tallied (GPU-s + VRAM GB-s)",
          round(fs.service_score("TenantA"), 3), round(10.0 + (100 / 1024) * 10.0, 3))
    j, _ = fs.select([a2, b1], view)
    check("under-served tenant jumps an earlier arrival", j.job_id, 2)
    _, idx = fs.select([b1], [pview(0, queue_len=2), pview(1, queue_len=0)])
    check("spreads to the least-loaded partition", idx, 1)
    _, idx = fs.select([b1], [pview(0, admit=50, queue_len=0),
                              pview(1, admit=400, queue_len=3)])
    check("but only across partitions that can admit the job", idx, 1)

    print("\n[4] Legacy select_server plugins still work through select()")
    pooled = PooledScheduler()
    jobs = [job(0, arrival=0.0), job(1, arrival=0.1)]
    prop = pooled.select(jobs, [pview(0, free=False, queue_len=1), pview(1)])
    check("pooled adapts: first job on first free server",
          (prop[0].job_id, prop[1]), (0, 1))
    check("no free server -> None (job held in global queue)",
          pooled.select(jobs, [pview(0, free=False, queue_len=1)]), None)
    names = set(load_schedulers())
    for want in (FCFSScheduler.NAME, PriorityScheduler.NAME,
                 FairShareScheduler.NAME, PooledScheduler.NAME):
        check(f"loader finds {want!r}", want in names, True)

    print("\n[5] Engine integration: scheduler -> admission -> placement")
    partitions = build_partitions(caps_mb=[512, 512], total_vram_mb=1024)
    admission = AdmissionController(threshold_pct=0.9, safety_margin_mb=32,
                                    partitions=partitions, use_nvml=False)
    tenants = make_tenants(2)  # TenantA, TenantB
    # Six placeable jobs (100 MB) + one that can never fit (600+32 > 512*0.9).
    trace = [job(i, tenants[i % 2].tenant_id, mem_mb=100, arrival=i * 0.05)
             for i in range(6)]
    trace.append(job(6, "TenantA", mem_mb=600, arrival=0.1))
    sim = MultiTenantSimulation(queue.Queue(), threading.Event(),
                                capacity=len(partitions), tenants=tenants,
                                jobs=trace, scheduler=FCFSScheduler(),
                                partitions=partitions, admission=admission)
    sim.run()  # blocks until the finite trace is drained

    check("run terminates with every job accounted for", sim.completed, 7)
    check("all results recorded", len(sim.results), 7)
    rejected = [r for r in sim.results if r.get("admission") == REJECT]
    check("exactly the unplaceable job rejected",
          [r["job_id"] for r in rejected], [6])
    check("rejected tallied per tenant", sim.stats["TenantA"]["rejected"], 1)
    completed = [r for r in sim.results if r.get("admission") == ADMIT]
    check("the six placeable jobs all completed", len(completed), 6)
    check("no reservations left on any partition",
          [p.used_mb for p in partitions], [0, 0])
    check("admission logged every decision", len(admission.log) >= 7, True)
    check("no OOMs on admitted jobs",
          any(r.get("oom") for r in completed), False)

    print("\nScheduler check:", "PASSED" if failures == 0 else f"FAILED ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
