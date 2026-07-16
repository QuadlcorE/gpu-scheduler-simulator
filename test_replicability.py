"""
Replicability + scheduler-comparison demonstration.

This is the check the whole "seeded experiment" design exists to make:

  1. Same seed + same scheduler, run twice -> the generated job *trace*
     (job_id, tenant, type, size, iters, arrival time) is byte-for-byte
     identical. Only the measured wait/run times differ, because real PyTorch
     work takes real wall-clock time on each run.

  2. Same seed, two different schedulers -> the trace is again identical, so
     any difference in the wait-time outcomes is attributable to the scheduler,
     not to a different workload. That is the comparison the project is for.

Run it directly::

    python test_replicability.py

It writes the four CSVs it produces into a temp directory and prints a short
report; it exits non-zero if the trace is not reproducible.
"""

import queue
import sys
import tempfile
import threading
from pathlib import Path

from simulation.simulator import MultiTenantSimulation
from simulation.results import (
    TRACE_FIELDS, summarize, write_results_csv,
)
from Workload.workload import build_experiment
from Scheduling_control.loader import load_schedulers

# Small, fast experiment: a high arrival rate keeps the whole run to a couple of
# seconds so this is usable as a quick check rather than a long benchmark.
CAPACITY = 2
NUM_TENANTS = 2
COUNTS = {"light": 3, "medium": 1, "heavy": 0}
SEED = 42
ARRIVAL_RATE = 5.0  # mean jobs/sec per tenant (tight spacing => quick run)


def run_once(scheduler):
    """Run one full experiment to completion and return its results log."""
    tenants, jobs = build_experiment(NUM_TENANTS, COUNTS, SEED, ARRIVAL_RATE)
    event_q = queue.Queue()
    stop_flag = threading.Event()
    sim = MultiTenantSimulation(event_q, stop_flag, CAPACITY, tenants, jobs, scheduler)
    sim.run()  # blocks until the finite trace is drained
    return sim.results


def trace_of(results):
    """The deterministic subset of a results log, ordered by job_id, rounded so
    float formatting can't introduce spurious differences."""
    rows = sorted(results, key=lambda r: r["job_id"])
    out = []
    for r in rows:
        row = tuple(r[f] for f in TRACE_FIELDS if f != "arrival_s")
        out.append(row + (round(r["arrival_s"], 6),))
    return out


def main():
    schedulers = load_schedulers()
    if not schedulers:
        print("No schedulers found in schedulers/ -- cannot run the demonstration.")
        return 1

    names = list(schedulers)
    sched_a = names[0]
    sched_b = names[1] if len(names) > 1 else names[0]

    out_dir = Path(tempfile.mkdtemp(prefix="fyp_replicability_"))
    ok = True

    # --- 1. Same seed + same scheduler, twice -----------------------------
    print(f"[1] Reproducibility: '{sched_a}' x2, seed {SEED}")
    res_1a = run_once(schedulers[sched_a]())
    res_1b = run_once(schedulers[sched_a]())
    csv_1a = out_dir / "run1_first.csv"
    csv_1b = out_dir / "run1_second.csv"
    write_results_csv(csv_1a, res_1a)
    write_results_csv(csv_1b, res_1b)

    if trace_of(res_1a) == trace_of(res_1b):
        print(f"    OK  identical trace across both runs "
              f"({len(res_1a)} jobs: ids/tenants/types/sizes/iters/arrivals match)")
    else:
        print("    FAIL trace differs between two runs of the same seed!")
        ok = False
    print(f"        {csv_1a}")
    print(f"        {csv_1b}")
    print("        (global_wait_s / server_wait_s / run_s differ slightly -- "
          "real measured time, expected)")

    # --- 2. Same seed, two schedulers -------------------------------------
    print(f"\n[2] Comparison: '{sched_a}' vs '{sched_b}', same seed {SEED}")
    res_2a = run_once(schedulers[sched_a]())
    res_2b = run_once(schedulers[sched_b]())
    csv_2a = out_dir / "run2_schedA.csv"
    csv_2b = out_dir / "run2_schedB.csv"
    write_results_csv(csv_2a, res_2a)
    write_results_csv(csv_2b, res_2b)

    if trace_of(res_2a) == trace_of(res_2b):
        print("    OK  both schedulers ran the identical workload trace")
    else:
        print("    FAIL schedulers saw different traces -- comparison would be unfair!")
        ok = False

    sum_a = summarize(res_2a, CAPACITY)
    sum_b = summarize(res_2b, CAPACITY)

    def wait_totals(summary):
        g = sum(t["total_global_wait"] for t in summary["per_tenant"].values())
        s = sum(t["total_server_wait"] for t in summary["per_tenant"].values())
        return g, s

    ga, sa = wait_totals(sum_a)
    gb, sb = wait_totals(sum_b)
    print(f"    {sched_a:<32} total global wait {ga:6.2f}s  "
          f"server wait {sa:6.2f}s  util {sum_a['gpu_util']*100:5.1f}%")
    print(f"    {sched_b:<32} total global wait {gb:6.2f}s  "
          f"server wait {sb:6.2f}s  util {sum_b['gpu_util']*100:5.1f}%")
    print(f"        {csv_2a}")
    print(f"        {csv_2b}")

    print("\nReplicability check:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
