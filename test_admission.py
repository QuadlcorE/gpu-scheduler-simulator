"""
Unit checks for the §3.6 four-step admission decision.

Exercises the AdmissionController boundaries the report's algorithm defines:

  1. ADMIT at the exact threshold boundary (projected + margin == limit).
  2. QUEUE when the projection crosses the threshold -- and the safety-margin
     arithmetic (the same placement admits with margin 0, queues with margin).
  3. REJECT only for jobs that can never fit ANY partition; a job that merely
     doesn't fit the offered partition (but fits another) queues instead.
  4. THROTTLE_RUNNING when a partition is already over threshold with a heavy
     job holding it: flag raised, throttle event logged, and released by
     reconsider() once the partition drains -- without ever clearing a flag
     that apply_compute_shares set statically.
  5. decide() never mutates the ledger, every decision lands in the admission
     log, and the log exports to CSV.

NVML cross-checking is disabled throughout so the checks are deterministic on
any machine (GPU or not). Run it directly::

    python test_admission.py

It prints one line per check and exits non-zero on the first failure count > 0.
"""

import csv
import sys
import tempfile
from pathlib import Path

from Scheduling_control.admission import (
    ADMIT, QUEUE, REJECT, THROTTLE_RUNNING,
    ADMISSION_LOG_FIELDS, AdmissionController, write_admission_log_csv,
)
from simulation.partition import Partition
from Workload.workload import Job

failures = 0


def check(name, got, want):
    global failures
    ok = got == want
    if not ok:
        failures += 1
    print(f"    {'OK  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    return ok


def job(job_id, mem_mb, wtype="medium"):
    """A trace job with only the fields admission reads populated."""
    return Job(job_id=job_id, tenant_id="TenantA", size=128, iters=1,
               wtype=wtype, mem_mb=mem_mb)


def main():
    # One 1000 MB partition at threshold 0.9 => admission limit 900 MB.
    print("[1] Threshold boundary (cap 1000, threshold 0.9, margin 50)")
    p = Partition(index=0, vram_cap_mb=1000, compute_share=1.0)
    ac = AdmissionController(threshold_pct=0.9, safety_margin_mb=50,
                             partitions=[p], use_nvml=False)
    check("projected+margin == limit admits", ac.decide(job(0, 850), p), ADMIT)
    check("decide() does not reserve", p.used_mb, 0)

    # Occupy 200 MB, then offer 700 MB: 200+700+50 = 950 > 900 => QUEUE.
    p.reserve(job(1, 200))
    check("projected+margin over limit queues", ac.decide(job(2, 700), p), QUEUE)

    print("\n[2] Safety-margin arithmetic (same placement, margin 0 vs 50)")
    ac0 = AdmissionController(threshold_pct=0.9, safety_margin_mb=0,
                              partitions=[p], use_nvml=False)
    check("fits exactly with margin 0", ac0.decide(job(3, 700), p), ADMIT)
    check("the margin alone forces the queue", ac.decide(job(3, 700), p), QUEUE)
    p.release(job(1, 200))

    print("\n[3] REJECT only when no partition can ever fit the job")
    check("851+50 > 900 on the only partition rejects",
          ac.decide(job(4, 851), p), REJECT)
    big = Partition(index=1, vram_cap_mb=2000, compute_share=1.0)
    ac2 = AdmissionController(threshold_pct=0.9, safety_margin_mb=50,
                              partitions=[p, big], use_nvml=False)
    check("same job queues when a bigger partition exists",
          ac2.decide(job(4, 851), p), QUEUE)
    check("truly unplaceable demand still rejects",
          ac2.decide(job(5, 3000), p), REJECT)

    print("\n[4] THROTTLE_RUNNING: over threshold with a heavy job holding it")
    heavy = job(6, 950, wtype="heavy")  # 950 > cap*0.5, and 950 > limit 900
    p.reserve(heavy)
    out = ac.decide(job(7, 300), p)
    check("outcome is THROTTLE_RUNNING", out, THROTTLE_RUNNING)
    check("partition throttle flag raised", p.throttle.is_set(), True)
    check("raise event logged",
          [e["event"] for e in ac.throttle_log], ["raised"])

    ac.reconsider(p)  # still over threshold: must NOT release yet
    check("no release while still over threshold", p.throttle.is_set(), True)
    p.release(heavy)
    ac.reconsider(p)
    check("released once drained under threshold", p.throttle.is_set(), False)
    check("release event logged",
          [e["event"] for e in ac.throttle_log], ["raised", "released"])

    print("\n[5] A share-throttled partition's flag is never cleared by admission")
    q = Partition(index=2, vram_cap_mb=1000, compute_share=0.1)
    q.throttle.set()  # as apply_compute_shares would (share < fair share)
    ac3 = AdmissionController(threshold_pct=0.9, safety_margin_mb=50,
                              partitions=[q], use_nvml=False)
    q.reserve(job(8, 950, wtype="heavy"))
    check("still reports THROTTLE_RUNNING", ac3.decide(job(9, 300), q),
          THROTTLE_RUNNING)
    q.release(job(8, 950))
    ac3.reconsider(q)
    check("statically-set flag survives reconsider()", q.throttle.is_set(), True)

    print("\n[6] Admission log + CSV export")
    check("every decision logged", len(ac.log), 5)  # jobs 0, 2, 3, 4, 7
    check("counters add up",
          sorted(ac.snapshot()["counters"].items()),
          [(ADMIT, 1), (QUEUE, 2), (REJECT, 1), (THROTTLE_RUNNING, 1)])
    out_csv = Path(tempfile.mkdtemp(prefix="fyp_admission_")) / "admission_log.csv"
    n = write_admission_log_csv(out_csv, ac.log)
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    check("CSV rows written", n, 5)
    check("CSV header matches schema", rows[0], ADMISSION_LOG_FIELDS)
    check("CSV body row count", len(rows) - 1, 5)
    print(f"        {out_csv}")

    print("\n[7] threshold_pct accepts a percentage")
    check("90 normalizes to 0.9",
          AdmissionController(threshold_pct=90, use_nvml=False).threshold_pct, 0.9)

    print("\nAdmission check:", "PASSED" if failures == 0 else f"FAILED ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
