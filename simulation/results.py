"""
Per-run results log: the record of a finished experiment.

Every completed job leaves a row here (see ``MultiTenantSimulation.results``);
together those rows are what makes a run *evaluable*. Two things are built on
top of them:

    * :func:`write_results_csv` -- export the raw per-job rows to CSV, so runs
      under identical seeds can be compared quantitatively (and diffed) across
      scheduling algorithms.
    * :func:`summarize` -- fold the rows into a per-run summary (per-tenant mean
      waits, makespan, GPU utilization) for the results view shown when a run
      completes.

A result row is a plain dict with these keys (all times in seconds, relative to
run start unless noted):

    job_id, tenant, wtype, size, iters,
    arrival_s      -- planned arrival from the trace (deterministic from seed)
    global_wait_s  -- assign - submit  (wait for placement)
    server_wait_s  -- start  - assign  (wait behind other jobs on the server)
    run_s          -- measured execution wall time
    gpu            -- server index the job ran on
    start_s        -- when the server began executing it
    end_s          -- when it finished

The deterministic columns (job_id, tenant, wtype, size, iters, arrival_s) are
identical for a given (config, seed); the timing columns are real measurements
and vary slightly between runs.
"""

import csv
from collections import defaultdict

# Column order used for CSV export. Deterministic trace columns first, then the
# measured outcomes, so a diff of two CSVs surfaces the timing differences last.
CSV_FIELDS = [
    "job_id", "tenant", "wtype", "size", "iters", "arrival_s",
    "global_wait_s", "server_wait_s", "run_s", "gpu", "start_s", "end_s",
]

# The subset that is fully determined by (config, seed). Same seed + same config
# => these are byte-for-byte identical across runs and across schedulers, which
# is what lets two schedulers be compared fairly on the same workload.
TRACE_FIELDS = ["job_id", "tenant", "wtype", "size", "iters", "arrival_s"]


def write_results_csv(path, results):
    """Write `results` (a list of result-row dicts) to `path` as CSV.

    Rows are ordered by job_id so the file is stable and diff-friendly. Returns
    the number of rows written."""
    rows = sorted(results, key=lambda r: r["job_id"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})
    return len(rows)


def summarize(results, capacity):
    """Fold result rows into a per-run summary dict.

    Returns::

        {
            "jobs": int, "capacity": int,
            "makespan_s": float,     # first arrival -> last completion
            "total_run_s": float,    # summed measured execution time
            "gpu_util": float,       # total_run_s / (capacity * makespan), 0..1
            "per_tenant": {tenant: {jobs, avg_global_wait, avg_server_wait,
                                    total_global_wait, total_server_wait,
                                    total_run_s}},
        }

    An empty result set yields zeroed fields rather than raising, so a stopped
    run with no completed jobs still summarizes cleanly."""
    n = len(results)
    if n == 0:
        return {
            "jobs": 0, "capacity": capacity, "makespan_s": 0.0,
            "total_run_s": 0.0, "gpu_util": 0.0, "per_tenant": {},
        }

    start = min(r["arrival_s"] for r in results)
    end = max(r["end_s"] for r in results)
    makespan = max(end - start, 0.0)
    total_run = sum(r["run_s"] for r in results)
    # Fraction of the available GPU-seconds (capacity * makespan) actually spent
    # computing. Clamped to 1.0 to stay sane if measured compute slightly
    # exceeds the wall-clock span (overlap/rounding at the edges).
    denom = capacity * makespan
    gpu_util = min(total_run / denom, 1.0) if denom > 0 else 0.0

    groups = defaultdict(list)
    for r in results:
        groups[r["tenant"]].append(r)

    per_tenant = {}
    for tenant, rows in groups.items():
        k = len(rows)
        tg = sum(x["global_wait_s"] for x in rows)
        ts = sum(x["server_wait_s"] for x in rows)
        per_tenant[tenant] = {
            "jobs": k,
            "avg_global_wait": tg / k,
            "avg_server_wait": ts / k,
            "total_global_wait": tg,
            "total_server_wait": ts,
            "total_run_s": sum(x["run_s"] for x in rows),
        }

    return {
        "jobs": n, "capacity": capacity, "makespan_s": makespan,
        "total_run_s": total_run, "gpu_util": gpu_util,
        "per_tenant": per_tenant,
    }
