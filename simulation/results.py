"""
Per-run results log: the record of a finished experiment.

Every job that leaves the system leaves a row here (see
``MultiTenantSimulation.results``) -- completed jobs and admission-REJECTed
jobs alike; together those rows are what makes a run *evaluable*. Two things
are built on top of them:

    * the CSV exporters (:func:`write_results_csv`, :func:`write_telemetry_csv`,
      :func:`write_summary_csv`, and the :func:`export_run` bundle) -- so runs
      under identical seeds can be compared quantitatively (and diffed) across
      scheduling algorithms;
    * :func:`summarize` -- fold the rows (plus the run's measured telemetry
      timeline) into a per-run summary covering all seven evaluation metrics
      of report §3.9, for the results view shown when a run completes.

A result row is a plain dict with these keys (all times in seconds, relative to
run start unless noted):

    job_id, tenant, wtype, size, iters,
    mem_mb         -- declared VRAM demand (deterministic from seed)
    priority       -- scheduling priority (deterministic from seed)
    arrival_s      -- planned arrival from the trace (deterministic from seed)
    admission      -- final admission outcome ("admit" or "reject")
    global_wait_s  -- assign - submit  (wait for placement)
    server_wait_s  -- start  - assign  (wait behind other jobs on the server)
    run_s          -- measured execution wall time (0 for rejected jobs)
    gpu            -- partition/server index the job ran on (None if rejected)
    start_s        -- when the server began executing it (None if rejected)
    end_s          -- when it finished (rejection time for rejected jobs)
    oom            -- True if execution hit a caught CUDA OutOfMemoryError
    throttled      -- True if any iteration ran cooperatively throttled

The deterministic columns (job_id, tenant, wtype, size, iters, mem_mb,
priority, arrival_s) are identical for a given (config, seed); the remaining
columns are real measured outcomes and vary between runs.

§3.9 metric map (where each evaluation metric comes from):

    i.   GPU utilization %      summarize(): gpu_util (busy-time / makespan,
                                overall and per partition) + gpu_util_nvml_mean
                                (measured NVML utilization from telemetry)
    ii.  Avg VRAM utilization   summarize(): vram_util_mean over telemetry
    iii. Job wait time          per-tenant avg global/server waits (rows)
    iv.  Job throughput         summarize(): throughput_jps
    v.   OOM occurrence rate    summarize(): oom_rate (caught OOMs + rejects)
    vi.  Fairness index         summarize(): fairness_wait / fairness_service
                                (Jain's index over per-tenant values)
    vii. Partition stability    summarize(): per_partition[i].std_used_mb over
                                the telemetry timeline
"""

import csv
import math
import os
import re
from collections import defaultdict

# Column order used for CSV export. Deterministic trace columns first, then the
# measured outcomes, so a diff of two CSVs surfaces the timing differences last.
CSV_FIELDS = [
    "job_id", "tenant", "wtype", "size", "iters", "mem_mb", "priority",
    "arrival_s",
    "admission", "global_wait_s", "server_wait_s", "run_s", "gpu",
    "start_s", "end_s", "oom", "throttled",
]

# The subset that is fully determined by (config, seed). Same seed + same config
# => these are byte-for-byte identical across runs and across schedulers, which
# is what lets two schedulers be compared fairly on the same workload.
TRACE_FIELDS = ["job_id", "tenant", "wtype", "size", "iters", "mem_mb",
                "priority", "arrival_s"]

# Fixed telemetry columns (see GPUMonitoring.monitoring.TelemetrySampler);
# the per-partition ledger columns (part<i>_used_mb) are appended dynamically.
TELEMETRY_BASE_FIELDS = ["t", "gpu_util", "mem_used_mb", "mem_total_mb",
                         "proc_count", "proc_mem_mb"]

# Throttle-event rows as logged by the admission controller.
THROTTLE_LOG_FIELDS = ["timestamp", "partition", "event", "job_id"]

# Admission outcome recorded on rejected result rows (mirrors
# Scheduling_control.admission.REJECT; duplicated here so this module stays
# importable without the controller and its NVML dependencies).
_REJECT = "reject"

_PART_COL_RE = re.compile(r"^part(\d+)_used_mb$")


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


def write_telemetry_csv(path, samples):
    """Write a run's telemetry timeline (TelemetrySampler.samples) to CSV.

    Column order: the fixed NVML columns, then one ledger column per partition
    in partition-index order. Returns the number of rows written."""
    part_cols = sorted(
        {k for s in samples for k in s if _PART_COL_RE.match(k)},
        key=lambda k: int(_PART_COL_RE.match(k).group(1)),
    )
    fields = TELEMETRY_BASE_FIELDS + part_cols
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for s in samples:
            writer.writerow({k: s.get(k) for k in fields})
    return len(samples)


def write_throttle_log_csv(path, log):
    """Write the admission controller's throttle raise/release events to CSV.
    Returns the number of rows written."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=THROTTLE_LOG_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for r in log:
            writer.writerow({k: r.get(k) for k in THROTTLE_LOG_FIELDS})
    return len(log)


def _flatten(obj, prefix="", out=None):
    """Flatten nested summary dicts into dotted keys for the one-row CSV:
    {"per_tenant": {"tenant-1": {"jobs": 3}}} -> {"per_tenant.tenant-1.jobs": 3}."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}{k}.", out)
    else:
        out[prefix[:-1]] = obj
    return out


def write_summary_csv(path, summary):
    """Write a summarize() dict to CSV as one wide row (header + values),
    nested sections flattened to dotted column names. One row per run makes
    the files trivially concatenable across a sweep of experiment conditions.
    Returns the number of columns written."""
    flat = _flatten(summary)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    return len(flat)


def export_run(dir_path, results, capacity, admission_log=None,
               throttle_log=None, telemetry=None, summary=None):
    """Export one finished run as its full set of CSV artifacts.

    Creates `dir_path` (if needed) and writes:

        jobs.csv           -- the per-job result rows (write_results_csv)
        admission_log.csv  -- every admission decision (when provided)
        throttle_log.csv   -- throttle raise/release events (when provided)
        telemetry.csv      -- the measured NVML/ledger timeline (when provided)
        summary.csv        -- the folded §3.9 metrics, one wide row

    `summary` defaults to summarize(results, capacity, telemetry). Returns a
    dict {artifact name: path written}."""
    os.makedirs(dir_path, exist_ok=True)
    paths = {}

    jobs_path = os.path.join(dir_path, "jobs.csv")
    write_results_csv(jobs_path, results)
    paths["jobs"] = jobs_path

    if admission_log is not None:
        # Local import: keeps this module stdlib-only unless the admission
        # log (whose writer lives with the controller) is actually exported.
        from Scheduling_control.admission import write_admission_log_csv
        adm_path = os.path.join(dir_path, "admission_log.csv")
        write_admission_log_csv(adm_path, admission_log)
        paths["admission_log"] = adm_path

    if throttle_log is not None:
        thr_path = os.path.join(dir_path, "throttle_log.csv")
        write_throttle_log_csv(thr_path, throttle_log)
        paths["throttle_log"] = thr_path

    if telemetry is not None:
        tel_path = os.path.join(dir_path, "telemetry.csv")
        write_telemetry_csv(tel_path, telemetry)
        paths["telemetry"] = tel_path

    if summary is None:
        summary = summarize(results, capacity, telemetry)
    sum_path = os.path.join(dir_path, "summary.csv")
    write_summary_csv(sum_path, summary)
    paths["summary"] = sum_path

    return paths


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def jain_index(values):
    """Jain's fairness index over per-tenant values:

        J = (sum x)^2 / (n * sum x^2),  in (0, 1]; 1 = perfectly equal.

    An empty or all-zero input (nobody waited / nobody was served) is trivially
    fair and returns 1.0."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return 1.0
    sq = sum(v * v for v in vals)
    if sq <= 0:
        return 1.0
    total = sum(vals)
    return (total * total) / (n * sq)


def _mean(values):
    """Mean of a list, or None when it is empty."""
    return sum(values) / len(values) if values else None


def _pstdev(values):
    """Population standard deviation, or None when the list is empty."""
    if not values:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _telemetry_stats(telemetry):
    """Fold a telemetry timeline into the measured §3.9 quantities:

        gpu_util_nvml_mean  -- mean NVML device utilization % (metric i)
        vram_util_mean      -- mean of used/total VRAM per sample (metric ii)
        vram_used_mb_mean   -- same numerator in absolute MB
        partitions          -- {index: {"mean_used_mb", "std_used_mb"}} from
                               the ledger columns (metric vii: stability =
                               std-dev of usage over the timeline)

    NVML fields are None on CPU-only runs; the ledger columns are always
    sampled, so partition stability stays computable everywhere."""
    telemetry = telemetry or []
    utils = [s["gpu_util"] for s in telemetry if s.get("gpu_util") is not None]
    fracs, used = [], []
    for s in telemetry:
        u, t = s.get("mem_used_mb"), s.get("mem_total_mb")
        if u is not None:
            used.append(u)
            if t:
                fracs.append(u / t)

    per_part = defaultdict(list)
    for s in telemetry:
        for k, v in s.items():
            m = _PART_COL_RE.match(k)
            if m and v is not None:
                per_part[int(m.group(1))].append(v)

    return {
        "gpu_util_nvml_mean": _mean(utils),
        "vram_util_mean": _mean(fracs),
        "vram_used_mb_mean": _mean(used),
        "partitions": {
            i: {"mean_used_mb": _mean(vals), "std_used_mb": _pstdev(vals)}
            for i, vals in sorted(per_part.items())
        },
    }


def summarize(results, capacity, telemetry=None):
    """Fold result rows (+ the run's telemetry timeline) into a per-run summary
    covering all seven §3.9 evaluation metrics::

        {
            "jobs": int,             # rows in the results log (done + rejected)
            "completed": int,        # jobs that actually executed
            "rejected": int,         # admission-REJECTed jobs
            "oom_count": int,        # caught CUDA OOMs during execution
            "capacity": int,
            "makespan_s": float,     # first arrival -> last completion
            "total_run_s": float,    # summed measured execution time

            # i. GPU utilization
            "gpu_util": float,           # total_run_s / (capacity*makespan)
            "gpu_util_nvml_mean": float|None,  # mean measured NVML util %

            # ii. average VRAM utilization (measured, from telemetry)
            "vram_util_mean": float|None,      # mean used/total, 0..1
            "vram_used_mb_mean": float|None,

            # iv. throughput  /  v. OOM occurrence rate
            "throughput_jps": float,     # completed / makespan
            "oom_rate": float,           # (OOMs + rejects) / jobs

            # vi. fairness (Jain's index over per-tenant values)
            "fairness_wait": float,      # over mean total wait per tenant
            "fairness_service": float,   # over GPU-seconds served per tenant

            # vii. partition stability (+ per-partition busy time, metric i)
            "per_partition": {index: {busy_s, busy_frac,
                                      mean_used_mb, std_used_mb}},

            # iii. wait times live here, per tenant
            "per_tenant": {tenant: {jobs, rejected,
                                    avg_global_wait, avg_server_wait,
                                    total_global_wait, total_server_wait,
                                    total_run_s}},
        }

    Wait/service per-tenant figures are computed over *completed* jobs only;
    rejected jobs never ran, so they count toward `rejected`/`oom_rate` but do
    not dilute the wait averages. An empty result set yields zeroed fields
    rather than raising, so a stopped run with no completed jobs still
    summarizes cleanly."""
    tstats = _telemetry_stats(telemetry)

    n = len(results)
    if n == 0:
        return {
            "jobs": 0, "completed": 0, "rejected": 0, "oom_count": 0,
            "capacity": capacity, "makespan_s": 0.0, "total_run_s": 0.0,
            "gpu_util": 0.0,
            "gpu_util_nvml_mean": tstats["gpu_util_nvml_mean"],
            "vram_util_mean": tstats["vram_util_mean"],
            "vram_used_mb_mean": tstats["vram_used_mb_mean"],
            "throughput_jps": 0.0, "oom_rate": 0.0,
            "fairness_wait": 1.0, "fairness_service": 1.0,
            "per_partition": {
                i: {"busy_s": 0.0, "busy_frac": 0.0, **ps}
                for i, ps in tstats["partitions"].items()
            },
            "per_tenant": {},
        }

    done = [r for r in results if r.get("admission") != _REJECT]
    rejected = n - len(done)
    oom_count = sum(1 for r in results if r.get("oom"))

    start = min(r["arrival_s"] for r in results)
    end = max(r["end_s"] for r in results)
    makespan = max(end - start, 0.0)
    total_run = sum(r["run_s"] for r in results)
    # Fraction of the available GPU-seconds (capacity * makespan) actually spent
    # computing. Clamped to 1.0 to stay sane if measured compute slightly
    # exceeds the wall-clock span (overlap/rounding at the edges).
    denom = capacity * makespan
    gpu_util = min(total_run / denom, 1.0) if denom > 0 else 0.0

    # iv. throughput: completed jobs per second of makespan.
    throughput = len(done) / makespan if makespan > 0 else 0.0
    # v. OOM occurrence rate: executions that hit a caught CUDA OOM, plus
    # memory rejects (jobs admission refused outright as guaranteed OOMs).
    oom_rate = (oom_count + rejected) / n

    # Per-partition busy time (metric i, "busy-time per partition / makespan"),
    # merged with the telemetry-derived stability figures (metric vii).
    busy = defaultdict(float)
    for r in done:
        if r.get("gpu") is not None:
            busy[r["gpu"]] += r["run_s"]
    per_partition = {}
    for i in sorted(set(busy) | set(tstats["partitions"])):
        b = busy.get(i, 0.0)
        per_partition[i] = {
            "busy_s": b,
            "busy_frac": min(b / makespan, 1.0) if makespan > 0 else 0.0,
            "mean_used_mb": tstats["partitions"].get(i, {}).get("mean_used_mb"),
            "std_used_mb": tstats["partitions"].get(i, {}).get("std_used_mb"),
        }

    # iii. wait times, per tenant (completed jobs only -- rejected jobs never
    # ran, so they are counted but excluded from the wait averages).
    groups = defaultdict(list)
    rejected_by_tenant = defaultdict(int)
    for r in results:
        if r.get("admission") == _REJECT:
            rejected_by_tenant[r["tenant"]] += 1
        else:
            groups[r["tenant"]].append(r)

    per_tenant = {}
    for tenant in sorted(set(groups) | set(rejected_by_tenant)):
        rows = groups.get(tenant, [])
        k = len(rows)
        tg = sum(x["global_wait_s"] for x in rows)
        ts = sum(x["server_wait_s"] for x in rows)
        per_tenant[tenant] = {
            "jobs": k,
            "rejected": rejected_by_tenant.get(tenant, 0),
            "avg_global_wait": tg / k if k else 0.0,
            "avg_server_wait": ts / k if k else 0.0,
            "total_global_wait": tg,
            "total_server_wait": ts,
            "total_run_s": sum(x["run_s"] for x in rows),
        }

    # vi. fairness: Jain's index over per-tenant mean total wait (equal waits
    # = fair treatment) and over per-tenant GPU-seconds served (equal service
    # share). Tenants with no completed jobs still count in the service index
    # (they received 0 service) but are skipped in the wait index (no waits
    # to average).
    wait_means = [t["avg_global_wait"] + t["avg_server_wait"]
                  for t in per_tenant.values() if t["jobs"] > 0]
    service = [t["total_run_s"] for t in per_tenant.values()]
    fairness_wait = jain_index(wait_means)
    fairness_service = jain_index(service)

    return {
        "jobs": n, "completed": len(done), "rejected": rejected,
        "oom_count": oom_count,
        "capacity": capacity, "makespan_s": makespan,
        "total_run_s": total_run,
        "gpu_util": gpu_util,
        "gpu_util_nvml_mean": tstats["gpu_util_nvml_mean"],
        "vram_util_mean": tstats["vram_util_mean"],
        "vram_used_mb_mean": tstats["vram_used_mb_mean"],
        "throughput_jps": throughput,
        "oom_rate": oom_rate,
        "fairness_wait": fairness_wait,
        "fairness_service": fairness_service,
        "per_partition": per_partition,
        "per_tenant": per_tenant,
    }
