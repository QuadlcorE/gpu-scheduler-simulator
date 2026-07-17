"""
Headless experiment runner -- generates Chapter 4's data without the GUI.

Sweeps a grid over the report's §3.8 experimental conditions and, for every
cell of the grid, runs **every selected scheduler on the identical seeded
trace** ("each scheduler was subjected to the same workload"): the trace is
fully determined by (mix, tenants, arrival rate, seed), so regenerating it per
scheduler run reproduces it byte-for-byte, and any difference in outcomes is
attributable to the policy alone.

Grid axes (each a repeatable CLI flag):

    --mix L,M,H         workload intensity mix: light/medium/heavy jobs per
                        tenant (§3.8 i), e.g. --mix 4,3,1 --mix 1,2,4
    --arrival-rates     mean jobs/s per tenant (§3.8 ii)
    --partitions        number of equal VRAM partitions (§3.8 iii)
    --thresholds        admission threshold %% of partition cap (§3.8 iv)
    --margins           admission safety margin in MB (§3.6 step 3)
    --seeds             trace seeds (one full grid per seed)

Output layout (one directory per condition, one run per scheduler under it)::

    <out>/
        sweep_summary.csv                   # one row per run: config + §3.9 metrics
        <condition_label>/
            <scheduler_slug>/
                config.json                 # the exact ExperimentConfig (citable)
                jobs.csv  admission_log.csv  throttle_log.csv
                telemetry.csv  summary.csv

Examples::

    # the default 3-scheduler comparison on one condition
    python run_experiments.py --schedulers fcfs priority fair

    # a threshold sweep across partition counts, two seeds
    python run_experiments.py --partitions 2 3 4 --thresholds 80 90 100 \
        --seeds 0 1 --out results/threshold_sweep
"""

import argparse
import itertools
import sys
import threading
import time
from pathlib import Path

from Scheduling_control.loader import load_schedulers
from simulation.config import ExperimentConfig
from simulation.results import _flatten, export_run
from simulation.simulator import MultiTenantSimulation


class _NullQueue:
    """Event-queue stand-in for headless runs: the engine's UI snapshots are
    simply discarded (there is no GUI polling them)."""

    def put(self, item):
        pass


def _parse_mix(text):
    """'4,3,1' -> {'light': 4, 'medium': 3, 'heavy': 1}."""
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"mix must be three comma-separated counts (light,medium,heavy): {text!r}")
    try:
        light, medium, heavy = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"mix counts must be integers: {text!r}")
    return {"light": light, "medium": medium, "heavy": heavy}


def _select_schedulers(available, patterns):
    """Resolve user-given name fragments against the loaded plugin display
    names (case-insensitive substring match). No patterns => all plugins.
    Raises SystemExit on a fragment that matches nothing or is ambiguous."""
    if not patterns:
        return dict(available)
    chosen = {}
    for pat in patterns:
        matches = [n for n in available if pat.lower() in n.lower()]
        if not matches:
            sys.exit(f"error: no scheduler matches {pat!r} "
                     f"(available: {', '.join(available)})")
        if len(matches) > 1:
            sys.exit(f"error: {pat!r} is ambiguous, matches: {', '.join(matches)}")
        chosen[matches[0]] = available[matches[0]]
    return chosen


def run_experiment(config, scheduler_cls):
    """Run one (condition, scheduler) cell headlessly to completion.

    Builds the seeded workload and a fresh scheduler instance (some are
    stateful), runs the engine synchronously, and returns the finished
    ``MultiTenantSimulation`` -- its results/admission/telemetry logs are the
    run's artifacts."""
    tenants, jobs = config.build_workload()
    sim = MultiTenantSimulation(
        _NullQueue(), threading.Event(), config.num_partitions, tenants, jobs,
        scheduler_cls(),
        threshold_pct=config.threshold_pct,
        safety_margin_mb=config.safety_margin_mb,
    )
    sim.run()  # blocks until the finite trace is drained
    return sim


def export_experiment(run_dir, config, sim):
    """Write one finished run's full artifact set (CSV bundle + the exact
    config that produced it) into `run_dir`. Returns the summary dict."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = sim.build_summary()
    export_run(
        str(run_dir), sim.results, sim.capacity,
        admission_log=sim.admission.log,
        throttle_log=sim.admission.throttle_log,
        telemetry=sim.telemetry.samples,
        summary=summary,
    )
    config.save(run_dir / "config.json")
    return summary


def write_sweep_summary(path, rows):
    """Write the whole sweep as one flat CSV: a row per run, config columns
    first, then the flattened §3.9 summary columns. Column set is the union
    across rows (per-tenant/per-partition columns vary with the condition)."""
    import csv
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sweep §3.8 experimental conditions headlessly; every "
                    "scheduler runs the identical seeded trace per condition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tenants", type=int, default=3,
                        help="number of tenants generating jobs")
    parser.add_argument("--mix", type=_parse_mix, action="append", default=None,
                        metavar="L,M,H",
                        help="light,medium,heavy jobs per tenant (repeatable; "
                             "default 4,3,1)")
    parser.add_argument("--arrival-rates", type=float, nargs="+", default=[0.5],
                        metavar="RATE", help="mean jobs/s per tenant")
    parser.add_argument("--partitions", type=int, nargs="+", default=[3],
                        metavar="N", help="number of equal VRAM partitions")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[90.0],
                        metavar="PCT", help="admission threshold %% of cap")
    parser.add_argument("--margins", type=int, nargs="+", default=[64],
                        metavar="MB", help="admission safety margin (MB)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        metavar="SEED", help="trace seeds")
    parser.add_argument("--schedulers", nargs="+", default=None, metavar="NAME",
                        help="scheduler name fragments (case-insensitive "
                             "substring of the plugin display name); default: "
                             "all plugins in schedulers/")
    parser.add_argument("--out", default=None, metavar="DIR",
                        help="results directory (default results/sweep_<timestamp>)")
    parser.add_argument("--list-schedulers", action="store_true",
                        help="list available scheduler plugins and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the run plan without executing anything")
    args = parser.parse_args(argv)

    available = load_schedulers()
    if args.list_schedulers:
        for name in available:
            print(name)
        return 0
    if not available:
        sys.exit("error: no scheduler plugins found in schedulers/")
    schedulers = _select_schedulers(available, args.schedulers)

    mixes = args.mix if args.mix else [{"light": 4, "medium": 3, "heavy": 1}]
    out_dir = Path(args.out) if args.out else (
        Path("results") / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}")

    # The full §3.8 grid: every combination of the swept conditions; every
    # selected scheduler runs inside each cell on the identical seeded trace.
    grid = list(itertools.product(mixes, args.arrival_rates, args.partitions,
                                  args.thresholds, args.margins, args.seeds))
    total_runs = len(grid) * len(schedulers)
    print(f"Sweep: {len(grid)} condition(s) x {len(schedulers)} scheduler(s) "
          f"= {total_runs} run(s) -> {out_dir}")

    sweep_rows = []
    run_no = 0
    for mix, rate, parts, thr, margin, seed in grid:
        base = dict(num_tenants=args.tenants, counts=mix, arrival_rate=rate,
                    num_partitions=parts, threshold_pct=thr,
                    safety_margin_mb=margin, seed=seed)
        cond_label = ExperimentConfig(**base).condition_label()
        for name, cls in schedulers.items():
            run_no += 1
            config = ExperimentConfig(scheduler=name, **base)
            run_dir = out_dir / cond_label / config.scheduler_slug()
            print(f"[{run_no}/{total_runs}] {cond_label} :: {name}", flush=True)
            if args.dry_run:
                continue
            t0 = time.perf_counter()
            sim = run_experiment(config, cls)
            summary = export_experiment(run_dir, config, sim)
            elapsed = time.perf_counter() - t0
            print(f"    {summary['completed']} completed, "
                  f"{summary['rejected']} rejected, "
                  f"{summary['oom_count']} OOM  |  "
                  f"makespan {summary['makespan_s']:.2f}s, "
                  f"util {summary['gpu_util'] * 100:.1f}%, "
                  f"fairness {summary['fairness_wait']:.3f}  "
                  f"({elapsed:.1f}s wall)", flush=True)
            # One flat row per run: the condition first, then the metrics.
            row = {f"config.{k}": v for k, v in _flatten(config.to_dict()).items()}
            row.update(_flatten(summary))
            sweep_rows.append(row)

    if args.dry_run:
        print("Dry run -- nothing executed.")
        return 0

    summary_path = out_dir / "sweep_summary.csv"
    write_sweep_summary(summary_path, sweep_rows)
    print(f"Done. Sweep summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
