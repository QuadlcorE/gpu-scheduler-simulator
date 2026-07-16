# Architecture

## Layers
- **Workload generation** (`Workload/workload.py`) — defines a `Job` (a real
  PyTorch matmul) and the three workload types (light / medium / heavy). Builds
  a run's finite, seeded **job trace**.
- **Scheduling and admission control** (`Scheduling_control/`) — decides which
  server (GPU) each pending job runs on. Pluggable (see below).
- **Simulation** (`simulation/`) — the execution engine: replays the trace,
  asks the scheduler where each job goes, runs it on a server thread, and
  records the results.
- **GPU monitoring** (`GPUMonitoring/`) — live hardware stats for the UI.
- **UI** (`UI/dashboard.py`) — Tkinter dashboard: experiment config, live grid
  and tables, and the end-of-run results view.

## Scheduler plugin contract
Each scheduling algorithm is one `.py` file in the top-level `schedulers/`
directory. To add one:

1. Subclass `Scheduler` (from `Scheduling_control.scheduler`).
2. Set a `NAME` class attribute — the label shown in the dashboard dropdown.
3. Implement `select_server(self, job, servers) -> int | None`.

The loader (`Scheduling_control/loader.py`) discovers every `Scheduler`
subclass in that directory at startup; the dashboard's **Reload schedulers**
button rescans without restarting.

`select_server` receives:
- `job` — the pending `Workload.Job` (`job.tenant_id`, `job.wtype`
  (light/medium/heavy), `job.size`, `job.iters` are available to inform
  placement).
- `servers` — a list of dicts, one per server slot, ordered by index:
  `{"index": int, "free": bool, "tenant": str | None, "queue_len": int}`.

Return the index of the server to run the job on. Servers have **unbounded
inboxes**, so returning a *busy* server's index is allowed — the job then waits
in that server's queue (its *server-queue wait*) behind whatever is ahead of it.
Return `None` to leave the job in the global queue (re-offered next dispatch).

Worked examples: `schedulers/pooled.py`, `schedulers/pinned.py`,
`schedulers/shortest_queue.py`.

## Experiment config (replicable runs)
A run is fully determined by **(number of tenants, per-tenant counts of each
workload type, seed)**, set in the dashboard's Configuration panel.
`Workload.build_experiment(num_tenants, counts, seed)` pre-generates the entire
job trace up front using a dedicated `random.Random(seed)` — never the global
RNG. Same config + same seed ⇒ byte-for-byte identical arrivals, sizes, and
iteration counts, which is what makes two schedulers comparable on one workload.

The engine replays that finite trace, releasing each job at its planned
`arrival_time`, and stops itself once every job has finished.

## Lifecycle timing (two kinds of wait)
Each job is timestamped through its lifecycle so the two waits are separable:

    submit_time -> assign_time -> start_time -> end_time
    global-queue wait = assign - submit   (waiting for placement)
    server-queue wait = start  - assign   (waiting behind other jobs)

## Results and evaluation
Every completed job leaves a finalized row in the run's **results log**
(`MultiTenantSimulation.results`). Helpers in `simulation/results.py`:
- `write_results_csv(path, results)` — export per-job rows (job id, tenant,
  type, size, iters, arrival, both wait times, run time, GPU) to CSV.
- `summarize(results, capacity)` — per-tenant mean waits, makespan, and GPU
  utilization.

The engine's final `done=True` snapshot carries both, so the dashboard shows a
results summary window (with CSV export) when a run completes.
`test_replicability.py` demonstrates the workflow: same seed twice ⇒ identical
trace; two schedulers on one seed ⇒ same workload, comparable outcomes.
