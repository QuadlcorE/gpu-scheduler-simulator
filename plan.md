1. Pluggable scheduler scripts
Create a schedulers/ directory where each .py file is one algorithm implementing the existing select_server(job, servers) contract (keep Scheduler as the base class; move Pooled/Pinned there as examples).
Write a small loader in Scheduling_control that scans schedulers/ at startup with importlib, finds Scheduler subclasses, and returns {display_name: class} — each script exposes a NAME attribute for the dropdown.
Replace the hardcoded SCHEDULERS dict in dashboard.py with the loader's output, and add a "Reload schedulers" button so a newly written script appears without restarting the app.
(Recommended) Enrich the servers view passed to select_server with per-server queue length and the job's type/size, so user-written algorithms have enough information to do something interesting (e.g. shortest-queue, size-aware placement).
2. Three workload types
Define WorkloadType presets in Workload/workload.py — light / medium / heavy — each with a fixed matrix size and iteration range (e.g. light 128², medium 384², heavy 768²; calibrate so heavy visibly takes seconds on your machine). Add a type field to Job.
Show the type in the UI: a column in the job table and ideally color intensity or a letter in the GPU grid boxes.
3. Replicable, configurable experiments
Rework Tenant so a run is a finite, pre-generated job trace: given (number of tenants, counts of light/medium/heavy per tenant, seed), generate the full list of jobs with arrival times up front using a dedicated random.Random(seed) instance — never the global RNG. Same seed + same config ⇒ identical arrivals, sizes, and iteration counts every run.
Replace the infinite _produce loops in the simulator with a replay of that trace (one dispatcher that releases each job at its arrival time, or per-tenant threads sleeping until each job's scheduled arrival).
Add the config controls to the UI panel: number of tenants (generate tenant IDs/colors dynamically instead of DEFAULT_TENANTS), spinboxes for the count of each workload type, and a seed field. The tenant stats panel must then be rebuilt per run rather than once in __init__.
Detect run completion (all trace jobs finished) and stop automatically, re-enabling the controls — a finite experiment should end itself.
4. Queue-time visibility
Add per-server queues: give each Server a real queue (unbounded inbox) and let the scheduler assign a job to a busy server, where it waits in that server's queue. Track queue length per server in the simulator state.
Timestamp the lifecycle in the simulator: submit_time (enters global queue), assign_time (leaves global queue for a server), start_time (server begins executing), end_time. Global-queue wait = assign − submit; server-queue wait = start − assign.
Surface it in the UI: add "Global wait" and "Server wait" columns to the job table (live-updating for waiting jobs), and per-tenant aggregates (average/total wait per queue type) in the tenant panel.
5. Wrap-up (makes the FYP evaluable)
Keep completed jobs in a per-run results log and export to CSV (job id, tenant, type, both wait times, run time, GPU) — this is what lets you compare scheduling algorithms quantitatively across identical seeded runs, which is presumably the point of the project.
Add a results summary view (per-tenant mean waits, makespan, GPU utilization) shown when a run completes.
Test replicability explicitly: run the same seed twice with the same scheduler and diff the CSVs; then run two schedulers on the same seed to demonstrate the comparison workflow.
Update architecture.md/memory.md to describe the scheduler plugin contract and the experiment config, so writing a new algorithm is documented.