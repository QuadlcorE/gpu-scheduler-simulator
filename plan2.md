# Plan 2 — Aligning the Project with the Report (Chapter 3)

This plan covers every code/report divergence **except the Dear ImGui swap** (UI stays
Tkinter), and folds in the report's later pages: the 4-step admission algorithm (§3.6),
the three named policies (§3.5), the resource-constraint layer (§3.4.3), the
experimental knobs (§3.8), and all seven evaluation metrics (§3.9).

## Guiding decision: how to make the "SimPy" claim true

The report commits to "SimPy (event simulation)" with real CUDA execution — a hybrid.
The cleanest way to honor that without giving up real GPU measurements is to rebuild
the engine around **`simpy.rt.RealtimeEnvironment`** (SimPy's wall-clock-synchronized
mode): arrivals, queueing, dispatch, and admission decisions become SimPy
processes/events (genuine DES modelling), while job execution still happens on real
server threads running PyTorch, signalling a SimPy event on completion. That is exactly
the hybrid §3.1–3.2 describes, and it stays honest for the write-up.

---

## Phase 1 — Memory-aware workload model (`Workload/workload.py`)

The whole admission story depends on jobs having an explicit VRAM demand. Currently
they don't.

1. Extend `WorkloadType` with a `mem_range_mb` (VRAM demand drawn per job) and a
   `batch` dimension, sized for the 4 GB single-GPU setup: light ≈ 100–250 MB,
   medium ≈ 400–700 MB, heavy ≈ 900–1400 MB. This gives you the **memory demand
   distribution** (§3.4.2 ii) as a first-class, controllable input.
2. Add to `Job`: `mem_mb` (declared demand) and `priority` (int, drawn from the seeded
   RNG — needed for §3.5.2).
3. Make `Job.run()` **actually allocate** its declared footprint: allocate a ballast
   tensor of `mem_mb` on the device alongside the GEMM tensors. This is what makes the
   hybrid claim real — NVML will genuinely show the memory the admission controller
   reasoned about, and a bad admission policy can genuinely OOM.
4. Wrap execution in `try/except torch.cuda.OutOfMemoryError` → record an OOM event in
   the job's metrics instead of crashing the server thread (feeds metric §3.9 v).
5. Add a cooperative **throttle hook**: `Job.run(device, throttle=...)` checks a shared
   flag each iteration and sleeps/slows when set (this implements "throttled if already
   running" from §3.6 and compute-share enforcement — queue-order-level control only,
   consistent with limitation ii, "no kernel-level preemption").
6. Thread `arrival_rate` all the way through `generate_trace`/`build_experiment` as a
   real config parameter (§3.8 ii) instead of the fixed module constant.

## Phase 2 — Simulated MIG partitions (`simulation/partition.py`, new)

Implements §3.4.1/§3.4.3 and Figure 3.2.

1. `Partition` dataclass: `index`, `vram_cap_mb`, `compute_share`, plus a
   **reservation ledger** — sum of `mem_mb` of jobs admitted-and-not-finished.
   `used_mb`/`headroom_mb` properties come from the ledger (policy-enforced logical
   limits, exactly as §3.4.1 says: no hardware isolation).
2. Partition set is built from config: number of partitions (§3.8 iii) and
   per-partition caps/shares, defaulting to N equal slices of total VRAM (e.g.
   3 × 1 GB on the 4 GB card). Total-memory limit = sum of caps ≤ physical VRAM
   (§3.4.3 i–ii).
3. The engine's `Server` threads become the execution arm of a partition (all mapping
   to `cuda:0` on the single-GPU setup); compute share is enforced softly via the
   throttle hook (share < fair share ⇒ throttled iterations).

## Phase 3 — Admission controller (`Scheduling_control/admission.py`, new)

Direct transcription of the §3.6 four-step algorithm:

1. `AdmissionController(threshold_pct, safety_margin_mb)` with one method
   `decide(job, partition) -> ADMIT | QUEUE | REJECT | THROTTLE_RUNNING`:
   - current = partition ledger usage (cross-checked against live NVML reading),
   - projected = current + `job.mem_mb`,
   - add `safety_margin_mb`,
   - admit iff projected + margin ≤ `vram_cap_mb × threshold_pct`.
2. Outcomes: over-threshold ⇒ **QUEUE**; job whose demand can never fit any partition
   cap ⇒ **REJECT** (recorded, feeds OOM-avoidance narrative); partition already over
   threshold with a heavy job running ⇒ raise its **throttle flag** and log a
   `ThrottleEvent`.
3. Both `threshold_pct` (§3.8 iv) and `safety_margin_mb` are experiment parameters.
4. Every decision is appended to an **admission log** (job, partition, projected MB,
   threshold, outcome, timestamp) — this becomes a CSV and the UI's "scheduling
   decisions / throttling events" feed.

## Phase 4 — The three report schedulers (`schedulers/`)

The current contract (`select_server(job, servers)`) can't express priority or
fair-share, because the engine hands jobs over one at a time in arrival order. Change
the contract to **queue-aware selection**:

- New contract: `select(pending_jobs, partitions) -> (job, partition_index) | None`,
  where `partitions` includes each partition's `headroom_mb`, `queue_len`, running
  tenant. The engine calls it repeatedly per dispatch pass; admission control then
  validates the proposed placement.
- **`schedulers/fcfs.py`** — pick the oldest pending job, first partition that passes
  admission (§3.5.1). (Retires `pooled.py` as the headline policy; pooled / pinned /
  shortest-queue can stay as extra plugins since the loader is generic.)
- **`schedulers/priority.py`** — pick the highest-priority pending job (ties by
  arrival). Higher priority *preempts queue order*, never running kernels — matching
  §3.5.2 and limitation ii verbatim.
- **`schedulers/fair_share.py`** — track per-tenant consumed service (GPU-seconds and
  VRAM-time); always serve the pending job of the most under-served tenant, spreading
  across partitions proportionally to prevent starvation (§3.5.3).

## Phase 5 — SimPy hybrid engine rewrite (`simulation/simulator.py`)

1. Replace the hand-rolled `_sleep_until` / heartbeat loop with a
   `simpy.rt.RealtimeEnvironment`:
   - an **arrival process** yields `env.timeout()` per trace gap and releases jobs
     into the global queue,
   - a **dispatch process** wakes on arrival/completion events, runs scheduler →
     admission → placement,
   - each placement fires the real server thread; completion triggers a SimPy event
     that re-enters dispatch.
2. Lifecycle timestamps (`submit/assign/start/end`) and the two-wait accounting
   survive unchanged — they're already correct and the report's queue-time story
   depends on them.
3. Snapshots gain: per-partition used/cap MB, throttle states, admission counters —
   same event-queue schema pattern the UI already consumes.

## Phase 6 — Monitoring upgrades (`GPUMonitoring/monitoring.py`)

Closes the §3.3.2 gaps:

1. Add `nvmlDeviceGetComputeRunningProcesses` → per-process memory usage and active
   PIDs (items ii and iv), `_safe`-wrapped like the rest.
2. Add a **run-scoped telemetry sampler**: a thread that, for the duration of a run,
   samples GPU util + VRAM at a fixed interval into a telemetry log. This is what
   makes "average VRAM utilization" and "partition-level resource stability"
   computable from *measured* data rather than the ledger alone.

## Phase 7 — Evaluation metrics (`simulation/results.py`)

Extend `summarize()` and the CSV exports to cover all of §3.9:

| §3.9 metric | Source |
|---|---|
| i. GPU utilization % | busy-time per partition ÷ makespan, plus mean NVML `gpu_util` from telemetry |
| ii. Avg VRAM utilization | mean of telemetry VRAM samples ÷ capacity |
| iii. Job wait time | existing global-wait + server-wait (already done) |
| iv. Job throughput | completed jobs ÷ makespan |
| v. OOM occurrence rate | (caught CUDA OOMs + memory rejects) ÷ total jobs |
| vi. Fairness index | Jain's index over per-tenant mean wait / service share |
| vii. Partition stability | std-dev of each partition's VRAM usage over the telemetry timeline |

Exports per run: `jobs.csv` (now with `mem_mb`, `priority`, admission outcome,
throttled flag, OOM flag), `admission_log.csv`, `telemetry.csv`, `summary.csv`.

## Phase 8 — Experiment config + headless runner

1. An `ExperimentConfig` dataclass capturing everything §3.8 varies: workload
   intensity mix, arrival rate, number of partitions, admission threshold %, safety
   margin, scheduler, seed — serializable to/from JSON so a run is a citable artifact.
2. `run_experiments.py` (new, headless CLI): sweeps a grid over the §3.8 conditions
   and runs **every scheduler on the identical seeded trace per condition** ("each
   scheduler was subjected to the same workload"), writing a results directory per
   cell. This is what will generate Chapter 4's data without babysitting the GUI.
3. Minimal Tkinter wiring (no framework change): add config fields for arrival rate,
   partition count, threshold %, safety margin; add a partition-usage bar
   (used/cap MB) to the grid; add counters / log view for admission decisions and
   throttle events.

## Suggested order and why

Phases 1→2→3 are the dependency spine (jobs need demands before partitions can cap
them before admission can gate them). Phase 4 needs the new contract from 3. Phase 5
(SimPy) is deliberately *after* the logic exists, so the rewrite moves working
behavior rather than building on sand. Phases 6–7 are independent and can interleave.
Phase 8 last, since it just parameterizes what exists.

## Report-side note

While editing the report anyway: hardware says "RTX-2030 / 4gb" — presumably RTX 2050
or similar; worth fixing alongside, since the partition sizing in code will reference
the real 4 GB figure.
