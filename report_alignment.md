# Project vs. Write-up: Required Changes

Audit of the code in this repo against `Project.pdf` (Abstract + Chapters 1–3,
44 pp., Feb 2026). Complements `report_requirements.txt`, which traced Chapter 3
only; this document adds Chapter 1, Chapter 2, the Abstract, and the Chapter 3
items that trace missed.

**Verified facts used below**
- Physical GPU: `NVIDIA GeForce RTX 2050`, `4096 MiB`, driver 592.82, Windows 11.
  The report's "RTX-2030" is not a real product.
- Schedulers present: `fcfs`, `priority`, `fair_share`, plus `pinned`, `pooled`,
  `shortest_queue`. **No memory-aware, no SJF, no round-robin, no unscheduled baseline.**
- `summarize()` computes no JCT field.
- `ExperimentConfig` has no field for per-partition caps or compute shares.

---

## Part 1 — Software changes

### S1. Implement Memory-Aware Scheduling (§3.5.4) — **blocking**

The Abstract claims *"four different scheduling policies, including FCFS,
Priority-Based, Fair-Share, and Memory-Aware"*, and §3.5.4 defines it: *"jobs are
scheduled based on available memory to reduce fragmentation and prevent OOM
conditions."* It does not exist. `report_requirements.txt` tallies only P1–P4 and
silently omits §3.5.4, so the "0 missing" claim there is wrong.

Add `schedulers/memory_aware.py`: best-fit selection — among pending jobs, pick
the one whose `mem_mb` is the largest that still fits some partition's
`admit_headroom_mb` (largest-first packing minimises fragmentation), and place it
on the partition with the *tightest* fit (smallest headroom that still admits it),
falling back to arrival order when nothing fits. The loader auto-discovers it
([loader.py:50](Scheduling_control/loader.py:50)), so no wiring is needed.

### S2. Implement the unscheduled / naive baseline — **blocking**

Three separate places promise it:
- Abstract: *"evaluated the efficiency of the proposed system against the baseline
  system without scheduling"*
- §1.4 contribution 2: *"comparing scheduling policies and an unscheduled approach"*
- §1.5: *"comparing the baseline policies with a non-scheduled (naive) method of execution"*

Nothing in the repo bypasses admission control, so the project currently cannot
produce its headline result — that admission control reduces OOM failures. `pooled.py`
is *not* this baseline; every placement it proposes is still validated by
[`AdmissionController.decide`](Scheduling_control/admission.py:160).

The engine hard-codes the admission gate into the dispatch loop
([simulator.py:421](simulation/simulator.py:421)), so the baseline cannot be a
plugin. Add a bypass at engine level: an `admission=None` / `enforce_admission=False`
mode where every proposal commits immediately, or a `NullAdmissionController` that
always returns `ADMIT` and logs the decision it *would* have made. That second form
is better for Chapter 4 — it yields, for the same trace, the count of placements
admission control would have blocked alongside the OOMs that actually resulted.
Expose it as `ExperimentConfig.enforce_admission: bool = True` and as a
`--baseline` flag in `run_experiments.py`.

Note that this baseline is the only run that will produce real
`torch.cuda.OutOfMemoryError` events, so it is also the only validation that the
OOM path in [workload.py:128](Workload/workload.py:128) works.

### S3. Fix the partition/memory calibration — **blocking, invalidates Chapter 4**

`build_partitions` slices the *full* 4096 MB by floor division
([partition.py:146](simulation/partition.py:146)); admission admits
`mem_mb + margin ≤ cap × threshold`. With the defaults (threshold 90 %, margin
64 MB) the largest admissible job is:

| `num_partitions` | cap/partition | admission limit | max admissible `mem_mb` | heavy jobs (900–1400 MB) rejected |
|---|---|---|---|---|
| 2 | 2048 MB | 1843 MB | 1779 MB | 0 % |
| 3 (default) | 1365 MB | 1228 MB | 1164 MB | **≈47 %** |
| 4 | 1024 MB | 922 MB | 857 MB | **100 %** |

Two consequences:

1. **§3.9 v (OOM occurrence rate) is corrupted.** `oom_rate = (ooms + rejects) / n`
   ([results.py:366](simulation/results.py:366)). Rejects are decided by
   `_fits_anywhere` from `mem_mb` alone, so on a fixed trace *every scheduler
   rejects exactly the same jobs*. The default config gives all policies an
   identical ~5–6 % OOM-rate floor that has nothing to do with scheduling.
2. **The §3.8 iii partition sweep is confounded.** Going 2 → 3 → 4 partitions
   moves heavy-job rejection 0 % → 47 % → 100 %. Any Chapter 4 finding about
   partition count is really a finding about heavy jobs becoming unschedulable.

Fixes, all needed:
- Add `reserved_overhead_mb` (~700–900 MB) to `build_partitions`, subtracted from
  physical VRAM before slicing. The card reports 4096 MiB but Windows WDDM plus a
  CUDA context leave ~3.2–3.4 GB usable, so `sum(caps) ≤ 4095` is not actually a
  safe constraint — the ledger currently permits more concurrent demand than the
  device can hold.
- Recalibrate `HEAVY.mem_range_mb` (and `MEDIUM`) against the *smallest* partition
  count you intend to sweep, so heavy jobs remain admissible throughout the sweep.
  With 4 partitions off a 3300 MB budget the cap is 825 MB → heavy must sit
  below ~680 MB.
- Split the metric: report `oom_rate` (measured `torch.cuda.OutOfMemoryError`
  events) and `reject_rate` (placements admission prevented) as separate fields.
  Conflating prevented-OOMs with actual OOMs is the central validity problem in
  the evaluation, and separating them is what makes the baseline comparison in S2
  meaningful.

### S4. Add Job Completion Time — **blocking**

JCT is the Abstract's first metric, §1.5's first metric, and gets the longest
conceptual treatment in §2.2.7 (*"the time from when a job is submitted until it
is completed"*). It is absent from §3.9's seven metrics and absent from
`summarize()`. It is derivable (`end_s - arrival_s`) but never derived.

Add `jct_s` to result rows and `CSV_FIELDS`, plus `avg_jct`, `p95_jct` overall and
per tenant in `summarize()`. Rejected rows should carry `jct_s = None`, not 0 —
they never completed, and a 0 would deflate the mean.

### S5. Guard against the QUEUE livelock

`_dispatch_proc` exits only on `completed >= total_jobs`
([simulator.py:306](simulation/simulator.py:306)). The NVML cross-check
([admission.py:186](Scheduling_control/admission.py:186)) turns ADMIT into QUEUE
whenever `job.mem_mb + margin > nvml_free`. On a 4 GB laptop GPU where the Windows
desktop compositor and a browser hold VRAM, `nvml_free` can sit below the demand of
every pending job while the ledger is empty and nothing is running: every job
queues forever, no completion ever fires, and the run hangs with only heartbeat
snapshots. The comment at [workload.py:134](Workload/workload.py:134) shows a
related stall has already been hit once.

Add a no-progress watchdog: if no job is running and none has been admitted for
*k* consecutive dispatch passes, either force-admit the head of the queue (and log
a distinct `admission` outcome so Chapter 4 can report it) or abort the run with a
diagnostic. Silent hangs during a sweep are much worse than a logged forced admit.

### S6. Make compute-share allocation reachable (§3.4.1 ii)

`apply_compute_shares` raises the throttle only when `share < 1/N`
([partition.py:182](simulation/partition.py:182)), and `ExperimentConfig` can only
express *equal* slices — it has `num_partitions` but no `caps_mb` or `shares`.
So §3.4.1 ii is dead code in every experiment the harness can run.

Either add `caps_mb: list | None` and `shares: list | None` to `ExperimentConfig`
(threaded into `build_partitions`, which already accepts both) and sweep one
unequal-share condition, or state in §3.10 that compute-share allocation is
implemented but not exercised. The first is ~20 lines and gives Chapter 4 a real
result; the second is honest but weakens §3.4.1.

Related, for §3.10 either way: the throttle is binary, not proportional. A
throttled job sleeps a fixed `THROTTLE_SLEEP_S = 0.05` per iteration
([workload.py:67](Workload/workload.py:67)) regardless of how far its share falls
below fair share, so "compute share allocation" is enforced as on/off, not as a
ratio.

### S7. Add repetitions and dispersion to the sweep

`run_experiments.py` defaults to `--seeds 0` — one run per cell. Every timing
metric here is *measured wall-clock* on a shared Windows desktop, so single runs
carry no error bars and Chapter 4 can make no statistical claim. Add `--repeats N`
(or require ≥5 seeds) and have `write_sweep_summary` also emit an aggregated table
with mean and standard deviation per (condition, scheduler). Cheap to add, and
without it the comparison chapter is anecdote.

### S8. Housekeeping

- **Extra plugins.** `pinned`, `pooled`, `shortest_queue` are not report policies,
  and `run_experiments.py` runs *all* plugins by default — Chapter 4 tables would
  contain three policies the methodology never defines. Either always pass
  `--schedulers`, or make the default the four report policies plus the baseline.
- **`memory.md`** ([memory.md:12](memory.md:12)) still documents the retired
  `select_server(job, servers)` contract and never mentions partitions, admission
  control, or SimPy.
- **`architecture.md`** (72 lines) predates partitions, admission control, the
  SimPy engine, and the §3.9 metric set.
- **`boilerplate.py` / `multitenant.py`** at the repo root are superseded
  prototypes using a plain `simpy.Environment()`. Move them to an `attic/`
  directory or delete them — a marker leaving them importable next to the real
  engine is an easy way to describe the wrong system in the write-up.

---

## Part 2 — Chapter 1 changes

### C1.1 §1.3 Aim and Objectives — rewrite (highest-value edit in Chapter 1)

Current text has two problems. The aim sentence says "assess" twice and reads as a
fragment: *"to examine, implement, and assess a light GPU workload scheduling
method … and assess its performance with different types of work."*

More seriously, **objective 1 names Shortest Job First and Round Robin**, neither
of which is implemented, and neither of which appears anywhere in Chapter 3. The
objectives also omit most of what was actually built: the simulated partition
layer, the evaluation harness, and the dashboard.

Replace with five measurable objectives that map onto Chapter 3 sections:

1. Design and implement a software-simulated MIG partition layer with per-partition
   VRAM caps and compute shares on a single consumer GPU (→ §3.4).
2. Implement four scheduling policies — FCFS, priority-based, fair-share, and
   memory-aware (→ §3.5).
3. Implement a VRAM-aware admission controller that admits, queues, throttles, or
   rejects jobs against a configurable threshold and safety margin (→ §3.6).
4. Implement an NVML-based monitoring layer providing real per-run GPU telemetry
   (→ §3.3.2).
5. Evaluate all policies against an unscheduled baseline on identical seeded
   workloads across the §3.8 conditions, using the §3.9 metrics (→ §3.8, §3.9).

If you would rather keep SJF, add `schedulers/sjf.py` — it is ~10 lines
(`min(pending, key=lambda j: j.iters)`) and Chapter 2 §2.2.5 already discusses SJF
and its starvation behaviour at length, so it would earn its place. Round Robin
should be dropped from the objectives: §3.10 ii commits to no kernel-level
preemption, and RR without preemption is not RR. If you want it anyway, implement
it as round-robin *across tenants at dispatch granularity* and say so explicitly.

### C1.2 §1.4 — remove the duplicated sentence

*"Rather than proposing a new scheduling framework, this research contributes to
the development of a lightweight version of these computing infrastructures,
particularly in addressing issues of unfair access and scheduling strategies for
resource utilization in multi-tenant GPU systems"* repeats, nearly verbatim, the
sentence two lines above it. It also straddles the pp. 3–4 break, which is
probably how it survived. Keep one copy.

### C1.3 §1.4 contribution 3 — align the metric list

Says *"fairness, responsiveness, wait time, JCT, and stability"*. "Stability" is
only measurable once §3.9 vii is named; make the wording match the final §3.9 set
so the contribution claim and the results chapter use the same vocabulary.

### C1.4 §1.5 — align the metrics list and add the software-virtualization detail

§1.5 lists four metrics (JCT, waiting times, GPU utilization, OOM failure
percentages). The Abstract lists five. §3.9 lists seven, and omits JCT. All three
must name the same set (see X1). §1.5 should also state *how* the software
virtualization works — logical partitions with policy-enforced VRAM caps, no
hardware isolation — since that one sentence is what justifies the whole approach,
and §3.4.1 currently carries it alone.

### C1.5 §1.2 — add explicit research questions

Optional but strengthens the chapter: the problem statement builds to *"it is
necessary to replicate representative GPU-sharing and scheduling behaviour on a
smaller scale"* without ever stating the questions the experiments answer. Two or
three RQs here would give Chapter 4 something to answer directly.

---

## Part 3 — Chapter 2 changes

### C2.1 Add §2.4 Summary and Research Gap — **missing section**

§2.1 states: *"There are four sections in the literature review chapter"* and
*"Finally, rounding up the chapter will be a concise summary highlighting key gaps
that inform the proposed work."* The chapter has three sections (2.1, 2.2, 2.3)
and no summary. The promised gap statement is the bridge into Chapter 3 and an
examiner will look for it. Write it around the gap the project actually fills:
prior work (AntMan, MARBLE, MIG) targets multi-GPU clusters or hardware-partitioned
datacentre cards, so its findings are not reproducible on a single consumer GPU
without MIG — hence a software-partitioned, single-GPU testbed.

### C2.2 §2.2.7 — define the three metrics the code computes but Chapter 2 never introduces

§2.2.7 conceptually defines four metrics: JCT, waiting time, GPU utilization, OOM
rate. Its closing sentence says *"Together these metrics, JCT, waiting time, GPU
utilization and OOM rate form a multidimensional evaluation framework."* But §3.9
measures seven. Three arrive in Chapter 3 with no theoretical grounding:

- **Fairness index.** This is the worst gap. §2.2.3 discusses max-min fairness,
  dominant resource fairness, and Pareto efficiency — but the code computes
  **Jain's fairness index** ([results.py:212](simulation/results.py:212)), which
  Chapter 2 never mentions and which is not in the reference list. Add a paragraph
  defining `J = (Σx)² / (n·Σx²)`, its (1/n, 1] range, and why it is the right
  summary here, citing Jain, Chiu & Hawe (1984).
- **Job throughput.** Define as completed jobs per unit time and connect it to the
  throughput-vs-fairness trade-off §2.2.5 already sets up.
- **Partition-level resource stability.** Define as variability (standard
  deviation) of per-partition VRAM occupancy over the run, and tie it to the
  internal-fragmentation discussion in §2.2.4.

Then update the closing sentence to list all seven.

### C2.3 §2.2.4 — remove the duplicated sentence

*"Due to this when a workload triggers the execution of a long running kernel, it
can monopolize the device until completion"* appears twice in consecutive
sentences on p. 11.

### C2.4 §2.2.5 — cover fair-share and memory-aware as policies

The section covers FCFS, SJF, RR, and priority. Two of the four implemented
policies — fair-share and memory-aware — get no policy-level treatment. Fair-share
is reachable from the fairness material already in §2.2.5; memory-aware connects to
the bin-packing/fragmentation discussion in §2.2.4. Without this, §3.5.3 and §3.5.4
appear with no literature behind them.

### C2.5 §2.3.4 — cite the tools, and be accurate about what SimPy is used for

*"Examples of these tools are SimPy which have been used to model GPU job arrival,
execution times and resource constraints"* has no citation, and SimPy is not in the
reference list. Add a citation (SimPy documentation, or a paper that uses it for
this). Also: this project's use of SimPy is a **hybrid** — the event calendar
models arrival/queueing/dispatch while execution is real CUDA work — which is
unusual enough to be worth one sentence here, since it is the methodological choice
§3.2 rests on. PyTorch is also uncited throughout despite being the workload
generation layer; add Paszke et al. (2019).

### C2.6 Reference list and citation-format repairs

- **`Mace, J., BOdik, P., Fonseca, R., & Retro, M. M.`** (ref 34) is mangled.
  "Retro" is the *system name*, not an author, and "BOdik" is miscapitalised. The
  correct form is Mace, Bodik, Fonseca & Roy, *Retro: Targeted Resource Management
  in Multi-tenant Distributed Systems*, NSDI '15. Also has no year and no page
  numbers.
- **`Ru, J., Grundy, J., & Keung, J. (2-14)`** — typo for 2014, in §2.2.3.
- **Author name inconsistency:** "Al Jahadali" (§2.2.3), "Al Jahdali" (§2.3.1),
  "AlJahdali" (ref 31). Pick the reference-list form.
- **`Wassemann`** (§2.3.4) vs **`Wassermann`** (ref 36).
- **`(Gao, et al)`** in §2.2.2 — no year. Should be Gao et al. (2020).
- **`Jia, Yang, Grundy, & Keung, 2017; Jia, Yang, Grundy, Keung, & Hao, 2021`** in
  §2.2.3 is a doubled citation of one 2021 paper (ref 32); the 2017 variant does
  not exist in the reference list.
- **`Bakhoda, 2009`** appears both with and without co-authors within §2.2.1.
- **`(Gupta, et al 2009; Wael, & Madi, 2025)`** in §1.2 — inconsistent comma
  placement; the same two styles alternate throughout Chapters 1–2. Pick one.
- Numbered reference list is unordered and in mixed formats. If the department
  requires APA, alphabetise and drop the numbering.

### C2.7 Abbreviations page

- **`MIG - Multi-instance Graphics Processing units`** → "Multi-Instance GPU".
- **`PRT - Predictable Response Time`** is defined but never used anywhere in the
  document. Remove it, or use it.
- `SJF` and `RR` are listed — keep only if C1.1 keeps them.

### C2.8 Typographical pass, Chapter 2

`continous` → continuous (§2.3.4); `activation's` → activations (§2.2.2);
`withing` → within (§2.2.3); `higlights` → highlights, `presense` → presence,
`peformance` → performance (§2.3.1); `Scheduliing` → Scheduling; `PCle` → PCIe
(§2.3.2). §2.2.2's *"A key constraint in practical GPU-accelerated computing
sharing is VRAMs"* needs rewriting.

---

## Part 4 — Chapter 3 changes

### C3.1 §3.7 Implementation Details — three factual corrections

- **`GPU model: RTX-2030`** is not a real product. The machine has an **NVIDIA
  GeForce RTX 2050** with 4096 MiB. Also state the driver version (592.82), the
  CUDA/PyTorch version, and the Python version — a methodology chapter needs them
  for replicability.
- **`Dear ImGui (Python binding)`** — the UI is **Tkinter** (`UI/dashboard.py`,
  1047 lines). Change the entry and add one sentence on why: Tkinter ships with
  CPython, so the dashboard has no GUI dependency to install, which matters for a
  tool whose selling point is being lightweight.
- **`Total VRAM: 4gb`** → "4 GB (4096 MiB)". If S3 is done, state the reserved
  overhead and the *usable* budget the partitions are sliced from — that number,
  not the 4096, is what the caps derive from.

The SimPy claim in §3.7 c stands as written: the engine genuinely runs on
`simpy.rt.RealtimeEnvironment` ([simulator.py:628](simulation/simulator.py:628)).

### C3.2 §3.3.4 User Interface Layer — Dear ImGui → Tkinter

Same change as above. The five listed display items are all genuinely implemented,
so only the library name changes.

### C3.3 §3.3.1 — list four policies, not three

§3.3.1 lists three supported policies (FCFS, priority, fair-share); §3.5 defines
four; the Abstract claims four. Add memory-aware here, and add the unscheduled
baseline as an explicitly non-policy control condition.

### C3.4 §3.5.3 Fair-Share Scheduling — the description does not match the design

Current text: *"GPU resources are distributed proportionally and fairly across
**partitions** to prevent starvation."* The implemented policy balances service
across **tenants**, not partitions ([fair_share.py](schedulers/fair_share.py)) —
which is the correct design, since tenants are the fairness principals in a
multi-tenant system and §2.2.3 frames fairness that way throughout.

Rewrite to describe what is implemented and why: each tenant's consumed service is
tallied as GPU-seconds plus VRAM GB-seconds, and the pending job of the
least-served tenant is dispatched first, with ties broken by fewest in-flight jobs
and then arrival order. State that the two dimensions are summed — this is a design
decision worth defending, since it is a simplification of the dominant-resource
fairness §2.2.3 introduces, and an examiner may well ask why DRF was not used.

### C3.5 §3.5.4 Memory-Aware Scheduling — expand once S1 exists

One sentence currently. Specify the placement rule (best-fit / tightest-fit), what
"available memory" means (the partition's admission headroom: threshold-scaled cap
minus ledger usage minus safety margin), and how it differs from admission control
— the scheduler *chooses* placements to minimise fragmentation, the controller
*vetoes* unsafe ones. That distinction is the cleanest way to justify having both.

### C3.6 §3.6 Admission Control Design — the four steps are incomplete

The implemented controller ([admission.py](Scheduling_control/admission.py)) does
more than the four steps describe, and the report understates its own contribution:

- **Step iv is wrong as written.** *"Compare the projected VRAM usage with partition
  capacity"* — the code compares against `capacity × threshold_pct`
  ([admission.py:174](Scheduling_control/admission.py:174)). Since the admission
  threshold percentage is a §3.8 experimental variable, the threshold belongs in the
  step, not just in §3.4.3.
- **REJECT is undocumented.** §3.6 gives two outcomes ("queued or throttled"); the
  code has four: ADMIT, QUEUE, REJECT, THROTTLE_RUNNING. REJECT — a job whose
  demand cannot fit *any* partition even when empty — is the outcome that most
  directly supports the OOM-avoidance claim and it is not in the report at all.
- **The NVML cross-check is undocumented.** Step 1 in the code reads the
  reservation ledger *and* cross-checks it against live NVML free VRAM, taking the
  more pessimistic value ([admission.py:186](Scheduling_control/admission.py:186)).
  This is the only place where the hybrid design's real-telemetry half feeds back
  into a scheduling decision — it deserves a named step.
- **The throttle-release rule is undocumented.** `reconsider()` clears a
  controller-raised throttle once the partition drops back under threshold. Without
  it, §3.6's throttling reads as one-way.
- **The "heavy job running" test is unstated.** THROTTLE_RUNNING requires that the
  largest live reservation exceeds `HEAVY_FRACTION = 0.5` of the cap. §3.6 says
  "throttled if already running" without saying what qualifies.

Rewrite §3.6 as a numbered algorithm with all four outcomes and both experimental
parameters (`threshold_pct`, `safety_margin_mb`) named.

### C3.7 §3.4.2 Workload Arrival Model — name the distributions

Currently three bullet labels with no distributions. A methodology chapter must
state them:

- inter-arrival times ~ Exponential(λ) per tenant, i.e. a Poisson arrival process
  ([workload.py:267](Workload/workload.py:267)), with λ the §3.8 ii arrival rate;
- memory demand ~ Uniform over the workload class's MB range;
- compute intensity ~ Uniform over the class's iteration range, scaled by a global
  multiplier;
- all draws come from a single seeded `random.Random(seed)`, so (mix, tenants, rate,
  seed) determines the trace exactly.

That last point is the replicability claim §3.8 depends on and it is currently made
nowhere in Chapter 3.

### C3.8 §3.3.3 / §3.4.2 — add the workload class table

§3.3.3 names four workload criteria and three classes but gives no values, so the
workload is unreproducible from the report. Add a table:

| Class | Matrix N | Batch | Iterations | VRAM demand (MB) |
|---|---|---|---|---|
| Light | 128 | 1 | 100–200 | 100–250 |
| Medium | 384 | 2 | 300–450 | 400–700 |
| Heavy | 768 | 4 | 600–800 | 900–1400 |

(Values from [workload.py:42](Workload/workload.py:42) — **update after S3
recalibration.**) Add a sentence on how memory demand is realised: each job
allocates a ballast tensor of its declared `mem_mb` alongside its GEMM operands, so
the footprint NVML sees is the footprint admission control reasoned about. This is
what makes the hybrid claim in §3.2 substantive rather than nominal, and it is
currently unstated.

### C3.9 §3.4.1 / §3.4.3 — state the actual partition configuration

§3.4.1 names the two partition attributes; §3.4.3 names three constraints. Neither
gives values. Add: N equal slices by floor division of the usable VRAM budget,
equal compute shares of 1/N, the default N = 3, and the resulting cap in MB. Add
the reserved-overhead constraint from S3 as a fourth §3.4.3 constraint — the
constraint that `sum(caps) ≤ physical VRAM` is *not* sufficient on a WDDM consumer
card, and saying so is a genuine methodological finding.

§3.4.1 should also state how the compute share is enforced: cooperatively, by a
throttle flag the job checks between kernels, with no kernel-level preemption
(consistent with §3.10 ii).

### C3.10 §3.8 Experimental Setup — currently four bullets, needs to be a protocol

Add:
- the actual swept values per axis (mixes, arrival rates, partition counts,
  thresholds, margins, seeds) and the resulting run count;
- **the unscheduled baseline as a fifth condition** (S2);
- **repetitions**: how many seeds per cell, and that results are reported as
  mean ± sd (S7). Single-run wall-clock measurements on a shared Windows desktop
  cannot support a comparison chapter;
- the fair-comparison mechanism: the trace is regenerated per scheduler from the
  same seed, so all policies face a byte-identical workload. `test_replicability.py`
  verifies this and is worth citing.

### C3.11 §3.7 — add a reproducibility / artifacts subsection

Chapter 3 never says what a run produces. Add a short subsection listing the module
layout (`Workload/`, `Scheduling_control/`, `simulation/`, `GPUMonitoring/`, `UI/`,
`schedulers/`), the plugin mechanism (drop a file in `schedulers/`, discovered at
load), and the per-run artifacts: `config.json`, `jobs.csv`, `admission_log.csv`,
`throttle_log.csv`, `telemetry.csv`, `summary.csv`, plus `sweep_summary.csv` across
a grid. Mention `run_experiments.py` as the headless harness that generates Chapter
4's data, and the three test scripts. This is what an examiner needs to believe the
results are reproducible, and it costs half a page.

### C3.12 §3.9 Evaluation Metrics — add JCT, split the OOM metric, define each formula

- **Add JCT** as metric i or ii (S4). It is the Abstract's headline metric and
  §2.2.7's most-developed concept; omitting it from §3.9 is the single largest
  internal inconsistency in the write-up.
- **Split metric v** into measured OOM rate and admission rejection rate (S3).
- **Give each metric its formula.** Seven bare labels are not a measurement
  protocol. In particular, "GPU utilization percentage" is ambiguous: the code
  reports *two* different quantities — busy-time ÷ (capacity × makespan) from the
  job log, and mean NVML device utilization from telemetry
  ([results.py:296](simulation/results.py:296)). These are not the same measurement
  and Chapter 4 will need to distinguish them, so §3.9 should define both.
- Name Jain's index explicitly for metric vi, and state the standard deviation of
  per-partition VRAM occupancy for metric vii.

### C3.13 §3.10 Methodological Limitations — five items, should be eight or nine

The five stated limitations all hold. Add the ones the implementation actually
incurs, all of which strengthen rather than weaken the chapter:

- **Ledger-based, not enforced, memory isolation.** A partition's VRAM cap is a
  policy accounting limit; nothing prevents a job from allocating past it. Only
  admission control keeps the ledger honest.
- **Cooperative, binary throttling.** A throttled job sleeps a fixed interval
  between kernels; the throttle is on/off, not proportional to the share deficit
  (S6).
- **Device-wide telemetry cannot be attributed per partition.** All partitions
  execute on the same `cuda:0` in one process, so NVML reports one PID and one
  device-wide VRAM figure. Per-partition usage in §3.9 vii comes from the
  reservation ledger, not from measurement. The report should be explicit about
  which of the seven metrics are *measured* and which are *modelled* — this is the
  most important honesty point in the chapter, and `telemetry.csv` records both
  side by side, so the gap between them is itself reportable.
- **Real-time execution bounds experiment scale.** `RealtimeEnvironment(factor=1)`
  pins simulated time to wall-clock time, so a run takes as long as the work does.
  This is the price of the hybrid design and it caps how large a sweep is feasible.
- **Wall-clock measurements on a non-dedicated desktop.** Windows background load
  and the desktop compositor's own VRAM use inject variance no seed controls
  (hence S7's repetitions).

---

## Part 5 — Figures

- **Figure 3.1 (component diagram)** — replace the Dear ImGui component with
  Tkinter. Add the simulated-partition layer if it is not already a distinct box:
  the current 4-component architecture in §3.3 does not include §3.4's simulation
  model layer, which is a fifth component in the code (`simulation/partition.py`).
- **Figure 3.2 (partition deployment)** — annotate with the real numbers: 3
  partitions, the per-partition cap in MB, and the reserved overhead, all on one
  physical `cuda:0`.
- **Figure 3.3 (admission activity diagram)** — must gain the REJECT branch, the
  NVML cross-check decision node, and the throttle-release transition (C3.6). As
  drawn from a two-outcome §3.6 it cannot match the implemented controller.

---

## Part 6 — Cross-document consistency

### X1. One metric list, used everywhere

Four places name metrics and no two agree:

| Location | Metrics named |
|---|---|
| Abstract | JCT, avg waiting time, GPU utilization, OOM failures, fairness index (5) |
| §1.4 | fairness, responsiveness, wait time, JCT, stability (5, different) |
| §1.5 | JCT, waiting times, GPU utilization, OOM failure % (4) |
| §3.9 | GPU util %, avg VRAM util, wait time, throughput, OOM rate, fairness, partition stability (7, **no JCT**) |

Fix §3.9 first (C3.12), then make the other three quote it.

### X2. One policy count

Abstract says four; §3.3.1 says three; §3.5 defines four; the code has three of
the four plus three undocumented extras. After S1 and C3.3, everything should read
four policies plus one unscheduled baseline.

### X3. Abstract wording

*"utilizing the SimPy library, which eliminates the need for hardware-based
Multi-Instance GPU (MIG) support"* — SimPy does not eliminate the need for MIG;
**software-level logical partitioning** does, and SimPy is the discrete-event
framework that models it. Reword. Also update "Dear ImGui-based visualization
dashboard" → Tkinter, and add JCT-consistent metric wording per X1.

### X4. Chapter 3 typographical pass

`and experimental system design` → *an* experimental (§3.2); `actial` → actual,
`arival` → arrival, `throuput` → throughput (§3.2); `trottle` → throttle,
`Scheduliing` → Scheduling, `the current memory GPU memory usage` (§3.3.1);
`ussage` → usage (§3.3.2); `Imgui` → ImGui (§3.3.4); `seperation` → separation
(§3.4.1); `constraints` → constrains (§3.4.3). Section headings §3.4.1 and §3.5.1
end with stray full stops.

---

## Suggested order

1. **S3** (calibration) — everything measured depends on it, and it is the one
   issue that would silently invalidate a finished Chapter 4.
2. **S1, S2** (memory-aware policy, unscheduled baseline) — the two claims the
   project currently cannot support.
3. **S4, S5** (JCT, livelock guard) — small, and S5 prevents a hung overnight sweep.
4. **S7** then run the real sweep.
5. Report edits: C3.1/C3.2/C3.6 (factual corrections) → C1.1/C2.1 (the two
   structural rewrites) → everything else.
6. **S6, S8** and the typographical passes last.

The three heaviest report edits are §1.3 (objectives), §2.4 (new section), and §3.6
(admission algorithm). Nothing else in Chapters 1–3 needs more than a paragraph.
