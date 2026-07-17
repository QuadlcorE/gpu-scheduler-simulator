"""
Experiment configuration -- the report's §3.8 experimental design, as data.

An :class:`ExperimentConfig` captures *everything* §3.8 varies between
experimental conditions, in one serializable object:

    i.   workload intensity mix   -- ``counts`` (light/medium/heavy jobs per
                                     tenant) and ``num_tenants``
    ii.  arrival rate             -- ``arrival_rate`` (mean jobs/s per tenant)
    iii. number of partitions     -- ``num_partitions`` (equal VRAM slices)
    iv.  admission threshold      -- ``threshold_pct`` (+ ``safety_margin_mb``)
         scheduling policy        -- ``scheduler`` (display name of the plugin)
         randomness               -- ``seed`` (fully determines the job trace)

It round-trips to/from JSON (:meth:`to_json` / :meth:`from_json`,
:meth:`save` / :meth:`load`), so the config written next to a run's CSVs is a
citable artifact: the exact condition that produced the data, re-runnable
byte-for-byte from the file alone.

The same (config minus scheduler) + same seed always generates the identical
job trace (see Workload.workload.generate_trace), which is what makes runs of
different schedulers under one config comparable -- "each scheduler was
subjected to the same workload".
"""

import json
from dataclasses import asdict, dataclass, field

from Workload.workload import DEFAULT_ARRIVAL_RATE, build_experiment

# The canonical workload-mix keys, in light->medium->heavy order (matches
# Workload.workload.WORKLOAD_TYPES).
MIX_KEYS = ("light", "medium", "heavy")


def _default_counts():
    return {"light": 4, "medium": 3, "heavy": 1}


@dataclass
class ExperimentConfig:
    """One experimental condition (§3.8), serializable to/from JSON.

    ``threshold_pct`` is stored as a fraction in (0, 1]; values > 1 are
    accepted as percentages (90 -> 0.9), mirroring AdmissionController, so
    both "0.9" and "90" mean the same condition in a config file."""

    num_tenants: int = 3
    # Jobs of each workload type *per tenant* (the intensity mix, §3.8 i).
    counts: dict = field(default_factory=_default_counts)
    # Mean jobs per second per tenant (§3.8 ii).
    arrival_rate: float = DEFAULT_ARRIVAL_RATE
    # Equal-slice partition count (§3.8 iii); the engine splits physical VRAM.
    num_partitions: int = 3
    # Admission threshold as a fraction of each partition's cap (§3.8 iv).
    threshold_pct: float = 0.9
    # Fixed headroom (MB) added before the threshold comparison (§3.6 step 3).
    safety_margin_mb: int = 64
    # Display name of the scheduler plugin (Scheduling_control.loader key).
    scheduler: str = "FCFS (first-come-first-served)"
    # Trace seed: fully determines arrivals, sizes, demands, and priorities.
    seed: int = 0

    def __post_init__(self):
        if self.threshold_pct > 1.0:  # accept "90" for 90 %
            self.threshold_pct = self.threshold_pct / 100.0
        self.validate()

    # ---- validation ---------------------------------------------------------
    def validate(self):
        """Raise ValueError on any out-of-range field. Called from
        __post_init__, so a constructed config is always valid."""
        if self.num_tenants < 1:
            raise ValueError(f"num_tenants must be >= 1: {self.num_tenants}")
        unknown = set(self.counts) - set(MIX_KEYS)
        if unknown:
            raise ValueError(f"unknown workload types in counts: {sorted(unknown)}")
        if any(int(v) < 0 for v in self.counts.values()):
            raise ValueError(f"workload counts must be >= 0: {self.counts}")
        if sum(int(v) for v in self.counts.values()) < 1:
            raise ValueError("counts must include at least one job per tenant")
        if self.arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be > 0: {self.arrival_rate}")
        if self.num_partitions < 1:
            raise ValueError(f"num_partitions must be >= 1: {self.num_partitions}")
        if not 0.0 < self.threshold_pct <= 1.0:
            raise ValueError(f"threshold_pct must be in (0, 1]: {self.threshold_pct}")
        if self.safety_margin_mb < 0:
            raise ValueError(f"safety_margin_mb must be >= 0: {self.safety_margin_mb}")
        if not self.scheduler:
            raise ValueError("scheduler must be a non-empty plugin display name")

    # ---- naming (for results directories / sweep tables) ---------------------
    def condition_label(self):
        """Filesystem-safe name for the *condition* (everything but the
        scheduler): one results directory per condition, holding one run per
        scheduler underneath."""
        mix = "-".join(str(int(self.counts.get(k, 0))) for k in MIX_KEYS)
        return (f"mix{mix}_tenants{self.num_tenants}"
                f"_rate{self.arrival_rate:g}_parts{self.num_partitions}"
                f"_thr{int(round(self.threshold_pct * 100))}"
                f"_margin{self.safety_margin_mb}_seed{self.seed}")

    def scheduler_slug(self):
        """Filesystem-safe form of the scheduler display name."""
        slug = "".join(c if c.isalnum() else "_" for c in self.scheduler)
        return slug.strip("_").lower() or "scheduler"

    # ---- building the run's inputs -------------------------------------------
    def build_workload(self):
        """Build this condition's (tenants, jobs): the seeded trace every
        scheduler under this config is subjected to."""
        counts = {k: int(self.counts.get(k, 0)) for k in MIX_KEYS}
        return build_experiment(self.num_tenants, counts, self.seed,
                                arrival_rate=self.arrival_rate)

    # ---- (de)serialization ----------------------------------------------------
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Build a config from a plain dict (e.g. parsed JSON). Unknown keys
        raise, so a typo in a config file fails loudly instead of silently
        running the default condition."""
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ExperimentConfig fields: {sorted(unknown)}")
        return cls(**data)

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def save(self, path):
        """Write the config as JSON next to a run's CSVs (the citable
        artifact). Returns `path`."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json() + "\n")
        return path

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())
