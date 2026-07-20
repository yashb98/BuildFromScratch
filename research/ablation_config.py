#!/usr/bin/env python3
"""research/ablation_config.py — the validated cohort builder for an RS-rigorous
ablation, so the NEXT run is a 'clean win', not a 'scoped win'.

The audit's three confounds in the prior ablation were:
  1. n<5 seeds  -> the seed CI was too wide to separate a real effect from noise.
  2. a SHARED hyperparameter (one wd / one LR for both arms) -> a technique that
     merely shifts the optimal wd/LR looks like a win or loss for the wrong reason.
  3. a FIXED train/val split -> seeds only captured init+shuffle noise, never the
     corpus-resampling variance, so the CI understated the true uncertainty.

This module builds + VALIDATES the cohort that closes all three:
  * each ARM carries its OWN hyperparameters (per-arm wd/LR, etc.), so every arm
    runs at ITS optimum and the comparison is confound-free.
  * n_seeds is first-class; a 'clean' verdict tier REQUIRES n_seeds >= 5, and
    validate() hard-REFUSES n_seeds < 3 (one or two seeds can never be a result).
  * randomized_split toggles data_decontam's per-cell split: randomized=True =>
    every (arm, seed) cell takes a DIFFERENT doc-disjoint split (full end-to-end
    variance); False => the legacy fixed split (init+shuffle noise only).
  * cells() enumerates the (arm, seed) grid with the EXACT split seed each cell
    must use (via data_decontam.cell_split_seed), so the launcher is deterministic.

Stdlib-only, pure, deterministic — no torch, no network, no model loads. The
contract is unit-tested (test_ablation_config.py) before any GPU time is spent.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from data_decontam import cell_split_seed

# Verdict tiers by seed count. A 'clean' (publishable, confound-free) result needs
# a real seed cohort; below that the verdict is explicitly demoted.
CLEAN_MIN_SEEDS = 5     # >=5 seeds -> eligible for the 'clean' tier
MIN_SEEDS = 3           # <3 seeds -> validate() refuses to build the cohort


@dataclass(frozen=True)
class Arm:
    """One arm of the ablation. `name` identifies it; `is_baseline` marks the
    control. `hparams` are PER-ARM (e.g. {'lr': 3e-4, 'wd': 0.1}) so each arm runs
    at its own tuned optimum — that is what turns a confounded comparison into a
    clean one. `hparams` must be a non-empty mapping of finite numeric (or string)
    values; an empty hparams means 'shared/untuned', which is exactly the confound
    we forbid."""
    name: str
    hparams: dict = field(default_factory=dict)
    is_baseline: bool = False

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("arm name must be a non-empty string")
        if not isinstance(self.hparams, dict) or not self.hparams:
            raise ValueError(
                f"arm {self.name!r} has no per-arm hparams — every arm must carry "
                "its OWN tuned hyperparameters (e.g. lr/wd) to avoid the shared-hparam confound")
        for k, v in self.hparams.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"arm {self.name!r}: hparam keys must be non-empty strings")
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                if v != v or v in (float("inf"), float("-inf")):
                    raise ValueError(f"arm {self.name!r}: hparam {k!r} is non-finite")
            elif not isinstance(v, str):
                raise ValueError(
                    f"arm {self.name!r}: hparam {k!r} must be number/bool/str, got {type(v).__name__}")


@dataclass
class CohortConfig:
    """A multi-seed, per-arm-tuned ablation cohort.

    Fields:
      arms          : list[Arm]  — exactly one must be the baseline; >=1 treatment.
      n_seeds       : int        — training seeds PER arm (>=3 to build at all;
                                    >=5 for the 'clean' tier).
      base_split_seed : int      — the run's base doc-disjoint split seed.
      val_fraction  : float      — fraction of docs routed to val (0,1).
      randomized_split : bool    — True => a DIFFERENT split per (arm,seed) cell
                                    (full variance); False => fixed split.
      seed_start    : int        — first training seed; cell seeds are
                                    seed_start .. seed_start+n_seeds-1.
      direction     : str        — 'lower_is_better' (PPL/loss) or 'higher_is_better'.
      notes         : str        — free-form provenance.
    """
    arms: list
    n_seeds: int = CLEAN_MIN_SEEDS
    base_split_seed: int = 0
    val_fraction: float = 0.05
    randomized_split: bool = False
    seed_start: int = 0
    direction: str = "lower_is_better"
    notes: str = ""

    def validate(self) -> "CohortConfig":
        """Refuse a cohort that cannot yield an honest stat. Returns self so it
        chains. Raises ValueError on any violation; NEVER silently downgrades."""
        if self.n_seeds < MIN_SEEDS:
            raise ValueError(
                f"n_seeds={self.n_seeds} < {MIN_SEEDS}: one or two seeds is not a "
                "result — a cohort needs at least 3 training seeds to estimate variance")
        if not isinstance(self.arms, (list, tuple)) or len(self.arms) < 2:
            raise ValueError("a cohort needs >=2 arms (one baseline + >=1 treatment)")
        arms = list(self.arms)
        for a in arms:
            if not isinstance(a, Arm):
                raise ValueError(f"each arm must be an Arm, got {type(a).__name__}")
            a.validate()
        names = [a.name for a in arms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate arm names: {names}")
        n_base = sum(1 for a in arms if a.is_baseline)
        if n_base != 1:
            raise ValueError(f"exactly one arm must have is_baseline=True (found {n_base})")
        if not (0.0 < self.val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0,1)")
        if self.direction not in ("lower_is_better", "higher_is_better"):
            raise ValueError(
                f"direction must be lower_is_better|higher_is_better, got {self.direction!r}")
        if not isinstance(self.randomized_split, bool):
            raise ValueError("randomized_split must be a bool")
        return self

    # ----- derived views (all assume validate() has been or will be called) -----

    @property
    def is_clean_tier(self) -> bool:
        """True iff this cohort is eligible for a 'clean' (publishable) verdict:
        >=5 seeds AND a randomized per-cell split (so the CI reflects corpus
        resampling, not just init+shuffle)."""
        return self.n_seeds >= CLEAN_MIN_SEEDS and self.randomized_split

    def verdict_tier(self) -> str:
        """The verdict tier this cohort's DESIGN can support (independent of the
        eventual numbers): 'clean' (>=5 seeds + randomized split), 'scoped'
        (enough seeds but fixed split and/or <5 seeds), or 'invalid' (<3 seeds)."""
        if self.n_seeds < MIN_SEEDS:
            return "invalid"
        if self.is_clean_tier:
            return "clean"
        return "scoped"

    def baseline_arm(self) -> Arm:
        for a in self.arms:
            if a.is_baseline:
                return a
        raise ValueError("no baseline arm — call validate() first")

    def treatment_arms(self) -> list:
        return [a for a in self.arms if not a.is_baseline]

    def seeds(self) -> list:
        return list(range(self.seed_start, self.seed_start + self.n_seeds))

    def cells(self) -> list:
        """Enumerate the full (arm, seed) grid as cell dicts the launcher consumes.
        Each cell carries its arm name, the arm's per-arm hparams, the training
        seed, and the EXACT split seed to use (fixed => base_split_seed for every
        cell; randomized => a distinct per-cell split seed via cell_split_seed).
        Deterministic and order-stable (arms outer, seeds inner)."""
        out = []
        for arm in self.arms:
            for s in self.seeds():
                split_seed = cell_split_seed(self.base_split_seed, s,
                                             randomized=self.randomized_split)
                out.append({
                    "arm": arm.name,
                    "is_baseline": arm.is_baseline,
                    "hparams": dict(arm.hparams),
                    "train_seed": s,
                    "split_seed": split_seed,
                    "val_fraction": self.val_fraction,
                    "randomized_split": self.randomized_split,
                })
        return out

    def to_dict(self) -> dict:
        """Serializable form for the ledger / on-disk cohort spec (provenance)."""
        d = asdict(self)
        d["verdict_tier"] = self.verdict_tier()
        d["is_clean_tier"] = self.is_clean_tier
        d["n_cells"] = len(self.arms) * self.n_seeds
        return d


def build_cohort(arms, *, n_seeds: int = CLEAN_MIN_SEEDS, base_split_seed: int = 0,
                 val_fraction: float = 0.05, randomized_split: bool = False,
                 seed_start: int = 0, direction: str = "lower_is_better",
                 notes: str = "") -> CohortConfig:
    """Build AND validate a cohort in one call (raises on any invalid spec).
    `arms` is a list of Arm objects OR plain dicts ({'name','hparams','is_baseline'})
    for convenience from a yaml/json config."""
    norm = []
    for a in arms:
        if isinstance(a, Arm):
            norm.append(a)
        elif isinstance(a, dict):
            norm.append(Arm(name=a.get("name"),
                            hparams=a.get("hparams", {}) or {},
                            is_baseline=bool(a.get("is_baseline", False))))
        else:
            raise ValueError(f"arm must be Arm or dict, got {type(a).__name__}")
    cfg = CohortConfig(arms=norm, n_seeds=n_seeds, base_split_seed=base_split_seed,
                       val_fraction=val_fraction, randomized_split=randomized_split,
                       seed_start=seed_start, direction=direction, notes=notes)
    return cfg.validate()
