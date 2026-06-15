#!/usr/bin/env python3
"""research/scorer_calibration.py — hold the §C15.3 idea-scorer accountable.

The scorer predicts a win-probability for each idea; this module scores the
SCORER against realised outcomes (Brier, log-loss, reliability bins, ECE) and
returns a shrinkage factor that pulls an over-confident scorer's outputs back
toward the prior until it earns trust. This is "taste is calibrated by being
wrong" made into a metric; it runs in /weekly-retro over the ledger's
predicted-vs-realised history.

Pure stdlib, CPU-only, deterministic — CI-safe and unit-tested.

Inputs are paired lists: preds = predicted win-probabilities in [0,1];
outcomes = realised binary outcomes in {0,1} (1 = the run was a win).
"""
from __future__ import annotations

import math

EPS = 1e-15


def _check(preds, outcomes):
    if len(preds) != len(outcomes):
        raise ValueError("preds and outcomes must be the same length")
    if not preds:
        raise ValueError("need at least one (pred, outcome) pair")
    for p in preds:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"prediction {p!r} not in [0,1]")
    for y in outcomes:
        if y not in (0, 1, 0.0, 1.0):
            raise ValueError(f"outcome {y!r} must be 0 or 1")


def brier(preds, outcomes) -> float:
    """Mean squared error of probabilistic predictions. 0 = perfect, 0.25 =
    always-0.5, 1 = confidently wrong."""
    _check(preds, outcomes)
    return sum((p - y) ** 2 for p, y in zip(preds, outcomes)) / len(preds)


def log_loss(preds, outcomes) -> float:
    """Mean negative log-likelihood (clipped to avoid inf on a confident miss)."""
    _check(preds, outcomes)
    total = 0.0
    for p, y in zip(preds, outcomes):
        p = min(max(p, EPS), 1 - EPS)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(preds)


def reliability_bins(preds, outcomes, n_bins=10):
    """Group predictions into equal-width probability bins; for each non-empty
    bin report mean predicted vs mean observed frequency (a calibration curve)."""
    _check(preds, outcomes)
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        # last bin is closed on the right so p == 1.0 lands somewhere
        members = [(p, y) for p, y in zip(preds, outcomes)
                   if (lo <= p < hi) or (b == n_bins - 1 and p == 1.0)]
        if not members:
            continue
        mp = sum(p for p, _ in members) / len(members)
        mo = sum(y for _, y in members) / len(members)
        bins.append({"bin": b, "lo": lo, "hi": hi, "count": len(members),
                     "mean_pred": mp, "mean_obs": mo, "gap": abs(mp - mo)})
    return bins


def ece(preds, outcomes, n_bins=10) -> float:
    """Expected Calibration Error: count-weighted mean |mean_pred − mean_obs|
    across bins. 0 = perfectly calibrated."""
    n = len(preds)
    return sum(b["count"] / n * b["gap"] for b in reliability_bins(preds, outcomes, n_bins))


def shrinkage_factor(calibration_ece: float, k: float = 2.0) -> float:
    """How much to TRUST the scorer's spread: 1/(1+k·ECE) ∈ (0,1]. ECE=0 → 1.0
    (trust fully); a badly-calibrated scorer → →0 (shrink its scores hard)."""
    if calibration_ece < 0:
        raise ValueError("ece must be >= 0")
    return 1.0 / (1.0 + k * calibration_ece)


def calibrate(p: float, shrink: float) -> float:
    """Shrink a raw prediction toward the 0.5 prior by `shrink` ∈ (0,1]:
    0.5 + shrink·(p−0.5). shrink=1 leaves p unchanged; shrink→0 → 0.5."""
    return 0.5 + shrink * (p - 0.5)


def report(preds, outcomes, n_bins=10) -> dict:
    """One-call summary for the weekly retro."""
    e = ece(preds, outcomes, n_bins)
    return {"n": len(preds), "brier": brier(preds, outcomes),
            "log_loss": log_loss(preds, outcomes), "ece": e,
            "shrinkage": shrinkage_factor(e),
            "bins": reliability_bins(preds, outcomes, n_bins)}
