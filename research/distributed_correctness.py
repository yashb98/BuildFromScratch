#!/usr/bin/env python3
"""research/distributed_correctness.py — the §C21 distributed-correctness GATE.

A multi-GPU (FSDP2 / TP / CP) training run is `inconclusive` BY CONSTRUCTION
unless it first proves it computes the SAME thing the single-GPU baseline does.
A distributed run can be fast AND wrong — a sharding/all-reduce/cast bug shifts
the loss while still descending plausibly — and an MFU number from a wrong run
is a lie. So this gate runs FIRST: before any tokens/sec or MFU figure from a
distributed run is trusted, its loss (or any eval metric) must match the
single-GPU baseline within a noise floor.

The law (§C21):

    |distributed_metric − single_gpu_baseline| <= tolerance

where `tolerance` is NOT invented — it is derived from the §C17 seed-noise floor
(the same machinery `eval_stats.py` uses for win/loss verdicts). The reason: a
distributed run that lands within the run-to-run training-seed scatter of the
baseline is indistinguishable from "the same computation, different RNG"; a run
that lands OUTSIDE that scatter is doing something different and must be caught.

This module is PURE and CPU-testable: it takes synthetic numbers (no torch, no
CUDA, no model, no I/O) and returns a structured pass/fail verdict. The GPU work
that PRODUCES the numbers lives off-box (`remote-launcher` / torchtitan); this is
the correctness logic that JUDGES them, fully unit-tested on CPU.

It reuses `eval_stats.py` for the noise-floor estimate so the tolerance is the
same statistic the rest of the system already trusts.
"""
from __future__ import annotations

import math

import eval_stats as es

# Below this absolute floor we never let the tolerance collapse to ~0. Two
# identical-seed baseline replicates can have near-zero scatter purely by luck;
# a literally-zero tolerance would then reject a correct distributed run for a
# 1e-7 floating-point reassociation difference (all-reduce changes summation
# order — bit-exactness is NOT expected across a different reduction tree). This
# is a relative floor applied to the baseline magnitude.
_MIN_REL_TOLERANCE = 1e-4


def noise_floor_from_seeds(baseline_seeds, k_sigma=2.0):
    """Derive a correctness tolerance from replicate single-GPU baseline seeds.

    The tolerance is `k_sigma` standard deviations of the baseline's seed-to-seed
    scatter (default 2σ ≈ the 95% band of run-to-run training noise). A
    distributed run inside this band is statistically indistinguishable from the
    baseline computation under a different RNG; outside it, it is doing something
    different.

    Requires >= 2 seeds (one run gives no variance estimate — and a single-GPU
    baseline you cannot vary is not a baseline you can gate against). Raises
    ValueError otherwise, so a caller can never silently gate against a fabricated
    zero-width floor.

    Returns (tolerance, baseline_mean, baseline_std).
    """
    seeds = list(baseline_seeds)
    if len(seeds) < 2:
        raise ValueError(
            f"need >= 2 baseline seeds to estimate a noise floor (got {len(seeds)}); "
            "a one-seed baseline gives no variance and cannot gate a distributed run"
        )
    if k_sigma <= 0:
        raise ValueError(f"k_sigma must be > 0, got {k_sigma!r}")
    mean, std, _ = es.mean_std_sem(seeds)  # raises on NaN/inf (diverged run)
    tolerance = k_sigma * std
    return tolerance, mean, std


def check_distributed_correctness(
    distributed_metric,
    single_gpu_baseline,
    tolerance=None,
    baseline_seeds=None,
    k_sigma=2.0,
    metric_name="loss",
):
    """The §C21 gate. Decide whether a distributed run matches its single-GPU
    baseline within the noise floor.

    Exactly ONE source of the tolerance must be supplied:
      * `tolerance=<float>` — an explicit absolute tolerance (e.g. a pre-computed
        noise floor), OR
      * `baseline_seeds=[...]` — replicate single-GPU baseline metrics, from which
        a `k_sigma`-σ tolerance is derived via `noise_floor_from_seeds`.

    `single_gpu_baseline` is the reference value the distributed run must match.
    When `baseline_seeds` is given and `single_gpu_baseline` is None, the seed
    MEAN is used as the reference (the natural single-GPU expectation).

    Returns a dict:
      passed            bool   — |Δ| <= tolerance (False also on any non-finite input)
      verdict           str    — "pass" | "fail"
      metric_name       str
      distributed_metric, single_gpu_baseline, tolerance  floats (as resolved)
      abs_delta         float  — |distributed − baseline|
      rel_delta         float  — |Δ| / |baseline|   (inf if baseline == 0)
      margin            float  — tolerance − |Δ|  (>=0 passes; how much headroom)
      n_baseline_seeds  int|None
      reason            str    — human-readable one-liner for the ledger/post-mortem

    HARD RULES:
      * A NaN/inf distributed_metric => FAIL (a diverged distributed run is the
        canonical fast-but-wrong case and must never pass by accident).
      * You may not pass BOTH `tolerance` and `baseline_seeds` (ambiguous source
        of truth), nor NEITHER (no floor => not gateable, §C21).
      * The effective tolerance is floored at `_MIN_REL_TOLERANCE * |baseline|`
        so a fluky zero-scatter baseline cannot reject a bit-reassociation-level
        difference that is correct-by-design across a different reduction tree.
    """
    if (tolerance is None) == (baseline_seeds is None):
        raise ValueError(
            "supply exactly one of tolerance=<float> or baseline_seeds=[...]; "
            "two sources is ambiguous, zero sources is ungateable (§C21)"
        )

    n_seeds = None
    if baseline_seeds is not None:
        derived_tol, seed_mean, seed_std = noise_floor_from_seeds(
            baseline_seeds, k_sigma=k_sigma
        )
        tolerance = derived_tol
        n_seeds = len(list(baseline_seeds))
        if single_gpu_baseline is None:
            single_gpu_baseline = seed_mean
    else:
        if single_gpu_baseline is None:
            raise ValueError(
                "single_gpu_baseline is required when tolerance is given explicitly"
            )
        tolerance = float(tolerance)
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")

    baseline = float(single_gpu_baseline)
    dist = float(distributed_metric)

    # Apply the relative floor so a bit-exact-impossible reduction-order diff
    # cannot fail against a coincidentally-zero-width baseline scatter.
    eff_tolerance = max(tolerance, _MIN_REL_TOLERANCE * abs(baseline))

    # Non-finite distributed metric => diverged run => hard FAIL (never silently
    # NaN-compares to True).
    if not math.isfinite(dist):
        return {
            "passed": False,
            "verdict": "fail",
            "metric_name": metric_name,
            "distributed_metric": dist,
            "single_gpu_baseline": baseline,
            "tolerance": eff_tolerance,
            "abs_delta": float("inf"),
            "rel_delta": float("inf"),
            "margin": float("-inf"),
            "n_baseline_seeds": n_seeds,
            "reason": (
                f"distributed {metric_name} is non-finite ({dist}) — diverged "
                "distributed run; fast-but-wrong, gate FAILS (§C21)"
            ),
        }

    abs_delta = abs(dist - baseline)
    rel_delta = abs_delta / abs(baseline) if baseline != 0 else float("inf")
    margin = eff_tolerance - abs_delta
    passed = abs_delta <= eff_tolerance

    if passed:
        reason = (
            f"distributed {metric_name}={dist:.6g} matches single-GPU "
            f"baseline={baseline:.6g} within tolerance={eff_tolerance:.6g} "
            f"(|Δ|={abs_delta:.6g}); MFU may now be trusted (§C21 pass)"
        )
    else:
        reason = (
            f"distributed {metric_name}={dist:.6g} DIVERGES from single-GPU "
            f"baseline={baseline:.6g}: |Δ|={abs_delta:.6g} > tolerance="
            f"{eff_tolerance:.6g} — fast-but-wrong; run is INCONCLUSIVE, do NOT "
            "trust its MFU (§C21 fail)"
        )

    return {
        "passed": passed,
        "verdict": "pass" if passed else "fail",
        "metric_name": metric_name,
        "distributed_metric": dist,
        "single_gpu_baseline": baseline,
        "tolerance": eff_tolerance,
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
        "margin": margin,
        "n_baseline_seeds": n_seeds,
        "reason": reason,
    }


def gate_or_raise(*args, **kwargs):
    """Convenience wrapper: run the gate and raise CorrectnessGateError on FAIL.

    For callers (the scaling-report driver) that want a hard stop — "do not even
    compute MFU if correctness failed". Returns the verdict dict on pass.
    """
    result = check_distributed_correctness(*args, **kwargs)
    if not result["passed"]:
        raise CorrectnessGateError(result["reason"], result)
    return result


class CorrectnessGateError(RuntimeError):
    """Raised by gate_or_raise when the §C21 distributed-correctness gate fails."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


__all__ = [
    "noise_floor_from_seeds",
    "check_distributed_correctness",
    "gate_or_raise",
    "CorrectnessGateError",
]
