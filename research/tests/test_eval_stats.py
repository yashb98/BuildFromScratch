"""Tests for research/eval_stats.py — the significance core of the hardened
eval suite (§C10). Pure math, no model: proves the new gate (a) flags a clean
win, (b) refuses a within-noise delta, and (c) — the audit's key point —
NEVER calls a single-seed comparison significant, however large the point gap.
"""
import math

import pytest

import eval_stats as es


def test_mean_std_sem_basic():
    mean, sd, sem = es.mean_std_sem([10.0, 12.0, 14.0])
    assert mean == 12.0
    assert math.isclose(sd, 2.0)                     # sample std of 10,12,14
    assert math.isclose(sem, 2.0 / math.sqrt(3))


def test_single_value_has_zero_variance():
    assert es.mean_std_sem([5.0]) == (5.0, 0.0, 0.0)


def test_subsample_noise_floor():
    assert math.isclose(es.subsample_noise_floor([28.5, 28.7, 28.6]), 0.2)


def test_clear_win_is_significant_lower_better():
    # PPL: treatment clearly below baseline, tight within-arm variance.
    r = es.seed_delta_significant([28.6, 28.7, 28.5], [23.5, 23.6, 23.4])
    assert r["significant"] is True
    assert r["improvement"] > 5 and r["ci95"][0] > 0 and r["warning"] is None


def test_within_noise_delta_not_significant():
    # Overlapping arms: the CI must straddle 0.
    r = es.seed_delta_significant([28.6, 28.7, 28.5], [28.5, 28.7, 28.6])
    assert r["significant"] is False
    assert r["ci95"][0] <= 0 <= r["ci95"][1]


def test_single_seed_never_significant():
    # The audit's core fix: one run per arm => no variance => not significant,
    # even with a huge point delta.
    r = es.seed_delta_significant([28.6], [23.5])
    assert r["significant"] is False
    assert r["n_baseline"] == 1 and "not significance-testable" in r["warning"]


def test_two_seeds_warns_but_can_be_significant():
    r = es.seed_delta_significant([28.6, 28.7], [23.5, 23.6])
    assert r["significant"] is True
    assert r["warning"] is not None and "n<3" in r["warning"]


def test_higher_is_better_direction():
    # Accuracy: treatment above baseline is the improvement.
    r = es.seed_delta_significant([0.40, 0.41, 0.39], [0.46, 0.47, 0.45],
                                  direction="higher_is_better")
    assert r["significant"] is True and r["improvement"] > 0


def test_higher_is_better_regression_not_significant_win():
    # Treatment WORSE on a higher-is-better metric -> improvement negative,
    # CI does not exclude 0 on the improving side.
    r = es.seed_delta_significant([0.46, 0.47, 0.45], [0.40, 0.41, 0.39],
                                  direction="higher_is_better")
    assert r["significant"] is False and r["improvement"] < 0


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        es.seed_delta_significant([1, 2], [1, 2], direction="sideways")
    with pytest.raises(ValueError):
        es.mean_std_sem([])


# ---- adversarial-review fixes: t-not-z, generators, NaN, sem=0 -------------

def test_uses_student_t_not_normal_z():
    # A borderline delta where the z(1.96) CI excludes 0 (would call it a win)
    # but the correct t(df~4 ~= 2.78) CI does NOT. The hardened gate must say NO.
    r = es.seed_delta_significant([28.4, 28.6, 28.8], [28.07, 28.27, 28.47])
    assert 0.30 < r["improvement"] < 0.36       # ~0.33 PPL point gap
    assert r["significant"] is False            # t-gate refuses; z-gate would not
    assert r["ci95"][0] < 0 < r["ci95"][1]
    assert r["df"] <= 5 and r["t_crit"] > 2.5   # small-n => large t critical value


def test_generators_preserve_n_and_verdict():
    # Generators must not be consumed-then-miscounted into n=0 / not-significant.
    r = es.seed_delta_significant((x for x in [28.6, 28.7, 28.5]),
                                  (x for x in [23.5, 23.6, 23.4]))
    assert r["n_baseline"] == 3 and r["n_treatment"] == 3
    assert r["significant"] is True and r["improvement"] > 5


def test_nan_and_inf_raise_valueerror():
    with pytest.raises(ValueError):
        es.mean_std_sem([28.6, float("nan"), 28.5])
    with pytest.raises(ValueError):
        es.seed_delta_significant([28.6, float("inf"), 28.5], [23.5, 23.6, 23.4])


def test_zero_within_arm_variance_is_not_significant():
    # Identical 'seeds' (really repeated evals of one ckpt) must NOT manufacture
    # a fake infinite-confidence win.
    r = es.seed_delta_significant([28.6, 28.6, 28.6], [28.5, 28.5, 28.5])
    assert r["significant"] is False
    assert r["pooled_sem"] == 0 and "distinct training seeds" in r["warning"]


def test_asymmetric_single_seed_not_significant():
    # One good arm, one single-seed arm -> still not significance-testable.
    r = es.seed_delta_significant([28.6, 28.7, 28.5], [23.5])
    assert r["significant"] is False and r["n_treatment"] == 1
