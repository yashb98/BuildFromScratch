"""Tests for research/scorer_calibration.py — Brier/log-loss/ECE + shrinkage.
Stdlib-only, deterministic (CI-safe)."""
import pytest

import scorer_calibration as sc


def test_brier_known_values():
    assert sc.brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert sc.brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_log_loss_perfect_is_near_zero_and_miss_is_large():
    assert sc.log_loss([1.0, 0.0], [1, 0]) < 1e-10
    assert sc.log_loss([0.0, 1.0], [1, 0]) > 10           # confident & wrong


def test_ece_zero_for_calibrated():
    # predictions match observed frequency exactly
    assert sc.ece([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_ece_high_for_overconfident():
    # always predicts 0.9 but only 50% are wins -> 0.4 gap in that bin
    e = sc.ece([0.9, 0.9, 0.9, 0.9], [1, 0, 1, 0])
    assert e == pytest.approx(0.4)


def test_shrinkage_factor_bounds():
    assert sc.shrinkage_factor(0.0) == pytest.approx(1.0)        # perfect -> trust
    assert sc.shrinkage_factor(0.4) == pytest.approx(1 / 1.8)    # 1/(1+2*0.4)
    assert 0.0 < sc.shrinkage_factor(5.0) < 0.2                  # awful -> shrink hard


def test_calibrate_shrinks_toward_half():
    assert sc.calibrate(0.9, 1.0) == pytest.approx(0.9)
    assert sc.calibrate(0.9, 0.0) == pytest.approx(0.5)
    assert sc.calibrate(0.9, 0.5) == pytest.approx(0.7)


def test_report_has_all_fields():
    r = sc.report([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
    assert set(r) >= {"n", "brier", "log_loss", "ece", "shrinkage", "bins"}
    assert r["n"] == 4 and 0 < r["shrinkage"] <= 1


def test_validation_errors():
    with pytest.raises(ValueError):
        sc.brier([0.5], [1, 0])           # length mismatch
    with pytest.raises(ValueError):
        sc.brier([], [])                  # empty
    with pytest.raises(ValueError):
        sc.brier([1.5], [1])              # pred out of range
    with pytest.raises(ValueError):
        sc.brier([0.5], [2])              # bad outcome
