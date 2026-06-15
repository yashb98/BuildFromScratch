"""Tests for research/scaling_fit.py — fit a Chinchilla-style law and predict a
held-out point. numpy+scipy are importorskip'd so the stdlib-only CI stays green;
these run wherever scipy is installed (locally)."""
import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import scaling_fit as sf

# ground-truth law used to synthesise exact (noise-free) points
E, A, ALPHA, B, BETA = 2.0, 500.0, 0.34, 400.0, 0.28


def _L(N, D):
    return E + A / N ** ALPHA + B / D ** BETA


def _points():
    pts = []
    for N in (1e8, 3e8, 6e8):          # 3 distinct N
        for D in (1e9, 3e9):
            pts.append({"N": N, "D": D, "L": _L(N, D)})
    return pts                          # 6 points, 5 coefficients


def test_fit_recovers_noise_free_law():
    out = sf.fit(_points())
    assert out["descriptive_only"] is False        # 3 distinct N
    assert out["max_rel_residual"] < 0.02          # near-exact on clean data


def test_predict_matches_truth():
    coef = sf.fit(_points())
    pred = sf.predict(coef, 9e8, 5e9)
    assert pred == pytest.approx(_L(9e8, 5e9), rel=0.05)


def test_holdout_extrapolation_is_accurate():
    pts = _points()
    out = sf.holdout_predict(pts, holdout_index=len(pts) - 1)  # hold out a big point
    assert out["rel_error"] < 0.1


def test_descriptive_only_flag_with_few_distinct_N():
    # 4 points but only 2 distinct N -> descriptive only
    pts = [{"N": 1e8, "D": 1e9, "L": _L(1e8, 1e9)},
           {"N": 1e8, "D": 3e9, "L": _L(1e8, 3e9)},
           {"N": 3e8, "D": 1e9, "L": _L(3e8, 1e9)},
           {"N": 3e8, "D": 3e9, "L": _L(3e8, 3e9)}]
    assert sf.fit(pts)["descriptive_only"] is True


def test_too_few_points_rejected():
    with pytest.raises(ValueError):
        sf.fit([{"N": 1e8, "D": 1e9, "L": 5.0}])
