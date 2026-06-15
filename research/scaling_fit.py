#!/usr/bin/env python3
"""research/scaling_fit.py — fit a Chinchilla-style scaling law to a set of
completed runs and (the load-bearing part) PREDICT a held-out larger model's
loss, reporting the prediction error. The §C-research-spine artifact that turns
"one point" into "a law that was right at held-out scale".

Form (Hoffmann 2022): L(N, D) = E + A / N**alpha + B / D**beta
    N = params, D = tokens, L = loss. Fit in log space with a Huber loss
    (robust to a noisy point), via scipy.optimize.least_squares.

Honesty (mirrors §C16): with < 3 distinct N the fit is DESCRIPTIVE ONLY — there
is no extrapolation claim. The comparability law (§C10) still applies upstream:
points from different model families / tokenizers must never share a fit; this
module fits whatever it is given, so the CALLER must pass comparable points.

Requires numpy + scipy (installed locally; the test importorskips them so the
stdlib-only CI stays green).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy.optimize import least_squares
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only where scipy is absent
    _HAVE_SCIPY = False


def _predict(params, N, D):
    E, A, alpha, B, beta = params
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def _residuals(params, N, D, L):
    # residual on log-loss is the standard robust choice
    return np.log(_predict(params, N, D)) - np.log(L)


def fit(points: list) -> dict:
    """points: [{"N": params, "D": tokens, "L": loss}, ...]. Returns the fitted
    coefficients + a descriptive_only flag."""
    if not _HAVE_SCIPY:
        raise RuntimeError("scaling_fit requires numpy + scipy")
    if len(points) < 4:
        raise ValueError("need >= 4 points to fit 5 coefficients")
    N = np.array([p["N"] for p in points], dtype=float)
    D = np.array([p["D"] for p in points], dtype=float)
    L = np.array([p["L"] for p in points], dtype=float)
    n_distinct_N = len(set(N.tolist()))
    # init: E ~ min loss, A/B ~ scale, exponents ~ 0.3 (Chinchilla-ish)
    p0 = [float(L.min()) * 0.9, 1.0e3, 0.34, 1.0e3, 0.28]
    lb = [0.0, 0.0, 0.0, 0.0, 0.0]
    ub = [float(L.min()), np.inf, 1.0, np.inf, 1.0]
    res = least_squares(_residuals, p0, args=(N, D, L), bounds=(lb, ub),
                        loss="huber", f_scale=0.1, max_nfev=10000)
    E, A, alpha, B, beta = (float(x) for x in res.x)
    fitted = _predict(res.x, N, D)
    rel_err = float(np.max(np.abs(fitted - L) / L))
    return {"E": E, "A": A, "alpha": alpha, "B": B, "beta": beta,
            "n_points": len(points), "n_distinct_N": n_distinct_N,
            "max_rel_residual": rel_err,
            "descriptive_only": n_distinct_N < 3,
            "form": "L = E + A/N^alpha + B/D^beta"}


def predict(coef: dict, N: float, D: float) -> float:
    if not _HAVE_SCIPY:
        raise RuntimeError("scaling_fit requires numpy + scipy")
    return float(_predict([coef["E"], coef["A"], coef["alpha"],
                           coef["B"], coef["beta"]], N, D))


def holdout_predict(points: list, holdout_index: int) -> dict:
    """Fit on all points except `holdout_index`, predict it, report the error —
    the actual evidence that the law extrapolates."""
    train = [p for i, p in enumerate(points) if i != holdout_index]
    held = points[holdout_index]
    coef = fit(train)
    pred = predict(coef, held["N"], held["D"])
    return {"held_out": held, "predicted_L": pred, "actual_L": held["L"],
            "rel_error": abs(pred - held["L"]) / held["L"],
            "coef": coef}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, required=True,
                    help='JSON list of {"N":..,"D":..,"L":..}')
    ap.add_argument("--holdout", type=int, default=None,
                    help="index to hold out and predict (extrapolation check)")
    a = ap.parse_args(argv)
    if not _HAVE_SCIPY:
        print(json.dumps({"error": "numpy + scipy required"}), file=sys.stderr)
        return 2
    pts = json.loads(a.points.read_text())
    out = holdout_predict(pts, a.holdout) if a.holdout is not None else fit(pts)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
