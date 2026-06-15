"""Pure statistics for eval comparability (§C10/§C15) — stdlib-only, no torch,
no model, no I/O. The eval-harness suite imports these so a "technique X beat
baseline" verdict rests on TRAINING-SEED variance and a real interval, not on a
single PPL point differenced against data-subsample wobble of one checkpoint.

Audit fix (2026-06-15): the prior gate was `abs(delta) > floor`, where
`floor = max-min over 3 disjoint DATA subsamples of ONE model` — corpus-sampling
jitter of a single checkpoint, NOT seed-to-seed training variance. A true effect
below seed noise would be declared significant. These functions replace that
gate; the legacy floor is kept (demoted) for back-compat.

The interval is a Student-t CI (NOT a normal-z CI): at the n≈3 seeds this gate is
built for, the z critical value (1.96) is ~30% too narrow and would declare wins
a proper t-test rejects. We use the two-sided t_{0.975} for the Welch–Satterthwaite
degrees of freedom, floored (conservative). This is the honest small-n choice;
it is still an approximation and is NOT a substitute for ≥3 real seeds.

Everything here is pure and deterministic and is unit-tested directly without
running a single eval (test_eval_stats.py).
"""
from __future__ import annotations

import math
import statistics

Z975 = 1.959963984540054  # two-sided 95% normal quantile (df -> inf fallback)

# Two-sided t_{0.975} by integer degrees of freedom (df 1..30). df>30 -> Z975.
# Source: standard t-table. Used instead of z because this gate runs at n≈3.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def t_critical(df):
    """Two-sided t_{0.975} for the given (possibly fractional) df, floored to an
    integer (smaller df -> larger t -> more conservative). The 1e-9 epsilon keeps
    a mathematically-integer df (e.g. Welch df==4 computed as 3.999999996) from
    floating down to 3; genuinely fractional dfs still floor (conservative).
    df>30 -> the z value."""
    d = max(1, int(math.floor(df + 1e-9)))
    return _T975.get(d, Z975)


def mean_std_sem(values):
    """(mean, sample_std, standard_error_of_mean) over >=1 replicate-seed runs.
    n==1 -> std/sem are 0.0 (no variance estimate exists; callers must treat a
    single seed as NOT significance-testable, see seed_delta_significant).
    Raises ValueError on empty input or any non-finite value (a NaN/inf PPL means
    a diverged run, which must surface explicitly, not as a cryptic stdlib crash)."""
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("need at least one value")
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("non-finite value (NaN/inf) in seeds — diverged run?")
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return mean, 0.0, 0.0
    sd = statistics.stdev(vals)               # sample std, n-1
    return mean, sd, sd / math.sqrt(len(vals))


def subsample_noise_floor(values):
    """LEGACY floor (finance contracts §6): max-min over n disjoint DATA
    subsamples of ONE checkpoint. Retained for back-compat and as a secondary
    sanity check, but it is NOT a training-variance test — prefer
    seed_delta_significant() for any cross-run win/loss verdict."""
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("need at least one value")
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("non-finite value (NaN/inf) in values")
    return max(vals) - min(vals)


def _welch_df(sem_b, n_b, sem_t, n_t):
    """Welch–Satterthwaite degrees of freedom from the per-arm SEMs and n's.
    Requires n_b, n_t >= 2 (callers guarantee this before calling)."""
    vb, vt = sem_b ** 2, sem_t ** 2            # variance of each arm's mean
    denom = (vb ** 2) / (n_b - 1) + (vt ** 2) / (n_t - 1)
    if denom == 0:                              # both arms zero-variance
        return float("inf")
    return (vb + vt) ** 2 / denom


def seed_delta_significant(baseline_seeds, treatment_seeds,
                           direction="lower_is_better"):
    """Two-sample difference-of-means with a Student-t CI on the improvement,
    using SEM across replicate TRAINING seeds for each arm.

    direction: "lower_is_better" (PPL/loss) or "higher_is_better" (accuracy).
    Returns a dict with the means, the improvement delta (positive == better in
    the given direction), the pooled SEM, the t-based 95% CI, the df + t used,
    and `significant` = the whole CI lies on the improving side of 0.

    HARD RULES (the audit's points):
      * <2 seeds in either arm -> NO variance estimate -> `significant` False
        with a warning (one run can never establish significance, however large
        the point delta).
      * zero within-arm variance in BOTH arms (sem==0) -> `significant` False
        with a warning (identical 'seeds' are almost always repeated evals of one
        checkpoint mislabeled as seeds, which would manufacture a fake win).

    Generators are accepted: inputs are materialized once so n is counted from
    the real data, not an exhausted iterator.
    """
    if direction not in ("lower_is_better", "higher_is_better"):
        raise ValueError(f"direction must be lower_is_better|higher_is_better, got {direction!r}")
    baseline_seeds = list(baseline_seeds)       # materialize (generator-safe)
    treatment_seeds = list(treatment_seeds)
    b_mean, _, b_sem = mean_std_sem(baseline_seeds)
    t_mean, _, t_sem = mean_std_sem(treatment_seeds)
    n_b, n_t = len(baseline_seeds), len(treatment_seeds)

    improvement = (b_mean - t_mean) if direction == "lower_is_better" else (t_mean - b_mean)
    pooled_sem = math.sqrt(b_sem ** 2 + t_sem ** 2)

    warning = None
    significant = False
    if n_b < 2 or n_t < 2:
        warning = (f"n<2 in an arm (baseline={n_b}, treatment={n_t}): no variance "
                   "estimate — not significance-testable; run >=3 training seeds")
        df, tcrit = 0.0, float("nan")
        ci_lo, ci_hi = float("nan"), float("nan")
    elif pooled_sem == 0:
        warning = ("zero within-arm variance — verify these are distinct training "
                   "seeds, not repeated evals of one checkpoint")
        df, tcrit = _welch_df(b_sem, n_b, t_sem, n_t), float("nan")
        ci_lo, ci_hi = improvement, improvement
    else:
        df = _welch_df(b_sem, n_b, t_sem, n_t)
        tcrit = t_critical(df)
        ci_lo, ci_hi = improvement - tcrit * pooled_sem, improvement + tcrit * pooled_sem
        significant = ci_lo > 0                  # whole CI on the improving side
        if n_b < 3 or n_t < 3:
            warning = f"n<3 in an arm (baseline={n_b}, treatment={n_t}): CI is wide/unreliable"

    return {
        "direction": direction,
        "baseline_mean": b_mean, "treatment_mean": t_mean,
        "improvement": improvement, "pooled_sem": pooled_sem,
        "df": df, "t_crit": tcrit, "ci95": [ci_lo, ci_hi],
        "significant": significant,
        "n_baseline": n_b, "n_treatment": n_t, "warning": warning,
    }
