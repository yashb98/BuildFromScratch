#!/usr/bin/env python3
"""research/scaling_ladder.py — infra for the SCALING-LADDER experiment (the #1 RS
lift): does an optimizer/architecture edge measured at small budget SURVIVE at
scale, or is it just an early-training speedup that converges away?

Two halves, both pure stdlib (no numpy/scipy, no torch, no I/O):

  1. PLAN — `build_ladder(...)`. Given a base config, a list of token budgets
     (e.g. [42e6, 170e6, 670e6, 1190e6]), a tokens-per-step, and a seed schedule
     (FEWER seeds allowed at larger budgets, where compute is dear), emit the
     per-cell run specs for BOTH arms (e.g. NorMuon vs AdamW): budget -> steps
     (via tok_per_step), per-cell seeds, tags, and a deterministic run_id. This
     is the exact list of training jobs the next run launches.

  2. ANALYZE — `fit_gap_trend(...)`. Given the per-budget BPB gaps measured after
     the runs complete (treatment minus baseline, in the metric's own units), do
     a simple ordinary-least-squares fit of gap vs log10(tokens). The SIGN of the
     slope is the verdict:
        slope ~ 0  and gap stays > floor  -> PERSISTS   (edge survives at scale)
        slope > 0  (gap shrinks toward 0) -> CONVERGES   (early-training speedup)
        slope < 0  (gap grows)            -> WIDENS       (edge compounds)
     "shrinks toward 0" is defined relative to the SIGN of the edge: a positive
     edge (treatment better, gap measured as improvement>0) converges when the
     fitted line trends toward 0; the code handles either edge direction.

Honesty (mirrors §C16 / §C17):
  * The verdict is DESCRIPTIVE with < 3 budget points (a 2-point "trend" is a
    line through 2 dots, not evidence) — `descriptive_only` is set and the
    verdict is suffixed accordingly.
  * A trend is only declared when the slope is resolvable above the measurement
    noise: the caller passes a `gap_noise` (the per-cell BPB noise floor from the
    seed-variance gate, research/eval_stats.py). If the fitted slope's effect over
    the budget range is within that floor, the verdict is FLAT/INCONCLUSIVE — we
    do NOT crown "persists" or "converges" off slope noise.

Everything is unit-tested directly on synthetic gaps (test_scaling_ladder.py)
without launching a single run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

# ----- gap-direction convention -------------------------------------------------
# Gaps are stored as IMPROVEMENT of treatment over baseline in the metric's own
# direction-corrected units: gap > 0 means the treatment arm is BETTER. For BPB
# (lower is better) the caller computes gap = baseline_bpb - treatment_bpb.
# "Converges away" therefore always means "the gap moves toward 0".


# ===== PART 1: PLAN — emit per-cell run specs ==================================

def _run_id(base_name: str, arm: str, tokens: float, seed: int) -> str:
    """Deterministic, collision-resistant run id for a single ladder cell."""
    tok_tag = _human_tokens(tokens)
    raw = f"{base_name}|{arm}|{int(round(tokens))}|{seed}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:6]
    return f"{base_name}_{arm}_{tok_tag}_s{seed}_{h}"


def _human_tokens(tokens: float) -> str:
    """42e6 -> '42M', 1.19e9 -> '1190M' (kept in M so the ladder reads uniformly)."""
    millions = tokens / 1e6
    if millions >= 1000:
        # keep whole-million resolution; 1190M, not 1.19B, so cells sort lexically
        return f"{int(round(millions))}M"
    if abs(millions - round(millions)) < 1e-6:
        return f"{int(round(millions))}M"
    return f"{millions:.1f}M"


def steps_for_budget(tokens: float, tok_per_step: float) -> int:
    """tokens / tok_per_step, rounded UP (a partial final step still runs).
    tok_per_step = global_batch_tokens = micro_batch * grad_accum * seq_len * dp."""
    if tokens <= 0:
        raise ValueError(f"tokens must be > 0, got {tokens}")
    if tok_per_step <= 0:
        raise ValueError(f"tok_per_step must be > 0, got {tok_per_step}")
    return int(math.ceil(tokens / tok_per_step))


def _normalize_seed_schedule(token_budgets, seeds_per_budget):
    """Resolve seeds_per_budget into a per-budget seed COUNT list.

    Accepts:
      * an int           -> same count at every budget;
      * a list/tuple     -> per-budget counts (len must match token_budgets);
      * a dict {tokens: count} -> explicit count keyed by budget (matched by
        nearest float key so 42e6 and 42000000.0 are the same cell).
    Fewer seeds at larger budgets is the intended use (cost). Every count must be
    >= 1; a budget with 0 seeds is a config error (it would silently drop a rung).
    """
    n = len(token_budgets)
    if isinstance(seeds_per_budget, int):
        counts = [seeds_per_budget] * n
    elif isinstance(seeds_per_budget, dict):
        counts = []
        for b in token_budgets:
            match = None
            for k, v in seeds_per_budget.items():
                if math.isclose(float(k), float(b), rel_tol=1e-9):
                    match = v
                    break
            if match is None:
                raise ValueError(f"seeds_per_budget dict has no entry for budget {b}")
            counts.append(match)
    else:
        counts = list(seeds_per_budget)
        if len(counts) != n:
            raise ValueError(
                f"seeds_per_budget list length {len(counts)} != "
                f"number of budgets {n}")
    for b, c in zip(token_budgets, counts):
        if int(c) < 1:
            raise ValueError(f"budget {b} has < 1 seed ({c}) — would drop the rung")
    return [int(c) for c in counts]


def build_ladder(base_config: dict,
                 token_budgets,
                 tok_per_step: float,
                 arms=("treatment", "baseline"),
                 seeds_per_budget=3,
                 base_seed: int = 0,
                 base_name: str = "ladder") -> dict:
    """Emit the full per-cell run-spec list for a scaling ladder.

    base_config: the shared config dict (model, data, lr schedule, ...). Each cell
        spec carries a COPY with `train_tokens` and `max_steps` set for that rung,
        plus an `arm` key — the launcher applies the arm's minimal diff.
    token_budgets: iterable of token counts (e.g. [42e6, 170e6, 670e6, 1190e6]).
    tok_per_step: tokens consumed per optimizer step (global batch tokens).
    arms: the two (or more) arms to compare, e.g. ("normuon", "adamw").
    seeds_per_budget: int | list | {tokens: count} — fewer seeds at larger budgets.
    base_seed: seeds are base_seed, base_seed+1, ... per cell (shared ACROSS arms
        at the same budget so the seed is a paired/blocking factor, not noise).

    Returns a dict with `cells` (flat list of run specs) and `summary` (counts +
    the seed schedule), so the caller can both launch and cost the ladder.
    """
    budgets = [float(b) for b in token_budgets]
    if not budgets:
        raise ValueError("token_budgets is empty")
    if len(set(arms)) < 2:
        raise ValueError("need at least 2 distinct arms to measure a gap")
    seed_counts = _normalize_seed_schedule(budgets, seeds_per_budget)

    cells = []
    for tokens, n_seeds in zip(budgets, seed_counts):
        steps = steps_for_budget(tokens, tok_per_step)
        seeds = [base_seed + i for i in range(n_seeds)]
        for seed in seeds:                       # seed is the OUTER paired factor
            for arm in arms:
                cfg = dict(base_config)
                cfg["arm"] = arm
                cfg["train_tokens"] = tokens
                cfg["max_steps"] = steps
                cfg["seed"] = seed
                cells.append({
                    "run_id": _run_id(base_name, arm, tokens, seed),
                    "arm": arm,
                    "train_tokens": tokens,
                    "tokens_human": _human_tokens(tokens),
                    "max_steps": steps,
                    "tok_per_step": tok_per_step,
                    "seed": seed,
                    "tags": [f"ladder:{base_name}", f"arm:{arm}",
                             f"budget:{_human_tokens(tokens)}", f"seed:{seed}",
                             "experiment:scaling-ladder"],
                    "config": cfg,
                })

    total_tokens = sum(c["train_tokens"] for c in cells)
    return {
        "base_name": base_name,
        "arms": list(arms),
        "tok_per_step": tok_per_step,
        "budgets": budgets,
        "seed_schedule": dict(zip((_human_tokens(b) for b in budgets), seed_counts)),
        "cells": cells,
        "summary": {
            "n_cells": len(cells),
            "n_budgets": len(budgets),
            "n_arms": len(arms),
            "total_train_tokens": total_tokens,
            "cells_per_budget": {
                _human_tokens(b): c * len(arms)
                for b, c in zip(budgets, seed_counts)
            },
        },
    }


# ===== PART 2: ANALYZE — fit the gap-vs-log(tokens) trend =====================

def _ols_line(xs, ys):
    """Ordinary-least-squares slope+intercept for y = slope*x + intercept.
    Pure stdlib. Returns (slope, intercept, r2). Requires >= 2 distinct x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("all token budgets are identical — no trend to fit")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    # R^2 (coefficient of determination); syy==0 (flat gaps) -> r2 defined as 1.0
    syy = sum((y - my) ** 2 for y in ys)
    if syy == 0:
        r2 = 1.0
    else:
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 - ss_res / syy
    return slope, intercept, r2


def fit_gap_trend(token_budgets, gaps, gap_noise: float = 0.0) -> dict:
    """Fit per-budget gaps (treatment improvement over baseline, gap>0 == better)
    against log10(tokens) and return the trend verdict.

    token_budgets: the budgets at which gaps were measured (one per gap).
    gaps: the measured BPB gaps (baseline_bpb - treatment_bpb per budget).
    gap_noise: per-cell measurement noise floor on a gap (from the seed-variance
        gate, eval_stats.seed_delta_significant). The slope is only acted on when
        its modeled effect across the budget range EXCEEDS this floor; otherwise
        the trend is FLAT/inconclusive (we never crown a verdict off slope noise).

    Verdict (string in `verdict`):
        PERSISTS   — edge present (|gap| at the top rung > noise) and slope flat
                     (edge does NOT trend toward 0): the advantage survives.
        WIDENS     — edge magnitude grows with scale (slope pushes |gap| up).
        CONVERGES  — edge shrinks toward 0 with scale (slope pulls |gap| down past
                     the noise floor at the largest budget): early-training speedup
                     that washes out — the honest "no real advantage at scale".
        FLAT       — slope effect within noise AND no resolvable edge: inconclusive.
    """
    budgets = [float(b) for b in token_budgets]
    gp = [float(g) for g in gaps]
    if len(budgets) != len(gp):
        raise ValueError(f"len(token_budgets)={len(budgets)} != len(gaps)={len(gp)}")
    if len(budgets) < 2:
        raise ValueError("need >= 2 budget points to fit a trend")
    if any(b <= 0 for b in budgets):
        raise ValueError("token budgets must be > 0 (log10 taken)")
    if any(not math.isfinite(g) for g in gp):
        raise ValueError("non-finite gap (NaN/inf) — a diverged run?")
    if gap_noise < 0:
        raise ValueError("gap_noise must be >= 0")

    xs = [math.log10(b) for b in budgets]
    slope, intercept, r2 = _ols_line(xs, gp)

    # Order by budget so "first"/"last" rung are the true extremes.
    order = sorted(range(len(budgets)), key=lambda i: budgets[i])
    x_lo, x_hi = xs[order[0]], xs[order[-1]]
    # fitted gap at the smallest and largest budget on the regression line
    gap_lo_fit = slope * x_lo + intercept
    gap_hi_fit = slope * x_hi + intercept
    # observed edge sign uses the MEAN gap (robust to a single noisy rung)
    mean_gap = sum(gp) / len(gp)
    edge_sign = 1 if mean_gap > 0 else (-1 if mean_gap < 0 else 0)

    # Modeled change in gap across the whole budget span (the load-bearing number).
    span_effect = gap_hi_fit - gap_lo_fit
    # Is there a real edge at the LARGEST budget (where it matters for "at scale")?
    edge_at_top = abs(gap_hi_fit)
    edge_resolved = edge_at_top > gap_noise
    # Is the slope's effect resolvable above per-cell noise?
    slope_resolved = abs(span_effect) > gap_noise

    descriptive_only = len(budgets) < 3

    # Decide direction of the slope RELATIVE to the edge sign:
    # span_effect * edge_sign > 0  -> |gap| growing in the edge's direction (WIDENS)
    # span_effect * edge_sign < 0  -> |gap| shrinking toward 0           (CONVERGES)
    toward_zero = (edge_sign != 0) and (span_effect * edge_sign < 0)

    if not slope_resolved:
        # slope is within noise: edge is flat across scale
        if edge_resolved:
            verdict = "PERSISTS"
            rationale = ("gap is flat vs log(tokens) (slope effect within noise) "
                         "and the edge at the largest budget exceeds the noise "
                         "floor — the advantage survives at scale")
        else:
            verdict = "FLAT"
            rationale = ("gap is flat vs log(tokens) and within the noise floor at "
                         "the largest budget — no resolvable advantage either way")
    else:
        if toward_zero:
            # gap shrinking toward 0: does it land inside the noise floor at top?
            if not edge_resolved:
                verdict = "CONVERGES"
                rationale = ("gap shrinks toward 0 with scale and falls within the "
                             "noise floor at the largest budget — an early-training "
                             "speedup that converges away (no advantage at scale)")
            else:
                verdict = "CONVERGES"
                rationale = ("gap shrinks toward 0 with scale; still above noise at "
                             "the largest measured budget but trending out — the "
                             "edge is eroding, extend the ladder before claiming it")
        else:
            verdict = "WIDENS"
            rationale = ("gap grows with scale (slope amplifies the edge beyond the "
                         "noise floor) — the advantage compounds at scale")

    if descriptive_only:
        verdict_full = verdict + " (DESCRIPTIVE: <3 budgets — extend the ladder)"
    else:
        verdict_full = verdict

    return {
        "verdict": verdict_full,
        "verdict_code": verdict,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "mean_gap": mean_gap,
        "edge_sign": edge_sign,
        "gap_lo_fit": gap_lo_fit,
        "gap_hi_fit": gap_hi_fit,
        "span_effect": span_effect,
        "edge_at_top": edge_at_top,
        "edge_resolved": edge_resolved,
        "slope_resolved": slope_resolved,
        "toward_zero": toward_zero,
        "gap_noise": gap_noise,
        "n_budgets": len(budgets),
        "descriptive_only": descriptive_only,
        "rationale": rationale,
        "fit": "gap = slope * log10(tokens) + intercept (OLS)",
    }


# ===== CLI ====================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="emit per-cell run specs (JSON)")
    p_plan.add_argument("--config", type=Path, required=True,
                        help="JSON base config dict")
    p_plan.add_argument("--budgets", type=float, nargs="+", required=True,
                        help="token budgets, e.g. 42e6 170e6 670e6 1190e6")
    p_plan.add_argument("--tok-per-step", type=float, required=True)
    p_plan.add_argument("--arms", nargs="+", default=["treatment", "baseline"])
    p_plan.add_argument("--seeds", type=int, nargs="+", default=[3],
                        help="one int (all budgets) or one per budget")
    p_plan.add_argument("--name", default="ladder")

    p_fit = sub.add_parser("fit", help="fit gap-vs-log(tokens) trend (JSON)")
    p_fit.add_argument("--budgets", type=float, nargs="+", required=True)
    p_fit.add_argument("--gaps", type=float, nargs="+", required=True)
    p_fit.add_argument("--gap-noise", type=float, default=0.0)

    a = ap.parse_args(argv)
    if a.cmd == "plan":
        cfg = json.loads(a.config.read_text())
        seeds = a.seeds[0] if len(a.seeds) == 1 else a.seeds
        out = build_ladder(cfg, a.budgets, a.tok_per_step, arms=tuple(a.arms),
                           seeds_per_budget=seeds, base_name=a.name)
    else:
        out = fit_gap_trend(a.budgets, a.gaps, gap_noise=a.gap_noise)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
