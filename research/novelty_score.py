#!/usr/bin/env python3
"""research/novelty_score.py — the §C15.3 idea-scoring + selection math engine.

Turns the §C15 taste rubric into a portfolio allocator under uncertainty. Pure
stdlib (math/statistics/random), CPU-only, deterministic (any randomness takes an
injected seeded RNG) — CI-safe and unit-tested.

This module is the MATH only. The qualitative sub-scores (novelty, mechanism,
scaling prior, prior-art saturation, persona disagreement, …) are produced by the
/idea-selection skill + its agents and passed in as a dict of normalised values in
[0,1]; this engine combines them, attaches an uncertainty, applies the
explore/exploit acquisition, and selects a DIVERSE shortlist. Selection is a
bandit, not a sort (see §C15.3).

Score dict keys (all in [0,1] unless noted):
  promote : voi, novelty, scale, mech, transfer, durable, falsify
  bonus   : velocity            (research-velocity multiplier)
  cost    : cost                (total cost incl. engineering; 0=free, 1=max)
            harness             (1=trivial diff, →0 huge implementation surface)
  penalty : contam, blast
  gate    : isolate, power      (low isolate/power are penalised AND can hard-gate)
  sigma   : search_depth (1=thorough), persona_disagreement, calib_uncertainty
"""
from __future__ import annotations

import math
import statistics

EPS = 1e-9

DEFAULT_W = {
    # geometric-mean weights over the promote terms
    "voi": 1.4, "novelty": 1.2, "scale": 1.3, "mech": 1.0,
    "transfer": 0.9, "durable": 0.7, "falsify": 0.8,
    # multipliers / penalties
    "velocity": 0.5, "cost": 0.6, "harness": 0.4,
    "pen_contam": 0.4, "pen_blast": 0.5, "pen_isolate": 0.5, "pen_power": 0.6,
    # sigma aggregation
    "sig_base": 0.05, "sig_depth": 0.45, "sig_disagree": 0.4, "sig_calib": 0.3,
}

PROMOTE = ("voi", "novelty", "scale", "mech", "transfer", "durable", "falsify")


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def _g(s, key, default=0.5):
    v = s.get(key, default)
    return _clamp(float(v))


def weighted_geomean(values_weights):
    """exp( Σ w·ln(v) / Σ w ); values clamped away from 0 so one weak-but-nonzero
    term drags the score without zeroing it (a true zero is a gate, not a term)."""
    num = 0.0
    den = 0.0
    for v, w in values_weights:
        num += w * math.log(max(v, EPS))
        den += w
    return math.exp(num / den) if den else 0.0


def composite_mu(s: dict, weights: dict = None) -> float:
    """The expected priority μ ∈ [0,1]: a weighted geomean of the promote terms,
    scaled by the velocity bonus and cheapness, minus the penalties. Monotone
    increasing in every promote term, decreasing in every penalty."""
    w = {**DEFAULT_W, **(weights or {})}
    value = weighted_geomean([(_g(s, k), w[k]) for k in PROMOTE])
    value *= (1.0 + w["velocity"] * _g(s, "velocity", 0.0))      # velocity bonus
    value *= (1.0 - w["cost"] * _g(s, "cost", 0.5))              # cheaper → higher
    value *= (1.0 - w["harness"] * (1.0 - _g(s, "harness", 1.0)))  # big diff → lower
    pen = (w["pen_contam"] * _g(s, "contam", 0.0)
           + w["pen_blast"] * _g(s, "blast", 0.0)
           + w["pen_isolate"] * (1.0 - _g(s, "isolate", 1.0))
           + w["pen_power"] * (1.0 - _g(s, "power", 1.0)))
    # MULTIPLICATIVE penalty (not subtractive): a penalty SHRINKS μ proportionally
    # rather than driving it to exactly 0. The old `value - pen` zeroed ~79% of
    # gate-passing ideas (pen routinely exceeded value), collapsing the bandit to
    # insertion order. (isolate/power are also hard gates in passes_gates; here
    # they act as soft penalties — the tests assert they reduce μ.)
    return _clamp(value * (1.0 - _clamp(pen)))


def aggregate_sigma(s: dict, weights: dict = None) -> float:
    """Uncertainty on μ: shallow search, persona disagreement, and a poorly-
    calibrated scorer all widen it (→ more explore weight in the acquisition)."""
    w = {**DEFAULT_W, **(weights or {})}
    sigma = (w["sig_base"]
             + w["sig_depth"] * (1.0 - _g(s, "search_depth", 0.5))
             + w["sig_disagree"] * _g(s, "persona_disagreement", 0.0)
             + w["sig_calib"] * _g(s, "calib_uncertainty", 0.0))
    return _clamp(sigma)


def passes_gates(s: dict, gates: dict = None) -> bool:
    """Hard kills (§C15.3): a frontier idea that does not scale, cannot be
    measured above the noise floor, or cannot be isolated is rejected outright —
    no μ can rescue it."""
    g = {"scale_min": 0.15, "power_min": 0.15, "novelty_min": 0.1,
         "isolate_min": 0.1, **(gates or {})}
    return (_g(s, "scale") >= g["scale_min"]
            and _g(s, "power") >= g["power_min"]
            and _g(s, "novelty") >= g["novelty_min"]
            and _g(s, "isolate") >= g["isolate_min"])


def ucb(mu: float, sigma: float, kappa: float = 1.0) -> float:
    """Deterministic explore/exploit acquisition: μ + κ·σ. Higher κ explores more
    (favours high-uncertainty, high-upside ideas)."""
    return mu + kappa * sigma


def thompson(mu: float, sigma: float, rng) -> float:
    """Stochastic acquisition: a draw from N(μ,σ) clamped to [0,1]. `rng` is an
    injected random.Random so selection is reproducible from a pinned seed."""
    return _clamp(rng.gauss(mu, sigma))


def select_diverse(items, k, sim, lam=0.7):
    """Max-Marginal-Relevance shortlist: greedily pick the item maximising
    lam·score − (1−lam)·max similarity-to-already-picked, so the SET is diverse,
    not just each item high (anti-herding, §C15.3). `items` is a list of dicts
    with 'id' and 'score'; `sim(a,b)→[0,1]`. Returns the chosen ids in order."""
    pool = list(items)
    chosen = []
    while pool and len(chosen) < k:
        best, best_val = None, -math.inf
        for it in pool:
            redundancy = max((sim(it, c) for c in chosen), default=0.0)
            val = lam * it["score"] - (1.0 - lam) * redundancy
            if val > best_val:
                best, best_val = it, val
        chosen.append(best)
        pool.remove(best)
    return [c["id"] for c in chosen]


def rank(candidates, weights=None, gates=None, kappa=1.0):
    """Score, gate, and rank candidates by the UCB acquisition. Each candidate is
    {'id':..., 'scores': {...}}. Returns the survivors sorted best-first with
    mu/sigma/acq attached."""
    out = []
    for c in candidates:
        s = c["scores"]
        if not passes_gates(s, gates):
            continue
        mu = composite_mu(s, weights)
        sigma = aggregate_sigma(s, weights)
        out.append({"id": c["id"], "mu": mu, "sigma": sigma,
                    "acq": ucb(mu, sigma, kappa)})
    out.sort(key=lambda r: r["acq"], reverse=True)
    return out


def disagreement(persona_scores) -> float:
    """Persona-ensemble signal: stdev of N personas' scores for one idea, scaled
    to [0,1] (max stdev of values in [0,1] is 0.5). High disagreement = either
    genuinely novel or genuinely confused → route to a human."""
    if len(persona_scores) < 2:
        return 0.0
    return _clamp(statistics.pstdev(persona_scores) / 0.5)
