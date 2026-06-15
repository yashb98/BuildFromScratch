"""Tests for research/novelty_score.py — the §C15.3 scoring/selection math.
Stdlib-only, deterministic (CI-safe)."""
import random

import pytest

import novelty_score as ns

ALL_GOOD = {k: 1.0 for k in ns.PROMOTE}
ALL_GOOD.update({"velocity": 0.0, "cost": 0.0, "harness": 1.0,
                 "contam": 0.0, "blast": 0.0, "isolate": 1.0, "power": 1.0})


def test_composite_mu_maxes_at_one():
    assert ns.composite_mu(ALL_GOOD) == pytest.approx(1.0)


def test_mu_monotone_in_novelty():
    lo = {**ALL_GOOD, "novelty": 0.4}
    hi = {**ALL_GOOD, "novelty": 0.9}
    assert ns.composite_mu(hi) > ns.composite_mu(lo)


def test_penalties_reduce_mu():
    base = ns.composite_mu(ALL_GOOD)
    assert ns.composite_mu({**ALL_GOOD, "contam": 0.8}) < base
    assert ns.composite_mu({**ALL_GOOD, "blast": 0.8}) < base
    assert ns.composite_mu({**ALL_GOOD, "isolate": 0.0}) < base   # can't isolate
    assert ns.composite_mu({**ALL_GOOD, "power": 0.0}) < base     # below noise floor


def test_cost_and_harness_reduce_value():
    base = ns.composite_mu(ALL_GOOD)
    assert ns.composite_mu({**ALL_GOOD, "cost": 1.0}) < base
    assert ns.composite_mu({**ALL_GOOD, "harness": 0.0}) < base   # huge impl surface


def test_sigma_widens_on_shallow_search_and_disagreement():
    deep = ns.aggregate_sigma({"search_depth": 1.0, "persona_disagreement": 0.0})
    shallow = ns.aggregate_sigma({"search_depth": 0.0, "persona_disagreement": 0.0})
    assert shallow > deep
    disag = ns.aggregate_sigma({"search_depth": 1.0, "persona_disagreement": 0.9})
    assert disag > deep


def test_ucb_explore_bonus():
    assert ns.ucb(0.5, 0.2, kappa=1.0) == pytest.approx(0.7)
    assert ns.ucb(0.5, 0.2, kappa=2.0) > ns.ucb(0.5, 0.2, kappa=1.0)


def test_thompson_is_seeded_and_bounded():
    rng = random.Random(0)
    draws = [ns.thompson(0.6, 0.1, rng) for _ in range(2000)]
    assert all(0.0 <= d <= 1.0 for d in draws)
    assert abs(sum(draws) / len(draws) - 0.6) < 0.02      # mean ≈ μ
    # reproducible from the same seed
    assert ns.thompson(0.6, 0.1, random.Random(7)) == ns.thompson(0.6, 0.1, random.Random(7))


def test_gates_kill_unscalable_and_unmeasurable():
    assert ns.passes_gates(ALL_GOOD) is True
    assert ns.passes_gates({**ALL_GOOD, "scale": 0.05}) is False   # doesn't scale
    assert ns.passes_gates({**ALL_GOOD, "power": 0.05}) is False   # can't measure


def test_select_diverse_avoids_redundancy():
    # id0 & id1 are near-duplicates (group A); id2 is distinct (group B)
    items = [{"id": "id0", "score": 0.90, "g": "A"},
             {"id": "id1", "score": 0.85, "g": "A"},
             {"id": "id2", "score": 0.80, "g": "B"}]
    sim = lambda a, b: 1.0 if a["g"] == b["g"] else 0.0
    picked = ns.select_diverse(items, k=2, sim=sim, lam=0.7)
    assert picked == ["id0", "id2"]   # diverse beats the higher-scoring duplicate


def test_rank_filters_gated_and_orders_by_acq():
    cands = [
        {"id": "good", "scores": ALL_GOOD},
        {"id": "killed", "scores": {**ALL_GOOD, "scale": 0.0}},
    ]
    ranked = ns.rank(cands)
    assert [r["id"] for r in ranked] == ["good"]
    assert ranked[0]["acq"] >= ranked[0]["mu"]


def test_disagreement_scaling():
    assert ns.disagreement([0.5, 0.5]) == pytest.approx(0.0)
    assert ns.disagreement([0.2, 0.8]) == pytest.approx(0.6)   # pstdev 0.3 / 0.5
    assert ns.disagreement([0.5]) == 0.0
