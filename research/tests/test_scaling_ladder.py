"""Tests for research/scaling_ladder.py — pure stdlib, no GPU, no model loads.

Two halves:
  PLAN    — build_ladder emits the right per-cell run specs (budget->steps,
            shared seeds across arms, fewer seeds at larger budgets).
  ANALYZE — fit_gap_trend turns synthetic per-budget gaps into the correct
            PERSISTS / CONVERGES / WIDENS / FLAT verdict, with the noise floor
            and the <3-budget descriptive guard honored.
"""
import math

import pytest

import scaling_ladder as sl


# ----- PART 1: PLAN -----------------------------------------------------------

BASE = {"model": "qwen3-0.6b", "lr": 3e-3, "seq_len": 2048}
BUDGETS = [42e6, 170e6, 670e6, 1190e6]
TPS = 2048 * 256  # tokens per step (seq_len * global batch)


def test_steps_for_budget_rounds_up():
    assert sl.steps_for_budget(1000, 100) == 10
    assert sl.steps_for_budget(1001, 100) == 11      # partial final step still runs
    with pytest.raises(ValueError):
        sl.steps_for_budget(0, 100)
    with pytest.raises(ValueError):
        sl.steps_for_budget(1000, 0)


def test_build_ladder_cell_count_and_steps():
    out = sl.build_ladder(BASE, BUDGETS, TPS, arms=("normuon", "adamw"),
                          seeds_per_budget=3, base_name="qwen3")
    # 4 budgets * 3 seeds * 2 arms = 24 cells
    assert out["summary"]["n_cells"] == 24
    assert out["summary"]["n_budgets"] == 4
    assert out["summary"]["n_arms"] == 2
    # steps derived from tok_per_step, rounded up
    for c in out["cells"]:
        assert c["max_steps"] == math.ceil(c["train_tokens"] / TPS)


def test_seeds_shared_across_arms_at_same_budget():
    # the seed must be a PAIRED factor: both arms train on the same seed set at a
    # given budget, so seed is blocked out of the gap, not added as noise.
    out = sl.build_ladder(BASE, [42e6], TPS, arms=("normuon", "adamw"),
                          seeds_per_budget=3, base_seed=10)
    by_arm = {}
    for c in out["cells"]:
        by_arm.setdefault(c["arm"], set()).add(c["seed"])
    assert by_arm["normuon"] == by_arm["adamw"] == {10, 11, 12}


def test_fewer_seeds_at_larger_budgets_via_list():
    out = sl.build_ladder(BASE, BUDGETS, TPS, arms=("a", "b"),
                          seeds_per_budget=[3, 3, 2, 1])
    per = out["summary"]["cells_per_budget"]
    # cells_per_budget counts BOTH arms: 3*2, 3*2, 2*2, 1*2
    assert per["42M"] == 6
    assert per["170M"] == 6
    assert per["670M"] == 4
    assert per["1190M"] == 2


def test_fewer_seeds_via_dict_keyed_by_budget():
    out = sl.build_ladder(BASE, BUDGETS, TPS, arms=("a", "b"),
                          seeds_per_budget={42e6: 3, 170e6: 3, 670e6: 2, 1190e6: 1})
    assert out["summary"]["cells_per_budget"]["1190M"] == 2


def test_seed_schedule_length_mismatch_rejected():
    with pytest.raises(ValueError):
        sl.build_ladder(BASE, BUDGETS, TPS, seeds_per_budget=[3, 2])  # 2 != 4


def test_zero_seeds_rejected():
    with pytest.raises(ValueError):
        sl.build_ladder(BASE, BUDGETS, TPS, seeds_per_budget=[3, 3, 0, 1])


def test_single_arm_rejected():
    with pytest.raises(ValueError):
        sl.build_ladder(BASE, BUDGETS, TPS, arms=("only",))


def test_run_ids_unique_and_config_carries_budget():
    out = sl.build_ladder(BASE, BUDGETS, TPS, arms=("normuon", "adamw"))
    ids = [c["run_id"] for c in out["cells"]]
    assert len(ids) == len(set(ids))                 # no collisions
    for c in out["cells"]:
        assert c["config"]["train_tokens"] == c["train_tokens"]
        assert c["config"]["max_steps"] == c["max_steps"]
        assert c["config"]["arm"] == c["arm"]
        assert c["config"]["model"] == "qwen3-0.6b"  # base config preserved
        assert f"arm:{c['arm']}" in c["tags"]


def test_total_train_tokens_summed():
    out = sl.build_ladder(BASE, [42e6, 170e6], TPS, arms=("a", "b"),
                          seeds_per_budget=2)
    # 2 budgets * 2 seeds * 2 arms; each (42M+170M) per (seed,arm) pair... sum it
    expect = sum(c["train_tokens"] for c in out["cells"])
    assert out["summary"]["total_train_tokens"] == expect
    assert expect == (42e6 + 170e6) * 2 * 2          # 4 (seed,arm) combos


# ----- PART 2: ANALYZE --------------------------------------------------------

LADDER = [42e6, 170e6, 670e6, 1190e6]


def _gaps_from(slope, intercept):
    """Synthesize gaps that lie EXACTLY on gap = slope*log10(tokens)+intercept."""
    return [slope * math.log10(b) + intercept for b in LADDER]


def test_persists_flat_positive_edge():
    # flat, clearly-positive gap -> edge survives at scale
    gaps = [0.05, 0.051, 0.049, 0.05]
    out = sl.fit_gap_trend(LADDER, gaps, gap_noise=0.005)
    assert out["verdict_code"] == "PERSISTS"
    assert abs(out["slope"]) < 1e-2
    assert out["edge_resolved"] is True


def test_converges_positive_edge_shrinks_to_zero():
    # gap starts at 0.08 (42M) and decays to ~0 by 1190M -> early-training speedup
    # slope is negative (gap falls as log tokens rise); edge sign positive.
    gaps = _gaps_from(slope=-0.05, intercept=0.05 * math.log10(LADDER[-1]))
    # check it actually crosses near zero at the top rung
    assert gaps[0] > 0.05 and abs(gaps[-1]) < 0.01
    out = sl.fit_gap_trend(LADDER, gaps, gap_noise=0.01)
    assert out["verdict_code"] == "CONVERGES"
    assert out["toward_zero"] is True
    assert out["edge_resolved"] is False             # lands inside noise at top


def test_widens_edge_grows_with_scale():
    # positive edge that GROWS with scale -> advantage compounds
    gaps = _gaps_from(slope=0.04, intercept=-0.20)
    assert gaps[-1] > gaps[0] > 0
    out = sl.fit_gap_trend(LADDER, gaps, gap_noise=0.01)
    assert out["verdict_code"] == "WIDENS"


def test_flat_within_noise_is_inconclusive():
    # tiny gaps entirely inside the noise floor -> FLAT, no advantage either way
    gaps = [0.002, -0.001, 0.001, 0.0]
    out = sl.fit_gap_trend(LADDER, gaps, gap_noise=0.01)
    assert out["verdict_code"] == "FLAT"
    assert out["edge_resolved"] is False


def test_negative_edge_converges_handled():
    # treatment WORSE (gap<0) but converging toward 0 -> CONVERGES (edge erodes).
    gaps = _gaps_from(slope=0.05, intercept=-0.05 * math.log10(LADDER[-1]))
    assert gaps[0] < -0.05 and abs(gaps[-1]) < 0.01   # starts negative, rises to ~0
    out = sl.fit_gap_trend(LADDER, gaps, gap_noise=0.01)
    assert out["edge_sign"] == -1
    assert out["verdict_code"] == "CONVERGES"


def test_slope_within_noise_not_acted_on():
    # a real positive edge with a faint negative slope whose SPAN effect is inside
    # the noise floor must NOT be called CONVERGES — it stays PERSISTS.
    gaps = _gaps_from(slope=-0.002, intercept=0.10)
    out = sl.fit_gap_trend(LADDER, gaps, gap_noise=0.01)
    assert out["slope_resolved"] is False
    assert out["verdict_code"] == "PERSISTS"


def test_descriptive_only_with_two_budgets():
    out = sl.fit_gap_trend([42e6, 1190e6], [0.05, 0.05], gap_noise=0.005)
    assert out["descriptive_only"] is True
    assert "DESCRIPTIVE" in out["verdict"]


def test_three_budgets_not_descriptive():
    out = sl.fit_gap_trend([42e6, 170e6, 670e6], [0.05, 0.05, 0.05], gap_noise=0.005)
    assert out["descriptive_only"] is False
    assert "DESCRIPTIVE" not in out["verdict"]


def test_r2_near_one_on_exact_line():
    gaps = _gaps_from(slope=-0.03, intercept=0.30)
    out = sl.fit_gap_trend(LADDER, gaps, gap_noise=0.0)
    assert out["r2"] == pytest.approx(1.0, abs=1e-9)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        sl.fit_gap_trend([42e6, 170e6], [0.05])


def test_single_budget_rejected():
    with pytest.raises(ValueError):
        sl.fit_gap_trend([42e6], [0.05])


def test_non_finite_gap_rejected():
    with pytest.raises(ValueError):
        sl.fit_gap_trend(LADDER, [0.05, float("nan"), 0.05, 0.05])


def test_identical_budgets_rejected():
    with pytest.raises(ValueError):
        sl.fit_gap_trend([42e6, 42e6, 42e6], [0.05, 0.04, 0.06])


def test_negative_gap_noise_rejected():
    with pytest.raises(ValueError):
        sl.fit_gap_trend(LADDER, [0.05, 0.05, 0.05, 0.05], gap_noise=-0.1)
