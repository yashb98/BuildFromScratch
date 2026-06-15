"""Tests for research/taxonomy_gap.py — the §C15.3 structural-hole gap-finder.
Stdlib-only, deterministic (CI-safe)."""
import pytest

import taxonomy_gap as tg

AXES = [["a", "b"], ["x", "y", "z"]]


def test_neighbors_differ_in_exactly_one_axis():
    nbrs = tg.neighbors(("a", "x"), AXES)
    # axis0: (b,x); axis1: (a,y),(a,z)  -> 3 neighbours
    assert set(nbrs) == {("b", "x"), ("a", "y"), ("a", "z")}


def test_empty_cell_amid_full_scores_high():
    occ = {("b", "x"): 1, ("a", "y"): 1, ("a", "z"): 1}   # all (a,x) neighbours full
    assert tg.gap_score(("a", "x"), occ, AXES) == pytest.approx(1.0)


def test_full_cell_scores_low():
    occ = {("b", "x"): 1, ("a", "y"): 1, ("a", "z"): 1}
    hole = tg.gap_score(("a", "x"), occ, AXES)
    full = tg.gap_score(("a", "y"), occ, AXES)            # (a,y) is occupied
    assert full < hole


def test_isolated_empty_cell_scores_zero():
    occ = {("a", "x"): 1}                                  # (b,z) neighbours all empty
    assert tg.gap_score(("b", "z"), occ, AXES) == pytest.approx(0.0)


def test_sparse_cell_scores_between_empty_and_full():
    occ = {("b", "x"): 1, ("a", "y"): 1, ("a", "z"): 1, ("a", "x"): 3}
    sparse = tg.gap_score(("a", "x"), occ, AXES)           # count 3 -> emptiness 0.25
    assert 0.0 < sparse < 1.0


def test_rank_gaps_surfaces_the_hole_first():
    occ = {("b", "x"): 1, ("a", "y"): 1, ("a", "z"): 1}
    ranked = tg.rank_gaps(occ, AXES)
    assert ranked[0]["cell"] == ("a", "x")                 # the empty-amid-full cell
    assert ranked[0]["gap"] >= ranked[-1]["gap"]
