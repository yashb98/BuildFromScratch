"""Tests for research/harness_search/framework.py — the harness-search selection
+ promotion gate. Uses a SYNTHETIC scorer (no model, no GPU, instant) to pin the
behaviors that matter, above all the lesson our own experiment taught: a brittle
search-winner (timeout/invalid on an unseen seed) must NEVER be promoted, even
when it has the best search score.
"""
import pytest

import framework as fw

SEARCH_SEED = 1
HELDOUT = [2, 3, 4]

# name -> {search, heldout(list aligned to HELDOUT), valid(list, default all 1)}
SPEC = {
    "inc":     {"search": 0.866, "heldout": [0.869, 0.871, 0.870]},
    "good":    {"search": 0.920, "heldout": [0.921, 0.919, 0.923]},
    # BEST search score, but one held-out seed times out (score 0, valid 0.4):
    "brittle": {"search": 0.930, "heldout": [0.000, 0.922, 0.921], "valid": [0.4, 1, 1]},
    "tie":     {"search": 0.900, "heldout": [0.870, 0.872, 0.869]},
    "hi_search_lo_held": {"search": 0.950, "heldout": [0.881, 0.879, 0.882]},
    "lo_search_hi_held": {"search": 0.910, "heldout": [0.921, 0.919, 0.923]},
}


def scorer(path, seed):
    s = SPEC[str(path)]
    if seed == SEARCH_SEED:
        return s["search"], 1.0
    i = HELDOUT.index(seed)
    valid = s.get("valid", [1.0, 1.0, 1.0])[i]
    return s["heldout"][i], valid


def run(cands):
    return fw.select_and_promote(scorer, cands, "inc", SEARCH_SEED, HELDOUT,
                                 direction="higher_is_better")


def test_promotes_significant_nonbrittle():
    r = run(["good"])
    assert r["promoted"] is True
    assert r["challenger"] == "good" and r["significance"]["significant"] is True


def test_excludes_brittle_even_with_best_search_score():
    # 'brittle' has the highest search score (0.930) so it tops model-selection,
    # but it times out on one held-out seed -> must be excluded; 'good' wins.
    r = run(["brittle", "good"])
    assert "brittle" in r["excluded_brittle"]
    assert r["promoted"] is True and r["challenger"] == "good"


def test_keeps_incumbent_when_not_significant():
    r = run(["tie"])
    assert r["promoted"] is False
    assert "does not significantly beat" in r["reason"]


def test_all_brittle_no_promotion():
    r = run(["brittle"])
    assert r["promoted"] is False
    assert r["reason"] == "all candidates brittle on held-out"
    assert r["challenger"] is None


def test_selection_uses_heldout_not_search():
    # hi_search_lo_held has the best SEARCH (0.950) but worse HELD-OUT (~0.880);
    # lo_search_hi_held wins on held-out (~0.921). Selection must follow held-out.
    r = run(["hi_search_lo_held", "lo_search_hi_held"])
    assert r["challenger"] == "lo_search_hi_held" and r["promoted"] is True


def test_empty_pool_raises():
    with pytest.raises(ValueError):
        run([])
