"""Generic, task-agnostic harness-search framework — the reusable core of the
validated Meta-Harness paradigm (arXiv 2603.28052), with the safety lesson our
own experiment proved baked in as a HARD GATE.

What the experiment showed (2026-06-18, research/harness_search):
  * Automated search BEAT the hand-designed baseline (+5.3 pts, 95% CI [+4.5,+6.2]).
  * But selecting the winner by SEARCH score alone crowned a brittle, timeout-prone
    overfit candidate that scored 0.0 on an unseen seed. Naive selection = a trap.
  * Held-out + variance selection (eval_stats) recovered the true win and refused
    the brittle one.

So this framework NEVER promotes a discovered harness unless it (1) is non-brittle
on every held-out seed AND (2) SIGNIFICANTLY beats the incumbent on held-out via
the eval_stats Student-t gate. A search-set number alone can never promote.

Task-agnostic by design: a caller supplies a `scorer(harness_path, seed) ->
(score, valid_fraction)`; everything else here is pure and unit-tested with a
synthetic scorer (test_framework.py) — no model, no GPU, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # research/ -> eval_stats
import eval_stats as es  # noqa: E402


def _heldout_replicates(scorer, path, heldout_seeds):
    """Return (mean, per_seed_scores, brittle) for a harness on the held-out seeds.
    brittle == any seed timed out / was invalid (score<=0 or valid_fraction<1) —
    the exact failure mode that sank the full-traces arm's search-winner."""
    scores, brittle = [], False
    for s in heldout_seeds:
        sc, valid = scorer(path, s)
        scores.append(sc)
        if sc <= 0.0 or valid < 1.0:
            brittle = True
    return (sum(scores) / len(scores) if scores else 0.0), scores, brittle


def select_and_promote(scorer, candidate_paths, incumbent_path,
                       search_seed, heldout_seeds, direction="higher_is_better",
                       top_k=5, alpha=0.05):
    """Run the proven selection + promotion gate over a set of discovered
    candidate harnesses.

    scorer(path, seed) -> (score, valid_fraction).
    Steps:
      1. MODEL-SELECT on search_seed: rank candidates by search score (never the
         metric we promote on).
      2. Take the top_k by search score + the incumbent; get held-out replicates.
      3. EXCLUDE brittle candidates (any held-out seed invalid/timeout/<=0).
      4. Among non-brittle candidates, the best HELD-OUT mean is the challenger.
      5. PROMOTE only if the challenger SIGNIFICANTLY beats the incumbent on
         held-out (eval_stats.seed_delta_significant). Else keep the incumbent.

    Returns a dict report — deterministic given a deterministic scorer.
    """
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError("direction must be higher_is_better|lower_is_better")
    if not candidate_paths:
        raise ValueError("no candidate harnesses to select from")

    # 1. model selection on search
    search_ranked = sorted(
        ((scorer(c, search_seed)[0], c) for c in candidate_paths),
        key=lambda t: (-t[0] if direction == "higher_is_better" else t[0]),
    )
    shortlist = [c for _, c in search_ranked[:top_k]]

    # 2-3. held-out replicates + brittleness for incumbent and shortlist
    inc_mean, inc_rep, inc_brittle = _heldout_replicates(scorer, incumbent_path, heldout_seeds)
    evaluated = []
    for c in shortlist:
        mean, rep, brittle = _heldout_replicates(scorer, c, heldout_seeds)
        evaluated.append({"path": str(c), "heldout_mean": mean, "heldout": rep,
                          "brittle": brittle, "search": scorer(c, search_seed)[0]})

    # 4. challenger = best held-out among NON-BRITTLE candidates
    eligible = [e for e in evaluated if not e["brittle"]]
    excluded_brittle = [e["path"] for e in evaluated if e["brittle"]]
    if not eligible:
        return {"promoted": False, "reason": "all candidates brittle on held-out",
                "incumbent": str(incumbent_path), "incumbent_heldout_mean": inc_mean,
                "excluded_brittle": excluded_brittle, "challenger": None}
    better = (max if direction == "higher_is_better" else min)
    challenger = better(eligible, key=lambda e: e["heldout_mean"])

    # 5. promotion gate: significant held-out improvement over the incumbent
    h = es.seed_delta_significant(inc_rep, challenger["heldout"], direction=direction)
    promote = bool(h["significant"])
    return {
        "promoted": promote,
        "reason": ("significant held-out win over incumbent" if promote
                   else "challenger does not significantly beat incumbent on held-out"),
        "incumbent": str(incumbent_path), "incumbent_heldout_mean": inc_mean,
        "challenger": challenger["path"], "challenger_heldout_mean": challenger["heldout_mean"],
        "challenger_search": challenger["search"],
        "significance": {"improvement": h["improvement"], "ci95": h["ci95"],
                         "significant": h["significant"], "warning": h.get("warning")},
        "excluded_brittle": excluded_brittle,
        "n_candidates": len(candidate_paths), "n_eligible": len(eligible),
    }
