#!/usr/bin/env python3
"""research/calibration_pairs.py — the (predicted, realised) JOIN that feeds
§C15.3.8 scorer calibration.

`research/scorer_calibration.py` is the MATH (Brier / log-loss / reliability
bins / ECE / shrinkage); it takes two paired lists and has no idea where they
come from. This module is the missing half: it walks the ledger and produces
those lists, honestly, from

    techniques[].predicted_win_prob  <->  techniques[].run_ids[]  <->  runs[].verdict

READ-ONLY by construction: it opens ledger.json for reading and has no write
path at all (no ledger.py mutator is imported). Stdlib-only, pure CPU.

WHY THIS SHIPS WITHOUT A NUMBER. As of 2026-07-23 not one technique in the
ledger carries `predicted_win_prob` (`/idea-selection` has never run against
it), so the join yields n=0. `scorer_calibration.report` raises on an empty
pair list, and /weekly-retro puts the honest floor at ~10 pairs ("descriptive
only — n=<N> below the calibration floor"). So this module reports the JOIN and
the HYGIENE GAPS and stops there; publishing an ECE off n=0 would be inventing
a number. Once the pairs exist, the report half is one line:

    scorer_calibration.report(**calibration_pairs.scorer_inputs())

CLI:
  python3 research/calibration_pairs.py [--ledger PATH] [--per-run]
    -> {"pairs": [...], "pending": [...], "hygiene_gaps": [...], "n": N}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_LEDGER = Path(__file__).resolve().parent / "ledger" / "ledger.json"

# A RECORDED verdict is a realised outcome. The scorer earns credit only for a
# verified `win` (§C17/§C19: an unverified single-seed win is `inconclusive` by
# construction, so it scores 0 here); every other recorded verdict is "not a
# win" = 0. `promising`/`null` are the §C25.3 split of the old `directional`.
#
# TRAP, stated because the two look identical in prose: the verdict WORD "null"
# is a MEASURED no-effect result — realised, outcome 0 — and is NOT the same
# thing as a JSON `null` verdict, which means not-judged-yet and is EXCLUDED as
# pending. Contracts §C8 draws exactly this distinction.
OUTCOME = {"win": 1, "loss": 0, "inconclusive": 0,
           "null": 0, "promising": 0, "directional": 0}

# Statuses at or past `briefed` (a technique reaches `done` through `briefed`).
# §C15.3.8 requires predicted_win_prob from `briefed` onward, so any of these
# without the field is a hygiene gap, not a silent drop.
REACHED_BRIEFED = frozenset({"briefed", "queued", "running", "done"})

# /weekly-retro's honesty floor: below this the ECE bins are nearly empty, so
# report raw Brier/log-loss at most and never publish a shrinkage off it.
MIN_PAIRS_FOR_CALIBRATION = 10


def load_ledger(path=None) -> dict:
    """Parse ledger.json. Read-only; no validation side effects, no writes."""
    return json.loads(Path(path or DEFAULT_LEDGER).read_text())


def below_calibration_floor(n: int) -> bool:
    """True when n is too small for the reliability curve / shrinkage to mean
    anything (the caller must then label the report descriptive-only)."""
    return n < MIN_PAIRS_FOR_CALIBRATION


def collect(led: dict, per_run: bool = False) -> dict:
    """Join predictions to realised outcomes.

    Granularity, because it changes n and therefore every calibration number:
    the scorer predicts a win-probability for an IDEA, so the default emits ONE
    pair per technique — outcome 1 if any of its realised runs won, else 0.
    `per_run=True` emits one pair per run, which repeats the SAME prediction
    across a cohort's runs and inflates n on correlated outcomes; it exists for
    inspection, not for feeding the report.

    Returns {"pairs", "pending", "hygiene_gaps", "n"}. Every pair carries its
    technique slug and contributing run_ids so any number is traceable back to
    a ledger entry (verifiable-accuracy rule).
    """
    runs = {r.get("run_id"): r for r in led.get("runs", [])}
    pairs, pending, gaps = [], [], []

    for t in led.get("techniques", []):
        slug, status = t.get("slug"), t.get("status")
        pwp = t.get("predicted_win_prob")
        if pwp is None:
            if status in REACHED_BRIEFED:
                gaps.append({"kind": "briefed_without_prediction", "technique": slug,
                             "status": status,
                             "detail": "§C15.3.8 requires predicted_win_prob once "
                                       "a technique is briefed"})
            continue
        run_ids = t.get("run_ids") or []
        if not run_ids:
            gaps.append({"kind": "prediction_without_run", "technique": slug,
                         "status": status, "predicted_win_prob": pwp,
                         "detail": "prediction can never be realised: no run_ids[]"})
            continue

        realised = []
        for rid in run_ids:
            r = runs.get(rid)
            if r is None:
                gaps.append({"kind": "run_id_not_found", "technique": slug,
                             "run_id": rid,
                             "detail": "technique.run_ids[] names a run with no "
                                       "runs[] entry"})
                continue
            verdict = r.get("verdict")
            if verdict is None:          # JSON null = not judged yet
                pending.append({"technique": slug, "run_id": rid,
                                "predicted_win_prob": pwp,
                                "run_status": r.get("status")})
                continue
            realised.append((rid, OUTCOME.get(verdict, 0), verdict))

        if not realised:
            if status == "done":
                gaps.append({"kind": "done_without_realised_verdict",
                             "technique": slug, "run_ids": run_ids,
                             "detail": "technique is done but no run carries a "
                                       "verdict — outcome unrealisable"})
            continue
        if per_run:
            pairs.extend({"technique": slug, "predicted_win_prob": pwp,
                          "outcome": y, "run_ids": [rid], "verdicts": [v]}
                         for rid, y, v in realised)
        else:
            pairs.append({"technique": slug, "predicted_win_prob": pwp,
                          "outcome": 1 if any(y for _, y, _ in realised) else 0,
                          "run_ids": [rid for rid, _, _ in realised],
                          "verdicts": [v for _, _, v in realised]})

    return {"pairs": pairs, "pending": pending, "hygiene_gaps": gaps,
            "n": len(pairs)}


def scorer_inputs(result=None, **kwargs) -> dict:
    """The exact keyword shape `scorer_calibration.report(preds, outcomes)`
    wants, so the report half is `report(**scorer_inputs())`. Pass a `collect()`
    result, or nothing to join the default ledger."""
    if result is None:
        result = collect(load_ledger(kwargs.pop("ledger", None)), **kwargs)
    return {"preds": [p["predicted_win_prob"] for p in result["pairs"]],
            "outcomes": [p["outcome"] for p in result["pairs"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--per-run", action="store_true",
                    help="one pair per RUN instead of per technique (inflates n "
                         "on multi-run cohorts; inspection only)")
    a = ap.parse_args(argv)
    print(json.dumps(collect(load_ledger(a.ledger), per_run=a.per_run), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
