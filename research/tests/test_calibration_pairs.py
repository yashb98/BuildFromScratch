"""Regression tests for research/calibration_pairs.py — the (predicted,
realised) join behind §C15.3.8 scorer calibration.

The join is the half that can lie: it decides which outcomes count as realised,
what a JSON-null verdict means versus the verdict WORD "null", and how many
pairs a multi-run cohort contributes. Each of those is pinned below.

Every test builds its own in-memory/temp ledger; the real ledger is never read
or written (the one test that touches a file asserts the module never writes).
"""
import hashlib
import json

import calibration_pairs as cp
import scorer_calibration


def led(techniques=(), runs=()):
    """A minimal ledger dict — collect() reads only techniques[] and runs[]."""
    return {"schema_version": 1, "techniques": list(techniques),
            "runs": list(runs), "proposals": [], "never_repeat": []}


def tech(slug, pwp=None, status="candidate", run_ids=()):
    return {"slug": slug, "title": slug, "status": status,
            "predicted_win_prob": pwp, "run_ids": list(run_ids)}


def run(run_id, verdict, status="done"):
    return {"run_id": run_id, "type": "ablation", "status": status,
            "verdict": verdict, "metrics": {}}


# ------------------------------------------------------------------ the join

def test_empty_ledger_yields_nothing():
    out = cp.collect(led())
    assert out == {"pairs": [], "pending": [], "hygiene_gaps": [], "n": 0}


def test_win_scores_one_and_traces_to_its_run():
    out = cp.collect(led([tech("t", 0.7, "done", ["2026-07-01_m_t"])],
                         [run("2026-07-01_m_t", "win")]))
    assert out["n"] == 1
    p = out["pairs"][0]
    assert p["predicted_win_prob"] == 0.7 and p["outcome"] == 1
    assert p["technique"] == "t" and p["run_ids"] == ["2026-07-01_m_t"]


def test_every_non_win_verdict_scores_zero():
    """Only a VERIFIED win earns the scorer credit (§C17/§C19). The §C25.3 split
    words (null/promising) and the deprecated `directional` are all realised 0s."""
    for i, verdict in enumerate(("loss", "inconclusive", "null", "promising",
                                 "directional")):
        rid = f"2026-07-0{i + 1}_m_t"
        out = cp.collect(led([tech(f"t{i}", 0.6, "done", [rid])],
                             [run(rid, verdict)]))
        assert out["n"] == 1, verdict
        assert out["pairs"][0]["outcome"] == 0, verdict


def test_json_null_verdict_is_pending_not_a_zero():
    """The trap this module exists to not fall into: an unjudged run must be
    EXCLUDED, not scored 0 — scoring it would punish the scorer for a run that
    has not finished. Distinct from the verdict WORD "null", asserted above."""
    out = cp.collect(led([tech("t", 0.9, "running", ["2026-07-21_m_t"])],
                         [run("2026-07-21_m_t", None, status="launched")]))
    assert out["n"] == 0 and out["pairs"] == []
    assert out["pending"] == [{"technique": "t", "run_id": "2026-07-21_m_t",
                               "predicted_win_prob": 0.9,
                               "run_status": "launched"}]


def test_verdict_word_null_and_json_null_do_not_collide():
    """Both in ONE ledger: the word null is a realised 0, JSON null is pending."""
    out = cp.collect(led([tech("measured", 0.5, "done", ["2026-07-01_m_a"]),
                          tech("inflight", 0.5, "running", ["2026-07-21_m_b"])],
                         [run("2026-07-01_m_a", "null"),
                          run("2026-07-21_m_b", None)]))
    assert out["n"] == 1 and out["pairs"][0]["technique"] == "measured"
    assert [p["technique"] for p in out["pending"]] == ["inflight"]


def test_technique_granularity_is_one_pair_per_idea():
    """Default granularity: the scorer predicted once, so it is graded once —
    a 3-run cohort must not contribute 3 correlated pairs and treble n."""
    t = tech("t", 0.8, "done", ["2026-07-01_m_t", "2026-07-02_m_t", "2026-07-03_m_t"])
    rs = [run("2026-07-01_m_t", "loss"), run("2026-07-02_m_t", "win"),
          run("2026-07-03_m_t", "null")]
    agg = cp.collect(led([t], rs))
    assert agg["n"] == 1
    assert agg["pairs"][0]["outcome"] == 1          # any realised win -> 1
    assert agg["pairs"][0]["verdicts"] == ["loss", "win", "null"]
    per_run = cp.collect(led([t], rs), per_run=True)
    assert per_run["n"] == 3
    assert [p["outcome"] for p in per_run["pairs"]] == [0, 1, 0]


def test_aggregate_is_zero_when_no_run_won():
    out = cp.collect(led([tech("t", 0.8, "done", ["2026-07-01_m_t", "2026-07-02_m_t"])],
                         [run("2026-07-01_m_t", "loss"),
                          run("2026-07-02_m_t", "promising")]))
    assert out["n"] == 1 and out["pairs"][0]["outcome"] == 0


def test_pending_runs_do_not_block_a_realised_sibling():
    out = cp.collect(led([tech("t", 0.4, "running", ["2026-07-01_m_t", "2026-07-21_m_t"])],
                         [run("2026-07-01_m_t", "win"),
                          run("2026-07-21_m_t", None, status="running")]))
    assert out["n"] == 1 and out["pairs"][0]["run_ids"] == ["2026-07-01_m_t"]
    assert len(out["pending"]) == 1


# ------------------------------------------------------------ hygiene checks

def test_briefed_without_prediction_is_a_hygiene_gap():
    """§C15.3.8 requires the field from `briefed` onward — the gap is REPORTED,
    never silently dropped, which is the whole reason today's n is 0."""
    out = cp.collect(led([tech("b", None, "briefed"), tech("d", None, "done"),
                          tech("q", None, "queued"), tech("r", None, "running")]))
    assert {g["technique"] for g in out["hygiene_gaps"]} == {"b", "d", "q", "r"}
    assert {g["kind"] for g in out["hygiene_gaps"]} == {"briefed_without_prediction"}


def test_candidate_without_prediction_is_not_a_gap():
    """A candidate has not been briefed yet, so no prediction is owed."""
    out = cp.collect(led([tech("c", None, "candidate"),
                          tech("p", None, "proposal"),
                          tech("x", None, "rejected")]))
    assert out["hygiene_gaps"] == []


def test_prediction_without_runs_is_a_gap():
    out = cp.collect(led([tech("t", 0.6, "briefed", [])]))
    assert out["n"] == 0
    assert [g["kind"] for g in out["hygiene_gaps"]] == ["prediction_without_run"]


def test_dangling_run_id_is_a_gap_not_a_crash():
    out = cp.collect(led([tech("t", 0.6, "done", ["2026-07-01_m_missing"])]))
    assert out["n"] == 0
    g = out["hygiene_gaps"][0]
    assert g["kind"] == "run_id_not_found" and g["run_id"] == "2026-07-01_m_missing"


def test_done_technique_with_no_verdict_anywhere_is_a_gap():
    out = cp.collect(led([tech("t", 0.6, "done", ["2026-07-01_m_t"])],
                         [run("2026-07-01_m_t", None)]))
    assert out["n"] == 0
    assert [g["kind"] for g in out["hygiene_gaps"]] == ["done_without_realised_verdict"]


# ------------------------------------------- hand-off contract + read-only-ness

def test_scorer_inputs_feed_scorer_calibration_report():
    """The point of the module: its output must drop straight into the §C15.3.8
    math with no adapter. `report(**scorer_inputs(...))` is the whole report half."""
    out = cp.collect(led(
        [tech(f"t{i}", p, "done", [f"2026-07-{i + 1:02d}_m_t"])
         for i, p in enumerate((0.9, 0.8, 0.2, 0.1))],
        [run("2026-07-01_m_t", "win"), run("2026-07-02_m_t", "win"),
         run("2026-07-03_m_t", "loss"), run("2026-07-04_m_t", "loss")]))
    kw = cp.scorer_inputs(out)
    assert kw == {"preds": [0.9, 0.8, 0.2, 0.1], "outcomes": [1, 1, 0, 0]}
    rep = scorer_calibration.report(**kw)          # the one-liner, exercised
    assert rep["n"] == 4
    # a scorer this well-ordered has a small Brier; pin the exact value so a
    # silent re-mapping of outcomes cannot pass this test.
    assert round(rep["brier"], 4) == round(((0.1 ** 2) * 2 + (0.2 ** 2) * 2) / 4, 4)


def test_empty_join_would_raise_in_the_math_which_is_why_we_stop_short():
    """Documents the guard that makes an n=0 calibration report impossible:
    scorer_calibration refuses an empty pair list, so calibration_pairs must be
    read for its hygiene gaps, not asked for an ECE."""
    kw = cp.scorer_inputs(cp.collect(led()))
    assert kw == {"preds": [], "outcomes": []}
    assert cp.below_calibration_floor(0) and cp.below_calibration_floor(9)
    assert not cp.below_calibration_floor(cp.MIN_PAIRS_FOR_CALIBRATION)
    try:
        scorer_calibration.report(**kw)
    except ValueError as e:
        assert "at least one" in str(e)
    else:
        raise AssertionError("scorer_calibration.report accepted an empty pair list")


def test_module_never_writes_the_ledger(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(led([tech("t", 0.5, "done", ["2026-07-01_m_t"])],
                                [run("2026-07-01_m_t", "win")]), indent=2))
    before = hashlib.md5(p.read_bytes()).hexdigest()
    assert cp.main(["--ledger", str(p)]) == 0
    assert cp.main(["--ledger", str(p), "--per-run"]) == 0
    assert cp.scorer_inputs(ledger=p)["outcomes"] == [1]
    assert hashlib.md5(p.read_bytes()).hexdigest() == before


def test_cli_prints_the_four_keys(tmp_path, capsys):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(led([tech("t", None, "briefed")])))
    assert cp.main(["--ledger", str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"pairs", "pending", "hygiene_gaps", "n"}
    assert out["n"] == 0 and len(out["hygiene_gaps"]) == 1
