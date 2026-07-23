"""§C25 eval-completeness gate tests — pure CPU, stdlib."""
import sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import eval_completeness as EC


def test_self_test():
    EC._self_test()


def test_all_stages_registered():
    assert set(EC.STAGES) == set(EC.REGISTRY), "STAGES and REGISTRY must agree"


def test_full_battery_complete_per_stage():
    for st, spec in EC.REGISTRY.items():
        r = EC.check_completeness(st, spec["required"])
        assert r["complete"] and r["verdict_cap"] is None, st


def test_missing_hard_caps_below_win():
    # §C25.3: a missing HARD item caps the run at `promising` — the strongest verdict an
    # incomplete battery may carry. (Was the single word `directional` before the
    # 2026-07-22 vocabulary split; the cap itself is unchanged.)
    for st, spec in EC.REGISTRY.items():
        if not spec["required"]:
            continue
        r = EC.check_completeness(st, spec["required"][1:])  # drop first required
        assert not r["complete"] and r["verdict_cap"] == "promising", st
        assert r["verdict_cap"] in EC.NEUTRAL_VERDICTS
        assert spec["required"][0] in r["missing_hard"]


def test_hard_incomplete_never_yields_win():
    # the cap, exhaustively: no stage × no significance verdict may win with a HARD item absent
    for st, spec in EC.REGISTRY.items():
        if not spec["required"]:
            continue
        partial = spec["required"][1:]
        for sig in sorted(EC.ACCEPTED_SIGNIFICANCE):
            g = EC.gate_verdict(st, partial, sig)
            assert g["verdict"] != "win", (st, sig)
            assert g["verdict"] in EC.NEUTRAL_VERDICTS, (st, sig, g["verdict"])
            assert "incomplete-eval" in g["why"], (st, sig)


def test_null_and_promising_do_not_collapse():
    # the whole point of the 2026-07-22 split: under the SAME incomplete battery,
    # "found something, one gate short" and "found nothing" must not wear the same word.
    partial = EC.REGISTRY["sft"]["required"][1:]
    effect = EC.gate_verdict("sft", partial, "win")["verdict"]
    no_effect = EC.gate_verdict("sft", partial, "null")["verdict"]
    assert effect == "promising"
    assert no_effect == "null"
    assert effect != no_effect


def test_incomplete_loss_is_not_burned_as_never_repeat():
    # §C25.3: the cap "is NOT a never_repeat loss". ledger.py auto-appends never_repeat[] on
    # verdict=loss, so an incomplete battery must never emit one.
    partial = EC.REGISTRY["preference"]["required"][1:]
    g = EC.gate_verdict("preference", partial, "loss")
    assert g["verdict"] == "inconclusive" and g["verdict"] != "loss"
    assert g["verdict"] in EC.NEUTRAL_VERDICTS
    assert "never_repeat" in g["why"]


def test_deprecated_directional_is_never_emitted():
    assert "directional" in EC.DEPRECATED_VERDICTS
    seen = set()
    for st, spec in EC.REGISTRY.items():
        req = spec["required"]
        for items in (req, req[1:], [], ["valppl_n1_stage_headline"]):
            for sig in sorted(EC.ACCEPTED_SIGNIFICANCE):
                seen.add(EC.gate_verdict(st, items, sig)["verdict"])
    assert "directional" not in seen, seen
    assert seen <= (EC.VERDICTS - EC.DEPRECATED_VERDICTS), seen


def test_vocabulary_matches_the_ledger():
    # the NEUTRAL_VERDICTS reader: this gate speaks the ledger's vocabulary (§C8 authority),
    # and the two must not drift apart.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ledger"))
    import ledger
    assert EC.VERDICTS == frozenset(ledger.VERDICTS)
    assert EC.NEUTRAL_VERDICTS == frozenset(ledger.NEUTRAL_VERDICTS)
    assert "win" not in EC.NEUTRAL_VERDICTS and "loss" not in EC.NEUTRAL_VERDICTS
    assert set(EC.CAP_WHEN_INCOMPLETE.values()) <= EC.NEUTRAL_VERDICTS
    assert set(EC.CAP_WHEN_INCOMPLETE) == EC.ACCEPTED_SIGNIFICANCE  # every input has a cap


def test_cap_postcondition_raises_on_vocabulary_drift(monkeypatch):
    # what makes NEUTRAL_VERDICTS load-bearing: if the cap table is ever edited so an
    # incomplete battery maps onto `win`, gate_verdict refuses instead of writing it.
    monkeypatch.setitem(EC.CAP_WHEN_INCOMPLETE, "win", "win")
    partial = EC.REGISTRY["sft"]["required"][1:]
    with pytest.raises(ValueError, match="cap violated"):
        EC.gate_verdict("sft", partial, "win")


def test_founding_mistake_is_blocked():
    # a base-eval headline on ONLY n=1 val PPL: the sole signal is disallowed as a headline
    # (§C25.7.3), so no admissible effect measurement exists — not even `promising`.
    g = EC.gate_verdict("base-eval", ["valppl_n1_stage_headline"], "win")
    assert g["verdict"] == "inconclusive"
    assert "incomplete-eval" in g["why"] and "valppl_n1_stage_headline" in g["why"]


def test_founding_mistake_not_bypassed_by_a_second_item():
    # Regression (2026-07-23): the disallowed-sole-signal floor once fired only when EXACTLY one
    # item was present (`len(present) == 1`), so pairing the n=1 val-PPL headline with ANY second
    # item — even a report-only figure — let a confounded run reach `promising`/`win`. A figure is
    # not an admissible effect measurement, so the run must still floor to `inconclusive`.
    g = EC.gate_verdict("base-eval", ["valppl_n1_stage_headline", "figure"], "win")
    assert g["verdict"] == "inconclusive", g
    assert g["completeness"]["disallowed_sole_signal"] == ["valppl_n1_stage_headline"]
    assert g["completeness"]["verdict_cap"] == "inconclusive"   # ceiling matches the actual downgrade
    # A REAL admissible signal alongside it clears the sole-signal floor: only the HARD-battery cap
    # remains, so a real (but incomplete) effect is `promising` — never floored for sole-ness.
    real_item = EC.REGISTRY["base-eval"]["required"][0]
    assert real_item not in EC.DISALLOWED_SOLE_SIGNAL
    g2 = EC.gate_verdict("base-eval", ["valppl_n1_stage_headline", real_item], "win")
    assert not g2["completeness"]["disallowed_sole_signal"], g2
    assert g2["verdict"] == "promising", g2


def test_conditional_only_fires_when_active():
    arch = EC.REGISTRY["architecture"]["required"]
    assert EC.check_completeness("architecture", arch)["complete"]
    fired = EC.check_completeness("architecture", arch, {"touches_positions"})
    assert "length_extrapolation_curve" in fired["missing_hard"]


def test_unknown_stage_cannot_win():
    g = EC.gate_verdict("nope", ["a", "b"], "win")
    assert g["verdict"] == "inconclusive"
    assert "unknown lifecycle_stage" in g["why"]


def test_complete_and_significant_is_win():
    full = EC.REGISTRY["serving"]["required"]
    assert EC.gate_verdict("serving", full, "win")["verdict"] == "win"
    assert EC.gate_verdict("serving", full, "loss")["verdict"] == "loss"
    # a complete battery is what earns the right to a first-class negative / capped call
    assert EC.gate_verdict("serving", full, "null")["verdict"] == "null"
    assert EC.gate_verdict("serving", full, "promising")["verdict"] == "promising"


def test_unreadable_significance_fails_closed():
    full = EC.REGISTRY["serving"]["required"]
    for junk in (None, "", "directional", "WIN", "banana"):
        g = EC.gate_verdict("serving", full, junk)
        assert g["verdict"] == "inconclusive", junk
        assert g["significance_verdict"] == junk        # raw input kept for provenance
        assert g["significance_read_as"] == "inconclusive"


def test_every_result_stamps_the_vocabulary():
    known = EC.check_completeness("systems", EC.REGISTRY["systems"]["required"])
    unknown = EC.check_completeness("frobnicate", ["x"])
    assert known["verdict_vocab"] == unknown["verdict_vocab"] == EC.VERDICT_VOCAB
