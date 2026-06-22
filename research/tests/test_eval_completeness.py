"""§C25 eval-completeness gate tests — pure CPU, stdlib."""
import sys, pathlib
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


def test_missing_hard_caps_to_directional():
    for st, spec in EC.REGISTRY.items():
        if not spec["required"]:
            continue
        r = EC.check_completeness(st, spec["required"][1:])  # drop first required
        assert not r["complete"] and r["verdict_cap"] == "directional", st
        assert spec["required"][0] in r["missing_hard"]


def test_founding_mistake_is_blocked():
    # a base-eval / pretrain headline on ONLY n=1 val PPL must never be a win
    g = EC.gate_verdict("base-eval", ["valppl_n1_stage_headline"], "win")
    assert g["verdict"] == "directional"
    assert "incomplete-eval" in g["why"]


def test_conditional_only_fires_when_active():
    arch = EC.REGISTRY["architecture"]["required"]
    assert EC.check_completeness("architecture", arch)["complete"]
    fired = EC.check_completeness("architecture", arch, {"touches_positions"})
    assert "length_extrapolation_curve" in fired["missing_hard"]


def test_unknown_stage_cannot_win():
    assert EC.gate_verdict("nope", ["a", "b"], "win")["verdict"] == "inconclusive"


def test_complete_and_significant_is_win():
    full = EC.REGISTRY["serving"]["required"]
    assert EC.gate_verdict("serving", full, "win")["verdict"] == "win"
    assert EC.gate_verdict("serving", full, "loss")["verdict"] == "loss"
