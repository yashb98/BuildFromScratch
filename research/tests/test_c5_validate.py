"""Tests for c5_validate — the machine-checkable §C5 pre-launch lint (upgrade-plan #11)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import c5_validate as c5  # noqa: E402


def _complete(**over):
    e = {"run_id": "2026-07-23_m_x", "objective": "pretrain-ablation", "framework": "jax",
         "smoke": "pass", "budget": {"tokens": 1}, "probe": {"tokens_per_sec": 1},
         "eta_hours": 4.0, "resume_roundtrip": "pass", "sentinel": "armed",
         "guards": "verified"}
    e.update(over)
    return e


def test_complete_evidence_passes():
    r = c5.validate_c5(_complete())
    assert r["ok"] is True and r["missing_items"] == [] and r["missing_identity"] == []


def test_accepts_both_key_stylings():
    # numbered c5_x style instead of flat style — must also satisfy
    e = {"run_id": "r", "objective": "finetune", "framework": "pytorch",
         "c5_0_smoke": {"result": "pass"}, "c5_2_budget": {"tokens": 1},
         "c5_3_probe": {}, "c5_4_eta_hours": 3, "c5_5_resume": "pass",
         "c5_6_sentinel": "armed", "c5_7_guards": "verified"}
    # c5_3_probe is empty {} -> counts as MISSING (empty value), everything else present
    r = c5.validate_c5(e)
    assert r["missing_items"] == ["c5.3_probe"]
    e["c5_3_probe"] = {"tokens_per_sec": 1}
    assert c5.validate_c5(e)["ok"] is True


def test_missing_items_are_reported():
    r = c5.validate_c5(_complete(guards=None, sentinel=""))
    assert set(r["missing_items"]) == {"c5.7_guards", "c5.6_sentinel"}
    assert r["ok"] is False


def test_missing_identity_fails():
    e = _complete(); del e["framework"]
    r = c5.validate_c5(e)
    assert "framework" in r["missing_identity"] and r["ok"] is False


def test_non_dict_and_empty():
    assert c5.validate_c5("nope")["ok"] is False
    assert c5.validate_c5({})["ok"] is False


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.json"; good.write_text(__import__("json").dumps(_complete()))
    bad = tmp_path / "bad.json"; bad.write_text(__import__("json").dumps(_complete(guards=None)))
    assert c5.main([str(good)]) == 0
    assert c5.main([str(bad)]) == 1
    assert c5.main([str(tmp_path / "absent.json")]) == 2
