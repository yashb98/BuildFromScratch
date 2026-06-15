"""Chaos / recovery tests for research/loop_state.py — turn the orchestrator's
resume / dead-run / auto-resume-cap / fail-open behaviour from spec into
execution-verified assertions (§C22, §C4 recovery). Stdlib-only (CI-safe)."""
import json

import loop_state as ls


def test_resume_lands_at_stored_stage_not_s0(tmp_path):
    p = tmp_path / "loop_state.json"
    st = ls.default_state()
    st["stage"] = "S5"
    ls.save(p, st)
    assert ls.resume_point(p) == "S5"   # NOT "S0"


def test_corrupt_state_fails_open(tmp_path):
    p = tmp_path / "loop_state.json"
    p.write_text("{ this is not json")
    st = ls.load(p)                      # must NOT raise
    assert st["_recovered"] is True and st["stage"] == "S0"


def test_missing_state_fails_open(tmp_path):
    st = ls.load(tmp_path / "absent.json")
    assert st["_recovered"] is True and st["stage"] == "S0"


def test_malformed_or_bad_stage_fails_open(tmp_path):
    p = tmp_path / "loop_state.json"
    p.write_text(json.dumps({"x": 1}))             # valid json, no stage
    assert ls.load(p)["_recovered"] is True
    p.write_text(json.dumps({"stage": "S99"}))     # impossible stage
    assert ls.load(p)["_recovered"] is True


def test_auto_resume_cap(tmp_path):
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    assert ls.record_resume(p, cap=2)["decision"] == "resume"   # 1st
    assert ls.record_resume(p, cap=2)["decision"] == "resume"   # 2nd
    third = ls.record_resume(p, cap=2)                          # 3rd -> crashed
    assert third["decision"] == "crashed" and third["auto_resumes"] == 2


def test_reset_resumes_clears_counter(tmp_path):
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    ls.record_resume(p, cap=2)
    ls.reset_resumes(p)
    assert ls.load(p)["auto_resumes"] == 0


def test_advance_roundtrip_and_in_flight(tmp_path):
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    ls.advance(p, "S3", ts="2026-06-15", in_flight="2026-06-15_qwen3_x")
    st = ls.load(p)
    assert st["stage"] == "S3" and st["in_flight_run"] == "2026-06-15_qwen3_x"
    assert st["updated"] == "2026-06-15"


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    ls.advance(p, "S2")
    assert list(tmp_path.glob(".loopstate_*.tmp")) == []
    assert (tmp_path / "loop_state.json.bak").exists()  # prior version snapshotted


def test_bad_stage_rejected_on_write(tmp_path):
    p = tmp_path / "loop_state.json"
    import pytest
    with pytest.raises(ValueError):
        ls.advance(p, "S42")
    bad = ls.default_state(); bad["stage"] = "ZZ"
    with pytest.raises(ValueError):
        ls.save(p, bad)
