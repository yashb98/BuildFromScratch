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


def test_default_state_carries_full_live_schema(tmp_path):
    """Schema-fork guard: default_state() must emit the §C5 pinned 12-key shape,
    NOT a divergent 7-key one. In particular the recovery-critical flat in-flight
    fields must be present (as None) so a fail-open recovery is schema-shaped."""
    d = ls.default_state()
    for k in ("schema_version", "iteration_date", "stage", "in_flight_run",
              "train_pid", "ckpt_path", "resume_cmd", "auto_resumes",
              "last_radar", "objective", "notes", "updated"):
        assert k in d, f"default_state missing canonical key {k!r}"
    for k in ls.IN_FLIGHT_FIELDS:
        assert d[k] is None            # no in-flight run on a fresh/recovered default
    assert d["objective"] == "any"     # §C13 selection filter must not be defeated
    assert "last_marker" not in d and "iteration" not in d  # dead fork keys gone


def test_fail_open_recovery_has_recovery_keys(tmp_path):
    """A corrupt state recovered mid-flight must still expose the recovery keys
    as None (a clean 'nothing to resume'), never KeyError on a 7-key default."""
    p = tmp_path / "loop_state.json"
    p.write_text("}{ truncated garbage")
    st = ls.load(p)
    assert st["_recovered"] is True
    assert st["train_pid"] is None and st["ckpt_path"] is None and st["resume_cmd"] is None


def test_in_flight_fields_survive_resume_accounting(tmp_path):
    """The dead-trainer case: a VALID state carrying a live trainer's pid/ckpt/
    resume_cmd must preserve all three across record_resume — the recovery chain
    re-adopts the trainer from exactly these fields."""
    p = tmp_path / "loop_state.json"
    st = ls.default_state()
    st.update(stage="S7", in_flight_run="2026-07-19_qwen3_x",
              train_pid=424242, ckpt_path="/x/exp/ckpt_step_900.pt",
              resume_cmd="python train_ablation.py --resume /x/exp/ckpt_step_900.pt")
    ls.save(p, st)
    ls.record_resume(p, cap=2)                       # one dead-run recovery attempt
    back = ls.load(p)
    assert back["train_pid"] == 424242
    assert back["ckpt_path"] == "/x/exp/ckpt_step_900.pt"
    assert back["resume_cmd"].endswith("ckpt_step_900.pt")
    assert back["in_flight_run"] == "2026-07-19_qwen3_x"
    assert back["auto_resumes"] == 1                 # accounting still advanced


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
