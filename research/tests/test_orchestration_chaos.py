"""Chaos / recovery tests for research/loop_state.py — turn the orchestrator's
resume / dead-run / auto-resume-cap / fail-open behaviour from spec into
execution-verified assertions (§C22, §C4 recovery). Stdlib-only (CI-safe)."""
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

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


# --------------------------------------------------------------------------
# Crash-durability parity with ledger.py (§C8): the parent-dir fsync, the
# mode-preserving replace, and the advisory lock. loop_state.json is THE
# recovery-critical file — after a GB10 hard-lock it is what boot_resume.sh
# reads to know a run was in flight, so a write that survives only in the
# page cache loses the run.
# --------------------------------------------------------------------------

def test_save_fsyncs_the_parent_directory(tmp_path, monkeypatch):
    """A crash between os.replace and the directory-metadata flush can bring the
    state file back absent/truncated even though its CONTENTS were fsynced. Spy
    on os.fsync and assert at least one call targeted a DIRECTORY fd."""
    p = tmp_path / "loop_state.json"
    real_fsync = os.fsync
    synced = {"file": 0, "dir": 0}

    def spy(fd):
        try:
            synced["dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"] += 1
        except OSError:                      # never let the spy break the write
            pass
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    ls.save(p, ls.default_state())
    assert synced["file"] >= 1, "save() never fsynced the file contents"
    assert synced["dir"] >= 1, "save() never fsynced the PARENT DIR (rename not durable)"


def test_save_preserves_the_state_file_mode(tmp_path):
    """mkstemp creates 0600 and os.replace carries that onto the target, so
    without an explicit chmod one write silently makes the state unreadable to
    the other sanctioned writers (loop / @reboot hook / liveness cron)."""
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    assert stat.S_IMODE(p.stat().st_mode) == 0o644      # first creation, not 0600
    p.chmod(0o640)
    ls.advance(p, "S4")
    assert stat.S_IMODE(p.stat().st_mode) == 0o640      # preserved across the replace


# A concurrent writer: load -> (widened window) -> save, all inside the module's
# advisory lock, recording its own critical-section [enter, exit] window.
# CLOCK_MONOTONIC is system-wide on Linux, so windows from separate processes
# are directly comparable.
_WRITER = r'''
import json, sys, time
import loop_state as ls

state_path, out_path, start_at = sys.argv[1], sys.argv[2], float(sys.argv[3])
while time.monotonic() < start_at:      # release every writer at the same instant
    time.sleep(0.001)
with ls.locked(state_path):
    enter = time.monotonic()
    st = ls.load(state_path)
    n = st["auto_resumes"]
    time.sleep(0.05)                    # widen the read->write window
    st["auto_resumes"] = n + 1
    ls.save(state_path, st)
    left = time.monotonic()
with open(out_path, "w") as f:
    json.dump([enter, left], f)
'''


def test_lock_serialises_concurrent_writers(tmp_path):
    """Multiple writers are real here — boot_resume.sh, liveness_cron.sh and the
    loop itself all write this file. Four processes each increment auto_resumes
    under the lock; assert (a) no lost update and (b) no two critical sections
    overlapped. Unlocked, all four would read the same value and the counter
    would land at 1."""
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    env = dict(os.environ, PYTHONPATH=str(Path(ls.__file__).resolve().parent))
    n_writers = 4
    start_at = time.monotonic() + 1.0
    procs = []
    for i in range(n_writers):
        out = tmp_path / f"window_{i}.json"
        procs.append((out, subprocess.Popen(
            [sys.executable, "-c", _WRITER, str(p), str(out), repr(start_at)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    windows = []
    for out, proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, f"writer failed: {err}"
        windows.append(json.loads(out.read_text()))

    assert ls.load(p)["auto_resumes"] == n_writers, "lost update — writers raced"
    windows.sort()
    for (_, earlier_exit), (later_enter, _) in zip(windows, windows[1:]):
        assert earlier_exit <= later_enter, "critical sections overlapped"
    assert (tmp_path / "loop_state.json.lock").exists()


def test_lock_is_reentrant_within_one_process(tmp_path):
    """flock is per open-file-description, so a nested os.open + LOCK_EX would
    deadlock against ourselves. Timed, because a self-deadlock degrades to the
    bounded LOCK_TIMEOUT_S fail-open wait rather than an outright hang: finishing
    fast IS the assertion."""
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    t0 = time.monotonic()
    with ls.locked(p):
        with ls.locked(p):
            ls.advance(p, "S6", ts="2026-07-23")   # a mutator that locks again
        ls.register(p, train_pid=4242)
    elapsed = time.monotonic() - t0
    assert elapsed < ls.LOCK_TIMEOUT_S / 2, f"nested lock stalled for {elapsed:.1f}s"
    st = ls.load(p)
    assert st["stage"] == "S6" and st["train_pid"] == 4242
    assert ls._LOCKS == {}, "lock fd not released when the outermost block exited"


def test_acquire_lock_fails_open_when_contended(tmp_path):
    """A held lock must never brick a writer forever: with the wait exhausted,
    acquire_lock returns None (proceed unlocked) instead of blocking. A lost
    update is recoverable; a hung @reboot hook forfeits the once-per-boot
    recovery entirely."""
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    held = ls.acquire_lock(p)
    assert held is not None
    try:
        assert ls.acquire_lock(p, timeout=0.0) is None
    finally:
        ls.release_lock(held)
    fd = ls.acquire_lock(p, timeout=0.0)
    assert fd is not None                     # released -> free again
    ls.release_lock(fd)


# --------------------------------------------------------------------------
# register(): the sanctioned setter for the §C5 flat in-flight fields, so
# boot_resume.sh / ablation-runner stop open-coding load-mutate-save (which
# skips the .bak, the parent-dir fsync and the lock).
# --------------------------------------------------------------------------

def test_register_roundtrips_the_recovery_fields(tmp_path):
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    ls.register(p, in_flight_run="2026-07-23_qwen3_x", train_pid=4242,
                ckpt_path="/x/exp/ckpt_step_900.pt",
                resume_cmd="bash run_arms.sh --resume", auto_resumes=0,
                ts="2026-07-23")
    st = ls.load(p)
    assert st["in_flight_run"] == "2026-07-23_qwen3_x"
    assert st["train_pid"] == 4242
    assert st["ckpt_path"] == "/x/exp/ckpt_step_900.pt"
    assert st["resume_cmd"] == "bash run_arms.sh --resume"
    assert st["auto_resumes"] == 0 and st["updated"] == "2026-07-23"
    assert st["stage"] == "S0"          # registering a trainer is NOT a transition
    assert list(tmp_path.glob(".loopstate_*.tmp")) == []


def test_register_touches_only_the_fields_passed(tmp_path):
    """boot_resume.sh's real call: write back ONLY the new trainer pid after a
    relaunch, without disturbing the run id / ckpt / resume_cmd it just used."""
    p = tmp_path / "loop_state.json"
    st = ls.default_state()
    st.update(stage="S7", in_flight_run="2026-07-19_qwen3_x",
              train_pid=111, ckpt_path="/x/ckpt.pt", resume_cmd="bash r.sh",
              auto_resumes=1)
    ls.save(p, st)
    ls.register(p, train_pid=222)
    back = ls.load(p)
    assert back["train_pid"] == 222
    assert back["in_flight_run"] == "2026-07-19_qwen3_x"
    assert back["ckpt_path"] == "/x/ckpt.pt" and back["resume_cmd"] == "bash r.sh"
    assert back["auto_resumes"] == 1 and back["stage"] == "S7"


def test_register_clears_fields_when_a_run_finishes(tmp_path):
    p = tmp_path / "loop_state.json"
    st = ls.default_state()
    st.update(in_flight_run="2026-07-19_qwen3_x", train_pid=111,
              ckpt_path="/x/ckpt.pt", resume_cmd="bash r.sh")
    ls.save(p, st)
    ls.register(p, in_flight_run=None, train_pid=None, ckpt_path=None,
                resume_cmd=None, auto_resumes=0)
    back = ls.load(p)
    for k in ls.IN_FLIGHT_FIELDS:
        assert back[k] is None          # nothing to resume
    assert back["auto_resumes"] == 0


def test_register_rejects_shapes_the_recovery_chain_cannot_use(tmp_path):
    """boot_resume.sh does `[ -z "$RUN_ID" ]` on a stringified field and
    `pgrep`/`sentinel watch --pid` on the pid — a nested object or a bogus pid
    would silently break the @reboot recovery instead of failing loudly."""
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    with pytest.raises(ValueError):
        ls.register(p, in_flight_run={"run_id": "x"})    # §C5: the RUN_ID string
    with pytest.raises(ValueError):
        ls.register(p, train_pid=0)
    with pytest.raises(ValueError):
        ls.register(p, train_pid=-7)
    with pytest.raises(ValueError):
        ls.register(p, train_pid=True)                   # bool is an int subclass
    with pytest.raises(ValueError):
        ls.register(p, ckpt_path=17)
    with pytest.raises(ValueError):
        ls.register(p, auto_resumes=-1)
    assert ls.load(p)["train_pid"] is None               # nothing was written


def test_register_on_a_corrupt_state_fails_open_and_repairs(tmp_path):
    """The realistic post-hard-lock case: the state file came back truncated.
    register must still land a usable recovery pointer (fail-open load ->
    canonical schema -> durable save), not raise on the recovery path."""
    p = tmp_path / "loop_state.json"
    p.write_text("{ truncated")
    ls.register(p, in_flight_run="2026-07-23_qwen3_x", train_pid=9,
                resume_cmd="bash r.sh")
    back = ls.load(p)
    assert back["_recovered"] is False                   # a valid file again
    assert back["in_flight_run"] == "2026-07-23_qwen3_x" and back["train_pid"] == 9
    assert back["stage"] == "S0" and "_recovered" not in p.read_text()


def test_register_cli_is_the_replacement_for_open_coded_writes(tmp_path):
    """The exact shell surface boot_resume.sh should use instead of its inline
    `st = ls.load(p); st['train_pid'] = pid; ls.save(p, st)` heredoc."""
    p = str(tmp_path / "loop_state.json")
    ls.save(p, ls.default_state())
    assert ls.main(["--path", p, "register", "--train-pid", "1234",
                    "--in-flight-run", "2026-07-23_qwen3_x",
                    "--resume-cmd", "bash run_arms.sh", "--ts", "2026-07-23"]) == 0
    st = ls.load(p)
    assert st["train_pid"] == 1234 and st["in_flight_run"] == "2026-07-23_qwen3_x"
    assert ls.main(["--path", p, "register", "--clear", "train_pid"]) == 0
    assert ls.load(p)["train_pid"] is None
    # usage errors are exit 2, never a silent no-op
    assert ls.main(["--path", p, "register"]) == 2                       # nothing to do
    assert ls.main(["--path", p, "register", "--train-pid", "5",
                    "--clear", "train_pid"]) == 2                        # contradictory
    assert ls.main(["--path", p, "register", "--train-pid", "0"]) == 2   # invalid pid
