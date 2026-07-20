"""Guard-logic tests for research/boot_resume.sh — the generic @reboot cross-reboot
recovery hook. Drives its `--dry-run` decision path over crafted loop_state.json states
with BFS_ROOT redirected to a fixture, so the safety-critical resume/refuse decisions are
execution-verified WITHOUT a GPU or a real trainer (§C22, §C4/§C5/§C6 recovery).

Covers the guards that gate an unattended relaunch on a box that hard-locks:
  - nothing to resume            (no in-flight run recorded)
  - a run to resume, all clear   -> would-resume
  - sentinel-kill marker present -> refuse (a kill is not auto-resumable at same config)
  - auto-resume cap reached       -> refuse (never loop-crash the box)
  - preflight failing             -> do not launch (exit 1)
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                      # BuildFromScratch/
BOOT = REPO / "research" / "boot_resume.sh"
REAL_LOOP_STATE = REPO / "research" / "loop_state.py"

SENTINEL_OK = "import sys; sys.exit(0)\n"
SENTINEL_FAIL = (
    "import sys\n"
    "sys.exit(1 if 'preflight' in sys.argv else 0)\n"
)


def _fixture(tmp_path, state: dict | None, *, preflight_ok=True, marker=False):
    """Build a minimal BFS_ROOT: research/loop_state.py (real), loop_state.json (crafted),
    a stub sentinel.py, and research/recovery/. Returns the root path."""
    root = tmp_path / "root"
    (root / "research" / "recovery").mkdir(parents=True)
    shutil.copy(REAL_LOOP_STATE, root / "research" / "loop_state.py")
    (root / "sentinel.py").write_text(SENTINEL_OK if preflight_ok else SENTINEL_FAIL)
    statef = root / "research" / "loop_state.json"
    if state is not None:
        statef.write_text(json.dumps(state, indent=2))
    if marker:
        (root / "research" / "loop_state.json.sentinel_kill").write_text("{}")
    return root


# Hermetic trainer pattern (boot_resume.sh's documented BOOT_TRAINER_RE test hook):
# matches no real process, so the §C4.5 concurrency gate cannot see a trainer that
# happens to be running on the box. Without this the default whitelist
# (`train_*.py|run_arms.sh|run_ladder.sh`) pgreps the REAL host, and every test whose
# decision lies past that gate silently degrades to `already-running` — i.e. the
# recovery-chain guards went unverified exactly when a trainer was live, which is the
# only time the recovery chain matters. (Caught 2026-07-20 beside a live train_hybrid.py;
# the later tests in this file already pass their own BOOT_TRAINER_RE.)
NO_TRAINER_RE = "__bfs_hermetic_no_such_trainer__"


def _run(root):
    r = subprocess.run(
        ["bash", str(BOOT), "--dry-run"],
        env={**os.environ, "BFS_ROOT": str(root),
             "BOOT_TRAINER_RE": NO_TRAINER_RE},
        capture_output=True, text=True, timeout=60,
    )
    # DECISION= is printed to stdout; log lines go to stderr.
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("DECISION=")), "")
    return line.replace("DECISION=", "").strip(), r.returncode


def _live_state(**over):
    st = {"schema_version": 1, "iteration_date": None, "stage": "S7",
          "in_flight_run": "2026-07-19_qwen3_x", "train_pid": 4242,
          "ckpt_path": "/x/ckpt.pt", "resume_cmd": "python3 train_ablation.py --resume /x/ckpt.pt",
          "auto_resumes": 0, "last_radar": None, "objective": "any", "notes": "", "updated": None}
    st.update(over)
    return st


def test_no_in_flight_run_is_nothing_to_resume(tmp_path):
    root = _fixture(tmp_path, _live_state(in_flight_run=None, resume_cmd=None))
    decision, rc = _run(root)
    assert decision == "nothing-to-resume" and rc == 0


def test_missing_state_file_is_nothing_to_resume(tmp_path):
    root = _fixture(tmp_path, None)                 # no loop_state.json at all (fail-open)
    decision, rc = _run(root)
    assert decision == "nothing-to-resume" and rc == 0


def test_clear_in_flight_run_would_resume(tmp_path):
    root = _fixture(tmp_path, _live_state())
    decision, rc = _run(root)
    assert decision == "would-resume" and rc == 0


def test_sentinel_kill_marker_refuses(tmp_path):
    root = _fixture(tmp_path, _live_state(), marker=True)
    decision, rc = _run(root)
    assert decision == "refuse-sentinel-kill" and rc == 0


def test_auto_resume_cap_refuses(tmp_path):
    root = _fixture(tmp_path, _live_state(auto_resumes=2))
    decision, rc = _run(root)
    assert decision == "cap-reached" and rc == 0


def test_preflight_failure_does_not_launch(tmp_path):
    root = _fixture(tmp_path, _live_state(), preflight_ok=False)
    decision, rc = _run(root)
    assert decision == "preflight-fail" and rc == 1


def test_guard_tests_are_hermetic_against_a_live_host_trainer(tmp_path):
    """Regression lock for the isolation fix above: a REAL trainer-shaped process on the
    box must not change any decision reached through `_run()`. Before the fix this test's
    scenario silently turned `would-resume` into `already-running`, so the whole
    recovery-guard suite went green-by-accident on an idle box and red during training."""
    tag = "h" + _uuid.uuid4().hex[:8]
    probe = tmp_path / f"train_{tag}.py"                  # matches the DEFAULT whitelist
    probe.write_text("import time; time.sleep(20)\n")
    proc = subprocess.Popen(["python3", str(probe), "20"])
    try:
        _time.sleep(1)                                     # let it appear in the process table
        assert subprocess.run(["pgrep", "-f", f"train_{tag}.py"],
                              capture_output=True).returncode == 0, "probe never started"
        root = _fixture(tmp_path, _live_state())
        decision, rc = _run(root)                          # must be blind to the host process
        assert (decision, rc) == ("would-resume", 0), decision
    finally:
        proc.kill()
        proc.wait()


def test_dry_run_has_no_side_effects(tmp_path):
    """--dry-run must not touch auto_resumes (no record-resume) or spawn anything."""
    root = _fixture(tmp_path, _live_state(auto_resumes=0))
    _run(root)
    st = json.loads((root / "research" / "loop_state.json").read_text())
    assert st["auto_resumes"] == 0                  # cap counter untouched by a dry run


# --- real-resume path tests (hermetic: SETTLE=0, own lock, a unique fake trainer) ---------
# Each uses a per-test-unique probe pattern so the guards can never match a real live trainer
# (run_tests.sh may run during training), and cleans up any process it spawns.
import os as _os
import signal as _signal
import time as _time
import uuid as _uuid


def _real_fixture(tmp_path, probe_tag, *, auto_resumes=0):
    """Fixture whose resume_cmd launches a uniquely-named python sleeper, with match
    patterns scoped to that unique tag so nothing else on the box can match."""
    probe = tmp_path / "root" / f"train_{probe_tag}.py"   # name matches train_<slug>.py
    root = _fixture(tmp_path, _live_state(
        auto_resumes=auto_resumes,
        resume_cmd=f"python3 {probe} 30",
    ))
    probe.write_text("import sys, time; time.sleep(int(sys.argv[1]) if len(sys.argv)>1 else 30)\n")
    env = {
        **os.environ,
        "BFS_ROOT": str(root),
        "BOOT_SETTLE_SECS": "0",
        "BOOT_LOCK": str(root / "boot.lock"),
        "BOOT_TRAINER_RE": f"train_{probe_tag}\\.py",
        "BOOT_PIDCAP_RE": f"python[0-9.]* .*train_{probe_tag}\\.py",
    }
    return root, env, probe_tag


def _kill_probe(tag):
    subprocess.run(["pkill", "-9", "-f", f"train_{tag}.py"], capture_output=True)


def _run_real(root, env):
    r = subprocess.run(["bash", str(BOOT)], env=env, capture_output=True, text=True, timeout=60)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("DECISION=")), "")
    return line.replace("DECISION=", "").strip(), r.returncode


def test_real_resume_launches_and_accounts(tmp_path):
    """Full real path: relaunch resume_cmd, capture the python trainer pid, bump the cap
    counter, write the new pid back — the untested branch the review flagged."""
    tag = "p" + _uuid.uuid4().hex[:8]
    root, env, tag = _real_fixture(tmp_path, tag)
    try:
        decision, rc = _run_real(root, env)
        assert decision == "resumed" and rc == 0
        st = json.loads((root / "research" / "loop_state.json").read_text())
        assert st["auto_resumes"] == 1                       # record-resume accounted exactly once
        assert isinstance(st["train_pid"], int) and st["train_pid"] > 0
        _os.kill(st["train_pid"], 0)                          # the written pid is a live process
    finally:
        _kill_probe(tag)


def test_flock_serializes_concurrent_resume(tmp_path):
    """The CRITICAL fix: two simultaneous @reboot invocations (duplicate crontab lines)
    must yield EXACTLY ONE resume, never two trainers (§C4.5). The flock guarantees it."""
    tag = "c" + _uuid.uuid4().hex[:8]
    root, env, tag = _real_fixture(tmp_path, tag)
    try:
        p1 = subprocess.Popen(["bash", str(BOOT)], env=env, stdout=subprocess.PIPE, text=True)
        p2 = subprocess.Popen(["bash", str(BOOT)], env=env, stdout=subprocess.PIPE, text=True)
        outs = [p1.communicate(timeout=60)[0], p2.communicate(timeout=60)[0]]
        decisions = sorted(
            next((ln.replace("DECISION=", "").strip()
                  for ln in o.splitlines() if ln.startswith("DECISION=")), "")
            for o in outs
        )
        # exactly one resumed; the loser bounced off the lock (or the re-probe)
        assert decisions.count("resumed") == 1, decisions
        assert decisions[0] in ("already-locked", "already-running"), decisions
        st = json.loads((root / "research" / "loop_state.json").read_text())
        assert st["auto_resumes"] == 1                       # cap counter bumped once, not twice
        running = subprocess.run(["pgrep", "-f", f"train_{tag}.py"], capture_output=True, text=True)
        assert len([l for l in running.stdout.split() if l]) == 1  # exactly ONE trainer, not two
    finally:
        _kill_probe(tag)


def test_broad_trainer_re_guards_slug_named_trainer(tmp_path):
    """The other CRITICAL fix: a running ablation-runner train_<SLUG>.py must trip the
    already-running guard (the old static whitelist missed it -> double-launch)."""
    tag = "g" + _uuid.uuid4().hex[:8]
    probe = tmp_path / "root" / f"train_{tag}.py"
    root = _fixture(tmp_path, _live_state(resume_cmd=f"python3 {probe} 30"))
    probe.write_text("import time; time.sleep(30)\n")
    proc = subprocess.Popen(["python3", str(probe), "30"])
    try:
        _time.sleep(1)                                       # let it show up in the process table
        env = {**os.environ, "BFS_ROOT": str(root),
               "BOOT_TRAINER_RE": f"train_{tag}\\.py"}        # default whitelist WOULD also match now
        r = subprocess.run(["bash", str(BOOT), "--dry-run"], env=env, capture_output=True, text=True, timeout=60)
        decision = next((ln.replace("DECISION=", "").strip()
                         for ln in r.stdout.splitlines() if ln.startswith("DECISION=")), "")
        assert decision == "already-running", r.stdout
    finally:
        proc.kill()
        proc.wait()
