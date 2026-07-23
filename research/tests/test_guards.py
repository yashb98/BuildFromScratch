"""Tests for the L1 memory guards every GPU process depends on — safe_cuda (torch)
and jax_safe_env (JAX). Upgrade-plan item 16 (batch7 Q2: test the killers first):
these had NO tests, yet their pure-CPU logic — fraction-range validation, env-var
composition, and the PREALLOCATE=true refusal — is the first line of defense against
the unified-memory over-allocation that hard-crashed the box on 2026-06-08.

Item 16's remainder (added 2026-07-23, the day the box hard-locked at 15:24 and the
ladder went unrecovered) covers the two *shell* killers that were named but untested:
  * run_arch_ladder.sh's trainer_alive() — the §C4.5 one-GPU-job-at-a-time gate, whose
    argv[0]-must-be-python check exists because a bare pgrep matched greps/editors/agent
    sessions and wrongly deferred all 15 cells on 2026-07-21 22:03;
  * the recovery chain liveness_cron.sh -> cron_runner.sh — sentinel's exit-4 routing and
    the flock that stops a resume overlapping the nightly fire.

All tests here are pure CPU and hermetic: the shell scripts are driven from tmp_path with
fake sentinel/claude/cron_runner stand-ins, so no trainer, no GPU work, no real ledger and
no real loop_state are ever touched. Fixture processes are `sleep`-alikes only.
"""
import contextlib
import fcntl
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ safe_cuda
import safe_cuda  # noqa: E402  (imports os only; torch is imported inside guard())


@pytest.mark.parametrize("bad", [0.0, -0.1, 0.96, 1.0, 1.5, 2.0])
def test_guard_rejects_unsafe_fraction_before_touching_torch(bad):
    """guard() validates 0 < fraction <= 0.95 and raises BEFORE the torch call, so an
    unsafe fraction is refused even on a box with no CUDA. This is the check that stops
    a 0.99 (or a typo'd 1.5) from handing the whole unified pool to one process."""
    with pytest.raises(ValueError):
        safe_cuda.guard(bad)


def test_safe_cuda_import_sets_alloc_conf():
    """Importing safe_cuda must compose PYTORCH_CUDA_ALLOC_CONF (expandable segments etc.)
    so fragmentation doesn't trip the OOM-killer on the shared pool."""
    assert "PYTORCH_CUDA_ALLOC_CONF" in os.environ
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] != ""


def test_guard_default_fraction_is_the_documented_085():
    import inspect
    sig = inspect.signature(safe_cuda.guard)
    assert sig.parameters["fraction"].default == 0.85   # §C1 default; must not drift


# ------------------------------------------------------------------ jax_safe_env
def _reload_jax_safe_env(env):
    """Import jax_safe_env fresh under a controlled environment, returning (module|exc)."""
    for k in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop("jax_safe_env", None)
    try:
        return importlib.import_module("jax_safe_env"), None
    except Exception as e:  # noqa: BLE001
        return None, e


def test_jax_safe_env_sets_no_prealloc_and_half_pool():
    saved = {k: os.environ.get(k) for k in
             ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")}
    try:
        mod, exc = _reload_jax_safe_env({})
        assert exc is None
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"  # no ~90GB startup grab
        assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"   # cap at ~60GB
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.modules.pop("jax_safe_env", None)


def test_jax_safe_env_refuses_prealloc_true():
    """If a caller already set PREALLOCATE=true (the ~90GB unified-pool grab that would
    crash the box), importing the guard must REFUSE loudly rather than proceed."""
    saved = {k: os.environ.get(k) for k in
             ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")}
    try:
        mod, exc = _reload_jax_safe_env({"XLA_PYTHON_CLIENT_PREALLOCATE": "true"})
        assert mod is None and isinstance(exc, RuntimeError)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.modules.pop("jax_safe_env", None)


# =========================================================== trainer_alive() (§C4.5)
# The arch-ladder driver's one-GPU-job-at-a-time gate. A FALSE POSITIVE stalls the
# ladder (2026-07-21: all 15 cells deferred by a pgrep that matched a grep); a FALSE
# NEGATIVE lets two trainers share the ~119 GB pool and OOM-kills the box. Tests drive
# the SHIPPED function text, extracted verbatim from run_arch_ladder.sh — never a
# reimplementation — against controlled sleeping fixtures. No fixture does GPU work.

LADDER_DRIVER = (ROOT / "HybridSSM-0.2B" / "experiments"
                 / "2026-07-21_hybrid-ssm-0.2b_arch-ladder" / "run_arch_ladder.sh")
PGREP_PATTERN = r"train_[A-Za-z0-9_]*\.py"   # the pattern the shipped function pgreps on


def _extract_shell_function(script: Path, name: str) -> str:
    """Return the verbatim `name () { ... }` block from a bash script.

    Fails loudly rather than silently yielding an empty body: if the driver renames or
    reshapes the guard, every test below must break instead of passing vacuously.
    """
    lines = script.read_text().splitlines()
    opens = [i for i, ln in enumerate(lines) if ln.startswith(f"{name} () {{")]
    assert len(opens) == 1, f"{name}() not found exactly once in {script}"
    i = opens[0]
    ends = [j for j in range(i + 1, len(lines)) if lines[j] == "}"]
    assert ends, f"unterminated {name}() in {script}"
    body = "\n".join(lines[i:ends[0] + 1])
    assert PGREP_PATTERN in body, "shipped pgrep pattern changed — fixture names are stale"
    return body


def _trainer_alive_probe(tmp_path: Path, name: str = "probe.sh") -> Path:
    """A standalone script that sources the shipped trainer_alive() and exits with its
    status (0 = a real trainer is alive). `set -uo pipefail` mirrors the driver."""
    (tmp_path / "fn.sh").write_text(_extract_shell_function(LADDER_DRIVER, "trainer_alive") + "\n")
    probe = tmp_path / name
    probe.write_text('set -uo pipefail\nsource "$(dirname "$0")/fn.sh"\ntrainer_alive\n')
    return probe


def _probe_rc(probe: Path) -> int:
    return subprocess.run(["bash", str(probe)], capture_output=True, text=True, timeout=60).returncode


def _require_idle_box(probe: Path):
    """Every case below is a transition test from 'no trainer'. If a real trainer is live
    on this box, skip rather than report a meaningless result."""
    if _probe_rc(probe) == 0:
        pytest.skip("a real trainer is alive on this box — cannot test the idle transition")


@pytest.fixture
def spawn():
    """Start throwaway fixture processes and guarantee they die with the test."""
    started = []

    def _spawn(argv):
        p = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append(p)
        deadline = time.time() + 10
        while time.time() < deadline:
            out = subprocess.run(["pgrep", "-f", PGREP_PATTERN],
                                 capture_output=True, text=True).stdout.split()
            if str(p.pid) in out:
                return p
            time.sleep(0.05)
        pytest.fail(f"fixture pid {p.pid} never matched {PGREP_PATTERN}")

    yield _spawn
    for p in started:
        p.kill()
        p.wait(timeout=10)


def test_trainer_alive_is_false_on_an_idle_box(tmp_path):
    """Baseline: with no trainer running the gate must NOT defer — otherwise the ladder
    never launches a cell (the 2026-07-21 stall)."""
    assert _probe_rc(_trainer_alive_probe(tmp_path)) == 1


def test_trainer_alive_detects_a_real_python_trainer(tmp_path, spawn):
    """True positive: a genuine `python3 .../train_*.py` must be seen, or §C4.5 is broken
    and two trainers can share the unified pool."""
    probe = _trainer_alive_probe(tmp_path)
    _require_idle_box(probe)
    trainer = tmp_path / "train_fixture.py"
    trainer.write_text("import time\ntime.sleep(120)\n")   # sleeps only — no GPU, no model
    spawn([sys.executable, str(trainer)])
    assert _probe_rc(probe) == 0


def test_trainer_alive_detects_a_versioned_interpreter(tmp_path, spawn):
    """The check is `case basename in python*)`, not an exact `python3` — a trainer started
    as python3.x still counts. (The driver's own sentinel re-pgrep uses `python[0-9.]*`.)"""
    probe = _trainer_alive_probe(tmp_path)
    _require_idle_box(probe)
    interp = tmp_path / "python3.99"
    interp.symlink_to(sys.executable)
    trainer = tmp_path / "train_versioned.py"
    trainer.write_text("import time\ntime.sleep(120)\n")
    spawn([str(interp), str(trainer)])
    assert _probe_rc(probe) == 0


def test_trainer_alive_ignores_a_shell_that_merely_mentions_a_trainer(tmp_path, spawn):
    """THE 2026-07-21 22:03 REGRESSION: a shell/grep/agent session whose cmdline contains
    train_*.py is matched by `pgrep -f` but its argv[0] is a shell, not a python
    interpreter. Counting it deferred all 15 cells while sentinel reported trainers=none."""
    probe = _trainer_alive_probe(tmp_path)
    _require_idle_box(probe)
    named = tmp_path / "train_hybrid.py"
    named.write_text("# not executed by this test\n")
    # trailing `:` keeps bash resident: with a lone final simple command bash execs it and
    # the cmdline (and the false positive we are testing for) disappears.
    spawn(["bash", "-c", f"grep -c . {named} >/dev/null; sleep 120; :"])
    assert _probe_rc(probe) == 1


def test_trainer_alive_ignores_a_monitoring_command_on_the_trainer_file(tmp_path, spawn):
    """Same class, non-shell argv[0]: a `tail -f train_hybrid.py` (or an editor holding the
    file open) mentions the trainer but is not one."""
    probe = _trainer_alive_probe(tmp_path)
    _require_idle_box(probe)
    watched = tmp_path / "train_hybrid.py"
    watched.write_text("# not executed by this test\n")
    spawn(["tail", "-f", str(watched)])
    assert _probe_rc(probe) == 1


def test_trainer_alive_never_counts_itself(tmp_path):
    """The `[ "$p" = "$$" ] && continue` line. Run the probe so that it is itself matched by
    the pgrep AND has a python argv[0] (`exec -a python3 bash <file named train_*.py>`):
    without the self-skip the driver would see its own pid and defer forever. Verified
    non-vacuous — deleting that line from the extracted body flips this rc to 0."""
    probe = _trainer_alive_probe(tmp_path, name="train_selfprobe.py")
    r = subprocess.run(["bash", "-c", 'exec -a python3 bash "$1"', "_", str(probe)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, r.stdout + r.stderr


# ================================================= recovery chain (§C6): exit 4 -> resume
# liveness_cron.sh routes ONLY sentinel's exit 4 ("in-flight run's PID is dead") to
# cron_runner.sh "/research-loop resume"; cron_runner.sh's flock stops that resume
# overlapping a live session. On 2026-07-23 this chain was never armed (no BuildFromScratch
# crontab lines) so the dead ladder went unrecovered — the scripts must at least be correct
# for when they ARE armed. Both scripts hardcode their absolute paths, so each test copies
# the shipped text and substitutes ONLY those constants (count==1 asserted, so a drift in
# the script breaks the test loudly instead of testing a stale string). Nothing real is
# invoked: sentinel, claude and (where not under test) cron_runner are fakes in tmp_path.

LIVENESS_CRON = ROOT / "research" / "liveness_cron.sh"
CRON_RUNNER = ROOT / "research" / "cron_runner.sh"
REPO_CONST = "REPO=/home/yashb98/Downloads/BuildFromScratch"
CLAUDE_CONST = "CLAUDE=/home/yashb98/.local/bin/claude"
LOCK_CONST = "LOCK=/tmp/research-loop.lock"


def _repoint(script: Path, dest: Path, subs):
    """Copy a shipped script to dest with ONLY the given constant lines rewritten."""
    text = script.read_text()
    for old, new in subs:
        assert text.count(old) == 1, f"expected exactly one {old!r} in {script}"
        text = text.replace(old, new)
    dest.write_text(text)
    dest.chmod(0o755)
    return dest


def _recorder(path: Path, record: Path) -> Path:
    """An executable stand-in that appends its argv to `record` and exits 0."""
    path.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "{record}"\nexit 0\n')
    path.chmod(0o755)
    return path


def _fake_repo(tmp_path, liveness_rc, real_cron_runner=False):
    """A fake REPO tree for liveness_cron.sh: a sentinel.py that exits `liveness_rc`, and a
    cron_runner.sh that is either a recorder or the SHIPPED script (repointed at a fake
    claude). Returns (patched liveness_cron, cron_record, claude_record, lock)."""
    repo = tmp_path / "repo"
    (repo / "research").mkdir(parents=True)
    (repo / "sentinel.py").write_text(
        "import os, sys\n"
        "print('[fake-sentinel]', sys.argv[1:])\n"
        "sys.exit(int(os.environ['FAKE_LIVENESS_RC']))\n")
    cron_record = tmp_path / "cron_runner_argv.txt"
    claude_record = tmp_path / "claude_argv.txt"
    lock = tmp_path / "research-loop.lock"
    if real_cron_runner:
        _recorder(tmp_path / "claude", claude_record)
        _repoint(CRON_RUNNER, repo / "research" / "cron_runner.sh",
                 [(REPO_CONST, f"REPO={repo}"),
                  (CLAUDE_CONST, f"CLAUDE={tmp_path / 'claude'}"),
                  (LOCK_CONST, f"LOCK={lock}")])
    else:
        _recorder(repo / "research" / "cron_runner.sh", cron_record)
    patched = _repoint(LIVENESS_CRON, tmp_path / "liveness_cron.sh",
                       [(REPO_CONST, f"REPO={repo}")])
    return patched, cron_record, claude_record, lock


def _run_liveness(patched, rc, *args):
    env = dict(os.environ, FAKE_LIVENESS_RC=str(rc))
    return subprocess.run(["bash", str(patched), *args],
                          capture_output=True, text=True, timeout=120, env=env)


@contextlib.contextmanager
def _hold_lock(lock: Path):
    """Hold cron_runner's flock from this process, exactly as a live session would."""
    fd = os.open(str(lock), os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_liveness_cron_routes_exit4_to_a_resume(tmp_path):
    """sentinel exit 4 = an in-flight run whose PID is dead. This is the ONLY code that
    turns that signal into recovery; if it stops firing, a dead run stays dead (2026-07-23)."""
    patched, cron_record, _, _ = _fake_repo(tmp_path, 4)
    r = _run_liveness(patched, 4)
    assert r.returncode == 0, r.stderr
    assert cron_record.read_text().splitlines() == ["/research-loop resume"]


@pytest.mark.parametrize("rc", [0, 1, 2, 3, 5])
def test_liveness_cron_launches_nothing_except_on_exit4(tmp_path, rc):
    """rc 0 = idle or running fine; anything else = probe glitch. Both must fail OPEN: a
    spurious session could relaunch on top of a healthy run and break §C4.5."""
    patched, cron_record, _, _ = _fake_repo(tmp_path, rc)
    r = _run_liveness(patched, rc)
    assert r.returncode == 0, r.stderr
    assert not cron_record.exists()


def test_liveness_cron_logs_the_probe_result(tmp_path):
    """The cron is unattended: its only observability is research/cron_logs/liveness.log."""
    patched, _, _, _ = _fake_repo(tmp_path, 0)
    _run_liveness(patched, 0)
    log = (tmp_path / "repo" / "research" / "cron_logs" / "liveness.log").read_text()
    assert "liveness rc=0 -> no action" in log


def test_liveness_cron_does_not_hang_on_a_bad_settle_arg(tmp_path):
    """The @reboot cron passes a settle delay (180). A non-numeric arg must not become
    `sleep abc` or an infinite wait — the reboot recovery would never fire."""
    patched, cron_record, _, _ = _fake_repo(tmp_path, 4)
    started = time.time()
    r = _run_liveness(patched, 4, "abc")
    assert r.returncode == 0, r.stderr
    assert time.time() - started < 30
    assert cron_record.read_text().splitlines() == ["/research-loop resume"]


def test_recovery_chain_reaches_the_cli_with_a_resume_prompt(tmp_path):
    """End-to-end through the SHIPPED cron_runner.sh: exit 4 must land as
    `claude -p "/research-loop resume"`. The CLI itself is a recorder — no session, no
    trainer, no GPU. (The real relaunch decision happens inside that session's S1/§C5.)"""
    patched, _, claude_record, _ = _fake_repo(tmp_path, 4, real_cron_runner=True)
    r = _run_liveness(patched, 4)
    assert r.returncode == 0, r.stderr
    assert claude_record.read_text().splitlines() == [
        "-p", "/research-loop resume", "--dangerously-skip-permissions"]


def test_flock_blocks_a_second_concurrent_session(tmp_path):
    """The flock is what keeps the 30-min liveness cron, the @reboot cron and the nightly
    fire from running three Claude sessions at once against one ledger and one GPU."""
    _, _, claude_record, lock = _fake_repo(tmp_path, 4, real_cron_runner=True)
    runner = tmp_path / "repo" / "research" / "cron_runner.sh"
    with _hold_lock(lock):
        r = subprocess.run([str(runner), "/research-loop"], capture_output=True,
                           text=True, timeout=120)
    assert r.returncode == 0, r.stderr          # skipping a fire is not an error
    assert not claude_record.exists()
    logs = list((tmp_path / "repo" / "research" / "cron_logs").glob("*.log"))
    assert any("skipping this fire" in p.read_text() for p in logs)


def test_flock_is_released_so_the_next_fire_runs(tmp_path):
    """A lock that is never released would silently retire the loop — the failure mode is
    indistinguishable from 'nothing to do', so assert the fire after the holder exits."""
    _, _, claude_record, lock = _fake_repo(tmp_path, 4, real_cron_runner=True)
    runner = tmp_path / "repo" / "research" / "cron_runner.sh"
    with _hold_lock(lock):
        subprocess.run([str(runner), "/research-loop"], capture_output=True,
                       text=True, timeout=120)
    r = subprocess.run([str(runner), "/research-loop"], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert claude_record.read_text().splitlines() == [
        "-p", "/research-loop", "--dangerously-skip-permissions"]


def test_recovery_resume_cannot_overlap_a_live_session(tmp_path):
    """The composite §C6 property: even when liveness says 'dead run, recover', the resume
    must be swallowed by the flock while a session still holds it — otherwise a slow
    nightly session plus a 30-min liveness fire could launch two loops (and two trainers)."""
    patched, _, claude_record, lock = _fake_repo(tmp_path, 4, real_cron_runner=True)
    with _hold_lock(lock):
        r = _run_liveness(patched, 4)
    assert r.returncode == 0, r.stderr
    assert not claude_record.exists()
