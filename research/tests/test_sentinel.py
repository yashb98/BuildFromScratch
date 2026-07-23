"""Regression tests for sentinel.py — the watchdog that keeps the GB10
unified-memory box from a hard crash. These pin the §C6 safety invariants:
the kill-at ceiling stays below the 0.85 per-process guard, the PID-reuse
guard refuses to signal a recycled PID, liveness fails OPEN on a bad state
file, and the kill path actually SIGTERMs the trainer and writes an atomic
marker.

Stdlib-only, CPU-only, no model load. The kill-path test spawns a real,
killable child of THIS process (a `sleep`), so nothing else on the box is
ever signaled. meminfo is monkeypatched to simulate memory pressure
deterministically — the box's real memory state is never relied upon.
"""
import json
import os
import subprocess
import time
import types

import pytest

import sentinel


def _sleeper(secs):
    return subprocess.Popen(["sleep", str(secs)])


def _dead_pid():
    """A PID that is definitively dead (spawned then reaped)."""
    p = _sleeper(0.01)
    pid = p.pid
    p.wait()
    return pid


# ---------------------------------------------------- the load-bearing invariant

def test_kill_at_default_is_below_guard():
    # §C6: the 0.80 default MUST stay strictly below safe_cuda's 0.85 guard,
    # or the sentinel would fire only after the per-process cliff it can't see.
    assert sentinel.DEFAULT_KILL_AT < 0.85


@pytest.mark.parametrize("bad", ["0.85", "0.90", "0.0", "-0.1"])
def test_watch_rejects_kill_at_outside_range(bad):
    # main() must reject any --kill-at not in (0, 0.85) with argparse exit 2,
    # before watch() ever runs.
    with pytest.raises(SystemExit) as ei:
        sentinel.main(["watch", "--pid", "1", "--kill-at", bad])
    assert ei.value.code == 2


# ---------------------------------------------------- PID / liveness primitives

def test_pid_alive_true_for_self():
    assert sentinel.pid_alive(os.getpid()) is True


def test_pid_alive_false_for_dead():
    assert sentinel.pid_alive(_dead_pid()) is False


def test_pid_reuse_guard():
    p = _sleeper(30)
    try:
        st = sentinel.proc_start_time(p.pid)
        assert st is not None
        assert sentinel.watched_alive(p.pid, st) is True
        # A recycled PID has a different start time -> must read as NOT alive.
        assert sentinel.watched_alive(p.pid, "0") is False
    finally:
        p.kill()
        p.wait()
    # Dead -> not alive even with the original ticks.
    assert sentinel.watched_alive(p.pid, st) is False


def test_pid_rss_nonneg_for_self():
    rss = sentinel.pid_rss_gib(os.getpid())
    assert rss is not None and rss >= 0


def test_meminfo_sane():
    total, avail = sentinel.meminfo()
    assert total > 0 and 0 <= avail <= total


# ---------------------------------------------------- liveness exit-code contract

def test_liveness_missing_state_fails_open(tmp_path):
    assert sentinel.liveness(str(tmp_path / "nope.json")) == 0


def test_liveness_bad_json_fails_open(tmp_path):
    s = tmp_path / "s.json"
    s.write_text("{ not json")
    assert sentinel.liveness(str(s)) == 0


def test_liveness_idle(tmp_path):
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"in_flight_run": None}))
    assert sentinel.liveness(str(s)) == 0


def test_liveness_alive(tmp_path):
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"in_flight_run": "r", "train_pid": os.getpid()}))
    assert sentinel.liveness(str(s)) == 0


def test_liveness_dead_returns_4(tmp_path):
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"in_flight_run": "r", "train_pid": _dead_pid()}))
    assert sentinel.liveness(str(s)) == 4


def test_liveness_non_int_pid_treated_as_dead(tmp_path):
    # Contract pin: a non-int train_pid (e.g. a JSON string) is treated as DEAD
    # (exit 4), failing toward a harmless resume rather than a missed-dead run.
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"in_flight_run": "r", "train_pid": str(os.getpid())}))
    assert sentinel.liveness(str(s)) == 4


def test_liveness_null_pid_fails_open(tmp_path):
    # Regression: an in-flight run with NO recorded pid (null/missing) is an
    # INCONSISTENT state we cannot probe — fail OPEN (exit 0), NOT exit 4, so the
    # */30 recovery cron does not spam `/research-loop resume` every 30 min.
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"in_flight_run": "r", "train_pid": None}))
    assert sentinel.liveness(str(s)) == 0
    s.write_text(json.dumps({"in_flight_run": "r"}))   # key absent entirely
    assert sentinel.liveness(str(s)) == 0


# ---------------------------------------------------- watch: disarm vs kill

def test_watch_disarms_cleanly_on_dead_pid(tmp_path):
    marker = tmp_path / "kill.json"
    rc = sentinel.watch(_dead_pid(), kill_at=0.80,
                        log_path=str(tmp_path / "w.log"),
                        interval=0.01, grace=1, marker_path=str(marker))
    assert rc == 0                     # clean disarm (the normal end of a run)
    assert not marker.exists()         # disarm must NOT write a kill marker


def test_watch_kills_under_pressure_and_writes_marker(tmp_path, monkeypatch):
    # Simulate the pool at 95% so the very first sample crosses kill-at.
    monkeypatch.setattr(sentinel, "meminfo", lambda: (1000, 50))
    child = _sleeper(30)
    marker = tmp_path / "kill.json"
    try:
        rc = sentinel.watch(child.pid, kill_at=0.80,
                            log_path=str(tmp_path / "w.log"),
                            interval=0.05, grace=2, marker_path=str(marker))
    finally:
        try:
            child.kill()
        except ProcessLookupError:
            pass
        child.wait()

    assert rc == 3                      # exit 3 == sentinel killed the trainer
    assert marker.exists()
    rec = json.loads(marker.read_text())
    for k in ("time", "killed_pid", "reason", "pool_usage", "pool_total_gb",
              "trainer_rss_gb", "kill_at", "kill_failed", "log"):
        assert k in rec, f"marker missing key {k!r}"
    assert rec["killed_pid"] == child.pid
    assert rec["kill_at"] == 0.80
    assert rec["kill_failed"] is None   # normal (un-denied) kill path
    assert rec["pool_usage"] >= 0.80


def test_watch_escalates_to_sigkill_after_grace(tmp_path, monkeypatch):
    # The last-resort branch: a trainer that IGNORES SIGTERM must still die via
    # SIGKILL within grace, with the marker written. This is the actual
    # box-saver when a process won't terminate politely.
    monkeypatch.setattr(sentinel, "meminfo", lambda: (1000, 50))   # 95% pressure
    ready = tmp_path / "ready"
    code = ("import signal,time,sys,pathlib;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "pathlib.Path(sys.argv[1]).write_text('1');"
            "time.sleep(60)")
    child = subprocess.Popen(["python3", "-c", code, str(ready)])
    # wait until the SIGTERM-ignore handler is actually installed
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(0.05)
    assert ready.exists(), "child never installed its SIGTERM handler"
    marker = tmp_path / "kill.json"
    logf = tmp_path / "w.log"
    died = False
    try:
        rc = sentinel.watch(child.pid, kill_at=0.80, log_path=str(logf),
                            interval=0.05, grace=2, marker_path=str(marker))
        # the SIGTERM-ignoring child must ACTUALLY be dead now (only SIGKILL could)
        for _ in range(60):
            if child.poll() is not None:
                died = True
                break
            time.sleep(0.05)
    finally:
        try:
            child.kill()
        except ProcessLookupError:
            pass
        child.wait()
    assert rc == 3
    assert died, "sentinel did not actually kill the SIGTERM-ignoring child"
    assert marker.exists()
    log = logf.read_text()
    assert "survived" in log and "SIGKILL" in log     # escalation path taken


# ------------------------------------------- pgrep self-match regression (#1 MLOps blocker)
# 2026-06-17 digest, Health: the GPU-free watcher's `pgrep -f 'train.*\.py'`
# self-matched its OWN command line (the pattern appears in pgrep's argv and in
# the shell wrapper that launched it), so it always reported a trainer and the
# loop never closed. These pin that the watcher (a) never counts itself / its
# probe / its ancestors, and (b) DOES detect a real trainer and reports idle
# when none exist.

def test_self_pid_set_includes_self():
    pids = sentinel.self_pid_set()
    assert os.getpid() in pids
    # the test process has at least one ancestor (the shell / pytest launcher)
    assert len(pids) >= 1
    assert all(isinstance(p, int) and p > 0 for p in pids)


def test_is_probe_flags_pgrep_self_match():
    # The exact self-matching command line from the digest.
    assert sentinel.is_probe(["pgrep", "-f", "train.*\\.py"]) is True
    assert sentinel.is_probe(["bash", "-c", "pgrep -f 'train.*\\.py'"]) is True
    assert sentinel.is_probe(["python3", "sentinel.py", "preflight"]) is True
    assert sentinel.is_probe(["python", "-c", "import safe_cuda"]) is True
    # a genuine trainer argv is NOT a probe
    assert sentinel.is_probe(["python3", "train.py", "--lr", "3e-4"]) is False


def test_is_trainer_rejects_self_and_probe():
    me = os.getpid()
    self_pids = {me}
    # (a) the watcher's own pgrep child carrying the pattern -> NOT a trainer
    assert sentinel.is_trainer(424242, ["pgrep", "-f", "train.*\\.py"],
                               self_pids) is False
    # the sentinel process itself, even with the pattern in argv -> NOT a trainer
    assert sentinel.is_trainer(424243, ["python3", "sentinel.py", "preflight"],
                               self_pids) is False
    # our own PID, even with a perfectly trainer-shaped argv -> excluded by PID
    assert sentinel.is_trainer(me, ["python3", "train.py"], self_pids) is False
    # an ancestor PID (e.g. the launching shell that carries the pattern)
    assert sentinel.is_trainer(99, ["bash", "-c", "pgrep -f train.py"],
                               {me, 99}) is False


def test_is_trainer_detects_real_trainer():
    self_pids = {os.getpid()}
    # (b) a real launched training job IS detected
    assert sentinel.is_trainer(
        555, ["python3", "train.py", "--lr", "3e-4"], self_pids) is True
    assert sentinel.is_trainer(
        556, ["/usr/bin/python3", "/x/pretrain.py"], self_pids) is True
    assert sentinel.is_trainer(
        557, ["torchrun", "--nproc_per_node=8", "train_qwen.py"], self_pids) is True
    # a launcher without the trainer pattern is NOT counted
    assert sentinel.is_trainer(
        558, ["python3", "serve.py"], self_pids) is False
    # 'train' as a substring of a non-launcher arg (e.g. a browser URL) -> no
    assert sentinel.is_trainer(
        559, ["firefox", "https://x/train_schedule"], self_pids) is False
    # empty argv (raced exit) -> no
    assert sentinel.is_trainer(560, [], self_pids) is False


def test_find_trainers_does_not_count_itself(monkeypatch):
    # END-TO-END self-match guard: pgrep returns ONLY this process's own PID
    # (exactly what `pgrep -f sentinel`-style self-match yields). The watcher
    # must report ZERO trainers, not count its own probe.
    me = os.getpid()

    def fake_run(cmd, *a, **k):
        # pgrep self-matches -> returns our own PID; nvidia-smi has no apps.
        out = f"{me}\n" if cmd[0] == "pgrep" else ""
        return types.SimpleNamespace(stdout=out)

    monkeypatch.setattr(sentinel.subprocess, "run", fake_run)
    monkeypatch.setattr(sentinel, "_proc_argv",
                        lambda pid: ["python3", "sentinel.py", "preflight"])
    hits = sentinel.find_trainers([])
    assert hits == [], f"watcher counted itself: {hits}"


def test_find_trainers_detects_real_then_idle(monkeypatch):
    # A REAL matching process (a live sleeper we pretend is a trainer) is
    # detected; when pgrep returns nothing, the watcher reports idle.
    child = _sleeper(30)
    try:
        def fake_run_hit(cmd, *a, **k):
            out = f"{child.pid}\n" if cmd[0] == "pgrep" else ""
            return types.SimpleNamespace(stdout=out)
        monkeypatch.setattr(sentinel.subprocess, "run", fake_run_hit)
        monkeypatch.setattr(
            sentinel, "_proc_argv",
            lambda pid: ["python3", "train.py", "--lr", "3e-4"])
        hits = sentinel.find_trainers([])
        assert any(str(child.pid) in h for h in hits), hits
    finally:
        child.kill()
        child.wait()

    # No matches -> idle (empty hit list).
    monkeypatch.setattr(sentinel.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=""))
    assert sentinel.find_trainers([]) == []


def test_find_trainers_pgrep_unavailable_notes(monkeypatch):
    def boom(*a, **k):
        raise OSError("no pgrep")
    monkeypatch.setattr(sentinel.subprocess, "run", boom)
    notes = []
    assert sentinel.find_trainers(notes) == []
    assert any("pgrep unavailable" in n for n in notes)


# ---------------------------------------------------- preflight health gate

def _all_healthy(monkeypatch):
    monkeypatch.setattr(sentinel, "meminfo", lambda: (1000, 900))         # 90% avail
    monkeypatch.setattr(sentinel, "find_trainers", lambda notes: [])      # none running
    monkeypatch.setattr(sentinel.shutil, "disk_usage",
                        lambda p: types.SimpleNamespace(free=200e9))       # 200 GB free
    monkeypatch.setattr(sentinel.os, "getloadavg", lambda: (1.0, 1.0, 1.0))


def test_preflight_ok_when_all_healthy(monkeypatch, capsys):
    _all_healthy(monkeypatch)
    assert sentinel.preflight() == 0
    assert "PREFLIGHT OK" in capsys.readouterr().out


def test_preflight_low_memory(monkeypatch, capsys):
    _all_healthy(monkeypatch)
    monkeypatch.setattr(sentinel, "meminfo", lambda: (1000, 100))         # 10% < 30%
    assert sentinel.preflight() == 1
    assert "low memory" in capsys.readouterr().out


def test_preflight_trainer_running(monkeypatch, capsys):
    _all_healthy(monkeypatch)
    monkeypatch.setattr(sentinel, "find_trainers",
                        lambda notes: ["pgrep: 123 python train_x.py"])
    assert sentinel.preflight() == 1
    assert "trainer already running" in capsys.readouterr().out


def test_preflight_low_disk(monkeypatch, capsys):
    _all_healthy(monkeypatch)
    monkeypatch.setattr(sentinel.shutil, "disk_usage",
                        lambda p: types.SimpleNamespace(free=50e9))        # < 100 GB
    assert sentinel.preflight() == 1
    assert "disk" in capsys.readouterr().out


def test_preflight_high_load(monkeypatch, capsys):
    _all_healthy(monkeypatch)
    monkeypatch.setattr(sentinel.os, "getloadavg", lambda: (9999.0, 1.0, 1.0))
    assert sentinel.preflight() == 1
    assert "load" in capsys.readouterr().out


def test_preflight_fails_closed_on_bad_meminfo(monkeypatch, capsys):
    _all_healthy(monkeypatch)

    def boom():
        raise OSError("no /proc/meminfo")
    monkeypatch.setattr(sentinel, "meminfo", boom)
    assert sentinel.preflight() == 1
    assert "failing closed" in capsys.readouterr().out


# ------------------------------------------- thermal-kill path (upgrade-plan item 16)
# The thermal guard is the NEWEST, least-mature safety layer, added because the GB10
# hard-locks from heat — and it had ZERO tests. Both readers MUST fail OPEN (an
# unreadable sensor returns None/False, never a fabricated high temp that would kill a
# healthy trainer), and the throttle parse must not confuse "Not Active" with "Active".

class _FakeProc:
    def __init__(self, stdout): self.stdout = stdout

def test_gpu_thermal_parses_temp_and_throttle(monkeypatch):
    monkeypatch.setattr(sentinel.subprocess, "run",
                        lambda *a, **k: _FakeProc("72, Not Active, Not Active\n"))
    assert sentinel.gpu_thermal() == (72.0, False)
    # hw slowdown active -> throttling True
    monkeypatch.setattr(sentinel.subprocess, "run",
                        lambda *a, **k: _FakeProc("91, Active, Not Active\n"))
    assert sentinel.gpu_thermal() == (91.0, True)
    # sw slowdown active -> throttling True
    monkeypatch.setattr(sentinel.subprocess, "run",
                        lambda *a, **k: _FakeProc("88, Not Active, Active\n"))
    assert sentinel.gpu_thermal() == (88.0, True)

def test_gpu_thermal_not_active_is_not_active(monkeypatch):
    """The exact substring trap: 'Not Active' must NOT read as throttling."""
    monkeypatch.setattr(sentinel.subprocess, "run",
                        lambda *a, **k: _FakeProc("60, Not Active, Not Active\n"))
    _, throttling = sentinel.gpu_thermal()
    assert throttling is False

def test_gpu_thermal_fails_open(monkeypatch):
    """A dead/absent nvidia-smi must never fabricate heat (else it kills a healthy run)."""
    def boom(*a, **k): raise OSError("no nvidia-smi")
    monkeypatch.setattr(sentinel.subprocess, "run", boom)
    assert sentinel.gpu_thermal() == (None, False)
    monkeypatch.setattr(sentinel.subprocess, "run",
                        lambda *a, **k: _FakeProc("garbage no digits\n"))
    assert sentinel.gpu_thermal() == (None, False)

def test_hottest_soc_c_picks_max(monkeypatch, tmp_path):
    zones = []
    for milli in (45000, 78000, 60000):        # 45C, 78C, 60C
        z = tmp_path / f"zone{milli}"; z.write_text(str(milli))
        zones.append(str(z))
    monkeypatch.setattr("glob.glob", lambda pat: zones)   # hottest_soc_c imports glob locally
    assert sentinel.hottest_soc_c() == 78.0

def test_hottest_soc_c_fails_open(monkeypatch):
    """No readable zones -> None, never a fabricated temperature."""
    monkeypatch.setattr("glob.glob", lambda pat: [])
    assert sentinel.hottest_soc_c() is None

def test_thermal_constants_debounced_and_ordered():
    """A single nvidia-smi blip must not trigger a kill (>=2 consecutive), and WARN < KILL."""
    assert sentinel.TEMP_KILL_CONSECUTIVE >= 2, "single-sample kill would false-fire on a blip"
    assert sentinel.TEMP_WARN_C < sentinel.TEMP_KILL_C
    assert sentinel.TEMP_KILL_C >= 85, "thermal ceiling set high on purpose (load temps uninstrumented)"
