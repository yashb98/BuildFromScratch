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
