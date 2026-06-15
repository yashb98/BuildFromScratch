#!/usr/bin/env python3
"""sentinel.py — resource watchdog for the GB10 unified-memory box.

Lives at the BuildFromScratch repo root, next to safe_cuda.py. STDLIB-ONLY and
CPU-only by design: it must run while the box is under memory pressure, which
is exactly when a torch import (which itself allocates) would fail or make
things worse. Never add torch/psutil/numpy imports here.

Contract home: .claude/skills/research-loop/references/contracts.md (§C6).

Modes
-----
preflight
    Exit 0 if the box is healthy enough to start GPU work; exit 1 with EVERY
    failing reason on stdout otherwise (§C5.1, §C4.5, §C4.6). Checks:
      * MemAvailable/MemTotal >= 30%            (/proc/meminfo)
      * no other trainer running                (pgrep -af + nvidia-smi
                                                 --query-compute-apps)
      * >= 100 GB free on /home/yashb98         (shutil.disk_usage)
      * 1-min loadavg <= 2x cores               (os.getloadavg)

watch --pid P [--kill-at 0.80] [--log FILE] [--interval 30] [--grace 60]
      [--marker FILE]
    Kill-switch daemon armed alongside every unattended training run (§C5.6,
    §C6). Pins P's /proc start time at arm so a recycled PID is never
    signaled. Every --interval seconds (default 30) it samples pool usage
    (/proc/meminfo) AND P's RSS (/proc/P/status VmRSS — §C6 "free + process
    RSS"):
      * P dead (or zombie, or PID recycled) -> log it, exit 0 — the normal
        end of a run.
      * pool usage (MemTotal-MemAvailable)/MemTotal >= --kill-at ->
        re-verify P's identity, SIGTERM P, wait up to --grace s (default 60),
        SIGKILL if still alive, write the reason + trainer RSS to the log AND
        (atomically, tmp+rename) to the marker file
        research/loop_state.json.sentinel_kill, exit 3. --marker overrides
        the marker path for SELF-TESTS ONLY; production arms never pass it.

Why --kill-at defaults to 0.80
------------------------------
safe_cuda.guard(0.85) caps ONE torch process at 85% of the unified pool, but
the OS, dataloader workers, the editor and everything else draw from the SAME
~119 GB pool — total pressure can hit the kernel's global OOM-killer cliff
before any single process reaches its 0.85 cap (it has taken down the desktop
on this box — contracts §C1). Killing the trainer at 80% of *pool* usage keeps the box below
the cliff that the per-process 0.85 guard cannot see. 0.80 < 0.85, always.

liveness [--state FILE]
    CPU-only, ~instant in-flight liveness probe for the 30-min cron (§C6
    observing-time). Reads research/loop_state.json: exit 0 if in_flight_run
    is null (nothing to do) or its train_pid is alive (running fine); exit 4
    with a reason if the in-flight run's PID is dead. The cron wrapper invokes
    "/research-loop resume" ONLY on exit 4 — so a silently-dead run is caught
    in <=30 min instead of waiting for the next nightly fire, and a healthy
    run starts no Claude session.

Exit codes
----------
0  preflight healthy / watch: watched pid exited on its own (clean disarm) /
   liveness: nothing to recover (idle or running)
1  preflight unhealthy (reasons on stdout)
2  argparse usage error (argparse default)
3  watch: sentinel killed the trainer (marker written)
4  liveness: in-flight run's PID is dead -> recovery needed
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MARKER = REPO_ROOT / "research" / "loop_state.json.sentinel_kill"  # §C6
LOOP_STATE = REPO_ROOT / "research" / "loop_state.json"  # §C6 liveness

# Thresholds (§C5.1, §C4.6, §C6). Change here, not at call sites.
MEM_AVAIL_FLOOR = 0.30   # preflight: MemAvailable must be >= 30% of pool
DISK_FLOOR_GB = 100      # §C4.6: never go below 100 GB free
DISK_PATH = "/home/yashb98"
LOAD_FACTOR = 2.0        # preflight: loadavg(1m) must be <= 2x cores
DEFAULT_KILL_AT = 0.80   # watch: must stay < safe_cuda's 0.85 (see docstring)
DEFAULT_INTERVAL = 30.0  # watch: sample period, seconds
DEFAULT_GRACE = 60.0     # watch: SIGTERM -> SIGKILL grace, seconds
TRAINER_PATTERN = "train"  # §C5.1: pgrep -af train


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def meminfo():
    """Return (total_kib, available_kib) from /proc/meminfo."""
    fields = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            fields[key.strip()] = int(rest.strip().split()[0])  # kB (KiB)
    return fields["MemTotal"], fields["MemAvailable"]


def find_trainers(notes):
    """Evidence lines for running trainers; appends degraded-check notes."""
    hits = []
    me = os.getpid()
    try:
        out = subprocess.run(
            ["pgrep", "-af", TRAINER_PATTERN],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
        notes.append("pgrep unavailable — process-name check skipped")
    for line in out.splitlines():
        pid_s, _, cmd = line.strip().partition(" ")
        if not pid_s.isdigit() or int(pid_s) == me:
            continue
        if "sentinel.py" in cmd or "pgrep" in cmd:
            continue
        # Count only things that look like a training job, not e.g. a
        # browser process that happens to contain 'train' in an argument.
        if any(t in cmd for t in ("python", "torchrun", "accelerate")):
            hits.append(f"pgrep: {pid_s} {cmd}")
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if line:  # on GB10 used_memory may print [N/A]; pid still works
                hits.append(f"nvidia-smi compute app: {line}")
    except (OSError, subprocess.TimeoutExpired):
        notes.append("nvidia-smi unavailable — GPU compute-app check skipped")
    return hits


def preflight() -> int:
    reasons, notes = [], []
    try:
        total_kib, avail_kib = meminfo()
        frac = avail_kib / total_kib
        if frac < MEM_AVAIL_FLOOR:
            reasons.append(
                f"low memory: MemAvailable {avail_kib / 2**20:.1f} GiB is "
                f"{frac:.0%} of pool (floor {MEM_AVAIL_FLOOR:.0%})"
            )
    except (OSError, KeyError, ValueError) as e:  # fail CLOSED
        frac = float("nan")
        reasons.append(f"cannot read /proc/meminfo ({e}) — failing closed")
    trainers = find_trainers(notes)
    if trainers:
        reasons.append(
            "trainer already running (§C4.5: one trainer at a time):\n    "
            + "\n    ".join(trainers)
        )
    free_gb = shutil.disk_usage(DISK_PATH).free / 1e9
    if free_gb < DISK_FLOOR_GB:
        reasons.append(
            f"disk: {free_gb:.0f} GB free on {DISK_PATH} "
            f"< {DISK_FLOOR_GB} GB floor (§C4.6)"
        )
    load1 = os.getloadavg()[0]
    cores = os.cpu_count() or 1
    if load1 > LOAD_FACTOR * cores:
        reasons.append(
            f"load: 1-min loadavg {load1:.1f} > {LOAD_FACTOR:g}x {cores} cores"
        )
    for n in notes:
        print(f"[sentinel] note: {n}")
    if reasons:
        print("PREFLIGHT FAIL")
        for r in reasons:
            print(f"- {r}")
        return 1
    print(
        f"PREFLIGHT OK mem_available={frac:.0%} disk_free={free_gb:.0f}GB "
        f"load1={load1:.2f} cores={cores} trainers=none"
    )
    return 0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by someone else
    try:  # a zombie cannot allocate; treat as dead
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def proc_start_time(pid: int):
    """Process start time (clock ticks; /proc/<pid>/stat field 22) or None.

    Pinned at arm time and re-checked before any signal: if the trainer died
    and the kernel recycled the PID, the start time differs and the sentinel
    disarms instead of killing an unrelated process (SKILL hard rule 8).
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            # Fields after the ')' that ends comm: index 0 is field 3 (state),
            # so field 22 (starttime) is index 19.
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def watched_alive(pid: int, start_ticks) -> bool:
    """Alive, not a zombie, AND same start time as when armed (PID-reuse guard)."""
    return pid_alive(pid) and proc_start_time(pid) == start_ticks


def pid_rss_gib(pid: int):
    """Watched process RSS in GiB (/proc/<pid>/status VmRSS), or None.

    §C6: the watch daemon samples free + process RSS every 30 s, so the kill
    log and marker can attribute pool pressure to the trainer vs everything
    else drawing on the shared pool.
    """
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20  # kB (KiB) -> GiB
    except (OSError, ValueError, IndexError):
        pass
    return None  # gone, or a zombie (zombies have no VmRSS line)


def watch(pid, kill_at, log_path, interval, grace, marker_path=None) -> int:
    marker = Path(marker_path) if marker_path else MARKER
    logf = open(log_path, "a", buffering=1) if log_path else None

    def log(msg):
        line = f"[{utcnow()}] [sentinel] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")

    start_ticks = proc_start_time(pid)  # identity pin (PID-reuse guard)
    log(
        f"armed: pid={pid} kill_at={kill_at:.2f} interval={interval:g}s "
        f"grace={grace:g}s start_ticks={start_ticks} marker={marker}"
    )
    samples = 0
    while True:
        if not watched_alive(pid, start_ticks):
            log(f"watched pid {pid} exited on its own; disarming (no kill)")
            return 0
        try:
            total_kib, avail_kib = meminfo()
        except (OSError, KeyError, ValueError) as e:
            log(f"WARN: meminfo read failed ({e}); keeping watch")
            time.sleep(interval)
            continue
        usage = (total_kib - avail_kib) / total_kib
        rss_gib = pid_rss_gib(pid)  # §C6: free + process RSS every sample
        rss_s = f"{rss_gib:.1f} GiB" if rss_gib is not None else "n/a"
        if usage >= kill_at:
            reason = (
                f"pool usage {usage:.1%} >= kill-at {kill_at:.0%} "
                f"(MemAvailable {avail_kib / 2**20:.1f} GiB of "
                f"{total_kib / 2**20:.1f} GiB; trainer rss {rss_s})"
            )
            if not watched_alive(pid, start_ticks):  # re-verify identity
                log(f"watched pid {pid} exited on its own; disarming (no kill)")
                return 0
            log(f"KILL: {reason} -> SIGTERM {pid}")
            kill_failed = None
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError as e:
                kill_failed = f"SIGTERM denied: {e}"
                log(f"ERROR: {kill_failed}")
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and watched_alive(pid, start_ticks):
                time.sleep(1)
            if watched_alive(pid, start_ticks):
                log(f"pid {pid} survived {grace:g}s grace -> SIGKILL")
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as e:
                    kill_failed = f"SIGKILL denied: {e}"
                    log(f"ERROR: {kill_failed}")
            else:
                log(f"pid {pid} exited within grace period")
            payload = {
                "time": utcnow(),
                "killed_pid": pid,
                "reason": reason,
                "pool_usage": round(usage, 4),
                "pool_total_gb": round(total_kib / 2**20, 1),
                "trainer_rss_gb": (
                    round(rss_gib, 2) if rss_gib is not None else None
                ),
                "kill_at": kill_at,
                "kill_failed": kill_failed,  # null on the normal path
                "log": str(log_path) if log_path else None,
            }
            marker.parent.mkdir(parents=True, exist_ok=True)
            tmp = marker.with_name(marker.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(tmp, marker)  # atomic: Phase 3 never sees torn JSON
            log(f"marker written: {marker}")
            return 3
        samples += 1
        if samples % 20 == 0:  # heartbeat every ~10 min at default interval
            log(f"heartbeat: pool usage {usage:.1%}, pid {pid} alive, rss {rss_s}")
        time.sleep(interval)


def liveness(state_path=None) -> int:
    """In-flight liveness probe for the 30-min cron (§C6 observing-time).

    Exit 0 = nothing to recover (no in-flight run, or its PID is alive).
    Exit 4 = the in-flight run's PID is dead -> the wrapper triggers resume.
    Fails OPEN (exit 0) on a missing/unreadable state file: a transient read
    glitch must not spam '/research-loop resume'; the nightly fire still runs.
    """
    sp = Path(state_path) if state_path else LOOP_STATE
    try:
        state = json.loads(sp.read_text())
    except FileNotFoundError:
        print("[sentinel] liveness: no loop_state.json — nothing to do")
        return 0
    except (OSError, ValueError) as e:
        print(f"[sentinel] liveness: cannot read state ({e}) — failing open")
        return 0
    run = state.get("in_flight_run")
    if not run:
        print("[sentinel] liveness: no in-flight run — idle")
        return 0
    pid = state.get("train_pid")
    if isinstance(pid, int) and pid_alive(pid):
        print(f"[sentinel] liveness: in-flight {run} pid {pid} alive — ok")
        return 0
    print(f"[sentinel] liveness: in-flight {run} pid {pid} is DEAD — "
          f"recovery needed (exit 4)")
    return 4


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="sentinel.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("preflight", help="health gate before any GPU work")
    lv = sub.add_parser("liveness", help="in-flight PID probe for the 30-min cron")
    lv.add_argument("--state", default=None,
                    help="override loop_state.json path (self-test hook)")
    w = sub.add_parser("watch", help="kill-switch daemon beside a trainer")
    w.add_argument("--pid", type=int, required=True,
                   help="trainer PID to watch")
    w.add_argument("--kill-at", type=float, default=DEFAULT_KILL_AT,
                   help=f"pool-usage kill fraction (default "
                        f"{DEFAULT_KILL_AT}; must be < 0.85)")
    w.add_argument("--log", default=None, help="append timestamped log here")
    w.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                   help="sample period seconds (default 30; <30 is for tests)")
    w.add_argument("--grace", type=float, default=DEFAULT_GRACE,
                   help="SIGTERM->SIGKILL grace seconds (default 60)")
    w.add_argument("--marker", default=None,
                   help="override kill-marker path (SELF-TEST hook only; "
                        "production arms use the default §C6 marker)")
    args = p.parse_args(argv)
    if args.mode == "preflight":
        return preflight()
    if args.mode == "liveness":
        return liveness(args.state)
    if not (0.0 < args.kill_at < 0.85):
        p.error(f"--kill-at {args.kill_at} must be in (0, 0.85): it has to "
                f"fire BELOW the safe_cuda 0.85 per-process guard (§C6)")
    return watch(args.pid, args.kill_at, args.log, args.interval, args.grace,
                 args.marker)


if __name__ == "__main__":
    sys.exit(main())
