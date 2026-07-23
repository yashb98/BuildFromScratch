#!/usr/bin/env python3
"""research/loop_state.py — the atomic interface to loop_state.json, the
research-loop's resumable S0–S9 state (mirrors how ledger.py is the only
interface to ledger.json). Stdlib-only, CPU-only, no network.

Why this exists: the loop's resume / dead-run / auto-resume-cap behaviour was
specified in the orchestrator skill but never execution-verified. Putting the
state transitions in a small tested module turns "resume lands at S5 not S0",
"a corrupt state fails OPEN", and "auto_resumes is capped at 2" from prose into
the assertions in test_orchestration_chaos.py.

Design rules (all tested):
  - Atomic write: tempfile + os.replace in the same dir; prior version -> .bak.
  - DURABLE write (§C8 parity with ledger.save): fsync(file) -> mode-preserving
    chmod -> os.replace -> fsync(PARENT DIR). The parent-dir fsync is what makes
    the rename itself survive the documented GB10 hard-lock; without it the file
    contents are on disk but the directory entry can be lost on reboot, and the
    @reboot recovery chain (boot_resume.sh) loses the in-flight run.
  - LOCKED read-modify-write (§C8 parity with ledger.acquire_lock): every mutator
    holds an exclusive advisory flock on <loop_state.json>.lock, so the loop, the
    @reboot recovery hook and the liveness cron cannot lose-update each other.
  - FAIL-OPEN read: a missing OR corrupt state file returns a safe fresh default
    (flagged `_recovered`) and NEVER raises — a watchdog/liveness probe must keep
    running even if the state file is garbage.
  - resume_point() returns the STORED stage, so a killed run resumes where it
    died, not from S0.
  - auto-resume cap: at most MAX_AUTO_RESUMES automatic resumes; the next dead
    run is classified `crashed`, not resumed (prevents an infinite resume loop).
  - register() is the sanctioned setter for the recovery-critical flat in-flight
    fields, so callers stop open-coding their own load->mutate->save (which skips
    the .bak, the parent-dir fsync and the lock).
  - timestamps are caller-supplied (derived at runtime, §C2) so this module
    never reads a wall clock and stays deterministic/testable.

Exit codes (CLI): 0 success; 2 validation/usage error.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

SCHEMA_VERSION = 1
STAGES = tuple(f"S{i}" for i in range(10))  # S0..S9
MAX_AUTO_RESUMES = 2
LOCK_TIMEOUT_S = 10.0   # bounded wait for the advisory lock — see acquire_lock()


# Recovery-critical flat in-flight fields (§C5) — the ones /ablation-runner writes
# and the recovery chain reads to re-adopt a killed trainer. A fresh/recovered
# default carries them as None (= "no in-flight run"), so a fail-open read is
# schema-shaped like a live file instead of silently dropping these keys.
IN_FLIGHT_FIELDS = ("in_flight_run", "train_pid", "ckpt_path", "resume_cmd")


def default_state() -> dict:
    """The §C5 pinned bootstrap schema, verbatim from research-loop/SKILL.md
    (+ `updated`, which the writer sets on the first advance). Emitting the FULL
    live 12-key shape here is what closes the schema fork: a fail-open recovery
    (missing/corrupt file) now yields the same keys a live file has — in
    particular the recovery-critical `train_pid`/`ckpt_path`/`resume_cmd` are
    present as None rather than absent, so downstream recovery reads a clean
    "nothing to resume" instead of KeyError-ing on a divergent 7-key default."""
    return {"schema_version": SCHEMA_VERSION, "iteration_date": None,
            "stage": "S0", "in_flight_run": None, "train_pid": None,
            "ckpt_path": None, "resume_cmd": None, "auto_resumes": 0,
            "last_radar": None, "objective": "any", "notes": "",
            "updated": None}


def acquire_lock(path, timeout: float = LOCK_TIMEOUT_S):
    """§C8 cross-process safety, mirroring ledger.acquire_lock: an exclusive
    advisory lock on <loop_state.json>.lock, serializing the load->mutate->save
    sequence so the @reboot recovery (research/boot_resume.sh), the liveness
    cron and the loop itself cannot lose-update each other. Fail-open: if
    locking is unavailable, return None and proceed (a missing lock must never
    brick the recovery path).

    ONE deliberate difference from ledger.py: the wait is BOUNDED. ledger.py
    blocks forever on LOCK_EX, but this file sits on the once-per-boot recovery
    path whose callers impose no timeout of their own, so an unbounded block
    would silently forfeit the recovery. After `timeout` we proceed unlocked —
    a lost update is recoverable, a hung @reboot hook is not."""
    p = Path(path)
    try:
        fd = os.open(str(p.with_name(p.name + ".lock")), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return None
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            if time.monotonic() >= deadline:
                os.close(fd)
                return None
            time.sleep(0.02)


def release_lock(fd):
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


_LOCKS: dict[str, list] = {}   # realpath of the state file -> [depth, fd]


@contextlib.contextmanager
def locked(path, timeout: float = LOCK_TIMEOUT_S):
    """Hold the advisory lock across a whole load->mutate->save block. Every
    mutator below enters it, and an external caller may wrap several mutations
    in ONE critical section.

    Re-entrant within a process: flock is per open-file-description, so a nested
    os.open + LOCK_EX would deadlock against ourselves — nested entries just bump
    a depth counter and reuse the outermost fd."""
    key = os.path.realpath(str(path))
    slot = _LOCKS.get(key)
    if slot is None:
        slot = _LOCKS[key] = [0, acquire_lock(path, timeout)]
    slot[0] += 1
    try:
        yield
    finally:
        slot[0] -= 1
        if slot[0] <= 0:
            _LOCKS.pop(key, None)
            release_lock(slot[1])


def _fsync_parent_dir(p: Path) -> None:
    """§C8 durability: fsync the PARENT DIR so the os.replace is itself
    persisted. Without this, on the documented GB10 unified-memory hard-crash
    the new directory entry can be lost on reboot — the state file comes back
    absent or truncated even though its contents were fsynced, and the recovery
    chain loses the in-flight run it was supposed to re-adopt. Best-effort: a
    filesystem that refuses the directory fsync must never fail the write."""
    try:
        dfd = os.open(str(p.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def load(path) -> dict:
    """Fail-open: missing/corrupt -> fresh default flagged `_recovered=True`.
    Never raises. A valid file is returned as-is (with `_recovered=False`)."""
    p = Path(path)
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or "stage" not in data:
            raise ValueError("malformed")
        if data.get("stage") not in STAGES:
            raise ValueError("bad stage")
        data["_recovered"] = False
        return data
    except (OSError, json.JSONDecodeError, ValueError):
        d = default_state()
        d["_recovered"] = True
        return d


def save(path, state: dict) -> None:
    """Atomic + DURABLE write, plus the .bak snapshot; strips the transient
    `_recovered` flag. Same sequence as ledger.save (§C8): fsync the contents,
    preserve the file mode, os.replace, then fsync the parent directory."""
    if state.get("stage") not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {state.get('stage')!r}")
    p = Path(path)
    out = {k: v for k, v in state.items() if k != "_recovered"}
    out.setdefault("schema_version", SCHEMA_VERSION)
    if p.exists():
        shutil.copy2(p, p.with_name(p.name + ".bak"))
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".loopstate_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates the temp file 0600 and os.replace would keep that —
        # preserve the existing state file's mode (or 0644 on first creation),
        # or a root/cron-written state silently stops being readable by the
        # other sanctioned writers.
        os.chmod(tmp, (os.stat(p).st_mode & 0o777) if p.exists() else 0o644)
        os.replace(tmp, p)  # atomic on POSIX, same filesystem
        _fsync_parent_dir(p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def advance(path, stage: str, ts: str | None = None, in_flight=...) -> dict:
    """Move to `stage` and persist. `in_flight` updated only if passed
    (sentinel `...` means leave unchanged). All other live keys — including the
    recovery-critical `train_pid`/`ckpt_path`/`resume_cmd` — are preserved
    verbatim across the transition (load returns a valid file as-is)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    with locked(path):
        st = load(path)
        st["stage"] = stage
        if ts is not None:
            st["updated"] = ts
        if in_flight is not ...:
            st["in_flight_run"] = in_flight
        save(path, st)
    return st


def _check_run_id(v):
    if v is None or isinstance(v, str):
        return v
    raise ValueError(f"in_flight_run must be the RUN_ID string or None, "
                     f"never a nested object (§C5): got {type(v).__name__}")


def _check_pid(v):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"train_pid must be a positive int or None, got {v!r}")
    return v


def _check_str_or_path(name, v):
    if v is None:
        return None
    if isinstance(v, os.PathLike):
        return os.fspath(v)
    if not isinstance(v, str):
        raise ValueError(f"{name} must be a string or None, got {type(v).__name__}")
    return v


def _check_resumes(v):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"auto_resumes must be an int >= 0, got {v!r}")
    return v


def register(path, *, in_flight_run=..., train_pid=..., ckpt_path=...,
             resume_cmd=..., auto_resumes=..., ts: str | None = None) -> dict:
    """Set the recovery-critical FLAT in-flight fields (§C5) atomically, under
    the lock. This is the sanctioned writer for what /ablation-runner records at
    launch and what boot_resume.sh re-adopts after a reboot — it exists so those
    callers stop open-coding their own load->mutate->save of loop_state.json (an
    open-coded write skips the .bak snapshot, the parent-dir fsync and the lock,
    i.e. exactly the durability this module owns).

    Only the fields you pass are touched (the `...` sentinel means "leave
    unchanged"); pass None to CLEAR one — a finished run has no in-flight
    pid/ckpt. The stage is never moved: registering a trainer is not a state
    transition (use advance() for that)."""
    updates = {}
    if in_flight_run is not ...:
        updates["in_flight_run"] = _check_run_id(in_flight_run)
    if train_pid is not ...:
        updates["train_pid"] = _check_pid(train_pid)
    if ckpt_path is not ...:
        updates["ckpt_path"] = _check_str_or_path("ckpt_path", ckpt_path)
    if resume_cmd is not ...:
        updates["resume_cmd"] = _check_str_or_path("resume_cmd", resume_cmd)
    if auto_resumes is not ...:
        updates["auto_resumes"] = _check_resumes(auto_resumes)
    with locked(path):
        st = load(path)
        st.update(updates)
        if ts is not None:
            st["updated"] = ts
        save(path, st)
    return st


def resume_point(path) -> str:
    """The stage a resumed iteration should re-enter (the stored stage)."""
    return load(path)["stage"]


def record_resume(path, ts: str | None = None, cap: int = MAX_AUTO_RESUMES) -> dict:
    """Account one dead-run recovery attempt. While auto_resumes < cap, increment
    and decide `resume`; once the cap is reached, decide `crashed` (no further
    auto-resume) WITHOUT incrementing past the cap. The whole read-modify-write
    is locked: two recovery paths racing must not BOTH read `auto_resumes=1`
    and both decide `resume` (that is how a resume loop re-crashes the box)."""
    with locked(path):
        st = load(path)
        used = int(st.get("auto_resumes", 0))
        if used < cap:
            st["auto_resumes"] = used + 1
            decision = "resume"
        else:
            decision = "crashed"
        if ts is not None:
            st["updated"] = ts
        save(path, st)
    return {"decision": decision, "auto_resumes": st["auto_resumes"], "cap": cap}


def reset_resumes(path, ts: str | None = None) -> dict:
    """A clean completed iteration clears the resume counter for the next one."""
    with locked(path):
        st = load(path)
        st["auto_resumes"] = 0
        if ts is not None:
            st["updated"] = ts
        save(path, st)
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    a_ = sub.add_parser("advance"); a_.add_argument("stage"); a_.add_argument("--ts", default=None)
    sub.add_parser("resume-point")
    rr = sub.add_parser("record-resume"); rr.add_argument("--ts", default=None)
    # `register` is the shell-side entry point for the §C5 flat in-flight fields
    # (the sanctioned replacement for a caller's own load-mutate-save heredoc).
    rg = sub.add_parser("register", help="set the flat in-flight recovery fields (§C5)")
    rg.add_argument("--in-flight-run", default=None)
    rg.add_argument("--train-pid", type=int, default=None)
    rg.add_argument("--ckpt-path", default=None)
    rg.add_argument("--resume-cmd", default=None)
    rg.add_argument("--auto-resumes", type=int, default=None)
    rg.add_argument("--ts", default=None)
    rg.add_argument("--clear", action="append", default=[],
                    choices=sorted(IN_FLIGHT_FIELDS),
                    help="explicitly null a field (a finished run clears these); "
                         "repeatable. Omitted fields are left unchanged. "
                         "(auto_resumes is a counter — zero it with --auto-resumes 0.)")
    a = ap.parse_args(argv)
    if a.cmd == "show":
        print(json.dumps(load(a.path), indent=2))
    elif a.cmd == "advance":
        print(json.dumps(advance(a.path, a.stage, a.ts), indent=2))
    elif a.cmd == "resume-point":
        print(resume_point(a.path))
    elif a.cmd == "record-resume":
        print(json.dumps(record_resume(a.path, a.ts), indent=2))
    elif a.cmd == "register":
        given = {"in_flight_run": a.in_flight_run, "train_pid": a.train_pid,
                 "ckpt_path": a.ckpt_path, "resume_cmd": a.resume_cmd,
                 "auto_resumes": a.auto_resumes}
        kw = {k: v for k, v in given.items() if v is not None}
        for f in a.clear:
            if f in kw:
                print(f"error: --clear {f} conflicts with the value given for it",
                      file=sys.stderr)
                return 2
            kw[f] = None
        if not kw:
            print("error: register needs at least one field to set or --clear",
                  file=sys.stderr)
            return 2
        try:
            st = register(a.path, ts=a.ts, **kw)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps({k: st.get(k) for k in
                          IN_FLIGHT_FIELDS + ("auto_resumes", "updated")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
