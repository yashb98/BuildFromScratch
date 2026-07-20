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
  - FAIL-OPEN read: a missing OR corrupt state file returns a safe fresh default
    (flagged `_recovered`) and NEVER raises — a watchdog/liveness probe must keep
    running even if the state file is garbage.
  - resume_point() returns the STORED stage, so a killed run resumes where it
    died, not from S0.
  - auto-resume cap: at most MAX_AUTO_RESUMES automatic resumes; the next dead
    run is classified `crashed`, not resumed (prevents an infinite resume loop).
  - timestamps are caller-supplied (derived at runtime, §C2) so this module
    never reads a wall clock and stays deterministic/testable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
STAGES = tuple(f"S{i}" for i in range(10))  # S0..S9
MAX_AUTO_RESUMES = 2


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
    """Atomic write + .bak snapshot. Strips the transient `_recovered` flag."""
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
        os.replace(tmp, p)  # atomic on POSIX, same filesystem
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
    st = load(path)
    st["stage"] = stage
    if ts is not None:
        st["updated"] = ts
    if in_flight is not ...:
        st["in_flight_run"] = in_flight
    save(path, st)
    return st


def resume_point(path) -> str:
    """The stage a resumed iteration should re-enter (the stored stage)."""
    return load(path)["stage"]


def record_resume(path, ts: str | None = None, cap: int = MAX_AUTO_RESUMES) -> dict:
    """Account one dead-run recovery attempt. While auto_resumes < cap, increment
    and decide `resume`; once the cap is reached, decide `crashed` (no further
    auto-resume) WITHOUT incrementing past the cap."""
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
    a = ap.parse_args(argv)
    if a.cmd == "show":
        print(json.dumps(load(a.path), indent=2))
    elif a.cmd == "advance":
        print(json.dumps(advance(a.path, a.stage, a.ts), indent=2))
    elif a.cmd == "resume-point":
        print(resume_point(a.path))
    elif a.cmd == "record-resume":
        print(json.dumps(record_resume(a.path, a.ts), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
