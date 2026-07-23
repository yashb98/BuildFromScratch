#!/usr/bin/env python3
"""research/adopt_run.py — the ADOPTED-RUN protocol (upgrade-plan #10).

The audit's uncomfortable finding: the out-of-loop launch is the DOMINANT mode on
this box (~24 hand-driven runs vs ~2 through /ablation-runner). The live arch-ladder
is the pattern in full — a hand-written `c5_evidence.json`, a hand-added ledger
entry, a raw `setsid nohup ... run_arch_ladder.sh`, and `loop_state.json` mutated by
an inline python snippet. The contracts had NO provision for any of that, so every
such launch was contract-VIOLATING rather than contract-GOVERNED. Pretending it does
not happen is how the record rots; this module makes the path legal and mechanically
complete instead.

Two verbs, both DRY-RUN BY DEFAULT (nothing mutates without `--apply`):

  adopt <run-dir>
      Make a manual launch indistinguishable from a sanctioned one:
        (a) §C5 lint  — validate <run-dir>/c5_evidence.json by importing
            c5_validate.validate_c5(); a lint failure REFUSES the adoption. No
            evidence, no adoption: that is the whole point of the protocol.
        (b) ledger    — create the run entry through research/ledger/ledger.py's
            CLI (never by hand). Idempotent: an existing entry is left untouched.
        (c) digest    — write a stub under research/digests/ so §C9's "a digest is
            always written" cannot be silently skipped by an out-of-loop launch.
        (d) loop_state— register in_flight_run / train_pid / ckpt_path / resume_cmd
            through loop_state.register() (the sanctioned setter), so the recovery
            chain (boot_resume.sh, liveness cron) can re-adopt the trainer.

  reconcile
      The other half, and the more valuable one: an in-flight run whose train_pid is
      no longer alive. It (i) reports the stale pointer and (ii) with `--apply`
      transitions the ledger entry to a terminal STATUS and clears the loop_state
      pointer. It NEVER invents a verdict — dying is not a result, and the terminal
      status itself is read off disk (a `verdict.json` in the artifacts dir means the
      run finished -> `done`; its absence means it did not -> `crashed`).

Exit codes:
  0  success / nothing to do (clean or alive)
  1  refused (§C5 lint failure, unreadable evidence, identity mismatch, dead pid at
     adopt) or a step failed
  2  argparse usage error (bad flags) — every other refusal is 1
  4  reconcile only, dry run: a STALE in-flight run was detected and needs `--apply`
     (mirrors `sentinel.py liveness` exit 4 = run dead, so a cron can key on it)

Stdlib-only, pure CPU, no network, no GPU, never launches or resumes a trainer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEDGER_PY = HERE / "ledger" / "ledger.py"
DEFAULT_LEDGER = HERE / "ledger" / "ledger.json"
DEFAULT_STATE = HERE / "loop_state.json"
DEFAULT_DIGESTS = HERE / "digests"
PROTOCOL_VERSION = "adopt-run/v1"

# A launch is not a result: the ledger STATUS moves, the verdict never does.
TERMINAL_IF_FINISHED = "done"      # artifacts carry a verdict.json -> the run finished
TERMINAL_IF_DIED = "crashed"       # no verdict.json and the pid is gone -> it died
IN_FLIGHT_STATUSES = ("launched", "running", "resumed")

# §C13 objective -> §C8 run type. A run's objective is the thing the c5 evidence
# records; `type` is the ledger's own vocabulary, so map rather than guess.
OBJECTIVE_TO_RUN_TYPE = {"pretrain-ablation": "ablation", "finetune": "finetune"}


def _load_module(name: str, path: Path):
    """Import a sibling research/ module by ABSOLUTE PATH (cwd-independent, so the
    headless loop and a cron can call this from anywhere). Registered in
    sys.modules under its real name so a caller that already did `import
    loop_state` shares ONE module object — two copies would mean two lock
    registries flocking the same file against each other."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


c5_validate = _load_module("c5_validate", HERE / "c5_validate.py")
loop_state = _load_module("loop_state", HERE / "loop_state.py")


def _ledger_vocabulary():
    """RUN_TYPES/RUN_STATUS straight from ledger.py (§C8 is the authority), loaded
    by path under a private name and deliberately NOT registered in sys.modules —
    same discipline as research/eval_completeness.py, so it cannot shadow the plain
    `import ledger` used by the ledger's own tests."""
    spec = importlib.util.spec_from_file_location("_adopt_ledger_vocab", LEDGER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # ledger.py's CLI is under __main__: no side effects
    return frozenset(mod.RUN_TYPES), frozenset(mod.RUN_STATUS)


RUN_TYPES, RUN_STATUS = _ledger_vocabulary()


# ---------------------------------------------------------------- small helpers

def _now_iso() -> str:
    """Local wall-clock stamp, runtime-derived (§C2). loop_state never reads a
    clock itself — the caller supplies `ts`, and this is that caller."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _today() -> str:
    return date.today().isoformat()


def pid_alive(pid: int) -> bool:
    """Alive and not a zombie. Same test as sentinel.pid_alive (a zombie cannot
    allocate, so it is dead for our purposes); duplicated rather than imported so
    this module stays importable without the repo root on sys.path."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by someone else
    except (OSError, TypeError, ValueError):
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _result_of(value):
    """§C5 items are recorded either as a bare token (`"pass"`) or as an object
    (`{"result": "pass", "detail": ...}`). Return the token in both stylings."""
    if isinstance(value, dict):
        return value.get("result")
    return value


def _rel(path: Path) -> str:
    """Repo-relative path when the target is inside the repo, else absolute — the
    ledger stores repo-relative artifact paths."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _ledger_cli(ledger_path: Path, args: list) -> tuple:
    """Run one research/ledger/ledger.py subcommand. Returns (rc, stdout, stderr).
    The ledger CLI is the ONLY sanctioned writer (§C11) — this module never opens
    ledger.json itself, for reading or writing."""
    proc = subprocess.run([sys.executable, str(LEDGER_PY)] + args
                          + ["--ledger", str(ledger_path)],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def ledger_run_entry(ledger_path: Path, run_id: str):
    """The run entry for `run_id`, or None. Read through the CLI's `query` verb so
    even the READ goes through the sanctioned interface."""
    rc, out, err = _ledger_cli(ledger_path, ["query", "--collection", "runs"])
    if rc != 0:
        raise RuntimeError(f"ledger query failed (exit {rc}): {(err or out).strip()}")
    try:
        runs = json.loads(out).get("runs", [])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ledger query returned non-JSON: {e}")
    return next((r for r in runs if r.get("run_id") == run_id), None)


# ---------------------------------------------------------------- digest stub

def digest_stub_path(digest_dir: Path, run_id: str, started: str) -> Path:
    """`<date>_adopted_<run_id>.md`, deliberately NOT `<date>.md`: the loop's own S9
    digest owns that name, and an adopted run must never clobber (or race) it. The
    stub still globs into research/digests/*.md, which is how digests are found."""
    return Path(digest_dir) / f"{started}_adopted_{run_id}.md"


def digest_stub_text(*, run_id, evidence, c5_report, ledger_action, state_action,
                     train_pid, evidence_path, artifacts_dir, ts) -> str:
    """The §C9 acceptance artifact for an out-of-loop launch. It records that the
    run EXISTS and is governed — and says, in words, that it holds no result."""
    ev = evidence
    satisfied = ", ".join(sorted(c5_report["satisfied"]))
    lines = [
        f"# Adopted-run digest stub — {run_id}",
        "",
        f"> **Out-of-loop launch, adopted {ts} by `{PROTOCOL_VERSION}`.** This run was",
        "> started by hand — not by `/research-loop` and not by `/ablation-runner`.",
        "> `research/adopt_run.py adopt` validated its §C5 evidence, recorded the ledger",
        "> entry through `ledger.py`, and registered it in `loop_state.json` through",
        "> `loop_state.register()`. This stub exists so §C9's \"a digest is always",
        "> written\" cannot be skipped by an out-of-loop launch.",
        "",
        "## What was adopted",
        f"- **run_id:** `{run_id}`",
        f"- **model_dir:** `{ev.get('model_dir')}`  ·  **technique:** `{ev.get('technique_slug')}`",
        f"- **objective:** `{ev.get('objective')}`  ·  **framework:** `{ev.get('framework')}`"
        f"  ·  **lifecycle_stage:** `{ev.get('lifecycle_stage')}`",
        f"- **artifacts:** `{artifacts_dir}`",
        f"- **§C5 evidence:** `{evidence_path}` — lint **PASS** ({len(c5_report['satisfied'])}/7: {satisfied})",
        f"- **ledger:** {ledger_action}  ·  **loop_state:** {state_action}"
        + (f"  ·  **train_pid:** {train_pid}" if train_pid else ""),
        "",
        "## Results",
        "_None._ A launch is not a result. This stub records only that the run exists and",
        "is governed; no number, no verdict, and no claim is made here. The verdict is",
        "written when the run finishes and `/eval-harness` scores it (§C10), subject to",
        "the §C25 per-stage battery (a missing HARD item caps it below `win`).",
        "",
        "## To close this run",
        "1. Score the finished artifacts with `/eval-harness` (the only source of",
        "   cross-run comparable numbers, §C10) and stamp `suite_version`.",
        "2. Write `verdict.json` in the artifacts dir, then record `metrics`/`verdict`",
        "   on the ledger entry via `ledger.py update-run`.",
        "3. If the trainer dies instead, run `python3 research/adopt_run.py reconcile`",
        "   — it moves the entry to a terminal status and clears the in-flight pointer",
        "   WITHOUT inventing a verdict.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- verb: adopt

def adopt(run_dir, *, ledger_path=DEFAULT_LEDGER, state_path=DEFAULT_STATE,
          digest_dir=DEFAULT_DIGESTS, train_pid=None, resume_cmd=None,
          ckpt_path=None, run_type=None, ts=None, apply=False,
          force=False) -> dict:
    """Adopt a manual launch. Every precondition is checked BEFORE any mutation, so
    a refusal leaves the ledger, the digests and loop_state untouched."""
    ts = ts or _now_iso()
    run_dir = Path(run_dir)
    report = {"verb": "adopt", "ok": False, "dry_run": not apply, "ts": ts,
              "run_dir": str(run_dir), "run_id": None, "c5": None,
              "actions": [], "errors": []}

    def err(msg):
        report["errors"].append(msg)
        return report

    # (a) §C5 lint — no evidence, no adoption.
    if not run_dir.is_dir():
        return err(f"run dir not found: {run_dir}")
    evidence_file = run_dir / "c5_evidence.json"
    try:
        evidence = json.loads(evidence_file.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return err(f"cannot read {evidence_file}: {e} — §C5 evidence must exist "
                   "BEFORE launch, and an adopted run is held to the same bar")
    c5_report = c5_validate.validate_c5(evidence)
    report["c5"] = c5_report
    if not c5_report["ok"]:
        return err("§C5 lint FAILED — refusing to adopt. missing_items="
                   f"{c5_report['missing_items']} missing_identity="
                   f"{c5_report['missing_identity']}")
    sat = c5_report["satisfied"]

    # Identity. The RUN_ID is the run dir's name by convention (§C4.4); a mismatch
    # means the evidence belongs to a different run and the entry would point at
    # the wrong artifacts.
    run_id = evidence["run_id"]
    report["run_id"] = run_id
    if run_dir.name != run_id and not force:
        return err(f"run dir name {run_dir.name!r} != c5 run_id {run_id!r} — refusing "
                   "(the ledger entry would point at the wrong artifacts); --force to override")

    objective = evidence.get("objective")
    run_type = run_type or OBJECTIVE_TO_RUN_TYPE.get(objective)
    if run_type not in RUN_TYPES:
        return err(f"cannot determine a valid ledger run type: objective={objective!r} "
                   f"-> {run_type!r}; pass --run-type (valid: {sorted(RUN_TYPES)})")

    # A RUN_ID starts with the launch date (§C8 YYYY-MM-DD_<model>_<slug>), so the
    # start date is READ off the id rather than defaulted to today — the bug that
    # once gave a run started(2026-07-13) AFTER ended(2026-07-12).
    started = run_id[:10]
    model_dir = evidence.get("model_dir")
    if not model_dir and run_dir.parent.name == "experiments":
        model_dir = run_dir.parent.parent.name

    # A pid is only worth registering if it is ALIVE. Registering a dead pid as the
    # in-flight run is precisely the stale pointer `reconcile` exists to clean up.
    if train_pid is not None and not pid_alive(train_pid):
        return err(f"train_pid {train_pid} is not alive — refusing to register a dead "
                   "trainer as in-flight. Adopt without --train-pid to record a finished "
                   "run, or run `adopt_run.py reconcile` if this pid was the in-flight one")

    # loop_state pre-check: never strand a DIFFERENT in-flight run.
    st = loop_state.load(state_path)
    current = st.get("in_flight_run")
    if train_pid is not None and current not in (None, run_id) and not force:
        return err(f"loop_state already points at in_flight_run={current!r} — refusing to "
                   "overwrite it (that run would be stranded). Reconcile or finish it "
                   "first; --force to override")

    # ---- everything validated; from here on, mutate (or describe the mutation) ----
    evidence_path = _rel(evidence_file)
    artifacts_dir = _rel(run_dir)
    smoke_pass = _result_of(evidence[sat["c5.0_smoke"]]) == "pass"
    adopted_block = {"protocol": PROTOCOL_VERSION, "adopted_on": ts,
                     "launched_out_of_loop": True,
                     "c5_satisfied": sorted(sat), "evidence_path": evidence_path,
                     "train_pid": train_pid}

    sets = ["--set", f"budget={json.dumps(evidence[sat['c5.2_budget']])}",
            "--set", f"probe={json.dumps(evidence[sat['c5.3_probe']])}",
            "--set", f"eta_hours={json.dumps(evidence[sat['c5.4_eta']])}",
            "--set", f"evidence_path={evidence_path}",
            "--set", f"adopted={json.dumps(adopted_block)}"]
    for key in ("lifecycle_stage", "confound_check"):
        if evidence.get(key) not in (None, "", {}, []):
            sets += ["--set", f"{key}={json.dumps(evidence[key])}"]
    if train_pid is not None:
        sets += ["--set", "status=running"]   # a live trainer IS running (§C8 vocabulary)

    argv = ["add-run", "--run-id", run_id, "--type", run_type,
            "--started", started]
    if model_dir:
        argv += ["--model-dir", model_dir]
    if evidence.get("technique_slug"):
        argv += ["--technique-slug", evidence["technique_slug"]]
    if objective:
        argv += ["--objective", objective]
    if evidence.get("framework"):
        argv += ["--framework", evidence["framework"]]
    if smoke_pass:
        argv += ["--smoke", "pass"]
    argv += ["--artifacts-dir", artifacts_dir] + sets

    # (b) ledger entry — idempotent: an existing entry is NEVER clobbered.
    try:
        existing = ledger_run_entry(ledger_path, run_id)
    except RuntimeError as e:
        return err(str(e))
    if existing is not None:
        ledger_action = f"exists (status={existing.get('status')}) — left untouched"
        report["actions"].append({"step": "ledger", "action": "exists",
                                  "status": existing.get("status")})
    elif not apply:
        ledger_action = "would add-run"
        report["actions"].append({"step": "ledger", "action": "would-add-run",
                                  "argv": argv})
    else:
        rc, out, cli_err = _ledger_cli(ledger_path, argv)
        if rc != 0:
            report["actions"].append({"step": "ledger", "action": "FAILED",
                                      "exit": rc, "stderr": cli_err.strip()})
            return err(f"ledger add-run FAILED (exit {rc}): {(cli_err or out).strip()} "
                       "— NOT registering loop_state; the pointer must never name a run "
                       "the ledger does not know")
        ledger_action = "add-run"
        report["actions"].append({"step": "ledger", "action": "add-run", "argv": argv})

    # (c) digest stub — §C9 is not skippable by launching out of loop.
    stub = digest_stub_path(digest_dir, run_id, started)
    state_action = ("register in_flight_run/train_pid" if train_pid is not None
                    else "not registered (no live train_pid given)")
    if stub.exists():
        report["actions"].append({"step": "digest", "action": "exists",
                                  "path": _rel(stub)})
    else:
        text = digest_stub_text(run_id=run_id, evidence=evidence, c5_report=c5_report,
                                ledger_action=ledger_action, state_action=state_action,
                                train_pid=train_pid, evidence_path=evidence_path,
                                artifacts_dir=artifacts_dir, ts=ts)
        if apply:
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.write_text(text)
            report["actions"].append({"step": "digest", "action": "wrote",
                                      "path": _rel(stub)})
        else:
            report["actions"].append({"step": "digest", "action": "would-write",
                                      "path": _rel(stub), "bytes": len(text)})

    # (d) loop_state — through register(), the sanctioned setter. Never a direct write.
    if train_pid is None:
        report["actions"].append({"step": "loop_state", "action": "skipped",
                                  "why": "no --train-pid: nothing is in flight"})
    elif not apply:
        report["actions"].append({"step": "loop_state", "action": "would-register",
                                  "in_flight_run": run_id, "train_pid": train_pid,
                                  "ckpt_path": ckpt_path, "resume_cmd": resume_cmd})
    else:
        kw = {"in_flight_run": run_id, "train_pid": train_pid}
        if ckpt_path is not None:
            kw["ckpt_path"] = ckpt_path
        if resume_cmd is not None:
            kw["resume_cmd"] = resume_cmd
        try:
            loop_state.register(state_path, ts=ts, **kw)
        except (ValueError, OSError) as e:
            report["actions"].append({"step": "loop_state", "action": "FAILED",
                                      "error": str(e)})
            return err(f"loop_state.register failed: {e}")
        report["actions"].append({"step": "loop_state", "action": "registered", **kw})

    report["ok"] = True
    return report


# ------------------------------------------------------------ verb: reconcile

def reconcile(*, ledger_path=DEFAULT_LEDGER, state_path=DEFAULT_STATE,
              clear_resume=False, ts=None, today=None, apply=False) -> dict:
    """Detect an in-flight run whose trainer is gone, and (with --apply) close the
    transaction: terminal STATUS on the ledger entry + a cleared loop_state pointer.

    `state` is one of:
      clean          — no in_flight_run; nothing to do.
      alive          — the trainer is still running; nothing to do (never touch it).
      unverifiable   — an in_flight_run with NO train_pid: death cannot be proven, so
                       nothing is mutated. Fail closed (§C6) — a missing signal is not
                       evidence of death.
      stale          — the pid is gone. Report, and with --apply, reconcile.
      stale-unrecorded — stale AND the ledger has no entry for it (the out-of-loop
                       pathology in its purest form): the pointer is cleared, but no
                       ledger entry is invented. `adopt` is what creates entries.
    """
    ts = ts or _now_iso()
    today = today or _today()
    report = {"verb": "reconcile", "ok": True, "dry_run": not apply, "ts": ts,
              "state": None, "run_id": None, "train_pid": None,
              "actions": [], "errors": []}

    st = loop_state.load(state_path)
    run_id, pid = st.get("in_flight_run"), st.get("train_pid")
    report["run_id"], report["train_pid"] = run_id, pid
    if st.get("_recovered"):
        report["actions"].append({"step": "loop_state", "action": "note",
                                  "why": "state file missing/corrupt — read fail-open"})
    if not run_id:
        report["state"] = "clean"
        return report
    if pid is None:
        report["state"] = "unverifiable"
        report["actions"].append({"step": "liveness", "action": "skipped",
                                  "why": f"in_flight_run={run_id!r} has no train_pid; "
                                         "liveness cannot be proven — refusing to "
                                         "declare it dead"})
        return report
    if pid_alive(pid):
        report["state"] = "alive"
        report["actions"].append({"step": "liveness", "action": "alive",
                                  "pid": pid, "why": "trainer still running — untouched"})
        return report

    # --- the trainer is gone -------------------------------------------------
    report["state"] = "stale"
    report["actions"].append({"step": "liveness", "action": "dead", "pid": pid})
    try:
        entry = ledger_run_entry(ledger_path, run_id)
    except RuntimeError as e:
        report["ok"] = False
        report["errors"].append(str(e))
        return report

    new_status = None
    if entry is None:
        report["state"] = "stale-unrecorded"
        report["actions"].append({"step": "ledger", "action": "no-entry",
                                  "why": f"no ledger run {run_id!r} — this launch was "
                                         "never recorded; run `adopt` to create the "
                                         "entry. Refusing to invent one here"})
    elif entry.get("status") not in IN_FLIGHT_STATUSES:
        report["actions"].append({"step": "ledger", "action": "already-terminal",
                                  "status": entry.get("status")})
    else:
        # Terminal status is READ off disk, never assumed: a verdict.json in the
        # artifacts dir means the run reached its end; its absence means it did not.
        adir = entry.get("artifacts_dir")
        vpath = (ROOT / adir / "verdict.json") if adir else None
        finished = bool(vpath and vpath.exists())
        # §C8 vocabulary, never a free-text status (drift is caught by
        # test_terminal_statuses_are_ledger_vocabulary against ledger.RUN_STATUS).
        new_status = TERMINAL_IF_FINISHED if finished else TERMINAL_IF_DIED
        basis = (f"verdict.json present at {adir}/verdict.json"
                 if finished else "no verdict.json in the artifacts dir")
        reconciled = {"protocol": PROTOCOL_VERSION, "reconciled_on": ts,
                      "dead_train_pid": pid, "basis": basis,
                      "verdict_untouched": True}
        sets = ["--set", f"status={new_status}", "--set", f"ended={today}",
                "--set", f"reconciled={json.dumps(reconciled)}"]
        argv = ["update-run", run_id] + sets
        if not apply:
            report["actions"].append({"step": "ledger", "action": "would-update-run",
                                      "status": new_status, "basis": basis,
                                      "argv": argv})
        else:
            rc, out, cli_err = _ledger_cli(ledger_path, argv)
            if rc != 0:
                report["ok"] = False
                report["actions"].append({"step": "ledger", "action": "FAILED",
                                          "exit": rc, "stderr": cli_err.strip()})
                report["errors"].append(
                    f"ledger update-run FAILED (exit {rc}): {(cli_err or out).strip()} "
                    "— loop_state pointer left in place so the stale run is not lost")
                return report
            report["actions"].append({"step": "ledger", "action": "update-run",
                                      "status": new_status, "basis": basis})

    # Clear the stale pointer. `resume_cmd`/`ckpt_path` are PRESERVED by default:
    # boot_resume.sh needs BOTH in_flight_run and resume_cmd to act, so clearing the
    # pointer already disarms auto-resume, and keeping the command means a human can
    # still relaunch by hand from the exact recorded form. --clear-resume drops them.
    kw = {"in_flight_run": None, "train_pid": None}
    if clear_resume:
        kw.update({"ckpt_path": None, "resume_cmd": None})
    if not apply:
        report["actions"].append({"step": "loop_state", "action": "would-clear", **kw})
    else:
        try:
            loop_state.register(state_path, ts=ts, **kw)
        except (ValueError, OSError) as e:
            report["ok"] = False
            report["errors"].append(f"loop_state.register failed: {e}")
            return report
        report["actions"].append({"step": "loop_state", "action": "cleared", **kw})
    report["new_status"] = new_status
    return report


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    common.add_argument("--state", type=Path, default=DEFAULT_STATE,
                        help="loop_state.json (written only via loop_state.register)")
    common.add_argument("--ts", default=None, help="timestamp stamp; default: now (§C2)")
    common.add_argument("--apply", action="store_true",
                        help="actually mutate; without it nothing is written")

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("adopt", parents=[common],
                       help="record a manual launch as a governed run")
    p.add_argument("run_dir", type=Path, help="<MODEL_DIR>/experiments/<RUN_ID>/")
    p.add_argument("--train-pid", type=int, default=None,
                   help="live trainer pid to register as in-flight (must be alive)")
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--resume-cmd", default=None,
                   help="the exact detached relaunch form the recovery chain replays")
    p.add_argument("--run-type", default=None, choices=sorted(RUN_TYPES),
                   help="override the objective->type mapping (§C8 vocabulary)")
    p.add_argument("--digests", type=Path, default=DEFAULT_DIGESTS)
    p.add_argument("--force", action="store_true",
                   help="override the run-id/dir mismatch and in-flight-collision refusals")

    p = sub.add_parser("reconcile", parents=[common],
                       help="close out an in-flight run whose trainer is gone")
    p.add_argument("--clear-resume", action="store_true",
                   help="also null ckpt_path/resume_cmd (default: keep them so a "
                        "human can still relaunch by hand)")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                   help="`ended` date; default: today (§C2)")

    a = ap.parse_args(argv)
    if a.cmd == "adopt":
        r = adopt(a.run_dir, ledger_path=a.ledger, state_path=a.state,
                  digest_dir=a.digests, train_pid=a.train_pid,
                  resume_cmd=a.resume_cmd, ckpt_path=a.ckpt_path,
                  run_type=a.run_type, ts=a.ts, apply=a.apply, force=a.force)
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1
    r = reconcile(ledger_path=a.ledger, state_path=a.state,
                  clear_resume=a.clear_resume, ts=a.ts, today=a.date, apply=a.apply)
    print(json.dumps(r, indent=2))
    if not r["ok"]:
        return 1
    # exit 4 = "a stale in-flight run is sitting there and --apply was not given",
    # the same shape as `sentinel.py liveness` exit 4, so a cron can key on it.
    if r["state"] in ("stale", "stale-unrecorded") and not a.apply:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
