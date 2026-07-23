"""Tests for research/adopt_run.py — the adopted-run protocol (upgrade-plan #10).

The defect being guarded: the out-of-loop launch is the DOMINANT mode on this box
(~24 hand-driven runs vs ~2 through /ablation-runner), and it used to leave a
hand-written c5 file, a hand-added ledger entry and a hand-mutated loop_state.json
behind. Every test below runs against tmp_path fixtures — the real ledger.json and
loop_state.json are never read or written here (nor by adopt_run itself: it goes
through ledger.py's CLI and loop_state.register()).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import adopt_run
import ledger
import loop_state as ls

RUN_ID = "2026-07-23_testmodel_adopt-me"
TECH = "adopt-me"


# ------------------------------------------------------------------ fixtures

def _complete_evidence(**over):
    """A §C5-complete evidence dict in the numbered styling the live HybridSSM
    c5_evidence.json uses (verified against
    HybridSSM-0.2B/experiments/2026-07-21_hybrid-ssm-0.2b_arch-ladder/c5_evidence.json)."""
    e = {"run_id": RUN_ID, "model_dir": "TestModel", "objective": "pretrain-ablation",
         "framework": "jax", "technique_slug": TECH, "lifecycle_stage": "architecture",
         "c5_0_smoke": {"result": "pass", "detail": "1 step, exit 0"},
         "c5_1_concurrency": {"result": "pass"},
         "c5_2_budget": {"tokens_total": 42_000_000, "source": "brief"},
         "c5_3_probe": {"tokens_per_sec": 2863, "peak_mem_gb": 16.6, "fits": True},
         "c5_4_eta_hours": 4.5,
         "c5_5_resume": {"result": "pass"},
         "c5_6_sentinel": {"result": "armed"},
         "c5_7_guards": {"result": "verified"}}
    e.update(over)
    return e


@pytest.fixture
def run_dir(tmp_path):
    """<MODEL_DIR>/experiments/<RUN_ID>/ carrying a complete c5_evidence.json."""
    d = tmp_path / "TestModel" / "experiments" / RUN_ID
    d.mkdir(parents=True)
    (d / "c5_evidence.json").write_text(json.dumps(_complete_evidence(), indent=2))
    return d


@pytest.fixture
def seeded_ledger(ledger_path):
    """The conftest ledger + the technique entry, because §C8 referential integrity
    hard-fails an `ablation` run whose technique_slug has no technique entry."""
    ledger.main(["add-technique", "--slug", TECH, "--title", "Adopt Me",
                 "--ledger", str(ledger_path)])
    return ledger_path


@pytest.fixture
def state_path(tmp_path):
    p = tmp_path / "loop_state.json"
    ls.save(p, ls.default_state())
    return p


@pytest.fixture
def digests(tmp_path):
    return tmp_path / "digests"


def _runs(ledger_path):
    return json.loads(ledger_path.read_text())["runs"]


def _entry(ledger_path, run_id=RUN_ID):
    return next((r for r in _runs(ledger_path) if r["run_id"] == run_id), None)


def _dead_pid():
    """A pid that is definitely gone: spawn, wait, reap. Not a zombie (we waited),
    so pid_alive must read it as dead."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _adopt(run_dir, seeded_ledger, state_path, digests, **kw):
    return adopt_run.adopt(run_dir, ledger_path=seeded_ledger, state_path=state_path,
                           digest_dir=digests, ts="2026-07-23T18:00:00+01:00", **kw)


# ------------------------------------------------------------------ vocabulary

def test_terminal_statuses_are_ledger_vocabulary():
    """§C8 is the authority on run status/type words. If ledger.py ever renames one,
    this fails HERE instead of at 3am inside a reconcile that then writes a status
    the ledger rejects."""
    assert adopt_run.RUN_STATUS == ledger.RUN_STATUS
    assert adopt_run.RUN_TYPES == ledger.RUN_TYPES
    assert adopt_run.TERMINAL_IF_FINISHED in ledger.RUN_STATUS
    assert adopt_run.TERMINAL_IF_DIED in ledger.RUN_STATUS
    assert set(adopt_run.IN_FLIGHT_STATUSES) <= ledger.RUN_STATUS
    assert set(adopt_run.OBJECTIVE_TO_RUN_TYPE) == ledger.RUN_OBJECTIVES
    assert set(adopt_run.OBJECTIVE_TO_RUN_TYPE.values()) <= ledger.RUN_TYPES


# ------------------------------------------------------------------ pid liveness

def test_pid_alive_reads_reality():
    assert adopt_run.pid_alive(os.getpid()) is True
    assert adopt_run.pid_alive(_dead_pid()) is False


# ------------------------------------------------------------------ adopt: refusals

def test_adopt_refuses_incomplete_c5_and_writes_nothing(
        run_dir, seeded_ledger, state_path, digests):
    """No evidence, no adoption — and a refusal must be TOTAL: an incomplete c5 file
    must not leave a half-adopted run (entry but no digest, or a pointer with no entry)."""
    bad = _complete_evidence()
    del bad["c5_3_probe"]
    (run_dir / "c5_evidence.json").write_text(json.dumps(bad))

    r = _adopt(run_dir, seeded_ledger, state_path, digests, apply=True)

    assert r["ok"] is False
    assert r["c5"]["missing_items"] == ["c5.3_probe"]
    assert any("§C5 lint FAILED" in e for e in r["errors"])
    assert _runs(seeded_ledger) == []
    assert not digests.exists()
    assert ls.load(state_path)["in_flight_run"] is None


def test_adopt_refuses_run_id_dir_mismatch(run_dir, seeded_ledger, state_path, digests):
    """The evidence's run_id IS the artifacts pointer; a mismatch would file the
    entry against the wrong directory."""
    (run_dir / "c5_evidence.json").write_text(
        json.dumps(_complete_evidence(run_id="2026-07-23_othermodel_other-run")))
    r = _adopt(run_dir, seeded_ledger, state_path, digests, apply=True)
    assert r["ok"] is False and "refusing" in r["errors"][0]
    assert _runs(seeded_ledger) == []
    # --force is the documented override
    r = _adopt(run_dir, seeded_ledger, state_path, digests, apply=True, force=True)
    assert r["ok"] is True
    assert _entry(seeded_ledger, "2026-07-23_othermodel_other-run") is not None


def test_adopt_refuses_a_dead_train_pid(run_dir, seeded_ledger, state_path, digests):
    """Registering a dead pid as in-flight manufactures exactly the stale pointer
    `reconcile` exists to clean up."""
    r = _adopt(run_dir, seeded_ledger, state_path, digests,
               train_pid=_dead_pid(), apply=True)
    assert r["ok"] is False
    assert "not alive" in r["errors"][0] and "reconcile" in r["errors"][0]
    assert _runs(seeded_ledger) == []
    assert ls.load(state_path)["in_flight_run"] is None


def test_adopt_refuses_to_strand_another_in_flight_run(
        run_dir, seeded_ledger, state_path, digests):
    ls.register(state_path, in_flight_run="2026-07-21_other_run", train_pid=4242)
    r = _adopt(run_dir, seeded_ledger, state_path, digests,
               train_pid=os.getpid(), apply=True)
    assert r["ok"] is False and "stranded" in r["errors"][0]
    assert ls.load(state_path)["in_flight_run"] == "2026-07-21_other_run"
    assert _runs(seeded_ledger) == []


def test_adopt_refuses_unknown_run_type(run_dir, seeded_ledger, state_path, digests):
    (run_dir / "c5_evidence.json").write_text(
        json.dumps(_complete_evidence(objective="sideways")))
    r = _adopt(run_dir, seeded_ledger, state_path, digests, apply=True)
    assert r["ok"] is False and "run type" in r["errors"][0]
    assert _runs(seeded_ledger) == []


def test_adopt_does_not_register_a_pointer_when_the_ledger_write_fails(
        run_dir, tmp_path, state_path, digests):
    """Ordering guarantee: the loop_state pointer must never name a run the ledger
    does not know. A missing ledger makes ledger.py exit non-zero."""
    missing = tmp_path / "no_such_ledger.json"
    r = adopt_run.adopt(run_dir, ledger_path=missing, state_path=state_path,
                        digest_dir=digests, train_pid=os.getpid(), apply=True)
    assert r["ok"] is False and r["errors"]
    assert ls.load(state_path)["in_flight_run"] is None
    assert not digests.exists()


# ------------------------------------------------------------------ adopt: dry run

def test_adopt_dry_run_mutates_nothing(run_dir, seeded_ledger, state_path, digests):
    r = _adopt(run_dir, seeded_ledger, state_path, digests, train_pid=os.getpid())

    assert r["ok"] is True and r["dry_run"] is True
    actions = {a["step"]: a["action"] for a in r["actions"]}
    assert actions == {"ledger": "would-add-run", "digest": "would-write",
                       "loop_state": "would-register"}
    assert _runs(seeded_ledger) == []
    assert not digests.exists()
    assert ls.load(state_path)["in_flight_run"] is None


# ------------------------------------------------------------------ adopt: apply

def test_adopt_apply_writes_entry_digest_and_pointer(
        run_dir, seeded_ledger, state_path, digests):
    pid = os.getpid()
    resume = "setsid nohup bash run_arch_ladder.sh >/dev/null 2>&1 </dev/null &"
    r = _adopt(run_dir, seeded_ledger, state_path, digests, train_pid=pid,
               resume_cmd=resume, ckpt_path="ckpt.pkl", apply=True)
    assert r["ok"] is True and r["dry_run"] is False

    # (b) the ledger entry, carrying the §C5 evidence
    e = _entry(seeded_ledger)
    assert e is not None
    assert e["type"] == "ablation"            # mapped from objective pretrain-ablation
    assert e["objective"] == "pretrain-ablation" and e["framework"] == "jax"
    assert e["status"] == "running"           # a live trainer IS running
    assert e["verdict"] is None               # a launch is not a result
    assert e["smoke"] == "pass"
    assert e["started"] == "2026-07-23"       # read off the RUN_ID, not defaulted to today
    assert e["model_dir"] == "TestModel" and e["technique_slug"] == TECH
    assert e["eta_hours"] == 4.5
    assert e["budget"]["tokens_total"] == 42_000_000
    assert e["probe"]["tokens_per_sec"] == 2863
    assert e["lifecycle_stage"] == "architecture"
    assert e["adopted"]["launched_out_of_loop"] is True
    assert e["adopted"]["protocol"] == adopt_run.PROTOCOL_VERSION
    assert e["adopted"]["train_pid"] == pid
    assert e["evidence_path"].endswith("c5_evidence.json")
    # §C8 back-reference: the technique now knows about the run
    techs = json.loads(seeded_ledger.read_text())["techniques"]
    assert techs[0]["run_ids"] == [RUN_ID]

    # (c) the §C9 digest stub
    stub = adopt_run.digest_stub_path(digests, RUN_ID, "2026-07-23")
    assert stub.exists()
    text = stub.read_text()
    assert RUN_ID in text and "Out-of-loop launch" in text
    assert "A launch is not a result" in text
    assert "**PASS** (7/7" in text and "c5.0_smoke" in text

    # (d) the loop_state pointer, written through register()
    st = ls.load(state_path)
    assert st["in_flight_run"] == RUN_ID and st["train_pid"] == pid
    assert st["resume_cmd"] == resume and st["ckpt_path"] == "ckpt.pkl"
    assert st["stage"] == "S0"                # registering a trainer is not a transition


def test_adopt_without_pid_records_but_registers_nothing(
        run_dir, seeded_ledger, state_path, digests):
    """Adopting a run with no live trainer is legitimate (record a finished manual
    run) — it must NOT claim an in-flight pointer."""
    r = _adopt(run_dir, seeded_ledger, state_path, digests, apply=True)
    assert r["ok"] is True
    assert _entry(seeded_ledger)["status"] == "launched"   # ledger default, not "running"
    assert ls.load(state_path)["in_flight_run"] is None
    assert any(a["step"] == "loop_state" and a["action"] == "skipped" for a in r["actions"])


def test_adopt_is_idempotent_and_never_clobbers(
        run_dir, seeded_ledger, state_path, digests):
    """Re-adopting must not duplicate the run, overwrite a hand-curated entry, or
    rewrite a digest that already exists."""
    _adopt(run_dir, seeded_ledger, state_path, digests, apply=True)
    ledger.main(["update-run", RUN_ID, "--set", 'notes="hand-curated"',
                 "--ledger", str(seeded_ledger)])
    stub = adopt_run.digest_stub_path(digests, RUN_ID, "2026-07-23")
    stub.write_text("HAND EDITED\n")

    r = _adopt(run_dir, seeded_ledger, state_path, digests, apply=True)

    assert r["ok"] is True
    assert len([x for x in _runs(seeded_ledger) if x["run_id"] == RUN_ID]) == 1
    assert _entry(seeded_ledger)["notes"] == "hand-curated"
    assert stub.read_text() == "HAND EDITED\n"
    actions = {a["step"]: a["action"] for a in r["actions"]}
    assert actions["ledger"] == "exists" and actions["digest"] == "exists"


def test_adopt_maps_a_finetune_objective_to_a_finetune_run(
        run_dir, seeded_ledger, state_path, digests):
    (run_dir / "c5_evidence.json").write_text(
        json.dumps(_complete_evidence(objective="finetune")))
    r = _adopt(run_dir, seeded_ledger, state_path, digests, apply=True)
    assert r["ok"] is True
    assert _entry(seeded_ledger)["type"] == "finetune"


def test_adopt_accepts_the_flat_c5_key_styling(
        run_dir, seeded_ledger, state_path, digests):
    """c5_validate accepts both the numbered and the flat key styling; adopt must
    read the evidence through whichever key actually satisfied each §C5 item."""
    flat = {"run_id": RUN_ID, "model_dir": "TestModel", "objective": "pretrain-ablation",
            "framework": "jax", "technique_slug": TECH,
            "smoke": "pass", "budget": {"tokens": 7}, "probe": {"tokens_per_sec": 11},
            "eta_hours": 2.0, "resume": "pass", "sentinel": "armed", "guards": "verified"}
    (run_dir / "c5_evidence.json").write_text(json.dumps(flat))
    r = _adopt(run_dir, seeded_ledger, state_path, digests, apply=True)
    assert r["ok"] is True
    e = _entry(seeded_ledger)
    assert e["budget"] == {"tokens": 7} and e["probe"] == {"tokens_per_sec": 11}
    assert e["eta_hours"] == 2.0 and e["smoke"] == "pass"


def test_adopt_omits_the_smoke_stamp_when_smoke_did_not_pass(
        run_dir, seeded_ledger, state_path, digests):
    """§C5.0 is a gate, not a checkbox: evidence that RECORDS a smoke result other
    than pass must not be stamped `smoke=pass` in the ledger."""
    (run_dir / "c5_evidence.json").write_text(
        json.dumps(_complete_evidence(c5_0_smoke={"result": "fail", "detail": "OOM"})))
    r = _adopt(run_dir, seeded_ledger, state_path, digests, apply=True)
    assert r["ok"] is True
    assert _entry(seeded_ledger)["smoke"] is None


# ------------------------------------------------------------------ reconcile

def _reconcile(seeded_ledger, state_path, **kw):
    return adopt_run.reconcile(ledger_path=seeded_ledger, state_path=state_path,
                               ts="2026-07-23T18:30:00+01:00", today="2026-07-23", **kw)


@pytest.fixture
def in_flight(run_dir, seeded_ledger, state_path, digests):
    """An adopted, live run — then the trainer dies. Returns the dead pid."""
    _adopt(run_dir, seeded_ledger, state_path, digests, train_pid=os.getpid(),
           resume_cmd="setsid nohup bash run.sh &", ckpt_path="ckpt.pkl", apply=True)
    dead = _dead_pid()
    ls.register(state_path, train_pid=dead)     # the trainer is gone
    return dead


def test_reconcile_clean_state_is_a_noop(seeded_ledger, state_path):
    r = _reconcile(seeded_ledger, state_path, apply=True)
    assert r["state"] == "clean" and r["ok"] is True and r["actions"] == []


def test_reconcile_leaves_a_live_trainer_alone(
        run_dir, seeded_ledger, state_path, digests):
    _adopt(run_dir, seeded_ledger, state_path, digests, train_pid=os.getpid(), apply=True)
    r = _reconcile(seeded_ledger, state_path, apply=True)
    assert r["state"] == "alive"
    assert _entry(seeded_ledger)["status"] == "running"
    assert ls.load(state_path)["in_flight_run"] == RUN_ID


def test_reconcile_will_not_declare_death_without_a_pid(seeded_ledger, state_path):
    """Fail closed (§C6): an in-flight run with no train_pid cannot be PROVEN dead,
    so nothing is mutated — a missing signal is not evidence of death."""
    ls.register(state_path, in_flight_run=RUN_ID)
    r = _reconcile(seeded_ledger, state_path, apply=True)
    assert r["state"] == "unverifiable"
    assert ls.load(state_path)["in_flight_run"] == RUN_ID


def test_reconcile_dry_run_reports_but_does_not_mutate(
        in_flight, seeded_ledger, state_path):
    r = _reconcile(seeded_ledger, state_path)
    assert r["state"] == "stale" and r["dry_run"] is True
    actions = {a["step"]: a["action"] for a in r["actions"]}
    assert actions == {"liveness": "dead", "ledger": "would-update-run",
                       "loop_state": "would-clear"}
    assert _entry(seeded_ledger)["status"] == "running"          # untouched
    assert ls.load(state_path)["in_flight_run"] == RUN_ID        # untouched


def test_reconcile_apply_closes_the_transaction_without_inventing_a_verdict(
        in_flight, seeded_ledger, state_path):
    """Today's dead ladder, exactly: the pid is gone, so the entry goes terminal and
    the pointer is cleared — but dying is not a result, so verdict stays null."""
    r = _reconcile(seeded_ledger, state_path, apply=True)
    assert r["state"] == "stale" and r["ok"] is True

    e = _entry(seeded_ledger)
    assert e["status"] == "crashed"          # no verdict.json on disk -> it died
    assert e["ended"] == "2026-07-23"
    assert e["verdict"] is None              # THE assertion: no verdict was invented
    assert e["metrics"] == {}
    assert e["reconciled"]["dead_train_pid"] == in_flight
    assert e["reconciled"]["verdict_untouched"] is True
    assert json.loads(seeded_ledger.read_text())["never_repeat"] == []

    st = ls.load(state_path)
    assert st["in_flight_run"] is None and st["train_pid"] is None
    # the relaunch form is PRESERVED by default (clearing in_flight_run already
    # disarms boot_resume.sh, and a human may still want the exact command)
    assert st["resume_cmd"] == "setsid nohup bash run.sh &"
    assert st["ckpt_path"] == "ckpt.pkl"


def test_reconcile_marks_done_when_the_run_actually_finished(
        in_flight, run_dir, seeded_ledger, state_path):
    """The terminal status is READ off disk, not assumed: a verdict.json means the
    run reached its end, so `done` — and the verdict field is STILL not touched."""
    (run_dir / "verdict.json").write_text(json.dumps({"verdict": "null"}))
    r = _reconcile(seeded_ledger, state_path, apply=True)
    e = _entry(seeded_ledger)
    assert e["status"] == "done"
    assert e["verdict"] is None, "reconcile must never copy a verdict.json into the ledger"
    assert "verdict.json present" in r["actions"][1]["basis"]


def test_reconcile_clear_resume_drops_the_relaunch_form(
        in_flight, seeded_ledger, state_path):
    _reconcile(seeded_ledger, state_path, apply=True, clear_resume=True)
    st = ls.load(state_path)
    assert st["resume_cmd"] is None and st["ckpt_path"] is None


def test_reconcile_stale_unrecorded_clears_the_pointer_without_inventing_an_entry(
        seeded_ledger, state_path):
    """The out-of-loop pathology in its purest form: loop_state points at a run the
    ledger has never heard of. Clear the pointer; do NOT fabricate a run entry."""
    ls.register(state_path, in_flight_run="2026-07-21_ghost_never-recorded",
                train_pid=_dead_pid())
    r = _reconcile(seeded_ledger, state_path, apply=True)
    assert r["state"] == "stale-unrecorded"
    assert _runs(seeded_ledger) == []
    assert ls.load(state_path)["in_flight_run"] is None


def test_reconcile_does_not_re_close_a_terminal_entry(
        in_flight, seeded_ledger, state_path):
    ledger.main(["update-run", RUN_ID, "--set", 'status="done"',
                 "--set", 'verdict="promising"', "--ledger", str(seeded_ledger)])
    r = _reconcile(seeded_ledger, state_path, apply=True)
    actions = {a["step"]: a["action"] for a in r["actions"]}
    assert actions["ledger"] == "already-terminal"
    e = _entry(seeded_ledger)
    assert e["status"] == "done" and e["verdict"] == "promising"   # untouched
    assert ls.load(state_path)["in_flight_run"] is None            # pointer still cleared


# ------------------------------------------------------------------ CLI contract

def test_cli_exit_codes(run_dir, seeded_ledger, state_path, digests):
    """0 ok / 1 refused / 4 stale-detected-in-dry-run (mirrors sentinel liveness)."""
    base = ["--ledger", str(seeded_ledger), "--state", str(state_path)]
    assert adopt_run.main(["adopt", str(run_dir), "--digests", str(digests)] + base) == 0
    assert adopt_run.main(["reconcile"] + base) == 0        # nothing in flight -> clean

    ls.register(state_path, in_flight_run=RUN_ID, train_pid=_dead_pid())
    assert adopt_run.main(["reconcile"] + base) == 4        # detected, not applied
    assert adopt_run.main(["reconcile", "--apply"] + base) == 0
    assert adopt_run.main(["reconcile"] + base) == 0        # now clean

    (run_dir / "c5_evidence.json").write_text("{ not json")
    assert adopt_run.main(["adopt", str(run_dir), "--digests", str(digests)] + base) == 1


def test_cli_help_lists_both_verbs(capsys):
    with pytest.raises(SystemExit):
        adopt_run.main(["--help"])
    out = capsys.readouterr().out
    assert "adopt" in out and "reconcile" in out


def test_module_never_names_the_live_state_files_for_writing():
    """Guard the guard (§C8/§C11): adopt_run must reach ledger.json only through
    ledger.py's CLI and loop_state.json only through loop_state.register(). The
    literal filenames may appear as DEFAULT PATHS, but never in an open()."""
    src = Path(adopt_run.__file__).read_text()
    assert "open(" not in src.replace('open(f"/proc/', "")   # only the /proc pid probe
    assert src.count("write_text") == 1     # the digest stub is the ONE file it writes
    assert "stub.write_text(text)" in src
    assert "json.dump(" not in src          # dumps-to-argv only; never dump-to-file
    assert "loop_state.register(" in src    # the sanctioned loop_state setter
    assert "[sys.executable, str(LEDGER_PY)]" in src   # the sanctioned ledger writer
