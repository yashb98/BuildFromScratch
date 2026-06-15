"""Regression tests for research/ledger/ledger.py — the single source of
experiment truth (§C8). Covers schema validation, exit-code contract, atomic
write + .bak, never_repeat dedup/auto-sync, backward-compat defaulting, the
read-only-never-writes guarantee, and next-best ranking + objective filter.

Every test runs against an isolated temp ledger (the `ledger_path` fixture);
the real ledger is never read or written.
"""
import hashlib
import json
import re

import ledger


def run(ledger_path, *argv):
    """Invoke the CLI; return its exit code (catching the fail()/argparse
    sys.exit so the test can assert on the code instead of crashing)."""
    try:
        rc = ledger.main([*argv, "--ledger", str(ledger_path)])
        return 0 if rc is None else rc
    except SystemExit as e:
        return e.code


def reload(ledger_path):
    return json.loads(ledger_path.read_text())


# --------------------------------------------------------------- basic CRUD

def test_status_empty(ledger_path):
    assert run(ledger_path, "status") == 0


def test_add_technique_then_dup(ledger_path):
    assert run(ledger_path, "add-technique", "--slug", "t1", "--title", "X") == 0
    d = reload(ledger_path)
    assert len(d["techniques"]) == 1 and d["techniques"][0]["slug"] == "t1"
    assert run(ledger_path, "check-dup", "t1") == 1   # duplicate -> exit 1
    assert run(ledger_path, "check-dup", "t2") == 0   # new -> exit 0


def test_duplicate_add_fails(ledger_path):
    assert run(ledger_path, "add-technique", "--slug", "t1", "--title", "X") == 0
    assert run(ledger_path, "add-technique", "--slug", "t1", "--title", "Y") == 2


def test_update_missing_returns_3(ledger_path):
    assert run(ledger_path, "update-technique", "nope", "--set", "status=done") == 3
    assert run(ledger_path, "update-run", "2026-06-15_m_x", "--set", "status=done") == 3


# --------------------------------------------------------------- validation

def test_status_enum_rejected_via_set(ledger_path):
    # --set bypasses argparse choices, so this exercises validate(), not argparse.
    assert run(ledger_path, "add-technique", "--slug", "t", "--title", "X",
               "--set", "status=bogus") == 2


def test_run_id_format_enforced(ledger_path):
    assert run(ledger_path, "add-run", "--run-id", "badid", "--type", "eval") == 2
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_t1",
               "--type", "eval") == 0


def test_run_objective_rejects_any(ledger_path):
    # §C13: a run is concretely pretrain or finetune — never `any`.
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_m_s",
               "--type", "ablation", "--set", "objective=any") == 2


def test_date_must_be_date_only(ledger_path):
    # A full ISO datetime must be rejected, not silently stored (§C8).
    assert run(ledger_path, "add-technique", "--slug", "t", "--title", "X",
               "--set", "paper_date=2026-06-15T10:00:00Z") == 2
    assert run(ledger_path, "add-technique", "--slug", "t2", "--title", "X",
               "--set", "paper_date=2026/06/15") == 2


# --------------------------------------------------------------- never_repeat

def test_never_repeat_blocks_add(ledger_path):
    assert run(ledger_path, "add-never-repeat", "s1") == 0
    assert run(ledger_path, "add-technique", "--slug", "s1", "--title", "X") == 2


def test_verdict_loss_autosyncs_never_repeat(ledger_path):
    assert run(ledger_path, "add-technique", "--slug", "tt", "--title", "X") == 0
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_m_tt",
               "--type", "ablation", "--technique-slug", "tt",
               "--set", 'budget={"tokens":1,"source":"x"}',
               "--set", 'probe={"tokens_per_sec":1,"peak_mem_gb":1}',
               "--set", "eta_hours=1") == 0
    assert run(ledger_path, "update-run", "2026-06-15_m_tt",
               "--set", "verdict=loss") == 0
    assert "tt" in reload(ledger_path)["never_repeat"]


# --------------------------------------------------------- durability / safety

def test_mutation_creates_bak(ledger_path):
    run(ledger_path, "add-technique", "--slug", "t", "--title", "X")
    assert ledger_path.with_name("ledger.json.bak").exists()


def test_readonly_ops_never_write(ledger_path):
    run(ledger_path, "add-technique", "--slug", "a", "--title", "A",
        "--set", "paper_date=2026-06-01", "--set", "status=briefed")
    before = hashlib.md5(ledger_path.read_bytes()).hexdigest()
    run(ledger_path, "status")
    run(ledger_path, "query", "--collection", "all")
    run(ledger_path, "check-dup", "a")
    run(ledger_path, "next-best", "--cutoff", "2026-01-01", "--include-candidates")
    after = hashlib.md5(ledger_path.read_bytes()).hexdigest()
    assert before == after


def test_apply_defaults_backcompat(ledger_path):
    """Legacy entries predate objective/taxonomy/taste_score (techniques) and
    smoke/framework (runs); load() must inject defaults in memory and never
    reject them."""
    legacy = {
        "schema_version": 1,
        "techniques": [{"slug": "old", "title": "O", "status": "candidate",
                        "paper_date": "2026-06-01"}],
        "runs": [{"run_id": "2026-06-01_m_old", "type": "ablation",
                  "status": "done", "metrics": {}}],
        "proposals": [], "never_repeat": [],
    }
    ledger_path.write_text(json.dumps(legacy))
    data = ledger.load(ledger_path)  # validate() + apply_defaults()
    t = data["techniques"][0]
    assert t["objective"] == "any" and t["taxonomy"] == [] and t["taste_score"] is None
    r = data["runs"][0]
    assert r["smoke"] is None and r["framework"] is None
    assert data["papers"] == []      # §C16 optional collection injected


# --------------------------------------------------------------- next-best

def test_next_best_taste_ranking(ledger_path, capsys):
    for slug, taste in (("hi", "9"), ("lo", "3")):
        run(ledger_path, "add-technique", "--slug", slug, "--title", slug,
            "--set", "status=briefed", "--set", "paper_date=2026-06-01",
            "--set", f"taste_score={taste}")
    run(ledger_path, "add-technique", "--slug", "nul", "--title", "nul",
        "--set", "status=briefed", "--set", "paper_date=2026-06-01")
    capsys.readouterr()                       # clear add output
    run(ledger_path, "next-best", "--cutoff", "2026-01-01")
    out = json.loads(capsys.readouterr().out)
    assert [t["slug"] for t in out] == ["hi", "lo", "nul"]  # taste desc, null LAST


def test_next_best_objective_filter(ledger_path, capsys):
    for slug, obj in (("ft", "finetune"), ("pre", "pretrain-ablation"), ("an", "any")):
        run(ledger_path, "add-technique", "--slug", slug, "--title", slug,
            "--set", "status=briefed", "--set", "paper_date=2026-06-01",
            "--objective", obj)
    capsys.readouterr()
    run(ledger_path, "next-best", "--cutoff", "2026-01-01", "--objective", "finetune")
    out = json.loads(capsys.readouterr().out)
    # finetune keeps finetune + any; pretrain-ablation is dropped.
    assert sorted(t["slug"] for t in out) == ["an", "ft"]


# ------------------------------------------- §C8 lineage + cost reproducibility

LINEAGE_KEYS = {"git_commit", "env", "dataset_id", "dataset_revision",
                "data_hash", "artifact_sha256"}
COST_KEYS = {"wall_clock_min", "gpu_hours"}


def test_git_head_commit_is_hash_or_none():
    c = ledger.git_head_commit()
    assert c is None or re.fullmatch(r"[0-9a-f]{7,40}", c)


def test_new_run_has_lineage_and_cost_slots(ledger_path):
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_t",
               "--type", "eval") == 0
    r = reload(ledger_path)["runs"][0]
    assert set(r["lineage"]) == LINEAGE_KEYS
    assert set(r["cost"]) == COST_KEYS
    # git_commit is auto-captured (we are inside a git checkout) — proves the
    # slot is real, not a permanently-empty placeholder.
    assert r["lineage"]["git_commit"] is None or \
        re.fullmatch(r"[0-9a-f]{7,40}", r["lineage"]["git_commit"])


def test_lineage_populated_via_set(ledger_path):
    payload = ('lineage={"git_commit":"abc1234","env":"torch2.11","dataset_id":'
               '"d","dataset_revision":"rev","data_hash":"h","artifact_sha256":"s"}')
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_m_t",
               "--type", "eval", "--set", payload) == 0
    lin = reload(ledger_path)["runs"][0]["lineage"]
    assert lin["dataset_revision"] == "rev" and lin["artifact_sha256"] == "s"


def test_lineage_must_be_dict_or_null(ledger_path):
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_m_b",
               "--type", "eval", "--set", "lineage=5") == 2


def test_apply_defaults_injects_lineage_cost(ledger_path):
    legacy = {
        "schema_version": 1, "techniques": [],
        "runs": [{"run_id": "2026-06-01_m_old", "type": "ablation",
                  "status": "done", "metrics": {}}],
        "proposals": [], "never_repeat": [],
    }
    ledger_path.write_text(json.dumps(legacy))
    r = ledger.load(ledger_path)["runs"][0]
    assert set(r["lineage"]) == LINEAGE_KEYS and set(r["cost"]) == COST_KEYS


# ------------------------------------------- adversarial-review fixes (ledger)

def test_add_run_lineage_set_preserves_git_commit(ledger_path):
    # cmd_add_run merge site: a --set lineage at add time must still keep the
    # auto-captured git_commit (which is filled BEFORE the --set is applied).
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_m_a", "--type", "eval",
               "--set", 'lineage={"dataset_id":"fineweb"}') == 0
    lin = reload(ledger_path)["runs"][0]["lineage"]
    assert lin["dataset_id"] == "fineweb"
    assert lin["git_commit"] is None or re.fullmatch(r"[0-9a-f]{7,40}", lin["git_commit"])
    # git_commit must NOT have been dropped to a missing key by the --set
    assert "git_commit" in lin


def test_update_run_lineage_set_preserves_git_commit(ledger_path):
    # cmd_update_run merge site: the auto-captured git_commit must survive a
    # writer's incremental `--set lineage={...}` (merge, not wholesale replace).
    run(ledger_path, "add-run", "--run-id", "2026-06-15_m_t", "--type", "eval")
    gc0 = reload(ledger_path)["runs"][0]["lineage"]["git_commit"]
    assert run(ledger_path, "update-run", "2026-06-15_m_t",
               "--set", 'lineage={"dataset_id":"wikitext","data_hash":"deadbeef"}') == 0
    lin = reload(ledger_path)["runs"][0]["lineage"]
    assert lin["git_commit"] == gc0            # preserved, not dropped
    assert lin["dataset_id"] == "wikitext" and lin["data_hash"] == "deadbeef"


def test_apply_defaults_normalizes_partial_lineage(ledger_path):
    legacy = {
        "schema_version": 1, "techniques": [],
        "runs": [{"run_id": "2026-06-01_m_p", "type": "ablation", "status": "done",
                  "metrics": {}, "lineage": {"git_commit": "abc1234"}}],
        "proposals": [], "never_repeat": [],
    }
    ledger_path.write_text(json.dumps(legacy))
    lin = ledger.load(ledger_path)["runs"][0]["lineage"]
    assert set(lin) == LINEAGE_KEYS                       # all subkeys present
    assert lin["git_commit"] == "abc1234" and lin["env"] is None  # existing wins


def test_check_date_rejects_impossible_calendar_dates(ledger_path):
    assert run(ledger_path, "add-technique", "--slug", "a", "--title", "A",
               "--set", "paper_date=2026-13-45") == 2     # month 13, day 45
    assert run(ledger_path, "add-technique", "--slug", "b", "--title", "B",
               "--set", "paper_date=2026-02-30") == 2     # Feb 30
    assert run(ledger_path, "add-technique", "--slug", "c", "--title", "C",
               "--set", "paper_date=2026-02-28") == 0     # a real date still OK


def test_next_best_queued_ranks_first_despite_lower_taste(ledger_path, capsys):
    run(ledger_path, "add-technique", "--slug", "q", "--title", "Q",
        "--set", "status=queued", "--set", "paper_date=2026-06-01", "--set", "taste_score=1")
    run(ledger_path, "add-technique", "--slug", "b", "--title", "B",
        "--set", "status=briefed", "--set", "paper_date=2026-06-01", "--set", "taste_score=9")
    capsys.readouterr()
    run(ledger_path, "next-best", "--cutoff", "2026-01-01")   # default {queued, briefed}
    out = json.loads(capsys.readouterr().out)
    assert [t["slug"] for t in out] == ["q", "b"]   # queued first, even at taste 1 < 9


def test_readonly_ops_never_write_legacy_file(ledger_path):
    # The harder invariant: a LEGACY file missing the new keys (which
    # apply_defaults injects IN MEMORY) must stay byte-identical after reads.
    legacy = {
        "schema_version": 1,
        "techniques": [{"slug": "old", "title": "O", "status": "briefed",
                        "paper_date": "2026-06-01"}],
        "runs": [{"run_id": "2026-06-01_m_old", "type": "ablation",
                  "status": "done", "metrics": {}}],
        "proposals": [], "never_repeat": [],
    }
    ledger_path.write_text(json.dumps(legacy, indent=2))
    before = hashlib.md5(ledger_path.read_bytes()).hexdigest()
    run(ledger_path, "status")
    run(ledger_path, "query", "--collection", "all")
    run(ledger_path, "next-best", "--cutoff", "2026-01-01", "--include-candidates")
    assert hashlib.md5(ledger_path.read_bytes()).hexdigest() == before


def test_bak_equals_premutation_bytes(ledger_path):
    before = ledger_path.read_bytes()
    run(ledger_path, "add-technique", "--slug", "t", "--title", "X")
    assert ledger_path.with_name("ledger.json.bak").read_bytes() == before


def test_save_preserves_file_mode(ledger_path):
    ledger_path.chmod(0o640)
    run(ledger_path, "add-technique", "--slug", "t", "--title", "X")
    assert (ledger_path.stat().st_mode & 0o777) == 0o640
