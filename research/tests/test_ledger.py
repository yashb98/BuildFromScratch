"""Regression tests for research/ledger/ledger.py — the single source of
experiment truth (§C8). Covers schema validation, exit-code contract, atomic
write + .bak, never_repeat dedup/auto-sync, backward-compat defaulting, the
read-only-never-writes guarantee, and next-best ranking + objective filter.

Every test runs against an isolated temp ledger (the `ledger_path` fixture);
the real ledger is never read or written.
"""
import hashlib
import json
import pathlib
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


# ------------------------------------------------- caller contract (repo-wide lint)
# 2026-07-20: score_ladder.py shipped `--type pretrain-ablation` — that is an OBJECTIVE,
# not a run type. argparse rejected every invocation (exit 2), and the caller passed
# `check=False`, so the ledger write failed SILENTLY on every scoring pass of a multi-day
# ladder while the script still returned 0. The scaling-persistence result was scored and
# never landed. A wrong --type must break a test, not a production run.

REPO = pathlib.Path(ledger.__file__).resolve().parents[2]
_PY_TYPE = re.compile(r"""['"]--type['"]\s*,\s*['"]([^'"]+)['"]""")
_SH_TYPE = re.compile(r"--type[ \t]+['\"]?([A-Za-z][\w-]*)")


def _ledger_type_literals(root=None):
    """Every literal `--type` value handed to ledger.py by a caller under `root`
    (default: this repo). Skips ledger.py itself (it DEFINES the choices) and the test
    tree (which passes invalid values on purpose to assert they are rejected).

    Known limitation, stated rather than hidden: a file only qualifies if the literal
    text "ledger.py" appears in it, so a caller that assembles the path piecewise
    (`ROOT / "research" / "ledger" / "ledger.py"`) is invisible to this scan. Every
    current caller names it literally, and test_the_run_type_lint_is_not_vacuous fails
    loudly if that ever stops being true for all of them at once."""
    root = REPO if root is None else pathlib.Path(root)
    for path in sorted(root.rglob("*.py")) + sorted(root.rglob("*.sh")):
        parts = set(path.parts)
        if parts & {".git", "__pycache__", "tests"} or path.name == "ledger.py":
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "ledger.py" not in text or "add-run" not in text:
            continue
        rx = _PY_TYPE if path.suffix == ".py" else _SH_TYPE
        for m in rx.finditer(text):
            yield path.relative_to(root), m.group(1)


def test_every_caller_passes_a_valid_run_type():
    bad = [(p, t) for p, t in _ledger_type_literals() if t not in ledger.RUN_TYPES]
    assert not bad, ("caller(s) passing an invalid ledger --type: "
                     + "; ".join(f"{p} -> {t!r}" for p, t in bad)
                     + f"  (valid: {sorted(ledger.RUN_TYPES)})")


def test_the_run_type_lint_is_not_vacuous():
    """Guard the guard: if the scan ever matches nothing, the test above passes for the
    wrong reason and this whole check silently stops protecting anything."""
    assert list(_ledger_type_literals()), "repo-wide --type scan matched no caller at all"


def test_the_run_type_lint_catches_the_original_bug(tmp_path):
    """Prove the lint DETECTS the real defect, on a synthetic caller rather than by
    reverting the fixed one. Reproduces score_ladder.py's exact broken argv shape."""
    (tmp_path / "caller.py").write_text(
        'subprocess.run(["python3", str(ROOT / "research/ledger/ledger.py"), "add-run",\n'
        '                "--run-id", rid, "--type", "pretrain-ablation",\n'
        '                "--model-dir", "Qwen3-0.6B"], check=False)\n')
    (tmp_path / "caller.sh").write_text(
        'python3 research/ledger/ledger.py add-run --run-id "$RID" --type pretrain-ablation\n')
    found = dict(_ledger_type_literals(tmp_path))
    assert {p.name for p in found} == {"caller.py", "caller.sh"}, found
    assert set(found.values()) == {"pretrain-ablation"}
    assert all(t not in ledger.RUN_TYPES for t in found.values())   # ...and flagged invalid


def test_arxiv_id_extraction():
    assert ledger.arxiv_id("http://arxiv.org/abs/2606.14187") == "2606.14187"
    assert ledger.arxiv_id("https://arxiv.org/pdf/2606.14187v2") == "2606.14187"
    assert ledger.arxiv_id("https://blog.google/some-post") is None
    assert ledger.arxiv_id(None) is None


def test_title_jaccard_fuzzy():
    assert ledger.title_jaccard("Zeta: Dual Whitening!", "zeta dual whitening") == 1.0
    assert ledger.title_jaccard("A method for X", "A totally different thing") < 0.85
    assert ledger.title_jaccard("anything", "") == 0.0


def test_check_dup_catches_arxiv_and_fuzzy_title(ledger_path):
    """The audit gap: exact-slug dedup let the SAME paper re-enter under a different slug.
    check-dup now also catches it by arXiv id and by fuzzy title."""
    ledger.main(["add-technique", "--slug", "zeta", "--title", "Zeta: Dual Whitening for Matrix Opt",
                 "--source-url", "http://arxiv.org/abs/2606.14187", "--ledger", str(ledger_path)])
    # same paper, DIFFERENT slug, via arxiv id -> DUPLICATE
    assert run(ledger_path, "check-dup", "zeta-rebrand",
               "--source-url", "https://arxiv.org/pdf/2606.14187v3") == 1
    # same paper, different slug, via fuzzy title -> DUPLICATE
    assert run(ledger_path, "check-dup", "zeta-reworded",
               "--title", "zeta dual whitening for matrix opt") == 1
    # genuinely new -> NEW (no false positive)
    assert run(ledger_path, "check-dup", "new-thing",
               "--source-url", "http://arxiv.org/abs/2607.00001",
               "--title", "An unrelated widget method") == 0
    # exact-slug path still works with no url/title
    assert run(ledger_path, "check-dup", "zeta") == 1


def test_split_verdict_vocabulary(ledger_path):
    """2026-07-22: 'directional' was split into null / promising so a genuine negative
    result and a big-but-capped effect stop sharing one word. Both must be accepted, and
    neither may auto-append to never_repeat (only 'loss' does)."""
    run(ledger_path, "add-technique", "--slug", "t", "--title", "X")
    for i, verdict in enumerate(("null", "promising", "inconclusive", "directional")):
        rid = f"2026-07-22_m_v{i}"
        assert run(ledger_path, "add-run", "--run-id", rid, "--type", "eval") == 0
        assert run(ledger_path, "update-run", rid, "--set", f'verdict="{verdict}"') == 0, verdict
        assert reload(ledger_path)["runs"][-1]["verdict"] == verdict
    assert reload(ledger_path)["never_repeat"] == [], "neutral verdicts must not never_repeat"
    # a 'loss' still does
    rid = "2026-07-22_m_loss"
    run(ledger_path, "add-run", "--run-id", rid, "--type", "ablation", "--technique-slug", "t")
    run(ledger_path, "update-run", rid, "--set", 'verdict="loss"')
    assert "t" in reload(ledger_path)["never_repeat"]


# ------------------------------------------- run-key hygiene + fsck (§C8/§C10)
# 2026-07-23: an audit of the live ledger found 24 runs carrying 60+ ad-hoc
# top-level keys — among them the PPLs of four 2026-06-16 eval runs sitting
# beside an EMPTY metrics{}, invisible to everything that compares runs. `--set`
# accepts any key by design (that is how additive contract keys land), so the
# fix is an advisory, not a rejection: the existing ledger must stay writable.


def stderr_of(capsys):
    return capsys.readouterr().err


def warnings_of(capsys):
    """The warn() lines as text — one JSON object per line on stderr."""
    return [json.loads(ln)["warning"] for ln in stderr_of(capsys).splitlines() if ln]


def test_unknown_top_level_run_key_warns_but_succeeds(ledger_path, capsys):
    assert run(ledger_path, "add-run", "--run-id", "2026-07-23_m_t", "--type", "eval",
               "--set", "train_pid=1234") == 0
    err = stderr_of(capsys)
    assert "unrecognized top-level keys" in err and "train_pid" in err
    assert reload(ledger_path)["runs"][0]["train_pid"] == 1234   # stored, not dropped


def test_eval_shaped_key_gets_its_own_louder_warning(ledger_path, capsys):
    assert run(ledger_path, "add-run", "--run-id", "2026-07-23_m_t", "--type", "eval",
               "--set", "wikitext2_ppl=37.01", "--set", "train_pid=1") == 0
    # two SEPARATE advisories, each naming only its own keys — the eval one must
    # not be diluted into the generic "we don't know this key" line.
    ws = warnings_of(capsys)
    evalish = [w for w in ws if "EVAL-SHAPED top-level keys" in w]
    other = [w for w in ws if "unrecognized top-level keys" in w]
    assert len(evalish) == 1 and len(other) == 1, ws
    assert "wikitext2_ppl" in evalish[0] and "train_pid" not in evalish[0]
    assert "train_pid" in other[0] and "wikitext2_ppl" not in other[0]
    assert "metrics{}" in evalish[0] and "fsck --fix" in evalish[0]


def test_schema_and_contract_keys_do_not_warn(ledger_path, capsys):
    """No crying wolf: the §C8 core + the additive keys the contracts name
    (§C18 confound_check, §C25.1 lifecycle_stage, §C11 launched_by, §C20
    is_remote, §C5.2 prior_run_id/note) must pass silently."""
    assert run(ledger_path, "add-run", "--run-id", "2026-07-23_m_t", "--type", "eval",
               "--set", "lifecycle_stage=base-eval", "--set", 'launched_by="manual"',
               "--set", "is_remote=false", "--set", 'note="one line"',
               "--set", 'prior_run_id="2026-07-22_m_t"',
               "--set", 'metrics={"ppl":1.0}') == 0
    assert stderr_of(capsys) == ""


def test_update_run_warns_on_each_ad_hoc_set(ledger_path, capsys):
    run(ledger_path, "add-run", "--run-id", "2026-07-23_m_t", "--type", "eval")
    capsys.readouterr()
    assert run(ledger_path, "update-run", "2026-07-23_m_t", "--set", "arm=x") == 0
    assert "arm" in stderr_of(capsys)
    # re-setting the SAME ad-hoc key warns again — the surface is still growing
    assert run(ledger_path, "update-run", "2026-07-23_m_t", "--set", "arm=y") == 0
    assert "arm" in stderr_of(capsys)


def test_is_eval_key_is_narrow_enough_to_move_on():
    # fsck --fix MOVES what this matches, so a false positive would relocate prose.
    for k in ("wikitext2_ppl", "code_floor_abs", "final_val_loss", "suite_version",
              "self_floor", "headline_bpb", "task_acc", "delta_ci95", "code_corpus_id"):
        assert ledger.is_eval_key(k), k
    for k in ("note", "train_pid", "arm_plan", "headline", "guards", "purpose",
              "evidence_path", "resume_cmd", "steps", "design"):
        assert not ledger.is_eval_key(k), k


def _dirty_ledger(ledger_path, repo_root):
    """One run with a WRITTEN detail doc, one with a dangling pointer, and stray
    eval keys beside an empty metrics{} — the live ledger's three defects."""
    run(ledger_path, "add-run", "--run-id", "2026-07-23_m_kept", "--type", "eval")
    run(ledger_path, "add-run", "--run-id", "2026-07-23_m_gone", "--type", "eval",
        "--set", "wikitext2_ppl=37.01", "--set", 'suite_version="text-lm-v2"',
        "--set", "train_pid=99")
    doc = repo_root / "research" / "ledger" / "runs"
    doc.mkdir(parents=True, exist_ok=True)
    (doc / "2026-07-23_m_kept.md").write_text("# real detail doc\n")


def test_fsck_reports_without_fixing(ledger_path, tmp_path, capsys):
    _dirty_ledger(ledger_path, tmp_path)
    before = hashlib.md5(ledger_path.read_bytes()).hexdigest()
    capsys.readouterr()
    assert run(ledger_path, "fsck", "--repo-root", str(tmp_path)) == 1  # repairable
    out = json.loads(capsys.readouterr().out)
    assert [d["run_id"] for d in out["dangling_detail_md"]] == ["2026-07-23_m_gone"]
    assert {s["key"] for s in out["eval_keys_at_top_level"]} == {"wikitext2_ppl",
                                                                 "suite_version"}
    assert out["other_unknown_run_keys"] == [{"run_id": "2026-07-23_m_gone",
                                              "keys": ["train_pid"]}]
    assert out["fixed"] is None
    assert hashlib.md5(ledger_path.read_bytes()).hexdigest() == before  # read-only


def test_fsck_fix_nulls_only_the_dangling_pointer(ledger_path, tmp_path, capsys):
    """The honest repair: a pointer to a document that was never written becomes
    null. It is NOT back-filled from the entry — a generated stub would carry
    nothing the entry does not already say while looking like a real write-up."""
    _dirty_ledger(ledger_path, tmp_path)
    capsys.readouterr()
    assert run(ledger_path, "fsck", "--fix", "--repo-root", str(tmp_path)) == 0
    runs = {r["run_id"]: r for r in reload(ledger_path)["runs"]}
    assert runs["2026-07-23_m_gone"]["detail_md"] is None
    assert runs["2026-07-23_m_kept"]["detail_md"] == \
        "research/ledger/runs/2026-07-23_m_kept.md"
    assert not (tmp_path / "research/ledger/runs/2026-07-23_m_gone.md").exists()


def test_fsck_fix_moves_eval_keys_into_metrics_preserving_values(ledger_path,
                                                                 tmp_path):
    _dirty_ledger(ledger_path, tmp_path)
    assert run(ledger_path, "fsck", "--fix", "--repo-root", str(tmp_path)) == 0
    r = {x["run_id"]: x for x in reload(ledger_path)["runs"]}["2026-07-23_m_gone"]
    assert r["metrics"] == {"wikitext2_ppl": 37.01, "suite_version": "text-lm-v2"}
    assert "wikitext2_ppl" not in r and "suite_version" not in r
    assert r["train_pid"] == 99      # non-eval ad-hoc key is reported, never moved


def test_fsck_fix_is_idempotent(ledger_path, tmp_path, capsys):
    _dirty_ledger(ledger_path, tmp_path)
    run(ledger_path, "fsck", "--fix", "--repo-root", str(tmp_path))
    after_first = ledger_path.read_bytes()
    capsys.readouterr()
    assert run(ledger_path, "fsck", "--fix", "--repo-root", str(tmp_path)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fixed"] == {"detail_md_nulled": [], "eval_keys_moved": [],
                            "skipped_collision": []}
    assert ledger_path.read_bytes() == after_first     # byte-identical re-run


def test_fsck_refuses_to_clobber_a_metrics_collision(ledger_path, tmp_path, capsys):
    """Same name, two DIFFERENT values: a machine must not pick. Report and skip."""
    run(ledger_path, "add-run", "--run-id", "2026-07-23_m_c", "--type", "eval",
        "--set", 'metrics={"wikitext2_ppl":37.01}', "--set", "wikitext2_ppl=99.9")
    capsys.readouterr()
    assert run(ledger_path, "fsck", "--fix", "--repo-root", str(tmp_path)) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["fixed"]["skipped_collision"] == ["2026-07-23_m_c.wikitext2_ppl"]
    r = reload(ledger_path)["runs"][0]
    assert r["metrics"]["wikitext2_ppl"] == 37.01 and r["wikitext2_ppl"] == 99.9


def test_fsck_moves_an_identical_duplicate(ledger_path, tmp_path):
    """Same name, SAME value = no information at stake; the top-level copy goes."""
    run(ledger_path, "add-run", "--run-id", "2026-07-23_m_d", "--type", "eval",
        "--set", 'metrics={"self_floor":true}', "--set", "self_floor=true")
    assert run(ledger_path, "fsck", "--fix", "--repo-root", str(tmp_path)) == 0
    r = reload(ledger_path)["runs"][0]
    assert r["metrics"]["self_floor"] is True and "self_floor" not in r


def test_fsck_clean_ledger_exits_zero(ledger_path, tmp_path, capsys):
    run(ledger_path, "add-run", "--run-id", "2026-07-23_m_t", "--type", "eval",
        "--set", "detail_md=null")
    capsys.readouterr()
    assert run(ledger_path, "fsck", "--repo-root", str(tmp_path)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dangling_detail_md"] == [] and out["eval_keys_at_top_level"] == []


def test_the_run_type_lint_accepts_a_valid_caller(tmp_path):
    """No false positives: the FIXED argv shape must not be flagged. Mirrors the real
    score_ladder.py, which reaches the CLI through a `LEDGER = ROOT / ".../ledger.py"`
    constant — the literal still appears in the file, which is what the scan keys on."""
    (tmp_path / "ok.py").write_text(
        'LEDGER = ROOT / "research/ledger/ledger.py"\n'
        'subprocess.run([str(LEDGER), "add-run", "--run-id", rid,\n'
        '                "--type", "scaling-fit", "--objective", "pretrain-ablation"])\n')
    found = dict(_ledger_type_literals(tmp_path))
    assert list(found.values()) == ["scaling-fit"]
    assert all(t in ledger.RUN_TYPES for t in found.values())
