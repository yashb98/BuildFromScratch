#!/usr/bin/env python3
"""research/ledger/ledger.py — the ONLY read/write interface for ledger.json.

Schema authority: .claude/skills/research-loop/references/contracts.md (§C8).
Conventions and recipes: .claude/skills/experiment-ledger/SKILL.md.
Stdlib-only, pure CPU, no network. Atomic writes: tempfile + os.replace in
the same directory; the previous version is snapshotted to ledger.json.bak
before every mutation. Never hand-edit ledger.json (§C11).

Commands (stdout is JSON, except `status` which is a human summary):
  add-technique     --slug S --title T [--modality M --source-url U
                    --paper-date D --status candidate --score F --brief-path P
                    --objective any|pretrain-ablation|finetune (§C13)
                    --taxonomy TAG (repeatable / comma-list, §C12)
                    --taste-score F (§C15.2)]
  update-technique  <slug> [--add-run-id RID --objective O --taxonomy TAG
                    --taste-score F] --set k=v ...
  add-run           --run-id YYYY-MM-DD_<model>_<slug> --type T
                    [--model-dir D --technique-slug S
                    --objective pretrain-ablation|finetune (§C13)
                    --smoke pass (§C5.0) --framework pytorch|jax|cuda (§C14)]
                    --set k=v ...
  update-run        <run_id> [--objective O --smoke pass --framework F] --set k=v ...
  add-proposal      --slug S --kind K --md-path P [--status open]
  update-proposal   <slug> --set k=v ...
  add-never-repeat  <slug> [--reason TEXT]
  check-dup         <slug>                    # verdict JSON; exit 1 if duplicate
  query             [--status S] [--type T] [--since YYYY-MM-DD] [--collection C]
  next-best         --cutoff YYYY-MM-DD [--limit N] [--include-candidates]
                    [--objective any|pretrain-ablation|finetune (§C13 filter)]
  fsck              [--fix] [--repo-root DIR]  # integrity report; exit 1 if repairable
  status                                      # human summary

--set VALUES are JSON-parsed when possible ( --set score=8.5
--set 'budget={"tokens":1200000000,"source":"brief"}' ), else kept as strings.
--ledger PATH overrides the ledger location (smoke tests use a temp copy).

Auto-sync per §C8: a technique updated to status=rejected, or a run whose
verdict becomes loss, appends its slug to never_repeat[].

Exit codes:
  0  success (check-dup: slug is NEW / not blocked; fsck: clean or repaired)
  1  check-dup: DUPLICATE (slug in techniques[] or never_repeat[]).
     fsck without --fix: repairable integrity issues remain
  2  validation / usage / schema error
  3  target entry not found (update-* on a missing slug or run_id)
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_LEDGER = Path(__file__).resolve().parent / "ledger.json"
REPO_ROOT = Path(__file__).resolve().parents[2]   # detail_md paths are repo-relative
TOP_KEYS = ("techniques", "runs", "proposals", "never_repeat")

TECH_STATUS = {"candidate", "briefed", "queued", "running", "done", "rejected", "proposal"}
RUN_STATUS = {"launched", "running", "resumed", "done", "crashed", "killed"}
RUN_TYPES = {"ablation", "finetune", "eval", "dataset-prep",
             # system-upgrade run types (§C20-§C23): off-box scaling, replication,
             # serving/RAG/safeguards benches, and the sweep parent group.
             "scaling-fit", "replication", "serving-bench", "rag-eval",
             "safeguards-round", "sweep", "serve"}
# Verdict vocabulary (2026-07-22: split "directional", which compressed opposite
# realities — a genuine null and a big significant-but-capped effect wore the same word).
#   win         — HARD-complete, significant, single-variable + iso-FLOP (§C18 gate below).
#   promising   — a REAL, measured (usually significant) effect, capped BELOW win by a
#                 missing §C25 HARD item, n<3 seeds, or an unresolved confound.
#                 "found something, one gate short." NOT a never_repeat loss.
#   null        — measured; the contrast shows NO effect beyond the noise floor.
#                 A first-class negative result ("found nothing"), NOT a never_repeat loss.
#   loss        — worse than baseline; auto-appends to never_repeat.
#   inconclusive— could not be measured/interpreted (crash, undecidable confound, no comparand).
#   directional — DEPRECATED alias, kept valid for old entries; new runs use null|promising.
VERDICTS = {"win", "promising", "null", "loss", "inconclusive", "directional", None}
NEUTRAL_VERDICTS = {"promising", "null", "directional", "inconclusive"}  # measured, not a never_repeat loss
PROP_KINDS = {"new-build", "big-run", "needs-approval"}
PROP_STATUS = {"open", "accepted", "declined"}
PAPER_STATUS = {"drafting", "packaged", "submitted", "published", "abandoned"}  # §C16
# §C13 objective axis. Techniques may be `any` (registrar can't yet tell);
# runs are always one of the two concrete objectives (a run IS pretrain or
# fine-tune), never `any`. §C14 framework axis: a run's runtime.
TECH_OBJECTIVES = {"any", "pretrain-ablation", "finetune"}
RUN_OBJECTIVES = {"pretrain-ablation", "finetune"}
SMOKE_VALUES = {"pass", None}            # §C5.0 smoke gate; null until proven
FRAMEWORKS = {"pytorch", "jax", "cuda", "triton", None}  # §C14 (+ triton kernels, §C-systems)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_.+_.+$")  # YYYY-MM-DD_<model>_<slug>

# ------------------------------------------------- run-key hygiene (§C8/§C10)
# `--set k=v` accepts ANY key, which is what makes the CLI usable for the additive
# keys the contracts keep introducing — and also how 24 live runs accumulated 60+
# ad-hoc top-level keys, several of them measured EVAL numbers stranded outside an
# empty metrics{} (the four 2026-06-16 eval runs). Everything a consumer compares
# across runs reads metrics{}; a number at top level is invisible to it.
# So: recognize the schema keys, WARN on the rest, and never hard-fail — a
# hard-fail would make the already-written ledger unloadable, i.e. unwritable.
RUN_CORE_KEYS = frozenset({
    "run_id", "type", "model_dir", "technique_slug", "budget", "probe", "smoke",
    "framework", "eta_hours", "started", "ended", "status", "verdict", "metrics",
    "artifacts_dir", "lineage", "cost", "detail_md", "objective"})
# Additive keys the contracts name explicitly (§C8 "additive key, ledger.py only").
RUN_ADDITIVE_KEYS = frozenset({
    "confound_check",     # §C18 single-variable / iso-FLOP gate (validated above)
    "lifecycle_stage",    # §C25.1
    "launched_by",        # §C11 launch provenance
    "c5_lint",            # §C5 pre-launch lint result stamped by /ablation-runner (pass)
    "adopted", "evidence_path", "reconciled",  # §C10 adopted-run protocol (research/adopt_run.py)
    "is_remote", "cluster_shape",   # §C20 off-box runs
    "brief_path", "prior_run_id", "note"})  # §C5.2 brief inheritance (+its one-line note)
RUN_KEYS = RUN_CORE_KEYS | RUN_ADDITIVE_KEYS

# Eval-shaped names get their own, louder advisory: those are the ones that
# silently break cross-run comparison. Canonical target shape = the metrics{} of
# run 2026-06-27_qwen3-0.6b_sft-3seed.
EVAL_KEY_EXACT = frozenset({"suite_version", "self_floor"})
EVAL_KEY_SUFFIXES = ("_ppl", "_bpb", "_loss", "_acc", "_floor_abs",
                     "_corpus_id", "_ci95")


def is_eval_key(key: str) -> bool:
    """True for a top-level run key that names a measured eval quantity (or the
    §C10 suite stamp) and therefore belongs INSIDE metrics{}. Deliberately narrow:
    it must never sweep up prose/bookkeeping keys (`note`, `train_pid`, `arm_plan`),
    because fsck --fix MOVES what this matches."""
    return key in EVAL_KEY_EXACT or key.endswith(EVAL_KEY_SUFFIXES)


def unknown_run_keys(r: dict):
    return sorted(k for k in r if k not in RUN_KEYS)


def warn_unknown_run_keys(run_id, keys):
    """Advisory (never fatal) for top-level run keys outside the §C8 schema."""
    evalish = [k for k in keys if is_eval_key(k)]
    other = [k for k in keys if not is_eval_key(k)]
    if evalish:
        warn(f"run[{run_id}]: EVAL-SHAPED top-level keys {evalish} — comparable "
             "numbers must live INSIDE metrics{} (§C8/§C10); nothing that reads "
             "metrics{} will ever see them. Fix: ledger.py fsck --fix")
    if other:
        warn(f"run[{run_id}]: unrecognized top-level keys {other} — not in the §C8 "
             "run schema. Numbers belong in metrics{}, prose in detail_md")


def fail(msg: str, code: int = 2):
    print(json.dumps({"error": msg}), file=sys.stderr)
    sys.exit(code)


def warn(msg: str):
    print(json.dumps({"warning": msg}), file=sys.stderr)


def today() -> str:
    return date.today().isoformat()  # runtime-derived, never hardcoded (§C2)


def git_head_commit():
    """Best-effort short HEAD commit of the repo holding this file, recorded as
    run lineage (§C8). Local, no network; returns None if git is unavailable or
    this isn't a checkout — a null commit is stored rather than failing the add,
    so reproducibility is captured automatically and never silently empty."""
    try:
        repo = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _new_lineage():
    """§C8 reproducibility lineage. git_commit is auto-captured at add-run;
    the rest are populated by the writer skills (ablation-runner pins
    dataset_id/revision/data_hash + env at launch; eval-harness pins
    artifact_sha256 of the scored checkpoint)."""
    return {"git_commit": None, "env": None, "dataset_id": None,
            "dataset_revision": None, "data_hash": None, "artifact_sha256": None}


def _new_cost():
    """§C8 cost accounting, populated at run completion by the writer skill."""
    return {"wall_clock_min": None, "gpu_hours": None}


def check_enum(e: dict, key: str, allowed: set, label: str):
    if key in e and e[key] not in allowed:
        fail(f"{label}.{key} must be one of {sorted(str(a) for a in allowed)}, got {e[key]!r}")


def check_date(e: dict, key: str, label: str):
    v = e.get(key)
    if v is None:
        return
    if not isinstance(v, str) or not DATE_RE.match(v):
        fail(f"{label}.{key} must be YYYY-MM-DD or null, got {v!r}")
    try:
        date.fromisoformat(v)  # calendar validity: reject 2026-13-45 / 2026-02-30
    except ValueError:
        fail(f"{label}.{key} is not a real calendar date: {v!r}")


def validate_technique(t: dict):
    if not t.get("slug"):
        fail("technique.slug must be non-empty")
    check_enum(t, "status", TECH_STATUS, f"technique[{t['slug']}]")
    check_enum(t, "objective", TECH_OBJECTIVES, f"technique[{t['slug']}]")  # §C13
    check_date(t, "paper_date", f"technique[{t['slug']}]")
    check_date(t, "first_seen", f"technique[{t['slug']}]")
    if not isinstance(t.get("run_ids", []), list):
        fail(f"technique[{t['slug']}].run_ids must be a list")
    if not isinstance(t.get("taxonomy", []), list):       # §C12 axes touched
        fail(f"technique[{t['slug']}].taxonomy must be a list")
    ts = t.get("taste_score")                              # §C15.2 deep score
    if ts is not None and not isinstance(ts, (int, float)):
        fail(f"technique[{t['slug']}].taste_score must be a number or null, got {ts!r}")
    pwp = t.get("predicted_win_prob")                      # §C15.3 calibrated prob
    if pwp is not None and not (isinstance(pwp, (int, float)) and not isinstance(pwp, bool)
                                and 0.0 <= pwp <= 1.0):
        fail(f"technique[{t['slug']}].predicted_win_prob must be a number in [0,1] or "
             f"null, got {pwp!r}")


def validate_run(r: dict):
    if not r.get("run_id") or not RUN_ID_RE.match(r["run_id"]):
        fail(f"run.run_id must be YYYY-MM-DD_<model>_<slug> (§C8), got {r.get('run_id')!r}")
    check_enum(r, "type", RUN_TYPES, f"run[{r['run_id']}]")
    check_enum(r, "status", RUN_STATUS, f"run[{r['run_id']}]")
    check_enum(r, "verdict", VERDICTS, f"run[{r['run_id']}]")
    check_enum(r, "objective", RUN_OBJECTIVES, f"run[{r['run_id']}]")  # §C13: never `any`
    check_enum(r, "smoke", SMOKE_VALUES, f"run[{r['run_id']}]")        # §C5.0
    check_enum(r, "framework", FRAMEWORKS, f"run[{r['run_id']}]")      # §C14
    # §C8: all dates are date-only YYYY-MM-DD strings (lexicographic == ISO
    # order; `query --since` and weekly-retro depend on it). A full ISO
    # datetime (e.g. `date -Is`) must be rejected, not silently stored.
    check_date(r, "started", f"run[{r['run_id']}]")
    check_date(r, "ended", f"run[{r['run_id']}]")
    if not isinstance(r.get("metrics", {}), dict):
        fail(f"run[{r['run_id']}].metrics must be a JSON object")
    for sub in ("lineage", "cost"):  # §C8 reproducibility + cost (optional/null)
        if r.get(sub) is not None and not isinstance(r[sub], dict):
            fail(f"run[{r['run_id']}].{sub} must be a JSON object or null")
    # §C18 confound_check hygiene + single-variable WIN gate. The field is
    # optional (a bundle/multi-variable run is legitimately recordable — cf. the
    # IMU-1 matched-compute paper), but if present it must be well-formed, and a
    # recorded `verdict=win` on a launch-bearing run MUST be a single-variable,
    # FLOP-matched comparison — else the headline number is confounded /
    # non-comparable, the exact P0 the rubric names.
    cc = r.get("confound_check")
    if cc is not None:
        if not isinstance(cc, dict):
            fail(f"run[{r['run_id']}].confound_check must be a JSON object or null")
        if "n_vars" in cc and not (isinstance(cc["n_vars"], int)
                                   and not isinstance(cc["n_vars"], bool) and cc["n_vars"] >= 1):
            fail(f"run[{r['run_id']}].confound_check.n_vars must be a positive int")
        if "iso_flop" in cc and not isinstance(cc["iso_flop"], bool):
            fail(f"run[{r['run_id']}].confound_check.iso_flop must be a bool")
    if r.get("verdict") == "win" and r.get("type") in {"ablation", "finetune", "replication"}:
        if not (isinstance(cc, dict) and cc.get("n_vars") == 1 and cc.get("iso_flop") is True):
            fail(f"run[{r['run_id']}].verdict='win' on a {r.get('type')} requires "
                 'confound_check={"n_vars":1,"iso_flop":true} (§C18 single-variable, '
                 f"FLOP-matched); got {cc!r}. A multi-variable bundle must be recorded "
                 "as inconclusive or an explicit recipe-level result, not a bare win.")


def validate_proposal(p: dict):
    if not p.get("slug"):
        fail("proposal.slug must be non-empty")
    check_enum(p, "kind", PROP_KINDS, f"proposal[{p['slug']}]")
    check_enum(p, "status", PROP_STATUS, f"proposal[{p['slug']}]")
    check_date(p, "created", f"proposal[{p['slug']}]")


def validate_paper(p: dict):                                   # §C16
    if not p.get("slug"):
        fail("paper.slug must be non-empty")
    check_enum(p, "status", PAPER_STATUS, f"paper[{p['slug']}]")
    check_date(p, "created", f"paper[{p['slug']}]")
    check_date(p, "updated", f"paper[{p['slug']}]")
    if not isinstance(p.get("run_ids", []), list):
        fail(f"paper[{p['slug']}].run_ids must be a list")


def apply_defaults(data: dict):
    """Backward-compat (§C8): older entries predate objective/taxonomy/
    taste_score (techniques) and objective/smoke/framework (runs). Inject the
    defaults IN MEMORY so next-best/query/filters always see the keys; no key
    means "the default", never a validation failure. Pure read ops never save,
    so this does not rewrite the on-disk file — only a real mutation persists
    the now-complete entry. Idempotent."""
    for t in data.get("techniques", []):
        t.setdefault("objective", "any")        # §C13
        t.setdefault("taxonomy", [])            # §C12
        t.setdefault("taste_score", None)       # §C15.2
    for r in data.get("runs", []):
        r.setdefault("smoke", None)             # §C5.0
        r.setdefault("framework", None)         # §C14
        # Fill missing subkeys too (a legacy/partial dict must not KeyError when
        # a reader indexes lineage["env"] etc.); existing values win (§C8).
        r["lineage"] = {**_new_lineage(), **(r.get("lineage") or {})}
        r["cost"] = {**_new_cost(), **(r.get("cost") or {})}
        # `objective` for runs has no safe blanket default (a run is concretely
        # pretrain or fine-tune); leave it absent if a legacy run omits it so
        # the writer must supply it, but tolerate its absence on read.
    # §C16: papers[] is an optional fifth collection added after schema v1
    # shipped; older ledgers predate it. Inject the empty list in memory so
    # query/status/validate always see it; a real add-paper persists it.
    data.setdefault("papers", [])


def validate(data: dict):
    if data.get("schema_version") != SCHEMA_VERSION:
        fail(f"schema_version mismatch: file has {data.get('schema_version')!r}, "
             f"this CLI implements {SCHEMA_VERSION} (§C8)")
    for k in TOP_KEYS:
        if not isinstance(data.get(k), list):
            fail(f"top-level key {k!r} missing or not a list")
    for t in data["techniques"]:
        validate_technique(t)
    for r in data["runs"]:
        validate_run(r)
    for p in data["proposals"]:
        validate_proposal(p)
    for p in data.get("papers", []):       # §C16 — optional collection, .get for back-compat
        validate_paper(p)
    if not all(isinstance(s, str) for s in data["never_repeat"]):
        fail("never_repeat must contain only slugs (strings)")


def load(path: Path) -> dict:
    if not path.exists():
        fail(f"ledger not found: {path} — seed it with "
             '{"schema_version": 1, "techniques": [], "runs": [], '
             '"proposals": [], "never_repeat": []}')
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        fail(f"ledger is not valid JSON ({e}); restore from {path}.bak or git — never hand-edit")
    validate(data)
    apply_defaults(data)  # backward-compat: legacy entries gain new-key defaults in memory
    return data


def save(path: Path, data: dict):
    validate(data)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
        # §C8 durability: a once-per-day pre-mutation snapshot in backups/ so a
        # mid-iteration corruption can be rolled back to last-good (the .bak is a
        # single slot overwritten on every mutation; the ledger is gitignored, so
        # git is not a fallback). Best-effort; never blocks a write.
        try:
            bdir = path.parent / "backups"
            bdir.mkdir(exist_ok=True)
            daily = bdir / f"{path.stem}-{today()}.json"
            if not daily.exists():
                shutil.copy2(path, daily)
        except OSError:
            pass
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ledger_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates the temp file 0600 and os.replace would keep that —
        # preserve the existing ledger's mode (or 0644 on first creation).
        os.chmod(tmp, (os.stat(path).st_mode & 0o777) if path.exists() else 0o644)
        os.replace(tmp, path)  # atomic on POSIX, same filesystem
        # §C8 durability: fsync the PARENT DIR so the rename itself is persisted.
        # Without this, on the documented GB10 unified-memory hard-crash the new
        # directory entry can be lost on reboot, reverting a just-recorded
        # verdict/metrics even though the file contents were fsynced.
        try:
            dfd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def acquire_lock(path: Path):
    """§C8 cross-process safety: an exclusive advisory lock on <ledger>.lock,
    serializing the load->mutate->save sequence so a recovery cron racing a
    hand-run cannot lose-update. Fail-open: if locking is unavailable, return
    None and proceed (a missing lock must never brick the CLI)."""
    try:
        fd = os.open(str(path.with_name(path.name + ".lock")), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except OSError:
        return None


def release_lock(fd):
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def merge_subdicts(target, patch):
    """The nested `lineage`/`cost` objects must MERGE under the existing
    (factory + auto-captured) values rather than wholesale-replace them — a
    partial `--set lineage={...}` from a writer skill must not drop the
    auto-captured git_commit or sibling subkeys (§C8). Mutates `patch`."""
    for sub in ("lineage", "cost"):
        if isinstance(patch.get(sub), dict):
            patch[sub] = {**(target.get(sub) or {}), **patch[sub]}


def parse_sets(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            fail(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def parse_taxonomy(values):
    """--taxonomy is repeatable AND accepts a comma-list per use
    (`--taxonomy decoder-only --taxonomy "SLM,RoPE"` -> three tags). Trims
    blanks, de-dups, preserves first-seen order (§C12 axes touched)."""
    out = []
    for chunk in values or []:
        for tag in chunk.split(","):
            tag = tag.strip()
            if tag and tag not in out:
                out.append(tag)
    return out


def find(items, key, value):
    return next((x for x in items if x.get(key) == value), None)


def emit(obj):
    print(json.dumps(obj, indent=2))


def block_never_repeat(led, slug):
    if slug and slug not in led["never_repeat"]:
        led["never_repeat"].append(slug)


# ---------------------------------------------------------------- mutators

def cmd_add_technique(led, a):
    if find(led["techniques"], "slug", a.slug):
        fail(f"technique slug {a.slug!r} already exists — run check-dup before add (§C8)")
    if a.slug in led["never_repeat"]:
        fail(f"slug {a.slug!r} is blocked by never_repeat. Retry exception (§C8): a NEW "
             "paper >= CUTOFF_3M that materially changes the recipe -> new suffixed slug "
             "with supersedes=<old-slug>; see experiment-ledger SKILL.md")
    t = {"slug": a.slug, "title": a.title, "modality": a.modality,
         "source_url": a.source_url, "paper_date": a.paper_date,
         "first_seen": a.first_seen or today(), "status": a.status,
         "objective": a.objective, "taxonomy": parse_taxonomy(a.taxonomy),
         "score": a.score, "taste_score": a.taste_score,
         "brief_path": a.brief_path, "run_ids": []}
    t.update(parse_sets(a.set))
    led["techniques"].append(t)
    if t.get("status") == "rejected":
        block_never_repeat(led, t["slug"])
    return t


def cmd_update_technique(led, a):
    t = find(led["techniques"], "slug", a.slug)
    if t is None:
        fail(f"technique {a.slug!r} not found", 3)
    patch = parse_sets(a.set)
    if "slug" in patch:
        fail("technique.slug is immutable — add a new entry instead")
    # Dedicated flags (§C13/§C12/§C15) layer in before --set so --set still wins
    # on a deliberate collision, matching the rest of the CLI's precedence.
    if a.objective is not None:
        t["objective"] = a.objective
    if a.taxonomy:
        t["taxonomy"] = parse_taxonomy(a.taxonomy)
    if a.taste_score is not None:
        t["taste_score"] = a.taste_score
    t.update(patch)
    if a.add_run_id and a.add_run_id not in t.setdefault("run_ids", []):
        t["run_ids"].append(a.add_run_id)
    if t.get("status") == "rejected":
        block_never_repeat(led, t["slug"])
    return t


def cmd_add_run(led, a):
    if find(led["runs"], "run_id", a.run_id):
        fail(f"run_id {a.run_id!r} already exists")
    r = {"run_id": a.run_id, "type": a.type, "model_dir": a.model_dir,
         "technique_slug": a.technique_slug,
         "budget": None, "probe": None, "smoke": a.smoke,
         "framework": a.framework, "eta_hours": None,
         "started": a.started or today(), "ended": None,
         "status": "launched", "verdict": None, "metrics": {},
         "artifacts_dir": a.artifacts_dir,
         "lineage": _new_lineage(), "cost": _new_cost(),  # §C8
         # The PLANNED path of the detail doc — the owning skill writes it later.
         # If it is never written, `fsck --fix` nulls the pointer rather than
         # leaving the entry claiming a document that does not exist.
         "detail_md": f"research/ledger/runs/{a.run_id}.md"}
    r["lineage"]["git_commit"] = git_head_commit()  # auto-capture repo HEAD
    # §C13: a run's objective is one of the two concrete values — never `any`,
    # and never a bare null (the enum rejects null). Set it only when supplied
    # (via flag or --set), so the legacy no-objective add-run form still works
    # for backward compatibility.
    if a.objective is not None:
        r["objective"] = a.objective
    patch = parse_sets(a.set)
    merge_subdicts(r, patch)  # §C8: a partial lineage/cost --set merges, not replaces
    r.update(patch)
    # Only the --set keys can be off-schema here (every other key above is the
    # factory shape), so the advisory names exactly what THIS call invented.
    warn_unknown_run_keys(r["run_id"], [k for k in patch if k not in RUN_KEYS])
    # §C5 evidence must be recorded for ablation|finetune regardless of the
    # initial status — a --set status=... override must not mute the warning.
    if r["type"] in ("ablation", "finetune"):
        for k in ("budget", "probe", "eta_hours"):
            if not r.get(k):
                warn(f"run.{k} is empty — §C5 requires it recorded BEFORE an "
                     "unattended training launch")
    led["runs"].append(r)
    if r.get("technique_slug"):
        t = find(led["techniques"], "slug", r["technique_slug"])
        if t is None:
            # §C8 referential integrity: an orphan ablation/finetune/replication run
            # would let next-best re-select the "untried" technique and burn a
            # SECOND paid run. Hard-fail for launch-bearing types; warn for
            # free-floating ones (eval/dataset-prep/etc.).
            if r["type"] in ("ablation", "finetune", "replication"):
                fail(f"technique_slug {r['technique_slug']!r} has no technique entry — a "
                     f"{r['type']} run must reference a recorded technique (§C8); add the "
                     "technique first so its run_ids[] back-references this run", code=3)
            warn(f"technique_slug {r['technique_slug']!r} has no technique entry")
        elif r["run_id"] not in t.setdefault("run_ids", []):
            t["run_ids"].append(r["run_id"])
    if r.get("verdict") == "loss":
        block_never_repeat(led, r.get("technique_slug"))
    return r


def cmd_update_run(led, a):
    r = find(led["runs"], "run_id", a.run_id)
    if r is None:
        fail(f"run {a.run_id!r} not found", 3)
    patch = parse_sets(a.set)
    if "run_id" in patch:
        fail("run.run_id is immutable")
    if a.objective is not None:          # §C13
        r["objective"] = a.objective
    if a.smoke is not None:              # §C5.0 — record pass once the smoke gate clears
        r["smoke"] = a.smoke
    if a.framework is not None:          # §C14
        r["framework"] = a.framework
    merge_subdicts(r, patch)  # §C8: incremental lineage/cost updates merge, not replace
    r.update(patch)
    # Warn on EVERY off-schema --set, not just the first one to introduce a key:
    # the point is that the ad-hoc surface stops growing silently, and re-setting
    # an ad-hoc key is exactly how it keeps growing. Stateless, so no false quiet.
    warn_unknown_run_keys(r["run_id"], [k for k in patch if k not in RUN_KEYS])
    if r.get("verdict") == "loss":
        block_never_repeat(led, r.get("technique_slug"))
    return r


def cmd_add_proposal(led, a):
    if find(led["proposals"], "slug", a.slug):
        fail(f"proposal slug {a.slug!r} already exists")
    p = {"slug": a.slug, "kind": a.kind, "md_path": a.md_path,
         "created": a.created or today(), "status": a.status}
    p.update(parse_sets(a.set))
    led["proposals"].append(p)
    return p


def cmd_update_proposal(led, a):
    p = find(led["proposals"], "slug", a.slug)
    if p is None:
        fail(f"proposal {a.slug!r} not found", 3)
    patch = parse_sets(a.set)
    if "slug" in patch:
        fail("proposal.slug is immutable")
    p.update(patch)
    return p


def cmd_add_never_repeat(led, a):
    block_never_repeat(led, a.slug)
    t = find(led["techniques"], "slug", a.slug)
    if t is not None and a.reason:
        t["reject_reason"] = a.reason
    return {"slug": a.slug, "never_repeat": True, "reason": a.reason,
            "technique_entry_exists": t is not None}


def cmd_add_paper(led, a):                                     # §C16
    led.setdefault("papers", [])
    if find(led["papers"], "slug", a.slug):
        fail(f"paper slug {a.slug!r} already exists — use update-paper")
    run_ids = parse_taxonomy(a.run_ids)   # reuse the trim / de-dup / order parser
    for rid in run_ids:
        if find(led["runs"], "run_id", rid) is None:
            warn(f"paper.run_ids references {rid!r} with no run entry")
    p = {"slug": a.slug, "title": a.title, "status": a.status,
         "venue": a.venue, "arxiv_id": a.arxiv_id, "run_ids": run_ids,
         "md_path": a.md_path, "created": a.created or today(),
         "updated": a.created or today(), "notes": a.notes}
    p.update(parse_sets(a.set))
    led["papers"].append(p)
    return p


def cmd_update_paper(led, a):                                  # §C16
    led.setdefault("papers", [])
    p = find(led["papers"], "slug", a.slug)
    if p is None:
        fail(f"paper {a.slug!r} not found", 3)
    patch = parse_sets(a.set)
    if "slug" in patch:
        fail("paper.slug is immutable")
    if a.status is not None:
        p["status"] = a.status
    if a.venue is not None:
        p["venue"] = a.venue
    if a.arxiv_id is not None:
        p["arxiv_id"] = a.arxiv_id
    if a.add_run_id:
        if find(led["runs"], "run_id", a.add_run_id) is None:
            warn(f"paper.run_ids references {a.add_run_id!r} with no run entry")
        if a.add_run_id not in p.setdefault("run_ids", []):
            p["run_ids"].append(a.add_run_id)
    p.update(patch)
    p["updated"] = today()
    return p


# ---------------------------------------------------------------- readers

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)


def arxiv_id(url):
    """The bare arXiv id (e.g. '2606.14187') from a source URL, else None. Strips any
    version suffix so v1/v2 of the same paper dedup together."""
    if not url:
        return None
    m = _ARXIV_RE.search(url)
    return m.group(1) if m else None


def norm_title_tokens(title):
    """Lowercase alphanumeric token set of a title — the fuzzy-match key. Punctuation,
    case, and spacing differences collapse away so 'Zeta: Dual Whitening' and
    'zeta dual whitening' match."""
    if not title:
        return frozenset()
    return frozenset(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def title_jaccard(a_title, b_title):
    """Token Jaccard overlap of two titles in [0,1]; 0 if either is empty."""
    A, B = norm_title_tokens(a_title), norm_title_tokens(b_title)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


TITLE_DUP_THRESHOLD = 0.85   # >= this token-Jaccard flags a probable duplicate title


def cmd_check_dup(led, a) -> int:
    """Dedup a candidate against the ledger. Beyond the exact-slug + never_repeat check
    (which depended on three separate LLM calls minting IDENTICAL slugs — the audit's
    gap), it now ALSO catches the SAME paper entering under a different slug via its
    arXiv id (from --source-url) and a fuzzy title match (--title). Cheap and
    deterministic — no external fuzzy lib."""
    hits = []
    t = find(led["techniques"], "slug", a.slug)
    if t is not None:
        hits.append({"where": "techniques", "match": "slug", "slug": a.slug,
                     "status": t.get("status"), "first_seen": t.get("first_seen")})
    if a.slug in led["never_repeat"]:
        hits.append({"where": "never_repeat", "match": "slug", "slug": a.slug})

    cand_arxiv = arxiv_id(getattr(a, "source_url", None))
    cand_title = getattr(a, "title", None)
    if cand_arxiv or cand_title:
        for other in led["techniques"]:
            if other.get("slug") == a.slug:
                continue                                  # already reported above
            if cand_arxiv and arxiv_id(other.get("source_url")) == cand_arxiv:
                hits.append({"where": "techniques", "match": "arxiv_id",
                             "arxiv_id": cand_arxiv, "slug": other.get("slug")})
            elif cand_title:
                j = title_jaccard(cand_title, other.get("title"))
                if j >= TITLE_DUP_THRESHOLD:
                    hits.append({"where": "techniques", "match": "fuzzy_title",
                                 "jaccard": round(j, 3), "slug": other.get("slug"),
                                 "title": other.get("title")})
    emit({"slug": a.slug, "verdict": "DUPLICATE" if hits else "NEW", "hits": hits})
    return 1 if hits else 0


def cmd_query(led, a):
    date_key = {"techniques": "first_seen", "runs": "started",
                "proposals": "created", "papers": "created"}  # §C16
    out = {"techniques": [], "runs": [], "proposals": [], "papers": []}
    cols = ("techniques", "runs", "proposals", "papers")
    for c in (cols if a.collection == "all" else (a.collection,)):
        for it in led.get(c, []):
            if a.status is not None and it.get("status") != a.status:
                continue
            if a.type is not None:
                # --type matches runs.type / proposals.kind; techniques have neither
                if c == "techniques" or it.get("type" if c == "runs" else "kind") != a.type:
                    continue
            v = it.get(date_key[c])
            if a.since is not None and not (isinstance(v, str) and v >= a.since):
                continue  # ISO dates compare lexicographically
            out[c].append(it)
    emit(out)


def cmd_next_best(led, a):
    if not DATE_RE.match(a.cutoff):
        fail(f"--cutoff must be YYYY-MM-DD, derived at runtime per §C2, got {a.cutoff!r}")
    if a.objective is not None and a.objective not in TECH_OBJECTIVES:
        fail(f"--objective must be one of {sorted(TECH_OBJECTIVES)}, got {a.objective!r}")
    # `queued` is always eligible and ranks FIRST: it is a prior iteration's
    # unfulfilled GPU request (§C11) whose brief + dataset already exist —
    # without it, S6's "prep consumed the night" hand-off strands forever.
    statuses = {"queued", "briefed", "candidate"} if a.include_candidates \
        else {"queued", "briefed"}
    rank = {"queued": 0, "briefed": 1, "candidate": 2}
    blocked = set(led["never_repeat"])

    def objective_ok(t):
        # §C13: when the standing objective is set (and not `any`), keep only
        # techniques of that objective; an `any`-tagged technique fits either.
        if a.objective is None or a.objective == "any":
            return True
        return t.get("objective", "any") in (a.objective, "any")

    elig = [t for t in led["techniques"]
            if t.get("status") in statuses and t.get("slug") not in blocked
            and objective_ok(t)
            and isinstance(t.get("paper_date"), str) and t["paper_date"] >= a.cutoff]
    # §C8/§C15: within a status, rank by taste_score (the deep §C15.2 score)
    # with nulls LAST, then the cheap triage score, then date/slug. A None
    # taste_score sorts after any real number (huge negative key so -key is huge).
    def taste_key(t):
        ts = t.get("taste_score")
        return -ts if isinstance(ts, (int, float)) else float("inf")

    elig.sort(key=lambda t: (
        rank.get(t.get("status"), 3),
        taste_key(t),
        -(t["score"] if isinstance(t.get("score"), (int, float)) else float("-inf")),
        t.get("paper_date", ""), t.get("slug", "")))
    emit(elig[: a.limit] if a.limit else elig)


# ------------------------------------------------------------- integrity (fsck)

def dangling_detail_md(led, repo_root=REPO_ROOT):
    """Run entries whose `detail_md` names a document that is not on disk.
    A pointer to a file that does not exist is a claim the ledger cannot back."""
    return [{"run_id": r.get("run_id"), "detail_md": r["detail_md"]}
            for r in led.get("runs", [])
            if isinstance(r.get("detail_md"), str) and r["detail_md"]
            and not (Path(repo_root) / r["detail_md"]).exists()]


def stray_eval_keys(led):
    """Eval-shaped top-level run keys that belong inside metrics{} (§C8/§C10).
    `collides` = metrics{} already holds that key with a DIFFERENT value — two
    values under one name, which a machine must not silently pick between."""
    out = []
    for r in led.get("runs", []):
        m = r.get("metrics") or {}
        for k in unknown_run_keys(r):
            if is_eval_key(k):
                out.append({"run_id": r.get("run_id"), "key": k,
                            "collides": k in m and m[k] != r[k]})
    return out


def other_unknown_keys(led):
    """Non-eval off-schema top-level run keys. REPORTED ONLY — never auto-moved:
    `arm_plan`/`train_pid`/`headline` are bookkeeping or prose, and guessing a
    destination for them would invent structure the ledger never recorded."""
    out = []
    for r in led.get("runs", []):
        ks = [k for k in unknown_run_keys(r) if not is_eval_key(k)]
        if ks:
            out.append({"run_id": r.get("run_id"), "keys": ks})
    return out


def cmd_fsck(led, a) -> int:
    """Integrity report over runs[], with two SAFE repairs behind --fix.

    Repair 1 — a dangling `detail_md` is NULLED, not back-filled. A run .md is a
    narrative artifact (hypothesis, config, caveats) written by the owning skill;
    generating one from the entry would produce a document containing nothing the
    entry does not already say, indistinguishable later from a real write-up. The
    honest record of "no detail doc was ever written" is `null`, and it is
    machine-checkable. `add-run` re-plans the path for each new run, so the
    repair is re-runnable rather than one-shot.

    Repair 2 — an eval-shaped top-level key is MOVED into metrics{} with its value
    byte-identical (`metrics[k] = r.pop(k)`); a value collision is skipped and
    reported. Both repairs are idempotent: a second --fix finds nothing to do."""
    repo_root = a.repo_root or REPO_ROOT
    dangling = dangling_detail_md(led, repo_root)
    stray = stray_eval_keys(led)
    fixed = {"detail_md_nulled": [], "eval_keys_moved": [], "skipped_collision": []}
    if a.fix:
        for d in dangling:
            find(led["runs"], "run_id", d["run_id"])["detail_md"] = None
            fixed["detail_md_nulled"].append(d["run_id"])
        for s in stray:
            r = find(led["runs"], "run_id", s["run_id"])
            if s["collides"]:
                fixed["skipped_collision"].append(f"{s['run_id']}.{s['key']}")
                continue
            r.setdefault("metrics", {})[s["key"]] = r.pop(s["key"])
            fixed["eval_keys_moved"].append(f"{s['run_id']}.{s['key']}")
        if dangling or stray:
            save(a.ledger, led)
    emit({"dangling_detail_md": dangling, "eval_keys_at_top_level": stray,
          "other_unknown_run_keys": other_unknown_keys(led),
          "fixed": fixed if a.fix else None})
    # Exit 1 = repairable issues are still there (CI-gateable): unfixed, or a
    # collision --fix deliberately refused to resolve. Off-schema non-eval keys
    # are advisory and never move the exit code.
    remaining = fixed["skipped_collision"] if a.fix else (dangling or stray)
    return 1 if remaining else 0


def cmd_status(led, a):
    def counts(items):
        c = {}
        for it in items:
            c[str(it.get("status"))] = c.get(str(it.get("status")), 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "none"

    print(f"ledger: {a.ledger}  (schema v{led['schema_version']})")
    print(f"techniques ({len(led['techniques'])}): {counts(led['techniques'])}")
    print(f"runs ({len(led['runs'])}): {counts(led['runs'])}")
    inflight = [r for r in led["runs"] if r.get("status") in ("launched", "running", "resumed")]
    for r in inflight:
        print(f"  in-flight: {r['run_id']}  status={r['status']}  eta_hours={r.get('eta_hours')}")
    if not inflight:
        print("  in-flight: none")
    print(f"proposals ({len(led['proposals'])}): {counts(led['proposals'])}")
    for p in led["proposals"]:
        if p.get("status") == "open":
            print(f"  open: {p['slug']}  kind={p.get('kind')}  md={p.get('md_path')}")
    print(f"never_repeat ({len(led['never_repeat'])}): "
          + (", ".join(led["never_repeat"]) or "none"))
    dangling, stray = dangling_detail_md(led), stray_eval_keys(led)
    if dangling or stray:   # silent once clean, so the line means "act on this"
        print(f"integrity: {len(dangling)} dangling detail_md, {len(stray)} eval "
              "key(s) outside metrics{} — run `ledger.py fsck --fix`")
    papers = led.get("papers", [])                             # §C16
    print(f"papers ({len(papers)}): {counts(papers)}")
    for p in papers:
        if p.get("status") not in ("published", "abandoned"):
            print(f"  active: {p['slug']}  status={p.get('status')}  "
                  f"venue={p.get('venue')}  arxiv={p.get('arxiv_id')}")


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    common.add_argument("--set", action="append", default=[], metavar="KEY=VAL")

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add-technique", parents=[common])
    p.add_argument("--slug", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--modality", default=None)
    p.add_argument("--source-url", default=None)
    p.add_argument("--paper-date", default=None)
    p.add_argument("--first-seen", default=None, help="default: today (runtime)")
    p.add_argument("--status", default="candidate", choices=sorted(TECH_STATUS))
    p.add_argument("--objective", default="any", choices=sorted(TECH_OBJECTIVES),
                   help="§C13: any|pretrain-ablation|finetune (default any)")
    p.add_argument("--taxonomy", action="append", default=[], metavar="TAG",
                   help="§C12 axes touched; repeatable or comma-list")
    p.add_argument("--score", type=float, default=None)
    p.add_argument("--taste-score", type=float, default=None,
                   help="§C15.2 deep score; null until briefed")
    p.add_argument("--brief-path", default=None)

    p = sub.add_parser("update-technique", parents=[common])
    p.add_argument("slug")
    p.add_argument("--add-run-id", default=None)
    p.add_argument("--objective", default=None, choices=sorted(TECH_OBJECTIVES),
                   help="§C13: any|pretrain-ablation|finetune")
    p.add_argument("--taxonomy", action="append", default=[], metavar="TAG",
                   help="§C12 axes touched; repeatable or comma-list (replaces)")
    p.add_argument("--taste-score", type=float, default=None,
                   help="§C15.2 deep score set after briefing")

    p = sub.add_parser("add-run", parents=[common])
    p.add_argument("--run-id", required=True, help="YYYY-MM-DD_<model>_<slug>")
    p.add_argument("--type", required=True, choices=sorted(RUN_TYPES))
    p.add_argument("--model-dir", default=None)
    p.add_argument("--technique-slug", default=None)
    p.add_argument("--objective", default=None, choices=sorted(RUN_OBJECTIVES),
                   help="§C13: pretrain-ablation|finetune (a run is never `any`)")
    p.add_argument("--smoke", default=None, choices=["pass"],
                   help="§C5.0: pass once the pre-launch smoke test clears")
    p.add_argument("--framework", default=None,
                   choices=["pytorch", "jax", "cuda", "triton"],
                   help="§C14 runtime")
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--started", default=None, help="default: today (runtime)")

    p = sub.add_parser("update-run", parents=[common])
    p.add_argument("run_id")
    p.add_argument("--objective", default=None, choices=sorted(RUN_OBJECTIVES),
                   help="§C13: pretrain-ablation|finetune")
    p.add_argument("--smoke", default=None, choices=["pass"], help="§C5.0")
    p.add_argument("--framework", default=None,
                   choices=["pytorch", "jax", "cuda", "triton"],
                   help="§C14 runtime")

    p = sub.add_parser("add-proposal", parents=[common])
    p.add_argument("--slug", required=True)
    p.add_argument("--kind", required=True, choices=sorted(PROP_KINDS))
    p.add_argument("--md-path", required=True)
    p.add_argument("--created", default=None, help="default: today (runtime)")
    p.add_argument("--status", default="open", choices=sorted(PROP_STATUS))

    p = sub.add_parser("update-proposal", parents=[common])
    p.add_argument("slug")

    p = sub.add_parser("add-never-repeat", parents=[common])
    p.add_argument("slug")
    p.add_argument("--reason", default=None)

    p = sub.add_parser("add-paper", parents=[common])           # §C16
    p.add_argument("--slug", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--status", default="drafting", choices=sorted(PAPER_STATUS),
                   help="§C16: drafting|packaged|submitted|published|abandoned")
    p.add_argument("--venue", default=None, help="arXiv / NeurIPS / workshop / …")
    p.add_argument("--arxiv-id", default=None)
    p.add_argument("--run-id", dest="run_ids", action="append", default=[],
                   metavar="RUN_ID", help="source run the paper reports; repeatable")
    p.add_argument("--md-path", default=None, help="research/papers/<slug>/")
    p.add_argument("--created", default=None, help="default: today (runtime)")
    p.add_argument("--notes", default=None)

    p = sub.add_parser("update-paper", parents=[common])        # §C16
    p.add_argument("slug")
    p.add_argument("--status", default=None, choices=sorted(PAPER_STATUS))
    p.add_argument("--venue", default=None)
    p.add_argument("--arxiv-id", default=None)
    p.add_argument("--add-run-id", default=None)

    p = sub.add_parser("check-dup", parents=[common])
    p.add_argument("slug")
    p.add_argument("--source-url", default=None,
                   help="also dedup by arXiv id extracted from this URL")
    p.add_argument("--title", default=None,
                   help="also dedup by fuzzy title match against existing techniques")

    p = sub.add_parser("query", parents=[common])
    p.add_argument("--status", default=None)
    p.add_argument("--type", default=None, help="matches runs.type / proposals.kind")
    p.add_argument("--since", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--collection", default="all",
                   choices=["all", "techniques", "runs", "proposals", "papers"])

    p = sub.add_parser("next-best", parents=[common])
    p.add_argument("--cutoff", required=True, metavar="YYYY-MM-DD",
                   help="CUTOFF_3M, derived at runtime (§C2)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include-candidates", action="store_true",
                   help="widen to status in {queued, briefed, candidate} "
                        "(orchestrator S4); default is {queued, briefed}")
    p.add_argument("--objective", default=None, choices=sorted(TECH_OBJECTIVES),
                   help="§C13 filter: keep techniques of this objective (an "
                        "`any` technique fits either); skipped when val is `any`")

    p = sub.add_parser("fsck", parents=[common])
    p.add_argument("--fix", action="store_true",
                   help="apply the safe repairs (null dangling detail_md; move "
                        "eval-shaped top-level keys into metrics{}); idempotent")
    p.add_argument("--repo-root", type=Path, default=None,
                   help="root that detail_md paths resolve against "
                        "(default: this checkout)")

    sub.add_parser("status", parents=[common])

    a = ap.parse_args(argv)
    # §C8: serialize the whole load->mutate->save under an exclusive lock so two
    # concurrent invocations (e.g. recovery cron + hand-run) cannot lose-update.
    lock_fd = acquire_lock(a.ledger)
    try:
        led = load(a.ledger)

        mutators = {"add-technique": cmd_add_technique,
                    "update-technique": cmd_update_technique,
                    "add-run": cmd_add_run, "update-run": cmd_update_run,
                    "add-proposal": cmd_add_proposal,
                    "update-proposal": cmd_update_proposal,
                    "add-never-repeat": cmd_add_never_repeat,
                    "add-paper": cmd_add_paper, "update-paper": cmd_update_paper}
        if a.cmd in mutators:
            result = mutators[a.cmd](led, a)
            save(a.ledger, led)
            emit(result)
            return 0
        if a.cmd == "check-dup":
            return cmd_check_dup(led, a)
        if a.cmd == "fsck":       # writes only under --fix, and only if dirty
            return cmd_fsck(led, a)
        if a.cmd == "query":
            cmd_query(led, a)
        elif a.cmd == "next-best":
            cmd_next_best(led, a)
        elif a.cmd == "status":
            cmd_status(led, a)
        return 0
    finally:
        release_lock(lock_fd)


if __name__ == "__main__":
    sys.exit(main())
