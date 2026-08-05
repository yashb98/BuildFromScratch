#!/usr/bin/env python3
"""build_private_heldout.py — materialize `bfs-private-v1`, the §C25 base-eval PRIVATE held-out split.

WHY THIS EXISTS
---------------
`research/eval_completeness.py` REGISTRY['base-eval']['required'] lists `private_heldout` as a HARD
item. It is the ONE base-eval HARD item that the 596M Qwen3 study could never close
(`research/eval/base_eval_verdict.md` §4: "private held-out split: ABSENT ... no private-vs-public BPB
divergence flag exists"), and it is the reason that stage is terminally capped below `win`. The
HybridSSM-0.2B base-eval stage would inherit exactly the same cap unless a private split exists BEFORE
the scoring pass is proposed. This builds one from data already on disk, on CPU, with no model load.

WHAT "PRIVATE" MEANS HERE (the unseen-ness argument, stated so it can be attacked)
----------------------------------------------------------------------------------
The training corpus of every HybridSSM checkpoint is a Qwen3-tokenized cache of
`HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streamed with NO pinned revision
(`Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py`, function
`stream_tokens`, the `load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train",
streaming=True)` call — there is no `revision=` kwarg). FineWeb-Edu is CommonCrawl-derived; the HF
dataset page lists dumps CC-MAIN-2013-20 ... CC-MAIN-2025-26 (fetched 2026-07-29).

This corpus is built ONLY from files authored inside this repository. The repository's first commit is
`84a96c0`, dated **2026-05-20** (`git log --reverse`). Every source file therefore postdates the
newest crawl the training dataset family can contain by >= 11 months, and none of it has ever been
published (the repo is local; no remote push of these paths). That TEMPORAL argument is the whole
basis of unseen-ness. It is NOT backed by a lexical overlap check against the training text, and
cannot be: the tokcache stores token ids only, and the stream was unpinned, so the exact documents the
model trained on are not recoverable from disk. Stated as a limitation, not hidden.

WHAT IT IS NOT
--------------
* Not a contamination PROOF. It is a divergence PROBE: BPB(private) vs BPB(public) is only
  interpretable next to an in-distribution control (the FineWeb-Edu `val` split inside the tokcache),
  because private-vs-public also differs by DOMAIN, not just by seen-ness.
* Not human-web text. Much of this repo's prose and code was written by an LLM agent, which plausibly
  sits closer to a language model's own output distribution than novel human text does, and may
  therefore score a LOWER BPB than a truly novel corpus would. Carried into the result doc.
* Not permanent. If this repository is ever published (the KaggleLab / GitHub-Pages publishing arm
  exists), `bfs-private-v1` is BURNED and must be rotated. The selection uses a seeded subset, so the
  unselected complement is reserved as the v2 pool. See MANIFEST.json -> burn_conditions.

FREEZE SEMANTICS (the thing that is easy to get wrong)
------------------------------------------------------
**The materialized shard IS the corpus; its sha256 IS the version.** The per-source-file hashes in
MANIFEST.json are PROVENANCE, not a liveness contract. This matters because this repository is a live
working tree with CONCURRENT writers (multiple agent sessions edit it while a trainer runs) — a source
file changing after the build is EXPECTED and does NOT invalidate an already-materialized held-out set.
Verification therefore has two levels, and conflating them would either cry wolf or hide a real problem:

  * SHARD drift  -> FATAL. The corpus text itself changed; every BPB number scored against it is void.
  * SOURCE drift -> INFORMATIONAL. The repo moved on since the build. The corpus is unaffected.
                    `--strict-sources` promotes it to a failure for the case where you are trying to
                    reproduce the build byte-for-byte from an unchanged tree.

A REBUILD is not idempotent across time on a live tree (the eligible file pool grows as sessions add
files). Build ONCE, pin, and treat the shard as immutable; to make a new corpus, bump SALT/CORPUS_ID.

DETERMINISM
-----------
Given a FIXED tree, selection is fully deterministic: a salted-hash order over an explicit
include/exclude rule set, then a whole-file greedy fill to a fixed byte cap.

USAGE (pure CPU, no GPU, no network, no model, no tokenizer — safe beside a live trainer)
    python3 build_private_heldout.py                       # build shards + MANIFEST.json
    python3 build_private_heldout.py --verify              # shard integrity (fatal) + source-drift report
    python3 build_private_heldout.py --verify --strict-sources   # also fail on source drift
    python3 build_private_heldout.py --stats               # dry-run: what would be selected, no writes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]                                  # research/eval/private_heldout_v1 -> repo root
SALT = "bfs-private-heldout-v1"                         # changing this changes the split => new version
CORPUS_ID = "bfs-private-v1"

# Per-shard byte cap. The eval geometry that consumes this (eval_suite_jax: SEQ=1024, STRIDE=512,
# MAX_WINDOWS=200) touches only the first (200-1)*512 + 1024 = 102,912 tokens, i.e. roughly 0.4 MB of
# English. 1.0 MB gives ~2.5x margin AND leaves the unselected complement as the v2 rotation pool.
SHARD_BYTES = 1_000_000
MIN_FILE_BYTES = 1_000        # skip stubs (a 200-byte README adds noise, not signal)
MAX_FILE_BYTES = 150_000      # skip giants so no single document dominates a 1 MB shard

# ── selection rules (explicit, auditable, mirrored verbatim into MANIFEST.json) ──────────────────
EXCLUDE_DIR_PARTS = {
    ".git",                 # object store, not authored text
    ".claude",              # local-only agent state; skill files have mixed/templated provenance
    "__pycache__",
    "skills_showcase",      # GENERATED from the skills (research/build_skills_page.py), not authored
    "node_modules",
    ".ipynb_checkpoints",
    "private_heldout_v1",   # this directory — self-reference would make the manifest unstable
}
# model OUTPUT, never model INPUT: generations.md files contain sampled text from checkpoints trained
# on the very corpus we are probing, plus the suite's public generation prompts verbatim.
EXCLUDE_BASENAMES = {"generations.md"}
EXCLUDE_REPO_SUFFIXES = ("plots/README.md",)            # auto-generated plot indexes, near-duplicate boilerplate
EXCLUDE_REPO_PATHS = {"research/eval/base_eval_hybridssm.md"}  # the deliverable that cites this corpus
# HARVESTED-from-the-public-web content. These directories are the ONE place in the repo where the text
# is not authored here: community-pulse / model-radar record fetched paper titles, abstract fragments and
# post text, and ml-research briefs quote recipes out of source papers. Small in absolute bytes, but a
# corpus whose whole claim is "never crawled" must not carry crawled fragments. Excluded wholesale.
EXCLUDE_REPO_PREFIXES = ("research/pulse/", "research/radar/", "research/briefs/")

SHARDS = {
    # shard -> (extension, output filename, the PUBLIC corpus it is the domain-paired counterpart of)
    "private_prose": (".md",  "private_prose_v1.txt", "wikitext2_val"),
    "private_code":  (".py",  "private_code_v1.txt",  "code_py"),
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sort_key(repo_path: str) -> str:
    """Deterministic, salt-dependent, path-only ordering. Path-only (not content) so the ORDER is
    stable when a file's body is edited — only its pinned hash changes, which --verify then catches."""
    return hashlib.sha256(f"{SALT}\x00{repo_path}".encode("utf-8")).hexdigest()


def excluded(p: pathlib.Path, repo_path: str) -> str | None:
    """Return the exclusion reason, or None if the file is eligible."""
    parts = set(p.parts)
    hit = parts & EXCLUDE_DIR_PARTS
    if hit:
        return f"dir:{sorted(hit)[0]}"
    if p.name in EXCLUDE_BASENAMES:
        return f"basename:{p.name}"
    if any(repo_path.endswith(s) for s in EXCLUDE_REPO_SUFFIXES):
        return "suffix:plots/README.md"
    if repo_path in EXCLUDE_REPO_PATHS:
        return "path:self-reference"
    for pre in EXCLUDE_REPO_PREFIXES:
        if repo_path.startswith(pre):
            return f"prefix:{pre} (web-harvested)"
    return None


def candidates(ext: str):
    """Eligible files for one shard, in deterministic salted order, with size gates applied."""
    out, skipped = [], {"excluded": 0, "too_small": 0, "too_big": 0}
    for p in sorted(REPO.rglob(f"*{ext}")):
        if not p.is_file():
            continue
        repo_path = p.relative_to(REPO).as_posix()
        if excluded(p, repo_path):
            skipped["excluded"] += 1
            continue
        n = p.stat().st_size
        if n < MIN_FILE_BYTES:
            skipped["too_small"] += 1
            continue
        if n > MAX_FILE_BYTES:
            skipped["too_big"] += 1
            continue
        out.append(repo_path)
    out.sort(key=sort_key)
    return out, skipped


def build_shard(ext: str, cap: int):
    """Greedy whole-file fill to `cap` bytes. Documents are joined with "\\n\\n", the SAME join the
    suite uses to assemble wikitext-2 (eval_suite_jax.load_corpora: '\\n\\n'.join(ex['text'] ...)),
    so the two corpora differ in content, not in assembly convention."""
    picked, files, total = [], [], 0
    pool, skipped = candidates(ext)
    for repo_path in pool:
        raw = (REPO / repo_path).read_bytes()
        # decode strictly: a file that is not valid UTF-8 is not text we want in a BPB denominator
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        add = len(raw) + (2 if picked else 0)            # the "\n\n" separator costs 2 bytes
        if total + add > cap:
            continue                                     # skip, don't stop: keeps the fill tight+deterministic
        picked.append(text)
        files.append({"path": repo_path, "bytes": len(raw), "sha256": sha256_bytes(raw)})
        total += add
    return "\n\n".join(picked), files, skipped, len(pool)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="shard integrity (FATAL on mismatch) + a source-provenance drift report")
    ap.add_argument("--strict-sources", action="store_true",
                    help="with --verify: also exit non-zero if any pinned SOURCE file drifted")
    ap.add_argument("--stats", action="store_true", help="dry run: report selection, write nothing")
    a = ap.parse_args()
    man_path = HERE / "MANIFEST.json"

    if a.verify:
        man = json.loads(man_path.read_text())
        fatal, drift = [], []
        for shard, s in man["shards"].items():
            out = HERE / s["file"]                       # ── level 1: the corpus itself (FATAL) ──
            if not out.exists():
                fatal.append(f"{shard}: shard file MISSING {s['file']}")
            elif sha256_bytes(out.read_bytes()) != s["sha256"]:
                fatal.append(f"{shard}: shard file DRIFTED {s['file']} — every BPB scored on it is void")
            for f in s["files"]:                         # ── level 2: provenance (informational) ──
                p = REPO / f["path"]
                if not p.exists():
                    drift.append(f"{shard}: source GONE     {f['path']}")
                elif sha256_bytes(p.read_bytes()) != f["sha256"]:
                    drift.append(f"{shard}: source CHANGED  {f['path']}")
        n = sum(len(s["files"]) for s in man["shards"].values())
        if fatal:
            print(f"VERIFY FAIL — corpus integrity ({len(fatal)}):")
            for b in fatal:
                print("  " + b)
            raise SystemExit(1)
        print(f"VERIFY PASS — corpus intact: {len(man['shards'])} shards match their pinned sha256")
        if drift:
            print(f"  source-provenance drift: {len(drift)}/{n} pinned files changed since the build "
                  f"(EXPECTED on a live tree with concurrent writers; the corpus is unaffected)")
            for d in drift[:20]:
                print("    " + d)
            if len(drift) > 20:
                print(f"    ... and {len(drift)-20} more")
            if a.strict_sources:
                raise SystemExit(2)
        else:
            print(f"  source-provenance: all {n} pinned files unchanged since the build")
        return

    shards, report = {}, {}
    for shard, (ext, fname, paired) in SHARDS.items():
        text, files, skipped, pool_n = build_shard(ext, SHARD_BYTES)
        blob = text.encode("utf-8")
        report[shard] = {"pool": pool_n, "picked": len(files), "bytes": len(blob), "skipped": skipped}
        shards[shard] = {
            "file": fname, "ext": ext, "paired_public_corpus": paired,
            "bytes": len(blob), "sha256": sha256_bytes(blob), "n_source_files": len(files),
            "pool_size": pool_n, "files": files,
        }
        if not a.stats:
            (HERE / fname).write_bytes(blob)

    if a.stats:
        print(json.dumps(report, indent=2))
        return

    manifest = {
        "corpus_id": CORPUS_ID,
        "schema_version": 1,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "built_by": "research/eval/private_heldout_v1/build_private_heldout.py",
        "git_commit": git_commit(),
        "c25_item": "private_heldout (research/eval_completeness.py REGISTRY['base-eval']['required'])",
        "salt": SALT,
        "freeze": {
            "the_corpus_is": "the materialized shard files; their sha256 is the version",
            "per_file_hashes_are": "PROVENANCE of what the shard was built from — not a liveness contract",
            "shard_drift": "FATAL — the corpus text changed; every BPB scored against it is void",
            "source_drift": ("INFORMATIONAL — this repository is a live working tree with concurrent agent "
                             "sessions writing to it, so pinned source files WILL change after the build. "
                             "That does not alter or invalidate an already-materialized shard."),
            "rebuild_is_not_idempotent_across_time": ("the eligible file pool grows as sessions add files, so "
                                                      "re-running the builder on a later tree yields a DIFFERENT "
                                                      "corpus. Build once, pin, treat as immutable; to make a new "
                                                      "corpus bump SALT + CORPUS_ID."),
        },
        "selection": {
            "root": "repository root (BuildFromScratch/)",
            "order": "sha256(SALT + '\\x00' + repo_relative_path), ascending — deterministic, content-independent",
            "fill": f"greedy whole-file to {SHARD_BYTES} bytes/shard; a file that would overflow is SKIPPED, not truncated",
            "join": "'\\n\\n' — identical to eval_suite_jax.load_corpora's wikitext assembly",
            "min_file_bytes": MIN_FILE_BYTES,
            "max_file_bytes": MAX_FILE_BYTES,
            "exclude_dir_parts": sorted(EXCLUDE_DIR_PARTS),
            "exclude_basenames": sorted(EXCLUDE_BASENAMES),
            "exclude_repo_suffixes": list(EXCLUDE_REPO_SUFFIXES),
            "exclude_repo_paths": sorted(EXCLUDE_REPO_PATHS),
            "exclude_repo_prefixes_web_harvested": list(EXCLUDE_REPO_PREFIXES),
            "encoding": "utf-8 strict; non-decodable files dropped",
        },
        "unseen_argument": {
            "type": "TEMPORAL (not lexical)",
            "repo_first_commit": "84a96c0 2026-05-20",
            "training_corpus": "HuggingFaceFW/fineweb-edu config sample-10BT, split train, streaming, revision UNPINNED",
            "training_corpus_source": ("Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/"
                                       "train_qwen3.py :: stream_tokens"),
            "crawl_range_fetched_2026-07-29": "CC-MAIN-2013-20 .. CC-MAIN-2025-26 (huggingface.co dataset page subset list)",
            "margin": "every source file postdates the newest possible crawl by >= 11 months",
            "not_verified": ("no lexical overlap check against the training text is possible — the tokcache "
                             "stores token ids only and the stream was unpinned, so the trained-on documents "
                             "are not recoverable from disk"),
        },
        "limitations": [
            "Much of this text was authored by an LLM agent; it may sit closer to a language model's own "
            "output distribution than novel human text, plausibly biasing BPB DOWN.",
            "Domain-shifted vs both public corpora (ML-research prose vs encyclopedic Wikipedia; "
            "research//experiment Python vs GitHub-general Python). A private-vs-public BPB gap is therefore "
            "NOT by itself a contamination signal — it must be read against the in-distribution control "
            "(the FineWeb-Edu `val` split inside the tokcache).",
            "Python sources include from-scratch reimplementations of published architectures (Qwen3, "
            "SmolLM2). They were written from a blank file, not copied, but they are structurally similar to "
            "public reference implementations.",
            "n=1 corpus: there is no second independent private set, so a private-side outlier cannot be "
            "cross-checked.",
        ],
        "burn_conditions": [
            "Publishing this repository (the KaggleLab / GitHub-Pages arm) makes bfs-private-v1 PUBLIC and "
            "voids the unseen-ness argument for every future model.",
            "Training any model on repository text.",
            "Any drift reported by `--verify` (a pinned source file edited after the build).",
            "Rotation pool for v2 = the unselected complement (pool_size - n_source_files per shard); bump "
            "SALT and CORPUS_ID rather than re-running v1.",
        ],
        "shards": shards,
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for shard, s in shards.items():
        print(f"[{shard}] {s['file']}: {s['bytes']:,} B from {s['n_source_files']} files "
              f"(pool {s['pool_size']}) sha256={s['sha256'][:16]}")
    print(f"[manifest] {man_path}")


if __name__ == "__main__":
    main()
