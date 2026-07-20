#!/usr/bin/env python3
"""data_decontam.py — document-disjoint train/val splitting + n-gram decontamination.

The audit found the build's eval split was the *sequential continuation* of the
train stream with NO independent dedup or benchmark decontamination, so any
reported val PPL/BPB was leak-suspect (an RS rejects it). This module is the
shared, UNIT-TESTED fix the data path calls:

  * is_val_doc(doc, seed, val_fraction) — deterministic, order-independent routing
    of a WHOLE document to train or val by a seeded hash, so train/val are
    DOCUMENT-DISJOINT (a given doc always lands in the same bucket; no doc spans
    the boundary, unlike a token-count cut).
  * decontaminate_val(val_docs, train_docs, n, threshold) — drops val docs whose
    n-gram (default 13) Jaccard-style overlap with the train corpus exceeds
    `threshold`, reusing eval_metrics' contamination primitives.
  * decontam_report(...) — the on-disk evidence (docs_dropped, split_seed, n,
    threshold) the §C8 ledger lineage + the §C22 provenance span point at.

Stdlib-only, pure, deterministic — no torch, no network. The logic is tested so
that even before the (paid) run executes, the decontam guarantee is proven.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eval_metrics import build_ngram_index, ngram_contamination


def _doc_hash01(doc: str, seed: int) -> float:
    """Deterministic uniform-in-[0,1) hash of a document, mixed with `seed`.
    Order-independent: the same document always maps to the same value, so its
    train/val assignment never depends on where it appeared in the stream."""
    h = hashlib.sha256(f"{seed}\x00{doc}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def is_val_doc(doc: str, seed: int, val_fraction: float) -> bool:
    """Route a WHOLE document to val (True) or train (False). ~`val_fraction` of
    documents land in val, disjoint from train, independent of stream order."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1)")
    return _doc_hash01(doc, seed) < val_fraction


def decontaminate_val(val_docs, train_docs, n: int = 13, threshold: float = 0.8):
    """Drop val docs that overlap the train corpus by more than `threshold` of
    their n-grams. Returns (kept_docs, dropped_idx, per_item_overlap). Uses 13-gram
    word overlap by default (the GPT-3-era decontamination convention)."""
    idx = build_ngram_index(train_docs, n)
    _, per_item = ngram_contamination(val_docs, idx, n, threshold=threshold)
    kept, dropped = [], []
    for i, ov in enumerate(per_item):
        (dropped if ov > threshold else kept).append(i)
    kept_docs = [val_docs[i] for i in kept]
    return kept_docs, dropped, per_item


def cell_split_seed(base_seed: int, cell_seed: int, *, randomized: bool) -> int:
    """Derive the *split* seed for one cohort cell from the run's base split seed
    and the cell's training seed.

    randomized=False -> the FIXED split: every cell gets the SAME doc-disjoint
      split (return base_seed), so train/val membership is held constant and the
      only thing the seeds vary is init + data-order shuffle. This is the existing
      behaviour and the right choice when you want to isolate optimization noise.

    randomized=True -> a DIFFERENT doc-disjoint split per cell, derived by hashing
      (base_seed, cell_seed) into a fresh split seed. Each cell then resamples
      which documents are train vs val, so the seed spread captures FULL
      end-to-end variance — corpus resampling included — not just init+shuffle.
      This is the honest variance estimate for an RS 'clean win' (a result that
      survives the train/val partition itself being a random draw).

    Deterministic: same (base_seed, cell_seed, randomized) -> same split seed, so
    a cohort is exactly reproducible from its config."""
    if not randomized:
        return int(base_seed)
    h = hashlib.sha256(f"split\x00{int(base_seed)}\x00{int(cell_seed)}".encode("utf-8")).digest()
    # 63-bit unsigned -> a stable, non-negative split seed distinct per cell.
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def split_and_decontaminate(docs, seed: int, val_fraction: float,
                            n: int = 13, threshold: float = 0.8, *,
                            cell_seed: int | None = None,
                            randomized: bool = False):
    """Convenience: doc-disjoint split, then decontaminate the val side against
    train. Returns (train_docs, kept_val_docs, report_dict).

    By default this is the FIXED-split path (unchanged): the doc-disjoint split is
    keyed on `seed` alone, identical for every call.

    Pass randomized=True together with the cell's training seed (`cell_seed`) to
    take a DIFFERENT doc-disjoint split for this cell — derived via
    cell_split_seed(seed, cell_seed, randomized=True). The report records both the
    base split seed and the effective per-cell split seed so the lineage is exact.
    `randomized=True` with `cell_seed=None` is an error (the cell seed is what
    makes the split differ)."""
    if randomized and cell_seed is None:
        raise ValueError("randomized=True requires a cell_seed to derive the per-cell split")
    eff_seed = cell_split_seed(seed, cell_seed if cell_seed is not None else 0,
                               randomized=randomized)
    train_docs, raw_val = [], []
    for d in docs:
        (raw_val if is_val_doc(d, eff_seed, val_fraction) else train_docs).append(d)
    kept_val, dropped, _ = decontaminate_val(raw_val, train_docs, n, threshold)
    report = decontam_report(eff_seed, val_fraction, n, threshold,
                             n_train_docs=len(train_docs), n_val_raw=len(raw_val),
                             docs_dropped=len(dropped))
    # lineage: distinguish the base split seed from the effective per-cell one
    report["base_split_seed"] = int(seed)
    report["cell_seed"] = None if cell_seed is None else int(cell_seed)
    report["randomized_split"] = bool(randomized)
    if randomized:
        report["method"] = ("randomized per-cell doc-disjoint seeded-hash split "
                            "+ 13-gram word-overlap decontam")
    return train_docs, kept_val, report


def decontam_report(seed, val_fraction, n, threshold, *, n_train_docs,
                    n_val_raw, docs_dropped) -> dict:
    """The on-disk decontamination evidence (write next to the token cache)."""
    return {
        "split_seed": seed,
        "val_fraction": val_fraction,
        "ngram_n": n,
        "overlap_threshold": threshold,
        "n_train_docs": n_train_docs,
        "n_val_docs_raw": n_val_raw,
        "docs_dropped": docs_dropped,
        "n_val_docs_kept": n_val_raw - docs_dropped,
        "method": "doc-disjoint seeded-hash split + 13-gram word-overlap decontam",
    }


def write_decontam_report(path, report: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2) + "\n")
