"""Prepare the dclm-edu TREATMENT slice for the data-selection A/B (vs the FineWeb-Edu control).

WHY direct-download not streaming: HF streaming of these huge web-dataset parquet shards is
network-throttled to ~20 tok/s (range-request overhead); a direct shard download is ~55 MB/s.
So we hf_hub_download whole parquet shards, then tokenize LOCALLY (fast, bounded RAM via pyarrow
batches). Token budget ~150M, matched to the de-confound 2000-step proxy so the data A/B is a
fast, directly-comparable cell.

Outputs (dataset-forge §R4/R5 shape): uint32 train shards + a doc-level held-out eval split +
13-gram decontam of TRAIN vs the STANDING OOD eval corpora (wikitext2_val + code_py) so the A/B's
OOD-BPB judgment can never be contaminated by benchmark text leaking into training. Model's own
Qwen tokenizer (vocab 151,936 -> uint32). trust_remote_code never used (raw parquet).

Usage:  python prepare_dclm_edu.py [--target-tokens 150_000_000] [--limit-tokens N (bounded test)]
"""
from __future__ import annotations
import sys, os, json, time, argparse, pathlib, hashlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                  # BuildFromScratch/
sys.path.insert(0, str(ROOT))
import numpy as np

REPO, SHA = "HuggingFaceTB/dclm-edu", "dbad8ad71224482740cd9c9d353591adbf62fe04"
TOK_REPO = "Qwen/Qwen3-0.6B-Base"
SHARD_TOKENS = 50_000_000
EVAL_FRAC_DENOM = 200          # ~0.5% docs held out
DECONTAM_NGRAM = 13
DECONTAM_JACCARD = 0.8         # finance contracts §6
EOS = None                     # set from tokenizer


def word_ngrams(text, n=DECONTAM_NGRAM):
    w = text.split()
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)} if len(w) >= n else set()


def build_decontam_index():
    """13-gram hash set from the standing OOD eval corpora — small, bounded."""
    from datasets import load_dataset
    idx = set()
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation",
                      revision="b08601e04326c79dfdd32d625aee71d232d685c3")
    idx |= word_ngrams("\n\n".join(e["text"] for e in wt if e["text"].strip()))
    try:
        cp = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True,
                          revision="4db92d2ec0c1b4c41eeb439cfae16854511d9dcd")
        buf, tot = [], 0
        for e in cp:
            buf.append(e["content"]); tot += len(e["content"])
            if tot > 500_000:
                break
        idx |= word_ngrams("\n\n".join(buf))
    except Exception as e:
        print(f"  [decontam] code corpus skipped: {str(e)[:60]}")
    return idx


def is_contaminated(text, idx):
    g = word_ngrams(text)
    if not g:
        return False
    return len(g & idx) / len(g) > DECONTAM_JACCARD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=150_000_000)
    ap.add_argument("--limit-tokens", type=int, default=0)   # >0 = bounded test
    args = ap.parse_args()
    target = args.limit_tokens or args.target_tokens
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    tok = AutoTokenizer.from_pretrained(TOK_REPO)
    global EOS; EOS = tok.eos_token_id

    print("building decontam index (wikitext2 + code)...", flush=True)
    didx = build_decontam_index()
    print(f"  decontam 13-grams: {len(didx):,}", flush=True)

    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset", revision=SHA) if f.endswith(".parquet")]
    print(f"{len(files)} parquet shards; will pull as needed for {target:,} tokens", flush=True)

    train_buf, eval_toks = [], []
    shard_i, train_tokens, eval_tokens, n_docs, n_dropped = 0, 0, 0, 0, 0
    t0 = time.time()

    def flush_train():
        nonlocal shard_i
        if not train_buf:
            return
        arr = np.array(train_buf, dtype=np.uint32)
        arr.tofile(HERE / f"shard_{shard_i:05d}.bin")
        print(f"  wrote shard_{shard_i:05d}.bin ({len(arr):,} tok)", flush=True)
        shard_i += 1; train_buf.clear()

    for fi, fname in enumerate(files):
        if train_tokens >= target:
            break
        local = hf_hub_download(REPO, fname, repo_type="dataset", revision=SHA, local_dir=str(HERE / ".raw"))
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if not text or not text.strip():
                    continue
                n_docs += 1
                if is_contaminated(text, didx):
                    n_dropped += 1; continue
                ids = tok.encode(text) + [EOS]
                if hashlib.md5(text[:200].encode()).hexdigest().__hash__() % EVAL_FRAC_DENOM == 0:
                    eval_toks.extend(ids); eval_tokens += len(ids)
                else:
                    train_buf.extend(ids); train_tokens += len(ids)
                    if len(train_buf) >= SHARD_TOKENS:
                        flush_train()
            if train_tokens >= target:
                break
        os.remove(local)   # free disk after each shard
        print(f"  shard {fi}: train={train_tokens:,} eval={eval_tokens:,} dropped={n_dropped} "
              f"({(train_tokens)/(time.time()-t0):,.0f} tok/s)", flush=True)

    flush_train()
    np.array(eval_toks, dtype=np.uint32).tofile(HERE / "eval_tokens.bin")
    meta = {"dataset": REPO, "revision": SHA, "tokenizer_repo": TOK_REPO, "dtype": "uint32",
            "vocab_size": len(tok), "eos_id": EOS, "shards": shard_i}
    (HERE / "meta.json").write_text(json.dumps(meta, indent=2))
    stats = {"train_tokens": train_tokens, "eval_tokens": eval_tokens, "shards": shard_i,
             "n_docs": n_docs, "docs_dropped_decontam": n_dropped,
             "decontam_jaccard": DECONTAM_JACCARD, "wall_clock_s": round(time.time() - t0),
             "sample_decode": tok.decode(train_buf[:40]) if train_buf else "(flushed)"}
    (HERE / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"DONE -> {train_tokens:,} train + {eval_tokens:,} eval tokens, {shard_i} shards, "
          f"{n_dropped} docs decontam-dropped, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
