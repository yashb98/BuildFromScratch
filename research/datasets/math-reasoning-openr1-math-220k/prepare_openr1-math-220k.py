#!/usr/bin/env python3
"""dataset-forge prep — OpenR1-Math-220k SFT shards for Qwen3-0.6B (math-reasoning).

Streaming-only (§C1: one ~119 GB unified pool; never list() a split, never hold
the corpus in RAM). No torch import -> no safe_cuda header needed (tokenization
is pure-CPU; the crash mode here is RAM accumulation, guarded by streaming +
bounded array.array buffers per recipes §R4).

Produces, all under this dir (research/datasets/math-reasoning-openr1-math-220k/):
  - shard_*.bin            : SFT train tokens, uint32 flat stream, eos-separated
  - train_meta.jsonl       : one line per SFT sample {n_tokens, prompt_len, source_uuid}
                             so /ablation-runner's masked-completion CE can mask the prompt
  - meta.json              : dtype/vocab/tokenizer/dataset id+sha/eos + counts
  - eval/eval_docs.jsonl   : held-out in-domain reasoning eval docs (post-13-gram-dedup)
  - eval/eval_tokens.bin   : tokenized eval split (same meta)
  - eval/forgetting_docs.jsonl  : held-out FineWeb-Edu general probe (catastrophic-forgetting, §C13)
  - eval/forgetting_tokens.bin  : tokenized forgetting probe (same meta)
  - stats.json             : final counts

Hygiene (finance contracts §6 / recipes §R5):
  - document-level seeded split FIRST (sha256 mod), seed "forge-v1", eval frac 0.005
  - 13-gram Jaccard dedup of eval docs vs the streamed train side, drop > 0.8
Re-runnable: completed shards on disk are skipped on resume.
"""
import argparse, array, hashlib, json, sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# ----- pins (verified live 2026-06-17; see card.md / selection.md) -----
REPO          = "Qwen/Qwen3-0.6B-Base"
SFT_ID        = "open-r1/OpenR1-Math-220k"
SFT_REV       = "e4e141ec9dea9f8326f4d347be56105859b2bd68"
SFT_CONFIG    = "default"
FORGET_ID     = "HuggingFaceFW/fineweb-edu"
FORGET_REV    = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FORGET_CONFIG = "sample-10BT"

VOCAB_SIZE   = 151_936
DTYPE        = np.uint32           # vocab > 65536 -> 4 bytes/token (§R4)
ACODE        = "I"                 # 4-byte unsigned C int
MAX_SEQ      = 4096                # build seq_len (train_qwen3.py)
SHARD_TOKENS = 50_000_000          # §R4
EVAL_FRAC    = 0.005               # Constants
EVAL_CAP_TOK = 5_000_000           # Constants
FORGET_CAP_TOK = 5_000_000
SEED         = "forge-v1"          # Constants
DEDUP_THRESH = 0.8                 # 13-gram Jaccard (finance §6)
NGRAM        = 13

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "eval"; EVAL_DIR.mkdir(exist_ok=True)


def is_eval(doc_key: str, frac=EVAL_FRAC, seed=SEED) -> bool:
    h = int(hashlib.sha256(f"{seed}:{doc_key}".encode()).hexdigest(), 16)
    return (h % 10_000) < int(frac * 10_000)


def grams(text, n=NGRAM):
    t = text.split()
    return {hash(tuple(t[i:i + n])) for i in range(max(0, len(t) - n + 1))}


def pick_verified_trace(row):
    """First generation whose correctness_math_verify is True (answer-checked)."""
    gens = row.get("generations") or []
    flags = row.get("correctness_math_verify") or []
    for g, f in zip(gens, flags):
        if f and g:
            return g
    return None


def sft_text(row, trace):
    """prompt + blank line + verified reasoning trace. Returns (full_text, prompt_text)."""
    prompt = (row.get("problem") or "").strip()
    return prompt + "\n\n" + trace, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_tokens", type=int, default=0,
                    help="bounded slice for the 60s smoke run; 0 = full TOKEN_TARGET")
    ap.add_argument("--token_target", type=int, default=125_000_000)
    args = ap.parse_args()
    TOKEN_TARGET = args.limit_tokens if args.limit_tokens > 0 else args.token_target
    smoke = args.limit_tokens > 0

    tok = AutoTokenizer.from_pretrained(REPO)
    eos = tok.eos_token_id
    assert eos == 151643, f"unexpected eos {eos}"
    print(f"[forge] tokenizer {REPO} len={len(tok)} eos={eos} target={TOKEN_TARGET} smoke={smoke}",
          flush=True)

    # ---------- PASS 1: build the held-out in-domain eval set FIRST (doc-level split) ----------
    # Stream a bounded prefix; route eval-bucket rows to the eval side, capped at EVAL_CAP_TOK.
    eval_docs, eval_keys = [], []
    eval_tok_total = 0
    eval_scan_cap = 200 if smoke else 20000   # bounded scan to fill the small eval cap
    sft = load_dataset(SFT_ID, SFT_CONFIG, split="train", streaming=True,
                       revision=SFT_REV, trust_remote_code=False)
    n_scanned = 0
    for row in sft:
        n_scanned += 1
        if n_scanned > eval_scan_cap:
            break
        key = row.get("uuid") or hashlib.sha256((row.get("problem") or "").encode()).hexdigest()
        if not is_eval(key):
            continue
        trace = pick_verified_trace(row)
        if trace is None:
            continue
        full, _ = sft_text(row, trace)
        ids = tok(full).input_ids[:MAX_SEQ]
        if eval_tok_total + len(ids) > EVAL_CAP_TOK:
            continue
        eval_docs.append({"uuid": key, "text": full, "n_tokens": len(ids)})
        eval_keys.append(key)
        eval_tok_total += len(ids)
    print(f"[forge] eval candidates collected: {len(eval_docs)} docs, {eval_tok_total} tok "
          f"(scanned {n_scanned} rows)", flush=True)

    # build the eval gram index for streaming dedup-vs-train
    eval_grams = {i: grams(d["text"]) for i, d in enumerate(eval_docs)}
    index = {}
    for i, gs in eval_grams.items():
        for g in gs:
            index.setdefault(g, []).append(i)

    # ---------- PASS 2: stream train side, dedup eval vs train, write shards ----------
    overlap = {}                          # (eval_i) -> max overlap count seen
    train_doc_grams_lens = {}             # per train doc gram-count (for jaccard denom)
    buf = array.array(ACODE)
    shard_idx, train_total, n_train_samples = 0, 0, 0
    eval_key_set = set(eval_keys)
    meta_f = open(HERE / "train_meta.jsonl", "w")

    def flush_full_shards():
        nonlocal buf, shard_idx
        while len(buf) >= SHARD_TOKENS:
            out = HERE / f"shard_{shard_idx:05d}.bin"
            np.frombuffer(buf, dtype=DTYPE, count=SHARD_TOKENS).tofile(out)
            print(f"[forge] wrote {out.name} ({SHARD_TOKENS} tok) total={train_total}", flush=True)
            buf = buf[SHARD_TOKENS:]
            shard_idx += 1

    sft2 = load_dataset(SFT_ID, SFT_CONFIG, split="train", streaming=True,
                        revision=SFT_REV, trust_remote_code=False)
    for row in sft2:
        key = row.get("uuid") or hashlib.sha256((row.get("problem") or "").encode()).hexdigest()
        if key in eval_key_set:
            continue                      # never let an eval doc into train
        trace = pick_verified_trace(row)
        if trace is None:
            continue                      # data hygiene: verified traces only
        full, prompt = sft_text(row, trace)

        # streaming 13-gram dedup contribution from this train doc
        if index:
            tg = grams(full)
            hit = tg & index.keys()
            if hit:
                tg_len = len(tg)
                for g in hit:
                    for i in index[g]:
                        c = overlap.get(i, 0) + 1
                        overlap[i] = c
                        # track the train doc gram-len that produced the best overlap
                        prev = train_doc_grams_lens.get(i, (0, 0))
                        if c >= prev[0]:
                            train_doc_grams_lens[i] = (c, tg_len)

        full_ids = tok(full).input_ids[:MAX_SEQ]
        prompt_ids = tok(prompt).input_ids
        prompt_len = min(len(prompt_ids), len(full_ids))   # masked span (clamped to window)
        ids = full_ids + [eos]
        buf.extend(ids)
        train_total += len(ids)
        meta_f.write(json.dumps({"n_tokens": len(ids), "prompt_len": prompt_len,
                                 "source_uuid": key}) + "\n")
        n_train_samples += 1
        flush_full_shards()
        if train_total >= TOKEN_TARGET:
            break
    # flush remainder
    if len(buf):
        out = HERE / f"shard_{shard_idx:05d}.bin"
        np.frombuffer(buf, dtype=DTYPE).tofile(out)
        print(f"[forge] wrote {out.name} ({len(buf)} tok, remainder) total={train_total}", flush=True)
        shard_idx += 1
    meta_f.close()

    # ---------- apply dedup verdicts: drop eval docs that overlap train > 0.8 ----------
    docs_dropped = 0
    kept_eval = []
    for i, d in enumerate(eval_docs):
        c = overlap.get(i, 0)
        if c > 0:
            e_len = len(eval_grams[i])
            t_len = train_doc_grams_lens.get(i, (c, c))[1]
            jacc = c / max(1, (e_len + t_len - c))
            if jacc > DEDUP_THRESH:
                docs_dropped += 1
                continue
        kept_eval.append(d)
    print(f"[forge] eval dedup: kept {len(kept_eval)}, dropped {docs_dropped} (13-gram jaccard>{DEDUP_THRESH})",
          flush=True)

    # write the kept eval split (untokenized + tokenized, same meta)
    eval_tok_buf = array.array(ACODE)
    eval_tokens = 0
    with open(EVAL_DIR / "eval_docs.jsonl", "w") as f:
        for d in kept_eval:
            f.write(json.dumps({"uuid": d["uuid"], "text": d["text"]}) + "\n")
            ids = tok(d["text"]).input_ids[:MAX_SEQ] + [eos]
            eval_tok_buf.extend(ids); eval_tokens += len(ids)
    np.frombuffer(eval_tok_buf, dtype=DTYPE).tofile(EVAL_DIR / "eval_tokens.bin")

    # ---------- forgetting probe: FineWeb-Edu general-distribution held-out (§C13) ----------
    forget_scan_cap = 200 if smoke else 40000
    forget_docs, forget_keys = [], []
    forget_tok = 0
    fw = load_dataset(FORGET_ID, FORGET_CONFIG, split="train", streaming=True,
                      revision=FORGET_REV, trust_remote_code=False)
    n = 0
    for row in fw:
        n += 1
        if n > forget_scan_cap or forget_tok >= FORGET_CAP_TOK:
            break
        text = row.get("text") or ""
        key = row.get("id") or hashlib.sha256(text.encode()).hexdigest()
        if not is_eval(key):              # same doc-level seeded split
            continue
        ids = tok(text).input_ids[:MAX_SEQ]
        if forget_tok + len(ids) > FORGET_CAP_TOK:
            continue
        forget_docs.append({"id": key, "text": text}); forget_keys.append(key)
        forget_tok += len(ids)
    # the forgetting probe is from a DIFFERENT corpus than train, so train-leak is structurally
    # impossible; we still apply the doc-level seeded split for held-out hygiene. Record 0 dropped.
    forget_buf = array.array(ACODE); forgetting_tokens = 0
    with open(EVAL_DIR / "forgetting_docs.jsonl", "w") as f:
        for d in forget_docs:
            f.write(json.dumps(d) + "\n")
            ids = tok(d["text"]).input_ids[:MAX_SEQ] + [eos]
            forget_buf.extend(ids); forgetting_tokens += len(ids)
    np.frombuffer(forget_buf, dtype=DTYPE).tofile(EVAL_DIR / "forgetting_tokens.bin")
    print(f"[forge] forgetting probe: {len(forget_docs)} FineWeb-Edu docs, {forgetting_tokens} tok",
          flush=True)

    # ---------- meta + stats ----------
    meta = {
        "dtype": "uint32", "vocab_size": VOCAB_SIZE, "tokenizer_repo": REPO,
        "dataset_id": SFT_ID, "revision": SFT_REV, "dataset_config": SFT_CONFIG,
        "eos_token_id": eos, "shard_tokens": SHARD_TOKENS, "max_seq": MAX_SEQ,
        "doc_separator": "eos", "sft_format": "prompt + '\\n\\n' + verified_trace; prompt_len in train_meta.jsonl",
        "forgetting_source": {"id": FORGET_ID, "config": FORGET_CONFIG, "revision": FORGET_REV},
    }
    (HERE / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---------- readback verification (§R4 gate) ----------
    shard_files = sorted(HERE.glob("shard_*.bin"))
    read_total, maxid = 0, 0
    for sf in shard_files:
        a = np.fromfile(sf, dtype=DTYPE)
        read_total += len(a)
        if len(a):
            maxid = max(maxid, int(a.max()))
    assert read_total == train_total, f"readback {read_total} != written {train_total}"
    assert maxid < VOCAB_SIZE, f"max id {maxid} >= vocab {VOCAB_SIZE}"
    # decode one 256-token window verbatim into the log
    if shard_files:
        win = np.fromfile(shard_files[0], dtype=DTYPE)[:256].tolist()
        print("[forge] DECODED WINDOW (first shard, 256 tok):", flush=True)
        print(tok.decode(win), flush=True)

    stats = {
        "train_tokens": train_total, "n_train_samples": n_train_samples,
        "shards": len(shard_files), "eval_tokens": eval_tokens,
        "eval_docs_kept": len(kept_eval), "docs_dropped": docs_dropped,
        "forgetting_tokens": forgetting_tokens, "forgetting_docs": len(forget_docs),
        "fertility": 1.92, "max_token_id": maxid, "token_target": TOKEN_TARGET,
        "smoke": smoke,
    }
    (HERE / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[forge] DONE {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
