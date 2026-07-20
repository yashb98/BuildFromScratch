#!/usr/bin/env python3
"""Prepare the GRPO TRAINING prompt+answer set (P1 of research/rlvr/plan.md §3, unlocked by
the Phase-1 GO on 2026-07-02): verifiable-reward math prompts for the exploratory Dr.GRPO arm
and its iso-compute RFT + random-reward controls.

Sets (→ {prompt, gold} JSONL, same template + extractor as math-eval-v1 / math-acc-v1):
  - GSM8K TRAIN (openai/gsm8k main:train, 7,473) — gold = number after '####'.
  - MATH train levels 1-3 (EleutherAI/hendrycks_math, all subjects) — gold = last \\boxed{}
    of the reference solution via the PINNED math-acc-v1 extractor. Skipped with a recorded
    reason if the dataset is unavailable (GSM8K-only v1 is still runnable).

DECONTAMINATION (direction matters for a TRAINING set):
  - vs the EVAL sets (math-eval-v1 gsm8k_test + math500): 13-gram overlap > 0.5 → DROPPED
    from *_clean.jsonl (training on an eval neighbor corrupts the decision metric).
  - vs the SFT problems (OpenR1 used uuids): FLAGGED ONLY (`sft_overlap`) — seeing SFT data
    again in RL is data reuse, not eval contamination; recorded for attribution honesty.

CPU-only, no torch. Run: python3 research/datasets/grpo-math-prompts-v1/prepare_grpo_prompts.py
Smoke: --limit 20.
"""
from __future__ import annotations
import argparse, json, os, sys, pathlib, re

ROOT = pathlib.Path("/home/yashb98/Downloads/BuildFromScratch")
sys.path.insert(0, str(ROOT))
from research.eval_metrics import build_ngram_index, ngram_contamination
from research.eval_math_acc import _last_boxed, EXTRACTOR_VERSION

HF_CACHE = "/home/yashb98/projects/qwen-distill/hf_cache"
OUT = ROOT / "research/datasets/grpo-math-prompts-v1"
EVAL_DIR = ROOT / "research/datasets/math-eval-v1"
SFT_META = ROOT / "research/datasets/math-reasoning-openr1-math-220k/train_meta.jsonl"
NGRAM_N, FLAG_THRESHOLD = 13, 0.5
PROMPT_TMPL = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_LEVEL = re.compile(r"Level (\d)")


def gsm8k_gold(answer: str) -> str:
    tail = answer.rsplit("####", 1)[-1]
    m = _NUM.search(tail)
    return (m.group(0).replace(",", "") if m else tail.strip())


def load_gsm8k_train(limit=None):
    os.environ["HF_HOME"] = HF_CACHE
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [{"prompt": PROMPT_TMPL.format(q=r["question"]), "question": r["question"],
             "gold": gsm8k_gold(r["answer"]), "source": "openai/gsm8k:train"} for r in ds]


def load_math_l13(limit=None):
    """MATH train levels 1-3, all subjects; gold via the pinned math-acc-v1 boxed extractor.
    Returns (items, err) — err recorded if unavailable (GSM8K-only v1 still valid)."""
    os.environ["HF_HOME"] = HF_CACHE
    from datasets import load_dataset
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    items = []
    try:
        for sub in subjects:
            ds = load_dataset("EleutherAI/hendrycks_math", sub, split="train",
                              trust_remote_code=False)
            for r in ds:
                m = _LEVEL.search(r.get("level") or "")
                if not m or int(m.group(1)) > 3:
                    continue
                gold = _last_boxed(r.get("solution") or "")
                if not gold:
                    continue
                items.append({"prompt": PROMPT_TMPL.format(q=r["problem"]),
                              "question": r["problem"], "gold": gold,
                              "level": int(m.group(1)), "subject": sub,
                              "source": f"EleutherAI/hendrycks_math/{sub}:train"})
            if limit and len(items) >= limit:
                return items[:limit], None
        return items, None
    except Exception as e:
        return items, f"{type(e).__name__}: {str(e)[:140]}"


def overlap_tag(items, index, key):
    _, ov = ngram_contamination([it["question"] for it in items], index, NGRAM_N, FLAG_THRESHOLD)
    n = 0
    for it, o in zip(items, ov):
        it[key] = round(o, 4)
        n += o > FLAG_THRESHOLD
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "prep.log", "w")
    def say(*a): print(*a, flush=True); print(*a, file=log)

    say(f"[1/5] GSM8K train{' SMOKE' if args.limit else ''}")
    gsm = load_gsm8k_train(args.limit)
    say(f"      gsm8k_train={len(gsm)}")

    say("[2/5] MATH train levels 1-3 (hendrycks_math)")
    math_items, math_err = load_math_l13(args.limit)
    say(f"      math_l13={len(math_items)}" + (f"  (ERR: {math_err})" if math_err else ""))

    say("[3/5] eval-set decontam index (math-eval-v1 gsm8k_test + math500 questions)")
    eval_qs = []
    for f in ("gsm8k_test.jsonl", "math500.jsonl"):
        for line in open(EVAL_DIR / f):
            r = json.loads(line)
            eval_qs.append(r["prompt"].split("\n\nPlease reason", 1)[0])
    eval_idx = build_ngram_index(eval_qs, NGRAM_N)
    say(f"      eval questions={len(eval_qs)}  index_13grams={len(eval_idx):,}")
    n_gsm_drop = overlap_tag(gsm, eval_idx, "eval_overlap")
    n_math_drop = overlap_tag(math_items, eval_idx, "eval_overlap") if math_items else 0
    say(f"      eval-overlap DROPS: gsm8k={n_gsm_drop}  math_l13={n_math_drop}")

    say("[4/5] vs-SFT flag (OpenR1 used problems; info-only)")
    sft_status = "done"
    try:
        os.environ["HF_HOME"] = HF_CACHE
        from datasets import load_dataset
        used = {json.loads(l)["source_uuid"] for l in open(SFT_META)}
        ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
        ds = ds.select_columns(["uuid", "problem"])
        sft_idx = build_ngram_index([r["problem"] for r in ds if r["uuid"] in used and r["problem"]], NGRAM_N)
        n_gsm_sft = overlap_tag(gsm, sft_idx, "sft_overlap")
        n_math_sft = overlap_tag(math_items, sft_idx, "sft_overlap") if math_items else 0
        say(f"      sft-overlap FLAGS (kept): gsm8k={n_gsm_sft}  math_l13={n_math_sft}")
    except Exception as e:
        sft_status = f"PENDING ({type(e).__name__}: {str(e)[:100]})"
        n_gsm_sft = n_math_sft = None
        say(f"      vs-SFT flag unavailable: {sft_status}")

    say(f"[5/5] writing {OUT}")
    def clean(items):
        return [x for x in items if x["eval_overlap"] <= FLAG_THRESHOLD]
    keys = ["prompt", "gold", "source", "eval_overlap", "sft_overlap", "level", "subject"]
    def w(path, items):
        with open(path, "w") as f:
            for it in items:
                f.write(json.dumps({k: it[k] for k in keys if k in it}) + "\n")
    w(OUT / "gsm8k_train.jsonl", gsm)
    w(OUT / "gsm8k_train_clean.jsonl", clean(gsm))
    if math_items:
        w(OUT / "math_l13_train.jsonl", math_items)
        w(OUT / "math_l13_train_clean.jsonl", clean(math_items))
    report = {
        "prepared_for": "research/rlvr/plan.md P1 (GRPO prompt set) — unlocked by Phase-1 GO 2026-07-02",
        "extractor": EXTRACTOR_VERSION, "ngram_n": NGRAM_N, "flag_threshold": FLAG_THRESHOLD,
        "eval_decontam": "vs math-eval-v1 (gsm8k_test+math500) — overlaps DROPPED from *_clean",
        "sft_flag_status": sft_status,
        "gsm8k_train": {"n": len(gsm), "eval_dropped": n_gsm_drop, "clean": len(clean(gsm)),
                         "sft_flagged": n_gsm_sft},
        "math_l13_train": {"n": len(math_items), "eval_dropped": n_math_drop,
                            "clean": len(clean(math_items)), "sft_flagged": n_math_sft,
                            "load_error": math_err},
        "smoke_limit": args.limit,
    }
    json.dump(report, open(OUT / "decontam_report.json", "w"), indent=2)
    say("DONE", json.dumps({k: report[k] for k in ("gsm8k_train", "math_l13_train")}))
    log.close()


if __name__ == "__main__":
    main()
