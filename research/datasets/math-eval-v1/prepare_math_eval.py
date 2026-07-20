#!/usr/bin/env python3
"""Prepare the held-out math-accuracy EVAL band (P2 of research/rlvr/plan.md §C27): the
decision-metric sets for the Phase-1 pass@k go/no-go and every downstream reasoning run.

Two sets → {prompt, gold} JSONL that research.eval_math_acc.run_math_acc consumes directly:
  - GSM8K test  (openai/gsm8k, 1319 items)  — gold = the number after '####'.
  - MATH-500    (HuggingFaceH4/MATH-500, 500 items) — gold = the dataset `answer` field.

DECONTAMINATION (the load-bearing hygiene): both are checked with a 13-gram overlap against
the ACTUAL 35,231 OpenR1-Math-220k problems the model was SFT'd on (identified by the
source_uuids in the prepared SFT set). This matters because OpenR1-Math contains MATH-style
problems, so MATH-500 could overlap the SFT data — scoring on a memorized problem would fake
a reasoning signal. Items over the flag threshold are marked `contaminated: true` and excluded
from the `*_clean.jsonl` the go/no-go scores; the full set + a decontam_report are kept.

CPU-only, no torch (forge convention). Offline-first via the on-box HF cache. Run:
  python3 research/datasets/math-eval-v1/prepare_math_eval.py
Smoke: add --limit 20 (tiny, for a fast wiring check — does NOT gate the frozen sets).
"""
from __future__ import annotations
import argparse, json, os, re, sys, pathlib

ROOT = pathlib.Path("/home/yashb98/Downloads/BuildFromScratch")
sys.path.insert(0, str(ROOT))
from research.eval_metrics import build_ngram_index, ngram_contamination

HF_CACHE = "/home/yashb98/projects/qwen-distill/hf_cache"     # GSM8K + OpenR1 already cached here
OUT = ROOT / "research/datasets/math-eval-v1"
SFT_META = ROOT / "research/datasets/math-reasoning-openr1-math-220k/train_meta.jsonl"
NGRAM_N = 13                       # the plan's 13-gram decontam
FLAG_THRESHOLD = 0.5               # eval item flagged if >50% of its 13-grams appear in SFT
PROMPT_TMPL = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm8k_gold(answer: str) -> str:
    tail = answer.rsplit("####", 1)[-1]
    m = _NUM.search(tail)
    return (m.group(0).replace(",", "") if m else tail.strip())


def load_sets(limit=None):
    os.environ["HF_HOME"] = HF_CACHE
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit:
        gsm, m500 = gsm.select(range(min(limit, len(gsm)))), m500.select(range(min(limit, len(m500))))
    gsm_items = [{"prompt": PROMPT_TMPL.format(q=r["question"]), "question": r["question"],
                  "gold": gsm8k_gold(r["answer"]), "source": "openai/gsm8k:test"} for r in gsm]
    m500_items = [{"prompt": PROMPT_TMPL.format(q=r["problem"]), "question": r["problem"],
                   "gold": str(r["answer"]).strip(), "level": r.get("level"),
                   "subject": r.get("subject"), "source": "HuggingFaceH4/MATH-500:test"} for r in m500]
    return gsm_items, m500_items


def sft_problem_texts():
    """The raw text of the OpenR1 problems actually used in the SFT set (by uuid), for the
    vs-SFT decontam. OpenR1 was STREAMED by the SFT prep (no processed offline cache), so this
    succeeds only if OpenR1 is loadable; on offline-miss it returns ([], n_used, reason) and the
    caller records the vs-SFT decontam as PENDING (online) rather than silently skipping it."""
    os.environ["HF_HOME"] = HF_CACHE
    from datasets import load_dataset
    used = {json.loads(l)["source_uuid"] for l in open(SFT_META)}
    try:
        ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
        ds = ds.select_columns(["uuid", "problem"])   # project away the huge generations/messages
        texts = [r["problem"] for r in ds if r["uuid"] in used and r["problem"]]
        return texts, len(used), None
    except Exception as e:
        return [], len(used), f"{type(e).__name__}: {str(e)[:120]}"


def decontam(items, index):
    _, overlap = ngram_contamination([it["question"] for it in items], index, NGRAM_N, FLAG_THRESHOLD)
    n_flag = 0
    for it, ov in zip(items, overlap):
        it["ngram_overlap"] = round(ov, 4)
        it["contaminated"] = ov > FLAG_THRESHOLD
        n_flag += it["contaminated"]
    return n_flag


def write_jsonl(path, items, keys):
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps({k: it[k] for k in keys if k in it}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="tiny smoke subset; does not gate the frozen sets")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "prep.log", "w")
    def say(*a): print(*a); print(*a, file=log)

    say(f"[1/4] loading GSM8K-test + MATH-500 (offline cache {HF_CACHE}){' SMOKE' if args.limit else ''}")
    gsm, m500 = load_sets(args.limit)
    say(f"      gsm8k_test={len(gsm)}  math500={len(m500)}")

    say(f"[2/4] building 13-gram decontam index from the SFT problems (OpenR1, used uuids)")
    sft_texts, n_uuid, sft_err = sft_problem_texts()
    if sft_texts:
        index = build_ngram_index(sft_texts, NGRAM_N)
        decontam_target = f"{len(sft_texts)} OpenR1 SFT problems (of {n_uuid} used uuids)"
        sft_status = "done"
        say(f"      sft_problems={len(sft_texts)} (of {n_uuid} used uuids)  index_13grams={len(index):,}")
    else:
        index = None                            # vs-SFT decontam PENDING (OpenR1 not offline-cached)
        decontam_target = f"cross-eval only (vs-SFT PENDING online: {sft_err})"
        sft_status = f"PENDING-ONLINE ({sft_err})"
        say(f"      OpenR1 unavailable offline ({sft_err}); vs-SFT decontam DEFERRED — cross-eval check only")

    say(f"[3/4] decontaminating (n={NGRAM_N}, flag>{FLAG_THRESHOLD}); target={decontam_target}")
    if index is not None:
        n_gsm_flag, n_m5_flag = decontam(gsm, index), decontam(m500, index)
    else:                                       # cross-eval leakage: gsm vs math500 and vice versa
        n_gsm_flag = decontam(gsm, build_ngram_index([x["question"] for x in m500], NGRAM_N))
        n_m5_flag = decontam(m500, build_ngram_index([x["question"] for x in gsm], NGRAM_N))
    say(f"      gsm8k flagged={n_gsm_flag}/{len(gsm)}   math500 flagged={n_m5_flag}/{len(m500)}")

    say(f"[4/4] writing {OUT}")
    gk = ["prompt", "gold", "source", "ngram_overlap", "contaminated"]
    mk = gk + ["level", "subject"]
    write_jsonl(OUT / "gsm8k_test.jsonl", gsm, mk)
    write_jsonl(OUT / "math500.jsonl", m500, mk)
    write_jsonl(OUT / "gsm8k_test_clean.jsonl", [x for x in gsm if not x["contaminated"]], gk)
    write_jsonl(OUT / "math500_clean.jsonl", [x for x in m500 if not x["contaminated"]], mk)
    report = {
        "prepared_for": "research/rlvr/plan.md Phase-1 pass@k go/no-go",
        "consumer": "research.eval_math_acc.run_math_acc (extractor math-acc-v1)",
        "ngram_n": NGRAM_N, "flag_threshold": FLAG_THRESHOLD,
        "decontam_target": decontam_target,
        "sft_decontam_status": sft_status,
        "rerun_vs_sft_online": "HF_HUB_OFFLINE=0 python3 research/datasets/math-eval-v1/prepare_math_eval.py  (fetches OpenR1 problems online for the vs-SFT check)",
        "index_13grams": len(index) if index is not None else 0,
        "gsm8k_test": {"n": len(gsm), "contaminated": n_gsm_flag, "clean": len(gsm) - n_gsm_flag},
        "math500": {"n": len(m500), "contaminated": n_m5_flag, "clean": len(m500) - n_m5_flag},
        "max_overlap_gsm8k": max((x["ngram_overlap"] for x in gsm), default=0.0),
        "max_overlap_math500": max((x["ngram_overlap"] for x in m500), default=0.0),
        "smoke_limit": args.limit,
    }
    json.dump(report, open(OUT / "decontam_report.json", "w"), indent=2)
    say("DONE", json.dumps(report["gsm8k_test"]), json.dumps(report["math500"]))
    log.close()


if __name__ == "__main__":
    main()
