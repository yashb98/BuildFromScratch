"""Zero-shot downstream-benchmark scorer — the `text-lm-v3` capability axis.

The MISSING piece between the raw model and the already-tested aggregators in
`research/eval_metrics.py` (`accuracy_wilson_ci`, `ece`): it loads the benchmarks,
runs lm-eval-harness-style **loglikelihood** multiple-choice + LAMBADA last-token
scoring, and returns per-item arrays (`correct`, `correct_norm`, `confidence`) that
feed those aggregators directly.

DESIGN — model-agnostic & CPU-testable. The scorer takes a `logits_fn` callable
(`LongTensor[1,T] -> FloatTensor[1,T,V]`), NOT a model, so the loglik math is
unit-tested here with a deterministic stub and the SAME code runs on the real
checkpoint under eval-harness. This module NEVER trains and NEVER loads a model or
allocates CUDA — the CALLER (eval-harness template) loads the checkpoint behind
`safe_cuda.guard(...)` and passes `logits_fn`.

HONEST SCALE CAVEAT (carried into every result): at 596M params / ~1.19B training
tokens the model sits near chance on most multiple-choice tasks. `accuracy_wilson_ci`
makes that explicit — a CI overlapping the chance line means NO signal, and must be
read as such, never as a win. **LAMBADA (last-token) is the signal-bearing task at
this scale.** Report numbers; do not over-read a near-chance accuracy.

Task ids + pinned revisions are filled by `smoke_pin()` after a real streaming
load (`trust_remote_code=False`); any task needing TRC is dropped to a proposal,
never auto-run (suite versioning protocol).
"""
from __future__ import annotations
import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

LogitsFn = Callable[[torch.Tensor], torch.Tensor]  # [1,T] -> [1,T,V]

# Pinned revisions (streaming-smoke-loaded TRC=False, 2026-06-22). A task with
# revision=None is NOT run (never an unpinned load). PIQA is a script-based dataset
# (found piqa.py) → needs trust_remote_code → DEFERRED to a proposal, never auto-run.
TASKS = {
    "lambada":    {"hf": "EleutherAI/lambada_openai", "config": "default", "split": "test",       "revision": "900124bf3b8235c6daf21033af9948b3f07346c4", "kind": "lambada", "chance": None},
    "arc_easy":   {"hf": "allenai/ai2_arc",           "config": "ARC-Easy", "split": "test",       "revision": "210d026faf9955653af8916fad021475a3f00453", "kind": "mc", "chance": 0.25},
    "hellaswag":  {"hf": "Rowan/hellaswag",           "config": None,       "split": "validation", "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76", "kind": "mc", "chance": 0.25},
    "winogrande": {"hf": "allenai/winogrande",        "config": "winogrande_debiased", "split": "validation", "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "kind": "mc", "chance": 0.50},
    "piqa":       {"hf": "ybisk/piqa",                "config": None,       "split": "validation", "revision": None, "kind": "mc", "chance": 0.50, "note": "script-based -> needs TRC -> proposal, not auto-run"},
}


# ───────────────────────── core loglikelihood (the correctness-critical math) ─────────────────────────
@torch.no_grad()
def loglikelihood(logits_fn: LogitsFn, ctx_ids: Sequence[int], cont_ids: Sequence[int]):
    """Sum log p(cont | ctx) under the model, lm-eval style.

    The logprob of the continuation token at absolute position p is
    log_softmax(logits[p-1])[token_p]; summed over the continuation span. Also
    returns `is_greedy` (argmax matched at every continuation step) — LAMBADA's metric.
    Requires a non-empty context (position 0 has no preceding logit, like lm-eval).
    """
    if len(ctx_ids) == 0:
        raise ValueError("context must be non-empty (no logit precedes position 0)")
    full = list(ctx_ids) + list(cont_ids)
    ids = torch.tensor(full, dtype=torch.long).unsqueeze(0)
    logits = logits_fn(ids)[0].float()                       # [T, V]
    start = len(ctx_ids)
    logp = logits[start - 1:len(full) - 1].log_softmax(dim=-1)  # predicts tokens start..end-1
    tgt = torch.tensor(full[start:], dtype=torch.long)
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [n_cont]
    is_greedy = bool((logp.argmax(dim=-1) == tgt).all().item())
    return float(tok_lp.sum().item()), len(cont_ids), is_greedy


def _enc(tokenizer, s: str):
    return tokenizer.encode(s, add_special_tokens=False)


# ───────────────────────── per-task scoring → per-item arrays for the aggregators ─────────────────────────
def score_multiple_choice(logits_fn, tokenizer, items, bos_id=None):
    """items: [{context:str, choices:[str], gold:int}]. Continuations get a leading
    space (lm-eval convention). Returns dict of per-item arrays for eval_metrics."""
    correct, correct_norm, conf = [], [], []
    for it in items:
        ctx = _enc(tokenizer, it["context"])
        if not ctx:                                    # empty ctx -> prepend BOS so position 0 has a predecessor
            ctx = [bos_id if bos_id is not None else (tokenizer.bos_token_id or tokenizer.eos_token_id or 0)]
        lps, lp_norms = [], []
        for ch in it["choices"]:
            cont = _enc(tokenizer, ch if ch.startswith(" ") else " " + ch)
            lp, n, _ = loglikelihood(logits_fn, ctx, cont)
            lps.append(lp); lp_norms.append(lp / max(1, n))
        lps_t = torch.tensor(lps)
        pred, pred_norm = int(lps_t.argmax()), int(torch.tensor(lp_norms).argmax())
        correct.append(int(pred == it["gold"]))
        correct_norm.append(int(pred_norm == it["gold"]))
        conf.append(float(F.softmax(lps_t, dim=0).max()))   # model's confidence in its pick (for ECE)
    return {"correct": correct, "correct_norm": correct_norm, "confidence": conf, "n": len(items)}


def score_lambada(logits_fn, tokenizer, items):
    """items: [{text:str}]. Accuracy = greedy match of the final word's tokens
    given everything before it (the standard LAMBADA-OpenAI metric)."""
    correct, target_lps = [], []
    for it in items:
        text = it["text"].strip()
        if " " not in text:
            continue
        ctx_str, last = text.rsplit(" ", 1)
        ctx = _enc(tokenizer, ctx_str)
        cont = _enc(tokenizer, " " + last)
        if not ctx or not cont:
            continue
        lp, _, greedy = loglikelihood(logits_fn, ctx, cont)
        correct.append(int(greedy)); target_lps.append(lp)
    return {"correct": correct, "confidence": None, "n": len(correct),
            "target_logprob_mean": (sum(target_lps) / len(target_lps)) if target_lps else float("nan")}


# ───────────────────────── benchmark loaders (schema mapping per dataset) ─────────────────────────
def _norm_items(task: str, rows):
    """Map each dataset's native schema to {context, choices, gold} or {text}."""
    out = []
    if task == "lambada":
        return [{"text": r["text"]} for r in rows]
    if task == "arc_easy":
        for r in rows:
            ch = r["choices"]; labels = list(ch["label"]); texts = list(ch["text"])
            key = r["answerKey"]
            if key not in labels:    # a few items use numeric keys
                continue
            out.append({"context": f"Question: {r['question']}\nAnswer:", "choices": texts, "gold": labels.index(key)})
    elif task == "hellaswag":
        for r in rows:
            ctx = (r["ctx_a"] + " " + r["ctx_b"]).strip() if r.get("ctx_b") else r["ctx"]
            out.append({"context": ctx, "choices": list(r["endings"]), "gold": int(r["label"])})
    elif task == "piqa":
        for r in rows:
            out.append({"context": r["goal"], "choices": [r["sol1"], r["sol2"]], "gold": int(r["label"])})
    elif task == "winogrande":
        for r in rows:
            opt = [r["option1"], r["option2"]]; gold = int(r["answer"]) - 1
            # standard winogrande: replace the blank with each option, score the suffix
            out.append({"context": r["sentence"].split("_")[0].rstrip(),
                        "choices": [(o + r["sentence"].split("_", 1)[1]) for o in opt], "gold": gold})
    return out


def load_task(task: str, limit: int | None = None):
    """Streaming load at the PINNED revision, TRC=False. Returns normalized items."""
    from datasets import load_dataset
    t = TASKS[task]
    if t["revision"] is None:
        raise RuntimeError(f"task {task} has no pinned revision — run smoke_pin() and update TASKS first")
    ds = load_dataset(t["hf"], t["config"], split=t["split"], revision=t["revision"],
                      streaming=True, trust_remote_code=False)
    rows = []
    for i, r in enumerate(ds):
        if limit and i >= limit:
            break
        rows.append(r)
    return _norm_items(task, rows)


def run_downstream(logits_fn, tokenizer, tasks=None, limit=None):
    """Score each pinned task → per-task results ready for eval_metrics.accuracy_wilson_ci.
    Tasks with no pinned revision are skipped with a recorded reason (never silently dropped)."""
    tasks = tasks or list(TASKS)
    results = {}
    for task in tasks:
        if TASKS[task]["revision"] is None:
            results[task] = {"status": "skipped — revision not pinned (run smoke_pin)"}
            continue
        items = load_task(task, limit=limit)
        if TASKS[task]["kind"] == "lambada":
            r = score_lambada(logits_fn, tokenizer, items)
        else:
            r = score_multiple_choice(logits_fn, tokenizer, items)
        r["task"], r["chance"], r["hf"], r["revision"] = task, TASKS[task]["chance"], TASKS[task]["hf"], TASKS[task]["revision"]
        results[task] = r
    return results


# ───────────────────────── CPU self-test of the loglik / MC / lambada math (no model, no network) ─────────────────────────
def _self_test():
    """Deterministic stub `logits_fn`: token t's logit favors token (t+1)%V, so the
    greedy continuation of any prefix is the +1 sequence. Verifies the loglik math,
    argmax MC selection, length-normalization, and LAMBADA greedy detection."""
    V = 11
    def stub(ids):                                   # [1,T] -> [1,T,V]: logits[:,p,(ids[p]+1)%V] high
        T = ids.shape[1]
        out = torch.full((1, T, V), -5.0)
        for p in range(T):
            out[0, p, (int(ids[0, p]) + 1) % V] = 5.0
        return out

    class Tok:                                        # identity tokenizer over space-separated ints
        bos_token_id = 0; eos_token_id = 0
        def encode(self, s, add_special_tokens=False):
            return [int(x) for x in s.split()] if s.strip() else []

    tok = Tok()
    # loglikelihood: ctx=[3], greedy cont is [4] (=(3+1)); [4] greedy, [5] not.
    lp_g, n, greedy = loglikelihood(stub, [3], [4]); assert greedy and n == 1, (lp_g, greedy)
    _, _, ng = loglikelihood(stub, [3], [5]); assert not ng
    # MC: context "1", gold choice "2" (=greedy of 1) must beat "9"
    mc = score_multiple_choice(stub, tok, [{"context": "1", "choices": ["2", "9"], "gold": 0}])
    assert mc["correct"] == [1] and mc["correct_norm"] == [1], mc
    # MC picks the WRONG one when gold is the non-greedy choice -> correct=0 (honest)
    mc2 = score_multiple_choice(stub, tok, [{"context": "1", "choices": ["2", "9"], "gold": 1}])
    assert mc2["correct"] == [0], mc2
    # LAMBADA: "3 4" -> ctx "3", last "4" is greedy -> correct; "3 9" -> not
    lam = score_lambada(stub, tok, [{"text": "3 4"}, {"text": "3 9"}])
    assert lam["correct"] == [1, 0], lam
    # feeds the existing aggregator
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from eval_metrics import accuracy_wilson_ci
    point, ci_low, ci_high = accuracy_wilson_ci([1, 1, 0, 1])   # returns (point, ci_low, ci_high)
    assert 0.0 <= ci_low <= point <= ci_high <= 1.0, (point, ci_low, ci_high)
    print("eval_downstream self-test PASS (loglik · MC argmax · norm · LAMBADA greedy · aggregator handoff)")


if __name__ == "__main__":
    _self_test()
