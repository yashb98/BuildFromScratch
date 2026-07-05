#!/usr/bin/env python3
r"""Corrected HELD-OUT reasoning verdict for the SFT 3-seed cohort (post-training plan §2A).

Why this exists — the confound score_verdict.py only flags:
  The in-loop reasoning PPL that score_verdict.py parses is (a) measured on TRAIN windows and
  (b) over a DIFFERENT token set per arm. The masked SFT cells score RESPONSE tokens only
  (14.262 / 159,582 toks) while the --no_mask control scores ALL tokens (15.083 / 163,840 toks),
  because sft_train.eval_reasoning_ce reuses each arm's *training* mask. Subtracting those two
  numbers is apples-to-oranges, so that SFT-vs-control branch is advisory only.

This script re-scores EVERY arm's checkpoint on ONE fixed HELD-OUT eval set with ONE fixed
masking, so SFT (masked) vs CONTROL (--no_mask) is finally a clean iso-FLOP comparison and the
absolute numbers are generalization (held-out), not a train-set proxy.

Metrics — identical construction for every checkpoint {base, sft_seed0/1/2, ctrl_seed0/1/2}:
  * reasoning_ppl_masked : held-out OpenR1-Math eval split (research/datasets/.../eval/eval_tokens.bin:
      66 docs, doc-level seeded split + 13-gram dedup vs train, NEVER trained on), RESPONSE-masked.
      prompt_len is reconstructed per doc from eval_docs.jsonl ("prompt + '\n\n' + trace") with the
      dataset tokenizer — approximate only at the prompt/trace junction (a few tokens), so we also
      report the exact, tokenizer-free:
  * reasoning_ppl_full   : same held-out set, NO masking — robustness cross-check. The SFT-vs-control
      sign should agree with the masked metric; the verdict requires BOTH to clear the gate.
  * forgetting_ppl       : held-out FineWeb-Edu (eval/forgetting_tokens.bin), full-sequence — the
      §C13 catastrophic-forgetting / retention probe vs the frozen base (a reasoning gain that wrecks
      general knowledge is not a win).

Verdict: SFT-vs-control + SFT-vs-base 95% CIs via research/eval_stats (the same unit-tested Welch-t
score_verdict.py uses). Writes reasoning_verdict.json + plots/reasoning_verdict.png.

GPU, one model at a time, safe_cuda-guarded; meant to fire AFTER cohort.done (training finished,
GPU free). `--smoke` validates the held-out data plumbing on CPU (no model load).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import sft_train as S          # sets sys.path; imports safe_cuda BEFORE torch, plus numpy/torch/model
import numpy as np             # noqa: E402  (sft_train already imported it; cached)
import torch                   # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eval_stats import seed_delta_significant, mean_std_sem  # noqa: E402  (research on path via sft_train)

HERE = pathlib.Path(__file__).resolve().parent
DS = S.DATASET
EVAL_DIR = DS / "eval"
EOS = 151643
SEQ = 4096
IGNORE = S.IGNORE


# ----------------------------------------------------------------- data
def load_docs_tokens(bin_path) -> list[np.ndarray]:
    """Split a flat eos-separated uint32 token stream into per-doc int64 arrays (trailing eos
    kept, exactly as the dataset built it). eos count == n_docs, so this aligns 1:1 with the
    eval_docs.jsonl order (same write loop in prepare_openr1-math-220k.py)."""
    t = np.fromfile(bin_path, dtype=np.uint32).astype(np.int64)
    eos_idx = np.where(t == EOS)[0]
    docs, start = [], 0
    for e in eos_idx:
        docs.append(t[start:e + 1])      # include trailing eos
        start = e + 1
    if start < t.size:                   # trailing doc without an eos
        docs.append(t[start:])
    return docs


def reasoning_prompt_lens(n_docs: int):
    """prompt_len per held-out reasoning doc, reconstructed from eval_docs.jsonl with the dataset
    tokenizer (offline). The eval split saved only {uuid,text}, NOT prompt_len, and text is
    'prompt + "\\n\\n" + trace' (prepare_openr1-math-220k.sft_text). Returns None (-> masked metric
    skipped) if the tokenizer is unavailable."""
    try:
        from transformers import AutoTokenizer
        repo = json.loads((DS / "meta.json").read_text())["tokenizer_repo"]
        tok = AutoTokenizer.from_pretrained(repo)
    except Exception as e:  # noqa: BLE001
        print("[reasoning_eval] tokenizer unavailable -> masked metric skipped:", str(e)[:90])
        return None
    docs = [json.loads(l) for l in (EVAL_DIR / "eval_docs.jsonl").read_text().splitlines()]
    assert len(docs) == n_docs, f"doc/token-stream mismatch: {len(docs)} docs vs {n_docs} token-segments"
    pls = []
    for d in docs:
        prompt = d["text"].split("\n\n", 1)[0]               # the problem statement (before the trace)
        pls.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))
    return pls


# ----------------------------------------------------------------- scoring
def _ce_sum_count(logits: torch.Tensor, labels: torch.Tensor, chunk: int = 8192):
    """(sum NLL over response tokens, count) — chunked, ignore_index=IGNORE, never materializing the
    full (N,vocab) fp32 logits (§C1). Mirrors sft_train.masked_chunked_ce but returns sum+count so
    PPL can be aggregated correctly across windows and docs."""
    flat = logits.reshape(-1, logits.size(-1))
    tgt = labels.reshape(-1)
    tot = flat.new_zeros((), dtype=torch.float32)
    n = 0
    for i in range(0, flat.size(0), chunk):
        tt = tgt[i:i + chunk]
        cnt = int((tt != IGNORE).sum())
        if cnt == 0:
            continue
        lt = flat[i:i + chunk].float()
        tot = tot + F.cross_entropy(lt, tt, ignore_index=IGNORE, reduction="sum")
        n += cnt
    return float(tot), n


@torch.no_grad()
def score_ppl(model, docs, device, prompt_lens=None):
    """Held-out PPL over docs. If prompt_lens is given, mask the first prompt_len tokens of each doc
    (response-only); else score all tokens. Each doc is next-token shifted then chopped into
    non-overlapping <=SEQ windows. Returns (ppl, scored_tokens, scored_tokens, total_tokens)."""
    model.eval()
    total_nll, total_n, scored_toks, all_toks = 0.0, 0, 0, 0
    for di, ids in enumerate(docs):
        if len(ids) < 2:
            continue
        labels = ids.copy()
        if prompt_lens is not None:
            pl = min(prompt_lens[di], len(ids))
            labels[:pl] = IGNORE
        inp_all = ids[:-1]
        tgt_all = labels[1:]                  # predict token t+1 from token t (sft_train semantics)
        all_toks += len(ids)
        scored_toks += int((tgt_all != IGNORE).sum())
        for s in range(0, len(inp_all), SEQ):
            inp = torch.from_numpy(inp_all[s:s + SEQ].copy()).long().unsqueeze(0).to(device)
            tgt = torch.from_numpy(tgt_all[s:s + SEQ].copy()).long().unsqueeze(0).to(device)
            logits = model(input_ids=inp)["logits"]
            sn, c = _ce_sum_count(logits, tgt)
            total_nll += sn
            total_n += c
    ppl = float(np.exp(total_nll / max(1, total_n))) if total_n else float("nan")
    return ppl, total_n, scored_toks, all_toks


def load_model(ckpt_path, device, dtype=torch.bfloat16):
    """Load a checkpoint into a fresh Qwen3 model, strict=True (same path as sft_train)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = S.Qwen3Config(**{k: v for k, v in ck["config"].items()
                           if k in S.Qwen3Config.__dataclass_fields__})
    m = S.Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
    m.load_state_dict(ck["model"], strict=True)
    return m


def discover_arms():
    """{name: checkpoint_path} for base + whichever sft/ctrl seed checkpoints exist on disk."""
    arms = {"base": S.BASE_CKPT}
    for s in (0, 1, 2):
        for pre in ("sft", "ctrl"):
            p = HERE / f"checkpoint_{pre}_seed{s}.pt"
            if p.exists():
                arms[f"{pre}_seed{s}"] = p
    return arms


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help="CPU-only: validate held-out data plumbing (doc split, prompt_len, resp_frac); no model")
    a = ap.parse_args()

    reasoning_docs = load_docs_tokens(EVAL_DIR / "eval_tokens.bin")
    prompt_lens = reasoning_prompt_lens(len(reasoning_docs))

    if a.smoke:
        tot = sum(len(d) for d in reasoning_docs)
        print(f"[smoke] reasoning docs={len(reasoning_docs)} tokens={tot:,}")
        if prompt_lens is not None:
            masked = sum(min(prompt_lens[i], len(reasoning_docs[i])) for i in range(len(reasoning_docs)))
            print(f"[smoke] prompt_lens reconstructed for {len(prompt_lens)} docs; "
                  f"mean prompt_len={sum(prompt_lens)/len(prompt_lens):.1f}; "
                  f"response fraction={1 - masked / max(1, tot):.4f} (train in-loop was ~0.974)")
        fdocs = load_docs_tokens(EVAL_DIR / "forgetting_tokens.bin")
        print(f"[smoke] forgetting docs={len(fdocs)} tokens={sum(len(d) for d in fdocs):,}")
        arms = discover_arms()
        print(f"[smoke] arms present: {list(arms)}")
        print("[smoke] OK — data plumbing valid; GPU scoring deferred to the real run.")
        return 0

    if not torch.cuda.is_available():
        print("CUDA required."); return 1
    S.safe_cuda.guard(0.85)
    device = torch.device("cuda")
    forget_docs = load_docs_tokens(EVAL_DIR / "forgetting_tokens.bin")
    arms = discover_arms()

    res = {}
    for name, ck in arms.items():
        m = load_model(ck, device)
        r_full, nfull, _, _ = score_ppl(m, reasoning_docs, device, prompt_lens=None)
        entry = {"reasoning_ppl_full": round(r_full, 4), "reasoning_full_toks": nfull}
        if prompt_lens is not None:
            r_mask, nmask, stok, atok = score_ppl(m, reasoning_docs, device, prompt_lens=prompt_lens)
            entry["reasoning_ppl_masked"] = round(r_mask, 4)
            entry["reasoning_resp_toks"] = nmask
            entry["resp_frac"] = round(stok / max(1, atok), 4)
        f_ppl, nf, _, _ = score_ppl(m, forget_docs, device, prompt_lens=None)
        entry["forgetting_ppl"] = round(f_ppl, 4)
        res[name] = entry
        del m
        torch.cuda.empty_cache()
        print(name, json.dumps(entry))

    def col(metric, pre):
        return [res[f"{pre}_seed{s}"][metric] for s in (0, 1, 2)
                if f"{pre}_seed{s}" in res and metric in res[f"{pre}_seed{s}"]]

    metric = "reasoning_ppl_masked" if prompt_lens is not None else "reasoning_ppl_full"
    sft, ctrl = col(metric, "sft"), col(metric, "ctrl")
    base = res.get("base", {}).get(metric)
    out = {"arms": res, "headline_metric": metric,
           "eval": {"reasoning_set": "held-out OpenR1-Math eval split (66 docs, seeded split + 13-gram dedup vs train)",
                    "forgetting_set": "held-out FineWeb-Edu (sample-10BT)",
                    "masking": "response (prompt_len reconstructed from eval_docs)" if prompt_lens is not None
                    else "none (full-sequence only — tokenizer unavailable)"}}

    if len(sft) == 3 and len(ctrl) == 3:
        r = seed_delta_significant(ctrl, sft, direction="lower_is_better")
        out["sft_vs_control"] = {"improvement_ppl": r["improvement"], "ci95": r["ci95"],
                                 "significant": r["significant"], "warning": r["warning"]}
        sff, cff = col("reasoning_ppl_full", "sft"), col("reasoning_ppl_full", "ctrl")
        rf = seed_delta_significant(cff, sff, direction="lower_is_better")
        out["sft_vs_control_fullseq"] = {"improvement_ppl": rf["improvement"], "ci95": rf["ci95"],
                                         "significant": rf["significant"],
                                         "note": "robustness cross-check (exact, tokenizer-free)"}
        both = r["significant"] and rf["significant"]
        out["overall_verdict"] = (
            "win — SFT (response-masking) beats the iso-FLOP --no_mask control on held-out reasoning "
            "PPL; masked AND full-sequence CIs both exclude 0" if both else
            ("directional — masked and full-sequence comparisons disagree on significance; "
             "treat as not-yet-separable" if r["significant"] != rf["significant"] else
             "inconclusive — SFT reasoning gain not separable from just-more-tokens at this budget"))
    else:
        out["overall_verdict"] = (f"incomplete — need 3 sft + 3 ctrl checkpoints "
                                  f"(have sft={len(sft)}, ctrl={len(ctrl)})")

    if base is not None and len(sft) == 3:
        rb = seed_delta_significant([base, base, base], sft, direction="lower_is_better")
        out["sft_vs_base"] = {"improvement_ppl": rb["improvement"], "ci95": rb["ci95"],
                              "significant": rb["significant"],
                              "note": "base repeated x3 -> reflects SFT seed-variance only (sanity check, not a paired test)"}

    fb = res.get("base", {}).get("forgetting_ppl")
    out["forgetting"] = {"base_ppl": fb,
                         "sft_mean": round(mean_std_sem(col("forgetting_ppl", "sft"))[0], 4) if col("forgetting_ppl", "sft") else None,
                         "ctrl_mean": round(mean_std_sem(col("forgetting_ppl", "ctrl"))[0], 4) if col("forgetting_ppl", "ctrl") else None,
                         "note": "held-out FineWeb-Edu PPL; >> base = catastrophic forgetting (§C13)"}

    (HERE / "reasoning_verdict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    try:
        sys.path.insert(0, str(S.ROOT / "research"))
        import eval_plots as P
        if len(sft) == 3:
            labels = ["base", "control\n(no_mask)", "SFT\n(masked)"]
            vals = [base, mean_std_sem(ctrl)[0], mean_std_sem(sft)[0]]
            P.bar_with_ci(labels, vals, title="SFT 3-seed: HELD-OUT reasoning PPL (corrected, iso-FLOP)",
                          ylabel=f"{metric} (lower=better)", out=HERE / "plots" / "reasoning_verdict.png",
                          zero_line=False)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", str(e)[:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
