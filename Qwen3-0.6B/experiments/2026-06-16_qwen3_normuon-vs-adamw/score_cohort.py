#!/usr/bin/env python3
"""Score all 6 ablation checkpoints through the text-lm-v2 suite logic (same
SEQ/STRIDE/corpora/bits_per_byte as eval-harness) and emit per-cell BPB+PPL.

Sequential on the single GB10 (parallel GPU evals would over-commit -> crash).
Corpora are loaded+tokenized ONCE; each checkpoint is a forward-only pass. The
BPB is the cross-comparable metric the verdict is computed on.
"""
from __future__ import annotations
import json, math, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
FAITHFUL = ROOT / "Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "Qwen3-0.6B"))
import safe_cuda  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from model import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from eval_metrics import bits_per_byte  # noqa: E402  (tested)

SUITE_VERSION = "text-lm-v2"
SEQ, STRIDE, MAX_WINDOWS = 1024, 512, 200            # text-lm-v2 constants
WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
CODEPARROT_REV = "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd"
CODE_CHARS = 500_000
RES = pathlib.Path(__file__).resolve().parent / "results"
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
CELLS = [f"{arm}_seed{s}" for arm in ("adamw", "normuon") for s in (0, 1, 2)]


@torch.no_grad()
def score(model, ids, tok):
    nlls, n, nbytes = [], 0, 0
    for i, b in enumerate(range(0, len(ids) - SEQ, STRIDE)):
        if i >= MAX_WINDOWS:
            break
        chunk = ids[b:b + SEQ].unsqueeze(0).to(DEVICE)
        logits = model(chunk)["logits"][..., :-1, :].float()
        labels = chunk[..., 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), reduction="sum")
        nlls.append(loss.item()); n += labels.numel()
        nbytes += len(tok.decode(labels[0].tolist()).encode("utf-8"))
    sum_nll = sum(nlls)
    return {"ppl": math.exp(sum_nll / n), "bpb": bits_per_byte(sum_nll, nbytes),
            "n_tokens": n, "n_bytes": nbytes}


def load_corpora(tok):
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation",
                      revision=WIKITEXT_REV)
    wt_text = "\n\n".join(e["text"] for e in wt if e["text"].strip())
    cp = load_dataset("codeparrot/codeparrot-clean-valid", split="train",
                      streaming=True, revision=CODEPARROT_REV)
    buf, total = [], 0
    for e in cp:
        buf.append(e["content"]); total += len(e["content"]) + 2
        if total > CODE_CHARS:
            break
    out = {}
    out["wikitext2_val"] = tok(wt_text, return_tensors="pt").input_ids[0]
    out["code_py"] = tok("\n\n".join(buf), return_tensors="pt").input_ids[0]
    return out


def main():
    safe_cuda.guard(0.85)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    print("loading corpora (once)...")
    corpora = load_corpora(tok)
    results = {"suite_version": SUITE_VERSION, "cells": {}}
    for cell in CELLS:
        ckpt = RES / f"checkpoint_{cell}.pt"
        t0 = time.time()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model = Qwen3ForCausalLM(Qwen3Config()).to(device=DEVICE, dtype=DTYPE).eval()
        model.load_state_dict(sd, strict=True)
        cell_res = {c: score(model, ids, tok) for c, ids in corpora.items()}
        results["cells"][cell] = cell_res
        del model; torch.cuda.empty_cache()
        print(f"  {cell}: wikitext bpb={cell_res['wikitext2_val']['bpb']:.4f} "
              f"ppl={cell_res['wikitext2_val']['ppl']:.2f} | code bpb="
              f"{cell_res['code_py']['bpb']:.4f}  ({time.time()-t0:.0f}s)")
        (RES / "cohort_bpb.json").write_text(json.dumps(results, indent=2))
    print(f"DONE -> {RES/'cohort_bpb.json'}")


if __name__ == "__main__":
    main()
