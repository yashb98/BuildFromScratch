#!/usr/bin/env python3
"""Score all 12 ablation checkpoints (4 arms × 3 seeds) through the text-lm-v2 suite
logic (same SEQ/STRIDE/corpora/bits_per_byte as eval-harness) → per-cell BPB+PPL.

Run AFTER cohort.done (all 12 arm_<cell>.done present). Sequential on the single GB10
(parallel GPU evals over-commit → crash); run `python3 sentinel.py preflight` first.

KEY vs the NorMuon sibling: the +arch arm checkpoints carry IMU-1's extra params
(gate_proj, vr_*), so each cell is loaded into `model_imu1` with the arm's arch flags
(arch-on only for the `arch` arm; bit-identical to faithful model.py when off). The
checkpoints live at the RUN-DIR ROOT (checkpoint_<cell>.pt), not results/.
"""
from __future__ import annotations
import json, math, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                  # BuildFromScratch/
MODERN = next((ROOT / "Qwen3-0.6B").glob("builds/*_reproduce-modernized_*"))
for p in (str(ROOT), str(ROOT / "research"), str(ROOT / "Qwen3-0.6B"), str(MODERN)):
    sys.path.insert(0, p)
import safe_cuda  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import model_imu1 as M  # noqa: E402  (flags-off == faithful model.py, bit-exact)
from eval_metrics import bits_per_byte  # noqa: E402  (tested)

SUITE_VERSION = "text-lm-v2"
SEQ, STRIDE, MAX_WINDOWS = 1024, 512, 200
WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
CODEPARROT_REV = "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd"
CODE_CHARS = 500_000
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
ARMS = ("baseline", "wsd", "zloss", "arch")
CELLS = [f"{arm}_seed{s}" for arm in ARMS for s in (0, 1, 2)]


@torch.no_grad()
def score(model, ids, tok):
    nlls, n, nbytes = [], 0, 0
    for i, b in enumerate(range(0, len(ids) - SEQ, STRIDE)):
        if i >= MAX_WINDOWS:
            break
        chunk = ids[b:b + SEQ].unsqueeze(0).to(DEVICE)
        logits = model(input_ids=chunk)["logits"][..., :-1, :].float()
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
    return {"wikitext2_val": tok(wt_text, return_tensors="pt").input_ids[0],
            "code_py": tok("\n\n".join(buf), return_tensors="pt").input_ids[0]}


def main():
    missing = [c for c in CELLS if not (HERE / f"checkpoint_{c}.pt").exists()]
    if missing:
        print(f"NOT READY — missing checkpoints: {missing}"); return 1
    safe_cuda.guard(0.85)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    print("loading corpora (once)...")
    corpora = load_corpora(tok)
    results = {"suite_version": SUITE_VERSION, "cells": {}}
    for cell in CELLS:
        arch_on = cell.startswith("arch")
        cfg = M.Qwen3Config(use_value_residual=arch_on, use_layernorm_scaling=arch_on,
                            use_head_gating=arch_on)
        t0 = time.time()
        sd = torch.load(HERE / f"checkpoint_{cell}.pt", map_location="cpu",
                        weights_only=False)["model"]
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model = M.Qwen3ForCausalLM(cfg).to(device=DEVICE, dtype=DTYPE).eval()
        model.load_state_dict(sd, strict=True)
        cell_res = {c: score(model, ids, tok) for c, ids in corpora.items()}
        results["cells"][cell] = cell_res
        del model; torch.cuda.empty_cache()
        print(f"  {cell}: wikitext bpb={cell_res['wikitext2_val']['bpb']:.4f} "
              f"ppl={cell_res['wikitext2_val']['ppl']:.2f} | code bpb="
              f"{cell_res['code_py']['bpb']:.4f}  ({time.time()-t0:.0f}s)")
        (HERE / "cohort_bpb.json").write_text(json.dumps(results, indent=2))
    print(f"DONE -> {HERE/'cohort_bpb.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
