#!/usr/bin/env python3
"""RULER-style effective-context ladder on the anneal cohort — supplies the §C25
mid-training HARD item `effective_context_length_ruler` for run
2026-06-30_qwen3-0.6b_midtrain-anneal (the item that caps the verdict at directional).

Reuses the tested loader `research/eval_longcontext.py` (passkey ladder — the base is
not instruction-tuned, so passkey is the honest base-model retrieval probe) upgraded
for a per-checkpoint MEASUREMENT rather than a one-shot gate:
  * 6 rungs (512/1024/2048/4096/6144/8192 — spans AND exceeds the 4096 trained len),
    5 depths × 8 keys = 40 samples/rung, Wilson 95% CI per rung.
  * ONE grid built once (seed 0) and scored by ALL checkpoints → paired comparison.
  * ECL reported two ways: rel-0.85 (§C25: longest rung ≥ 0.85 × short-ctx acc,
    short-ctx = the 512 rung) and abs-0.5 (step-0 diagnostic continuity).
  * Cells: the 6 anneal arms + the un-annealed base (regression reference).

Eval-only, no training. safe_cuda-guarded; sequential (one GPU job, §C4.5).
Run `python3 sentinel.py preflight` first. --smoke = 1 rung × 4 samples × base only.
"""
from __future__ import annotations
import json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                  # BuildFromScratch/
MD = ROOT / "Qwen3-0.6B"
MODERN = next(MD.glob("builds/*_reproduce-modernized_*"))
for p in (str(ROOT), str(ROOT / "research"), str(MD), str(MODERN)):
    sys.path.insert(0, p)
import safe_cuda  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import model_imu1 as M  # noqa: E402  (flags-off == faithful model.py)
from eval_longcontext import ladder, score_passkey  # noqa: E402  (tested loader)
from eval_metrics import accuracy_wilson_ci  # noqa: E402

LENGTHS = (512, 1024, 2048, 4096, 6144, 8192)
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
N_PER_CELL = 8                                          # 5 depths × 8 = 40 samples/rung
TRAINED_LEN, REL, ABS_T = 4096, 0.85, 0.5
BASE_CKPT = (MD / "builds" / "2026-06-08_reproduce-faithful_qwen3-0.6b" /
             "checkpoint_qwen3_baseline2tpp.pt")
CELLS = [f"{arm}_seed{s}" for arm in ("fineweb", "mix") for s in (0, 1, 2)]


def make_generate(model, tok):
    @torch.no_grad()
    def generate(ids, max_new_tokens=8):
        x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
        out = []
        for _ in range(max_new_tokens):
            logits = model(input_ids=x)["logits"][0, -1]
            nxt = int(logits.argmax())
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device="cuda")], dim=1)
            if x.shape[1] > 8400:                       # safety
                break
        return tok.decode(out)
    return generate


def ecl_from(per_rung):
    """(rel-0.85 ECL, abs-0.5 ECL) from {rung: accuracy}."""
    short = per_rung[min(per_rung)]
    rel_ecl = abs_ecl = 0
    for L in sorted(per_rung):
        if short > 0 and per_rung[L] >= REL * short:
            rel_ecl = L
        if per_rung[L] >= ABS_T:
            abs_ecl = L
    return rel_ecl, abs_ecl


def main():
    smoke = "--smoke" in sys.argv
    safe_cuda.guard(0.85)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    lengths = LENGTHS[:1] if smoke else LENGTHS
    n_per = 2 if smoke else N_PER_CELL
    depths = DEPTHS[:2] if smoke else DEPTHS
    grid = {}                                           # ONE grid for all ckpts (paired)
    import eval_longcontext as elc
    import random
    rng = random.Random(0)
    for L in lengths:
        cell = []
        for d in depths:
            for _ in range(n_per):
                pk = rng.randint(10000, 99999)
                cell.append(elc.make_passkey_sample(tok, L, d, pk, rng))
        grid[L] = cell
    n_rung = len(next(iter(grid.values())))
    print(f"grid: {len(grid)} rungs × {n_rung} samples (paired across checkpoints)", flush=True)

    cells = ["base"] if smoke else ["base"] + CELLS
    results = {"suite": "ecl-ladder-v1 (passkey, eval_longcontext loader)",
               "lengths": list(lengths), "depths": list(depths), "n_per_rung": n_rung,
               "trained_len": TRAINED_LEN, "rel_threshold": REL, "abs_threshold": ABS_T,
               "cells": {}}
    for cell in cells:
        t0 = time.time()
        ckpt = BASE_CKPT if cell == "base" else HERE / f"checkpoint_{cell}.pt"
        cfg = M.Qwen3Config(use_value_residual=False, use_layernorm_scaling=False,
                            use_head_gating=False)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model = M.Qwen3ForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
        model.load_state_dict(sd, strict=True)
        gen = make_generate(model, tok)
        per_rung_acc, per_rung_ci = {}, {}
        for L, samples in grid.items():
            r = score_passkey(gen, tok, samples)
            hits = [it["hit"] for it in r["items"]]
            acc, lo, hi = accuracy_wilson_ci(hits)
            per_rung_acc[L], per_rung_ci[L] = acc, [lo, hi]
        rel_ecl, abs_ecl = ecl_from(per_rung_acc)
        results["cells"][cell] = {"per_rung_accuracy": per_rung_acc,
                                  "per_rung_wilson95": per_rung_ci,
                                  "ecl_rel085_shortctx": rel_ecl, "ecl_abs05": abs_ecl}
        del model; torch.cuda.empty_cache()
        accs = " ".join(f"{L}:{a:.2f}" for L, a in per_rung_acc.items())
        print(f"  {cell}: {accs} | ECL rel={rel_ecl} abs={abs_ecl} ({time.time()-t0:.0f}s)",
              flush=True)
        (HERE / ("ecl_ladder_smoke.json" if smoke else "ecl_ladder.json")
         ).write_text(json.dumps(results, indent=2))
    print(f"DONE -> {'ecl_ladder_smoke.json' if smoke else 'ecl_ladder.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
