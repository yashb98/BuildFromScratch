#!/usr/bin/env python3
"""Single-variable iso-FLOP ablation: NorMuon vs AdamW on the 2D hidden weights.

Both arms use the IDENTICAL faithful Qwen3-0.6B architecture, the SAME
decontaminated FineWeb-Edu data (fixed split seed 0), the SAME token budget,
schedule shape, seq/batch, and the SAME treatment of embeddings + 1D params
(AdamW, wd=0). The ONLY thing that changes between arms is how the 2D hidden
matrices are optimized:

  * adamw   : 2D hidden weights -> AdamW   @ peak_lr,  wd
  * normuon : 2D hidden weights -> NorMuon @ normuon_lr, wd   (embeddings/1D -> AdamW)

=> a §C18 single-variable comparison (the variable is the 2D-weight optimizer;
each optimizer runs at its own tuned LR, which is the fair way to compare
optimizers — disclosed in the run write-up). Per-cell seed varies model init +
DataLoader shuffle only; the data split is fixed so all 6 cells train on the same
corpus and are scored on the eval-harness's fixed pinned corpora.

Reuses the faithful build's proven data/eval/packing code by import (no copy).
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]            # BuildFromScratch/
FAITHFUL = ROOT / "Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FAITHFUL))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # local normuon.py

import safe_cuda  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import train_qwen3 as tq  # noqa: E402  — reuse stream_tokens/evaluate/chunked CE/packing
from model import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from normuon import NorMuon  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SEQ_LEN, MICRO_BATCH, GRAD_ACCUM = 4096, 4, 4
WARMUP, END_RATIO, SPLIT_SEED = 50, 0.1, 0          # fixed data split across cells
TOK_PER_STEP = SEQ_LEN * MICRO_BATCH * GRAD_ACCUM   # 65,536


def lr_factor(step, total):
    if step < WARMUP:
        return (step + 1) / WARMUP
    progress = min(1.0, (step - WARMUP) / max(1, total - WARMUP))
    return END_RATIO + (1 - END_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))


def build_optims(model, optimizer, peak_lr, normuon_lr, wd):
    """2D hidden matrices (non-embed) vs the rest (embeddings/norms/1D). Returns a
    list of (optimizer, base_lr) so the shared cosine factor can be applied to
    each. The 'rest' params are treated IDENTICALLY in both arms (AdamW, wd=0)."""
    twod, rest = [], []
    for name, p in model.named_parameters():
        (twod if (p.dim() == 2 and "embed_tokens" not in name) else rest).append(p)
    rest_opt = AdamW([{"params": rest, "weight_decay": 0.0}],
                     lr=peak_lr, betas=(0.9, 0.95), eps=1e-8)
    if optimizer == "adamw":
        twod_opt = AdamW([{"params": twod, "weight_decay": wd}],
                         lr=peak_lr, betas=(0.9, 0.95), eps=1e-8)
        return [(twod_opt, peak_lr), (rest_opt, peak_lr)], len(twod), len(rest)
    twod_opt = NorMuon(twod, lr=normuon_lr, weight_decay=wd, beta1=0.95, beta2=0.95)
    return [(twod_opt, normuon_lr), (rest_opt, peak_lr)], len(twod), len(rest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--optimizer", choices=["adamw", "normuon"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=600)            # ~39M tokens
    ap.add_argument("--peak_lr", type=float, default=2.4e-3)     # faithful-tuned AdamW LR
    ap.add_argument("--normuon_lr", type=float, default=0.011)   # IMU-1-tuned NorMuon 2D LR
    ap.add_argument("--weight_decay", type=float, default=0.1)   # on 2D, BOTH arms (held equal)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--resume_every", type=int, default=200,
                    help="save a full-state resume checkpoint every N optimizer steps "
                         "(crash-survival: the box hard-locks under load every ~5-10h)")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--tag", default=None, help="override the output tag (default: <optimizer>_seed<seed>)")
    a = ap.parse_args()

    arm = a.optimizer
    tag = a.tag if a.tag else f"{arm}_seed{a.seed}"
    log_path = RESULTS / f"{tag}.log"
    ckpt_path = RESULTS / f"checkpoint_{tag}.pt"
    resume_path = RESULTS / f"resume_{tag}.pt"
    token_budget = a.steps * TOK_PER_STEP

    if not torch.cuda.is_available():
        print("CUDA required."); return 1
    safe_cuda.guard(a.mem_fraction)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tq.log(f"[{tag}] start: optimizer={arm} seed={a.seed} steps={a.steps} "
           f"tok_budget={token_budget:,} peak_lr={a.peak_lr} normuon_lr={a.normuon_lr} "
           f"wd={a.weight_decay}", log_path)

    tokenizer = AutoTokenizer.from_pretrained(tq.REPO)
    # FIXED data split (seed 0) for ALL cells -> identical corpus; cached after cell 1.
    train_tokens, val_tokens = tq.stream_tokens(
        tokenizer, n_train=token_budget + 2_000_000, n_val=300_000,
        log_path=log_path, seed=SPLIT_SEED)
    pack = tq.PackedTextDataset(train_tokens, SEQ_LEN)

    # per-cell seed varies init + shuffle order ONLY (data is fixed above)
    tq.set_seed(a.seed)
    loader = DataLoader(pack, batch_size=MICRO_BATCH, shuffle=True, drop_last=True)
    tq.log(f"[{tag}] {len(pack):,} train windows of {SEQ_LEN}", log_path)

    model = Qwen3ForCausalLM(Qwen3Config()).to(device=device, dtype=dtype)
    model.train()
    optims, n_2d, n_rest = build_optims(model, arm, a.peak_lr, a.normuon_lr, a.weight_decay)
    tq.log(f"[{tag}] param split: {n_2d} 2D->{arm} | {n_rest} rest->AdamW", log_path)

    # ---- crash-survival resume (model + optim state + step + RNG). The box hard-locks
    # under sustained load every ~5-10h but a 420M rung needs ~16-18h, so progress MUST
    # survive a reboot. Saved every --resume_every steps (atomic + fsync). A fresh run
    # with no resume_<tag>.pt behaves EXACTLY as before (identical numerics). ----
    def save_resume(cur_step, cur_base_ppl, cur_t0):
        payload = {
            "step": cur_step,
            "model": model.state_dict(),
            "optims": [o.state_dict() for o, _ in optims],
            "baseline_ppl": cur_base_ppl,
            "elapsed": time.time() - cur_t0,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
            "meta": {"tag": tag, "arm": arm, "seed": a.seed, "total_steps": a.steps},
        }
        tmp = resume_path.with_suffix(".pt.tmp")
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush(); os.fsync(f.fileno())      # durable BEFORE rename (box hard-locks)
        os.replace(tmp, resume_path)             # atomic swap: old ckpt intact until this

    resume_step, resume_elapsed, resumed = 0, 0.0, False
    if resume_path.exists():
        try:
            # load to CPU: RNG-state must stay a CPU ByteTensor for set_rng_state; model
            # load_state_dict / optim load_state_dict move their tensors to the param
            # device themselves. weights_only=False: our own ckpt holds RNG/optim objects.
            ck = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            for o, s in zip((o for o, _ in optims), ck["optims"]):
                o.load_state_dict(s)
            random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
            torch.set_rng_state(ck["rng"]["torch"]); torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
            resume_step = int(ck["step"]); base_ppl = ck["baseline_ppl"]
            resume_elapsed = float(ck.get("elapsed", 0.0)); resumed = True
            tq.log(f"[{tag}] RESUMED from step {resume_step}/{a.steps} "
                   f"(baseline PPL={base_ppl:.2f}, {resume_elapsed/3600:.2f}h prior compute)", log_path)
        except Exception as e:                   # corrupt/incompatible -> fail closed to fresh
            tq.log(f"[{tag}] resume ckpt unreadable ({e!r}); starting fresh", log_path)
            resume_step, resume_elapsed, resumed = 0, 0.0, False

    train_model = model if a.no_compile else torch.compile(model)
    if not resumed:
        base_ppl, _ = tq.evaluate(model, val_tokens, device, SEQ_LEN)
        tq.log(f"[{tag}] baseline (random-init) val PPL={base_ppl:.2f}", log_path)

    # Return the baseline-eval's reserved allocator blocks to the unified pool BEFORE the
    # first training step. Fresh runs otherwise stack ~40GB of step-1 logits on top of the
    # ~48GB still-reserved eval pool (different shapes → not reused), a transient that hit
    # ~88% of the shared pool and tripped sentinel's 0.80 kill (and neared the box's crash
    # cliff). Resumed runs skip the eval, so this only helps fresh runs / the smoke; it is
    # numerically inert (frees cached-but-unused memory only). (2026-07-11)
    torch.cuda.empty_cache()

    step, accum, t0 = resume_step, 0, time.time() - resume_elapsed
    for o, _ in optims:
        o.zero_grad(set_to_none=True)
    done = step >= a.steps
    while not done:
        for inp, lbl in loader:
            inp, lbl = inp.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
            logits = train_model(input_ids=inp)["logits"]
            loss = tq.chunked_cross_entropy(logits, lbl)
            (loss / GRAD_ACCUM).backward()
            accum += 1
            if accum < GRAD_ACCUM:
                continue
            factor = lr_factor(step, a.steps)
            for o, base in optims:
                for g in o.param_groups:
                    g["lr"] = base * factor
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            for o, _ in optims:
                o.step()
            for o, _ in optims:
                o.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            if step % a.log_every == 0:
                dt = time.time() - t0
                tps = step * TOK_PER_STEP / max(1e-9, dt)
                tq.log(f"[{tag}] step {step}/{a.steps} loss {loss.item():.4f} "
                       f"lr {a.peak_lr*factor:.2e} |grad| {float(grad_norm):.2f} "
                       f"tok/s {tps:,.0f} mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", log_path)
            if step % a.resume_every == 0:
                save_resume(step, base_ppl, t0)  # crash-survival; a kill just dies fast (frees pool)
            if step >= a.steps:
                done = True
                break

    trained_ppl, _ = tq.evaluate(model, val_tokens, device, SEQ_LEN)
    tq.log(f"[{tag}] DONE in {(time.time()-t0)/60:.1f} min; fineweb-val PPL "
           f"{base_ppl:.2f} -> {trained_ppl:.2f} (the eval-harness does the real scoring)", log_path)
    torch.save({
        "model": model.state_dict(), "config": Qwen3Config().__dict__,
        "step": a.steps, "tok_seen": token_budget, "arm": arm, "seed": a.seed,
        "fineweb_val_ppl": trained_ppl, "baseline_ppl": base_ppl,
        "recipe": {"optimizer": arm, "peak_lr": a.peak_lr, "normuon_lr": a.normuon_lr,
                   "weight_decay_2d": a.weight_decay, "seq_len": SEQ_LEN,
                   "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
                   "warmup": WARMUP, "end_ratio": END_RATIO, "split_seed": SPLIT_SEED,
                   "total_steps": a.steps},
    }, ckpt_path)
    tq.log(f"[{tag}] saved {ckpt_path}", log_path)
    if resume_path.exists():
        os.remove(resume_path)                   # rung complete: resume ckpt no longer needed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
