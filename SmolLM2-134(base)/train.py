"""
Minimal training loop for SmolLM2-135M (from random init).

Faithful to the 135M recipe (per paper §6 + nanotron config_smollm2_135M.yaml):
    - AdamW(β=(0.9, 0.95))                  paper §4.1
    - WSD schedule, 20% decay phase         paper §6
    - Peak LR = 3.0e-3                      paper §6
    - bf16 mixed precision                  model card
    - Gradient clip 1.0                     nanotron clip_grad
    - Weight decay 0.01                     nanotron weight_decay
    - Sequence length 2048                  nanotron sequence_length

Deviations from a full reproduction (intentional, for a single-GPU starter):
    - Tiny dataset (wikitext-103-raw) instead of the 2T-token mixture. The
      point here is to validate the training stack end-to-end on something you
      can actually feed; serious pretraining needs the curated mixture.
    - Small global batch — set to whatever fits. Paper uses 64×H100; we don't.
    - Token packing is rudimentary (truncation, no cross-document attention).

    python train.py --steps 100
"""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model_full import SmolLM2ForCausalLM, SmolLM2Config


# ---------------------------------------------------------------------------
# WSD (Warmup-Stable-Decay) scheduler — Hu et al. 2024 / Hägele et al. 2024.
#   1. Linear warmup from 0 to peak_lr over `warmup_steps`.
#   2. Constant at peak_lr through (1 - decay_frac) of total_steps.
#   3. Linear decay to 0 over the final decay_frac of total_steps.
# Paper §6: 135M uses 20% decay. nanotron config: warmup=2000, decay_frac=0.20.
# ---------------------------------------------------------------------------
def make_wsd_scheduler(optimizer, warmup_steps: int, total_steps: int, decay_frac: float = 0.2):
    decay_start = int(total_steps * (1.0 - decay_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < decay_start:
            return 1.0
        decay_progress = (step - decay_start) / max(1, total_steps - decay_start)
        return max(0.0, 1.0 - decay_progress)

    return LambdaLR(optimizer, lr_lambda)


class PackedTextDataset(torch.utils.data.Dataset):
    """Pre-tokenized contiguous windows of `seq_len`."""
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.tokens = token_ids
        self.seq_len = seq_len
        self.n_windows = (len(token_ids) - 1) // seq_len

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def build_dataset(tokenizer, seq_len: int):
    """Tokenize wikitext-103-raw-v1 train split, pack into seq_len windows."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    eos = tokenizer.eos_token_id
    all_ids = []
    for ex in ds:
        text = ex["text"]
        if not text.strip():
            continue
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        all_ids.append(eos)
    tokens = torch.tensor(all_ids, dtype=torch.long)
    print(f"Packed {len(tokens):,} tokens into windows of {seq_len}.")
    return PackedTextDataset(tokens, seq_len)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, help="total optimizer steps")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2, help="micro-batch (per accum step)")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3.0e-3, help="peak LR; paper §6")
    ap.add_argument("--warmup_steps", type=int, default=20,
                    help="tiny demo default; nanotron config_smollm2_135M.yaml: 2000")
    ap.add_argument("--weight_decay", type=float, default=0.01,
                    help="nanotron config_smollm2_135M.yaml: 0.01")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="nanotron config_smollm2_135M.yaml: clip_grad=1.0")
    ap.add_argument("--decay_frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--compile", action="store_true",
                    help="wrap model in torch.compile(mode='reduce-overhead')")
    ap.add_argument("--ckpt_path", default="checkpoint.pt")
    ap.add_argument("--resume", default=None,
                    help="path to checkpoint to resume from (restores model, optim, sched, RNG)")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if device.type == "cpu":
        dtype = torch.float32

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    print("Building dataset (this tokenizes wikitext-103 once — first run is slow)...")
    dataset = build_dataset(tokenizer, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    print("Initializing model from scratch (random init)...")
    cfg = SmolLM2Config()
    model = SmolLM2ForCausalLM(cfg).to(device=device, dtype=dtype)
    model.train()

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.dim() < 2 else decay).append(p)
    optim = AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )
    sched = make_wsd_scheduler(optim, args.warmup_steps, args.steps, decay_frac=args.decay_frac)

    train_model = model
    if args.compile:
        print("Compiling model with torch.compile(mode='reduce-overhead')...")
        train_model = torch.compile(model, mode="reduce-overhead")

    start_step = 0
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "optim" in ck:
            optim.load_state_dict(ck["optim"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        if "rng_torch" in ck:
            torch.set_rng_state(ck["rng_torch"])
        if "rng_cuda" in ck and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        if "rng_numpy" in ck:
            np.random.set_state(ck["rng_numpy"])
        if "rng_python" in ck:
            random.setstate(ck["rng_python"])
        start_step = ck.get("step", 0)
        print(f"  resumed at step={start_step}")

    def save_ckpt(step: int):
        torch.save({
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "step": step,
            "training_recipe": {
                "steps": args.steps,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "decay_frac": args.decay_frac,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "seed": args.seed,
                "dtype": args.dtype,
                "compile": args.compile,
            },
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        }, args.ckpt_path)

    step = start_step
    accum = 0
    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    while step < args.steps:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = train_model(input_ids=inputs)["logits"]
            # .float() on logits: bf16 CE over 49k classes is numerically noisy
            # (the log-sum-exp needs the headroom); standard practice is fp32 CE.
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
            )
            (loss / args.grad_accum).backward()
            accum += 1

            if accum == args.grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                accum = 0
                step += 1

                if step % args.log_every == 0:
                    tokens_seen = (step - start_step) * args.grad_accum * args.batch_size * args.seq_len
                    dt = time.time() - t0
                    tps = tokens_seen / max(1e-9, dt)
                    cur_lr = sched.get_last_lr()[0]
                    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
                    print(f"step {step:5d} | loss {loss.item():.4f} | lr {cur_lr:.2e} | "
                          f"|grad| {float(grad_norm):.3f} | mem {peak_mem_mb:.0f}MB | "
                          f"tok/s {tps:,.0f} | tokens {tokens_seen:,}")

                if step >= args.steps:
                    break

    save_ckpt(step)
    print(f"Saved {args.ckpt_path} after {step} steps.")


if __name__ == "__main__":
    main()
