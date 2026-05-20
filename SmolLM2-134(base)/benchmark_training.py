"""
Measure training throughput on the live device for SmolLM2-135M
under a realistic continued-pretraining config (bf16, AdamW, full backward).

Result is tokens/sec, which lets us extrapolate training time for any
domain-corpus size and token budget.
"""
from __future__ import annotations

import argparse
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model_full import SmolLM2ForCausalLM, SmolLM2Config

ap = argparse.ArgumentParser()
ap.add_argument("--compile", action="store_true",
                help="wrap model in torch.compile(mode='reduce-overhead') before measuring")
args = ap.parse_args()

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

cfg = SmolLM2Config()
model = SmolLM2ForCausalLM(cfg).to(device=device, dtype=dtype)
model.train()
opt = AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

run_model = model
if args.compile:
    print("Compiling model with torch.compile(mode='reduce-overhead')...")
    run_model = torch.compile(model, mode="reduce-overhead")

CONFIGS = [
    # (seq_len, micro_batch, grad_accum)  — total step tokens = seq*mb*ga
    (1024, 4, 1),
    (1024, 8, 1),
    (2048, 4, 1),
    (2048, 2, 4),   # gradient accumulation
]

def step(seq_len, mb, ga):
    ids = torch.randint(0, cfg.vocab_size, (mb, seq_len), device=device)
    opt.zero_grad(set_to_none=True)
    for _ in range(ga):
        logits = run_model(input_ids=ids)["logits"]
        loss = F.cross_entropy(
            logits[..., :-1, :].reshape(-1, logits.size(-1)).float(),
            ids[..., 1:].reshape(-1),
        )
        (loss / ga).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

print(f"Device: {device}   dtype: {dtype}")
print()
print(f"{'seq_len':>8}  {'mb':>4}  {'ga':>4}  {'tok/step':>10}  "
      f"{'sec/step':>10}  {'tok/s':>10}  {'tok/s eff':>12}")
print("-" * 80)
for seq, mb, ga in CONFIGS:
    # Warmup
    try:
        for _ in range(2):
            step(seq, mb, ga)
        if device.type == "cuda":
            torch.cuda.synchronize()
        # Timed
        N = 5
        t0 = time.time()
        for _ in range(N):
            step(seq, mb, ga)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / N
        tok = seq * mb * ga
        print(f"{seq:>8}  {mb:>4}  {ga:>4}  {tok:>10,}  "
              f"{dt:>10.3f}  {tok/dt:>10,.0f}  {tok/dt:>12,.0f}")
    except torch.cuda.OutOfMemoryError:
        print(f"{seq:>8}  {mb:>4}  {ga:>4}  OOM")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"{seq:>8}  {mb:>4}  {ga:>4}  OOM ({type(e).__name__})")
            torch.cuda.empty_cache()
        else:
            raise

if device.type == "cuda":
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak CUDA memory used: {mem:.2f} GB")
