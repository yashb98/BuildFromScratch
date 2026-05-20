"""
Continued pretraining of SmolLM2-135M on roneneldan/TinyStories.

- Starts from the official HF safetensors loaded into our SmolLM2ForCausalLM.
- Default 100M-token budget (~1/4 epoch over the full TinyStories train split).
- Recipe: AdamW(0.9, 0.95), bf16, peak LR 3e-4 (10x lower than from-scratch),
  weight_decay 0.01, clip_grad 1.0, WSD(warmup 200, decay 20%).

Outputs:
    checkpoint_tinystories.pt        - final state_dict + full training recipe
    results/tinystories_train.csv    - per-step (step, loss, lr, grad_norm, peak_mem_mb, tok_seen)
    results/tinystories_train.log    - stdout log
    results/tinystories_before.txt   - base-model generations + eval-loss
    results/tinystories_after.txt    - trained-model generations + eval-loss
    results/tinystories_eval.csv     - mid-training PPL trace (when --eval_every > 0)
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_full import SmolLM2ForCausalLM, SmolLM2Config
from train import PackedTextDataset, make_wsd_scheduler
from verify import REPO, load_official_weights_into_ours

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)

SAMPLE_PROMPTS = [
    "Once upon a time, there was a little",
    "The brave little mouse",
    "In a faraway forest,",
]


def now():
    return time.strftime("%H:%M:%S")


def log(msg: str, log_path: pathlib.Path):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


@torch.no_grad()
def evaluate(model, val_tokens, device, seq_len: int, max_windows: int = 200):
    """Sliding-window CE perplexity over val_tokens (no overlap)."""
    was_training = model.training
    model.eval()
    nlls, n = [], 0
    for begin in range(0, min(len(val_tokens) - seq_len, max_windows * seq_len), seq_len):
        ids = val_tokens[begin:begin + seq_len].unsqueeze(0).to(device)
        logits = model(ids)["logits"][..., :-1, :].float()
        labels = ids[..., 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), reduction="sum")
        nlls.append(loss.item())
        n += labels.numel()
    if was_training:
        model.train()
    return math.exp(sum(nlls) / n), n


@torch.no_grad()
def sample(model, tokenizer, prompts, device, max_new: int = 80,
           temperature: float = 0.7, top_k: int = 40):
    was_training = model.training
    model.eval()
    out = []
    for p in prompts:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        gen = model.generate(ids, max_new_tokens=max_new, temperature=temperature, top_k=top_k)
        out.append(tokenizer.decode(gen[0], skip_special_tokens=True))
    if was_training:
        model.train()
    return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--token_budget", type=int, default=100_000_000)
    ap.add_argument("--peak_lr", type=float, default=3e-4,
                    help="10x lower than fresh-init 3e-3; we are not starting from random")
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--decay_frac", type=float, default=0.20)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=2000,
                    help="0 disables mid-training eval")
    ap.add_argument("--ckpt_every", type=int, default=4000)
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--compile", action="store_true",
                    help="wrap model in torch.compile (mode='reduce-overhead')")
    ap.add_argument("--ckpt_path", default="checkpoint_tinystories.pt")
    ap.add_argument("--resume", default=None,
                    help="path to checkpoint to resume from (restores model, optim, sched, RNG)")
    return ap.parse_args()


def main():
    args = parse_args()
    log_path = RESULTS / "tinystories_train.log"
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if device.type == "cpu":
        dtype = torch.float32
    log(f"device={device}  dtype={dtype}  seed={args.seed}", log_path)
    log(f"args={vars(args)}", log_path)

    # --- Tokenizer & model -------------------------------------------------
    log("Loading tokenizer + official SmolLM2-135M weights...", log_path)
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
    model = SmolLM2ForCausalLM(SmolLM2Config())
    load_official_weights_into_ours(model, hf.state_dict())
    del hf
    model = model.to(device=device, dtype=dtype)

    # --- Dataset -----------------------------------------------------------
    log("Downloading + tokenizing TinyStories (train + validation)...", log_path)
    t0 = time.time()
    train_ds = load_dataset("roneneldan/TinyStories", split="train")
    val_ds = load_dataset("roneneldan/TinyStories", split="validation")
    eos = tokenizer.eos_token_id
    log(f"  train: {len(train_ds):,} stories, val: {len(val_ds):,}", log_path)

    train_buf = []
    target_train_tokens = args.token_budget + 2_000_000
    for ex in train_ds:
        text = ex["text"].strip()
        if not text:
            continue
        train_buf.extend(tokenizer.encode(text, add_special_tokens=False))
        train_buf.append(eos)
        if len(train_buf) >= target_train_tokens:
            break
    train_tokens = torch.tensor(train_buf, dtype=torch.long)
    log(f"  packed {len(train_tokens):,} train tokens in {time.time()-t0:.1f}s", log_path)

    val_buf = []
    for ex in val_ds:
        text = ex["text"].strip()
        if not text:
            continue
        val_buf.extend(tokenizer.encode(text, add_special_tokens=False))
        val_buf.append(eos)
        if len(val_buf) >= 200_000:
            break
    val_tokens = torch.tensor(val_buf, dtype=torch.long)
    log(f"  packed {len(val_tokens):,} val tokens", log_path)

    pack = PackedTextDataset(train_tokens, args.seq_len)
    loader = DataLoader(pack, batch_size=args.micro_batch, shuffle=True, drop_last=True)
    log(f"  {len(pack):,} train windows of {args.seq_len}", log_path)

    # --- Step budget -------------------------------------------------------
    tok_per_step = args.seq_len * args.micro_batch * args.grad_accum
    total_steps = args.token_budget // tok_per_step
    log(f"  tok/step={tok_per_step:,}  total_steps={total_steps:,}", log_path)

    # --- Baseline eval + samples (skipped on resume to keep timing apples-to-apples) ---
    if args.resume is None:
        log("Baseline (BEFORE) eval...", log_path)
        base_ppl, base_n = evaluate(model, val_tokens, device, args.seq_len)
        log(f"  baseline TinyStories-val PPL={base_ppl:.3f}  ({base_n:,} target tokens)", log_path)
        log("Baseline generations:", log_path)
        base_samples = sample(model, tokenizer, SAMPLE_PROMPTS, device, args.max_new)
        with open(RESULTS / "tinystories_before.txt", "w") as f:
            f.write(f"# Baseline SmolLM2-135M on TinyStories\n")
            f.write(f"Validation PPL: {base_ppl:.4f}  ({base_n:,} target tokens)\n\n")
            for p, s in zip(SAMPLE_PROMPTS, base_samples):
                f.write(f"PROMPT: {p}\nOUTPUT: {s}\n\n")
                log(f"  [{p!r}]  {s[:140]}", log_path)
    else:
        base_ppl = None

    # --- Optimizer / scheduler --------------------------------------------
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    optim = AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )
    sched = make_wsd_scheduler(optim, args.warmup_steps, total_steps, args.decay_frac)

    # --- Optional torch.compile -------------------------------------------
    train_model = model
    if args.compile:
        log("Compiling model with torch.compile(mode='reduce-overhead')...", log_path)
        train_model = torch.compile(model, mode="reduce-overhead")

    # --- Resume -----------------------------------------------------------
    start_step = 0
    tok_seen = 0
    if args.resume is not None:
        log(f"Resuming from {args.resume}", log_path)
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
        tok_seen = ck.get("tok_seen", start_step * tok_per_step)
        base_ppl = ck.get("baseline_ppl", base_ppl)
        log(f"  resumed at step={start_step}  tok_seen={tok_seen:,}", log_path)

    # --- Training loop ----------------------------------------------------
    csv_path = RESULTS / "tinystories_train.csv"
    eval_csv_path = RESULTS / "tinystories_eval.csv"
    csv_f = open(csv_path, "a" if args.resume else "w", newline="")
    csv_w = csv.writer(csv_f)
    if args.resume is None:
        csv_w.writerow(["step", "loss", "lr", "grad_norm", "peak_mem_mb", "tok_seen"])
    eval_f = open(eval_csv_path, "a" if args.resume else "w", newline="")
    eval_w = csv.writer(eval_f)
    if args.resume is None:
        eval_w.writerow(["step", "val_ppl", "delta_vs_base"])

    def save_ckpt(step: int, tok_seen: int, trained_ppl: float | None = None):
        ck = {
            "model": model.state_dict(),
            "config": SmolLM2Config().__dict__,
            "step": step,
            "tok_seen": tok_seen,
            "baseline_ppl": base_ppl,
            "trained_ppl": trained_ppl,
            "training_recipe": {
                "seq_len": args.seq_len,
                "micro_batch": args.micro_batch,
                "grad_accum": args.grad_accum,
                "token_budget": args.token_budget,
                "peak_lr": args.peak_lr,
                "warmup_steps": args.warmup_steps,
                "decay_frac": args.decay_frac,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "seed": args.seed,
                "dtype": args.dtype,
                "compile": args.compile,
                "total_steps": total_steps,
            },
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        }
        torch.save(ck, args.ckpt_path)

    model.train()
    step = start_step
    accum = 0
    t0 = time.time()
    log(f"Training {total_steps:,} steps to {args.token_budget:,} tokens "
        f"(starting at step {step})...", log_path)
    optim.zero_grad(set_to_none=True)
    done = False
    while not done:
        for inp, lbl in loader:
            inp = inp.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            logits = train_model(input_ids=inp)["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                lbl.reshape(-1),
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
                tok_seen = step * tok_per_step
                cur_lr = sched.get_last_lr()[0]
                peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
                csv_w.writerow([step, float(loss.item()), cur_lr,
                                float(grad_norm), peak_mem_mb, tok_seen])
                csv_f.flush()

                if step % args.log_every == 0:
                    dt = time.time() - t0
                    tps = (tok_seen - start_step * tok_per_step) / max(1e-9, dt)
                    remaining = args.token_budget - tok_seen
                    eta = remaining / max(1.0, tps)
                    log(f"step {step:>6}/{total_steps}  loss {loss.item():.4f}  "
                        f"lr {cur_lr:.2e}  |grad| {float(grad_norm):.3f}  "
                        f"mem {peak_mem_mb:.0f}MB  "
                        f"tok/s {tps:>7,.0f}  "
                        f"tok {tok_seen/1e6:5.1f}M/{args.token_budget/1e6:.0f}M  "
                        f"ETA {eta/60:5.1f} min", log_path)

                if args.eval_every > 0 and step % args.eval_every == 0:
                    val_ppl, _ = evaluate(model, val_tokens, device, args.seq_len)
                    delta = (val_ppl - base_ppl) if base_ppl is not None else float("nan")
                    eval_w.writerow([step, val_ppl, delta])
                    eval_f.flush()
                    log(f"  [eval @ step {step}] TinyStories-val PPL={val_ppl:.3f}  "
                        f"(Δ vs base {delta:+.3f})", log_path)

                if step % args.ckpt_every == 0 or step == total_steps:
                    save_ckpt(step, tok_seen)
                    log(f"  [ckpt @ step {step}] saved {args.ckpt_path}", log_path)

                if step >= total_steps:
                    done = True
                    break
    csv_f.close()
    eval_f.close()
    log(f"Training complete in {(time.time()-t0)/60:.1f} min.", log_path)

    # --- Post-training eval + samples -------------------------------------
    log("Trained-model (AFTER) eval...", log_path)
    trained_ppl, trained_n = evaluate(model, val_tokens, device, args.seq_len)
    if base_ppl is not None:
        log(f"  AFTER PPL = {trained_ppl:.3f}   "
            f"(BEFORE was {base_ppl:.3f}; improvement {base_ppl - trained_ppl:+.3f} = "
            f"{100*(base_ppl - trained_ppl)/base_ppl:+.1f}%)", log_path)
    else:
        log(f"  AFTER PPL = {trained_ppl:.3f}  (no baseline available on resume)", log_path)
    after_samples = sample(model, tokenizer, SAMPLE_PROMPTS, device, args.max_new)
    with open(RESULTS / "tinystories_after.txt", "w") as f:
        f.write(f"# SmolLM2-135M after {args.token_budget:,} tokens of TinyStories\n")
        f.write(f"Validation PPL: {trained_ppl:.4f}  ({trained_n:,} target tokens)\n")
        if base_ppl is not None:
            f.write(f"BEFORE→AFTER PPL: {base_ppl:.4f} → {trained_ppl:.4f}\n\n")
        for p, s in zip(SAMPLE_PROMPTS, after_samples):
            f.write(f"PROMPT: {p}\nOUTPUT: {s}\n\n")
            log(f"  [{p!r}]  {s[:140]}", log_path)

    save_ckpt(step, tok_seen, trained_ppl=trained_ppl)
    log(f"Saved {args.ckpt_path}", log_path)


if __name__ == "__main__":
    main()
