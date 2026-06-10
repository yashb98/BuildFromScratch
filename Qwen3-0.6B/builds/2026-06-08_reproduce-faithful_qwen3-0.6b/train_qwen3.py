"""
From-scratch training for Qwen3-0.6B (random init) — Recipe A (Faithful-scaled).

Recipe (training_plan.md, anchored to Qwen3 tech report §recipe-baseline):
    - AdamW(betas=(0.9, 0.95), eps=1e-8)        Qwen3 defaults
    - cosine schedule, peak 1.7e-3 -> end 3.2e-4  paper LR shape (scaled budget)
    - weight_decay 0.01, grad_clip 1.0           paper / SmolLM2 train.py
    - bf16, seq_len 4096                          paper stage-1 context
    - data: FineWeb-Edu sample-10BT (streamed)   closest public proxy

HARDWARE GUARDRAILS (this is a GB10 unified-memory box — see safe_cuda.py):
    - safe_cuda.guard() caps the process so an over-allocation raises a clean
      torch.OutOfMemoryError instead of crashing the whole machine.
    - micro_batch=4 x grad_accum=4 (effective 16 seqs). The throughput probe
      proved micro_batch>=8 OOMs at seq_len=4096; do NOT raise micro_batch.
    - cross-entropy is CHUNKED: never materialize the full (N, 151936) fp32
      logits tensor (~40 GB at the training shape — that crash is why this
      file exists).

Defaults run the ~1000-step SMOKE validation (loss goes down, ckpts save, no
OOM over a sustained run). Scale up via --steps / --token_budget for the real run.

    python train_qwen3.py                 # smoke: 1000 steps, compiled
    python train_qwen3.py --steps 15000   # a real Config-A-ish run
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # BuildFromScratch/
sys.path.insert(0, str(REPO_ROOT))
import safe_cuda  # noqa: E402,F401  — sets expandable_segments before torch touches CUDA

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_DIR = pathlib.Path(__file__).resolve().parents[2]  # Qwen3-0.6B/
sys.path.insert(0, str(MODEL_DIR))
from model import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

REPO = "Qwen/Qwen3-0.6B-Base"
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

SAMPLE_PROMPTS = [
    "The capital of France is",
    "In a recent study, researchers found that",
    "def add(a, b):",
]


def now():
    return time.strftime("%H:%M:%S")


def log(msg: str, log_path: pathlib.Path):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Mean token CE without materializing the full (N, vocab) fp32 tensor.
    At vocab=151936 the naive logits.float() is ~40 GB at the training shape and
    crashed this box on 2026-06-08; chunking caps the fp32 working set."""
    flat = logits.reshape(-1, logits.size(-1))
    tgt = targets.reshape(-1)
    total = flat.new_zeros((), dtype=torch.float32)
    n = flat.size(0)
    for i in range(0, n, chunk):
        total = total + F.cross_entropy(
            flat[i:i + chunk].float(), tgt[i:i + chunk], reduction="sum"
        )
    return total / n


def make_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int,
                          peak_lr: float, end_lr: float):
    """Linear warmup to peak, then cosine decay to end_lr (as a fraction of peak)."""
    floor = end_lr / peak_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * prog))

    return LambdaLR(optimizer, lr_lambda)


class PackedTextDataset(torch.utils.data.Dataset):
    """Pre-tokenized contiguous windows of seq_len (mirrors SmolLM2 train.py)."""
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


def stream_tokens(tokenizer, n_train: int, n_val: int, log_path) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream FineWeb-Edu sample-10BT, tokenize on the fly, fill a train buffer
    then a disjoint val buffer (no full-dataset download). Cached to disk so a
    re-run with the same budget skips the ~90s stream+tokenize."""
    cache = RESULTS / f"tokcache_{n_train}_{n_val}.pt"
    if cache.exists():
        d = torch.load(cache)
        log(f"  loaded cached tokens from {cache.name} ({len(d['train']):,} train + {len(d['val']):,} val)", log_path)
        return d["train"], d["val"]
    eos = tokenizer.eos_token_id or tokenizer.encode("<|endoftext|>")[0]
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    buf, val = [], []
    t0 = time.time()
    for ex in ds:
        text = ex.get("text", "").strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        (buf if len(buf) < n_train else val).extend(ids + [eos])
        if len(val) >= n_val:
            break
    log(f"  streamed {len(buf):,} train + {len(val):,} val tokens in {time.time()-t0:.1f}s", log_path)
    train_t = torch.tensor(buf[:n_train], dtype=torch.long)
    val_t = torch.tensor(val[:n_val], dtype=torch.long)
    torch.save({"train": train_t, "val": val_t}, cache)
    return train_t, val_t


@torch.no_grad()
def evaluate(model, val_tokens, device, seq_len: int, max_windows: int = 50):
    was_training = model.training
    model.eval()
    nlls, n = 0.0, 0
    for begin in range(0, min(len(val_tokens) - seq_len, max_windows * seq_len), seq_len):
        ids = val_tokens[begin:begin + seq_len + 1].unsqueeze(0).to(device)
        logits = model(ids[:, :-1])["logits"]
        loss = chunked_cross_entropy(logits, ids[:, 1:]) * (ids.size(1) - 1)
        nlls += loss.item()
        n += ids.size(1) - 1
    if was_training:
        model.train()
    return math.exp(nlls / max(1, n)), n


@torch.no_grad()
def sample(model, tokenizer, prompts, device, max_new: int = 60):
    was_training = model.training
    model.eval()
    out = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        gen = model.generate(ids, max_new_tokens=max_new, temperature=0.8, top_k=40)
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
    ap.add_argument("--steps", type=int, default=1000, help="total optimizer steps (default = smoke run)")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--micro_batch", type=int, default=4, help="DO NOT raise; >=8 OOMs at seq 4096 (probe-verified)")
    ap.add_argument("--grad_accum", type=int, default=4, help="effective batch = micro_batch * grad_accum seqs")
    ap.add_argument("--peak_lr", type=float, default=1.7e-3, help="Qwen3 paper baseline peak")
    ap.add_argument("--end_lr", type=float, default=3.2e-4, help="Qwen3 paper baseline floor")
    ap.add_argument("--warmup_steps", type=int, default=50, help="paper uses 1000; smaller for the smoke run")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mem_fraction", type=float, default=0.85, help="safe_cuda cap; lower (0.6) if editor/browser open")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=250, help="0 disables mid-training eval")
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--no_compile", action="store_true", help="disable torch.compile (default: ON)")
    ap.add_argument("--run_name", default="", help="tag for output files (csv/log/after/ckpt) so runs don't clobber each other")
    ap.add_argument("--ckpt_path", default=None, help="defaults to checkpoint_qwen3[_<run_name>].pt")
    ap.add_argument("--resume", default=None, help="path to a checkpoint to resume from (restores model/optim/sched/RNG/step)")
    return ap.parse_args()


def main():
    args = parse_args()
    tag = f"{args.run_name}_" if args.run_name else ""
    log_path = RESULTS / f"qwen3_{tag}train.log"
    csv_path = RESULTS / f"qwen3_{tag}train.csv"
    after_path = RESULTS / f"qwen3_{tag}after.txt"
    if args.ckpt_path is None:
        args.ckpt_path = str(pathlib.Path(__file__).resolve().parent /
                             f"checkpoint_qwen3{('_' + args.run_name) if args.run_name else ''}.pt")
    set_seed(args.seed)

    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1
    safe_cuda.guard(args.mem_fraction)
    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    log(f"device=cuda dtype={dtype} seed={args.seed}", log_path)
    log(f"args={vars(args)}", log_path)

    tok_per_step = args.seq_len * args.micro_batch * args.grad_accum
    token_budget = args.steps * tok_per_step
    log(f"tok/step={tok_per_step:,}  steps={args.steps:,}  token_budget={token_budget:,}", log_path)

    # --- Tokenizer + data --------------------------------------------------
    log("Loading tokenizer...", log_path)
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    log("Streaming FineWeb-Edu sample-10BT...", log_path)
    train_tokens, val_tokens = stream_tokens(
        tokenizer, n_train=token_budget + 2_000_000, n_val=300_000, log_path=log_path)
    pack = PackedTextDataset(train_tokens, args.seq_len)
    loader = DataLoader(pack, batch_size=args.micro_batch, shuffle=True, drop_last=True)
    log(f"  {len(pack):,} train windows of {args.seq_len}", log_path)

    # --- Model (random init) ----------------------------------------------
    log("Initializing Qwen3-0.6B from scratch (random init)...", log_path)
    cfg = Qwen3Config()
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
    model.train()

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    optim = AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )
    sched = make_cosine_scheduler(optim, args.warmup_steps, args.steps, args.peak_lr, args.end_lr)

    # --- Resume (restore model/optim/sched/RNG/step before any eval) -------
    start_step = 0
    base_ppl = None
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
        if ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        if "rng_numpy" in ck:
            np.random.set_state(ck["rng_numpy"])
        if "rng_python" in ck:
            random.setstate(ck["rng_python"])
        start_step = ck.get("step", 0)
        base_ppl = ck.get("baseline_ppl")
        log(f"  resumed at step={start_step} (baseline_ppl={base_ppl})", log_path)

    train_model = model
    if not args.no_compile:
        # DEFAULT mode (Inductor kernel fusion), NOT reduce-overhead: the latter's
        # CUDA graphs use a static output buffer that breaks under gradient
        # accumulation ("accessing tensor output of CUDAGraphs ... overwritten").
        # Default mode is ~as fast here (~6.9k vs 7.2k tok/s) without that footgun.
        log("Compiling model with torch.compile() [default mode]...", log_path)
        train_model = torch.compile(model)

    # --- Baseline (random-init) eval (skipped on resume) ------------------
    if args.resume is None:
        log("Baseline (random-init) eval...", log_path)
        base_ppl, base_n = evaluate(model, val_tokens, device, args.seq_len)
        log(f"  baseline val PPL={base_ppl:.2f} ({base_n:,} tokens) — expect ~vocab_size ({cfg.vocab_size:,}) at init", log_path)

    # --- CSV / ckpt (append on resume so the curve stays continuous) ------
    csv_f = open(csv_path, "a" if args.resume else "w", newline="")
    csv_w = csv.writer(csv_f)
    if args.resume is None:
        csv_w.writerow(["step", "loss", "lr", "grad_norm", "peak_mem_gb", "tok_seen"])

    def save_ckpt(step: int, tok_seen: int, trained_ppl=None):
        torch.save({
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "step": step,
            "tok_seen": tok_seen,
            "baseline_ppl": base_ppl,
            "trained_ppl": trained_ppl,
            "training_recipe": {
                "recipe": "A_faithful_scaled_cosine",
                "seq_len": args.seq_len, "micro_batch": args.micro_batch,
                "grad_accum": args.grad_accum, "token_budget": token_budget,
                "peak_lr": args.peak_lr, "end_lr": args.end_lr,
                "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip, "betas": (0.9, 0.95), "eps": 1e-8,
                "seed": args.seed, "dtype": args.dtype, "compile": not args.no_compile,
                "total_steps": args.steps,
            },
            "optim": optim.state_dict(), "sched": sched.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
        }, args.ckpt_path)

    # --- Training loop -----------------------------------------------------
    step, accum = start_step, 0
    t0 = time.time()
    log(f"Training to {args.steps:,} steps (starting at {step})...", log_path)
    optim.zero_grad(set_to_none=True)
    done = False
    while not done:
        for inp, lbl in loader:
            inp = inp.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            logits = train_model(input_ids=inp)["logits"]
            loss = chunked_cross_entropy(logits, lbl)
            (loss / args.grad_accum).backward()
            accum += 1
            if accum < args.grad_accum:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            tok_seen = step * tok_per_step
            cur_lr = sched.get_last_lr()[0]
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            csv_w.writerow([step, float(loss.item()), cur_lr, float(grad_norm), peak_gb, tok_seen])
            csv_f.flush()

            if step % args.log_every == 0:
                dt = time.time() - t0
                tps = tok_seen / max(1e-9, dt)
                eta = (token_budget - tok_seen) / max(1.0, tps)
                log(f"step {step:>5}/{args.steps}  loss {loss.item():.4f}  lr {cur_lr:.2e}  "
                    f"|grad| {float(grad_norm):.2f}  mem {peak_gb:.1f}GB  tok/s {tps:>6,.0f}  "
                    f"tok {tok_seen/1e6:.1f}M  ETA {eta/60:.1f}min", log_path)

            if args.eval_every and step % args.eval_every == 0:
                vp, _ = evaluate(model, val_tokens, device, args.seq_len)
                log(f"  [eval @ {step}] val PPL={vp:.2f} (Δ vs base {vp-base_ppl:+.2f})", log_path)

            if step % args.ckpt_every == 0 or step >= args.steps:
                save_ckpt(step, tok_seen)
                log(f"  [ckpt @ {step}] saved {args.ckpt_path}", log_path)

            if step >= args.steps:
                done = True
                break
    csv_f.close()
    log(f"Training complete in {(time.time()-t0)/60:.1f} min.", log_path)

    # --- After eval + samples ---------------------------------------------
    trained_ppl, _ = evaluate(model, val_tokens, device, args.seq_len)
    if base_ppl is not None:
        log(f"AFTER val PPL={trained_ppl:.2f} (BEFORE {base_ppl:.2f}; "
            f"improvement {base_ppl-trained_ppl:+.2f} = {100*(base_ppl-trained_ppl)/base_ppl:+.1f}%)", log_path)
    else:
        log(f"AFTER val PPL={trained_ppl:.2f} (no baseline available on resume)", log_path)
    samples = sample(model, tokenizer, SAMPLE_PROMPTS, device)
    with open(after_path, "w") as f:
        f.write(f"# Qwen3-0.6B from-scratch after {token_budget:,} tokens (smoke={args.steps==1000})\n")
        f.write(f"val PPL: {base_ppl if base_ppl is None else round(base_ppl,2)} -> {trained_ppl:.2f}\n\n")
        for p, s in zip(SAMPLE_PROMPTS, samples):
            f.write(f"PROMPT: {p}\nOUTPUT: {s}\n\n")
            log(f"  [{p!r}] {s[:120]}", log_path)
    save_ckpt(args.steps, args.steps * tok_per_step, trained_ppl=trained_ppl)
    log(f"Saved {args.ckpt_path}", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
