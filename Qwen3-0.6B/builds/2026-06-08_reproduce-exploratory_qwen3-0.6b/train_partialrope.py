"""Build 3 (Exploratory) trainer — partial RoPE.

Identical recipe to the faithful baseline (AdamW + cosine, best LR 2.4e-3) so the
ONLY difference vs baseline is partial_rotary_factor — a clean controlled ablation.
Reuses the faithful trainer's verified data/eval/scheduler helpers; swaps the model
for model_partialrope with a configurable rotated fraction.
"""
import sys, pathlib, time, argparse, math
import torch
from torch.utils.data import DataLoader

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_DIR = HERE.parents[1]
FAITHFUL = MODEL_DIR / "builds" / "2026-06-08_reproduce-faithful_qwen3-0.6b"
for p in (str(ROOT), str(MODEL_DIR), str(HERE), str(FAITHFUL)):
    sys.path.insert(0, p)

import safe_cuda  # noqa: F401
import model_partialrope as M
from train_qwen3 import (stream_tokens, evaluate, PackedTextDataset,
                         chunked_cross_entropy, make_cosine_scheduler)

RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)


def log(msg, path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=18150)        # ~2 TPP at 65,536 tok/step
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--peak_lr", type=float, default=2.4e-3)   # Phase A winner
    ap.add_argument("--end_lr", type=float, default=3.2e-4)
    ap.add_argument("--warmup_steps", type=int, default=900)
    ap.add_argument("--partial_rotary_factor", type=float, required=True)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--run_name", required=True)
    args = ap.parse_args()

    tag = f"{args.run_name}_"
    log_path = RESULTS / f"qwen3_{tag}train.log"
    if not torch.cuda.is_available():
        print("CUDA required."); return 1
    safe_cuda.guard(args.mem_fraction)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    cfg = M.Qwen3Config(partial_rotary_factor=args.partial_rotary_factor)
    tok_per_step = args.seq_len * args.micro_batch * args.grad_accum
    token_budget = args.steps * tok_per_step
    log(f"partial RoPE factor={args.partial_rotary_factor}  steps={args.steps:,}  "
        f"tok/step={tok_per_step:,}  budget={token_budget:,}  peak_lr={args.peak_lr}", log_path)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    train_tokens, val_tokens = stream_tokens(tokenizer, token_budget + 2_000_000, 300_000, log_path)
    loader = DataLoader(PackedTextDataset(train_tokens, args.seq_len),
                        batch_size=args.micro_batch, shuffle=True, drop_last=True)

    model = M.Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    model.train()
    log(f"rotary_dim={model.model.rotary_dim} / head_dim={cfg.head_dim}", log_path)

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8)
    sched = make_cosine_scheduler(optim, args.warmup_steps, args.steps, args.peak_lr, args.end_lr)

    train_model = model if args.no_compile else torch.compile(model)
    log("Baseline eval...", log_path)
    base_ppl, _ = evaluate(model, val_tokens, device, args.seq_len)
    log(f"  baseline val PPL={base_ppl:.2f}", log_path)

    def cycle(dl):
        while True:
            for b in dl:
                yield b
    it = cycle(loader)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        optim.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            inp, lbl = next(it)
            inp, lbl = inp.to(device), lbl.to(device)
            logits = train_model(input_ids=inp)["logits"]
            (chunked_cross_entropy(logits, lbl) / args.grad_accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step(); sched.step()
        if step % args.log_every == 0:
            tok = step * tok_per_step
            tps = tok / max(1e-9, time.time() - t0)
            eta = (token_budget - tok) / max(1.0, tps)
            log(f"step {step:>6}/{args.steps}  loss {chunked_cross_entropy(logits, lbl).item():.4f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  |grad| {float(gn):.2f}  "
                f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB  tok/s {tps:,.0f}  ETA {eta/3600:.1f}h", log_path)
        if args.eval_every and step % args.eval_every == 0:
            vp, _ = evaluate(model, val_tokens, device, args.seq_len)
            log(f"  [eval @ {step}] val PPL={vp:.2f}", log_path)
        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": step},
                       RESULTS / f"checkpoint_{tag}.pt")
    vp, _ = evaluate(model, val_tokens, device, args.seq_len)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": args.steps},
               RESULTS / f"checkpoint_{tag}.pt")
    (RESULTS / f"qwen3_{tag}after.txt").write_text(
        f"partial RoPE factor={args.partial_rotary_factor} rotary_dim={model.model.rotary_dim}\n"
        f"val PPL: {base_ppl:.2f} -> {vp:.2f}\n")
    log(f"DONE  final val PPL={vp:.2f}", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
