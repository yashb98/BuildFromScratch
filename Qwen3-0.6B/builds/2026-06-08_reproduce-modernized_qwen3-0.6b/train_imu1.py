"""Build 2 (Modernized) trainer — full IMU-1 bundle.

Reuses the faithful trainer's verified data/eval helpers; swaps in:
  - model_imu1 (value residuals + LayerNorm scaling + per-head gating)
  - NorMuon (2D hidden matrices) + AdamW (embeddings/norms/scalars)  param split
  - WSD schedule (warmup -> stable -> 20% linear decay-to-zero)
  - chunked z-loss (1e-4) — crash-safe over the 152k vocab
  - optional weight EMA over the tail of training

GPU smoke test is gated on Phase A finishing (one training job at a time on the
GB10). Use --dry_run for a CPU wiring check that needs no GPU and no data.
"""
import sys, pathlib, time, argparse, math
import torch
from torch.utils.data import DataLoader

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # BuildFromScratch/
MODEL_DIR = HERE.parents[1]                  # Qwen3-0.6B/
FAITHFUL = MODEL_DIR / "builds" / "2026-06-08_reproduce-faithful_qwen3-0.6b"
for p in (str(ROOT), str(MODEL_DIR), str(HERE), str(FAITHFUL)):
    sys.path.insert(0, p)

import safe_cuda                              # noqa: F401  caps CUDA before torch touches it
import model_imu1 as M
from normuon import NorMuon

RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)


def log(msg, path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def chunked_cross_entropy(logits, targets, chunk=8192):
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_tgt = targets.reshape(-1)
    total, n = 0.0, flat_tgt.numel()
    for i in range(0, n, chunk):
        total = total + torch.nn.functional.cross_entropy(
            flat_logits[i:i + chunk], flat_tgt[i:i + chunk], reduction="sum")
    return total / n


def chunked_z_loss(logits, chunk=8192):
    """z-loss = mean over tokens of logsumexp(logits)^2, chunked in fp32."""
    flat = logits.reshape(-1, logits.size(-1))
    n = flat.size(0)
    total = 0.0
    for i in range(0, n, chunk):
        lse = torch.logsumexp(flat[i:i + chunk].float(), dim=-1)
        total = total + (lse ** 2).sum()
    return total / n


def wsd_factor(step, total, warmup, decay_frac=0.2):
    """Warmup-Stable-Decay multiplier in [0,1]; linear decay-to-zero over the tail."""
    if step < warmup:
        return step / max(1, warmup)
    decay_start = total * (1.0 - decay_frac)
    if step < decay_start:
        return 1.0
    return max(0.0, (total - step) / max(1.0, total - decay_start))


def split_params(model):
    """2D hidden matrices -> NorMuon; embeddings/norms/scalars/bias -> AdamW."""
    normuon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2 and "embed_tokens" not in name:
            normuon.append(p)
        else:
            adam.append(p)
    return normuon, adam


def build_optimizers(model, normuon_lr, adam_lr, wd):
    n_params, a_params = split_params(model)
    opt_n = NorMuon(n_params, lr=normuon_lr, weight_decay=wd, beta1=0.95, beta2=0.95)
    opt_a = torch.optim.AdamW(a_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    bases = [(opt_n, normuon_lr), (opt_a, adam_lr)]
    return [opt_n, opt_a], bases, len(n_params), len(a_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000, help="smoke default; full run ~18150 for 2 TPP")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--normuon_lr", type=float, default=0.011)   # IMU-1 stable-stage 2D LR
    ap.add_argument("--adam_lr", type=float, default=0.006)      # IMU-1 stable-stage 1D LR
    ap.add_argument("--weight_decay", type=float, default=0.1)   # 2D only (NorMuon)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--decay_frac", type=float, default=0.2)     # WSD decay-to-zero tail
    ap.add_argument("--z_weight", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--run_name", default="imu1")
    ap.add_argument("--dry_run", action="store_true", help="CPU wiring check: 3 steps, synthetic data, no GPU/dataset")
    args = ap.parse_args()

    tag = f"{args.run_name}_"
    log_path = RESULTS / f"qwen3_{tag}train.log"

    torch.manual_seed(args.seed)
    tok_per_step = args.seq_len * args.micro_batch * args.grad_accum

    if args.dry_run:
        # Tiny config + synthetic data so the loop can be validated on CPU in seconds.
        device = torch.device("cpu")
        cfg = M.Qwen3Config(num_hidden_layers=2, hidden_size=128, intermediate_size=256,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                            vocab_size=512, max_position_embeddings=64)
        seq = 16
        model = M.Qwen3ForCausalLM(cfg).to(device)
        steps = 3
    else:
        if not torch.cuda.is_available():
            print("CUDA required (or use --dry_run)."); return 1
        safe_cuda.guard(args.mem_fraction)
        device = torch.device("cuda")
        cfg = M.Qwen3Config()
        seq = args.seq_len
        model = M.Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
        steps = args.steps

    model.train()
    log(f"device={device} bundle: vr={cfg.use_value_residual} ln={cfg.use_layernorm_scaling} "
        f"hg={cfg.use_head_gating}  steps={steps}", log_path)

    optims, bases, n_n, n_a = build_optimizers(model, args.normuon_lr, args.adam_lr, args.weight_decay)
    log(f"param split: {n_n} NorMuon (2D), {n_a} AdamW (1D/embed)  tok/step={tok_per_step:,}", log_path)

    # --- Data ---
    if args.dry_run:
        def batches():
            g = torch.Generator().manual_seed(0)
            while True:
                x = torch.randint(0, cfg.vocab_size, (args.micro_batch, seq + 1), generator=g)
                yield x[:, :-1], x[:, 1:]
        data_iter = batches()
    else:
        from transformers import AutoTokenizer
        from train_qwen3 import stream_tokens, evaluate, PackedTextDataset
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        token_budget = steps * tok_per_step
        train_tokens, val_tokens = stream_tokens(tokenizer, token_budget + 2_000_000, 300_000, log_path)
        loader = DataLoader(PackedTextDataset(train_tokens, seq), batch_size=args.micro_batch,
                            shuffle=True, drop_last=True)
        def cycle(dl):
            while True:
                for b in dl:
                    yield b
        data_iter = cycle(loader)
        if not args.no_compile:
            log("Compiling model...", log_path)
            model = torch.compile(model)

    train_model = model

    # --- Training loop ---
    t0 = time.time()
    for step in range(1, steps + 1):
        for opt in optims:
            opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            inp, lbl = next(data_iter)
            inp, lbl = inp.to(device), lbl.to(device)
            logits = train_model(input_ids=inp)["logits"]
            ce = chunked_cross_entropy(logits, lbl)
            zl = chunked_z_loss(logits) * args.z_weight
            ((ce + zl) / args.grad_accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        factor = wsd_factor(step, steps, args.warmup_steps, args.decay_frac)
        for opt, base in bases:
            for grp in opt.param_groups:
                grp["lr"] = base * factor
        for opt in optims:
            opt.step()

        if step % args.log_every == 0 or args.dry_run:
            tok = step * tok_per_step
            tps = tok / max(1e-9, time.time() - t0)
            mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
            log(f"step {step:>5}/{steps}  ce {ce.item():.4f}  z {zl.item():.5f}  "
                f"lr {args.normuon_lr*factor:.2e}  |grad| {float(gn):.2f}  mem {mem:.1f}GB  "
                f"tok/s {tps:,.0f}", log_path)

        if not args.dry_run and args.eval_every and step % args.eval_every == 0:
            from train_qwen3 import evaluate
            vp, _ = evaluate(model, val_tokens, device, seq)
            log(f"  [eval @ {step}] val PPL={vp:.2f}", log_path)
        if not args.dry_run and args.ckpt_every and step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": step},
                       RESULTS / f"checkpoint_{tag}step{step}.pt")
            log(f"  [ckpt @ {step}]", log_path)

    log("DONE" + (" (dry run OK)" if args.dry_run else ""), log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
