"""Single-variable ABLATION trainer for the IMU-1 bundle de-confounding ladder
(Phase 1). ONE parameterized trainer so every arm differs from the faithful
baseline by EXACTLY ONE flag (the §C18 single-variable gate is then trivially
auditable):

  --arch      {faithful|imu1}   model_imu1 with the 3 IMU-1 flags off|on
                                (flags-off is bit-identical to faithful model.py,
                                 verified max|Δlogits|=0.0)
  --schedule  {cosine|wsd}      faithful cosine-to-floor | WSD decay-to-zero
  --zloss     <float>           0.0 (none) | 1e-4 (IMU-1 z-loss)
  --optimizer {adamw|normuon}   faithful single-AdamW | NorMuon(2D)+AdamW split
                                (Phase 1 holds this at adamw — the NorMuon axis is
                                 already isolated; optimizer↔LR is confounded)

Baseline arm = --arch faithful --schedule cosine --zloss 0 --optimizer adamw,
which reproduces train_qwen3.py's recipe (AdamW lr 1.7e-3, betas (0.9,0.95),
eps 1e-8, wd-split 0.01/0.0, cosine warmup→floor=end_lr/peak_lr, chunked fp32 CE,
no aux loss). Each Phase-1 arm flips ONE of {schedule, zloss, arch}.

Copied from the modernized build's train_imu1.py (never edits a canonical file,
§C4.4); reuses the faithful build's verified data/eval helpers (same FineWeb-Edu
sample-10BT doc-disjoint decontaminated stream + eval). Carries the safe_cuda
header (§C1) and chunked CE (151,936 vocab). ADDS --resume (template lacked it):
restores model + ALL optimizers + step + RNG so the cohort survives nights.

Usage (GPU):  python train_ablation.py --arch faithful --schedule cosine --zloss 0 \
                  --optimizer adamw --steps 2000 --warmup_steps 100 --seed 0 \
                  --ckpt_path <dir>/checkpoint_baseline_seed0.pt --run_name baseline_seed0
       (CPU wiring check, no GPU/data):  add --dry_run
"""
import sys, pathlib, time, argparse, math
import torch
from torch.utils.data import DataLoader

HERE = pathlib.Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[1]                   # Qwen3-0.6B/
ROOT = HERE.parents[2]                        # BuildFromScratch/
# resolve build dirs at RUNTIME (no baked dated names, §C2)
MODERN = next(MODEL_DIR.glob("builds/*_reproduce-modernized_*"))
FAITHFUL = next(MODEL_DIR.glob("builds/*_reproduce-faithful_*"))
for p in (str(ROOT), str(MODEL_DIR), str(MODERN), str(FAITHFUL), str(HERE)):
    sys.path.insert(0, p)

import safe_cuda                              # noqa: F401  caps CUDA before torch touches it
import model_imu1 as M                        # flags-off == faithful model.py (bit-exact verified)
from normuon import NorMuon


def log(msg, path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def chunked_cross_entropy(logits, targets, chunk=8192):
    """fp32 chunked CE (matches the faithful trainer's .float() cast)."""
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
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


def cosine_factor(step, total, warmup, floor):
    """Faithful schedule: linear warmup, then cosine peak→floor (floor=end/peak)."""
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * prog))


def wsd_factor(step, total, warmup, decay_frac=0.2):
    """WSD: warmup → stable(1.0) → linear decay-to-ZERO over the last decay_frac."""
    if step < warmup:
        return step / max(1, warmup)
    decay_start = total * (1.0 - decay_frac)
    if step < decay_start:
        return 1.0
    return max(0.0, (total - step) / max(1.0, total - decay_start))


def split_2d(model):
    """2D non-embedding matrices vs the rest (1D norms/scalars/bias + embeddings)."""
    mats, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (mats if (p.dim() == 2 and "embed_tokens" not in name) else rest).append(p)
    return mats, rest


def build_optimizers(model, mode, peak_lr, normuon_lr, adam_lr, wd):
    """adamw: ONE AdamW over all params, wd-split (dim>=2 -> wd, dim<2 -> 0) — the
    faithful recipe. normuon: NorMuon on 2D matrices + AdamW on the rest (IMU-1).
    Returns (optims, bases=[(opt, base_lr)...], n_a, n_b)."""
    if mode == "adamw":
        decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
            lr=peak_lr, betas=(0.9, 0.95), eps=1e-8)
        return [opt], [(opt, peak_lr)], len(decay), len(no_decay)
    mats, rest = split_2d(model)
    opt_n = NorMuon(mats, lr=normuon_lr, weight_decay=0.1, beta1=0.95, beta2=0.95)
    opt_a = torch.optim.AdamW(rest, lr=adam_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    return [opt_n, opt_a], [(opt_n, normuon_lr), (opt_a, adam_lr)], len(mats), len(rest)


def _strip(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    # axis toggles (the single variables)
    ap.add_argument("--arch", choices=["faithful", "imu1"], default="faithful")
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    ap.add_argument("--optimizer", choices=["adamw", "normuon"], default="adamw")
    ap.add_argument("--zloss", type=float, default=0.0)        # 0 = none; 1e-4 = IMU-1
    ap.add_argument("--data", choices=["fineweb","dclm"], default="fineweb")  # data-selection A/B
    # faithful recipe knobs
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--peak_lr", type=float, default=1.7e-3)   # faithful AdamW LR
    ap.add_argument("--end_lr", type=float, default=3.2e-4)    # cosine floor target
    ap.add_argument("--normuon_lr", type=float, default=0.011)
    ap.add_argument("--adam_lr", type=float, default=0.006)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--run_name", default="ablation")
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    log_path = HERE / f"train_{args.run_name}.log"
    ckpt_path = pathlib.Path(args.ckpt_path) if args.ckpt_path else HERE / f"checkpoint_{args.run_name}.pt"
    arch_on = (args.arch == "imu1")
    torch.manual_seed(args.seed)
    tok_per_step = args.seq_len * args.micro_batch * args.grad_accum

    if args.dry_run:
        device = torch.device("cpu")
        cfg = M.Qwen3Config(num_hidden_layers=2, hidden_size=128, intermediate_size=256,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                            vocab_size=512, max_position_embeddings=64,
                            use_value_residual=arch_on, use_layernorm_scaling=arch_on,
                            use_head_gating=arch_on)
        seq, steps = 16, 5
        model = M.Qwen3ForCausalLM(cfg).to(device)
    else:
        if not torch.cuda.is_available():
            print("CUDA required (or use --dry_run)."); return 1
        safe_cuda.guard(args.mem_fraction)
        device = torch.device("cuda")
        cfg = M.Qwen3Config(use_value_residual=arch_on, use_layernorm_scaling=arch_on,
                            use_head_gating=arch_on)
        seq, steps = args.seq_len, args.steps
        model = M.Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)

    model.train()
    log(f"ARM arch={args.arch} schedule={args.schedule} optimizer={args.optimizer} "
        f"zloss={args.zloss} | vr={cfg.use_value_residual} ln={cfg.use_layernorm_scaling} "
        f"hg={cfg.use_head_gating} | steps={steps} seed={args.seed} tok/step={tok_per_step:,}", log_path)

    optims, bases, n_a, n_b = build_optimizers(model, args.optimizer, args.peak_lr,
                                               args.normuon_lr, args.adam_lr, args.weight_decay)
    floor = args.end_lr / args.peak_lr
    log(f"optimizer split: {n_a} / {n_b} param tensors", log_path)

    start_step = 1
    if args.resume and pathlib.Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(_strip(ck["model"]))
        for o, st in zip(optims, ck["optims"]):
            o.load_state_dict(st)
        start_step = ck["step"] + 1
        if ck.get("rng") is not None:
            torch.set_rng_state(ck["rng"].cpu() if hasattr(ck["rng"], "cpu") else ck["rng"])
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu() if hasattr(ck["cuda_rng"], "cpu") else ck["cuda_rng"])
        log(f"RESUMED from {args.resume} @ step {start_step}", log_path)

    # --- Data ---
    if args.dry_run:
        def batches():
            g = torch.Generator().manual_seed(0)
            while True:
                x = torch.randint(0, cfg.vocab_size, (args.micro_batch, seq + 1), generator=g)
                yield x[:, :-1], x[:, 1:]
        data_iter, val_tokens = batches(), None
    else:
        from transformers import AutoTokenizer
        from train_qwen3 import stream_tokens, evaluate, PackedTextDataset
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        token_budget = steps * tok_per_step
        if args.data == "dclm":
            import numpy as np, glob as _g
            DD = str(ROOT / "research/datasets/data-selection-dclm-edu")
            sh = sorted(_g.glob(DD + "/shard_*.bin"))
            tr = np.concatenate([np.fromfile(p, dtype=np.uint32) for p in sh]).astype(np.int64)
            train_tokens = torch.from_numpy(tr)
            val_tokens = torch.from_numpy(np.fromfile(DD + "/eval_tokens.bin", dtype=np.uint32).astype(np.int64)[:300_000])
            log(f"DATA=dclm-edu: {len(tr):,} train + {len(val_tokens):,} val tokens (shards {len(sh)})", log_path)
        else:
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

    def save_ckpt(step):
        torch.save({"model": _strip(model.state_dict()), "optims": [o.state_dict() for o in optims],
                    "step": step, "config": cfg.__dict__,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None},
                   ckpt_path)

    # --- Training loop ---
    t0 = time.time()
    for step in range(start_step, steps + 1):
        for opt in optims:
            opt.zero_grad(set_to_none=True)
        ce = zl = None
        for _ in range(args.grad_accum):
            inp, lbl = next(data_iter)
            inp, lbl = inp.to(device), lbl.to(device)
            logits = model(input_ids=inp)["logits"]
            ce = chunked_cross_entropy(logits, lbl)
            loss = ce
            if args.zloss > 0:
                zl = chunked_z_loss(logits) * args.zloss
                loss = ce + zl
            (loss / args.grad_accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        factor = (cosine_factor(step, steps, args.warmup_steps, floor) if args.schedule == "cosine"
                  else wsd_factor(step, steps, args.warmup_steps, args.decay_frac))
        for opt, base in bases:
            for grp in opt.param_groups:
                grp["lr"] = base * factor
        for opt in optims:
            opt.step()

        if step % args.log_every == 0 or args.dry_run:
            tok = (step - start_step + 1) * tok_per_step
            tps = tok / max(1e-9, time.time() - t0)
            mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
            zmsg = f"z {zl.item():.5f}  " if zl is not None else ""
            log(f"step {step:>5}/{steps}  ce {ce.item():.4f}  {zmsg}lr {bases[0][1]*factor:.2e}  "
                f"|grad| {float(gn):.2f}  mem {mem:.1f}GB  tok/s {tps:,.0f}", log_path)

        if not args.dry_run and args.eval_every and step % args.eval_every == 0:
            from train_qwen3 import evaluate
            vp, _ = evaluate(model, val_tokens, device, seq)
            log(f"  [eval @ {step}] val PPL={vp:.2f}", log_path)
        if not args.dry_run and args.ckpt_every and step % args.ckpt_every == 0:
            save_ckpt(step)
            log(f"  [ckpt @ {step}] -> {ckpt_path.name}", log_path)

    if not args.dry_run:
        save_ckpt(steps)
    log("DONE" + (" (dry run OK)" if args.dry_run else ""), log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
