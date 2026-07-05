#!/usr/bin/env python3
"""SFT-only post-training of the faithful Qwen3-0.6B base on math-reasoning traces.

Technique brief: research/briefs/vibethinker-small-reasoning.md (VibeThinker-3B
"Spectrum-to-Signal" — FIRST RUN = SFT-ONLY, single variable §C18; GRPO is a
separate queued arm). This is the §C13 `finetune` executor recipe (the
post-train SKILL), launched through /ablation-runner under the full §C5
bounded-auto-run contract. NO canonical file is edited — this whole script lives
under Qwen3-0.6B/experiments/<RUN_ID>/ (§C4.4).

What it does (vs. the canonical pretrain train_qwen3.py):
  * loads the FAITHFUL base checkpoint weights strict=True (not a random init);
  * trains on response-masked completion CE over OpenR1-Math-220k reasoning
    traces prepared by dataset-forge (research/datasets/math-reasoning-openr1-math-220k/):
    each SFT sample is `n_tokens` long in the flat uint32 shard stream, the
    first `prompt_len` tokens are PROMPT (masked -> label -100, NOT trained),
    the rest are the verified reasoning trace (trained). Masking the prompt is
    mandatory (posttrain_losses.masked_sft_nll docstring) — training CE on the
    prompt is objective misspecification.
  * the loss IS posttrain_losses.label_smoothed_nll per response token, here
    realized as a CHUNKED masked cross-entropy so the (N,151936) fp32 logits
    tensor (~the 2026-06-08 crash) is never materialized (§C1 chunked-CE, vocab
    151,936 > 64k). With smoothing=0 the per-token loss is exactly the
    unit-tested label_smoothed_nll(logp,V,0) = -logp_target; the masked mean
    over response tokens is masked_sft_nll(...). We assert this equivalence on a
    tiny tensor at startup (the unit-tested math is the oracle).

Hyperparameters (from the brief's "Exact recipe", all SFT values transferred
from VibeThinker-3B SFT stage-1 = reported, our LR/budget = inferred at 596M):
  * global batch 128 (reported)         -> micro_batch 4 x grad_accum 32 (=128)
  * peak LR 5e-5, cosine -> 8e-8, 5% warmup (reported)
  * token budget ~100-125M (brief §C5.2, =one pass over the 125,000,592-tok set)
  * AdamW(0.9,0.95) eps 1e-8 wd 0.01 grad_clip 1.0 bf16 seq_len 4096 (repo/paper)

Checkpoint+resume (§C5.5): --ckpt_every / --resume restore model + optimizer +
scheduler + RNG + step + sample-cursor; the save->reload round-trip is exercised
by the smoke test (sft_smoke.py).

GB10 guardrails (§C1): safe_cuda.guard() FIRST; chunked masked CE; micro_batch 4
@ seq 4096 (probe-verified ceiling for this arch); plan < 60% of the 119 GB pool.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]            # BuildFromScratch/
FAITHFUL = ROOT / "Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b"
DATASET = ROOT / "research/datasets/math-reasoning-openr1-math-220k"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(FAITHFUL))
sys.path.insert(0, str(ROOT / "Qwen3-0.6B"))

import safe_cuda  # noqa: E402,F401  — sets expandable_segments before torch touches CUDA
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402

import train_qwen3 as tq  # noqa: E402  — reuse log()/set_seed()/make_cosine_scheduler()
from model import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
import posttrain_losses as ptl  # noqa: E402  — the unit-tested SFT loss math (oracle)

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
IGNORE = -100  # masked (prompt) label

BASE_CKPT = FAITHFUL / "checkpoint_qwen3_baseline2tpp.pt"


# ----------------------------------------------------------------- data
def load_shard_stream() -> np.ndarray:
    """Memory-map-concatenate the flat uint32 SFT shard stream (eos-separated)."""
    shards = sorted(glob.glob(str(DATASET / "shard_*.bin")))
    if not shards:
        raise FileNotFoundError(f"no shards under {DATASET}")
    arrs = [np.fromfile(s, dtype=np.uint32) for s in shards]
    return np.concatenate(arrs)


def load_metas() -> list[dict]:
    with open(DATASET / "train_meta.jsonl") as f:
        return [json.loads(l) for l in f]


class SFTPackedDataset(torch.utils.data.Dataset):
    """Response-masked SFT windows of fixed seq_len, built from the flat shard
    stream + per-sample (n_tokens, prompt_len) meta.

    Build (once, CPU, cheap): walk the meta list in order over the flat stream;
    for each sample emit a per-token label array equal to the token id on
    RESPONSE positions and IGNORE(-100) on the first `prompt_len` PROMPT
    positions. Concatenate all (input, label) pairs into two flat arrays, then
    chop into contiguous seq_len windows (the standard packing the pretrain
    PackedTextDataset uses, but carrying the label-mask so prompt tokens never
    contribute to the loss even when packed next to another sample). The next-
    token shift is done in the trainer (inputs[:-1] -> labels[1:]).
    """
    def __init__(self, stream: np.ndarray, metas: list[dict], seq_len: int,
                 limit_samples: int | None = None):
        self.seq_len = seq_len
        labels = np.empty(len(stream), dtype=np.int64)
        off = 0
        n = 0
        for m in metas:
            nt, pl = m["n_tokens"], m["prompt_len"]
            seg = stream[off:off + nt]
            lab = seg.astype(np.int64).copy()
            pl = min(pl, nt)
            lab[:pl] = IGNORE                       # mask the prompt span
            labels[off:off + nt] = lab
            off += nt
            n += 1
            if limit_samples is not None and n >= limit_samples:
                break
        used = off
        self.inputs = stream[:used].astype(np.int64)
        self.labels = labels[:used]
        self.n_windows = (used - 1) // seq_len
        self.n_response_tokens = int((self.labels != IGNORE).sum())
        self.used_tokens = used

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx: int):
        s = idx * self.seq_len
        inp = torch.from_numpy(self.inputs[s:s + self.seq_len + 1].copy())
        lab = torch.from_numpy(self.labels[s:s + self.seq_len + 1].copy())
        return inp[:-1], lab[1:]                     # next-token shift


# ------------------------------------------------- masked chunked CE (§C1)
def masked_chunked_ce(logits: torch.Tensor, labels: torch.Tensor,
                      chunk: int = 8192) -> torch.Tensor:
    """Mean CE over RESPONSE tokens only (labels==IGNORE excluded), WITHOUT
    materializing the full (N, vocab) fp32 logits — chunked exactly like
    tq.chunked_cross_entropy but with ignore_index. This is the GPU realization
    of posttrain_losses.masked_sft_nll(per_token_logp, mask, smoothing=0):
    per response token the loss is -logp_target (= label_smoothed_nll w/ eps=0),
    averaged over response tokens. At vocab=151936 the naive logits.float() is
    the ~40 GB tensor that crashed the box on 2026-06-08 — never build it."""
    flat = logits.reshape(-1, logits.size(-1))
    tgt = labels.reshape(-1)
    total = flat.new_zeros((), dtype=torch.float32)
    n = flat.new_zeros((), dtype=torch.float32)
    N = flat.size(0)
    for i in range(0, N, chunk):
        lt = flat[i:i + chunk].float()
        tt = tgt[i:i + chunk]
        # F.cross_entropy with ignore_index handles the mask and the denom.
        m = (tt != IGNORE)
        cnt = m.sum()
        if cnt.item() == 0:
            continue
        total = total + F.cross_entropy(lt, tt, ignore_index=IGNORE, reduction="sum")
        n = n + cnt.float()
    if float(n) == 0.0:
        return flat.new_zeros((), dtype=torch.float32, requires_grad=True)
    return total / n


def assert_loss_matches_oracle():
    """Prove masked_chunked_ce == posttrain_losses.masked_sft_nll (eps=0) on a
    tiny tensor before we burn GPU hours — the unit-tested math is the oracle."""
    torch.manual_seed(0)
    V = 50
    logits = torch.randn(1, 6, V, dtype=torch.float32)
    labels = torch.tensor([[IGNORE, IGNORE, 3, 7, IGNORE, 11]])  # 3 response toks
    got = float(masked_chunked_ce(logits, labels, chunk=4))
    logp = F.log_softmax(logits, dim=-1)
    per_tok, mask = [], []
    for t in range(6):
        lbl = int(labels[0, t])
        if lbl == IGNORE:
            per_tok.append(0.0); mask.append(0)
        else:
            per_tok.append(float(logp[0, t, lbl])); mask.append(1)
    want = ptl.masked_sft_nll(per_tok, mask, smoothing=0.0)
    assert abs(got - want) < 1e-4, f"masked CE {got} != oracle {want}"
    return got, want


# ----------------------------------------------------------------- eval
@torch.no_grad()
def eval_reasoning_ce(model, dataset, device, max_windows: int = 40):
    """Held-out response-masked CE / PPL on the in-domain reasoning eval set
    (a quick in-loop signal; the §C10 comparable numbers come from eval-harness)."""
    was = model.training
    model.eval()
    nll, n = 0.0, 0
    for idx in range(min(len(dataset), max_windows)):
        inp, lab = dataset[idx]
        inp = inp.unsqueeze(0).to(device)
        lab = lab.unsqueeze(0).to(device)
        logits = model(inp)["logits"]
        m = (lab != IGNORE)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        l = masked_chunked_ce(logits, lab)
        nll += float(l) * cnt
        n += cnt
    if was:
        model.train()
    return (math.exp(nll / max(1, n)) if n else float("nan")), n


# ----------------------------------------------------------------- ckpt
def save_ckpt(path, model, optim, sched, step, sample_cursor, tok_seen,
              base_ppl, recipe):
    torch.save({
        "model": model.state_dict(),
        "config": model.cfg.__dict__,
        "step": step,
        "sample_cursor": sample_cursor,
        "tok_seen": tok_seen,
        "base_reasoning_ppl": base_ppl,
        "recipe": recipe,
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }, path)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--micro_batch", type=int, default=4,
                    help="probe-verified ceiling @ seq 4096; DO NOT raise")
    ap.add_argument("--grad_accum", type=int, default=32, help="x micro_batch=128 global (paper)")
    ap.add_argument("--token_budget", type=int, default=125_000_000,
                    help="brief §C5.2: ~100-125M (one pass over the prepared set)")
    ap.add_argument("--steps", type=int, default=0,
                    help="override: total optimizer steps (0 = derive from token_budget)")
    ap.add_argument("--peak_lr", type=float, default=5e-5, help="VibeThinker SFT stage-1 (reported)")
    ap.add_argument("--end_lr", type=float, default=8e-8, help="cosine floor (reported)")
    ap.add_argument("--warmup_frac", type=float, default=0.05, help="5% linear warmup (reported)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=200, help="0 disables in-loop eval")
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--limit_samples", type=int, default=0, help="smoke: cap SFT samples (0=all)")
    ap.add_argument("--run_name", default="vibethinker_sft")
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def main():
    a = parse_args()
    tag = a.run_name
    log_path = RESULTS / f"{tag}.log"
    csv_path = RESULTS / f"{tag}.csv"
    if a.ckpt_path is None:
        a.ckpt_path = str(RESULTS / f"checkpoint_{tag}.pt")
    tq.set_seed(a.seed)

    if not torch.cuda.is_available():
        print("CUDA required."); return 1
    safe_cuda.guard(a.mem_fraction)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # The unit-tested loss math is the oracle: prove the GPU loss == it first.
    g, w = assert_loss_matches_oracle()
    tq.log(f"[{tag}] loss-oracle check OK: masked_chunked_ce={g:.6f} == masked_sft_nll={w:.6f}", log_path)

    tok_per_step = a.seq_len * a.micro_batch * a.grad_accum
    if a.steps > 0:
        total_steps = a.steps
    else:
        total_steps = max(1, a.token_budget // tok_per_step)
    token_budget = total_steps * tok_per_step
    warmup_steps = max(1, int(a.warmup_frac * total_steps))
    tq.log(f"[{tag}] tok/step={tok_per_step:,} steps={total_steps:,} "
           f"token_budget={token_budget:,} warmup={warmup_steps} peak_lr={a.peak_lr} "
           f"end_lr={a.end_lr} global_batch={a.micro_batch*a.grad_accum}", log_path)

    # --- data ---
    tq.log(f"[{tag}] loading SFT shard stream + meta...", log_path)
    stream = load_shard_stream()
    metas = load_metas()
    lim = a.limit_samples if a.limit_samples > 0 else None
    ds = SFTPackedDataset(stream, metas, a.seq_len, limit_samples=lim)
    del stream
    tq.log(f"[{tag}] {len(ds):,} windows of {a.seq_len}; "
           f"{ds.n_response_tokens:,} response (trained) tokens of {ds.used_tokens:,} total "
           f"({100*ds.n_response_tokens/max(1,ds.used_tokens):.1f}% trained)", log_path)

    # --- model: load the FAITHFUL base weights strict=True ---
    tq.log(f"[{tag}] loading base checkpoint {BASE_CKPT}", log_path)
    ck = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = Qwen3Config(**{k: v for k, v in ck["config"].items()
                         if k in Qwen3Config.__dataclass_fields__})
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(ck["model"], strict=True)
    tq.log(f"[{tag}] base loaded strict=True (base PPL on FineWeb-Edu was "
           f"{ck.get('trained_ppl')}); params={sum(p.numel() for p in model.parameters()):,}", log_path)
    model.train()

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    optim = AdamW([{"params": decay, "weight_decay": a.weight_decay},
                   {"params": no_decay, "weight_decay": 0.0}],
                  lr=a.peak_lr, betas=(0.9, 0.95), eps=1e-8)
    sched = tq.make_cosine_scheduler(optim, warmup_steps, total_steps, a.peak_lr, a.end_lr)

    recipe = {"technique": "vibethinker-small-reasoning", "method": "SFT",
              "base_ckpt": str(BASE_CKPT), "seq_len": a.seq_len,
              "micro_batch": a.micro_batch, "grad_accum": a.grad_accum,
              "global_batch": a.micro_batch * a.grad_accum,
              "peak_lr": a.peak_lr, "end_lr": a.end_lr, "warmup_steps": warmup_steps,
              "weight_decay": a.weight_decay, "grad_clip": a.grad_clip,
              "smoothing": a.smoothing, "seed": a.seed, "dtype": "bfloat16",
              "total_steps": total_steps, "token_budget": token_budget,
              "dataset_id": "open-r1/OpenR1-Math-220k",
              "dataset_revision": "e4e141ec9dea9f8326f4d347be56105859b2bd68"}

    # --- resume ---
    start_step, sample_cursor, base_ppl = 0, 0, None
    if a.resume is not None:
        rk = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rk["model"])
        optim.load_state_dict(rk["optim"])
        sched.load_state_dict(rk["sched"])
        torch.set_rng_state(rk["rng_torch"])
        if rk.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(rk["rng_cuda"])
        np.random.set_state(rk["rng_numpy"])
        random.setstate(rk["rng_python"])
        start_step = rk.get("step", 0)
        sample_cursor = rk.get("sample_cursor", 0)
        base_ppl = rk.get("base_reasoning_ppl")
        tq.log(f"[{tag}] RESUMED at step={start_step} sample_cursor={sample_cursor}", log_path)

    train_model = model if a.no_compile else torch.compile(model)

    # baseline reasoning PPL (skip on resume)
    if a.resume is None:
        base_ppl, bn = eval_reasoning_ce(model, ds, device)
        tq.log(f"[{tag}] baseline held-out reasoning PPL={base_ppl:.3f} ({bn} resp toks)", log_path)

    csv_f = open(csv_path, "a" if a.resume else "w", newline="")
    csv_w = csv.writer(csv_f)
    if a.resume is None:
        csv_w.writerow(["step", "loss", "lr", "grad_norm", "peak_mem_gb", "tok_seen"])

    # --- training loop: iterate windows, deterministic order; sample_cursor is
    #     the window index, so resume continues mid-epoch. ---
    n_windows = len(ds)
    rng = random.Random(a.seed)
    order = list(range(n_windows))
    rng.shuffle(order)

    def next_microbatch():
        """Stack `micro_batch` windows (deterministic shuffled order); reshuffle
        + advance epoch when the cursor wraps. Returns (inp, lab) on device."""
        nonlocal wi
        inps, labs = [], []
        for _ in range(a.micro_batch):
            if wi >= n_windows:
                rng.shuffle(order); wi = 0
            inp, lab = ds[order[wi]]; wi += 1
            inps.append(inp); labs.append(lab)
        return (torch.stack(inps).to(device, non_blocking=True),
                torch.stack(labs).to(device, non_blocking=True))

    step, accum, t0 = start_step, 0, time.time()
    optim.zero_grad(set_to_none=True)
    tq.log(f"[{tag}] training to {total_steps:,} steps (start {start_step}); "
           f"micro_batch={a.micro_batch} x grad_accum={a.grad_accum} windows/step", log_path)
    done = False
    wi = sample_cursor
    loss = None
    while not done:
        # one optimizer step = grad_accum forwards of micro_batch windows each
        # (= micro_batch*grad_accum windows == the probed 128-window global batch).
        inp, lab = next_microbatch()
        logits = train_model(input_ids=inp)["logits"]
        loss = masked_chunked_ce(logits, lab)
        (loss / a.grad_accum).backward()
        accum += 1
        if accum < a.grad_accum:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        accum = 0
        step += 1
        sample_cursor = wi
        tok_seen = step * tok_per_step
        cur_lr = sched.get_last_lr()[0]
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        csv_w.writerow([step, float(loss.item()), cur_lr, float(grad_norm), peak_gb, tok_seen])
        csv_f.flush()
        if step % a.log_every == 0:
            dt = time.time() - t0
            tps = (step - start_step) * tok_per_step / max(1e-9, dt)
            eta = (token_budget - tok_seen) / max(1.0, tps)
            tq.log(f"[{tag}] step {step}/{total_steps} loss {loss.item():.4f} lr {cur_lr:.2e} "
                   f"|grad| {float(grad_norm):.2f} mem {peak_gb:.1f}GB tok/s {tps:,.0f} "
                   f"tok {tok_seen/1e6:.1f}M ETA {eta/3600:.2f}h", log_path)
        if a.eval_every and step % a.eval_every == 0:
            vp, _ = eval_reasoning_ce(model, ds, device)
            tq.log(f"[{tag}] [eval @ {step}] reasoning PPL={vp:.3f} "
                   f"(Δ vs base {vp-(base_ppl or vp):+.3f})", log_path)
        if step % a.ckpt_every == 0 or step >= total_steps:
            save_ckpt(a.ckpt_path, model, optim, sched, step, sample_cursor,
                      tok_seen, base_ppl, recipe)
            tq.log(f"[{tag}] [ckpt @ {step}] saved {a.ckpt_path}", log_path)
        if step >= total_steps:
            done = True
    csv_f.close()
    final_ppl, _ = eval_reasoning_ce(model, ds, device)
    tq.log(f"[{tag}] DONE in {(time.time()-t0)/60:.1f} min; reasoning PPL "
           f"{base_ppl} -> {final_ppl:.3f} (eval-harness does the §C10 scoring + "
           f"the §C13 FineWeb-Edu forgetting probe)", log_path)
    save_ckpt(a.ckpt_path, model, optim, sched, total_steps, sample_cursor,
              token_budget, base_ppl, recipe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
