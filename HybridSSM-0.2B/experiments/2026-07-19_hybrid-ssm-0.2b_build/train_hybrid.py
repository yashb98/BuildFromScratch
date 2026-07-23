"""train.py — HybridSSM-0.2B trainer (JAX/Flax). §C5.0 smoke-capable + ckpt/resume for the recovery chain.

Smoke (§C5.0): `python3 train.py --smoke` → tiny config, synthetic data, a few steps; asserts imports,
build, finite+trending loss, ckpt save→reload round-trip, exit 0. Real run adds `--data <shards>` +
budget; launched detached under sentinel watch + boot_resume (§C5). Optimizer = AdamW now (optax); the
Muon port (muon_jax.py) is a pre-full-run refinement (the paper uses Muon; AdamW is a valid arm too).
"""
import sys
sys.path.insert(0, "/home/yashb98/Downloads/BuildFromScratch")
import jax_safe_env  # noqa: E402  (before jax, §C1)
import argparse  # noqa: E402
import dataclasses  # noqa: E402
import pickle  # noqa: E402
import pathlib  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import serialization  # noqa: E402
import model as M  # noqa: E402


def make_batch(rng, B, T, V):
    ids = np.asarray(jax.random.randint(rng, (B, T + 1), 0, V))
    return ids[:, :-1], ids[:, 1:]                       # inputs, next-token targets


def load_tokens(path):
    """Load a FineWeb-Edu tokcache (.pt from train_qwen3.stream_tokens — real, decontaminated,
    Qwen3-tokenized; reused for BPB-comparability with the 596M study). Returns uint32 arrays."""
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    tr = d["train"].numpy().astype(np.uint32)
    va = d["val"].numpy().astype(np.uint32)
    return tr, va


def real_batch(rng, toks, B, T):
    """Sample B random contiguous windows of T+1 tokens from the flat token stream."""
    n = toks.shape[0]
    starts = np.asarray(jax.random.randint(rng, (B,), 0, n - T - 1))
    win = np.stack([toks[s:s + T + 1] for s in starts]).astype(np.int32)
    return win[:, :-1], win[:, 1:]


def loss_fn(params, apply, ids, tgt, chunk):
    hidden, emb = apply({"params": params}, ids)
    N = hidden.shape[0] * hidden.shape[1]
    return M.chunked_cross_entropy(hidden.reshape(N, -1), emb, tgt.reshape(N), chunk=chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--mixer", default="ssm")
    ap.add_argument("--attn_every", type=int, default=2)
    ap.add_argument("--nope_on_full", action="store_true")
    ap.add_argument("--ckpt", default="checkpoint.pkl")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None, help="FineWeb-Edu tokcache .pt (real run)")
    ap.add_argument("--tokens", type=int, default=0, help="token budget → steps (real run)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--done_marker", default=None)
    a = ap.parse_args()

    if a.smoke:
        cfg = M.HybridConfig(vocab_size=4096, d_model=256, n_layers=6, n_heads=8, n_kv_heads=2,
                             head_dim=32, ffn=512, mixer=a.mixer, attn_every=a.attn_every,
                             nope_on_full=a.nope_on_full)
        # overfit ONE fixed batch — the real "can it learn" signal (random-per-step has no structure)
        B, T, steps, chunk = 2, 64, 30, 512
    else:
        cfg = M.HybridConfig(mixer=a.mixer, attn_every=a.attn_every, nope_on_full=a.nope_on_full)
        B, T, chunk = a.batch, a.seq, 8192
        steps = a.tokens // (B * T) if a.tokens else a.steps

    toks = valtoks = None
    if a.data:
        toks, valtoks = load_tokens(a.data)
        print(f"[data] loaded {toks.shape[0]:,} train + {valtoks.shape[0]:,} val tokens from {a.data}", flush=True)

    rng = jax.random.PRNGKey(a.seed)
    net = M.HybridSSM(cfg)
    ids0, _ = make_batch(rng, B, T, cfg.vocab_size)
    params = net.init(rng, jnp.asarray(ids0))["params"]
    ne, tot = M.count_params(params)
    print(f"[build] non-embed={ne/1e6:.1f}M total={tot/1e6:.1f}M cfg={cfg.mixer} attn_every={cfg.attn_every} "
          f"nope={cfg.nope_on_full} steps={steps} tok/step={B*T}", flush=True)

    if a.smoke or not a.data:
        opt = optax.adamw(a.lr, weight_decay=0.1)
    else:
        sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, a.warmup, max(steps, a.warmup + 1), end_value=a.lr * 0.1)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.1))
    opt_state = opt.init(params)
    start_step = 0
    if a.resume and pathlib.Path(a.resume).exists():
        with open(a.resume, "rb") as f:
            blob = pickle.load(f)
        params = serialization.from_bytes(params, blob["params"])
        opt_state = serialization.from_bytes(opt_state, blob["opt_state"])
        start_step = blob["step"]
        # Restore the PRNG stream so a resumed run continues the SAME data-window sequence it
        # would have had uninterrupted. Without this, `rng` stays PRNGKey(seed) from above and the
        # loop's first split at step=start_step reproduces step 0's subkey — the resumed run then
        # REPLAYS its own steps 0..start_step-1 data windows across its remaining budget. That is a
        # per-arm data-repetition confound the uninterrupted arms don't carry; it bit the 85M rung
        # after the 2026-07-23 thermal kills (swa128_nope_85M_s0 was killed at step ~7800/11718).
        if "rng" in blob:
            rng = jnp.asarray(blob["rng"])
            print(f"[resume] from {a.resume} at step {start_step} (rng stream restored — exact continuation)", flush=True)
        else:
            print(f"[resume] from {a.resume} at step {start_step} — WARNING: checkpoint predates "
                  f"rng-checkpointing; data windows for steps 0..{start_step-1} will REPLAY across the "
                  f"remaining budget (data-repetition confound). Discard and rerun clean to avoid it.", flush=True)

    @jax.jit
    def step(params, opt_state, ids, tgt):
        loss, grads = jax.value_and_grad(loss_fn)(params, net.apply, ids, tgt, chunk)
        gnorm = optax.global_norm(grads)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, gnorm

    def save(step_i):
        # `rng` is captured at call time (late binding): it holds the post-split stream position at
        # the end of step step_i-1, which is exactly what a resume at step_i must restore to be exact.
        blob = {"params": serialization.to_bytes(params), "opt_state": serialization.to_bytes(opt_state),
                "step": step_i, "rng": np.asarray(rng)}
        tmp = a.ckpt + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(blob, f)
        pathlib.Path(tmp).replace(a.ckpt)

    fixed = make_batch(jax.random.PRNGKey(1234), B, T, cfg.vocab_size) if a.smoke else None
    losses = []
    # NOTE (2026-07-21 fix): the bound is `steps`, NOT `start_step + steps`. The old form
    # made a RESUMED run repeat the whole budget from the resume point — arm ssm_base_s0
    # resumed at 400 and ran 21,156 steps (173.3M tok) against a declared 170.0M (+1.93%).
    # Harmless at n=1, fatal for §C18 iso-FLOP across a ladder: an arm that crashes and
    # resumes would silently receive more compute than one that doesn't, in proportion to
    # its resume point, and nothing in the logs would say so. It also ran the optax cosine
    # schedule (decay_steps=steps) past its end for those extra steps.
    for s in range(start_step, steps):
        rng, sk = jax.random.split(rng)
        if a.smoke:
            ids, tgt = fixed                                 # overfit one fixed batch
        elif toks is not None:
            ids, tgt = real_batch(sk, toks, B, T)            # real FineWeb-Edu windows
        else:
            ids, tgt = make_batch(sk, B, T, cfg.vocab_size)
        params, opt_state, loss, gnorm = step(params, opt_state, jnp.asarray(ids), jnp.asarray(tgt))
        loss = float(loss)
        losses.append(loss)
        assert np.isfinite(loss), f"non-finite loss at step {s}"
        if a.smoke or s % 20 == 0:
            print(f"[step {s}] loss={loss:.4f} grad_norm={float(gnorm):.3f}", flush=True)
        if valtoks is not None and a.eval_every and (s + 1) % a.eval_every == 0:
            rng, vk = jax.random.split(rng)
            vids, vtgt = real_batch(vk, valtoks, B, T)
            vloss = float(loss_fn(params, net.apply, jnp.asarray(vids), jnp.asarray(vtgt), chunk))
            print(f"[eval step {s+1}] val_loss={vloss:.4f}", flush=True)
        if (s + 1) % a.ckpt_every == 0:
            save(s + 1)

    if not a.smoke:
        save(steps)
        if a.done_marker:
            pathlib.Path(a.done_marker).touch()
        # `losses` is empty when a resume finds the cell already complete (start_step >= steps),
        # which is the idempotent no-op the ladder driver relies on — don't IndexError on it.
        final = f"{losses[-1]:.4f}" if losses else "n/a (already complete at resume)"
        print(f"[done] {steps} steps · final loss={final}", flush=True)

    if a.smoke:
        # ckpt round-trip: save, reload into a fresh param tree, assert identical
        save(steps)
        with open(a.ckpt, "rb") as f:
            blob = pickle.load(f)
        fresh = net.init(jax.random.PRNGKey(99), jnp.asarray(ids0))["params"]
        restored = serialization.from_bytes(fresh, blob["params"])
        maxd = max(float(jnp.abs(x - y).max()) for x, y in
                   zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(restored)))
        trend = losses[-1] < losses[0] - 1.0             # overfitting one batch must clearly drop loss
        print(f"[smoke] loss {losses[0]:.4f} -> {losses[-1]:.4f} trending_down={trend} | "
              f"ckpt_roundtrip max|Δ|={maxd:.2e} | step={blob['step']}", flush=True)
        ok = np.isfinite(losses).all() and maxd < 1e-6 and trend
        print(f"SMOKE {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
