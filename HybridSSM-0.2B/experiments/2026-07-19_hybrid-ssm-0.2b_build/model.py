"""HybridSSM-0.2B — a from-scratch hybrid attention-SSM decoder LM (JAX/Flax).

Novel design (brief hybrid-attention-rethink / arXiv 2606.15378): interleaves full GQA attention with
an efficient mixer (SelectiveSSM or SlidingWindowAttention). The composition is configurable so the
single-variable ablations (mixer type · attention fraction · NoPE-on-full-attn) are ONE flag each.

No bit-exact oracle (novel) — verify.py cross-checks the scan vs its sequential reference (ssm.py) and
the full forward vs an independent numpy attention at ~1e-2. Chunked CE for the 151,936 vocab (§C1).
"""
from __future__ import annotations
import dataclasses
import math
import jax
import jax.numpy as jnp
import flax.linen as nn

from ssm import SelectiveSSM, SlidingWindowAttention


@dataclasses.dataclass(frozen=True)
class HybridConfig:
    vocab_size: int = 151_936
    d_model: int = 768
    n_layers: int = 24
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn: int = 2048
    rope_theta: float = 1e4
    rms_eps: float = 1e-6
    # ---- the ablation toggles ----
    attn_every: int = 2          # full-attention layer every k layers (2 → 1:1, 4 → 1:3)
    mixer: str = "ssm"           # efficient-layer mixer: "ssm" | "swa128"
    swa_window: int = 128
    nope_on_full: bool = False   # NoPE on full-attention layers (the paper's headline knob)

    def is_full(self, i: int) -> bool:
        return (i % self.attn_every) == 0


def rms_norm(x, weight, eps):
    dt = x.dtype
    x = x.astype(jnp.float32)
    x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return (weight * x).astype(dt)


def rope(x, theta):
    # x: [B,T,H,D] → rotary. D even.
    B, T, H, D = x.shape
    half = D // 2
    inv = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    ang = jnp.arange(T, dtype=jnp.float32)[:, None] * inv[None, :]        # [T,half]
    cos = jnp.cos(ang)[None, :, None, :]
    sin = jnp.sin(ang)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).astype(x.dtype)


class GQAAttention(nn.Module):
    cfg: HybridConfig
    use_rope: bool = True

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        B, T, _ = x.shape
        nq, nkv, hd = c.n_heads, c.n_kv_heads, c.head_dim
        q = nn.Dense(nq * hd, use_bias=False, name="q")(x).reshape(B, T, nq, hd)
        k = nn.Dense(nkv * hd, use_bias=False, name="k")(x).reshape(B, T, nkv, hd)
        v = nn.Dense(nkv * hd, use_bias=False, name="v")(x).reshape(B, T, nkv, hd)
        if self.use_rope:
            q, k = rope(q, c.rope_theta), rope(k, c.rope_theta)
        rep = nq // nkv
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(hd)
        causal = jnp.tril(jnp.ones((T, T), bool))
        scores = jnp.where(causal[None, None], scores, -1e30)
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(B, T, nq * hd)
        return nn.Dense(c.d_model, use_bias=False, name="o")(out)


class MLP(nn.Module):
    cfg: HybridConfig

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        g = nn.Dense(c.ffn, use_bias=False, name="gate")(x)
        u = nn.Dense(c.ffn, use_bias=False, name="up")(x)
        return nn.Dense(c.d_model, use_bias=False, name="down")(nn.silu(g) * u)


class Block(nn.Module):
    cfg: HybridConfig
    layer_idx: int

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        n1 = self.param("norm1", nn.initializers.ones, (c.d_model,))
        n2 = self.param("norm2", nn.initializers.ones, (c.d_model,))
        h = rms_norm(x, n1, c.rms_eps)
        if c.is_full(self.layer_idx):
            mix = GQAAttention(c, use_rope=not c.nope_on_full, name="attn")(h)
        elif c.mixer == "swa128":
            mix = SlidingWindowAttention(c.d_model, c.n_heads, c.swa_window, name="swa")(h)
        else:
            mix = SelectiveSSM(c.d_model, name="ssm")(h)
        x = x + mix
        x = x + MLP(c, name="mlp")(rms_norm(x, n2, c.rms_eps))
        return x


class HybridSSM(nn.Module):
    cfg: HybridConfig

    @nn.compact
    def __call__(self, input_ids):
        c = self.cfg
        emb = self.param("embed", nn.initializers.normal(1.0 / math.sqrt(c.d_model)),
                         (c.vocab_size, c.d_model))
        x = emb[input_ids]
        # Rematerialize each block: recompute its activations in the backward pass instead of
        # holding all 24 layers' SSM-scan / attention intermediates (~61GB → sentinel-killed at
        # batch4). Numerically identical (same params/schedule → the step-400 ckpt resumes cleanly).
        BlockR = nn.remat(Block)
        for i in range(c.n_layers):
            x = BlockR(c, i, name=f"block_{i}")(x)
        x = rms_norm(x, self.param("norm_f", nn.initializers.ones, (c.d_model,)), c.rms_eps)
        return x, emb            # hidden + tied embedding (logits = x @ emb.T, formed chunked)


def chunked_cross_entropy(hidden, emb, targets, chunk=8192, ignore_index=-100):
    """CE over the tied head WITHOUT materializing (N, vocab) logits (§C1). Streams the vocab in
    chunks for a numerically-stable log-sum-exp + the target logit."""
    N, d = hidden.shape
    V = emb.shape[0]
    hidden = hidden.astype(jnp.float32)
    emb = emb.astype(jnp.float32)
    mask = targets != ignore_index
    tgt = jnp.where(mask, targets, 0)
    tgt_logit = jnp.sum(hidden * emb[tgt], axis=-1)             # [N] correct-class logit
    # streaming max + sumexp over vocab chunks
    running_max = jnp.full((N,), -jnp.inf)
    running_sum = jnp.zeros((N,))
    for s in range(0, V, chunk):
        logits_c = hidden @ emb[s:s + chunk].T                  # [N, chunk]
        cmax = jnp.max(logits_c, axis=-1)
        new_max = jnp.maximum(running_max, cmax)
        running_sum = running_sum * jnp.exp(running_max - new_max) + \
            jnp.sum(jnp.exp(logits_c - new_max[:, None]), axis=-1)
        running_max = new_max
    lse = running_max + jnp.log(running_sum)
    nll = lse - tgt_logit
    return jnp.sum(jnp.where(mask, nll, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


def count_params(params, exclude_embed=True):
    flat = jax.tree_util.tree_leaves(params)
    total = sum(p.size for p in flat)
    emb = 0
    for path, p in jax.tree_util.tree_flatten_with_path(params)[0]:
        if "embed" in jax.tree_util.keystr(path):
            emb += p.size
    return total - (emb if exclude_embed else 0), total
