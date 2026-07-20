"""Efficient-mixer layers for HybridSSM-0.2B (novel design; brief hybrid-attention-rethink).

Two mixers the hybrid interleaves with full GQA attention:
  - SelectiveSSM : a Mamba-2-style diagonal selective state-space mixer, computed as a linear
    recurrence via jax.lax.associative_scan (JAX-native). h_t[c,n] = a_t[c]·h_{t-1}[c,n] + bx_t[c,n],
    y_t[c] = sum_n C_t[n]·h_t[c,n] + D[c]·x_t[c], gated by z. Input-dependent a (selective).
  - SlidingWindowAttention : causal attention restricted to a window (the SWA ablation arm).

CORRECTNESS-FIRST: `selective_scan` (parallel) and `selective_scan_ref` (explicit sequential loop)
must agree — verify.py gates on it before any training (§C5.0 / §C14 novel-design ~1e-2 cross-check).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn
import math


# ───────────────────────── selective-SSM core (pure functions) ─────────────────────────
def _combine(l, r):
    """Associative combine for the linear recurrence h_t = a_t·h_{t-1} + u_t.
    Element = (a, u) with a:[...,C] (per-channel decay, broadcast over state) and u:[...,C,N].
    (a1,u1) ⊕ (a2,u2) = (a1·a2, a2·u1 + u2)."""
    a1, u1 = l
    a2, u2 = r
    return a1 * a2, a2[..., None] * u1 + u2


def selective_scan(a, bx, C, D, x):
    """Parallel selective scan.
      a  : [B,T,C]     per-channel decay a_t (0<a<1)
      bx : [B,T,C,N]   input term b_t[n]·x_t[c]  (already formed)
      C  : [B,T,N]     output projection C_t
      D  : [C]         skip
      x  : [B,T,C]
    returns y : [B,T,C]"""
    a_cum, h = jax.lax.associative_scan(_combine, (a, bx), axis=1)   # h:[B,T,C,N]
    y = jnp.einsum("btcn,btn->btc", h, C) + D * x
    return y


def selective_scan_ref(a, bx, C, D, x):
    """Explicit sequential reference (the correctness oracle for the parallel scan)."""
    B, T, Cn, N = bx.shape
    h = jnp.zeros((B, Cn, N), bx.dtype)
    ys = []
    for t in range(T):
        h = a[:, t][..., None] * h + bx[:, t]           # [B,C,N]
        ys.append(jnp.einsum("bcn,bn->bc", h, C[:, t]))
    return jnp.stack(ys, axis=1) + D * x


class SelectiveSSM(nn.Module):
    d_model: int
    d_state: int = 16
    expand: int = 2
    dt_min: float = 1e-3
    dt_max: float = 1e-1

    @nn.compact
    def __call__(self, x):
        B, T, _ = x.shape
        d_in = self.expand * self.d_model
        # projections: x-path + z gate
        xz = nn.Dense(2 * d_in, use_bias=False, name="in_proj")(x)
        xin, z = jnp.split(xz, 2, axis=-1)                       # [B,T,d_in] each
        xin = nn.silu(xin)
        # input-dependent dt, B, C
        dt = nn.softplus(nn.Dense(d_in, name="dt_proj")(x))     # [B,T,d_in] > 0
        Bc = nn.Dense(self.d_state, use_bias=False, name="B_proj")(x)   # [B,T,N]
        Cc = nn.Dense(self.d_state, use_bias=False, name="C_proj")(x)   # [B,T,N]
        # A (negative, per-channel-state), D skip
        A_log = self.param("A_log", lambda k: jnp.log(jnp.linspace(1.0, self.d_state, d_in)))
        A = -jnp.exp(A_log)                                     # [d_in] < 0  (diag, state broadcast)
        D = self.param("D", nn.initializers.ones, (d_in,))
        a = jnp.exp(dt * A)                                     # [B,T,d_in]  in (0,1)
        bx = (dt[..., None] * Bc[:, :, None, :]) * xin[..., None]   # [B,T,d_in,N]
        y = selective_scan(a, bx, Cc, D, xin)                  # [B,T,d_in]
        y = y * nn.silu(z)                                     # gate
        return nn.Dense(self.d_model, use_bias=False, name="out_proj")(y)


class SlidingWindowAttention(nn.Module):
    """Causal attention restricted to the last `window` tokens (the SWA ablation mixer)."""
    d_model: int
    n_heads: int = 12
    window: int = 128

    @nn.compact
    def __call__(self, x):
        B, T, _ = x.shape
        hd = self.d_model // self.n_heads
        qkv = nn.Dense(3 * self.d_model, use_bias=False, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        shp = (B, T, self.n_heads, hd)
        q, k, v = q.reshape(shp), k.reshape(shp), v.reshape(shp)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(hd)
        i = jnp.arange(T)[:, None]
        j = jnp.arange(T)[None, :]
        mask = (j <= i) & (j > i - self.window)                # causal AND within window
        scores = jnp.where(mask[None, None], scores, -1e30)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(B, T, self.d_model)
        return nn.Dense(self.d_model, use_bias=False, name="o")(out)
