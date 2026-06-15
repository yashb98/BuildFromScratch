#!/usr/bin/env python3
"""flop_accounting.py — training-FLOP accounting for transformer runs.

The single source of the FLOPs-per-token convention used by mfu_meter.py and by
the §C18 iso-FLOP single-variable-ablation gate. Stdlib-only, pure CPU, no model
load — safe to run anytime, including while a trainer is active (§C1).

Convention (PaLM App. B, arXiv:2204.02311): the dense matmuls cost ~6N FLOPs per
token over a fwd+bwd step (2N fwd + 4N bwd), and self-attention adds a
sequence-dependent term. We use the form the §C5.3 launch probe records:

    flops_per_token = 6*N  +  12 * L * H * Q * T

where N = non-embedding parameter count, L = layers, H = attention heads,
Q = head_dim, T = sequence length (so H*Q = d_model). The 12·L·d_model·T term is
the QKᵀ + AV cost over fwd+bwd; it is the part that makes a longer-context or
deeper arm cost more FLOPs at the SAME token budget — which is exactly why
"token-matched" is not "FLOP-matched" and why §C18 compares train_flops, not
tokens.

This is an approximation (it omits embeddings, norms, and the softmax itself);
it is the same approximation used to report MFU across the field, so MFU numbers
computed from it are comparable. Reported values must always state the formula.
"""
from __future__ import annotations

import argparse
import json
import sys


def flops_per_token(n_params: int, n_layers: int, n_heads: int,
                    head_dim: int, seq_len: int) -> float:
    """6N + 12·L·H·Q·T. n_params is the NON-embedding parameter count (the part
    that does FLOPs every token; tied embeddings are a lookup, not a matmul)."""
    for name, v in (("n_params", n_params), ("n_layers", n_layers),
                    ("n_heads", n_heads), ("head_dim", head_dim),
                    ("seq_len", seq_len)):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"{name} must be a positive number, got {v!r}")
    dense = 6.0 * n_params
    attn = 12.0 * n_layers * n_heads * head_dim * seq_len
    return dense + attn


def train_flops(n_params: int, n_layers: int, n_heads: int, head_dim: int,
                seq_len: int, tokens: int) -> float:
    """Total training FLOPs to consume `tokens` tokens at this config."""
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens!r}")
    return flops_per_token(n_params, n_layers, n_heads, head_dim, seq_len) * tokens


def iso_flop(a: float, b: float, tol: float = 0.05) -> bool:
    """§C18: two arms are FLOP-matched iff their train_flops differ within `tol`
    (default 5%). Used to BLOCK a comparison whose arms did unequal compute."""
    if a <= 0 or b <= 0:
        raise ValueError("train_flops must be positive")
    return abs(a - b) / max(a, b) <= tol


def from_config(cfg: dict) -> dict:
    """Accept a config dict and return the FLOP accounting. Keys:
    n_params, n_layers, n_heads, head_dim, seq_len, [tokens]."""
    fpt = flops_per_token(cfg["n_params"], cfg["n_layers"], cfg["n_heads"],
                          cfg["head_dim"], cfg["seq_len"])
    out = {"flops_per_token": fpt,
           "formula": "6*N + 12*L*H*Q*T",
           "n_params": cfg["n_params"]}
    if cfg.get("tokens"):
        out["train_flops"] = fpt * cfg["tokens"]
        out["tokens"] = cfg["tokens"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-params", type=float, required=True,
                    help="non-embedding parameter count N")
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--tokens", type=float, default=None)
    a = ap.parse_args(argv)
    cfg = {"n_params": a.n_params, "n_layers": a.n_layers, "n_heads": a.n_heads,
           "head_dim": a.head_dim, "seq_len": a.seq_len, "tokens": a.tokens}
    print(json.dumps(from_config(cfg), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
