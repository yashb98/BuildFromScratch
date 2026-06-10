"""
Phase-4 tests for the Qwen3-0.6B from-scratch implementation.

Tests build a SHRUNK config (tiny vocab + layers) for speed — these probe
mechanics (shapes, grads, state-dict roundtrip, no-NaN generate), not
numerical equivalence vs HF (that's the Phase-5 verify gate).

The param-count test instantiates the FULL config and checks against 596M
within 1% — that's the one slow test, but it's the right gate that
hidden_size / head_dim / vocab math is correct.

    pytest -v test_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make Qwen3-0.6B/model.py importable regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parents[2]   # .../Qwen3-0.6B
sys.path.insert(0, str(ROOT))

from model import Qwen3Config, Qwen3ForCausalLM, num_params  # noqa: E402


def _tiny_cfg() -> Qwen3Config:
    """Shrunk config for fast tests. Preserves QK-Norm + GQA + head_dim
    indirection (the parts whose shapes can be wrong)."""
    return Qwen3Config(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,        # GQA 2:1
        head_dim=16,                  # decoupled from hidden//n_heads (64/4=16, coincides here — but path is the same)
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )


def test_forward_shapes():
    cfg = _tiny_cfg()
    m = Qwen3ForCausalLM(cfg).eval()
    for B, T in [(1, 1), (1, 8), (2, 16), (4, 32)]:
        x = torch.randint(0, cfg.vocab_size, (B, T))
        out = m(x)
        assert out["logits"].shape == (B, T, cfg.vocab_size), \
            f"logits shape mismatch B={B} T={T}: {out['logits'].shape}"


def test_backward():
    cfg = _tiny_cfg()
    m = Qwen3ForCausalLM(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(x, labels=x)
    out["loss"].backward()
    # Every param with requires_grad should have received a gradient.
    # With tied embeddings, lm_head.weight aliases embed_tokens.weight, so
    # only ONE of the two sees a .grad attribute (PyTorch deduplicates).
    seen = set()
    for n, p in m.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        assert p.grad is not None, f"No grad on {n}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad on {n}"


def test_param_count_full_config():
    """Full-config param count must match published 596M within 1%."""
    cfg = Qwen3Config()
    m = Qwen3ForCausalLM(cfg)
    n = num_params(m)
    expected = 596_049_920
    rel_err = abs(n - expected) / expected
    assert rel_err < 0.01, f"Param count {n:,} differs from {expected:,} by {rel_err:.2%}"


def test_generate_no_nan():
    cfg = _tiny_cfg()
    m = Qwen3ForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = m.generate(x, max_new_tokens=16, temperature=0.8, top_k=10)
    assert out.shape == (1, 4 + 16)
    assert ((out >= 0) & (out < cfg.vocab_size)).all()


def test_load_state_dict_roundtrip():
    cfg = _tiny_cfg()
    a = Qwen3ForCausalLM(cfg).eval()
    b = Qwen3ForCausalLM(cfg).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        la = a(x)["logits"]
        lb = b(x)["logits"]
    assert torch.allclose(la, lb), "state_dict roundtrip changed outputs"


def test_qk_norm_present():
    """Regression guard: QK-Norm weights must exist in state_dict — this
    is the Qwen3-specific bit that would silently break verify if dropped."""
    cfg = _tiny_cfg()
    m = Qwen3ForCausalLM(cfg)
    sd_keys = m.state_dict().keys()
    assert any(k.endswith("self_attn.q_norm.weight") for k in sd_keys), \
        "q_norm.weight missing from state_dict"
    assert any(k.endswith("self_attn.k_norm.weight") for k in sd_keys), \
        "k_norm.weight missing from state_dict"
