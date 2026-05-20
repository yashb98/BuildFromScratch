"""
Pytest wrappers around the architecture parity checks.

Runs the same gates as verify.py and the first half of compare_with_hf.py,
but with proper assertions and pytest exit codes so CI can rely on them.

Run from the repo root:
    pytest tests/ -v

If the official SmolLM2-135M weights can't be downloaded (no internet, HF
outage, missing transformers cache), the suite skips rather than failing.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# Make the project root importable when running pytest from anywhere.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def models():
    """Load HF reference and our model once per test session (download once, reuse)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from model_full import SmolLM2ForCausalLM, SmolLM2Config
        from verify import REPO, load_official_weights_into_ours
    except ImportError as e:
        pytest.skip(f"missing dependency: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(REPO)
        hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"can't load {REPO} (no internet or HF cache miss?): {e}")

    ours = SmolLM2ForCausalLM(SmolLM2Config())
    load_official_weights_into_ours(ours, hf.state_dict())
    ours.eval()
    return tokenizer, hf, ours


def test_param_count(models):
    """Unique parameter count must equal 134,515,008 (the published 135M)."""
    _, _, ours = models
    n = sum(p.numel() for p in ours.parameters())
    assert n == 134_515_008, (
        f"got {n:,}; expected 134,515,008 — most likely embeddings are not tied"
    )


def test_tied_embeddings(models):
    """lm_head.weight must be the same tensor as embed_tokens.weight (tied)."""
    _, _, ours = models
    same_ptr = ours.lm_head.weight.data_ptr() == ours.model.embed_tokens.weight.data_ptr()
    assert same_ptr, "lm_head and embed_tokens are not sharing storage — tying is broken"


def test_short_prompt_logit_parity(models):
    """The canonical bit-exact gate: max|Δ logits| < 1e-3 on a short prompt."""
    tokenizer, hf, ours = models
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        hf_out = hf(ids).logits
        our_out = ours(ids)["logits"]
    max_abs = (hf_out - our_out).abs().max().item()
    assert max_abs < 1e-3, f"max|Δlogits| = {max_abs:.3e}; architectural mismatch"


def test_short_prompt_argmax_match(models):
    """Next-token argmax must agree (the loosest possible parity check)."""
    tokenizer, hf, ours = models
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        hf_next = hf(ids).logits[0, -1].argmax().item()
        our_next = ours(ids)["logits"][0, -1].argmax().item()
    assert hf_next == our_next, (
        f"HF picks {tokenizer.decode([hf_next])!r}, "
        f"ours picks {tokenizer.decode([our_next])!r}"
    )


def test_long_context_logit_parity(models):
    """RoPE sanity at 512 tokens: same threshold should hold."""
    tokenizer, hf, ours = models
    long_text = ("In the field of language modeling, " * 60)[:2000]
    ids = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=512).input_ids
    with torch.no_grad():
        hf_log = hf(ids).logits[0, -1].float()
        our_log = ours(ids)["logits"][0, -1].float()
    max_abs = (hf_log - our_log).abs().max().item()
    assert max_abs < 1e-3, (
        f"long-context max|Δ| = {max_abs:.3e}; positional encoding is the likely culprit"
    )
    assert hf_log.argmax() == our_log.argmax(), "long-context next-token argmax disagrees"


def test_per_layer_hidden_state_parity(models):
    """Every transformer block output must agree — catches per-layer divergence."""
    tokenizer, hf, ours = models
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    hf_states: list[torch.Tensor] = []
    our_states: list[torch.Tensor] = []

    def hf_hook(_, __, out):
        hf_states.append(out[0] if isinstance(out, tuple) else out)

    def our_hook(_, __, out):
        our_states.append(out)

    hf_handles = [l.register_forward_hook(hf_hook) for l in hf.model.layers]
    our_handles = [l.register_forward_hook(our_hook) for l in ours.model.layers]
    try:
        with torch.no_grad():
            hf(ids)
            ours(ids)
    finally:
        for h in hf_handles + our_handles:
            h.remove()

    assert len(hf_states) == len(our_states) == 30, (
        f"expected 30 layers, got hf={len(hf_states)} ours={len(our_states)}"
    )
    for i, (a, b) in enumerate(zip(hf_states, our_states)):
        diff = (a - b).abs().max().item()
        assert diff < 1e-3, f"layer {i}: max|Δ hidden| = {diff:.3e}"
