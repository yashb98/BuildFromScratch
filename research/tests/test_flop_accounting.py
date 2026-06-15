"""Tests for flop_accounting.py — the 6N + 12·L·H·Q·T convention and the §C18
iso-FLOP gate. Stdlib-only (CI-safe)."""
import pytest

import flop_accounting as fa


def test_flops_per_token_exact():
    # 6*100 + 12*2*4*8*16 = 600 + 12288 = 12888
    assert fa.flops_per_token(100, 2, 4, 8, 16) == 12888.0


def test_dense_term_dominates_for_big_model():
    # tiny context => attention term ~ negligible; ~6N
    f = fa.flops_per_token(1_000_000_000, 1, 1, 1, 1)
    assert f == 6.0e9 + 12.0


def test_train_flops_scales_with_tokens():
    fpt = fa.flops_per_token(100, 2, 4, 8, 16)
    assert fa.train_flops(100, 2, 4, 8, 16, tokens=1000) == fpt * 1000


def test_iso_flop_within_and_beyond_tol():
    assert fa.iso_flop(1000.0, 1040.0, tol=0.05) is True    # 3.8% apart
    assert fa.iso_flop(1000.0, 1100.0, tol=0.05) is False   # 9.1% apart


def test_rejects_nonpositive():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            fa.flops_per_token(bad, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        fa.train_flops(100, 1, 1, 1, 1, tokens=0)


def test_from_config_roundtrip():
    cfg = {"n_params": 100, "n_layers": 2, "n_heads": 4, "head_dim": 8,
           "seq_len": 16, "tokens": 1000}
    out = fa.from_config(cfg)
    assert out["flops_per_token"] == 12888.0
    assert out["train_flops"] == 12888.0 * 1000
    assert out["formula"] == "6*N + 12*L*H*Q*T"
