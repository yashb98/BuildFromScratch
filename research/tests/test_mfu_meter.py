"""Tests for mfu_meter.py — MFU/HFU arithmetic, pinned peak values, the
estimated-GB10 caveat, and refuse-unknown-device. Stdlib-only (CI-safe)."""
import pytest

import mfu_meter as mm


CFG = {"n_params": 1_000_000_000, "n_layers": 1, "n_heads": 1,
       "head_dim": 1, "seq_len": 1}  # fpt = 6e9 + 12


def test_peak_values_pinned_to_datasheet():
    # guard against accidental edits of the honest denominators (§C2)
    assert mm.PEAK_FLOPS["h100-sxm"]["bf16_dense_tflops"] == 989.5
    assert mm.PEAK_FLOPS["a100-sxm"]["bf16_dense_tflops"] == 312.0
    assert mm.PEAK_FLOPS["a100-sxm"]["estimated"] is False


def test_mfu_matches_formula():
    tps = 50_000.0
    out = mm.compute_mfu(CFG, tps, "a100-sxm")
    fpt = 6.0e9 + 12.0
    expected = (fpt * tps) / (312.0e12)
    assert out["mfu"] == pytest.approx(expected, rel=1e-12)
    assert out["achieved_tflops"] == pytest.approx(fpt * tps / 1e12, rel=1e-12)


def test_hfu_equals_mfu_without_ckpt_and_exceeds_with():
    base = mm.compute_mfu(CFG, 50_000.0, "h100-sxm", act_ckpt=False)
    assert base["hfu"] == pytest.approx(base["mfu"], rel=1e-12)
    ck = mm.compute_mfu(CFG, 50_000.0, "h100-sxm", act_ckpt=True)
    assert ck["hfu"] == pytest.approx(ck["mfu"] * (8.0 / 6.0), rel=1e-12)
    assert ck["hfu"] > ck["mfu"]


def test_gb10_flagged_estimated_with_caveat():
    out = mm.compute_mfu(CFG, 40_000.0, "gb10")
    assert out["peak_is_estimated"] is True
    assert out["caveat"] is not None and "ESTIMATED" in out["caveat"]


def test_unknown_device_refused_not_guessed():
    with pytest.raises(ValueError):
        mm.compute_mfu(CFG, 1000.0, "rtx-9090")
    with pytest.raises(ValueError):
        mm.device_peak_tflops("made-up")


def test_nonpositive_throughput_rejected():
    with pytest.raises(ValueError):
        mm.compute_mfu(CFG, 0.0, "h100-sxm")
