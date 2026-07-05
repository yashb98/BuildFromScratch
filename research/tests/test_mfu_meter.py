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


# ---- measured-roofline support (Frontier-Eng "real measured MFU") ----

def test_measured_peak_drops_estimated_and_caveat():
    tps = 50_000.0
    # A real measured H100 dense bf16 peak (e.g. ~830 TFLOP/s achievable, below
    # the 989.5 datasheet ceiling). estimated must be False, caveat dropped.
    out = mm.compute_mfu(CFG, tps, "h100-sxm", measured_peak_tflops=830.0)
    fpt = 6.0e9 + 12.0
    expected = (fpt * tps) / (830.0e12)
    assert out["mfu"] == pytest.approx(expected, rel=1e-12)
    assert out["peak_is_estimated"] is False
    assert out["peak_is_measured"] is True
    assert out["caveat"] is None
    assert out["device_peak_tflops"] == 830.0


def test_measured_peak_overrides_estimated_gb10():
    # Even on the estimated GB10, a measured peak makes the result non-estimated.
    out = mm.compute_mfu(CFG, 40_000.0, "gb10", measured_peak_tflops=98.0)
    assert out["peak_is_estimated"] is False
    assert out["peak_is_measured"] is True
    assert out["caveat"] is None
    assert out["device_peak_tflops"] == 98.0


def test_gb10_without_measured_still_estimated():
    # Regression: the datasheet/estimated path is unchanged when no peak given.
    out = mm.compute_mfu(CFG, 40_000.0, "gb10")
    assert out["peak_is_estimated"] is True
    assert out["peak_is_measured"] is False
    assert out["caveat"] is not None and "ESTIMATED" in out["caveat"]


def test_device_measured_requires_a_measured_peak():
    with pytest.raises(ValueError):
        mm.compute_mfu(CFG, 1000.0, "measured")  # no peak supplied
    # but works when one is given
    out = mm.compute_mfu(CFG, 1000.0, "measured", measured_peak_tflops=500.0)
    assert out["peak_is_measured"] is True
    assert out["peak_is_estimated"] is False


def test_measured_peak_must_be_positive():
    with pytest.raises(ValueError):
        mm.compute_mfu(CFG, 1000.0, "h100-sxm", measured_peak_tflops=0.0)
    with pytest.raises(ValueError):
        mm.compute_mfu(CFG, 1000.0, "h100-sxm", measured_peak_tflops=-1.0)


def test_record_measured_peak_provenance():
    rec = mm.record_measured_peak(
        830.0, device="h100-sxm", method="gemm_microbench",
        gemm_shape=(8192, 8192, 8192),
        command="python bench_gemm.py --dtype bf16 --m 8192 --n 8192 --k 8192",
        measured_on="rented-h100-2026-06-17")
    assert rec["bf16_dense_tflops"] == 830.0
    assert rec["estimated"] is False
    assert rec["measured"] is True
    assert rec["sparsity"] == "dense"
    assert rec["dtype"] == "bf16"
    assert rec["method"] == "gemm_microbench"
    assert rec["gemm_shape"] == (8192, 8192, 8192)
    # feeds straight into compute_mfu
    out = mm.compute_mfu(CFG, 50_000.0, "h100-sxm", measured_peak=rec)
    assert out["peak_is_measured"] is True
    assert out["device_peak_tflops"] == 830.0


def test_record_measured_peak_refuses_sparse_and_fp4():
    # the dense-no-sparsity discipline: a sparse or FP4 number is NOT a peak
    with pytest.raises(ValueError):
        mm.record_measured_peak(1600.0, device="h100-sxm",
                                method="gemm_microbench", sparsity="2:4")
    with pytest.raises(ValueError):
        mm.record_measured_peak(2000.0, device="gb10",
                                method="gemm_microbench", dtype="fp4")
    with pytest.raises(ValueError):
        mm.record_measured_peak(0.0, device="h100-sxm", method="gemm_microbench")
    with pytest.raises(ValueError):
        mm.record_measured_peak(830.0, device="h100-sxm", method="vibes")


def test_measured_peak_dict_and_scalar_must_agree():
    rec = mm.record_measured_peak(830.0, device="h100-sxm",
                                  method="gemm_microbench")
    # consistent scalar+dict is fine
    out = mm.compute_mfu(CFG, 1000.0, "h100-sxm",
                         measured_peak_tflops=830.0, measured_peak=rec)
    assert out["device_peak_tflops"] == 830.0
    # contradictory scalar+dict is refused
    with pytest.raises(ValueError):
        mm.compute_mfu(CFG, 1000.0, "h100-sxm",
                       measured_peak_tflops=900.0, measured_peak=rec)
