"""Tests for research/kernel_oracle.py — the correctness-first kernel gate (§C21).

The single most important property: the oracle PASSES a correct candidate and
FAILS (discards) a subtly-wrong one, across a shape x dtype sweep, WITHOUT a GPU.
A fast-but-wrong kernel must never be allowed through to the roofline stage.

Pure CPU, numpy-only. torch is optional and only exercised opportunistically.
"""
import math

import numpy as np
import pytest

import kernel_oracle as ko


# ----------------------------------------------------------- tolerance policy

def test_dtype_name_normalizes_aliases():
    assert ko._dtype_name("torch.float16") == "float16"
    assert ko._dtype_name("fp32") == "float32"
    assert ko._dtype_name("bf16") == "bfloat16"
    assert ko._dtype_name("half") == "float16"
    assert ko._dtype_name(None) == "float32"


def test_tolerance_per_dtype_is_looser_for_low_precision():
    assert ko.tolerance_for("float32")["rtol"] < ko.tolerance_for("float16")["rtol"]
    assert ko.tolerance_for("float16")["rtol"] < ko.tolerance_for("bfloat16")["rtol"]
    # unknown dtype falls back to the conservative fp32 tol, never silently loose
    assert ko.tolerance_for("int4-quark") == ko.DEFAULT_TOL


# ----------------------------------------------------------- compare()

def test_compare_identical_passes():
    a = np.linspace(-3, 3, 100)
    r = ko.compare(a, a.copy(), dtype="float32")
    assert r["passed"] is True
    assert r["n_mismatch"] == 0
    assert r["max_abs_err"] == 0.0


def test_compare_accepts_plain_lists():
    r = ko.compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r["passed"] is True


def test_compare_small_drift_within_fp16_tol_passes_but_fails_fp32():
    ref = np.ones(64)
    cand = ref + 5e-4                       # drift ~5e-4
    assert ko.compare(ref, cand, dtype="float16")["passed"] is True
    assert ko.compare(ref, cand, dtype="float32")["passed"] is False


def test_compare_shape_mismatch_is_structural_fail():
    r = ko.compare(np.zeros((4, 4)), np.zeros((4, 5)))
    assert r["passed"] is False
    assert "shape mismatch" in r["reason"]


def test_compare_nan_in_candidate_is_hard_fail():
    ref = np.ones(10)
    cand = ref.copy()
    cand[3] = np.nan
    r = ko.compare(ref, cand, dtype="bfloat16")   # even at the loosest tol
    assert r["passed"] is False
    assert "non-finite" in r["reason"]


def test_compare_inf_is_hard_fail():
    ref = np.ones(10)
    cand = ref.copy()
    cand[0] = np.inf
    assert ko.compare(ref, cand)["passed"] is False


# ----------------------------------------------------------- the gate verdict

SHAPES = [(8, 256), (1, 4096), (32, 1024)]
DTYPES = ("float32", "float16", "bfloat16")


def test_gate_passes_correct_rmsnorm_candidate():
    v = ko.gate(ko.rmsnorm_ref, ko.rmsnorm_candidate_correct, SHAPES, DTYPES)
    assert v["passed"] is True
    assert v["n_failed"] == 0
    assert v["n_cells"] == len(SHAPES) * len(DTYPES)
    assert "PASS" in v["summary"]


def test_gate_discards_subtly_wrong_rmsnorm_candidate():
    v = ko.gate(ko.rmsnorm_ref, ko.rmsnorm_candidate_wrong, SHAPES, DTYPES)
    assert v["passed"] is False
    assert v["n_failed"] >= 1
    assert "DISCARDED" in v["summary"]
    # every failing cell carries a numeric reason, not a crash
    for cell in v["cells"]:
        if not cell["passed"]:
            assert cell["n_mismatch"] != 0


def test_gate_discards_wrong_silu_candidate():
    # silu vs sigmoid: same shape, very different values -> must fail
    v = ko.gate(ko.silu_ref, ko.silu_candidate_wrong, SHAPES, ("float32",))
    assert v["passed"] is False


def test_gate_passes_silu_against_itself():
    v = ko.gate(ko.silu_ref, ko.silu_ref, SHAPES, DTYPES)
    assert v["passed"] is True


def test_gate_reference_and_candidate_get_identical_inputs():
    # If the two fns received different random inputs, an identity-vs-identity
    # comparison would fail. It must pass -> same inputs feed both arms.
    seen = {}

    def ref(x, dtype="float32"):
        seen["ref"] = x.copy()
        return x * 2.0

    def cand(x, dtype="float32"):
        seen["cand"] = x.copy()
        return x * 2.0

    v = ko.gate(ref, cand, [(16,)], ("float32",))
    assert v["passed"] is True
    np.testing.assert_array_equal(seen["ref"], seen["cand"])


def test_gate_is_deterministic_across_runs():
    v1 = ko.gate(ko.rmsnorm_ref, ko.rmsnorm_candidate_wrong, SHAPES, DTYPES, seed=7)
    v2 = ko.gate(ko.rmsnorm_ref, ko.rmsnorm_candidate_wrong, SHAPES, DTYPES, seed=7)
    assert v1["n_failed"] == v2["n_failed"]
    for c1, c2 in zip(v1["cells"], v2["cells"]):
        assert c1["max_abs_err"] == c2["max_abs_err"]


def test_gate_candidate_that_raises_is_fail_not_crash():
    def boom(x, dtype="float32"):
        raise RuntimeError("kernel launch failed")

    v = ko.gate(ko.rmsnorm_ref, boom, [(8, 8)], ("float32",))
    assert v["passed"] is False
    assert "raised" in v["cells"][0]["reason"]


def test_gate_rejects_empty_sweeps():
    with pytest.raises(ValueError):
        ko.gate(ko.silu_ref, ko.silu_ref, [], ("float32",))
    with pytest.raises(ValueError):
        ko.gate(ko.silu_ref, ko.silu_ref, [(8,)], ())


def test_gate_custom_input_factory_is_used():
    # a two-input op (add) needs a factory yielding two arrays
    def factory(shape, dtype, rng):
        return rng.standard_normal(shape), rng.standard_normal(shape)

    def add_ref(a, b, dtype="float32"):
        return np.asarray(a, np.float64) + np.asarray(b, np.float64)

    def add_wrong(a, b, dtype="float32"):
        return np.asarray(a, np.float64) - np.asarray(b, np.float64)  # BUG

    assert ko.gate(add_ref, add_ref, [(32,)], ("float32",),
                   input_factory=factory)["passed"] is True
    assert ko.gate(add_ref, add_wrong, [(32,)], ("float32",),
                   input_factory=factory)["passed"] is False


# ----------------------------------------------------------- reference ops

def test_rmsnorm_ref_unit_rms_input():
    # a vector with mean-square exactly 1 -> RMSNorm(eps->0) returns it ~unchanged
    x = np.array([[1.0, -1.0, 1.0, -1.0]])
    out = ko.rmsnorm_ref(x, eps=0.0)
    np.testing.assert_allclose(out, x, rtol=0, atol=1e-12)


def test_rmsnorm_ref_applies_weight():
    x = np.array([[3.0, 4.0]])
    w = np.array([2.0, 0.5])
    out = ko.rmsnorm_ref(x, weight=w, eps=0.0)
    base = ko.rmsnorm_ref(x, eps=0.0)
    np.testing.assert_allclose(out, base * w, rtol=0, atol=1e-12)


def test_wrong_rmsnorm_actually_differs_on_a_concrete_input():
    # mean-then-square vs square-then-mean: on a zero-mean input the wrong one
    # divides by ~sqrt(eps) -> blows up, proving the bug is real (not cosmetic).
    x = np.array([[2.0, -2.0, 2.0, -2.0]])   # mean 0, mean-square 4
    good = ko.rmsnorm_candidate_correct(x, eps=1e-6)
    bad = ko.rmsnorm_candidate_wrong(x, eps=1e-6)
    assert not np.allclose(good, bad)


# ----------------------------------------------------------- torch path (opt)

def test_compare_accepts_torch_tensors_if_available():
    torch = pytest.importorskip("torch")
    ref = torch.linspace(-2, 2, 50)
    cand = ref.clone()
    r = ko.compare(ref, cand, dtype="float32")
    assert r["passed"] is True
    # a torch tensor with a real drift still fails at fp32 tol
    cand2 = ref + 0.01
    assert ko.compare(ref, cand2, dtype="float32")["passed"] is False
