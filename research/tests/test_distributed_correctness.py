"""Tests for research/distributed_correctness.py — the §C21 correctness GATE.

Pure math, no torch/CUDA/model. Proves the gate (a) PASSES a distributed run
that matches the single-GPU baseline within the seed-noise floor, (b) FAILS a
fast-but-wrong run whose loss is shifted beyond the floor, (c) hard-FAILS a
diverged (NaN/inf) distributed run, (d) derives its tolerance from real baseline
seed scatter (never an invented number), and (e) refuses to gate with no floor
or two floors. This is the "fast-but-wrong is caught BEFORE any MFU is trusted"
law, exercised entirely on CPU with synthetic numbers.
"""
import math

import pytest

import distributed_correctness as dc


# ---- noise_floor_from_seeds --------------------------------------------------

def test_noise_floor_is_k_sigma_of_seed_scatter():
    seeds = [2.50, 2.52, 2.48]  # single-GPU baseline loss across 3 seeds
    tol, mean, std = dc.noise_floor_from_seeds(seeds, k_sigma=2.0)
    assert math.isclose(mean, 2.50, abs_tol=1e-9)
    assert math.isclose(tol, 2.0 * std)
    assert tol > 0


def test_noise_floor_requires_two_seeds():
    with pytest.raises(ValueError):
        dc.noise_floor_from_seeds([2.5])


def test_noise_floor_rejects_nonfinite_seed():
    with pytest.raises(ValueError):
        dc.noise_floor_from_seeds([2.5, float("nan")])


def test_noise_floor_rejects_nonpositive_ksigma():
    with pytest.raises(ValueError):
        dc.noise_floor_from_seeds([2.5, 2.6], k_sigma=0)


# ---- the gate: pass / fail ---------------------------------------------------

def test_matching_distributed_run_passes_with_explicit_tolerance():
    r = dc.check_distributed_correctness(
        distributed_metric=2.505,
        single_gpu_baseline=2.50,
        tolerance=0.05,
    )
    assert r["passed"] is True
    assert r["verdict"] == "pass"
    assert r["margin"] >= 0
    assert "trust" in r["reason"].lower() or "pass" in r["reason"].lower()


def test_fast_but_wrong_run_fails():
    # Distributed loss shifted by 0.3 — way outside a 0.05 floor. The canonical
    # sharding/all-reduce bug: descends, but to the wrong place.
    r = dc.check_distributed_correctness(
        distributed_metric=2.80,
        single_gpu_baseline=2.50,
        tolerance=0.05,
    )
    assert r["passed"] is False
    assert r["verdict"] == "fail"
    assert r["margin"] < 0
    assert "wrong" in r["reason"].lower() or "diverge" in r["reason"].lower()


def test_diverged_nan_distributed_run_hard_fails():
    r = dc.check_distributed_correctness(
        distributed_metric=float("nan"),
        single_gpu_baseline=2.50,
        tolerance=10.0,  # huge tolerance must NOT rescue a NaN
    )
    assert r["passed"] is False
    assert math.isinf(r["abs_delta"])


def test_inf_distributed_run_hard_fails():
    r = dc.check_distributed_correctness(
        distributed_metric=float("inf"),
        single_gpu_baseline=2.50,
        tolerance=10.0,
    )
    assert r["passed"] is False


# ---- tolerance derived from seeds (the §C17/§C21 link) -----------------------

def test_gate_derives_tolerance_from_baseline_seeds():
    seeds = [2.50, 2.52, 2.48]
    # Distributed run lands at the seed mean -> trivially inside the floor.
    r = dc.check_distributed_correctness(
        distributed_metric=2.50,
        single_gpu_baseline=None,  # use the seed mean as reference
        baseline_seeds=seeds,
        k_sigma=2.0,
    )
    assert r["passed"] is True
    assert r["n_baseline_seeds"] == 3
    assert math.isclose(r["single_gpu_baseline"], 2.50, abs_tol=1e-9)


def test_gate_fails_when_outside_seed_floor():
    seeds = [2.50, 2.501, 2.499]  # very tight scatter -> tiny floor
    r = dc.check_distributed_correctness(
        distributed_metric=2.70,  # far outside the tight floor
        single_gpu_baseline=None,
        baseline_seeds=seeds,
        k_sigma=2.0,
    )
    assert r["passed"] is False


# ---- relative floor: reduction-order reassociation must not fail -------------

def test_bit_reassociation_within_relative_floor_passes():
    # Zero-scatter baseline (lucky identical seeds) but a 1e-7 relative diff from
    # a different all-reduce summation order must NOT be rejected.
    base = 2.5
    r = dc.check_distributed_correctness(
        distributed_metric=base * (1 + 1e-7),
        single_gpu_baseline=base,
        tolerance=0.0,  # the explicit floor is 0; relative floor must rescue
    )
    assert r["passed"] is True


# ---- argument validation -----------------------------------------------------

def test_rejects_no_tolerance_source():
    with pytest.raises(ValueError):
        dc.check_distributed_correctness(2.5, 2.5)


def test_rejects_two_tolerance_sources():
    with pytest.raises(ValueError):
        dc.check_distributed_correctness(
            2.5, 2.5, tolerance=0.1, baseline_seeds=[2.5, 2.6]
        )


def test_explicit_tolerance_requires_baseline():
    with pytest.raises(ValueError):
        dc.check_distributed_correctness(2.5, None, tolerance=0.1)


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError):
        dc.check_distributed_correctness(2.5, 2.5, tolerance=-0.1)


# ---- gate_or_raise -----------------------------------------------------------

def test_gate_or_raise_passes_through_on_pass():
    r = dc.gate_or_raise(2.505, 2.50, tolerance=0.05)
    assert r["passed"] is True


def test_gate_or_raise_raises_on_fail():
    with pytest.raises(dc.CorrectnessGateError) as exc:
        dc.gate_or_raise(2.80, 2.50, tolerance=0.05)
    assert exc.value.result["passed"] is False
    assert "fail" in str(exc.value).lower() or "wrong" in str(exc.value).lower()
