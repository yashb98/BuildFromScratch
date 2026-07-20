"""Correctness-first oracle for the Triton-kernel dev harness (§C21).

The Frontier-Eng kernel lift writes a candidate kernel (e.g. a fused RMSNorm)
and claims it is faster than the model's reference op. This module is the GATE
that stands BETWEEN "I wrote a kernel" and "I am allowed to benchmark it": a
candidate kernel is run against a REFERENCE op across a sweep of shapes x dtypes
and must agree within a per-dtype tolerance (an allclose-style check). A
fast-but-wrong kernel is DISCARDED here, never rooflined, never reported — the
§C21 correctness-first discipline. A kernel that is 3x faster but disagrees in
the 4th decimal at fp16 is not a speedup, it is a bug.

WHY THIS IS A CPU MODULE
------------------------
The subtle, get-it-wrong-able logic of a kernel gate is NOT the GPU launch — it
is the comparison policy: which tolerance per dtype, how a NaN/Inf is treated
(a single non-finite element fails the gate, never "allclose-with-NaN==True"),
how a shape-mismatch or dtype-mismatch is surfaced (a structural failure, not a
numeric one), and how a sweep aggregates into a single PASS/FAIL verdict. All of
that is pure Python/numpy and is proven correct WITHOUT a GPU here, mirroring the
eval_stats.py / posttrain_losses.py pattern: the SKILL is not "documented but
fake" — only the actual Triton launch on a profileable GPU is the off-box part.

To prove the gate itself works we ship two numpy reference ops (RMSNorm, SiLU)
AND, for each, a deliberately-subtly-wrong "candidate" (an epsilon dropped, a
mean used where the mean-square belongs). The oracle PASSES the correct candidate
and FAILS the wrong one — that is the test of the gate, on CPU.

TORCH/TRITON ARE OPTIONAL. The module imports on a CPU-only box: torch is only
touched inside helpers guarded by `torch is not None` / `torch.cuda.is_available()`,
and triton is never imported at module load. `compare()` works on anything
array-like (numpy arrays, torch tensors, or plain nested lists) by normalizing to
numpy first, so the same gate logic that runs in CI also runs against real device
tensors off-box (a torch tensor is moved to CPU+numpy before comparison).
"""
from __future__ import annotations

import math

import numpy as np

try:  # torch is optional — the gate logic must import & run without it (§ CPU-only)
    import torch  # noqa: F401
except Exception:  # pragma: no cover - exercised only on a torch-less box
    torch = None


# --------------------------------------------------------------------------
# Per-dtype tolerances. fp32 is tight; reduced precision is looser because the
# kernel's accumulation order legitimately differs from the reference's. These
# are the allclose (rtol, atol) pair, matched to torch.testing.assert_close's
# defaults for the corresponding dtype so the CPU gate and an off-box
# torch.testing gate agree on what "close enough" means.
# --------------------------------------------------------------------------
DTYPE_TOL = {
    "float32": {"rtol": 1.3e-6, "atol": 1e-5},
    "float64": {"rtol": 1e-7, "atol": 1e-8},
    "float16": {"rtol": 1e-3, "atol": 1e-5},
    "bfloat16": {"rtol": 1.6e-2, "atol": 1e-5},
}
DEFAULT_TOL = {"rtol": 1.3e-6, "atol": 1e-5}


def tolerance_for(dtype) -> dict:
    """The (rtol, atol) pair for a dtype, accepting a numpy dtype, a torch dtype,
    or a plain name string ('float16', 'bf16', 'fp32', ...). Unknown dtypes fall
    back to the conservative fp32 tolerance (never silently loose)."""
    name = _dtype_name(dtype)
    return dict(DTYPE_TOL.get(name, DEFAULT_TOL))


def _dtype_name(dtype) -> str:
    """Normalize any dtype-ish object to a canonical numpy-style name string."""
    if dtype is None:
        return "float32"
    raw = str(dtype)
    # 'torch.float16' -> 'float16'; 'torch.bfloat16' -> 'bfloat16'
    raw = raw.split(".")[-1].strip().lower()
    aliases = {
        "fp32": "float32", "f32": "float32", "float": "float32",
        "fp64": "float64", "f64": "float64", "double": "float64",
        "fp16": "float16", "f16": "float16", "half": "float16",
        "bf16": "bfloat16",
    }
    return aliases.get(raw, raw)


def _to_numpy(x):
    """Coerce an array-like (numpy array, torch tensor on any device, or a nested
    Python list/scalar) to a contiguous numpy float64 array WITHOUT requiring a
    GPU import path at module load. A torch tensor is detached, moved to CPU, and
    upcast — bf16 has no numpy dtype, so we always go through float for the
    comparison (the tolerance, not the storage dtype, encodes the precision)."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().to("cpu", dtype=torch.float64).numpy()
    arr = np.asarray(x)
    if arr.dtype == object:
        raise TypeError("ragged / non-numeric array-like passed to oracle")
    return arr.astype(np.float64, copy=False)


def _allclose_report(ref, cand, rtol, atol):
    """numpy-allclose with a diagnostic. Returns (passed, max_abs, max_rel,
    n_bad, reason). A NaN/Inf anywhere (in either array) is a HARD FAIL — we
    never let np.allclose's equal_nan path mask a kernel that produced NaNs.
    The element-wise rule mirrors numpy/torch: |c-r| <= atol + rtol*|r|."""
    if ref.shape != cand.shape:
        return False, math.inf, math.inf, ref.size, (
            f"shape mismatch: reference {ref.shape} vs candidate {cand.shape}")

    finite_ref = np.isfinite(ref)
    finite_cand = np.isfinite(cand)
    if not finite_ref.all() or not finite_cand.all():
        n_bad = int((~finite_cand).sum())
        where = "candidate" if not finite_cand.all() else "reference"
        return False, math.inf, math.inf, max(n_bad, 1), (
            f"non-finite (NaN/Inf) values in {where} output — fast-but-wrong, discarded")

    abs_err = np.abs(cand - ref)
    tol = atol + rtol * np.abs(ref)
    bad = abs_err > tol
    n_bad = int(bad.sum())
    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    # relative error guarded against divide-by-zero
    denom = np.abs(ref)
    rel = np.where(denom > 0, abs_err / denom, 0.0)
    max_rel = float(rel.max()) if rel.size else 0.0
    passed = n_bad == 0
    reason = "ok" if passed else (
        f"{n_bad}/{ref.size} elements exceed tol "
        f"(max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, rtol={rtol:.1e}, atol={atol:.1e})")
    return passed, max_abs, max_rel, n_bad, reason


def compare(reference_out, candidate_out, dtype="float32"):
    """Single allclose-style comparison of one candidate output vs the reference.

    reference_out / candidate_out: any array-like (numpy, torch tensor, list).
    dtype: drives the tolerance (the candidate's compute precision), NOT the
           comparison storage — both sides are upcast to float64 first.

    Returns a result dict: passed, dtype, rtol, atol, max_abs_err, max_rel_err,
    n_mismatch, n_total, reason. A shape/dtype-structural problem and a numeric
    drift are BOTH reported as passed=False but with distinguishable reasons.
    """
    tol = tolerance_for(dtype)
    ref = _to_numpy(reference_out)
    cand = _to_numpy(candidate_out)
    passed, max_abs, max_rel, n_bad, reason = _allclose_report(
        ref, cand, tol["rtol"], tol["atol"])
    return {
        "passed": passed,
        "dtype": _dtype_name(dtype),
        "rtol": tol["rtol"], "atol": tol["atol"],
        "max_abs_err": max_abs, "max_rel_err": max_rel,
        "n_mismatch": n_bad, "n_total": int(ref.size),
        "reason": reason,
    }


def gate(reference_fn, candidate_fn, shapes, dtypes=("float32",),
         input_factory=None, seed=0):
    """Run the full correctness gate: the candidate is compared against the
    reference across the CROSS PRODUCT of `shapes` x `dtypes`. The verdict is
    PASS iff EVERY cell passes — a kernel correct at fp32 but wrong at one shape
    in bf16 is DISCARDED (§C21: a kernel must be correct everywhere it claims to
    run, not on average).

    reference_fn / candidate_fn: callables taking the SAME generated input
        (a tuple of numpy arrays) and an optional `dtype` keyword; each returns
        an array-like output. The reference is the source of truth (the model's
        own op, ported to numpy on CPU; the model.py op on GPU off-box).
    shapes:  iterable of shape tuples, e.g. [(8, 256), (1, 4096)].
    dtypes:  iterable of dtype names ('float32', 'float16', 'bfloat16', ...).
    input_factory: optional callable(shape, dtype, rng) -> tuple-of-inputs. The
        default makes a single standard-normal input array of `shape` (the
        common single-tensor elementwise/norm op). Both fns get the SAME inputs.
    seed: RNG seed so the sweep is deterministic and reproducible.

    Returns a verdict dict: passed (bool, the gate), n_cells, n_failed, cells
    (per-cell result dicts incl. shape/dtype), and a one-line `summary`. The
    contract: callers MUST NOT benchmark/roofline a candidate unless
    verdict['passed'] is True.
    """
    shapes = [tuple(s) for s in shapes]
    dtypes = [_dtype_name(d) for d in dtypes]
    if not shapes:
        raise ValueError("need at least one shape to sweep")
    if not dtypes:
        raise ValueError("need at least one dtype to sweep")
    if input_factory is None:
        input_factory = _default_input_factory

    cells = []
    n_failed = 0
    for di, dt in enumerate(dtypes):
        for si, shape in enumerate(shapes):
            # deterministic, distinct RNG per cell
            rng = np.random.default_rng(seed + 1009 * di + si)
            inputs = input_factory(shape, dt, rng)
            try:
                ref_out = _call(reference_fn, inputs, dt)
                cand_out = _call(candidate_fn, inputs, dt)
                cell = compare(ref_out, cand_out, dtype=dt)
            except Exception as exc:  # a candidate that raises is a FAIL, not a crash
                cell = {
                    "passed": False, "dtype": dt, "rtol": float("nan"),
                    "atol": float("nan"), "max_abs_err": math.inf,
                    "max_rel_err": math.inf, "n_mismatch": -1, "n_total": -1,
                    "reason": f"candidate raised {type(exc).__name__}: {exc}",
                }
            cell["shape"] = shape
            cells.append(cell)
            if not cell["passed"]:
                n_failed += 1

    passed = n_failed == 0
    summary = (
        f"GATE {'PASS' if passed else 'FAIL'}: {len(cells) - n_failed}/{len(cells)} "
        f"cells ok across {len(shapes)} shapes x {len(dtypes)} dtypes"
        + ("" if passed else " — candidate DISCARDED (correctness-first §C21)"))
    return {
        "passed": passed,
        "n_cells": len(cells),
        "n_failed": n_failed,
        "cells": cells,
        "summary": summary,
    }


def _call(fn, inputs, dtype):
    """Call a reference/candidate fn with the generated inputs, passing `dtype`
    only if the fn accepts it (kwarg-tolerant so simple numpy refs need no dtype
    param)."""
    try:
        return fn(*inputs, dtype=dtype)
    except TypeError:
        return fn(*inputs)


def _default_input_factory(shape, dtype, rng):
    """Default single-input generator: one standard-normal array of `shape`.
    Returns a 1-tuple (so reference/candidate fns are called as fn(x))."""
    x = rng.standard_normal(size=shape).astype(np.float32)
    return (x,)


# --------------------------------------------------------------------------
# Reference ops (numpy) + matched candidates. These exist to PROVE the gate on
# CPU: the *_ref is the source of truth; the *_correct candidate must PASS and
# the *_wrong candidate must FAIL. Off-box, the reference_fn is the model's own
# op (model.py) and the candidate_fn wraps the Triton kernel.
# --------------------------------------------------------------------------

def rmsnorm_ref(x, weight=None, eps=1e-6, dtype="float32"):
    """Reference RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * weight.
    This is the numpy oracle the candidate Triton RMSNorm kernel is gated against."""
    x = np.asarray(x, dtype=np.float64)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    out = x / np.sqrt(ms + eps)
    if weight is not None:
        out = out * np.asarray(weight, dtype=np.float64)
    return out


def rmsnorm_candidate_correct(x, weight=None, eps=1e-6, dtype="float32"):
    """A *correct* alternate implementation (different but equivalent algebra:
    reciprocal-sqrt factored out). Stands in for a correct Triton kernel — the
    gate must PASS this."""
    x = np.asarray(x, dtype=np.float64)
    inv = 1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    out = x * inv
    return out if weight is None else out * np.asarray(weight, dtype=np.float64)


def rmsnorm_candidate_wrong(x, weight=None, eps=1e-6, dtype="float32"):
    """A SUBTLY WRONG RMSNorm: it normalizes by sqrt(mean(x)^2 + eps) instead of
    sqrt(mean(x^2) + eps) — i.e. mean-then-square rather than square-then-mean
    (a classic fused-kernel bug). Numerically close on some inputs, catastrophic
    on others. The gate MUST FAIL this and DISCARD it (the whole point)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.mean(x, axis=-1, keepdims=True)          # BUG: mean before square
    out = x / np.sqrt(m * m + eps)
    return out if weight is None else out * np.asarray(weight, dtype=np.float64)


def silu_ref(x, dtype="float32"):
    """Reference SiLU / swish: x * sigmoid(x)."""
    x = np.asarray(x, dtype=np.float64)
    return x * (1.0 / (1.0 + np.exp(-x)))


def silu_candidate_wrong(x, dtype="float32"):
    """WRONG SiLU: plain sigmoid(x) (forgot the elementwise * x). Must FAIL."""
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))
