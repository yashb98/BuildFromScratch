"""Tests for the L1 memory guards every GPU process depends on — safe_cuda (torch)
and jax_safe_env (JAX). Upgrade-plan item 16 (batch7 Q2: test the killers first):
these had NO tests, yet their pure-CPU logic — fraction-range validation, env-var
composition, and the PREALLOCATE=true refusal — is the first line of defense against
the unified-memory over-allocation that hard-crashed the box on 2026-06-08.

All tests here are pure CPU: the validation/env logic runs BEFORE any torch/jax call,
so no GPU (and no torch/jax install) is needed to exercise the safety-critical paths.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ safe_cuda
import safe_cuda  # noqa: E402  (imports os only; torch is imported inside guard())


@pytest.mark.parametrize("bad", [0.0, -0.1, 0.96, 1.0, 1.5, 2.0])
def test_guard_rejects_unsafe_fraction_before_touching_torch(bad):
    """guard() validates 0 < fraction <= 0.95 and raises BEFORE the torch call, so an
    unsafe fraction is refused even on a box with no CUDA. This is the check that stops
    a 0.99 (or a typo'd 1.5) from handing the whole unified pool to one process."""
    with pytest.raises(ValueError):
        safe_cuda.guard(bad)


def test_safe_cuda_import_sets_alloc_conf():
    """Importing safe_cuda must compose PYTORCH_CUDA_ALLOC_CONF (expandable segments etc.)
    so fragmentation doesn't trip the OOM-killer on the shared pool."""
    assert "PYTORCH_CUDA_ALLOC_CONF" in os.environ
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] != ""


def test_guard_default_fraction_is_the_documented_085():
    import inspect
    sig = inspect.signature(safe_cuda.guard)
    assert sig.parameters["fraction"].default == 0.85   # §C1 default; must not drift


# ------------------------------------------------------------------ jax_safe_env
def _reload_jax_safe_env(env):
    """Import jax_safe_env fresh under a controlled environment, returning (module|exc)."""
    for k in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop("jax_safe_env", None)
    try:
        return importlib.import_module("jax_safe_env"), None
    except Exception as e:  # noqa: BLE001
        return None, e


def test_jax_safe_env_sets_no_prealloc_and_half_pool():
    saved = {k: os.environ.get(k) for k in
             ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")}
    try:
        mod, exc = _reload_jax_safe_env({})
        assert exc is None
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"  # no ~90GB startup grab
        assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"   # cap at ~60GB
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.modules.pop("jax_safe_env", None)


def test_jax_safe_env_refuses_prealloc_true():
    """If a caller already set PREALLOCATE=true (the ~90GB unified-pool grab that would
    crash the box), importing the guard must REFUSE loudly rather than proceed."""
    saved = {k: os.environ.get(k) for k in
             ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")}
    try:
        mod, exc = _reload_jax_safe_env({"XLA_PYTHON_CLIENT_PREALLOCATE": "true"})
        assert mod is None and isinstance(exc, RuntimeError)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.modules.pop("jax_safe_env", None)
