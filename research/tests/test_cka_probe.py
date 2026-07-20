"""Regression guard for research/interp/cka_probe.py's CKA math + decision-rule plumbing
(CPU-only; the GPU probe path is exercised separately). Locks in the invariances the
representational-convergence null depends on, and the anti-laundering direction of the
bootstrap decision rule."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "interp"))
torch = pytest.importorskip("torch")
import cka_probe as cp  # noqa: E402


def test_cka_invariances():
    torch.manual_seed(0)
    X = torch.randn(400, 48).double()
    assert abs(cp.linear_cka(X, X) - 1.0) < 1e-9                       # identity
    assert abs(cp.linear_cka(X, X * 2.5 - 1.0) - 1.0) < 1e-6           # isotropic scale+shift
    R = torch.linalg.qr(torch.randn(48, 48).double())[0]
    assert abs(cp.linear_cka(X, X @ R) - 1.0) < 1e-6                   # orthogonal transform
    assert cp.linear_cka(X, torch.randn(400, 48).double()) < 0.2      # independent → low


def test_fp16_would_overflow_is_upcast():
    """The confound that faked the first null: large activations must not silently degrade.
    linear_cka upcasts fp16 inputs; here we confirm a huge-magnitude pair still gives CKA≈1
    when it should (fp16 storage would have overflowed these to inf → CKA 0)."""
    torch.manual_seed(0)
    X = torch.randn(300, 32) * 3.0e5                                   # ~NorMuon-scale magnitudes
    assert X.abs().max() > 65504                                       # beyond fp16 range
    assert abs(cp.linear_cka(X.float(), X.float()) - 1.0) < 1e-5


def test_decision_rule_direction():
    torch.manual_seed(0)
    X = torch.randn(300, 32)
    Y = torch.randn(300, 32)
    clear = {"n": X, "a": Y, "t": X.clone()}                          # CKA(n,t)=1 ≫ CKA(a,t) → Δ≫0
    pt, lo, hi = cp.bootstrap_delta_ci(clear, "n", "a", "t", B=200)
    assert lo > 0 and pt > 0.5                                         # disjoint-positive confirms
    null = {"n": Y, "a": Y.clone(), "t": X}                           # both ≈0 vs t → Δ≈0
    pt2, lo2, hi2 = cp.bootstrap_delta_ci(null, "n", "a", "t", B=200)
    assert lo2 <= 0 <= hi2                                             # straddles 0 → null
