"""CPU unit tests for the chunked-torch CCE fused linear cross-entropy reference.

These are the HARD correctness gate of SPEC_cce_fused_linear_ce.md, run on CPU at
tiny sizes so the whole CCE mechanism is proven WITHOUT a GPU:

  * forward loss matches the fp32 unfused oracle  F.cross_entropy(H@W.T, y)  (atol 1e-3)
  * backward dH and dW match the oracle's autograd grads (rtol/atol 1e-2)
  * the (N, V) logits tensor is NEVER materialized — enforced by an INDEPENDENT
    global torch.matmul shape guard, not just the module's self-report.
  * the online log-sum-exp equals a direct full-vocab logsumexp
  * ignore_index rows, reduction='sum'/'none', and the eps gradient filter behave
  * the eps filter is exercised in a regime where it ACTUALLY DROPS MASS (peaked,
    non-zero-mean W) and a too-aggressive eps is DETECTED (so these tests are not a
    rubber stamp), and a bf16-ROUNDED emulation of the Triton kernel path clears the
    same 1e-2 grad gate.

SCOPE (honest): this file gates the pure-torch reference (cce_linear_ce), which shares
the Triton kernel's math. It does NOT run the Triton kernels (needs CUDA) — their
bf16 tensor-core numerics and tile scheduling are gated off-box by
cce_triton.gate_against_reference(). CPU-green != kernel-correct.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

# cce_linear_ce lives under research/kernel/, which conftest does not add to sys.path.
_KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernel"
if str(_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNEL_DIR))

import cce_linear_ce  # noqa: E402
from cce_linear_ce import linear_cross_entropy, EPS_BF16  # noqa: E402


# --------------------------------------------------------------------------- helpers
V_TINY, N_TINY, D_TINY = 512, 64, 32
CHUNK = 128   # 4 vocab blocks over V=512 — strictly smaller than V


def _make_inputs(N=N_TINY, D=D_TINY, V=V_TINY, dtype=torch.float32, seed=0, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    H = (torch.randn(N, D, generator=g, dtype=torch.float32) * scale).to(dtype)
    W = (torch.randn(V, D, generator=g, dtype=torch.float32) * scale).to(dtype)
    y = torch.randint(0, V, (N,), generator=g)
    return H, W, y


def _oracle(H, W, y, ignore_index=-100, reduction="mean"):
    """The HARD oracle: fp32 unfused CE. Returns (loss, dH, dW)."""
    Href = H.detach().float().clone().requires_grad_(True)
    Wref = W.detach().float().clone().requires_grad_(True)
    logits = Href @ Wref.t()                        # (N, V) fp32 — the tensor CCE avoids
    loss = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction=reduction)
    if reduction == "none":
        loss.sum().backward()
    else:
        loss.backward()
    return loss.detach(), Href.grad.detach(), Wref.grad.detach()


def _fused(H, W, y, chunk_size=CHUNK, ignore_index=-100, reduction="mean",
           grad_filter_eps=0.0):
    """Run the chunked reference in fp32 and collect (loss, dH, dW)."""
    Hf = H.detach().float().clone().requires_grad_(True)
    Wf = W.detach().float().clone().requires_grad_(True)
    loss = linear_cross_entropy(Hf, Wf, y, chunk_size=chunk_size,
                                ignore_index=ignore_index, reduction=reduction,
                                grad_filter_eps=grad_filter_eps)
    if reduction == "none":
        loss.sum().backward()
    else:
        loss.backward()
    return loss.detach(), Hf.grad.detach(), Wf.grad.detach()


def _fraction_softmax_below_eps(H, W, eps):
    """Test-only (dense, tiny V): fraction of full-vocab softmax entries below eps.
    Used to PROVE a filter test is actually in a regime where the filter drops mass."""
    logits = H.float() @ W.float().t()
    p = torch.softmax(logits, dim=1)
    return float((p < eps).float().mean())


class _MatmulShapeGuard:
    """Globally wrap torch.matmul (thread-safe: a plain attribute swap, so it intercepts
    the autograd engine's backward thread too) and record every output shape. Asserts on
    exit that no GEMM ever produced the forbidden (N, V) logits tensor.

    CONTRACT (pinned): the reference deliberately routes EVERY big GEMM through
    torch.matmul (never the @ operator) so this independent guard can observe them. A
    refactor to @ would blind the guard — but it fails LOUDLY ("saw no matmul at all")
    rather than silently passing, so the contract self-enforces. This does NOT trust
    cce_linear_ce.LAST_STATS."""

    def __init__(self, N, V):
        self.N, self.V = N, V
        self.shapes = []
        self._orig = None

    def __enter__(self):
        self._orig = torch.matmul

        def _wrapped(a, b, *args, **kwargs):
            out = self._orig(a, b, *args, **kwargs)
            try:
                self.shapes.append(tuple(out.shape))
            except Exception:
                pass
            return out

        torch.matmul = _wrapped
        return self

    def __exit__(self, *exc):
        torch.matmul = self._orig
        return False

    def assert_never_full_logits(self):
        assert self.shapes, ("guard saw no matmul at all — the reference MUST use "
                             "torch.matmul (not the @ operator) for the guard to observe GEMMs")
        forbidden = (self.N, self.V)
        for shp in self.shapes:
            assert shp != forbidden, f"(N, V) logits tensor {forbidden} was materialized!"
            # Backstop pinned to the VOCAB axis (dim-1 spanning the full vocab), not any
            # wide tensor: a 2-D GEMM output (N rows) x (>= V cols) is the full-logits
            # tile. Safe against the (N, D) backward-dH product because D (1024) < V.
            assert not (len(shp) == 2 and shp[0] == self.N and shp[1] >= self.V), \
                f"a full-vocab-width logits tile {shp} was formed"


# --------------------------------------------------------------------------- the gate

def test_forward_loss_matches_fp32_oracle():
    H, W, y = _make_inputs()
    ref_loss, _, _ = _oracle(H, W, y)
    fz_loss, _, _ = _fused(H, W, y, grad_filter_eps=0.0)
    torch.testing.assert_close(fz_loss, ref_loss, atol=1e-3, rtol=0)


def test_backward_grads_match_oracle_filter_off():
    """Filter OFF -> the reference is the exact CE; grads must match tightly."""
    H, W, y = _make_inputs()
    _, ref_dH, ref_dW = _oracle(H, W, y)
    _, fz_dH, fz_dW = _fused(H, W, y, grad_filter_eps=0.0)
    torch.testing.assert_close(fz_dH, ref_dH, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fz_dW, ref_dW, rtol=1e-2, atol=1e-2)


def test_backward_grads_match_oracle_with_eps_filter():
    """Filter ON at eps=2**-12 -> still within the loosened bf16 tolerance."""
    H, W, y = _make_inputs()
    _, ref_dH, ref_dW = _oracle(H, W, y)
    _, fz_dH, fz_dW = _fused(H, W, y, grad_filter_eps=EPS_BF16)
    torch.testing.assert_close(fz_dH, ref_dH, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fz_dW, ref_dW, rtol=1e-2, atol=1e-2)


def test_full_logits_tensor_is_never_materialized():
    """The load-bearing memory claim: run the full fwd+bwd under a global matmul
    shape guard and assert (N, V) is never formed."""
    H, W, y = _make_inputs()
    Hf = H.float().clone().requires_grad_(True)
    Wf = W.float().clone().requires_grad_(True)
    with _MatmulShapeGuard(N_TINY, V_TINY) as guard:
        loss = linear_cross_entropy(Hf, Wf, y, chunk_size=CHUNK, grad_filter_eps=EPS_BF16)
        loss.backward()
    guard.assert_never_full_logits()
    # widest vocab tile the impl self-reports must be a chunk, never the full vocab
    assert cce_linear_ce.LAST_STATS["max_logits_tile_cols"] == CHUNK
    assert cce_linear_ce.LAST_STATS["max_logits_tile_cols"] < V_TINY


def test_online_lse_equals_direct_logsumexp():
    """The streaming (running max, running sum) LSE must equal a direct full-vocab
    logsumexp computed on the (small) dense logits."""
    H, W, y = _make_inputs()
    H32, W32 = H.float(), W.float()
    lse, zy, max_cols = cce_linear_ce._online_lse_and_target(
        H32, W32, y, chunk_size=CHUNK, ignore_index=-100)
    dense = H32 @ W32.t()                                  # (N, V) — allowed here (tiny, test-only)
    direct_lse = torch.logsumexp(dense, dim=1)
    torch.testing.assert_close(lse, direct_lse, rtol=1e-5, atol=1e-5)
    # gathered target logit equals the directly-indexed logit
    direct_zy = dense[torch.arange(N_TINY), y]
    torch.testing.assert_close(zy, direct_zy, rtol=1e-5, atol=1e-5)
    assert max_cols == CHUNK


def test_loss_is_independent_of_chunk_size():
    """Correctness must not depend on the vocab block width (streaming LSE is exact)."""
    H, W, y = _make_inputs()
    losses = []
    for cs in (32, 128, 300, V_TINY):     # incl. 300 (non-divisor of 512) and the full vocab
        l, _, _ = _fused(H, W, y, chunk_size=cs, grad_filter_eps=0.0)
        losses.append(float(l))
    for l in losses[1:]:
        assert abs(l - losses[0]) < 1e-5, f"loss depends on chunk_size: {losses}"


def test_remainder_vocab_block_is_handled():
    """V not a multiple of chunk_size (V=512, chunk=300 -> blocks 300 + 212) must
    still match the oracle — the remainder-mask path."""
    H, W, y = _make_inputs()
    ref_loss, ref_dH, ref_dW = _oracle(H, W, y)
    fz_loss, fz_dH, fz_dW = _fused(H, W, y, chunk_size=300, grad_filter_eps=0.0)
    torch.testing.assert_close(fz_loss, ref_loss, atol=1e-3, rtol=0)
    torch.testing.assert_close(fz_dH, ref_dH, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fz_dW, ref_dW, rtol=1e-2, atol=1e-2)


def test_ignore_index_rows_dropped_like_oracle():
    """Rows labelled ignore_index contribute 0 to loss and 0 to gradient — matching
    F.cross_entropy(ignore_index=...)."""
    H, W, y = _make_inputs()
    y = y.clone()
    y[::7] = -100                                         # ignore every 7th row
    ref_loss, ref_dH, ref_dW = _oracle(H, W, y, ignore_index=-100)
    fz_loss, fz_dH, fz_dW = _fused(H, W, y, ignore_index=-100, grad_filter_eps=0.0)
    torch.testing.assert_close(fz_loss, ref_loss, atol=1e-3, rtol=0)
    torch.testing.assert_close(fz_dH, ref_dH, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fz_dW, ref_dW, rtol=1e-2, atol=1e-2)
    # the ignored rows have exactly-zero hidden gradient
    assert torch.count_nonzero(fz_dH[::7]) == 0


def test_reduction_sum_matches_oracle():
    H, W, y = _make_inputs()
    ref_loss, ref_dH, ref_dW = _oracle(H, W, y, reduction="sum")
    fz_loss, fz_dH, fz_dW = _fused(H, W, y, reduction="sum", grad_filter_eps=0.0)
    torch.testing.assert_close(fz_loss, ref_loss, atol=1e-2, rtol=0)  # sum is ~N*mean -> looser abs
    torch.testing.assert_close(fz_dH, ref_dH, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fz_dW, ref_dW, rtol=1e-2, atol=1e-2)


def test_reduction_none_matches_oracle_per_row():
    H, W, y = _make_inputs()
    ref_loss, _, _ = _oracle(H, W, y, reduction="none")
    fz_loss, _, _ = _fused(H, W, y, reduction="none", grad_filter_eps=0.0)
    assert fz_loss.shape == (N_TINY,)
    torch.testing.assert_close(fz_loss, ref_loss, atol=1e-3, rtol=0)


def test_bf16_inputs_upcast_and_match_oracle_loosely():
    """bf16 inputs (the production dtype) are upcast to fp32 internally; loss still
    tracks the fp32 oracle within bf16 noise."""
    H, W, y = _make_inputs(dtype=torch.bfloat16)
    ref_loss, _, _ = _oracle(H, W, y)                    # oracle upcasts bf16->fp32
    fz_loss, _, _ = _fused(H, W, y, grad_filter_eps=EPS_BF16)
    torch.testing.assert_close(fz_loss.float(), ref_loss, atol=5e-3, rtol=0)


# --------------------------------------------------- eps filter: exercised WHERE IT BITES

def test_default_eps_filter_active_and_within_tol_on_peaked_nonzero_mean_W():
    """The CCE headline trick, tested where it actually FILTERS. A peaked softmax at a
    larger vocab (so many entries fall below eps) AND a NON-zero-mean W (so the dropped
    mass does NOT conveniently cancel via E_p[W]~0, the trap that made the flat/zero-mean
    case pass for free). Under production mean-reduction the default eps must still hold
    the SPEC's 1e-2 backward gate. (The residual RELATIVE error at scale is the open
    pretraining-scale A/B in Known-open — this bounds the ABSOLUTE error the gate uses.)"""
    g = torch.Generator().manual_seed(11)
    N, D, V = 128, 64, 4096
    H = torch.randn(N, D, generator=g) * 0.7
    W = torch.randn(V, D, generator=g) * 0.7 + 0.5       # non-zero mean shift
    y = torch.randint(0, V, (N,), generator=g)

    # PROVE the filter is genuinely active here (drops a large fraction of entries),
    # so this is not a no-op test that would pass with the filter disabled.
    frac = _fraction_softmax_below_eps(H, W, EPS_BF16)
    assert frac > 0.5, f"filter not exercised in this regime ({frac:.1%} below eps)"

    _, ref_dH, ref_dW = _oracle(H, W, y)                 # exact fp32 CE
    _, f_dH, f_dW = _fused(H, W, y, chunk_size=512, grad_filter_eps=EPS_BF16)
    torch.testing.assert_close(f_dH, ref_dH, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(f_dW, ref_dW, rtol=1e-2, atol=1e-2)


def test_too_aggressive_eps_is_detectable():
    """Guards against a rubber-stamp: an absurd eps=0.5 drops nearly all softmax mass,
    and with a strongly NON-zero-mean W under SUM reduction (grads O(1)) the resulting
    gradient MUST visibly break the 1e-2 gate. If this did not raise, the suite could
    not tell a correct filter from a broken one."""
    g = torch.Generator().manual_seed(7)
    N, D, V = 64, 32, 512
    H = torch.randn(N, D, generator=g) * 0.3
    W = torch.randn(V, D, generator=g) * 0.3 + 0.8       # strong non-zero mean
    y = torch.randint(0, V, (N,), generator=g)
    _, ref_dH, _ = _oracle(H, W, y, reduction="sum")
    _, bad_dH, _ = _fused(H, W, y, reduction="sum", grad_filter_eps=0.5)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(bad_dH, ref_dH, rtol=1e-2, atol=1e-2)


def test_bf16_rounded_gemm_emulation_within_tol():
    """CPU proxy for the Triton kernel's NUMERIC PATH (which the pure-fp32 tests do not
    exercise): store H/W as bf16, recompute logits with bf16-rounded operands (fp32
    accumulate, as the kernel's tensor-core dot does), keep dlogit in fp32, and store
    dH/dW as bf16. It must still clear the 1e-2 grad gate at a production-ish N. The
    definitive check remains the off-box cce_triton.gate_against_reference()."""
    g = torch.Generator().manual_seed(5)
    N, D, V = 256, 64, 2048
    Hf = (torch.randn(N, D, generator=g) * 0.5)
    Wf = (torch.randn(V, D, generator=g) * 0.5)
    y = torch.randint(0, V, (N,), generator=g)

    # oracle: exact fp32 CE
    ref_loss, ref_dH, ref_dW = _oracle(Hf, Wf, y)

    def _bf(x):                                          # bf16 round-trip (operand rounding)
        return x.bfloat16().float()

    # emulate the kernel: bf16-rounded logit operands, fp32 accumulate + softmax,
    # fp32 dlogit, bf16-rounded W/H in the accumulate GEMMs, bf16-stored outputs.
    Hb, Wb = _bf(Hf), _bf(Wf)
    logits = torch.matmul(Hb, Wb.t())                   # fp32 out, bf16-rounded operands
    lse = torch.logsumexp(logits, dim=1, keepdim=True)
    p = torch.exp(logits - lse)
    p = torch.where(p >= EPS_BF16, p, torch.zeros_like(p))   # eps filter
    onehot = F.one_hot(y, V).float()
    dlogit = (p - onehot) / N                            # mean reduction, fp32
    dH = torch.matmul(dlogit, Wb).bfloat16().float()    # bf16-rounded W operand, bf16 store
    dW = torch.matmul(dlogit.t(), Hb).bfloat16().float()

    torch.testing.assert_close(dH, ref_dH, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dW, ref_dW, rtol=1e-2, atol=1e-2)


# --------------------------------------------------- edge cases matching F.cross_entropy

def test_out_of_range_label_raises_like_cross_entropy():
    """y >= V or a stray negative (non-ignore) label must RAISE, exactly as
    F.cross_entropy raises IndexError — never silently absorbed into a wrong loss."""
    H, W, y = _make_inputs()
    Hf = H.float().clone().requires_grad_(True)
    Wf = W.float().clone().requires_grad_(True)
    y_hi = y.clone(); y_hi[0] = V_TINY                   # == V -> out of range
    with pytest.raises(ValueError):
        linear_cross_entropy(Hf, Wf, y_hi, chunk_size=CHUNK)
    y_neg = y.clone(); y_neg[0] = -5                     # negative, not ignore_index
    with pytest.raises(ValueError):
        linear_cross_entropy(Hf, Wf, y_neg, chunk_size=CHUNK)


def test_all_ignored_batch_returns_nan_like_oracle():
    """A fully-masked micro-batch is 0/0 -> NaN in F.cross_entropy(mean); the fused ref
    must match (NOT a silent 0.0 that hides a labeling bug)."""
    H, W, y = _make_inputs()
    y = torch.full_like(y, -100)
    ref = F.cross_entropy(H.float() @ W.float().t(), y, ignore_index=-100)  # NaN
    Hf = H.float().clone().requires_grad_(True)
    Wf = W.float().clone().requires_grad_(True)
    fz = linear_cross_entropy(Hf, Wf, y, chunk_size=CHUNK, ignore_index=-100, grad_filter_eps=0.0)
    assert torch.isnan(ref) and torch.isnan(fz), (float(ref), float(fz))


def test_cpu_fallback_entry_matches_reference():
    """cce_linear_ce.triton_linear_cross_entropy transparently falls back to the chunked
    reference on CPU (no CUDA), so it imports + runs here and equals linear_cross_entropy."""
    H, W, y = _make_inputs()
    Hf1 = H.float().clone().requires_grad_(True)
    Wf1 = W.float().clone().requires_grad_(True)
    loss1 = cce_linear_ce.triton_linear_cross_entropy(Hf1, Wf1, y, grad_filter_eps=0.0)
    loss1.backward()
    Hf2 = H.float().clone().requires_grad_(True)
    Wf2 = W.float().clone().requires_grad_(True)
    loss2 = linear_cross_entropy(Hf2, Wf2, y, grad_filter_eps=0.0)
    loss2.backward()
    torch.testing.assert_close(loss1, loss2)
    torch.testing.assert_close(Hf1.grad, Hf2.grad)
    torch.testing.assert_close(Wf1.grad, Wf2.grad)


def test_eps_constant_and_bf16_sub_ulp_rationale():
    """CORRECTED bf16 facts (the old comment was numerically wrong). bf16 has 7 EXPLICIT
    mantissa bits, so ULP(1.0)=2**-7 and the round-to-1.0 boundary is ~2**-8. 2**-12 is
    well BELOW that boundary -> it is a CONSERVATIVE SUB-ULP filter floor (it IS truncated
    when added to 1.0), NOT 'the smallest non-truncated bf16 magnitude'."""
    assert EPS_BF16 == 2.0 ** -12
    # power of two -> exactly representable in bf16 (round-trips)
    assert float(torch.tensor(EPS_BF16, dtype=torch.bfloat16).float()) == EPS_BF16
    one = torch.tensor(1.0, dtype=torch.bfloat16)
    # sub-ULP: 2**-12 and 2**-8 both round (1.0 + x) back to 1.0; 2**-7 (= 1 ULP) does not.
    assert (one + torch.tensor(2.0 ** -12, dtype=torch.bfloat16)) == one   # truncated -> filter is safe
    assert (one + torch.tensor(2.0 ** -8, dtype=torch.bfloat16)) == one    # still rounds to 1.0
    assert (one + torch.tensor(2.0 ** -7, dtype=torch.bfloat16)) != one    # 1 ULP -> representable
