"""Chunked (vocab-blocked) fused linear cross-entropy — the pure-torch CCE reference,
plus the framework-agnostic `triton_linear_cross_entropy` entry point that dispatches
to the Triton kernel on CUDA and FALLS BACK to this chunked-torch reference on CPU.

Deliverable #1 of the /kernel-dev CCE lift (SPEC_cce_fused_linear_ce.md,
CCE = "Cut Your Losses in Large-Vocabulary LMs", arXiv 2411.09009). It computes

    loss = mean_i ( logsumexp_v(H[i] @ W[v])  -  (H[i] @ W[y_i]) )

together with the gradients d_hidden (dH) and d_lm_head_weight (dW), WITHOUT ever
materializing the full (N, V) logits tensor. For Qwen3-0.6B (V = 151,936, D = 1024,
N = 16,384 per micro-batch) that (N, V) fp32 tensor is ~9.96 GB; this reference
never allocates it — the largest intermediate along the vocab axis is one chunk of
`chunk_size` columns, i.e. (N, chunk_size).

WHY A PURE-TORCH REFERENCE (and not "just the Triton kernel")
------------------------------------------------------------
Two reasons, both load-bearing:
  1. It is the CPU-testable ORACLE-adjacent implementation: it runs and is
     numerically correct on CPU at tiny sizes (V=512, N=64, D=32) so the whole
     mechanism — online log-sum-exp, the indexed target logit, the block-wise
     softmax-minus-onehot backward, the eps gradient filter — is unit-tested TODAY
     without a GPU. See research/tests/test_cce_linear_ce.py.
  2. It is itself a genuine memory-saving implementation. The vocab loop means peak
     activation for the CE term is O(N * chunk_size) instead of O(N * V). On a box
     where over-allocation OOM-kills the whole machine (§C1), that alone is useful
     even before the Triton kernel is rooflined off-box.

CORRECTNESS ORACLE (the HARD gate — SPEC §"Correctness oracle"):
    ref = torch.nn.functional.cross_entropy(H.float() @ W.float().T, y)
    forward loss:  atol 1e-3
    backward dH/dW: rtol/atol 1e-2  (bf16, loosened for the eps filter)
NOT bit-exact by design: the eps=2**-12 gradient filter drops sub-bf16-resolution
softmax entries (the CCE key trick), which perturbs the gradient below the bf16
noise floor.

The math here mirrors the Triton kernel (cce_triton.py) line-for-line so the two can
be diffed during hand review. NOTE (honest scope): this reference runs every big GEMM
in fp32; the Triton kernel runs the LOGIT recompute with bf16 tensor-core operands
(fp32 accumulate) and stores bf16 dH/dW. The CPU suite covers the fp32 path AND a
bf16-rounded emulation of the kernel path, but the *actual* kernel numerics are gated
only off-box by cce_triton.gate_against_reference().
"""
from __future__ import annotations

import importlib

import torch

# ---------------------------------------------------------------------------
# eps=2**-12 gradient-filter threshold (CCE, arXiv 2411.09009 §4).
#
# HONEST bf16 rationale (the earlier comment here was numerically wrong): bf16 has
# 7 EXPLICIT mantissa bits, so ULP(1.0) = 2**-7 and the round-to-1.0 boundary is
# ~2**-8 (half a ULP; (1.0 + 2**-8) rounds back to 1.0, (1.0 + 2**-7) does not).
# 2**-12 sits ~32x BELOW that boundary, so it is a deliberately CONSERVATIVE
# SUB-ULP filter: a softmax entry below it, sitting next to the O(1) onehot term
# in the gradient, changes nothing a bf16-resolution accumulate could represent.
# It is NOT "the smallest non-truncated bf16 magnitude" (that is ~2**-7); the value
# is a safe filter floor, and the CCE speedup comes from *skipping* the resulting
# provably-zero blocks, not from bf16 representability per se.
# ---------------------------------------------------------------------------
EPS_BF16 = 2.0 ** -12  # == 0.000244140625  (exactly representable in bf16)

# Populated on every forward() call — a self-reported audit trail the unit test
# cross-checks against an INDEPENDENT torch.matmul shape guard. Records the widest
# vocab-axis tile ever formed so a regression that accidentally builds (N, V) is
# caught even if the independent guard were ever removed.
LAST_STATS: dict = {}


def _validate_labels(y, V, ignore_index):
    """Fail LOUDLY on out-of-range labels, exactly as F.cross_entropy raises an
    IndexError. A fused CE that silently absorbs y>=V or stray-negative y (its
    onehot never subtracted, its target logit never gathered) would hide a
    data-pipeline bug that stock CE surfaces immediately. Cheap on (N,) ints."""
    valid = (y != ignore_index)
    bad = valid & ((y < 0) | (y >= V))
    if bool(bad.any()):
        n_bad = int(bad.sum())
        raise ValueError(
            f"{n_bad} label(s) out of range [0, {V}) with ignore_index={ignore_index} "
            f"(first bad value {int(y[bad][0])}); F.cross_entropy would raise IndexError.")


def _online_lse_and_target(H32, W, y, chunk_size, ignore_index):
    """FORWARD core. One streaming pass over the vocab in blocks of `chunk_size`.

    Returns (lse, zy, max_tile_cols):
      lse (N,) fp32 : the exact per-row log-sum-exp of the full-vocab logits,
                      accumulated with the numerically-stable online (running max
                      m + running sum s) recurrence — never storing all V logits.
      zy  (N,) fp32 : the indexed correct-token logit H[i] @ W[y_i] (0 for rows
                      whose label == ignore_index; their loss is masked later).
      max_tile_cols : the widest vocab tile formed (== chunk_size except the
                      remainder block) — proves (N, V) is never allocated.
    """
    N, D = H32.shape
    V = W.shape[0]
    device = H32.device

    # Online log-sum-exp accumulators, fp32 (SPEC: "fp32 accumulation for the LSE").
    m = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)  # running max
    s = torch.zeros((N,), dtype=torch.float32, device=device)                # running sum of exp(logit - m)
    zy = torch.zeros((N,), dtype=torch.float32, device=device)               # gathered correct-token logit

    row_idx = torch.arange(N, device=device)
    max_tile_cols = 0

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        cols = end - start
        max_tile_cols = max(max_tile_cols, cols)

        Wc = W[start:end].float()                     # (C, D) fp32 view->copy of one vocab block
        # torch.matmul (NOT the @ operator) is used deliberately for EVERY big GEMM so
        # the unit test's global torch.matmul shape guard can observe them and assert
        # none is (N, V). This is a load-bearing contract — do not refactor to @.
        logits = torch.matmul(H32, Wc.t())            # (N, C) fp32  <-- ONLY (N, C), never (N, V)

        # --- streaming, numerically-stable log-sum-exp update ---
        cmax = logits.amax(dim=1)                     # (N,) max over this block
        new_m = torch.maximum(m, cmax)                # new running max
        # s <- s * exp(m - new_m) + sum_c exp(logit_c - new_m). exp(-inf)=0 handles the
        # first block (m starts at -inf) with no NaN because new_m is finite there.
        s = s * torch.exp(m - new_m) + torch.exp(logits - new_m.unsqueeze(1)).sum(dim=1)
        m = new_m

        # --- gather the correct-token logit for rows whose label lands in this block ---
        in_blk = (y >= start) & (y < end)             # (N,) bool; false for ignore_index=-100
        if bool(in_blk.any()):
            rows = row_idx[in_blk]
            local = y[in_blk] - start                 # column within this block
            zy[rows] = logits[rows, local]

    lse = m + torch.log(s)                            # (N,) exact full-vocab logsumexp
    return lse, zy, max_tile_cols


def _blocked_backward(H, W, y, lse, chunk_size, ignore_index, row_scale, grad_filter_eps):
    """BACKWARD core. Recompute the SAME blocked logits (memory-optimal recompute,
    not caching) and accumulate dH, dW block by block in fp32.

    For each vocab block the local gradient wrt the logits is
        dlogit = softmax(logit) - onehot(y)           (the standard CE gradient)
    scaled per row by row_scale (the reduction/grad_output factor, 0 for
    ignore_index rows). Then, exactly as the fused kernel does:
        dH   += dlogit @ W_block          (N,C)@(C,D) -> (N,D)
        dW_block += dlogit.T @ H          (C,N)@(N,D) -> (C,D)

    GRADIENT FILTER (CCE): softmax entries below grad_filter_eps are zeroed before
    forming dlogit. The correct-token -1 (onehot) term is subtracted AFTER the
    filter, so the exact target gradient is always preserved even if its own
    softmax prob happened to fall below eps.
    """
    H32 = H.float()
    N, D = H32.shape
    V = W.shape[0]
    device = H32.device

    dH32 = torch.zeros((N, D), dtype=torch.float32, device=device)   # fp32 accumulate (SPEC)
    dW32 = torch.zeros((V, D), dtype=torch.float32, device=device)   # fp32 accumulate (SPEC)
    row_idx = torch.arange(N, device=device)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)

        Wc = W[start:end].float()                     # (C, D)
        logits = torch.matmul(H32, Wc.t())            # (N, C) fp32 — recomputed, never (N, V)

        # softmax via the saved (exact) lse: p = exp(logit - lse)
        p = torch.exp(logits - lse.unsqueeze(1))      # (N, C) in [0, 1]

        # --- eps gradient filter: drop softmax entries below the bf16 floor ---
        if grad_filter_eps and grad_filter_eps > 0.0:
            p = torch.where(p >= grad_filter_eps, p, torch.zeros((), dtype=p.dtype, device=device))

        # --- subtract the onehot (correct token) AFTER filtering: exact target term ---
        in_blk = (y >= start) & (y < end)
        if bool(in_blk.any()):
            rows = row_idx[in_blk]
            local = y[in_blk] - start
            p[rows, local] -= 1.0

        # per-row scale: grad_output * (reduction factor); 0 for ignore_index rows,
        # so their dlogit row is 0 and contributes nothing to dH/dW.
        dlogit = p * row_scale.unsqueeze(1)           # (N, C) fp32

        dH32 += torch.matmul(dlogit, Wc)              # (N, C) @ (C, D) -> (N, D)
        dW32[start:end] += torch.matmul(dlogit.t(), H32)  # (C, N) @ (N, D) -> (C, D)

    return dH32.to(H.dtype), dW32.to(W.dtype)


class _LinearCrossEntropy(torch.autograd.Function):
    """autograd.Function wiring the blocked forward/backward. Only H and W receive
    gradients; y and the scalar config args return None."""

    @staticmethod
    def forward(ctx, H, W, y, chunk_size, ignore_index, reduction, grad_filter_eps):
        if H.dim() != 2 or W.dim() != 2 or H.shape[1] != W.shape[1]:
            raise ValueError(f"expected H (N,D), W (V,D) with matching D; got {tuple(H.shape)}, {tuple(W.shape)}")
        if y.dim() != 1 or y.shape[0] != H.shape[0]:
            raise ValueError(f"expected y (N,) matching H rows; got {tuple(y.shape)}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        V = W.shape[0]
        _validate_labels(y, V, ignore_index)

        H32 = H.float()
        lse, zy, max_tile_cols = _online_lse_and_target(H32, W, y, chunk_size, ignore_index)

        valid = (y != ignore_index)                   # (N,) bool
        n_valid = int(valid.sum().item())
        per_row = lse - zy                            # (N,) fp32 per-row CE
        per_row = torch.where(valid, per_row, torch.zeros((), dtype=per_row.dtype, device=per_row.device))

        if reduction == "mean":
            # Match F.cross_entropy exactly: an all-ignored batch (n_valid==0) is 0/0 ->
            # NaN, NOT a silent 0.0 that would hide a fully-masked (labeling-bug) batch.
            denom = torch.tensor(float(n_valid), dtype=per_row.dtype, device=per_row.device)
            loss = per_row.sum() / denom
        elif reduction == "sum":
            loss = per_row.sum()
        elif reduction == "none":
            loss = per_row
        else:
            raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")

        ctx.save_for_backward(H, W, y, lse)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.grad_filter_eps = grad_filter_eps
        ctx.n_valid = n_valid

        LAST_STATS.clear()
        LAST_STATS.update(
            N=int(H.shape[0]), D=int(H.shape[1]), V=int(W.shape[0]),
            chunk_size=int(chunk_size), n_chunks=(int(W.shape[0]) + chunk_size - 1) // chunk_size,
            max_logits_tile_cols=int(max_tile_cols), n_valid=n_valid, reduction=reduction,
        )
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        H, W, y, lse = ctx.saved_tensors
        valid = (y != ctx.ignore_index)

        # Build the per-row scale = grad_output * d(reduction)/d(per_row_i), 0 for ignored rows.
        if ctx.reduction == "mean":
            base = grad_output / max(ctx.n_valid, 1)          # scalar tensor
            row_scale = torch.where(valid, base.to(torch.float32).expand(y.shape[0]),
                                    torch.zeros((), dtype=torch.float32, device=y.device))
        elif ctx.reduction == "sum":
            row_scale = torch.where(valid, grad_output.to(torch.float32).expand(y.shape[0]),
                                    torch.zeros((), dtype=torch.float32, device=y.device))
        else:  # "none": grad_output is (N,)
            row_scale = torch.where(valid, grad_output.to(torch.float32),
                                    torch.zeros((), dtype=torch.float32, device=y.device))

        dH, dW = _blocked_backward(
            H, W, y, lse, ctx.chunk_size, ctx.ignore_index, row_scale, ctx.grad_filter_eps)
        # forward signature: (H, W, y, chunk_size, ignore_index, reduction, grad_filter_eps)
        return dH, dW, None, None, None, None, None


def linear_cross_entropy(
    H: torch.Tensor,
    W: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int = 8192,
    ignore_index: int = -100,
    reduction: str = "mean",
    grad_filter_eps: float = EPS_BF16,
) -> torch.Tensor:
    """Fused linear cross-entropy, vocab-chunked, never materializing (N, V).

    Args:
        H: (N, D) hidden states (any float dtype; upcast to fp32 internally).
        W: (V, D) lm_head weight (tied to embed). Its .grad slot receives dW.
        y: (N,) int64 target token ids. ignore_index rows are dropped from the mean.
        chunk_size: vocab block width. Peak CE activation is O(N * chunk_size).
            Correctness is independent of chunk_size (streaming LSE is exact);
            it is purely a memory/perf knob. Production: 8192 (19 blocks over 151,936).
        ignore_index: label value excluded from loss + gradient (default -100, as
            torch.nn.functional.cross_entropy).
        reduction: 'mean' | 'sum' | 'none'.
        grad_filter_eps: softmax entries below this are zeroed in the backward
            (CCE trick). Pass 0.0 to disable the filter (exact CE gradient).

    Returns:
        Scalar loss (mean/sum) or (N,) per-row loss (none).
    """
    return _LinearCrossEntropy.apply(
        H, W, y, chunk_size, ignore_index, reduction, grad_filter_eps)


# ======================================================================================
# Framework-agnostic entry point: Triton on CUDA, chunked-torch reference otherwise.
# ======================================================================================
def _load_cce_triton():
    """Import the sibling cce_triton module by whatever import path is live (added to
    sys.path as a top-level module by the tests / kernel-dev harness, or as a package
    submodule). Returns the module or None if it cannot be imported."""
    for name in ("cce_triton", "research.kernel.cce_triton"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def triton_linear_cross_entropy(
    H: torch.Tensor,
    W: torch.Tensor,
    y: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    grad_filter_eps: float = EPS_BF16,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """The production entry point. Dispatches to the fused Triton kernel when Triton
    is importable AND all inputs are on CUDA; otherwise transparently FALLS BACK to
    the chunked-torch reference above (so this function imports and runs on a
    triton-less / CPU-only box — e.g. in CI).

    The Triton path has NO chunk_size knob (blocking is set by the kernel launch
    tiles); chunk_size only steers the CPU/torch fallback. Semantics are identical
    across both paths (same math, same eps filter, same reduction policy)."""
    on_cuda = bool(H.is_cuda and W.is_cuda and y.is_cuda)
    if on_cuda:
        mod = _load_cce_triton()
        if mod is not None and getattr(mod, "HAS_TRITON", False):
            return mod.triton_linear_cross_entropy(
                H, W, y, ignore_index=ignore_index, reduction=reduction,
                grad_filter_eps=grad_filter_eps)
    # CPU / triton-less fallback: the unit-tested chunked reference.
    return linear_cross_entropy(
        H, W, y, chunk_size=chunk_size, ignore_index=ignore_index,
        reduction=reduction, grad_filter_eps=grad_filter_eps)
