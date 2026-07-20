"""Fused linear cross-entropy — Triton kernels (the perf target of the CCE lift).

Deliverable #2 of SPEC_cce_fused_linear_ce.md. This is the fused forward + backward
Triton implementation of the SAME mechanism as the pure-torch reference in
cce_linear_ce.py. It is written to be reviewed LINE-FOR-LINE against that reference
(the mechanism is identical; only the loop nest and memory hierarchy differ).

>>> IMPORTANT — NOT YET GPU-VALIDATED <<<
The GB10 GPU is occupied for hours, so these kernels have NOT been compiled or run on
hardware yet. They are provided for HAND REVIEW of the numerics (LSE stability, the
softmax-minus-onehot backward, the eps filter, block/tile indexing, dtype and fp32
accumulation). The correctness GATE (§C21) is gate_against_reference() below — run it
on a profileable GPU BEFORE ANY benchmark. A kernel that fails the gate is DISCARDED,
never rooflined.

WHAT THE CPU SUITE DOES *NOT* COVER (be honest): research/tests/test_cce_linear_ce.py
gates the fp32 torch reference and a bf16-ROUNDED CPU emulation of this kernel's path,
but it cannot run these Triton kernels. The bf16 tensor-core logit recompute, the exact
tile scheduling, register/SRAM occupancy, and the D-tiled accumulation below are proven
ONLY by the off-box gate. CPU-green != kernel-correct.

VOCAB REMAINDER (correcting an earlier false note): V = 151,936 = 2**7 * 1187, so it IS
divisible by 16/32/64/128. At the default divisor tiles (BLOCK_V in {64,128}) the
`mask_v = offs_v < V` remainder branch is INACTIVE (always true) and is dead code in
production. It exists only so an autotuner may legally pick a non-divisor BLOCK_V (e.g.
256 -> remainder 128); if it ever does, that masked path must be exercised on GPU first.

DESIGN (three kernels, no atomics)
----------------------------------
  fwd    : grid over row-blocks. Streaming online LSE over vocab-blocks in SRAM
           (fp32 running max m + running sum s), plus the indexed target logit.
           Writes per-row LSE (N,) and ZY (N,). The scalar loss is reduced on the
           host from those two (N,) vectors — cheap, and still never (N, V).
  bwd_dH : grid over row-blocks. Each program OWNS the dH rows for its block (no
           atomics). D is TILED (outer loop over BLOCK_D output columns); the logit
           recompute uses the SAME inner BLOCK_D contraction + bf16 operands as the
           forward, so exp(logit - lse) reconstructs a softmax consistent with the
           forward's LSE. Never holds a (·, D) tile whole.
  bwd_dW : grid over vocab-blocks. Each program OWNS the dW rows for its block; same
           D-tiled structure.

Resource fix (was a blocker): the previous version held (BLOCK_N, D)/(BLOCK_V, D) fp32
accumulators and whole-D operand tiles at D=1024 (128-256 KB) — a near-certain
out-of-resource / heavy-spill on a single SM. Both backward kernels now tile D so the
widest live tile is (BLOCK_*, BLOCK_D). The cost is that the logits are recomputed once
per output-D block (D/BLOCK_D times); that is a memory-safe DEFAULT the off-box tuner
can trade back for speed (larger BLOCK_D, or a cached-dlogit variant) once it compiles.

Numeric-consistency fix: forward and both backward kernels contract the hidden dim in
the SAME BLOCK_D order with the SAME bf16 operands, so the backward-recomputed logits
match the forward's (bf16 matmul is non-associative — mismatched tiling would make the
per-row softmax not sum to 1 against the saved LSE).

Precision fix: the ACCUMULATE GEMMs (dlogit @ W, dlogit.T @ H) keep dlogit in fp32 and
cast the W/H operand to fp32, matching the fp32 reference (the old kernel downcast
dlogit to bf16, injecting ~0.4% error). The off-box engineer may switch these to bf16
tensor-core for speed IF the gate still passes at fp32-cast.

Reference template: apple/ml-cross-entropy (MIT). CCE: arXiv 2411.09009.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only / triton-less box
    triton = None
    tl = None
    HAS_TRITON = False

# CCE gradient-filter threshold. See cce_linear_ce.EPS_BF16 for the (corrected) bf16
# rationale: 2**-12 is a CONSERVATIVE SUB-ULP floor (bf16 ULP(1.0) = 2**-7, round-to-1
# boundary ~2**-8), NOT "the smallest non-truncated bf16 magnitude". Accumulation here
# is fp32, so the filter's justification is the SPEC's <0.02% dropped-mass-for-peaked-
# softmax + the loosened 1e-2 tolerance + the block-skip sparsity — valid for peaked
# (trained-model) softmax, still an open question at pretraining scale.
EPS_BF16 = 2.0 ** -12


# ======================================================================================
# Kernels (only defined when Triton is importable; otherwise the wrappers raise clearly)
# ======================================================================================
if HAS_TRITON:

    @triton.jit
    def _cce_fwd_kernel(
        H_ptr, W_ptr, Y_ptr, LSE_ptr, ZY_ptr,
        N, V, D,
        stride_hn, stride_hd,
        stride_wv, stride_wd,
        IGNORE_INDEX: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Forward: per-row online log-sum-exp over the full vocab + indexed target logit.

        One program handles BLOCK_N rows. It streams over the vocab in BLOCK_V-wide
        blocks, and within each vocab block contracts the full hidden dim D in BLOCK_D
        steps. All reductions are fp32. Never materializes (N, V): the widest live logit
        tile is (BLOCK_N, BLOCK_V)."""
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # (BN,)
        mask_n = offs_n < N

        # target ids as int64 (labels are int64); ignore rows read as IGNORE_INDEX so
        # they can never match a (non-negative) vocab column.
        y = tl.load(Y_ptr + offs_n, mask=mask_n, other=IGNORE_INDEX).to(tl.int64)   # (BN,)

        m = tl.full((BLOCK_N,), -float("inf"), tl.float32)       # running max
        s = tl.zeros((BLOCK_N,), tl.float32)                     # running sum of exp(logit - m)
        zy = tl.zeros((BLOCK_N,), tl.float32)                    # gathered target logit

        for v0 in range(0, V, BLOCK_V):
            offs_v = v0 + tl.arange(0, BLOCK_V)                  # (BV,) int32
            mask_v = offs_v < V                                 # REMAINDER MASK (inactive at divisor tiles)
            offs_v64 = offs_v.to(tl.int64)                      # int64 for the target compare

            # ---- logits tile (BN, BV) fp32, contract D in BLOCK_D steps (bf16 operands) ----
            acc = tl.zeros((BLOCK_N, BLOCK_V), tl.float32)
            for d0 in range(0, D, BLOCK_D):
                cur_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = cur_d < D
                h = tl.load(
                    H_ptr + offs_n[:, None] * stride_hn + cur_d[None, :] * stride_hd,
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0,
                )                                                # (BN, BD) bf16
                w = tl.load(
                    W_ptr + offs_v[:, None] * stride_wv + cur_d[None, :] * stride_wd,
                    mask=mask_v[:, None] & mask_d[None, :], other=0.0,
                )                                                # (BV, BD) bf16
                acc += tl.dot(h, tl.trans(w))                   # (BN, BV) += (BN,BD)@(BD,BV), fp32 acc

            # mask padded vocab columns to -inf so they never enter max/sum/gather
            acc = tl.where(mask_v[None, :], acc, -float("inf"))

            # ---- streaming stable LSE update ----
            cmax = tl.max(acc, axis=1)                          # (BN,)
            new_m = tl.maximum(m, cmax)
            p = tl.exp(acc - new_m[:, None])                    # (BN, BV); masked cols -> exp(-inf)=0
            s = s * tl.exp(m - new_m) + tl.sum(p, axis=1)       # exp(-inf)=0 on the first block
            m = new_m

            # ---- gather the correct-token logit for rows whose label is in this block ----
            in_blk = (y[:, None] == offs_v64[None, :]) & mask_v[None, :]   # (BN, BV) bool
            zy += tl.sum(tl.where(in_blk, acc, 0.0), axis=1)    # exactly one hit per matching row

        lse = m + tl.log(s)                                     # (BN,)
        tl.store(LSE_ptr + offs_n, lse, mask=mask_n)
        tl.store(ZY_ptr + offs_n, zy, mask=mask_n)

    @triton.jit
    def _cce_recompute_logits(
        H_ptr, W_ptr, offs_n, mask_n, offs_v, mask_v,
        stride_hn, stride_hd, stride_wv, stride_wd,
        D, BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Shared helper: recompute one (BN, BV) logits tile with the EXACT same
        BLOCK_D contraction order + bf16 operands as the forward, so the backward's
        softmax is consistent with the saved forward LSE."""
        acc = tl.zeros((BLOCK_N, BLOCK_V), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            cur_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = cur_d < D
            h = tl.load(
                H_ptr + offs_n[:, None] * stride_hn + cur_d[None, :] * stride_hd,
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            w = tl.load(
                W_ptr + offs_v[:, None] * stride_wv + cur_d[None, :] * stride_wd,
                mask=mask_v[:, None] & mask_d[None, :], other=0.0,
            )
            acc += tl.dot(h, tl.trans(w))
        return acc

    @triton.jit
    def _cce_bwd_dh_kernel(
        H_ptr, W_ptr, Y_ptr, LSE_ptr, SCALE_ptr, DH_ptr,
        N, V, D,
        stride_hn, stride_hd,
        stride_wv, stride_wd,
        stride_dhn, stride_dhd,
        EPS: tl.constexpr, IGNORE_INDEX: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
        TILE_SKIP: tl.constexpr,
    ):
        """Backward dH. Grid over row-blocks; each program owns dH rows [pid_n] (no
        atomics). D is TILED via an OUTER loop over BLOCK_D output columns, so the
        accumulator is only (BLOCK_N, BLOCK_D) — never (BLOCK_N, D).

        dH[:, d_out] = sum_v dlogit[:, v] * W[v, d_out],  dlogit = filter(softmax) - onehot."""
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)         # (BN,)
        mask_n = offs_n < N

        y = tl.load(Y_ptr + offs_n, mask=mask_n, other=IGNORE_INDEX).to(tl.int64)  # (BN,)
        lse = tl.load(LSE_ptr + offs_n, mask=mask_n, other=0.0)         # (BN,) fp32
        scale = tl.load(SCALE_ptr + offs_n, mask=mask_n, other=0.0)     # (BN,) fp32 per-row grad scale

        for d_out in range(0, D, BLOCK_D):                      # OUTER over output D
            offs_do = d_out + tl.arange(0, BLOCK_D)
            mask_do = offs_do < D
            acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)      # small (BN, BD) accumulator

            for v0 in range(0, V, BLOCK_V):
                offs_v = v0 + tl.arange(0, BLOCK_V)
                mask_v = offs_v < V                             # REMAINDER MASK
                offs_v64 = offs_v.to(tl.int64)

                logits = _cce_recompute_logits(
                    H_ptr, W_ptr, offs_n, mask_n, offs_v, mask_v,
                    stride_hn, stride_hd, stride_wv, stride_wd,
                    D, BLOCK_N, BLOCK_V, BLOCK_D)
                logits = tl.where(mask_v[None, :], logits, -float("inf"))

                p = tl.exp(logits - lse[:, None])               # softmax (BN, BV) in [0,1]
                p = tl.where(p >= EPS, p, 0.0)                  # eps gradient filter (elementwise, exact)

                in_blk = (y[:, None] == offs_v64[None, :]) & mask_v[None, :]
                dlogit = tl.where(in_blk, p - 1.0, p)           # subtract onehot AFTER filter
                dlogit = dlogit * scale[:, None]
                dlogit = tl.where(mask_v[None, :], dlogit, 0.0) # zero padded cols

                # W tile for the output-D columns only (BV, BD) — never (BV, D)
                w_out = tl.load(
                    W_ptr + offs_v[:, None] * stride_wv + offs_do[None, :] * stride_wd,
                    mask=mask_v[:, None] & mask_do[None, :], other=0.0,
                )
                # fp32 accumulate GEMM (operand cast to fp32 to match the fp32 reference).
                # TILE_SKIP is a compile-time toggle for the CCE block-skip: when a tile's
                # dlogit is provably all-zero (fully filtered, no target) the accumulate is
                # a no-op and can be skipped. Data-dependent control flow is version-
                # sensitive in Triton, so it DEFAULTS OFF (always-accumulate = the
                # correct, always-compilable fallback); the elementwise filter above
                # already guarantees correctness. Flip on off-box after confirming it
                # compiles + measuring the real skip rate on a PEAKED (trained) softmax.
                if TILE_SKIP:
                    if tl.max(tl.abs(dlogit)) > 0.0:
                        acc += tl.dot(dlogit, w_out.to(tl.float32))
                else:
                    acc += tl.dot(dlogit, w_out.to(tl.float32))

            tl.store(
                DH_ptr + offs_n[:, None] * stride_dhn + offs_do[None, :] * stride_dhd,
                acc, mask=mask_n[:, None] & mask_do[None, :],
            )

    @triton.jit
    def _cce_bwd_dw_kernel(
        H_ptr, W_ptr, Y_ptr, LSE_ptr, SCALE_ptr, DW_ptr,
        N, V, D,
        stride_hn, stride_hd,
        stride_wv, stride_wd,
        stride_dwv, stride_dwd,
        EPS: tl.constexpr, IGNORE_INDEX: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
        TILE_SKIP: tl.constexpr,
    ):
        """Backward dW. Grid over vocab-blocks; each program owns dW rows [pid_v] (no
        atomics). D is TILED via an OUTER loop over BLOCK_D output columns, so the
        accumulator is only (BLOCK_V, BLOCK_D) — never (BLOCK_V, D).

        dW[:, d_out] = sum_n dlogit[n, :].T * H[n, d_out]."""
        pid_v = tl.program_id(0)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)         # (BV,)
        mask_v = offs_v < V                                      # REMAINDER MASK
        offs_v64 = offs_v.to(tl.int64)

        for d_out in range(0, D, BLOCK_D):                      # OUTER over output D
            offs_do = d_out + tl.arange(0, BLOCK_D)
            mask_do = offs_do < D
            acc = tl.zeros((BLOCK_V, BLOCK_D), tl.float32)      # small (BV, BD) accumulator

            for n0 in range(0, N, BLOCK_N):
                offs_n = n0 + tl.arange(0, BLOCK_N)
                mask_n = offs_n < N

                y = tl.load(Y_ptr + offs_n, mask=mask_n, other=IGNORE_INDEX).to(tl.int64)
                lse = tl.load(LSE_ptr + offs_n, mask=mask_n, other=0.0)
                scale = tl.load(SCALE_ptr + offs_n, mask=mask_n, other=0.0)

                logits = _cce_recompute_logits(
                    H_ptr, W_ptr, offs_n, mask_n, offs_v, mask_v,
                    stride_hn, stride_hd, stride_wv, stride_wd,
                    D, BLOCK_N, BLOCK_V, BLOCK_D)
                logits = tl.where(mask_v[None, :], logits, -float("inf"))

                p = tl.exp(logits - lse[:, None])               # (BN, BV)
                p = tl.where(p >= EPS, p, 0.0)                  # eps filter

                in_blk = (y[:, None] == offs_v64[None, :]) & mask_v[None, :]
                dlogit = tl.where(in_blk, p - 1.0, p)
                dlogit = dlogit * scale[:, None]
                dlogit = tl.where(mask_n[:, None] & mask_v[None, :], dlogit, 0.0)

                h_out = tl.load(
                    H_ptr + offs_n[:, None] * stride_hn + offs_do[None, :] * stride_hd,
                    mask=mask_n[:, None] & mask_do[None, :], other=0.0,
                )
                if TILE_SKIP:
                    if tl.max(tl.abs(dlogit)) > 0.0:
                        acc += tl.dot(tl.trans(dlogit), h_out.to(tl.float32))
                else:
                    acc += tl.dot(tl.trans(dlogit), h_out.to(tl.float32))

            tl.store(
                DW_ptr + offs_v[:, None] * stride_dwv + offs_do[None, :] * stride_dwd,
                acc, mask=mask_v[:, None] & mask_do[None, :],
            )


# ======================================================================================
# Python wrappers / autograd glue
# ======================================================================================
def _require_triton_cuda(H, W, y):
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not importable — the Triton CCE kernel needs a CUDA box with "
            "triton installed. Use cce_linear_ce.linear_cross_entropy for the CPU reference.")
    if not (H.is_cuda and W.is_cuda and y.is_cuda):
        raise RuntimeError("Triton CCE kernel requires all inputs on CUDA.")


# Default launch tiles. PLACEHOLDERS for an off-box autotune sweep — chosen so that no
# (·, D) tile is ever held whole (D is tiled by BLOCK_D). Not tuned for occupancy.
# BLOCK_V is kept a divisor of V=151936 (64) so the remainder mask stays inactive.
_FWD_TILES = dict(BLOCK_N=64, BLOCK_V=64, BLOCK_D=128)
_DH_TILES = dict(BLOCK_N=32, BLOCK_V=64, BLOCK_D=128)
_DW_TILES = dict(BLOCK_N=32, BLOCK_V=64, BLOCK_D=128)
# CCE block-skip. OFF by default (always-accumulate = correct + always compiles).
# Turn on off-box only after confirming the data-dependent branch compiles on the
# target Triton/GB10 AND measuring the real skip rate on a peaked softmax.
_TILE_SKIP = False


def _validate_labels(y, V, ignore_index):
    valid = (y != ignore_index)
    bad = valid & ((y < 0) | (y >= V))
    if bool(bad.any()):
        raise ValueError(
            f"{int(bad.sum())} label(s) out of range [0, {V}) with ignore_index={ignore_index}; "
            f"F.cross_entropy would raise IndexError.")


class _TritonLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, W, y, ignore_index, reduction, grad_filter_eps):
        _require_triton_cuda(H, W, y)
        N, D = H.shape
        V = W.shape[0]
        _validate_labels(y, V, ignore_index)

        lse = torch.empty((N,), dtype=torch.float32, device=H.device)
        zy = torch.empty((N,), dtype=torch.float32, device=H.device)

        grid = (triton.cdiv(N, _FWD_TILES["BLOCK_N"]),)
        _cce_fwd_kernel[grid](
            H, W, y, lse, zy,
            N, V, D,
            H.stride(0), H.stride(1),
            W.stride(0), W.stride(1),
            IGNORE_INDEX=ignore_index, **_FWD_TILES,
        )

        valid = (y != ignore_index)
        n_valid = int(valid.sum().item())
        per_row = torch.where(valid, lse - zy, torch.zeros((), dtype=torch.float32, device=H.device))
        if reduction == "mean":
            # 0/0 -> NaN for an all-ignored batch, matching F.cross_entropy.
            denom = torch.tensor(float(n_valid), dtype=torch.float32, device=H.device)
            loss = per_row.sum() / denom
        elif reduction == "sum":
            loss = per_row.sum()
        elif reduction == "none":
            loss = per_row
        else:
            raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")

        ctx.save_for_backward(H, W, y, lse)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.grad_filter_eps = grad_filter_eps
        ctx.n_valid = n_valid
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        H, W, y, lse = ctx.saved_tensors
        N, D = H.shape
        V = W.shape[0]
        valid = (y != ctx.ignore_index)

        # per-row grad scale (same policy as the torch reference)
        if ctx.reduction == "mean":
            base = (grad_output / max(ctx.n_valid, 1)).to(torch.float32)
            row_scale = torch.where(valid, base.expand(N), torch.zeros((), dtype=torch.float32, device=H.device))
        elif ctx.reduction == "sum":
            row_scale = torch.where(valid, grad_output.to(torch.float32).expand(N),
                                    torch.zeros((), dtype=torch.float32, device=H.device))
        else:  # none
            row_scale = torch.where(valid, grad_output.to(torch.float32),
                                    torch.zeros((), dtype=torch.float32, device=H.device))
        row_scale = row_scale.contiguous()

        dH = torch.zeros_like(H)
        dW = torch.zeros_like(W)
        eps = ctx.grad_filter_eps if ctx.grad_filter_eps else 0.0

        grid_dh = (triton.cdiv(N, _DH_TILES["BLOCK_N"]),)
        _cce_bwd_dh_kernel[grid_dh](
            H, W, y, lse, row_scale, dH,
            N, V, D,
            H.stride(0), H.stride(1),
            W.stride(0), W.stride(1),
            dH.stride(0), dH.stride(1),
            EPS=eps, IGNORE_INDEX=ctx.ignore_index, TILE_SKIP=_TILE_SKIP, **_DH_TILES,
        )

        grid_dw = (triton.cdiv(V, _DW_TILES["BLOCK_V"]),)
        _cce_bwd_dw_kernel[grid_dw](
            H, W, y, lse, row_scale, dW,
            N, V, D,
            H.stride(0), H.stride(1),
            W.stride(0), W.stride(1),
            dW.stride(0), dW.stride(1),
            EPS=eps, IGNORE_INDEX=ctx.ignore_index, TILE_SKIP=_TILE_SKIP, **_DW_TILES,
        )
        # forward signature: (H, W, y, ignore_index, reduction, grad_filter_eps) -> 6 inputs,
        # so backward MUST return exactly 6 grads (dH, dW, then one None per config arg).
        return dH, dW, None, None, None, None


def triton_linear_cross_entropy(
    H: torch.Tensor,
    W: torch.Tensor,
    y: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    grad_filter_eps: float = EPS_BF16,
) -> torch.Tensor:
    """Triton fused linear cross-entropy (CUDA only). Same signature/semantics as
    cce_linear_ce.linear_cross_entropy but without the chunk_size knob (blocking is set
    by the kernel launch tiles). See module docstring for the correctness gate. For a
    CPU-safe entry that falls back to the reference, use
    cce_linear_ce.triton_linear_cross_entropy."""
    return _TritonLinearCrossEntropy.apply(H, W, y, ignore_index, reduction, grad_filter_eps)


# ======================================================================================
# §C21 HARD correctness gate (OFF-BOX, needs CUDA). MUST pass before any roofline.
# ======================================================================================
def gate_against_reference(N=1024, D=1024, V=8192, dtype=None, device="cuda",
                           atol_loss=1e-3, rtol_grad=2e-2, seed=0):
    """OFF-BOX correctness gate (§C21). Compares the Triton kernel to the model's own
    fp32 unfused CE and returns a verdict dict. MUST pass before any roofline.

    THREE fixes vs the original vacuous gate:
      1. PEAKED inputs (0.5 scale, matching the meaningful CPU-test regime) so the
         softmax concentrates and the eps filter actually DROPS mass — not the
         near-uniform regime where the filter was a no-op and every check was benign.
      2. RELATIVE grad tolerance: atol is tied to the largest reference-grad magnitude
         (rtol_grad * |ref|.max()), because under mean-reduction the gradients are
         O(1/N) ~ 1e-4 and a FIXED atol=1e-2 was ~100x larger than the signal — the old
         gate greened an all-zero backward. The check is now scaled to the data.
      3. NEGATIVE CONTROLS: the gate only PASSES if a zero-gradient AND a half-gradient
         candidate both FAIL the same comparison. If they don't, the tolerance is
         vacuous and the gate returns passed=False (fail closed) regardless of the
         real candidate — so a broken/absent backward can never be certified.

    Also reports the oracle-INDEPENDENT eps bound: filter-on vs filter-off FUSED grads.
    Not run in CI (needs CUDA + the busy GB10/rented GPU). Mirrors SPEC tolerances."""
    import torch.nn.functional as F
    if dtype is None:
        dtype = torch.bfloat16
    g = torch.Generator(device=device).manual_seed(seed)
    # Peaked: 0.5 scale -> logit std ~ 0.25*sqrt(D), a concentrated softmax over V.
    H = torch.randn(N, D, device=device, dtype=dtype, generator=g) * 0.5
    W = torch.randn(V, D, device=device, dtype=dtype, generator=g) * 0.5
    y = torch.randint(0, V, (N,), device=device, generator=g)

    # --- reference: fp32 unfused CE (the HARD oracle) ---
    Href = H.detach().clone().float().requires_grad_(True)
    Wref = W.detach().clone().float().requires_grad_(True)
    ref_loss = F.cross_entropy(torch.matmul(Href, Wref.t()), y)
    ref_loss.backward()
    ref_dH, ref_dW = Href.grad, Wref.grad

    # --- candidate: Triton fused (default eps filter ON) ---
    Hc = H.detach().clone().requires_grad_(True)
    Wc = W.detach().clone().requires_grad_(True)
    cand_loss = triton_linear_cross_entropy(Hc, Wc, y)
    cand_loss.backward()
    cand_dH, cand_dW = Hc.grad.float(), Wc.grad.float()

    # relative, data-scaled tolerances (fix #2)
    dH_atol = rtol_grad * float(ref_dH.abs().max())
    dW_atol = rtol_grad * float(ref_dW.abs().max())

    def _passes(cand, ref, atol):
        return bool(((cand - ref).abs() <= atol + rtol_grad * ref.abs()).all())

    loss_ok = abs(float(cand_loss) - float(ref_loss)) <= atol_loss
    dH_ok = _passes(cand_dH, ref_dH, dH_atol)
    dW_ok = _passes(cand_dW, ref_dW, dW_atol)

    # negative controls (fix #3): zero AND half grads MUST fail, else the gate is vacuous.
    zero_fails = (not _passes(torch.zeros_like(ref_dH), ref_dH, dH_atol)) and \
                 (not _passes(torch.zeros_like(ref_dW), ref_dW, dW_atol))
    half_fails = (not _passes(0.5 * ref_dH, ref_dH, dH_atol)) and \
                 (not _passes(0.5 * ref_dW, ref_dW, dW_atol))
    non_vacuous = zero_fails and half_fails

    # oracle-independent eps bound: filter-off vs filter-on FUSED grads.
    Hc0 = H.detach().clone().requires_grad_(True)
    Wc0 = W.detach().clone().requires_grad_(True)
    loss0 = triton_linear_cross_entropy(Hc0, Wc0, y, grad_filter_eps=0.0)
    loss0.backward()
    eps_dH_gap = float((Hc.grad.float() - Hc0.grad.float()).abs().max())
    eps_dW_gap = float((Wc.grad.float() - Wc0.grad.float()).abs().max())

    passed = bool(loss_ok and dH_ok and dW_ok and non_vacuous)
    return {
        "passed": passed,
        "loss_ref": float(ref_loss), "loss_cand": float(cand_loss),
        "d_loss": abs(float(ref_loss) - float(cand_loss)), "loss_ok": loss_ok,
        "dH_max_err": float((cand_dH - ref_dH).abs().max()), "dH_ok": dH_ok, "dH_atol": dH_atol,
        "dW_max_err": float((cand_dW - ref_dW).abs().max()), "dW_ok": dW_ok, "dW_atol": dW_atol,
        "ref_dH_absmax": float(ref_dH.abs().max()), "ref_dW_absmax": float(ref_dW.abs().max()),
        "negative_control_zero_fails": zero_fails,
        "negative_control_half_fails": half_fails,
        "non_vacuous": non_vacuous,
        "eps_filter_dH_gap": eps_dH_gap, "eps_filter_dW_gap": eps_dW_gap,
        "note": ("passed=True requires the real candidate to match AND the zero/half "
                 "negative controls to FAIL (proving the tolerance is not vacuous)."),
    }


if __name__ == "__main__":  # pragma: no cover - off-box smoke
    if HAS_TRITON and torch.cuda.is_available():
        import json
        print(json.dumps(gate_against_reference(N=1024), indent=2, default=str))
    else:
        print("Triton/CUDA unavailable — cannot run the Triton CCE gate here. "
              "Use cce_linear_ce.linear_cross_entropy on CPU for the unit-tested reference.")
