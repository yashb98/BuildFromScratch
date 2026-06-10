# Build 2 — Modernized (full IMU-1 bundle) — VERIFIED SPEC

Primary sources (fetched + verified 2026-06-08):
- IMU-1: arXiv **2602.02522** (Jan 2026) — bundle + hyperparameters
- NorMuon: arXiv **2510.05491** (Oct 2025) — optimizer algorithm

User decision: implement the FULL bundle (not single-variable), smoke-test first, then full 2-TPP run. This is a "full recipe vs faithful baseline" comparison — it intentionally confounds many changes; NOT a controlled single-variable test.

## NorMuon optimizer (2510.05491) — VERIFIED
- Momentum: `M_t = β1·M_{t-1} + (1-β1)·G_t`, **β1=0.95**
- Orthogonalize: `O_t = NS5(M_t)` — Frobenius-normalize `X0=M/‖M‖_F`, then 5 Newton-Schulz iters of `X = a·X + b·(XXᵀ)X + c·(XXᵀ)²X`. Coeffs **NOT in paper**; cites Jordan 2024 → use standard Muon **(a,b,c)=(3.4445, -4.7750, 2.0315)** [LABELED as standard, not paper-quoted].
- Row 2nd-moment: `v_t = β2·v_{t-1} + (1-β2)·mean_cols(O_t⊙O_t)`, **β2=0.95**
- Normalize: `Ô_t = O_t / (√V_t + ε)` (V_t = v_t broadcast across cols); ε value not specified → use 1e-8.
- LR scale: `η̂ = 0.2·η·√(m·n) / ‖Ô_t‖_F`
- Update: `W = W − η·λ·W − η̂·Ô_t`
- Scope: NorMuon on **2D hidden matrices** (q/k/v/o, gate/up/down proj). Adam on **embeddings, unembedding, norms, scalars/bias**.

## IMU-1 architecture additions (2602.02522) — VERIFIED
- Value residuals (Eq4): `V(l) = s·(α1·V_local(l) + α2·V(1)) / √(α1²+α2²)`, init `(s,α1,α2)=(1,1,0)` (learnable). Needs first-layer V piped to all layers.
- LayerNorm scaling (Eq5): `LN_l(x) = (1/√l)·Norm(x)`, layer index l=1..L.
- Per-head gating (Eq3): `out_h = 2·σ(g_h)·Attn_h`, `g = W_g·x`, W_g ∈ R^{d×n_h}.
- QK-norm: ALREADY in faithful Qwen3 baseline (no change).
- muP: **NOT specified in paper text → OMITTED** (purpose is cross-scale HP transfer; we train one fixed scale, so moot). Honest deviation.

## Training recipe (2602.02522) — VERIFIED (their 3-stage / 72B-token schedule)
- NorMuon 2D LR 0.011 (stable) / 0.0115 (decay) / 0.003 (mid); 1D LR 0.006/0.006/0.002
- WD 0.1 (2D only); warmup 2500; WSD decay fraction 20%; z-loss 1e-4; grad-accum 2; EMA β=0.8 (final 10 ckpts)
- Cautious WD (Eq7): `Δw = −λw if sign(u)=sign(w) else 0`, u = orthogonalized update.

## Adaptation to OUR budget (2 TPP, ~1.2B tok, ~18,150 steps, single run)
- Single WSD stage (not 3 stages): warmup ~5% + stable + 20% decay-to-zero.
- Keep NorMuon 2D LR ~0.011 / 1D ~0.006 (paper stable-stage), WD 0.1, z-loss 1e-4, β1=β2=0.95.
- seq_len 4096, micro_batch 4 × grad_accum 4 (our memory limit), bf16.
- EMA of weights over final checkpoints (optional; include if cheap).
- Smoke test first (~1000 steps) to validate the bundle trains + loss descends, THEN full run.

## Verify gate for Build 2
Unchanged components (vocab/RMSNorm/SwiGLU/GQA/embeddings) must still match faithful when the NEW components are disabled (value-residual α2=0, gating g→large so 2σ=1? no — disable gate, LN-scale=1). Build a "bundle-off == faithful" bit-exact test mirroring Build 3's verify.
