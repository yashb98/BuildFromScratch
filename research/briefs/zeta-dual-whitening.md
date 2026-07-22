# Brief: Zeta: Dual Whitening for Matrix Optimization via Coordinate-Adaptive Preconditioning (zeta-dual-whitening)
- researched: 2026-06-19 · by: ml-research · fetch_level: fulltext-arxiv-html
- paper_date: 2026-06-12 (cutoff_3m this run: 2026-03-19)
- modality: text · verdict: runnable-now
- objective: pretrain-ablation (§C13) · taste_score: 7.5 (§C15.2)

## Sources (all fetch-verified this run, §C3)
| url | what it is | accessed | replication status |
|---|---|---|---|
| https://export.arxiv.org/api/query?id_list=2606.14187 | arXiv metadata (date 2026-06-12, cs.LG, authors Chen et al.) | 2026-06-19 | primary |
| https://arxiv.org/html/2606.14187 | full paper (Algorithm 2, §4 config, §4.1–4.4 results, App. A.5/D) | 2026-06-19 | claim, unreplicated |
| https://github.com/AIGCodeOS/aigcode_zeta_optimizer | authors' reference code (cited in abstract; not fetched/run here) | 2026-06-19 | author code |
| https://kvfrans.com/matrix-whitening/ | "What really matters in matrix-whitening optimizers?" (adjacent critical analysis) | 2026-06-19 | independently discussed, not re-measured |
| https://arxiv.org/abs/2509.02046 | "Fantastic Pretraining Optimizers and Where to Find Them" (optimizer-eval rigor critique) | 2026-06-19 | background |

## What it changes
A drop-in **optimizer swap** (no architecture/data change). Zeta is a Muon-family
matrix optimizer that fixes a stated Muon weakness: Muon's Newton–Schulz (NS) step
assumes a well-conditioned input, but raw momentum matrices have severe
coordinate-wise scale heterogeneity (the paper verifies this with a chi-square
uniformity test). Zeta applies **dual whitening**: (1) *coordinate whitening* —
an AdamW-style per-entry second-moment normalization `G̃ = M/(√V+ε)` — then (2)
*spectral whitening* — `U = NewtonSchulz(G̃, K=5)`, with the standard Muon
update `ΔW = 0.2·√(mn)/(‖U‖_F+ε)·U`, decoupled weight decay. It is applied to **2D
hidden matrices** (attention + FFN projections); biases/LayerNorm/embeddings fall
back to AdamW — the **identical param-split shape as our verified NorMuon run**.

## Taxonomy (§C12 — axes touched)
architecture: decoder-only · training-stage: base (pretrain) · optimizer · modality: text · size-band: SLM (0.6B)

## Objective (§C13)
- type: pretrain-ablation (changes training from step 0 — the optimizer for 2D matrices)

## Exact recipe
From `arxiv.org/html/2606.14187` Algorithm 2 + §4 "Training Configuration":

| Hyperparameter | Value | Flag | Provenance |
|---|---|---|---|
| Zeta β₁ (momentum) | 0.95 | reported | §4 |
| Zeta β₂ (2nd moment) | 0.99 | reported | §4 |
| Newton–Schulz iters K | 5 | reported | §4 |
| NS coeffs (a,b,c) | 3.4445, −4.7750, 2.0315 | reported | "standard values, cited" |
| RMS scale | 0.2·√(mn)/‖U‖_F | reported | §4 (standard Muon scaling) |
| weight decay (decoupled) | 0.1 | reported | §4 |
| ε | not reported | inferred → 1e-8 | Alg.2 shows ε, no value |
| LR schedule | cosine, 1% warmup | reported | §4 |
| peak LR (Qwen3-0.6B) | 9e-4 | reported | §4 per-model table |
| param scope | 2D matrices → Zeta; biases/LN/embeds → AdamW | reported | §4 dual-path |
| paper batch / seq (0.6B) | 256 × 4096 = 1.05M tok/step | reported | §4 |
| paper budget (0.6B) | ~20.9B tok (20k iters) ≈ 35 TPP | reported | §4 |
| custom CUDA kernel | none (standard PyTorch matmul/elementwise) | reported | "not specified"; we reimplement |

LR-tuning protocol (paper): each optimizer's LR is **tuned individually on
Qwen3-0.6B**, then unified for larger models (§4) — a clean, non-confounded control.

## Recommended budget (scaled)
- paper budget: ~20.9B tok (35 TPP) on Qwen3-0.6B — far above our regime.
- scaling reasoning (inferred): match the **verified NorMuon-vs-AdamW ablation budget** so the two optimizer results are directly comparable on the SAME baseline/metric → Zeta's lift can be ranked against NorMuon's measured **+0.4743 bpb** win.
- **TOKEN_BUDGET — Qwen3-0.6B: 6-cell cohort (2 arms × 3 seeds), 640 steps × 65,536 tok = 41.9M tok/cell, 251.7M total** (source §C5.2: parity with run `2026-06-16_qwen3_normuon-vs-adamw`). Optional confirmation at 2000 steps if the 640-step signal is borderline. (SmolLM2-134M: not the target here — Qwen3 is the paper's tested model.)

## Framework / runtime fit (§C14)
- recommendation: **pytorch** — the baseline (faithful Qwen3-0.6B) is PyTorch and the verify gate is PyTorch-only (§C14a); Zeta is a drop-in optimizer reimplemented in-repo as a single `.py` (exactly as `normuon.py` was), no port.
- portability / kernel flags: none. NS iteration = matmuls + element-wise ops → no custom CUDA, no x86-only dep, **aarch64-safe by construction** (we implement from Algorithm 2; the authors' code is reference only). A fused NS/whitening Triton kernel is a *future* optimization opportunity, not needed to run.

## Baseline + win condition
- baseline checkpoint / recipe: `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/checkpoint_qwen3_baseline2tpp.pt` (596,049,920 params, single AdamW @ tuned LR; provenance: faithful build README + checkpoint inventory read this run). The A/B trains BOTH arms from random init at matched budget (Type-STEP0): **AdamW baseline vs Zeta(2D)+AdamW(1D)**.
- win = across-seed (3-seed) wikitext-2 **BPB** (text-lm-v2, §C10) for Zeta significantly **lower** than the AdamW baseline — 95% CI excludes 0 (`eval_stats.seed_delta_significant`) and clears the noise floor; code-PPL reported as the OOD check. **Bonus comparison:** rank Zeta's lift against NorMuon's measured +0.4743 bpb [0.4435, 0.5052] (same budget/baseline → "does this newer Muon-variant beat the one we verified?").

## Research-taste verdict (§C15.2)
- taste_score: 7.5 · axes: mechanism **strong** (whitening-conditions-NS is principled, chi-square-verified) · evidence **good but single-org** (clean per-optimizer LR control + β-grid robustness across 0.6B/1.7B/8B/MoE + downstream, no independent repro yet) · reproduction **none found as of 2026-06-19** · scaling-to-our-scale **strong** (paper tests OUR exact model Qwen3-0.6B) · ROI **high** (drop-in, ~4% wall-clock overhead, AdamW-equal memory, same cheap shape as the NorMuon win) · simplicity/blast-radius **low** (one optimizer, one .py) · safety **high** (optimizer-only, reversible).
- Weighting mechanism + evidence + ROI over buzz: this is a principled, cheap, single-variable optimizer ablation on the project's exact model, directly comparable to an already-verified win — the strongest "another one" in the candidate pool. The one real discount is zero independent reproduction (it's 7 days old), mitigated by the cheap first budget and the paper's clean controls. Below the NorMuon precedent only because that one is already verified in-repo; as a *new* pick it is the highest-ROI optimizer candidate.

## Reproductions & criticism
- **No independent reproduction of Zeta found as of 2026-06-19** (paper is 7 days old). Queries run: "Zeta dual whitening optimizer Muon variant reproduction results"; arXiv/HN surfaced only adjacent Muon-variant work (MuonAll 2511.06086, Muon-in-ViT 2605.24770, Muon convergence 2509.15816), not Zeta itself.
- Adjacent critical context (cite, don't over-weight): kvfrans "What really matters in matrix-whitening optimizers?" (questions which whitening components actually matter — relevant since Zeta's gain is the *coordinate*-whitening add-on) and "Fantastic Pretraining Optimizers and Where to Find Them" (2509.02046, warns optimizer wins often shrink under honest matched-LR eval). Both raise the bar for our own controlled A/B.

## Failure modes & abort criteria
- **LR-coupling confound** (Zeta's RMS-scaled LR is on a different scale than AdamW's; the verdict must not be an LR artifact). Mitigation: use AdamW-best 2.4e-3 (from the verified sweep) for the baseline and the paper's reported 9e-4 for Zeta; if the result is borderline, a 3-point Zeta LR mini-sweep is required before claiming a win.
- **NS instability** if coordinate whitening fails to isotropize → NaN/Inf in the optimizer step ⇒ **instant abort**.
- loss > AdamW-baseline-at-equal-tokens by >10% for 2 consecutive logged evals ⇒ abort.
- grad-norm > 10 or NaN/Inf at any step ⇒ instant abort.
- tokens/sec degraded > 15% vs the run's own §C5.3 probe (paper claims ~4% wall-clock overhead; a larger hit means the in-repo NS/whitening impl is inefficient) ⇒ flag, not necessarily abort.
- no eval signal after 25% of TOKEN_BUDGET when the paper predicts an early speedup ⇒ likely `inconclusive` at this budget.

## GB10 feasibility (§C1)
- Memory (≤60% of ~119 GB pool): params 596M×2B (bf16) ≈ 1.2 GB; grads ≈ 1.2 GB; optimizer state **O(2mn)** for 2D matrices (M+V, fp32) + AdamW(2 states) for the rest ≈ AdamW-equal (~5 GB); activations + **chunked CE** over 151,936 vocab (REQUIRED, §C1 — already in `train_ablation.py`). The faithful 2-TPP run measured **52.4 GB peak** at this exact config; Zeta's O(2mn) state ≈ AdamW → same envelope, comfortably under 60%.
- modality fit: text decoder-only → the faithful Qwen3-0.6B checkpoint is the baseline. ✓
- aarch64 deps: **none** — pure PyTorch, reimplemented in-repo from Algorithm 2 (the `normuon.py` pattern); no pip wheel, no x86 CUDA kernel.
- The §C5.3 measured probe (run by ablation-runner) is the launch authority; this is paper-math only.

## Verdict
**runnable-now** — recipe complete, baseline checkpoint + FineWeb-Edu data already on the box (same pipeline as the de-confound cohort), memory fits the measured 52 GB envelope, pure-PyTorch/aarch64-safe. Queue it; launch the instant the GPU frees from the IMU-1 de-confound cohort (one-trainer-at-a-time, §C4.5).
