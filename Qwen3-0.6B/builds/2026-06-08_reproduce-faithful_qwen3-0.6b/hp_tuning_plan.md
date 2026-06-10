# HP Tuning Plan — Qwen3-0.6B, reproduce-faithful

- **Target:** `Qwen/Qwen3-0.6B-Base` (596M params)
- **Build mode:** reproduce-faithful
- **Recipe:** A — Faithful-scaled (cosine LR, AdamW)
- **Data:** FineWeb-Edu sample-10BT
- **Seq len:** 4096 (matches Qwen3 stage-1)
- **Hardware:** NVIDIA GB10, bf16
- **Date:** 2026-06-08

## Honest caveat on "faithful" HPs

**The Qwen3 technical report does NOT disclose exact pretraining HPs for the 0.6B base.** Quoted directly from a WebFetch of arXiv:2505.09388:

> "we develop scaling laws for optimal hyper-parameters (e.g., learning rate scheduler, and batch size) predictions based on three pre-training stages"

and:

> "we set the predicted optimal learning rate and batch size strategy for each dense or MoE model"

But no concrete per-size HP table is published. Various third-party summaries report a paper-baseline experiment around peak LR 1.7e-3 / warmup 1000 / 500B tokens / 4M-token batch, with cosine decay to 3.2e-4. **I am treating those as best-available signals, not verified ground truth.** Where there is no Qwen3-published value, I fall back to:

- Modern LLM convention (AdamW β=0.9, 0.95; eps 1e-8; wd 0.01; grad_clip 1.0)
- The existing `SmolLM2-134(base)/train.py` recipe in this codebase, for consistency

## HP surface

| HP | Default for Recipe A | Sensible range | Sensitivity | Source |
|---|---|---|---|---|
| Optimizer | AdamW | AdamW, Schedule-Free AdamW, Lion | Low (AdamW dominates) | LLM standard |
| β₁ | 0.9 | 0.9 fixed | Low | Standard |
| β₂ | **0.95** | 0.95–0.999 | Medium — lower β₂ adapts faster to non-stationary gradients (LLM-pretrain norm); 0.999 is GPT-3 default | β₂=0.95 is the modern LLM-pretrain default; SmolLM2 recipe uses it. One summary of Qwen3 quotes 0.999, but unverified vs paper. |
| eps | 1e-8 | 1e-8 fixed | Low | Standard |
| weight_decay | 0.01 | 0.01–0.1 | Medium | SmolLM2 uses 0.01; one Qwen3 summary quotes 0.1 (unverified). Sticking with 0.01 for consistency. |
| grad_clip | 1.0 | 0.5–1.0 | Low | Qwen team norm; SmolLM2 default |
| Peak LR | **1.7e-3** | 1e-3 to 4e-3 | **High** — top sensitivity | Qwen3 paper baseline (per third-party summary, unverified). SmolLM2 uses 3e-3. |
| End LR (cosine) | 3.2e-4 | 0 to peak/5 | Medium | Qwen3 paper baseline (unverified). |
| LR schedule | **cosine** | cosine / WSD / D2Z | Medium — picked per Recipe A | Qwen3 paper |
| Warmup steps | scaled to ~5% of total steps | 100–1000 | Medium — too short causes spikes, too long wastes compute at small budgets | Paper baseline: 1000 over 500B tokens; we scale proportionally |
| Sequence length | 4096 | 2048–4096 | Low for loss; medium for context coverage | User-selected; matches Qwen3 stage-1 |
| Effective batch (tokens) | 131K–524K | 64K–4M | Medium (interacts with LR) | Paper baseline: 4M. We are 8–32× below that. |
| Token budget | 0.5–2B | 100M–10B | High for final loss | Chinchilla optimal is ~12B for 596M params (20 tok/param); we're 6–24× under. |
| Dtype | bf16 | bf16, fp16 | Low (bf16 standard) | Modern LLM standard |
| Init scheme | Normal(0, 0.02) | locked by config | Low | HF config |
| Init seed | 0 | 0 fixed for repro | n/a | Conv |

## Candidate configs

All share: AdamW(β=0.9, 0.95), eps=1e-8, weight_decay=0.01, grad_clip=1.0, bf16, seq_len=4096, init Normal(0, 0.02), seed=0.

| Knob | Config A — Faithful, modest | Config B — Compute-adjusted (Recommended) | Config C — SmolLM2-anchor (comparison) |
|---|---|---|---|
| Schedule | cosine | cosine | cosine |
| Peak LR | **1.7e-3** | **1.7e-3** | **3.0e-3** |
| End LR | 3.2e-4 | 3.2e-4 | 3.2e-4 |
| Warmup steps | 200 | 400 | 200 |
| Token budget | **0.5B** | **1B** | 0.5B |
| Per-step batch (sequences) | 8 × accum 4 = 32 | 16 × accum 4 = 64 | 8 × accum 4 = 32 |
| Per-step batch (tokens) | 131,072 | 262,144 | 131,072 |
| Total optimizer steps | ~3,815 | ~3,815 | ~3,815 |
| Wall-clock estimate (GB10) | **TBD** — ask user for TPS estimate | **TBD** | **TBD** |

### Why three configs

- **Config A** matches the paper's LR shape exactly at a budget that's small enough to iterate quickly. Smallest cost — best for a "did we wire this all up right" first run.
- **Config B (recommended)** doubles the budget at the same recipe — gets closer to a meaningful PPL improvement vs random-init, still finishes within one workday. Bigger per-step batch reduces gradient noise.
- **Config C** swaps in SmolLM2's 3e-3 peak LR as a comparison anchor: does this codebase's existing recipe outperform the (unverified) Qwen3 paper LR on a same-budget run? Useful negative result.

## Throughput estimate (MEASURED 2026-06-08 — `results/throughput_probe.json`)

Measured on this GB10 with `throughput_probe.py` (50 steps after 5 warmup, bf16, seq_len=4096, chunked cross-entropy, process capped at 85% of the unified pool):

| | micro_batch | tokens/sec | sec/step | peak mem |
|---|---|---|---|---|
| baseline (no compile) | **4** | 3,788 | 4.33 | 68.2 GB |
| `torch.compile` (reduce-overhead) | **4** | 7,167 | 2.29 | 52.4 GB |

`torch.compile` speedup: **1.89×**.

**Critical memory finding:** `micro_batch=16` and `micro_batch=8` at seq_len=4096 **do not fit** — both raised a clean CUDA OOM and the probe backed off to **micro_batch=4** (the largest that fits, even on a freshly-booted box with the full ~109 GB cap). This is on the GB10 **unified memory** pool (~119 GB shared CPU+GPU; torch reports 128.5 GB total). **The plan's Config B "bs=16" is infeasible as a single micro-batch and must be realized as `micro_batch=4 × gradient_accumulation=4`.** (See the unified-memory guardrail in the `from-scratch-build` skill — over-allocation here hard-crashed the box on 2026-06-08; that is why the probe now caps memory and backs off.)

**Wall-clock at micro_batch=4, compiled (7,167 tok/s):**
- Config A (0.5B tokens): ~19.4 hr
- Config B (1B tokens): ~38.8 hr
- 2B-token 3-config sweep: **~77.5 hr** (~3.2 days); baseline/no-compile would be ~146.7 hr.

## Recommendation

**Config B — Compute-adjusted.**

- LR shape stays paper-anchored (1.7e-3 cosine to 3.2e-4)
- Token budget is 2× Config A — meaningfully better signal vs random-init
- Per-step batch is 2× — lower gradient noise, more stable training
- Warmup proportional to total steps (~10% of run)
- Still finishes within a single workday on a GB10 at any reasonable TPS

## Provenance & expected results (per config)

Each run is anchored to a specific source. **Critical:** the Qwen3 report does *not* publish the per-size 0.6B pretraining recipe (see "Honest caveat" above), so A/B are faithful to the paper's *baseline LR shape* only, and C plus the Chinchilla budget bump are deliberate, separately-cited deviations. No paper verifies that any of these three is *the* Qwen3-0.6B recipe — that recipe was never released.

| Run | LR / budget | Backing source | Verification status | What this run tests | Expected outcome |
|---|---|---|---|---|---|
| **A — Faithful, modest** | 1.7e-3 cosine, 0.5B tok | Qwen3 Technical Report, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) — scaling-law *baseline experiment* (peak 1.7e-3 → 3.2e-4, warmup 1000/500B) | ⚠️ Paper-anchored LR *shape*; per-size HP table NOT in paper | "Did we wire the faithful recipe up right?" — cheapest sanity run | Val PPL ~184K (random) → roughly **100–200** on FineWeb-Edu held-out; ~24× under Chinchilla → fluent but shallow; large gap to released Qwen3-0.6B-Base |
| **B — Compute-adjusted (Recommended)** | 1.7e-3 cosine, **1B tok, 2× batch** | Same Qwen3 TR LR shape **+** Chinchilla scaling law (Hoffmann et al., [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)) for the 2× token / 2× batch deviation | ⚠️ LR paper-anchored; budget bump is Chinchilla-justified, not Qwen3-published | Best *faithful* result achievable in ~one workday | **Lowest PPL of the three** (more tokens + lower gradient noise); still ~12× under Chinchilla → still a real gap to Qwen3-Base. This is the number we'd report as "our faithful repro." |
| **C — SmolLM2-anchor** | **3.0e-3** cosine, 0.5B tok | This repo's SmolLM2-135M repro recipe → SmolLM2 paper, [arXiv:2502.02737](https://arxiv.org/abs/2502.02737) | ✅ Known-good LR in *this* codebase (controlled comparison vs A at matched 0.5B) | Does the paper-anchored 1.7e-3 actually beat a known-good 3e-3 at equal compute? | Either **A ≤ C** (Qwen3 LR holds up → validates paper signal) or **C < A** (higher LR wins → useful negative result). Both are real findings. |

> Naming note: the earlier `training_plan.md` defined Recipe C as **WSD + decay-to-zero**, backed by Schedule-Free / D2Z ([arXiv:2507.09846](https://arxiv.org/pdf/2507.09846), 2026). That WSD recipe was deferred to the **modernized build (Build 2)**, not this faithful sweep. The operative C here is the SmolLM2-LR cosine anchor above.

### What the three runs together let us learn

1. **A vs B** — the *token-budget / batch-size* effect at a fixed (paper-anchored) LR. Isolates "more compute → how much lower PPL" on this exact model+data, quantifying the Chinchilla under-training penalty empirically.
2. **A vs C** — the *learning-rate* effect at matched 0.5B compute. Since Qwen3 never published the 0.6B LR, this is the **only genuinely novel, verifiable result of the sweep**: which LR is actually better here.
3. **All three** — the deliverable is the **relative ranking at matched compute**, plus a reusable, verified training harness. It is NOT a match to Qwen3-0.6B-Base's absolute quality (impossible at ≤1B tokens) — the success signal is PPL improvement vs random-init and a clean A/B/C ordering.

## Open questions

1. **Throughput measurement**: do you want me to run a 50-step throughput probe before Phase 8 (to lock the wall-clock estimate), or just estimate and proceed?
2. **`torch.compile`**: enable for the training run? Pays ~30s upfront compile cost for typically 1.3–1.6× steady-state speedup. SmolLM2's `train.py` keeps it as an `--compile` flag (default off).
3. **HP MODE**: single run (Config B), or mini-sweep of A vs B vs C? Sweep = 3× wall-clock; sweep result is a `sweep_summary.json` picking the best by held-out PPL.
