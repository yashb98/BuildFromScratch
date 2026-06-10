# Qwen3-0.6B — from-scratch reproduction + research experiment

A single-file PyTorch reproduction of [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
(596M-param decoder-only transformer), **verified bit-exact** against the official
HuggingFace weights, used as the base for a **three-build experiment**: reproduce it
faithfully, then apply recent (2026) research methods and measure — at matched
compute — whether they improve on the faithful baseline.

> **Status: Phase B running (as of 2026-06-10).** Architecture verified, learning-rate
> sweep complete, and the four matched-compute training runs are underway (~8 days on a
> single GB10). Numbers below are updated as runs finish.

## The model

Decoder-only transformer: 28 layers × 1024 hidden, GQA (16 query / 8 KV heads, head_dim
128), SwiGLU FFN (intermediate 3072), RMSNorm pre-norm, **QK-norm**, RoPE θ=1e6, tied
embeddings, vocab 151,936. The from-scratch [`model.py`](model.py) reproduces it to the bit:

```
verify.json:  max_abs_error = 0.0   argmax_match = true   ("The capital of France is" -> " Paris")
              params = 596,049,920 (exact)
```

That 0.0 logit error vs the official weights is the foundation of the whole project: it
proves the implementation *is* Qwen3-0.6B, so any later result is attributable to the
change we made, not a bug.

## The three builds

Each build lives under [`builds/`](builds/) with its own model, training script, verify
gate, and plan documents (every method is verified against its primary-source paper).

| Build | Folder | What changes | Backing paper(s) |
|---|---|---|---|
| **1. Faithful** | `…_reproduce-faithful_…` | Nothing — exact arch, AdamW + cosine. The baseline. | Qwen3 Tech Report ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)) |
| **2. Modernized** | `…_reproduce-modernized_…` | Full **IMU-1 bundle**: NorMuon optimizer + value residuals + LayerNorm-scaling + per-head gating + cautious weight decay + WSD schedule + z-loss | IMU-1 ([arXiv:2602.02522](https://arxiv.org/abs/2602.02522)), NorMuon ([arXiv:2510.05491](https://arxiv.org/abs/2510.05491)) |
| **3. Exploratory** | `…_reproduce-exploratory_…` | **Partial RoPE** — rotate only 25% / 10% of head dims, pass the rest through | "Fractional Rotation, Full Potential?" ([arXiv:2603.11611](https://arxiv.org/abs/2603.11611)) |

Every build's verify gate proves the *unchanged* components stay bit-identical to the
faithful model, so the comparison isolates exactly the one thing that changed.

## What's going on right now

The experiment runs in phases, **one GPU job at a time** (the GB10 shares one ~119 GB
CPU+GPU memory pool — two concurrent trainings overcommit and crash the box):

1. **✅ Verify gate** — bit-exact equivalence to HF (`max|Δlogits| = 0.0`).
2. **✅ Phase A — learning-rate sweep** (3 runs @ 131M tokens) to pick the LR:

   | Peak LR | final val PPL |
   |---|---|
   | 1.7e-3 | 46.89 |
   | **2.4e-3** | **46.31 ← best** |
   | 3.0e-3 | 49.28 (too hot) |

3. **🔄 Phase B — the real matched-compute runs** (4 runs @ **2 tokens/param** = 1.19B
   tokens each, LR 2.4e-3): faithful baseline → IMU-1 bundle → partial-RoPE 25% →
   partial-RoPE 10%. This is what produces the defensible result.

### Headline result so far — reproduction gap

Measured with identical eval code on the identical 300k-token FineWeb-Edu val slice:

```
ORIGINAL  Qwen3-0.6B-Base (36T tokens)        val PPL = 13.40
OURS      best from-scratch (131M tokens)     val PPL = 46.31   -> 3.5x gap
```

**We reproduced Qwen3-0.6B to within 3.5× perplexity of the original using ~275,000× less
data.** A buggy reimplementation would not land within a single-digit multiple of a
36T-token model — so the gap is *data scale*, not correctness. (A scaling check supports
this: 65.5M tokens → 96 PPL, 131M → 46, original 36T → 13.4.)

A **preliminary** Build-2 smoke run (full IMU-1 bundle, 65.5M tokens) reached **39.83
PPL — beating the faithful recipe's 46.31 that needed 131M tokens** (~2× sample
efficiency, matching the papers' claims). This is **not yet proven**: it's smoke-grade and
the full bundle confounds ~6 changes at once. Phase B is the controlled test.

## What we'll try / produce once Phase B is done

- **The matched-compute comparison table** — faithful baseline vs IMU-1 vs partial-RoPE
  25/10, all at 2 TPP, anchored to the original's 13.40. This finally *proves* (not hints)
  whether the research-driven changes beat the faithful baseline at equal compute.
- **`proof.ipynb`** — the bit-exact equivalence demonstration (load HF, load ours, show 0.0).
- **`results.ipynb`** — the full comparison + scaling-curve + per-build ablation, with an
  explicit limitations section (2 TPP is ~80× below the methods' validated regime → expect
  *directional* signal, and the IMU-1 bundle is intentionally confounded).
- **Candidate follow-ups** (depending on results): a NorMuon-only run to *deconfound* the
  bundle; a higher-TPP scaling-curve run to push the gap to the original further down; and
  applying the winning method as the new default recipe.

## Layout

```
Qwen3-0.6B/
├── model.py                 # the architecture, one file — verified bit-exact vs HF
├── verify.py                # parity gate
└── builds/
    ├── 2026-06-08_target-survey/                  # model-choice survey
    ├── 2026-06-08_reproduce-faithful_…/           # Build 1: baseline + LR sweep + eval
    ├── 2026-06-08_reproduce-modernized_…/         # Build 2: IMU-1 bundle (model_imu1, normuon)
    ├── 2026-06-08_reproduce-exploratory_…/        # Build 3: partial RoPE (model_partialrope)
    └── phase_b_driver.sh                           # runs the 4 matched-compute runs
```

> Checkpoints (`*.pt`, ~3.5 GB each) and token caches are intentionally **not** committed
> — regenerate them by running the training scripts.
