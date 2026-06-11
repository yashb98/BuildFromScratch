# Qwen3-0.6B — from-scratch reproduction + research experiment

A single-file PyTorch reproduction of [`Qwen/Qwen3-0.6B-Base`][hfbase] (596M-param
decoder-only transformer), **verified bit-exact** against the official HuggingFace
weights (`max |Δlogits| = 0.0`), used as the base for a **three-build experiment**:
reproduce it faithfully, then apply recent (2026) research methods and measure — at
matched compute — whether they beat the faithful baseline.

> **Status: Phase B running (run 2 of 4).** Architecture VERIFIED (bit-exact);
> Phase A LR sweep complete (`lr24 = 2.4e-3` won). Phase B = four matched-compute
> runs @ 2 TPP (1.19B tokens each). **Run 1 (faithful baseline) is DONE: val PPL
> 28.65**, which narrows the gap to the original from 3.5× (@131M) to **2.14×
> (@1.19B)**. Run 2 (IMU-1 bundle) is training; partial-RoPE runs follow. The
> "does a research method beat the baseline" finding is still **PENDING**.

> **This is an index.** Each build has its own detailed README — see
> [the three builds](#the-three-builds) for links. The architecture itself is the
> single, fully-commented [`model.py`](model.py) (every choice cited inline).

[hfbase]: https://huggingface.co/Qwen/Qwen3-0.6B-Base
[qwen3paper]: https://arxiv.org/abs/2505.09388

---

## Results so far

All perplexities use **identical eval code on the identical 300k-token FineWeb-Edu
val slice** ([`eval_original_vs_repro.py`](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/eval_original_vs_repro.py)),
so every row is directly comparable.

**Bit-exact reproduction** — `verify.json`: `max_abs_error = 0.0`, argmax `" Paris"`,
params **596,049,920**. Our `model.py` *is* Qwen3-0.6B.

**The reproduction gap (the headline):**

| Model | Training tokens | val PPL | Gap vs original |
|---|---|---|---|
| **Original** `Qwen3-0.6B-Base` | 36T | **13.40** | 1.0× |
| Our best (Phase A, `lr24`) | 131M | 46.31 | 3.5× |
| **Our faithful (Phase B baseline)** | **1.19B** | **28.65** | **2.14×** |

We reproduced Qwen3-0.6B to within **2.14× perplexity using ~275,000× less data** —
and each ~10× more data roughly halves the gap, confirming the gap is *data scale,
not correctness*.

**Scaling trend:** `65.5M → 131M → 1.19B → 36T  ≈  96 → 46 → 28.65 → 13.4`.

**Phase A LR sweep** (131M tokens, matched compute) → picked the LR:
`lr17` (1.7e-3) = 46.89 · **`lr24` (2.4e-3) = 46.31 ← best** · `lr30` (3.0e-3) = 49.28.

**Build-2 IMU-1 smoke** (⚠️ preliminary + confounded): 39.83 PPL @ **65.5M tokens** —
better than the faithful *smoke* (95.87 at the same 65.5M), a directional hint of
sample efficiency. **Do not compare it to the 28.65 baseline above** — that's 1.19B
tokens (18× more data). The real same-budget verdict (IMU-1 @ 2 TPP vs faithful 28.65)
is **PENDING**; the controlled test is Phase B.

> The matched-compute **IMU-1 / partial-RoPE vs the 28.65 baseline** comparison is
> still **PENDING** (runs 2–4 in progress).

---

## The three builds

Each is a self-contained folder with its own README, model/scripts, verify gate,
and results. Click through for the detail.

| Build | Folder (→ README) | What changes | Backing paper | Status |
|---|---|---|---|---|
| **1 · Faithful** | [`builds/…reproduce-faithful…`](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/README.md) | nothing — exact arch, AdamW + cosine (the baseline + shared harness) | [Qwen3 TR][qwen3paper] | ✅ baseline = **28.65** |
| **2 · Modernized** | [`builds/…reproduce-modernized…`](builds/2026-06-08_reproduce-modernized_qwen3-0.6b/README.md) | full **IMU-1 bundle**: NorMuon + value residuals + LayerNorm-scaling + per-head gating + cautious-WD + WSD + z-loss | [IMU-1](https://arxiv.org/abs/2602.02522), [NorMuon](https://arxiv.org/abs/2510.05491) | 🔄 Phase B run 2 |
| **3 · Exploratory** | [`builds/…reproduce-exploratory…`](builds/2026-06-08_reproduce-exploratory_qwen3-0.6b/README.md) | **partial RoPE** — rotate 25% / 10% of head dims, pass the rest through | [arXiv:2603.11611](https://arxiv.org/abs/2603.11611) | ⏳ queued |
| — · Survey | [`builds/…target-survey…`](builds/2026-06-08_target-survey/README.md) | why Qwen3-0.6B was chosen (model-selection phase, not a build) | — | ✅ done |

Each build's verify gate proves the *unchanged* components stay bit-identical to the
faithful model, so any result is attributable to the one thing that changed.

---

## Architecture at a glance

Full spec is in [`model.py`](model.py) (one file, every value cited to `config.json`).

| Field | Value | | Field | Value |
|---|---|---|---|---|
| Layers | 28 | | RoPE θ | **1e6** (not 1e4) |
| Hidden | 1024 | | RMSNorm eps | 1e-6 |
| Heads (Q/KV) | **16 / 8** (GQA 2:1) | | Vocab | 151,936 |
| head_dim | **128** (independent field) | | Tied embeddings | yes |
| FFN (SwiGLU) | 3072 | | Params | **596,049,920** |

The three things that differ from Llama/SmolLM2: **per-head QK-Norm** (RMSNorm on Q,K
*before* RoPE), `head_dim` is an **independent** config field (not hidden/n_heads), and
**RoPE θ = 1e6**. See the faithful build README and `model.py` comments for the rest.

---

## Repo layout

```
Qwen3-0.6B/
├── model.py                 # the architecture, one file — verified bit-exact vs HF
├── verify.py                # parity gate
├── README.md                # this index
└── builds/
    ├── 2026-06-08_target-survey/                  # model-choice survey      (README)
    ├── 2026-06-08_reproduce-faithful_…/           # Build 1 + shared harness (README)
    ├── 2026-06-08_reproduce-modernized_…/         # Build 2: IMU-1 bundle    (README)
    ├── 2026-06-08_reproduce-exploratory_…/        # Build 3: partial RoPE    (README)
    └── phase_b_driver.sh                           # runs the 4 matched-compute runs
```

Checkpoints (`*.pt`, ~3.5 GB each) and token caches are **gitignored** — regenerate
them with the training scripts.

---

## Setup

```bash
pip install torch transformers datasets safetensors accelerate
python verify.py        # parity gate — runs on CPU, no GPU needed
```

Training needs a CUDA GPU. On the GB10 unified-memory box, only **one training job at
a time** (CPU+GPU share one ~119 GB pool — two concurrent runs overcommit and crash
the machine); the scripts import [`safe_cuda`](../safe_cuda.py) to cap the process.

---

## Honest accounting (short)

- ✅ **Architecture** — verified bit-exact vs HF (`max|Δlogits| = 0.0`).
- ✅ **Reproduction gap is data, not skill** — 2.14× PPL gap against ~275,000× less
  data, with a clean scaling curve.
- ✅ **Phase A LR (2.4e-3)** — an original verified finding (Qwen3 never published the
  0.6B LR).
- 🔶 **Phase B** — baseline done (28.65); IMU-1 / partial-RoPE *vs* baseline **pending**.
- ⚠️ **2 TPP is ~80× below** the methods' validated regime → Phase B results will be
  **directional, not headline**; the IMU-1 bundle intentionally **confounds** ~6 changes.
- ⚠️ **Honest gaps** — muP omitted; NorMuon NS5 coeffs are the standard Muon values
  (not printed in the paper). See the modernized build README.

---

## Source map

| Topic | Source |
|---|---|
| Qwen3 architecture / recipe | [arXiv:2505.09388][qwen3paper] · live `config.json` |
| NorMuon optimizer | [arXiv:2510.05491](https://arxiv.org/abs/2510.05491) |
| IMU-1 bundle | [arXiv:2602.02522](https://arxiv.org/abs/2602.02522) |
| Partial RoPE | [arXiv:2603.11611](https://arxiv.org/abs/2603.11611) |
| RoPE · GQA · SwiGLU · RMSNorm | [2104.09864](https://arxiv.org/abs/2104.09864) · [2305.13245](https://arxiv.org/abs/2305.13245) · [2002.05202](https://arxiv.org/abs/2002.05202) · [1910.07467](https://arxiv.org/abs/1910.07467) |
