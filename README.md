# BuildFromScratch

From-scratch language-model reproductions and the tooling around them. Two
reproduction subprojects live here:

| Path | What it is |
|---|---|
| [`SmolLM2-134(base)/`](SmolLM2-134(base)/) | Single-file PyTorch reproduction of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), verified **bit-exact** against the official HuggingFace weights (`max |Δlogits| = 0.0`). Includes from-scratch training, continued pretraining on TinyStories, multi-axis parity diagnostics, in-domain vs OOD eval, and an `lm-evaluation-harness` wrapper. |
| [`Qwen3-0.6B/`](Qwen3-0.6B/) | Single-file PyTorch reproduction of [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), verified **bit-exact** (`max |Δlogits| = 0.0`), plus a **three-build research experiment** (faithful baseline / modernized *IMU-1* bundle / exploratory *partial-RoPE*) that applies recent 2026 papers and measures them at matched compute. Reproduced the faithful baseline to within **2.14×** of the original (using ~275,000× less data); the modernized **IMU-1 bundle then beat that baseline by 17.9% at matched compute** (23.52 vs 28.65), while exploratory partial-RoPE lost (0.25 = 29.54; the 0.10 variant died incomplete). A two-phase single-variable, 3-seed, iso-FLOP de-confound then **attributed that win** to NorMuon + the IMU-1 architecture modules (value-residual / layernorm-scaling / head-gating, all individually significant on canonical BPB), with schedule and z-loss **not** significant. See [its README](Qwen3-0.6B/README.md). |

> This repo is driven by local Claude Code skills (`from-scratch-build`,
> `finance-research-loop`) and a FastAPI agent-harness showcase; those are kept
> local-only and are **not** committed.

## Quickstart — the SmolLM2 reproduction

```bash
cd "SmolLM2-134(base)"

# Install pinned dependencies that produced the 0.0 logit-diff result.
pip install -e .                # or: pip install -r requirements.txt

# Architecture parity gate (the non-negotiable test before training).
pytest tests/ -v
# or, the script form:
python verify.py

# Sample from the official weights via our class.
python generate.py "Once upon a time"

# A toy training run (random init, wikitext-103 demo) — proves the loop works.
python train.py --steps 100

# Continued pretraining on TinyStories from official weights.
python train_tinystories.py --token_budget 10_000_000     # ~10M tokens for a quick run

# Resume a run that died:
python train_tinystories.py --resume checkpoint_tinystories.pt --token_budget 100_000_000

# Standardized benchmarks (lm-evaluation-harness wrapper):
pip install lm-eval
bash scripts/run_lm_eval.sh                          # base only
bash scripts/run_lm_eval.sh checkpoint_tinystories.pt # base + trained
```

See [`SmolLM2-134(base)/README.md`](SmolLM2-134(base)/README.md) for the full
architectural walkthrough, design-decision narrative, and reproduction recipe.

## Repository layout

```
BuildFromScratch/
├── README.md                      # this file
├── .gitignore
├── SmolLM2-134(base)/             # the SmolLM2-135M reproduction (see its README)
│   ├── model_full.py              # the architecture, one file
│   ├── verify.py                  # parity gate vs HF reference
│   ├── compare_with_hf.py         # 6-axis diagnostic suite
│   ├── train.py                   # from-scratch training (random init)
│   ├── train_tinystories.py       # continued pretraining on TinyStories
│   ├── eval_after_vs_base.py      # in-domain + OOD ppl comparison
│   ├── benchmark_training.py      # tok/s + peak memory measurement
│   ├── generate.py                # CLI sampler
│   ├── tests/test_parity.py       # pytest gate
│   ├── scripts/                   # lm-eval wrapper + checkpoint export
│   ├── results/                   # generated artifacts (JSON/CSV/PNG/MD)
│   ├── proof.ipynb                # pedagogical notebook
│   ├── results.ipynb              # end-to-end results notebook
│   ├── pyproject.toml             # pinned deps
│   └── README.md                  # the long-form architecture/recipe doc
├── Qwen3-0.6B/                    # the Qwen3-0.6B reproduction + 3-build experiment (see its README)
│   ├── model.py                  # the architecture, one file — verified bit-exact vs HF
│   ├── verify.py                 # parity gate
│   └── builds/                   # faithful / modernized (IMU-1) / exploratory (partial-RoPE)
├── safe_cuda.py                   # GB10 unified-memory guard (caps the CUDA process)
└── jax_safe_env.py                # JAX preallocation guard for the shared-memory box
```

