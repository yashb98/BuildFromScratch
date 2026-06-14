# Build 1 — Faithful Qwen3-0.6B (from scratch) + the shared experimental harness

**One line:** a bit-exact from-scratch reproduction of [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (596M params), trained from random init on FineWeb-Edu — and the trainer / sweep / eval / plotting tooling that every later build in this project reuses.

This folder is **two things at once**:

1. **Build 1 (Faithful)** — the reference build. Recipe A "faithful-scaled": AdamW + cosine LR anchored to the Qwen3 Technical Report's baseline LR shape, on a drastically reduced public-data budget.
2. **The shared harness** — `train_qwen3.py`, the verify/test scripts, the throughput probe, the Phase-A LR sweep, the original-vs-repro eval, and the plotting/notebook builders. Builds 2+ (modernized / exploratory) live in sibling folders but import or copy this trainer and reuse this eval code so their numbers are directly comparable.

> **The model implementation itself (`model.py`) is NOT in this folder.** It lives one level up at [`../../model.py`](../../model.py), shared by all builds. For the architecture (GQA, QK-Norm, RoPE θ=1e6, SwiGLU, tied embeddings, the 596M param breakdown) and the bit-exact verify story, read the **[parent README `../../README.md`](../../README.md)**. This README stays folder-scoped: what is *in here*, how to run it, and the results this build produced.

---

## Method (and the paper behind it)

- **Target:** `Qwen/Qwen3-0.6B-Base`, the 596M-param dense decoder.
- **Backing paper:** [Qwen3 Technical Report, arXiv:2505.09388](https://arxiv.org/abs/2505.09388) (May 2025). Architecture is faithful to the shipped `config.json`; the *training recipe* mirrors the report's scaling-law **baseline experiment** (cosine LR, peak `1.7e-3` → end `3.2e-4`, warmup 1000 over 500B tokens, 4M-token batch).
- **Honest caveat (from `hp_tuning_plan.md`):** the Qwen3 report does **not** publish the per-size 0.6B pretraining HP table. The `1.7e-3 / 1000 / 500B` figures are the paper's *baseline-experiment* anchor (via third-party summaries), treated as best-available signal — **not** verified ground truth for the 0.6B-Base production run. Where no Qwen3 value exists we fall back to modern-LLM convention (AdamW β=0.9/0.95, eps 1e-8, wd 0.01, grad-clip 1.0) and this repo's SmolLM2 recipe.
- **Data proxy:** FineWeb-Edu `sample-10BT` ([HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)), streamed and tokenized on the fly — the closest public analogue of Qwen3's knowledge-dense stage-1 mix that fits a single-GPU budget.
- **The unavoidable deviation = the token budget.** The paper used ~36T tokens; we use 131M (Phase A) to 1.19B (Phase B). So the success signal is **PPL improvement vs random-init** and a **clean A/B/C ranking at matched compute** — *not* matching released Qwen3-0.6B-Base's absolute quality (impossible at ≤1.2B tokens). Chinchilla-optimal for 596M is ~12B tokens (20 tok/param); even Phase B is ~10× under-trained.

Shared fixed knobs across every run here: **AdamW(β=0.9, 0.95), eps=1e-8, weight_decay=0.01, grad_clip=1.0, bf16, seq_len=4096, init Normal(0, 0.02), seed=0**, cosine schedule, end LR `3.2e-4`.

---

## Hardware guardrails (read before running anything)

This is a **NVIDIA GB10 unified-memory box**: CPU and GPU share one ~119 GB pool (torch reports ~128.5 GB total), with no separate VRAM. Over-allocation crashes the whole machine, not just the process. Two mitigations are baked into every script here:

- **`safe_cuda.guard(0.85)`** (imported from [`../../../safe_cuda.py`](../../../safe_cuda.py), the `BuildFromScratch/` root) caps the process at 85% (~109 GB) so an over-allocation raises a clean `torch.OutOfMemoryError` instead of a hard crash. Every entrypoint imports `safe_cuda` *before* torch touches CUDA.
- **Chunked cross-entropy** (`chunked_cross_entropy`, in `train_qwen3.py` and `throughput_probe.py`): the naive `logits.float()` at vocab=151,936 and the training shape is a ~40 GB fp32 tensor — that exact allocation hard-crashed the box on 2026-06-08, which is why these files exist. CE is computed in chunks of 8192 rows.
- **`micro_batch=4` is a hard ceiling at seq_len=4096.** The throughput probe proved `micro_batch ∈ {8, 16}` cleanly OOM even on a freshly-booted box; effective batch is realized as `micro_batch=4 × grad_accum=4` (16 seqs / 65,536 tokens per step). **Do not raise `--micro_batch`.**

---

## What's in this folder

### Plan docs (the decision trail)

| File | What it records |
|---|---|
| `architecture_plan.md` | Component-by-component faithful spec vs the HF `config.json` + `modeling_qwen3.py`; the param-count forward-calc (596,049,920); the QK-Norm / `head_dim=128` "this is the #1 difference vs Llama" callout; the verify-gate definition. |
| `training_plan.md` | What Qwen3-0.6B-Base was actually trained on (36T tokens, 3-stage); data-candidate table (FineWeb-Edu chosen); recipe-candidate table (A/B/C); Chinchilla sanity check. |
| `hp_tuning_plan.md` | The "faithful HPs are not published" caveat; HP-surface sensitivity table; candidate configs A/B/C; the **measured** throughput probe results; per-run provenance + expected-outcome table. |

### Code — the shared harness

| File | What it does | How to run |
|---|---|---|
| `train_qwen3.py` | The trainer. Random-init `Qwen3ForCausalLM`, streams+caches FineWeb-Edu, cosine LR, chunked CE, `torch.compile` (default mode), per-step CSV logging, mid-train eval, checkpoint + resume (model/optim/sched/RNG/step), end-of-run eval + sample generations. Writes `results/qwen3_<tag>_{train.log,train.csv,after.txt}` and `checkpoint_qwen3[_<tag>].pt`. | `python train_qwen3.py` (1000-step smoke) · see commands below |
| `test_model.py` | pytest mechanics suite on a *shrunk* config (shapes, backward grads, state-dict roundtrip, no-NaN generate) **plus** one full-config test asserting the param count is within 1% of 596,049,920, and a regression guard that `q_norm`/`k_norm` weights exist. | `pytest -v test_model.py` |
| `verify_run.py` | The Phase-5 bit-exact gate. Loads official Qwen3-0.6B-Base safetensors into our model (via `load_official_weights_into_ours` from `../../verify.py`), runs one prompt through both, computes `max\|Δlogits\|`, writes `results/verify.json`, asserts `< 1e-3` + argmax match. | `python verify_run.py` |
| `throughput_probe.py` | Measures steady-state tok/s + peak mem at the training shape, compile OFF then ON, backing off `micro_batch` on OOM. Drives the cost gate. Writes `results/throughput_probe.json`. | `python throughput_probe.py` |
| `run_lr_sweep.sh` | Phase-A matched-compute LR sweep: three 2000-step (~131M-token) runs, identical except peak LR — `lr17`=1.7e-3, `lr24`=2.4e-3, `lr30`=3.0e-3. | `./run_lr_sweep.sh` |
| `eval_original_vs_repro.py` | Evaluates the **published** Qwen3-0.6B-Base and our sweep checkpoints (`lr17/lr24/lr30`) with **identical eval code on the identical val slice** (from the shared tokcache) → the true reproduction gap. Writes `results/original_vs_repro.txt`. | `python eval_original_vs_repro.py` |
| `wait_then_eval_original.sh` | Polls until the sweep's `train_qwen3.py` processes exit, then launches the original-vs-repro eval — keeps the GB10 at one GPU job at a time. | `./wait_then_eval_original.sh` |
| `make_plots.py` | Reads a run's CSV+log → PNGs (loss/LR, LR schedule, grad-norm, peak-mem-vs-cap, val-PPL, combined dashboard) under `results/plots[_<run_name>]/`. | `python make_plots.py [--run_name <tag>]` |
| `_build_results_notebook.py` | Regenerates `results.ipynb` (live-computed from `results/`). Re-run after training to refresh. | `python _build_results_notebook.py` |
| `results.ipynb` | Executable results notebook for the smoke run. |

### `results/` — artifacts (every number in this README traces to one of these)

- `verify.json` — the bit-exact gate result.
- `throughput_probe.json` / `throughput_probe.log` — measured tok/s + peak mem.
- `original_vs_repro.txt` (+ `original_eval_run2.log`, `original_eval_wrapper.log`) — the published-vs-ours PPL comparison.
- Per-run **`qwen3_<tag>_{after.txt, train.csv, train.log}`** for: `lr17`, `lr24`, `lr30` (Phase A); `baseline2tpp` (Phase B run 1); plus `configA`, `smoke`/untagged (smoke runs).
- `sweep_driver.log`, `phase_b_driver.log` — the orchestration logs.
- `plots/` (smoke-run PNGs: dashboard, loss_curve, lr_schedule, grad_norm, peak_mem, val_ppl) and `plots_smoke/`.
- `tokcache_*.pt` — cached tokenized FineWeb-Edu train+val buffers (re-runs skip the ~26-min stream+tokenize). The `tokcache_133072000_300000.pt` val slice is what `eval_original_vs_repro.py` uses for an apples-to-apples comparison.
- `checkpoint_qwen3*.pt` live in **this build dir** (not in `results/`): `_lr17`, `_lr24`, `_lr30`, `_baseline2tpp`, and the smoke `checkpoint_qwen3.pt`.

---

## How to run

```bash
# 0) bit-exact architecture gate (downloads ~1.2 GB on first run)
pytest -v test_model.py        # mechanics + param-count + QK-Norm present
python verify_run.py           # writes results/verify.json, asserts max|Δ| < 1e-3

# 1) measure throughput / lock the cost estimate
python throughput_probe.py     # writes results/throughput_probe.json

# 2) smoke train (1000 steps, compiled) — proves the stack trains, no OOM
python train_qwen3.py

# 3) Phase A — matched-compute LR sweep (3 × ~131M-token runs)
./run_lr_sweep.sh
./wait_then_eval_original.sh    # original-vs-repro PPL once the sweep exits

# 4) plots / notebook for any run
python make_plots.py --run_name lr24
python _build_results_notebook.py
```

A Phase-B-style longer run is the same trainer with more steps (Phase B run 1 used
`--steps 18150 --warmup_steps 900 --peak_lr 2.4e-3 --run_name baseline2tpp`, the
winning LR from Phase A scaled to a 1.19B-token / 2-tokens-per-param budget).
`--resume <ckpt>` restores model/optim/sched/RNG/step for a continuous curve.

---

## Results

### 1) Bit-exact verify — **VERIFIED** (`results/verify.json`)

Loaded official Qwen3-0.6B-Base weights into our model, fp32, prompt `"The capital of France is"`:

| Metric | Value |
|---|---|
| `max_abs_error` | **0.0** |
| `relative_error` | 0.0 |
| HF next token | `' Paris'` (id 12095) |
| Our next token | `' Paris'` (id 12095) |
| argmax match | true |
| **passed** | **true** (tolerance `1e-3`) |

Not just under tolerance — **identical to the bit** over the full logit vector. The architecture is correct.

### 2) Throughput probe — **MEASURED** (`results/throughput_probe.json`)

GB10, bf16, seq_len=4096, `micro_batch=4` (the largest that fit; 8 and 16 OOM'd and the probe backed off):

| Pass | tokens/sec | sec/step | peak mem |
|---|---:|---:|---:|
| baseline (no compile) | 3,787.5 | 4.326 | 68.21 GB |
| `torch.compile` | 7,167.4 | 2.286 | 52.40 GB |

`torch.compile` speedup: **1.89×**. (Note: the *probe* uses `mode="reduce-overhead"`; the *trainer* uses default mode because reduce-overhead's CUDA-graph static buffer breaks under gradient accumulation — see the comment in `train_qwen3.py`.)

### 3) Phase A — matched-compute LR sweep — **VERIFIED** (per-run `qwen3_lr*_after.txt`)

Three runs, identical except peak LR; 2000 steps ≈ 131,072,000 tokens each; final FineWeb-Edu held-out PPL:

| Run | Peak LR | Final val PPL | Notes |
|---|---|---:|---|
| `lr17` | 1.7e-3 (Qwen3 paper anchor) | 46.89 | |
| **`lr24`** | **2.4e-3 (midpoint)** | **46.31** | **BEST** |
| `lr30` | 3.0e-3 (SmolLM2 anchor) | 49.28 | |

All start from random-init baseline PPL ≈ 183,922. **Finding:** the midpoint `2.4e-3` edged out both the paper-anchored `1.7e-3` and SmolLM2's `3.0e-3` at matched compute — the one genuinely novel, verifiable result of the sweep, since Qwen3 never published the 0.6B LR. `2.4e-3` was carried forward to Phase B.

### 4) Original vs reproduction — **VERIFIED** (`results/original_vs_repro.txt`)

Identical eval code, identical 300K-token val slice (50 windows × 4096):

| Model | Trained on | val PPL | Gap vs original |
|---|---|---:|---:|
| **ORIGINAL** Qwen3-0.6B-Base | 36T tokens | **13.40** | 1.0× |
| REPRO `lr17` | 131M tokens, from scratch | 46.89 | 3.5× |
| **REPRO `lr24` (best)** | 131M tokens, from scratch | **46.31** | **3.5×** |
| REPRO `lr30` | 131M tokens, from scratch | 49.28 | 3.7× |

Best repro `lr24` = **46.31 vs original 13.40 → 3.5× higher PPL** — expected, given ~275,000× less data (36T vs 131M).

### 5) Phase B baseline final — **VERIFIED / build-1 final** (`results/qwen3_baseline2tpp_after.txt`)

The faithful baseline re-trained at the Phase-A-winning LR (`2.4e-3`) on a **1,189,478,400-token (~2 tokens/param)** budget — 18,150 steps, warmup 900:

| | Value |
|---|---|
| Token budget | 1,189,478,400 (~1.19B) |
| Baseline (random-init) PPL | 185,810.49 |
| **Final val PPL** | **28.65** |
| Gap vs original (13.40) | **2.14×** |

This is the headline Build-1 number. Scaling 131M → 1.19B tokens at the same recipe **narrowed the gap to the original from 3.5× to 2.14×**. Mid-training eval (from `qwen3_baseline2tpp_train.log`) shows the monotone descent: PPL `60.10 (@2k) → 45.06 (@4k) → 39.44 (@6k) → 35.71 (@8k) → 33.04 (@10k) → 30.93 (@12k) → 29.59 (@14k) → 28.93 (@16k) → 28.66 (@18k) → 28.65 (final)`. Training ran ~2663 min (~44 hr) at ~7,480 tok/s; peak mem held flat at **52.4 GB** (well under the 109 GB cap). At 1.19B tokens the model reliably completes `"The capital of France is" → "Paris…"` (vs the lr24 @131M checkpoint, which did not).

---

## Status legend (per project's "brutal scrutiny" rule)

- **VERIFIED** — bit-exact verify, throughput probe, Phase-A sweep, original-vs-repro, and the Phase-B **baseline** final (run 1 of 4) are all complete and backed by files in `results/`.
- **Phase B comparison (sibling builds):** the **IMU-1 bundle finished at 23.52 — it beats this baseline's 28.65 by −17.9%** at matched 2 TPP (details in the modernized build's README). The **partial-RoPE** runs (25% / 10%) are still in progress; their vs-baseline result is not yet decided. These runs live with the modernized/exploratory builds, not this one.
- The smoke-run artifacts (`qwen3_after.txt` @65.5M tokens → PPL 95.87; `configA`) exist only to prove the stack trains end-to-end and are **not** comparison results.

---

## Gotchas

- **`model.py` is in the parent dir** (`../../model.py`), not here. All scripts add `Qwen3-0.6B/` to `sys.path` to import it.
- **Never raise `--micro_batch` above 4** at seq_len=4096 — it OOMs (probe-verified). Use `--grad_accum` for effective batch.
- **`safe_cuda` must import before torch** — it sets the memory cap and `expandable_segments` before CUDA is initialized. The entrypoints already do this; preserve the import order if you edit them.
- **Token cache files are huge** (`tokcache_1191478400_300000.pt` ≈ 9.5 GB; checkpoints ≈ 3.6 GB each) — they live in `results/` / this dir but are training scratch, not deliverables.
- **"Faithful" ≠ matching released Qwen3 quality.** It means faithful to the *architecture* (bit-exact) and to the paper's *LR shape*. The token budget is ~30,000× smaller than the real run; a 2.14× PPL gap at 1.19B tokens is the expected, honest outcome.
- The smoke `after.txt` says `smoke=True` whenever `--steps == 1000`; longer runs (including `baseline2tpp`) correctly report `smoke=False`.
