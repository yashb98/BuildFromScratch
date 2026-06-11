# Build 2 — Modernized (full IMU-1 bundle + NorMuon)

A from-scratch Qwen3-0.6B that swaps in the **full IMU-1 architecture+training recipe**
on top of this repo's faithful Qwen3 baseline. This is the "kitchen-sink" build: value
residuals, LayerNorm scaling, and per-head attention gating in the model, trained with
the **NorMuon** optimizer (+ cautious weight decay), a WSD schedule, and z-loss.

> **Scope note.** This folder is one of three parallel builds (faithful / modernized /
> exploratory). For the shared Qwen3 architecture (GQA 16/8, head_dim 128, QK-Norm,
> RoPE θ=1e6, RMSNorm ε=1e-6, tied embeddings, 151,936 vocab) see the parent README at
> [`../../README.md`](../../README.md). This README only covers what is **new in this
> build** and how to run it.

> **Honest framing.** This intentionally changes **many variables at once** (architecture
> *and* optimizer *and* schedule). It is a *full-recipe-vs-faithful* comparison, **not** a
> controlled single-variable ablation. Treat any quality delta vs the faithful baseline as
> "did the whole bundle help", not "did component X help".

---

## What's modernized (and the papers behind it)

| Component | What it does | Source (verified 2026-06-08) |
|---|---|---|
| **Value residuals** (Eq4) | Mixes each layer's local value with the first layer's value: `V(l) = s·(α1·V_local + α2·V(1))/√(α1²+α2²)`, learnable `(s,α1,α2)` init `(1,1,0)` | IMU-1, arXiv **2602.02522** |
| **LayerNorm scaling** (Eq5) | Scales each block's pre-norm output by `1/√l`, layer index `l=1..L` | IMU-1, arXiv **2602.02522** |
| **Per-head gating** (Eq3) | `out_h = 2·σ(g_h)·Attn_h`, `g = W_g·x`, one gate logit per query head | IMU-1, arXiv **2602.02522** |
| **NorMuon optimizer** | Muon (Newton-Schulz orthogonalized momentum) + per-neuron (row-wise) 2nd-moment normalization + RMS-matched LR scale, on 2D hidden matrices only | NorMuon, arXiv **2510.05491** |
| **Cautious weight decay** (Eq7) | Apply `−λw` only where `sign(update)==sign(w)` | IMU-1, arXiv **2602.02522** |
| **WSD + z-loss** | Warmup→Stable→linear-decay-to-zero LR; z-loss `1e-4` over the 152k vocab | IMU-1 training recipe (2602.02522) |

The init `(s,α1,α2)=(1,1,0)` and `ln_scale=1.0`/gating-off make the **bundle-off** model
**bit-identical** to the faithful baseline — that's the verify anchor (below). QK-Norm was
**already** in the faithful Qwen3 baseline, so it is not a "new" component here.

### Honest gaps / deviations (read these)

- **NS5 coefficients are standard-Muon, not paper-quoted.** NorMuon (2510.05491) does not
  print the Newton-Schulz quintic coefficients; it cites Jordan et al. 2024 (Muon). We use
  the **standard Muon quintic `(a,b,c)=(3.4445, −4.7750, 2.0315)`** and label it as such in
  [`normuon.py`](normuon.py). This is an inferred value, not a value lifted from the NorMuon paper.
- **muP is omitted.** IMU-1's text does not specify muP; its purpose is cross-scale HP
  transfer, and we train one fixed scale, so it is moot here. Documented deviation, not an oversight.
- **NorMuon ε = 1e-8** chosen (paper does not specify the value).
- **Single WSD stage**, not IMU-1's 3-stage / 72B-token schedule — adapted to our ~1.2B-token (2 TPP) budget.

See [`build2_spec.md`](build2_spec.md) for the full verified method spec with every equation and hyperparameter.

---

## Files in this folder

| File | What it is |
|---|---|
| [`build2_spec.md`](build2_spec.md) | The **verified method spec** — every equation, hyperparameter, source citation, and the honest deviations above. Read this first. |
| [`model_imu1.py`](model_imu1.py) | Qwen3 model **+ IMU-1 bundle** behind config toggles (`use_value_residual`, `use_layernorm_scaling`, `use_head_gating`). All-off ⇒ faithful. |
| [`normuon.py`](normuon.py) | **NorMuon** optimizer + cautious WD, plus the `_newton_schulz5` quintic. 2D matrices only (caller routes 1D/embeds to Adam). |
| [`train_imu1.py`](train_imu1.py) | Trainer: NorMuon/AdamW param split, WSD schedule, chunked CE + chunked z-loss, EMA hook. Reuses the faithful trainer's data/eval helpers. |
| [`verify_imu1.py`](verify_imu1.py) | The verify gate (CPU/fp32, no GPU): bundle-off bit-exactness vs faithful + component-live + NorMuon-descends checks. |
| `results/` | Training logs + smoke checkpoints (see below). |

### `results/` contents

| File | What it is |
|---|---|
| `qwen3_imu1_smoke_train.log` | **1000-step smoke run** log (VERIFIED, complete). |
| `imu1_smoke_stdout.log` | Raw stdout of the smoke run (includes `safe_cuda` cap + compile warnings). |
| `qwen3_imu1_2tpp_train.log` | **Full 2-TPP run** log — **RUNNING / incomplete** (Phase B run 2). |
| `qwen3_imu1_train.log` | A 3-step CPU `--dry_run` wiring check. |
| `checkpoint_imu1_smoke_step500.pt`, `checkpoint_imu1_smoke_step1000.pt` | Smoke-run checkpoints (~1.19 GB each). |

---

## How to run

All commands are run from **this folder**. The trainer imports `safe_cuda` and the faithful
trainer's helpers (it adds the faithful build dir to `sys.path` itself).

### 1. Verify gate (CPU, no GPU, seconds)

```bash
python3 verify_imu1.py
```

What it asserts (and the **actual captured output**, re-run 2026-06-11 on CPU/fp32):

```
[1] bundle OFF        max_abs_err vs faithful = 0.0
[2] value-resid ON    max_abs_err vs faithful = 0.0  (expect 0 at init, a2=0)
[3] ln-scaling Δ=2.4634   head-gating Δ=1.8562  (both expect >0)
[4] full bundle       finite=True  loss=12.312  params=596,508,756
[5] NorMuon toy loss  2.0755 -> 1.3996  (expect decrease)

ALL PASS
```

Interpretation: **bundle-off is bit-identical to the faithful model** (max-abs-err `0.0`),
value-residual at init is also bit-identical (α2=0), LN-scaling and head-gating each
measurably change the output (they're live), the full bundle is finite and computes a loss,
and NorMuon descends a toy objective. The full-bundle model has **596,508,756 parameters**
(the bundle adds ~459k params — gating/value-residual scalars — over the faithful
596,049,920; both are "596M / 0.6B"-branded).

### 2. CPU dry-run (wiring check, no GPU, no dataset)

```bash
python3 train_imu1.py --dry_run
```

Tiny 2-layer synthetic-data loop, 3 steps. (See `results/qwen3_imu1_train.log` for a captured run.)

### 3. Smoke train (GPU, 1000 steps)

```bash
python3 train_imu1.py --run_name imu1_smoke --steps 1000
```

### 4. Full 2-TPP train (GPU)

```bash
python3 train_imu1.py --run_name imu1_2tpp --steps 18150
```

Key defaults (from [`train_imu1.py`](train_imu1.py)): `seq_len 4096`, `micro_batch 4`,
`grad_accum 4` ⇒ **65,536 tokens/step**; NorMuon 2D LR `0.011`, AdamW 1D LR `0.006`,
WD `0.1` (2D only), warmup `50` steps, WSD decay fraction `0.2`, z-loss `1e-4`, grad-clip
`1.0`, bf16, `mem_fraction 0.85`. `safe_cuda` caps CUDA at **85% of the 129 GB unified pool
(~109 GB)** so over-allocation errors cleanly instead of crashing the box.

---

## Results

### Verify gate — VERIFIED ✅

All checks pass (output above). The load-bearing result: **bundle-off == faithful,
`max_abs_err = 0.0` (bit-exact).** Source: live re-run of `verify_imu1.py` (2026-06-11).

### Smoke run (1000 steps, 65.5M tokens) — VERIFIED ✅

Full IMU-1 bundle + NorMuon trains stably and loss descends monotonically.
Source: [`results/qwen3_imu1_smoke_train.log`](results/qwen3_imu1_smoke_train.log).

| Checkpoint | val PPL |
|---|---|
| @250 | 92.38 |
| @500 | 59.51 |
| @750 | 49.61 |
| **@1000** | **39.83** |

- Final train CE @1000: **3.7812** (z-loss term ~`0.0154`).
- Throughput: ~**5,135 tok/s**, peak mem **66.1 GB**, on the GB10 unified-memory box.
- Param split: **224 NorMuon (2D) / 198 AdamW (1D + embeddings)**.
- Token budget: 1000 × 65,536 = **65,536,000 tokens** (~65.5M).

> **PRELIMINARY / not directly comparable.** The smoke run uses a *compressed* 1000-step
> WSD schedule (it starts decaying ~step 800), a tiny token budget, and the *full modernized
> recipe* (NorMuon LR 1.1e-2 + WSD), so its `39.83` is **not** apples-to-apples with the
> faithful baseline's eval at the same step count (different optimizer, schedule, and token
> totals). It is a "does the bundle train and descend" smoke test, not a quality verdict.

### Full 2-TPP run — PENDING ⏳ (RUNNING)

This is **Phase B, run 2 of the planned parallel builds** and is **mid-run** at the time of
writing. Source: [`results/qwen3_imu1_2tpp_train.log`](results/qwen3_imu1_2tpp_train.log).

- Config: `steps=18150`, **1,191,478,400 train tokens** (≈2 TPP), 65,536 tok/step, same 224/198 split.
- Last logged step in the file: **step 950 / 18150**, train CE **3.9375**, LR **1.10e-02**
  (just reached the stable plateau after the 50-step warmup), ~5,216 tok/s, 66.1 GB.
- **No final val PPL yet** — the run has not reached the WSD decay tail. Final number is `[not captured]`.

For context (from the parent README), the **faithful** baseline at the same 2-TPP / 1.19B-token
budget reached **val PPL 28.65**. The head-to-head modernized-vs-faithful number will be
fillable only once this run completes.

---

## Gotchas

- **`verify_imu1.py` result is not persisted** in `results/` — it prints to stdout and was
  re-run live for this README. Re-run it any time; it needs no GPU.
- **`train_imu1.py` reuses the faithful build's helpers** (`stream_tokens`, `evaluate`,
  `PackedTextDataset`, `train_qwen3`) and `safe_cuda` — it inserts those dirs into `sys.path`
  at import time, so run it from this folder and keep the sibling
  `2026-06-08_reproduce-faithful_qwen3-0.6b/` build in place.
- **One training job at a time on the GB10** (single unified-memory pool). The 2-TPP run is
  gated behind Phase A/other runs finishing; don't launch a second GPU job concurrently.
- **NorMuon is strictly 2D**: `step()` asserts `g.dim() == 2`. Anything 1D (norms, scalars,
  biases) or the embedding matrix must be routed to AdamW — `split_params()` handles this.
- **NS5 coefficients are standard-Muon, not from the NorMuon paper** (see Honest gaps). If you
  later find paper-quoted coefficients, change them in `normuon.py::_newton_schulz5`.
