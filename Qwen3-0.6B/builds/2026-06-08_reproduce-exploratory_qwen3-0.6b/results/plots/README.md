# Plots — Qwen3-0.6B EXPLORATORY (partial-RoPE) build

Generated CPU-only (matplotlib Agg, no GPU, no model loads) from real on-disk
training logs only. No smoothing, no extrapolation, no fabricated points.
Generator: `make_plots.py` (in this directory).

## Source logs
- **prope25** — `../qwen3_prope25_2tpp_train.log` — COMPLETED (DONE), 18150/18150 steps, final val PPL = 29.54.
- **prope10** — `../qwen3_prope10_2tpp_train.log` — INCOMPLETE, trainer DIED at step 5450/18150 (no DONE line; only 2 eval points exist).

## Figures

### `exploratory_prope_train_loss.png` / `.pdf`
Training loss (cross-entropy) vs step, prope25 (solid) and prope10 (dashed, X marks death point) overlaid.
Caption: Training loss vs step for partial-RoPE prope25 (factor=0.25, rotary_dim=32, COMPLETED 18150 steps) and prope10 (factor=0.10, rotary_dim=12, INCOMPLETE — died at step 5450/18150). Source: `qwen3_prope25_2tpp_train.log` + `qwen3_prope10_2tpp_train.log`; per-50-step loss parsed directly, no smoothing/extrapolation.

### `exploratory_prope_val_ppl.png` / `.pdf`
Validation perplexity vs step (log y), both runs.
Caption: Eval val-PPL vs step (log y) for prope25 (COMPLETED, final PPL=29.54 at step 18150) and prope10 (INCOMPLETE — died at step 5450, only eval @ 2000 & 4000 exist). Source: `qwen3_prope25_2tpp_train.log` (eval @ 2k..18k + DONE) + `qwen3_prope10_2tpp_train.log`; val-PPL parsed directly, no extrapolation.

## Exact data points plotted (cross-checkable against the source logs)

### prope25 (COMPLETED) — eval val-PPL
| step | val PPL | source line |
|------|---------|-------------|
| 2000  | 62.33 | `[eval @ 2000]`  |
| 4000  | 46.42 | `[eval @ 4000]`  |
| 6000  | 40.66 | `[eval @ 6000]`  |
| 8000  | 36.86 | `[eval @ 8000]`  |
| 10000 | 34.11 | `[eval @ 10000]` |
| 12000 | 31.97 | `[eval @ 12000]` |
| 14000 | 30.55 | `[eval @ 14000]` |
| 16000 | 29.85 | `[eval @ 16000]` |
| 18000 | 29.57 | `[eval @ 18000]` |
| 18150 | 29.54 | `DONE final val PPL=29.54` (terminal point) |

prope25 training loss: 363 points, every 50 steps from step 50 to step 18150 (parsed from `step N/18150 loss ...` lines).

### prope10 (INCOMPLETE — died at step 5450) — eval val-PPL
| step | val PPL | source line |
|------|---------|-------------|
| 2000 | 69.37 | `[eval @ 2000]` |
| 4000 | 50.71 | `[eval @ 4000]` |

prope10 training loss: 109 points, every 50 steps from step 50 to step 5450 (last logged step; trainer died here — no DONE line). Death point: step 5450, loss 3.6663.

## Reproduce
```
CUDA_VISIBLE_DEVICES="" python3 make_plots.py
```
