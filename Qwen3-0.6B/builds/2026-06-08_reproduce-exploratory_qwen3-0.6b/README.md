# Build 3 — Exploratory: partial RoPE

> **One line:** This build is the *faithful* Qwen3-0.6B from-scratch model with a
> single toggle added — **partial RoPE**: rotate only `partial_rotary_factor ×
> head_dim` of each head's dimensions and pass the rest through un-rotated. At
> `factor = 1.0` it collapses **bit-identically** back to the faithful baseline, so
> any measured training difference is attributable to the RoPE fraction and not a
> wiring change.

This is a **folder-scoped** README. For the shared Qwen3-0.6B architecture
(QK-Norm, GQA 16/8, SwiGLU, the spec sheet, parameter accounting, the three-build
experiment design, and the matched-compute methodology) see the parent README at
[`../../README.md`](../../README.md). Everything below is specific to *this* build.

---

## What this is (the experiment)

The faithful build applies RoPE to **all** of `head_dim = 128` dimensions of every
query/key head. This build makes the rotated fraction a config knob and runs it at
three settings:

| `partial_rotary_factor` | `rotary_dim` (rotated dims) | passed-through dims | role |
|---|---|---|---|
| `1.0`  | 128 | 0  | == faithful baseline (verify anchor) |
| `0.25` | 32  | 96 | Phase B run 3 (`prope25_2tpp`) — 🔄 **IN PROGRESS** (~48%, prelim ~3% behind baseline) |
| `0.10` | 12  | 116 | Phase B run 4 (`prope10_2tpp`) — ⏳ **queued** |

`rotary_dim = int(head_dim × factor)`, then rounded **down to even** (rotate_half
chunks in two), with `assert 2 ≤ rotary_dim ≤ head_dim`
(`model_partialrope.py:246-248`).

### The mechanism

The change lives in two places in `model_partialrope.py`:

1. **The cache** (`_build_rope_cache`, lines 93-98) is built over `rotary_dim`, not
   `head_dim`. `inv_freq` divides by `rotary_dim`:

   ```python
   inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, ...) / rotary_dim))
   ```

   This follows the standard HF `partial_rotary_factor` convention used by
   GPT-NeoX / Phi / StableLM: the frequency band spans the same range over fewer
   dims (it divides by `rotary_dim`, **not** `head_dim`). When
   `rotary_dim == head_dim` the cache is bit-identical to the full-RoPE cache.

2. **The application** (`_apply_rope`, lines 106-119) slices each head into a
   rotated part and a pass-through part:

   ```python
   rd = cos.shape[-1]                      # == rotary_dim
   q_rot, q_pass = q[..., :rd], q[..., rd:]
   k_rot, k_pass = k[..., :rd], k[..., rd:]
   q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
   k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
   q_out = torch.cat([q_rot, q_pass], dim=-1)
   k_out = torch.cat([k_rot, k_pass], dim=-1)
   ```

   When `rotary_dim == head_dim`, `q_pass`/`k_pass` are empty and this is the
   full-RoPE path bit-for-bit. Everything else — QK-Norm, GQA, SwiGLU, RMSNorm,
   tied embeddings, the `1e6` theta — is unchanged from the faithful model.

### Backing paper

> "Fractional Rotation, Full Potential?" — **arXiv:2603.11611** (Mar 2026).

Per the project survey, the paper's claim is that **~10% rotation reaches
convergence comparable to full RoPE at the ~135M scale**
(`model_partialrope.py:52-56`). This build is the controlled test of that claim at
the Qwen3-0.6B (596M) scale, against the faithful full-RoPE baseline at matched
compute.

---

## Files in this folder

| File | What it does |
|---|---|
| `model_partialrope.py` | The faithful Qwen3ForCausalLM with one added config field, `partial_rotary_factor` (in `Qwen3Config`), and a sliced `_apply_rope` + `rotary_dim`-sized RoPE cache. Submodule/param names mirror HF `transformers.models.qwen3` so the official safetensors load with no key remap. |
| `verify_partialrope.py` | The verify gate. Proves `factor=1.0` is bit-identical to `../../model.py` (the faithful reference) and that `0.25`/`0.10` give the right `rotary_dim`, stay finite, and actually change the output. CPU / fp32 — does not touch the GPU sweep. |
| `train_partialrope.py` | The trainer. Same AdamW + cosine recipe as the faithful baseline (peak LR `2.4e-3`, the Phase-A winner); the only difference vs baseline is `--partial_rotary_factor`. Reuses the faithful trainer's data/eval/scheduler helpers. |
| `results/` | Created on first training run. **Exists** — holds `qwen3_prope25_2tpp_train.log` (run 3 in progress); `prope10` artifacts appear once run 4 starts. |
| `__pycache__/` | Compiled bytecode; ignore. |

> There is **no separate `Qwen3Config` reimplementation** here — `model_partialrope.py`
> is a self-contained copy of the faithful model with the one toggle added. The
> canonical config values (`vocab_size 151_936`, `hidden_size 1024`,
> `num_hidden_layers 28`, `head_dim 128`, `rope_theta 1e6`, `rms_norm_eps 1e-6`,
> etc.) are documented inline against `config.json`; see the parent README for the
> full spec sheet.

---

## How to run

### Verify (the bit-identity gate) — run this first

```bash
cd builds/2026-06-08_reproduce-exploratory_qwen3-0.6b
python verify_partialrope.py        # CPU / fp32; no GPU needed
```

It loads the faithful reference (`../../model.py`), copies its weights into the
partial-RoPE model at each factor (so only the RoPE buffer differs), and checks
logit equality/difference.

### Train (Phase B runs — one factor per invocation)

`--partial_rotary_factor` and `--run_name` are required. The matched-compute Phase
B recipe (from `../phase_b_driver.sh`, runs 3 and 4):

```bash
# Run 3 — partial RoPE 25%
python train_partialrope.py --steps 18150 --partial_rotary_factor 0.25 \
  --warmup_steps 900 --eval_every 2000 --ckpt_every 2000 --log_every 50 \
  --run_name prope25_2tpp

# Run 4 — partial RoPE 10%
python train_partialrope.py --steps 18150 --partial_rotary_factor 0.10 \
  --warmup_steps 900 --eval_every 2000 --ckpt_every 2000 --log_every 50 \
  --run_name prope10_2tpp
```

Defaults that define the matched-compute budget (`train_partialrope.py:36-52`):
`--seq_len 4096`, `--micro_batch 4`, `--grad_accum 4` ⇒ **65,536 tok/step**;
`--steps 18150` ⇒ **~1.19B tokens** (`18150 × 65,536 = 1,189,478,400`), the same
"2 TPP" budget as the faithful baseline. Peak LR `2.4e-3`, end LR `3.2e-4`, warmup
`900`, weight decay `0.01`, grad clip `1.0`, `--mem_fraction 0.85`, bf16.

These are **not** launched standalone in practice — they are runs [3/4] and [4/4]
of `../phase_b_driver.sh`, which runs all four Phase B jobs sequentially on one GB10.

---

## Results so far

### Verify gate — VERIFIED ✅

Captured from a live CPU run of `verify_partialrope.py` in this folder:

```
[1] factor=1.0  rotary_dim=128  max_abs_err vs faithful = 0.0
[2] factor=0.25  rotary_dim= 32 (want 32)  finite=True  max|Δ| vs full-RoPE = 0.8728  -> OK
[2] factor=0.1   rotary_dim= 12 (want 12)  finite=True  max|Δ| vs full-RoPE = 0.8307  -> OK

ALL PASS
```

| Check | Result |
|---|---|
| `factor=1.0` vs faithful `../../model.py`, max abs logit error | **0.0** (bit-identical) |
| `factor=0.25` rotary_dim | **32** (= 128 × 0.25) |
| `factor=0.10` rotary_dim | **12** (= int(128 × 0.10) → 12, already even) |
| `factor=0.25` max\|Δ\| vs full-RoPE (same weights) | **0.8728** (finite, non-zero → RoPE is doing something) |
| `factor=0.10` max\|Δ\| vs full-RoPE (same weights) | **0.8307** (finite, non-zero) |
| Overall | **ALL PASS** |

The `0.25`/`0.10` deltas are measured with **identical weights** to the baseline
(the faithful state dict is loaded into both via `load_state_dict(..., strict=True)`),
so the only thing that changed is the RoPE buffer — this isolates the partial-RoPE
effect from any weight difference.

### Training (Phase B partial-RoPE runs)

Phase B (`../phase_b_driver.sh`) runs four matched-compute jobs **sequentially** on
one GB10: [1] faithful baseline → [2] IMU-1 bundle → [3] partial RoPE 25% → [4]
partial RoPE 10%. Status as of 2026-06-15:

- **[1/4] faithful baseline — DONE**, `val PPL 28.65` (the full-RoPE reference).
- **[2/4] IMU-1 bundle — DONE**, `val PPL 23.52`.
- **[3/4] `prope25_2tpp` — 🔄 IN PROGRESS** (~step 8,800 / 18,150, ~48%).
- **[4/4] `prope10_2tpp` — ⏳ queued.**

#### prope25 (factor 0.25) — PRELIMINARY mid-run (NOT final)

Same recipe as the baseline (AdamW, cosine, LR 2.4e-3) → the **only** variable is the
rotated-dim fraction, so this is the cleanest single-variable ablation in the project.
Matched-step val PPL (`results/qwen3_prope25_2tpp_train.log`) vs the full-RoPE baseline:

| eval step | Baseline (full RoPE) | Partial-RoPE 25% | Δ |
|---|---|---|---|
| @2000 | 60.10 | 62.33 | +3.7% |
| @4000 | 45.06 | 46.42 | +3.0% |
| @6000 | 39.44 | 40.66 | +3.1% |
| @8000 | 35.71 | 36.86 | +3.2% |

> ⚠️ **PRELIMINARY — not a result.** This run is ~halfway and has **not** reached its
> cosine decay tail; the final number is not yet captured. So far partial-RoPE 25% tracks
> a **steady ~3% behind** full RoPE — broadly consistent with the paper's "comparable
> convergence" claim (rotating only 25% of dims costs a little). The verdict (final PPL
> vs the baseline's 28.65) is still **OPEN** until the run completes, and `prope10` (10%)
> hasn't started. No partial-RoPE quality claim is final yet.

---

## Gotchas

- **Imports reach outside this folder.** `train_partialrope.py` inserts the repo
  root, the model dir (`Qwen3-0.6B/`), this folder, **and the faithful build dir**
  onto `sys.path`, then imports `safe_cuda` (from `BuildFromScratch/`) and the
  faithful trainer helpers `stream_tokens, evaluate, PackedTextDataset,
  chunked_cross_entropy, make_cosine_scheduler` from `train_qwen3`
  (`train_partialrope.py:12-23`). `verify_partialrope.py` imports `model` from
  `../../model.py` as the faithful reference. Run the scripts from this folder so
  those relative paths resolve.
- **`rotary_dim` is rounded down to even**, so a factor that lands on an odd dim
  count loses one dim (e.g. an odd target would round down). `0.25→32` and
  `0.10→12` are both already even, so no rounding loss at the swept settings.
- **`factor=1.0` is the anchor, not a separate experiment** — it is identical to
  Build 1 (faithful) and exists only to prove the wiring. The real ablation is
  `0.25` and `0.10` vs the full-RoPE baseline.
- **CUDA is required for training** (`train_partialrope.py:57-59`); verify runs
  CPU-only by design so it doesn't contend with the GPU sweep.

---

## Provenance

- Architecture / shared details: [`../../README.md`](../../README.md)
- Faithful reference model (verify target): [`../../model.py`](../../model.py)
- Phase B driver (sequential run order): [`../phase_b_driver.sh`](../phase_b_driver.sh)
- Backing paper: "Fractional Rotation, Full Potential?" — arXiv:2603.11611 (Mar 2026)
- Target model: [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
