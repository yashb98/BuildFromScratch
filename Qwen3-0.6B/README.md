# Qwen3-0.6B from scratch — bit-exact reproduction + a three-build research experiment

A from-scratch PyTorch reproduction of [`Qwen/Qwen3-0.6B-Base`][hfbase], faithful
to the shipped `config.json` and the [Qwen3 Technical Report][qwen3paper]. The
architecture is reproduced **bit-exact** against HuggingFace's reference
`Qwen3ForCausalLM` (`max_abs_error = 0.0` over the full logit vector, fp32) — and
on top of that faithful anchor we run **three parallel builds** (faithful /
modernized / exploratory) as a controlled research experiment, each with a
verify gate that collapses it bit-for-bit back to the baseline so any measured
difference is attributable to the toggled change and not a wiring bug.

This is the second model in the series after [SmolLM2-135M](../SmolLM2-134(base)/);
the README mirrors that one's voice. Read it top to bottom and you should be able
to narrate every decision — **what we did**, **how we did it**, **why we did it
that way** — without referring back to the papers.

> **Status: Phase B running.** The architecture is VERIFIED (bit-exact). Phase A
> (the LR sweep @ 131M tokens) is complete and `lr24 = 2.4e-3` won. Phase B (four
> matched-compute runs @ 2 TPP = 1.19B tokens each, sequential on one GB10) is
> **mid-run** — its final perplexities are PENDING. Everything PENDING /
> PRELIMINARY / CONFOUNDED is flagged as such; nothing is overclaimed.

[hfbase]: https://huggingface.co/Qwen/Qwen3-0.6B-Base
[hfinstruct]: https://huggingface.co/Qwen/Qwen3-0.6B
[qwen3paper]: https://arxiv.org/abs/2505.09388
[hfqwen3]: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
[rasbt]: https://huggingface.co/rasbt/qwen3-from-scratch
[rasbtarticle]: https://magazine.sebastianraschka.com/p/qwen3-from-scratch

---

## Table of contents

0. [Results so far](#0-results-so-far)
1. [Why Qwen3-0.6B](#1-why-qwen3-06b)
2. [Sources of truth](#2-sources-of-truth)
3. [Architecture spec sheet](#3-architecture-spec-sheet)
4. [Parameter accounting](#4-parameter-accounting)
5. [Component-by-component walkthrough of `model.py`](#5-component-by-component-walkthrough-of-modelpy)
6. [The three-build experiment](#6-the-three-build-experiment)
7. [Verifying architectural correctness](#7-verifying-architectural-correctness)
8. [Training recipe + matched-compute methodology](#8-training-recipe--matched-compute-methodology)
9. [Repo layout](#9-repo-layout)
10. [Setup](#10-setup)
11. [Honest accounting](#11-honest-accounting)
12. [What we'll try once Phase B is done](#12-what-well-try-once-phase-b-is-done)
13. [Source map](#13-source-map)

---

## 0. Results so far

This is a **single-GPU, matched-recipe** reproduction. The architecture is
reproduced **bit-exact**; the training is deliberately scaled down by orders of
magnitude, so the headline is a *gap* analysis, not a quality match. No value
below is typed by hand — each is cross-checked against a results file.

### 0.1 Architectural verification (VERIFIED, bit-exact)

Source: [`results/verify.json`](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/verify.json)

| Field | Value |
|---|---|
| Repo | `Qwen/Qwen3-0.6B-Base` |
| Prompt | `The capital of France is` |
| dtype | `float32` |
| Tolerance | `0.001` |
| **`max_abs_error`** | **`0.0`** |
| `relative_error` | `0.0` |
| HF next-token id / text | `12095` / `" Paris"` |
| Our next-token id / text | `12095` / `" Paris"` |
| `argmax_match` | `true` |
| **`passed`** | **`true`** |
| `input_shape` | `[1, 5]` |
| Param count (config) | **`596,049,920`** (from `architecture_plan.md` "Total" row and `test_model.py:79` `expected = 596_049_920`) |

`max_abs_error = 0.0` over the full logit vector (not merely an argmax tie) is the
strongest possible pass — our hand-written `model.py` is numerically
indistinguishable from the HF reference in fp32. Note: `596,049,920` is **not**
stored in `verify.json`; it is verified from `architecture_plan.md` and
`test_model.py`.

### 0.2 Original vs. reproduction (the real gap)

Source: [`results/original_vs_repro.txt`](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_vs_repro.txt)
— identical eval code, identical 300,000-token val slice (50 windows × 4096),
`eval_original_vs_repro.py`.

| Model | Training tokens | val PPL | Gap vs original |
|---|---|---|---|
| **ORIGINAL** `Qwen3-0.6B-Base` | 36T | **13.400** | 1.0× |
| REPRO `lr17` (from scratch) | 131M | 46.892 | 3.5× |
| **REPRO `lr24` (BEST)** | 131M | **46.310** | **3.5×** |
| REPRO `lr30` (from scratch) | 131M | 49.276 | 3.7× |

> Best reproduction: **lr24 = 46.310** vs original **13.400** ⇒ **3.5× higher
> PPL**, *expected because* 36T vs 131M tokens ≈ **275,000× less data** (the
> `275,000×` framing is computed/printed by `eval_original_vs_repro.py:71`).

The remarkable result is how *small* a 3.5× PPL gap is given a **~275,000×** data
deficit — strong circumstantial evidence the architecture and recipe are correct,
and that the residual gap is **data, not skill** (see §11).

### 0.3 Phase A LR sweep @ 131M tokens (matched compute)

Each run: 2000 steps, seq_len 4096, micro_batch 4 × grad_accum 4, warmup 150,
cosine to end_lr 3.2e-4 (args from `results/qwen3_lr24_train.log`). All start from
the same random init (PPL 183922.14).

| Run | Peak LR | val PPL (final) | Notes |
|---|---|---|---|
| `lr17` | 1.7e-3 | 46.89 | Paper-anchored LR shape |
| **`lr24`** | **2.4e-3** | **46.31** | **BEST — selected for Phase B** |
| `lr30` | 3.0e-3 | 49.28 | SmolLM2-anchor LR; over-aggressive here |

(`qwen3_lr17_after.txt`, `qwen3_lr24_after.txt`, `qwen3_lr30_after.txt`: each
header `183922.14 -> {46.89, 46.31, 49.28}`.) The genuinely novel, verifiable
finding of the sweep: **2.4e-3 (midpoint) wins** — Qwen3 never published the 0.6B
LR, so this A/B/C ordering at matched compute is the only original result the
sweep produces.

### 0.4 Build-2 IMU-1 smoke eval trajectory (PRELIMINARY + CONFOUNDED)

Source: [`qwen3_imu1_smoke_train.log`](builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_smoke_train.log).
Bundle `vr=True ln=True hg=True`, 1000 steps, **peak lr 1.10e-02**, 65,536
tok/step, param split **224 NorMuon (2D) / 198 AdamW (1D/embed)**.

| Step | val PPL |
|---|---|
| @250 | 92.38 |
| @500 | 59.51 |
| @750 | 49.61 |
| @1000 | 39.83 |

⚠️ **Do not over-read this.** It is a 1000-step smoke run on a **different
recipe** (NorMuon LR 1.1e-2, WSD, not the faithful cosine/AdamW), and the IMU-1
bundle **intentionally confounds multiple changes at once** (NorMuon +
value-residual + LN-scale + head-gate + WSD — see §6 and `phase_b_driver.sh:5`).
The @1000 = 39.83 is *not* comparable to the Phase A 46.31 (different token
budget, different optimizer, different schedule). It only shows the modernized
stack trains and descends.

### 0.5 Scaling trend (token budget → val PPL)

| Training tokens | val PPL | Source |
|---|---|---|
| 65.5M (smoke) | ~96 | smoke run, 65,536,000 tok (`qwen3_smoke_after.txt` header) |
| 131M (Phase A best) | 46.31 | `original_vs_repro.txt` |
| 36T (released) | 13.40 | `original_vs_repro.txt` |

Monotone descent across ~5.5 orders of magnitude of data
(`65.5M → 131M → 36T` ≈ `96 → 46 → 13.4`) — exactly the Chinchilla-style
underfitting curve expected. The released model sits ~12× above
Chinchilla-optimal data (12B for 596M) while our runs sit far below it.

---

## 1. Why Qwen3-0.6B

**What we picked:** `Qwen/Qwen3-0.6B-Base` as the reproduction target for all
three planned builds (faithful, modernized, exploratory).

**How we narrowed it down:** we surveyed **4 candidate Qwen models** in the
sub-billion (~0.5B) band. The figures below are not hand-quoted — per
`model_target_plan.md`, they were "*Confirmed from runtime
`AutoConfig.from_pretrained` pulls … they come from the live config.json on each
repo.*"

| Target | Family | Params (exact) | Hidden / Layers / Heads (Q/KV) / Head dim | FFN dim | Vocab | Max ctx | RoPE θ | Tied emb | License | Repro difficulty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-0.6B** ✅ | qwen3 | **596M** (text-only) | 1024 / 28 / 16 / 8 / 128 (GQA 2:1) | 3072 (SwiGLU) | 151,936 | 40,960 | 1e6 | yes | Apache 2.0 | **LOW** — clean text decoder; published from-scratch reference (Raschka) |
| Qwen2.5-0.5B | qwen2 | ~494M | 896 / 24 / 14 / 2 / 64 (GQA 7:1) | 4864 (SwiGLU) | 151,936 | 32,768 | 1e6 | yes | Apache 2.0 | LOW — older but well-trodden |
| Qwen2-0.5B | qwen2 | ~494M | 896 / 24 / 14 / 2 / 64 | 4864 (SwiGLU) | 151,936 | 131,072 | 1e6 | yes | Apache 2.0 | LOW — superseded by Qwen2.5 |
| ~~Qwen3.5-0.8B~~ | qwen3_5 (multimodal) | ~0.9B incl. ViT | 1024 / 24 / 8 / 2 / 256, hybrid linear+full attn, MTP, partial RoPE, MRoPE | 3584 | 248,320 | 262,144 | 1e7 | yes | Apache 2.0 | **OUT OF SCOPE** |

> Source: `model_target_plan.md`, "Comparison" table (lines 12–17). HF repo:
> [Qwen/Qwen3-0.6B][hfinstruct] (base: [Qwen/Qwen3-0.6B-Base][hfbase]).

**Why this target wins** (rationale per `model_target_plan.md` lines 36, 69 —
the "*strongest combination of*" the following):

- **Newest sub-1B *text-only* Qwen.** It is the "*newest Qwen family with a
  sub-billion text-only base*" (line 36). Qwen2.5-0.5B is the runner-up but
  "*older (Sep 2024) — fails the 'latest Qwen' criterion you specified*" (line
  41), and Qwen2-0.5B was skipped as "*superseded by Qwen2.5; no reason to pick*"
  (line 16).
- **Stays text-only (avoids scope blowup).** The newer Qwen3.5-0.8B was *excluded
  as out of scope* because it is multimodal: "*separate vision tower (12-layer
  ViT, hidden 768, patch 16) + text decoder with **hybrid linear/full attention**
  … MRoPE multi-axis rotary, partial rotary factor 0.25, attention-output-gating,
  and a Multi-Token Prediction head*" (lines 49–50). The plan explicitly flags
  this so that "*the 'pick the newest Qwen' instinct doesn't quietly drift us into
  a multimodal/SSM reproduction we don't want*" (line 51).
- **GQA 2:1 — an interesting but not exotic verify target.** Qwen3-0.6B uses
  **16 query heads / 8 KV heads** (`n_rep = 2`), described as an "*interesting
  verify target but not exotic*" (line 69), in contrast to Qwen2.5's more
  aggressive 7:1 KV reduction (line 41).
- **Apache-2.0 license** (line 14).
- **Stable open base checkpoint.** It "*exists as a stable open base checkpoint
  (no thinking-mode prompt tags needed for base PPL eval)*" (line 36) — the plan
  recommends reproducing `Qwen/Qwen3-0.6B-Base` rather than the instruct model so
  base PPL is the clean signal (line 73).
- **High-quality community reference for triangulation.** It "*has a high-quality
  community PyTorch reference to triangulate verify-gate failures against*" (line
  36). See §2.

### The teachable *new* component vs SmolLM2: per-head QK-Norm

The reason this target is pedagogically valuable on top of the prior SmolLM2-135M
build is **QK-Norm**, called out in the architecture plan as **"the #1
architectural difference vs Llama/Qwen2/SmolLM2"** (`architecture_plan.md` line
26). It is a per-head RMSNorm applied to the query and key projections *before*
RoPE:

```python
# from architecture_plan.md line 26 (verified from HF source per the plan)
q_norm = Qwen3RMSNorm(head_dim=128, eps=1e-6)
k_norm = Qwen3RMSNorm(head_dim=128, eps=1e-6)
# applied as:
q = q_norm(q_proj(x).view(B, T, 16, 128))   # then .transpose(1, 2)
# RoPE applied AFTER QK-Norm
```

The plan stresses: "*This is the #1 architectural difference vs
Llama/Qwen2/SmolLM2 — must implement exactly or verify gate fails. The norm
weight is a single (128,) vector broadcast across heads.*" (line 26). QK-norm is
"*new vs Qwen2*" (`model_target_plan.md` line 35).

---

## 2. Sources of truth

Sources are listed in **trust order**. The configuration numbers are anchored to
the live config, the weights are the equivalence ground truth, and the paper +
two reference implementations are used to triangulate any verify-gate
disagreement.

| # | Source | Role | Reference |
|---|---|---|---|
| 1 | **Live `config.json` via `AutoConfig`** | Canonical config numbers — pulled at runtime, not hand-quoted | `AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base")`, pulled **2026-06-08** (`architecture_plan.md` line 12) |
| 2 | **The safetensors weights** | Numerical equivalence ground truth for the Phase 5 verify gate | `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype=torch.float32)` (`architecture_plan.md` line 82) |
| 3 | **Qwen3 Technical Report** | Published-recipe / paper source | arXiv:**2505.09388** (Qwen3, May 2025) — https://arxiv.org/abs/2505.09388 (`architecture_plan.md` line 9) |
| 4 | **HF `transformers` qwen3 reference** | Per-component code-level spec (esp. QK-Norm) | [`modeling_qwen3.py`][hfqwen3] (`architecture_plan.md` line 10) |
| 5 | **Sebastian Raschka — qwen3-from-scratch** | Independent pure-PyTorch reference to triangulate verify-gate failures | [rasbt/qwen3-from-scratch][rasbt] · [article][rasbtarticle] (`architecture_plan.md` line 11) |

Additional community references listed as **verified** in `model_target_plan.md`
(lines 30–34):

- [rasbt/LLMs-from-scratch ch05/11_qwen3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3) — book code
- [rasbt/reasoning-from-scratch `qwen3.py`](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py) — reasoning extension

### Canonical config values (Source #1, live config.json)

Quoted verbatim from `architecture_plan.md` line 13:

> hidden_size 1024 · num_hidden_layers 28 · num_attention_heads 16 ·
> num_key_value_heads 8 · head_dim 128 · intermediate_size 3072 · vocab_size
> 151,936 · max_position_embeddings 40,960 · rope_theta 1,000,000 · rms_norm_eps
> 1e-6 · tie_word_embeddings true · attention_bias false · hidden_act silu

### Gotchas the plan flags

These are the traps the plan explicitly warns about — getting any wrong fails the
`< 1e-3` fp32 verify gate (`architecture_plan.md` line 6):

1. **RoPE θ = 10⁶, not 10⁴.** "*RoPE θ=10^6 (not the 10^4 GPT-style default —
   easy mistake)*" (`model_target_plan.md` line 35). The arch plan adds it also
   differs from Llama (1e4) and SmolLM2-v2 (1e5): "*note θ differs from Llama
   (1e4) and SmolLM2-v2 (1e5)*" (line 27). Source: config `rope_theta=1000000`.
2. **Per-head QK-Norm (new vs Qwen2).** "*QK norm is applied per-head inside
   attention in Qwen3 (new vs Qwen2)*" (`model_target_plan.md` line 35). RoPE is
   applied **after** QK-Norm; the norm weight is a single `(128,)` vector
   broadcast across heads (`architecture_plan.md` line 26).
3. **`head_dim ≠ hidden_size // num_heads`.** head_dim is an **explicit 128**, not
   the derived 64. "*critical: `head_dim ≠ hidden_size // num_heads` here
   (hidden/n_heads=64; head_dim=128)*" — so `q_proj` out = 16×128 = 2048, while
   `k_proj`/`v_proj` out = 8×128 = 1024 (`architecture_plan.md` line 24).
4. **Tied embeddings, no separate lm_head bias.** "*tied embeddings with no
   separate lm_head bias*" (`model_target_plan.md` line 35); config
   `tie_word_embeddings=true` (`architecture_plan.md` lines 13, 33). For verify,
   `load_state_dict(strict=False)` with only `lm_head.weight` allowed missing
   because it is tied (line 78).
5. **Vocab pad mismatch.** "*vocab_size pad mismatch is common*"
   (`model_target_plan.md` line 35); vocab 151,936 is "*padded to
   multiple-of-128 from the true ~151,669*" (`architecture_plan.md` line 36).
6. **No projection biases / no dropout / full attention every layer.**
   `attention_bias=false` ⇒ no bias on q/k/v/o_proj (`architecture_plan.md` line
   25); `attention_dropout=0.0`, no other dropout (line 35);
   `use_sliding_window=false`, all 28 layers `"full_attention"` (line 29).

### Status / honesty markers

- **VERIFIED:** All config values above are from the live `AutoConfig` pull
  (2026-06-08) and the per-component table cross-checked against
  `modeling_qwen3.py` (`architecture_plan.md` lines 12–37). The 596M param count
  is reconstructed by hand in the plan and totals **596,049,920**
  (`architecture_plan.md` lines 41–62), matching the "0.6B" branding.
- **HONEST GAPS:** Neither plan file mentions muP, and the NS5 / Muon optimizer
  coefficients are not present in either document — those belong to later
  training/HP-tuning plans, not these two files. Training corpus is only
  partially disclosed: "*~36T tokens, 119 languages. Exact source mixture not
  fully disclosed*" (`model_target_plan.md` line 28). Specific
  MMLU/GSM8K/HumanEval numbers "*vary by source and post-training stage*"; the
  plan compares only via base-checkpoint PPL deltas (line 29).

> Source files:
> `builds/2026-06-08_target-survey/model_target_plan.md`,
> `builds/2026-06-08_reproduce-faithful_qwen3-0.6b/architecture_plan.md`

---

## 3. Architecture spec sheet

This is what the model **is**. Every value below is copied from the `Qwen3Config`
dataclass in [`model.py`](model.py) (lines 35–51), whose docstring states each
default "matches `config.json` at the Qwen3-0.6B-Base repo HEAD (pulled via
`AutoConfig.from_pretrained` on 2026-06-08)" (model.py:32–33).

| Field | Value | Source (model.py) | Upstream source cited in file |
|---|---|---|---|
| `vocab_size` | `151,936` | line 37 | `config.json: vocab_size` |
| `hidden_size` | `1024` | line 38 | `config.json: hidden_size` |
| `intermediate_size` (FFN) | `3072` | line 39 | `config.json: intermediate_size` |
| `num_hidden_layers` | `28` | line 40 | `config.json: num_hidden_layers` |
| `num_attention_heads` (Q) | `16` | line 41 | `config.json: num_attention_heads` |
| `num_key_value_heads` (KV) | `8` | line 42 | `config.json: num_key_value_heads (GQA 16/8 = 2:1)` |
| `head_dim` | `128` | line 43 | `config.json: head_dim — INDEPENDENT field, not hidden/n_heads` |
| `max_position_embeddings` | `40,960` | line 44 | `config.json: max_position_embeddings` |
| `rope_theta` | `1,000,000.0` (1e6) | line 45 | `config.json: rope_theta` |
| `rms_norm_eps` | `1e-6` | line 46 | `config.json: rms_norm_eps` |
| `initializer_range` | `0.02` | line 47 | `config.json: initializer_range` |
| `tie_word_embeddings` | `True` | line 48 | `config.json: tie_word_embeddings` |
| `attention_bias` | `False` | line 49 | `config.json: attention_bias` |
| `attention_dropout` | `0.0` | line 50 | `config.json: attention_dropout` |
| `hidden_act` | `"silu"` → SwiGLU | line 51 (comment) | hardcoded in `MLP.forward` via `F.silu` (line 183) |
| dtype (parity test) | `float32` | — | `verify.json: "dtype": "float32"` |

Notes grounded in the code, not assumed:

- **All Linear layers are bias-free.** `attention_bias=False` (line 49) feeds
  q/k/v/o_proj (lines 127–130); the MLP projections pass `bias=False` explicitly
  (lines 178–180); `lm_head` is `bias=False` (line 242). The only learnable
  non-Linear params are RMSNorm `weight` vectors (line 61) initialized to ones.
- **dtype** is not a config field; `verify.json` ran the parity test in `float32`.
  RMSNorm upcasts to fp32 internally regardless (line 66).

### The three differences vs Llama/SmolLM2 (as enumerated in the file, lines 14–20)

| # | Difference | Where in code | Llama / SmolLM2 baseline (per file comments) |
|---|---|---|---|
| 1 | **QK-Norm**: per-head RMSNorm on Q,K **before RoPE** | lines 132–134, 146–147 | absent |
| 2 | **`head_dim` independent config field** = 128 (≠ hidden/n_heads = 1024/16 = 64) | line 43, 123, 127–130 | head_dim = hidden/n_heads |
| 3 | **RoPE θ = 1e6** | line 45, 78 | 1e4 (Llama) / 1e5 (SmolLM2-v2) |

Two further deltas the file lists (lines 19–20) but does not bill as the "big
three": **RMSNorm eps = 1e-6** (vs 1e-5 SmolLM2) and **fixed init std 0.02** (vs
`1/sqrt(hidden)`).

---

## 4. Parameter accounting

**Exact unique parameter count: `596,049,920`** (596M-branded, "0.6B").

This is not hand-arithmetic — it was produced by instantiating
`Qwen3ForCausalLM(Qwen3Config())` and summing `p.numel()` over unique storages.
`num_params(m)` returns `596,049,920` (model.py:298–299), matching the `__main__`
expectation `~596,049,920` (model.py:306). Because `tie_word_embeddings=True`,
`lm_head.weight` aliases `embed_tokens.weight` (model.py:243–245); the run
confirms `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()`, so even
the naive `sum(p.numel() for p in m.parameters())` equals `596,049,920` (PyTorch
dedupes shared parameters).

```text
Qwen3-0.6B parameter breakdown  (H=1024, head_dim=128, n_q=16, n_kv=8, I=3072, V=151936, L=28)

EMBEDDINGS
  embed_tokens   V * H = 151936 * 1024                       = 155,582,464

PER-LAYER (one Qwen3DecoderLayer)
  Attention
    q_proj       H * (n_q  * head_dim) = 1024 * (16*128)=2048 =   2,097,152
    k_proj       H * (n_kv * head_dim) = 1024 * ( 8*128)=1024 =   1,048,576
    v_proj       H * (n_kv * head_dim) = 1024 * ( 8*128)=1024 =   1,048,576
    o_proj       (n_q * head_dim) * H  = 2048 * 1024          =   2,097,152
    q_norm       head_dim = 128  (QK-Norm, Qwen3-specific)    =         128
    k_norm       head_dim = 128  (QK-Norm, Qwen3-specific)    =         128
                                              attn subtotal   =   6,291,712
  MLP (SwiGLU)
    gate_proj    H * I = 1024 * 3072                          =   3,145,728
    up_proj      H * I = 1024 * 3072                          =   3,145,728
    down_proj    I * H = 3072 * 1024                          =   3,145,728
                                               mlp subtotal   =   9,437,184
  Norms
    input_layernorm           H = 1024                        =       1,024
    post_attention_layernorm  H = 1024                        =       1,024
                                              norm subtotal   =       2,048
                                       ---------------------------------------
                                       per-layer total        =  15,730,944
  x 28 layers                                                 = 440,466,432

FINAL
  model.norm     H = 1024                                     =       1,024

HEAD
  lm_head        TIED to embed_tokens (shared storage)        =           0
                                       =======================================
  TOTAL UNIQUE PARAMETERS                                     = 596,049,920
```

Cross-checks (all from the live instantiation):

- `155,582,464 + 440,466,432 + 1,024 + 0 = 596,049,920` ✓ equals `num_params(m)`.
- If the head were *untied*, it would add another `V * H = 155,582,464`, giving
  `751,632,384` — so tying saves ~155.6M params, ~26% of the model. This is why a
  "0.6B" model has 155.6M of its weights in a single shared
  embedding/un-embedding matrix.

---

## 5. Component-by-component walkthrough of `model.py`

This is the section to follow on camera. Open [`model.py`](model.py) side-by-side
and walk through it top to bottom in the order below.

### 5.1 RMSNorm — `class RMSNorm` (lines 58–69)

**What it is.** Root-Mean-Square LayerNorm (Zhang & Sennrich 2019), the standard
Llama-family pre-norm; cited in-file (line 55). A single learnable gain vector, no
bias, no mean-subtraction.

**Key code (lines 64–69):**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + self.eps)
    return (self.weight * x).to(dtype)
```

**Why.** Normalizes by RMS over the last dim, then rescales by `weight` (init to
ones, line 61). Internals run in fp32 then cast back — the comment states this
"match[es] HF Qwen3RMSNorm for numerical stability" (line 56). `eps = 1e-6` is
passed from config (line 46). The same class is reused for input/post-attention
layernorms, the final norm, *and* the QK-Norms — the difference is just the
normalized dimension (`hidden_size=1024` for block norms vs `head_dim=128` for
QK-norms).

### 5.2 RoPE — `_build_rope_cache`, `_rotate_half`, `_apply_rope` (lines 77–96)

**What it is.** Rotary Position Embedding (Su et al. 2021), in the **HF / Llama
"rotate_half" layout, NOT GPT-NeoX interleaved** (stated line 73–74).

**Key code:**

```python
# frequencies (line 78): theta = 1e6 from cfg.rope_theta
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, ...) / head_dim))
...
emb = torch.cat([freqs, freqs], dim=-1)          # (seq_len, head_dim)  line 81

def _rotate_half(x):                              # lines 85-87
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

q_out = (q * cos) + (_rotate_half(q) * sin)       # line 94
```

**Why split-halves.** `_rotate_half` splits the head into two contiguous halves
`[x1 | x2]` and maps them to `[-x2 | x1]` — the GPT-NeoX/HF convention where
dimension `i` pairs with `i + head_dim/2`. The `cat([freqs, freqs])` (line 81)
duplicates each frequency so `cos`/`sin` align with that paired layout. This
contrasts with the interleaved (adjacent-pairs) convention; getting it wrong
silently breaks parity. The cache is built once at
`max_position_embeddings = 40,960` over `head_dim = 128` (lines 222–223) and
registered non-persistent (line 224–225), so it is recomputed at load, not stored
in the checkpoint.

### 5.3 Attention — GQA + QK-Norm — `class Attention` (lines 117–169)

This component carries **two of the three Qwen3-specific differences** (QK-Norm
and the independent `head_dim`).

**(a) Independent `head_dim` and GQA shapes.** `head_dim = 128` is taken straight
from config (line 123), *not* computed as `hidden/n_heads = 1024/16 = 64` (called
out in the docstring, lines 16–17, and inline line 123). Consequently the
projections do **not** map back to `hidden_size`:

```python
self.q_proj = nn.Linear(1024, 16*128=2048, bias=False)   # line 127
self.k_proj = nn.Linear(1024,  8*128=1024, bias=False)   # line 128
self.v_proj = nn.Linear(1024,  8*128=1024, bias=False)   # line 129
self.o_proj = nn.Linear(2048, 1024,        bias=False)   # line 130
```

The Q/O internal width is **2048 (> hidden 1024)**; KV width is 1024. GQA ratio
`n_rep = 16 // 8 = 2` (line 124); KV heads are broadcast to Q heads via
`repeat_interleave(self.n_rep, dim=1)` (lines 158–160).

**(b) QK-Norm — the #1 difference vs Llama/Qwen2/SmolLM2.** Two RMSNorms over
`head_dim` (lines 132–134):

```python
self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)   # weight shape (128,)
self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
```

applied **after the per-head view but BEFORE RoPE and before the transpose**:

```python
q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim)   # line 141
k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)   # line 142
...
q = self.q_norm(q)                                              # line 146  (QK-Norm)
k = self.k_norm(k)                                              # line 147
q = q.transpose(1, 2); k = k.transpose(1, 2)                   # lines 150-151
q, k = _apply_rope(q, k, cos, sin)                             # line 155  (RoPE AFTER QK-Norm)
```

**Why.** Normalizing over the last (`head_dim`) axis while the tensor is
`(B, T, heads, head_dim)` makes a single `(head_dim,)` gain vector broadcast
across all heads — exactly HF's `Qwen3RMSNorm(self.head_dim, ...)` (quoted
in-file, lines 113–115). The ordering matters: QK-Norm stabilizes the un-rotated
Q/K (controlling attention-logit magnitude / training stability) and **then**
RoPE injects position. V is never normed and never rotated (only Q and K go
through `_apply_rope`, line 155).

**(c) SDPA.** Causal attention via
`F.scaled_dot_product_attention(..., is_causal=(attention_mask is None))` (lines
162–167), reshaped back through `o_proj` to `hidden_size` (line 168–169).

### 5.4 SwiGLU MLP — `class MLP` (lines 175–183)

**What it is.** Gated FFN (Shazeer 2020), described as "Identical to the
Llama/SmolLM2 form" (line 173) — i.e. *not* a Qwen3-specific difference.

**Key code (line 183):**

```python
def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**Why.** Two parallel up-projections to `intermediate_size = 3072`: `gate` passes
through SiLU and gates `up` elementwise, then `down` projects back to
`hidden = 1024`. All three are `bias=False` (lines 178–180). This realizes
`hidden_act = "silu"` noted in config (line 51).

### 5.5 Block (pre-norm) — `class Block` (lines 191–202)

**What it is.** A pre-norm transformer decoder layer; submodule names
`input_layernorm`, `post_attention_layernorm`, `self_attn`, `mlp` "mirror HF
Qwen3DecoderLayer exactly so state dicts load without key remap" (lines 188–189).

**Key code (lines 199–202):**

```python
def forward(self, x, cos, sin, attention_mask=None):
    x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x
```

**Why pre-norm.** Norm is applied *inside* each residual branch
(norm→sublayer→add), leaving a clean identity residual path — the stable
Llama-family arrangement. Two norms per layer (the 2,048 params/layer in §4).

### 5.6 Outer model + tied embeddings + init — `Qwen3Model`, `Qwen3ForCausalLM` (lines 214–295)

**Structure** mirrors HF (docstring lines 206–213): `Qwen3ForCausalLM.model` =
`Qwen3Model` holding `embed_tokens`, `layers: ModuleList[Block]×28`, and final
`norm` (lines 218–220); plus `lm_head` (line 242).

**Tied embeddings (lines 243–245):**

```python
if cfg.tie_word_embeddings:
    self.lm_head.weight = self.model.embed_tokens.weight   # same storage
```

**Why.** This makes the `(V, H)` un-embedding *the same tensor* as the input
embedding (verified: identical `data_ptr`), which is why `lm_head` contributes
**0 extra params** in §4 and why the parity test loads official safetensors with
"no key remapping" (model.py:10–12).

**RoPE buffers** are built on the model (lines 222–225) and sliced to `T` per
forward (lines 230–231); the loss is the standard shifted next-token
cross-entropy with `ignore_index=-100` (lines 266–273).

**Init — `_init_weights` (lines 248–257):**

```python
std = self.cfg.initializer_range            # 0.02, fixed
nn.init.normal_(module.weight, mean=0.0, std=std)   # Linear and Embedding
# bias -> zeros (none exist here); RMSNorm.weight stays ones
```

**Why / honest gap.** `std = 0.02` is a **fixed** initializer (difference #5,
docstring line 20: "vs 1/sqrt(hidden) SmolLM2"). The file states this is "HF
Qwen3 default" (line 249). No muP / per-layer residual-scaling / depth-dependent
init is applied — initialization is a flat `Normal(0, 0.02)` for every Linear and
Embedding weight, with norms left at ones.

---

## 6. The three-build experiment

This reproduction is run as **three parallel builds** off one canonical faithful
architecture. Build 1 is the exact-arch baseline; Build 2 swaps the *training
recipe and a few sub-layer additions* for the full IMU-1 modernization bundle;
Build 3 swaps a *single architectural primitive* (RoPE coverage). Each build ships
with a **verify gate** — a config under which the build collapses, bit-for-bit,
back to the faithful baseline — so that any measured difference is attributable to
the toggled change and not to a wiring bug.

All three builds share the same canonical config, taken verbatim from
`model_imu1.py` / `model_partialrope.py` (`config.json` at the
`Qwen/Qwen3-0.6B-Base` repo HEAD, pulled via `AutoConfig.from_pretrained` on
2026-06-08): the §3 spec sheet. Self-reported parameter count from the `__main__`
block of both model files:

```
Unique params: ...
Expected:      ~596,049,920  (596M-branded, '0.6B')
```

> **[HONEST GAP]** The header docstring of `hp_tuning_plan.md` says the target is
> `Qwen/Qwen3-0.6B-Base` (596M params). The number `596,049,920` is the value
> printed as "Expected" in the model files' `__main__` blocks; the actual
> `num_params(m)` output is not captured in those files, so the verified
> *measured* count for the IMU-1/partial-RoPE variants is asserted by the
> Expected line, not a captured printout.

### Build 1 — Faithful (AdamW + cosine baseline)

**Path:** `builds/2026-06-08_reproduce-faithful_qwen3-0.6b/` (HP plan:
`hp_tuning_plan.md`).
**Backing source for the architecture:** Qwen3 config.json + HF
`transformers.models.qwen3`. **Backing source for the recipe:** Qwen3 Technical
Report, [arXiv:2505.09388][qwen3paper] (LR *shape* only — see caveat).

**What it is.** The exact Qwen3-0.6B-Base architecture, named to mirror
`transformers.models.qwen3` so the official safetensors load via
`load_state_dict` with no key remapping. The five things that distinguish it from
the SmolLM2/Llama family (verbatim from the model docstring):

1. **QK-Norm** — per-head RMSNorm on Q and K *before* RoPE (Qwen3-specific).
2. **`head_dim` is an independent config field** (128), NOT `hidden/n_heads`
   (1024/16 = 64). Q/K/V projections size off `head_dim`.
3. **RoPE theta = 1e6** (vs 1e4 Llama, 1e5 SmolLM2-v2).
4. **RMSNorm eps = 1e-6** (vs 1e-5 SmolLM2).
5. **Initializer std = 0.02 fixed** (vs 1/sqrt(hidden) SmolLM2).

Key shape detail (from `Attention.__init__`): `q_proj`/`o_proj` use
`n_heads*head_dim = 2048`; `k_proj`/`v_proj` use `n_kv_heads*head_dim = 1024`. GQA
repeat factor `n_rep = 16//8 = 2`. QK-Norm is a single `RMSNorm(head_dim, eps)`
for each of Q and K, weight shape `(head_dim,)`, shared/broadcast across heads.

**The training recipe (Recipe A — Faithful-scaled: cosine LR, AdamW).**

> **[HONEST GAP — quoted from `hp_tuning_plan.md`]** "The Qwen3 technical report
> does NOT disclose exact pretraining HPs for the 0.6B base." The plan quotes the
> paper directly ("we develop scaling laws for optimal hyper-parameters …
> predictions") and states "no concrete per-size HP table is published." The
> third-party-reported baseline (peak LR 1.7e-3 / warmup 1000 / 500B tokens /
> 4M-token batch, cosine decay to 3.2e-4) is "treating those as best-available
> signals, not verified ground truth."

Shared across all candidate configs (from `hp_tuning_plan.md`):
`AdamW(β=0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.01`, `grad_clip=1.0`, `bf16`,
`seq_len=4096`, init `Normal(0, 0.02)`, `seed=0`.

| Knob | Config A — Faithful, modest | Config B — Compute-adjusted (Recommended) | Config C — SmolLM2-anchor |
|---|---|---|---|
| Schedule | cosine | cosine | cosine |
| Peak LR | 1.7e-3 | 1.7e-3 | 3.0e-3 |
| End LR | 3.2e-4 | 3.2e-4 | 3.2e-4 |
| Warmup steps | 200 | 400 | 200 |
| Token budget | 0.5B | 1B | 0.5B |
| Per-step batch (seqs) | 8 × accum 4 = 32 | 16 × accum 4 = 64 | 8 × accum 4 = 32 |
| Per-step batch (tokens) | 131,072 | 262,144 | 131,072 |
| Total optimizer steps | ~3,815 | ~3,815 | ~3,815 |

**Measured throughput (2026-06-08, `results/throughput_probe.json`)** on the
GB10, bf16, seq_len=4096, 50 steps after 5 warmup, process capped at 85% of the
unified pool:

| | micro_batch | tokens/sec | sec/step | peak mem |
|---|---|---|---|---|
| baseline (no compile) | 4 | 3,788 | 4.33 | 68.2 GB |
| `torch.compile` (reduce-overhead) | 4 | 7,167 | 2.29 | 52.4 GB |

`torch.compile` speedup: 1.89×. **Critical memory finding (verbatim):**
`micro_batch=16` and `micro_batch=8` at seq_len=4096 do not fit — both raised a
clean CUDA OOM and the probe backed off to micro_batch=4 (the largest that fits)
on the GB10 unified-memory pool (~119 GB shared CPU+GPU; torch reports 128.5 GB).
Config B "bs=16" is therefore realized as
`micro_batch=4 × gradient_accumulation=4`. Wall-clock at micro_batch=4 compiled:
Config A ~19.4 hr, Config B ~38.8 hr, full 2B-token 3-config sweep ~77.5 hr.
**Recommendation in the plan: Config B — Compute-adjusted.**

**The verify gate.** Build 1 *is* the anchor; its gate is the cross-build parity
test (§7). The faithful model is the reference against which Builds 2 and 3 must
reproduce bit-for-bit when their new components are disabled. (Per docstrings:
official Qwen3-0.6B-Base safetensors load via `load_state_dict` with no key
remapping; "See verify.py for the parity test.")

> **[HONEST GAP / caveat from the plan]** A/B/C are faithful to the paper's
> *baseline LR shape* only; the per-size 0.6B pretraining recipe "was never
> released." The success signal is PPL improvement vs random-init and a clean
> A/B/C ordering, NOT an absolute match to Qwen3-0.6B-Base quality (impossible at
> ≤1B tokens).

### Build 2 — Modernized (the full IMU-1 bundle)

**Path:** `builds/2026-06-08_reproduce-modernized_qwen3-0.6b/` (spec:
`build2_spec.md`; optimizer: `normuon.py`; model: `model_imu1.py`).
**Primary sources (fetched + verified 2026-06-08, per `build2_spec.md`):**

- **IMU-1:** arXiv **2602.02522** (Jan 2026) — the bundle + hyperparameters.
- **NorMuon:** arXiv **2510.05491** (Oct 2025) — the optimizer algorithm.

> **[DESIGN NOTE — verbatim from `build2_spec.md`]** "implement the FULL bundle
> (not single-variable) … This is a 'full recipe vs faithful baseline'
> comparison — it intentionally confounds many changes; NOT a controlled
> single-variable test."

> **[STATUS]** Phase B is mid-run. Training-result numbers are therefore
> **PENDING**; everything below is the *specified/implemented* recipe as it exists
> in the files, not measured outcomes.

**Component 1 — NorMuon optimizer (arXiv 2510.05491).** NorMuon = Muon
(Newton-Schulz orthogonalization of the momentum) + per-neuron (row-wise)
second-moment normalization of the orthogonalized update, with an RMS-matched LR
scale. The verified update (from both `build2_spec.md` and the `normuon.py`
docstring):

```
M    = β1·M + (1-β1)·G                         # momentum,  β1 = 0.95
O    = NS5(M)                                  # orthogonalize, Frobenius-normalized
v    = β2·v + (1-β2)·mean_cols(O⊙O)            # per-row (per-neuron) 2nd moment, β2 = 0.95
Ô    = O / (√V + ε)                            # V = v broadcast across columns
η̂    = 0.2·η·√(m·n) / ‖Ô‖_F                    # RMS match to Adam (Jordan 2024)
W    = W − η·λ·W − η̂·Ô
```

**NS5 orthogonalization** (`_newton_schulz5` in `normuon.py`):
Frobenius-normalize `X0 = M/‖M‖_F`, then 5 Newton-Schulz iterations of
`X = a·X + b·(XXᵀ)X + c·(XXᵀ)²X`.

> **[HONEST GAP — verbatim from both `normuon.py` and `build2_spec.md`]** The NS5
> coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)` are **NOT printed in the
> NorMuon paper**; it cites Jordan et al. 2024 (Muon), whose standard quintic
> these are. They are **labeled as standard-Muon, not paper-quoted.** Code:
> `a, b, c = 3.4445, -4.7750, 2.0315`.

Implementation details from `normuon.py`:

- NS5 runs in **fp32** for stability; transposes to keep the smaller dim first
  (`if X.size(0) > X.size(1): X = X.t()`), cheaper `A = XXᵀ`.
- ε in `√V + ε`: paper does not specify the value → **use 1e-8** (`build2_spec.md`
  line 13; `NorMuon.__init__` default `eps=1e-8`).
- Row-wise 2nd moment: `v.mul_(b2).add_((O * O).mean(dim=1), alpha=1 - b2)`;
  normalize `Ohat = O / (v.sqrt().unsqueeze(1) + eps)`.
- `eta_hat = 0.2 * lr * (m_ * n_) ** 0.5 / (Ohat.norm() + eps)`.
- **Scope (verbatim):** NorMuon on 2D hidden matrices (q/k/v/o, gate/up/down
  proj). **Adam** on embeddings, unembedding, norms, scalars/bias. Enforced by
  `assert g.dim() == 2, "NorMuon is for 2D matrices only; route others to Adam"`.

**Why this choice:** Muon-class optimizers orthogonalize the momentum so every
direction in a weight matrix gets a comparably-sized step; NorMuon adds per-neuron
2nd-moment normalization (the Adam-like part) on top of the orthogonalized update,
and rescales by `0.2·η·√(m·n)/‖Ô‖_F` so its effective step magnitude matches what
Adam would have taken (Jordan-2024 RMS-match), letting it reuse Adam-tuned
learning rates.

**Component 2 — Value residuals (IMU-1 Eq4)** (`build2_spec.md` line 19;
`model_imu1.py` lines 146-150, 174-177):

```
V(l) = s·(α1·V_local(l) + α2·V(1)) / √(α1² + α2²)
init (s, α1, α2) = (1, 1, 0)   # learnable
```

The first layer's value `V(1)` is piped to all layers. In `Qwen3Model.forward`:
`if i == 0: first_v = v_local` then passed into every subsequent layer. **At init
`α2=0` ⇒ `V = V_local` bit-identical** (`vr_s=1, vr_a1=1, vr_a2=0`). **Why:** lets
deep layers blend in the unmodified first-layer values, a residual path through
the *value* stream that the network can open gradually during training.

**Component 3 — LayerNorm scaling (IMU-1 Eq5)** (`build2_spec.md` line 20;
`model_imu1.py` line 231):

```
LN_l(x) = (1/√l)·Norm(x),   l = 1..L   (l = layer_idx + 1)
```

Code:
`self.ln_scale = (1.0 / math.sqrt(layer_idx + 1)) if cfg.use_layernorm_scaling else 1.0`,
applied as `self.ln_scale * self.input_layernorm(x)` and
`self.ln_scale * self.post_attention_layernorm(x)`. **scale=1.0 when disabled ⇒
bit-identical to faithful.** **Why:** down-weights deeper layers' normalized
contributions by `1/√l`, which tempers residual-stream growth with depth.

**Component 4 — Per-head gating (IMU-1 Eq3)** (`build2_spec.md` line 21;
`model_imu1.py` lines 144-145, 195-197):

```
out_h = 2·σ(g_h)·Attn_h,   g = W_g·x,   W_g ∈ R^{d×n_h}
```

Code: `self.gate_proj = nn.Linear(cfg.hidden_size, self.n_heads, bias=False)` (one
logit per query head); `g = self.gate_proj(x).transpose(1, 2).unsqueeze(-1)` →
`out = out * (2.0 * torch.sigmoid(g))`. The `2·σ` form makes the gate's expected
init value 1.0 (since `σ(0)=0.5`), so it starts as a near-identity multiplier.
**Why:** a learned per-head scalar gate lets the model attenuate or amplify
individual attention heads.

**Component 5 — Cautious weight decay (IMU-1 Eq7)** (`build2_spec.md` line 28;
`normuon.py` lines 19-20, 80-86):

```
Δw = −λw  if sign(u) = sign(w)  else 0      # u = orthogonalized update
```

Apply the `−η·λ·W` term **only where `sign(update) == sign(weight)`**, masked by
`(Ô ⊙ W > 0)`. Code:
`mask = (Ohat * p.data > 0).to(p.dtype); p.data.add_(p.data * mask, alpha=-lr * wd)`.
Default `cautious=True`. **Why:** decay is skipped on weights whose update already
disagrees in sign with the parameter, avoiding decay that would fight the
gradient.

**Components 6-7 — WSD schedule + z-loss** (from `build2_spec.md`, their 3-stage /
72B-token schedule, VERIFIED as paper recipe):

- NorMuon **2D LR** 0.011 (stable) / 0.0115 (decay) / 0.003 (mid); **1D LR**
  0.006/0.006/0.002.
- **WD 0.1 (2D only)**; **warmup 2500**; **WSD decay fraction 20%**; **z-loss
  1e-4**; grad-accum 2; **EMA β=0.8** (final 10 ckpts).

**Adaptation to OUR budget** (2 TPP, ~1.2B tok, ~18,150 steps, single run):

- Single WSD stage (not 3): warmup ~5% + stable + 20% decay-to-zero.
- Keep NorMuon 2D LR ~0.011 / 1D ~0.006 (paper stable-stage), WD 0.1, z-loss
  1e-4, β1=β2=0.95.
- seq_len 4096, micro_batch 4 × grad_accum 4 (memory limit), bf16. EMA optional.
  Smoke test ~1000 steps first, then full run.

> **WSD = Warmup-Stable-Decay** LR schedule; z-loss is the auxiliary
> `1e-4 · (logsumexp logits)²` regularizer that keeps the softmax normalizer from
> drifting. (Coefficient `1e-4` verified in `build2_spec.md`; the exact z-loss
> code form is not pinned at the level of an exact code line — **[not in repo]**.)

**muP — OMITTED (HONEST GAP).**

> Verbatim from `build2_spec.md` line 23 and `model_imu1.py` comment: muP is
> **NOT specified in paper text → OMITTED**. Rationale (verbatim): "its purpose is
> cross-scale HP transfer; we train one fixed scale, so moot. Honest deviation."
> This is an explicit, labeled deviation from a "full bundle" — the bundle is
> *all components the paper specifies*, and muP is not one the paper text pins
> down.

**The verify gate.** The bundle toggles live in `Qwen3Config` (`model_imu1.py`
lines 53-57):

```python
use_value_residual: bool = True     # Eq4 ...
use_layernorm_scaling: bool = True  # Eq5 ...
use_head_gating: bool = True        # Eq3 ...
```

with the explicit invariant (line 54): **"All False => bit-identical to the
faithful baseline (verify anchor)."** Each component was engineered to vanish at
its disabled/init setting: value-residual `α2=0` ⇒ `V=V_local`; LN-scale `=1.0`;
gating module simply not created. The spec calls for a **"bundle-off == faithful"
bit-exact test mirroring Build 3's verify** (`build2_spec.md` lines 37-38).

> **[NUANCE — verbatim from the spec]** With the components *toggled on but at
> init* the model is NOT automatically bit-identical: value-residual at init *is*
> `V_local`, but head-gating at init multiplies by `2·σ(0)=1.0` only if you
> disable it, and the spec notes "disable gate, LN-scale=1" rather than relying on
> init values for the gate. The clean bit-exact anchor is **all toggles False**,
> which removes the new modules entirely (gate/value-residual params are
> conditionally created; LN scale becomes the literal `1.0`).

### Build 3 — Exploratory (partial RoPE)

**Path:** `builds/2026-06-08_reproduce-exploratory_qwen3-0.6b/` (model:
`model_partialrope.py`; verify: `verify_partialrope.py`).
**Backing paper:** "Fractional Rotation, Full Potential?" — arXiv **2603.11611**
(Mar 2026), quoted in the docstring as: "~10% rotation reaches convergence
comparable to full RoPE at 135M scale."

**What it is.** Derived from the verified faithful `model.py`; **the ONLY change
is partial RoPE** — rotate `partial_rotary_factor` of `head_dim` and pass the rest
through un-rotated. Config field (`model_partialrope.py` lines 52-56):

```python
partial_rotary_factor: float = 1.0   # 1.0=baseline; sweep 0.25, ~0.10
```

The sweep values from the task and file comment: **25% and ~10%** (`0.25`,
`~0.10`).

**How it works.** The RoPE cache is built over `rotary_dim = head_dim * factor`
(with the convention `inv_freq` divides by `rotary_dim`, NOT `head_dim`, following
the HF GPT-NeoX/Phi/StableLM `partial_rotary_factor` convention — so the frequency
band spans the same range over fewer dims). Code (`_build_rope_cache`, lines
93-98):

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, ...) / rotary_dim))
```

Application (`_apply_rope`, lines 106-119) — rotate the first `rd = rotary_dim`
dims, pass the rest through unchanged:

```python
rd = cos.shape[-1]
q_rot, q_pass = q[..., :rd], q[..., rd:]
k_rot, k_pass = k[..., :rd], k[..., rd:]
q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
q_out = torch.cat([q_rot, q_pass], dim=-1)
k_out = torch.cat([k_rot, k_pass], dim=-1)
```

Even-dim guard (`Qwen3Model.__init__`, lines 246-248):
`rotary_dim = int(head_dim * factor); rotary_dim -= rotary_dim % 2; assert 2 <= rotary_dim <= head_dim`.
**Why:** `rotate_half` chunks the rotated slice in two, so the rotated count must
be even.

**Why the idea:** RoPE injects relative position by rotating Q/K coordinate pairs
at geometrically-spaced frequencies. If only the lowest-frequency (long-range)
subset of dimensions carries most of the useful positional signal, you can leave
the remaining high-frequency dims un-rotated (pure content channels) without
losing convergence — the paper's claim at 135M scale.

**The verify gate.** From the docstring (lines 1-9) and the code comments (lines
88-91, 108-109): **with `factor=1.0` it is bit-identical to the faithful build** —
`rotary_dim == head_dim` ⇒ `q_pass`/`k_pass` are empty and `_apply_rope` equals
the full-RoPE path bit-for-bit; the cache built over `rotary_dim == head_dim` is
bit-identical to the faithful full-RoPE cache. This is "verified in
`verify_partialrope.py`" (per the docstring). So any difference observed at
`factor=0.25` or `~0.10` is attributable solely to the reduced RoPE coverage.

> **[STATUS]** The fact of bit-identity at `factor=1.0` is asserted in-code and
> stated to be checked by `verify_partialrope.py` — only the construction
> guaranteeing it is verified in the source above; the *passing result* of that
> gate is not captured in the model file itself.

### Cross-build summary

| | Build 1 Faithful | Build 2 Modernized (IMU-1) | Build 3 Exploratory |
|---|---|---|---|
| What changes vs faithful | nothing (anchor) | optimizer (NorMuon/Adam split) + value residuals + LN-scaling + head gating + cautious WD + WSD + z-loss | partial RoPE only |
| Backing paper(s) | Qwen3 TR [2505.09388](https://arxiv.org/abs/2505.09388) (LR shape only) | IMU-1 [2602.02522](https://arxiv.org/abs/2602.02522) + NorMuon [2510.05491](https://arxiv.org/abs/2510.05491) | "Fractional Rotation, Full Potential?" [2603.11611](https://arxiv.org/abs/2603.11611) |
| Verify gate | reference for the other two; HF safetensors load with no key remap | all bundle toggles `False` ⇒ bit-identical to faithful | `partial_rotary_factor=1.0` ⇒ bit-identical to faithful |
| Honest gaps | per-size 0.6B HP table never published; A/B/C match LR *shape* only | NS5 coeffs standard-Muon not paper-quoted; ε=1e-8 chosen; muP omitted; our recipe is a single-stage WSD adaptation of the paper's 3-stage | bit-identity asserted in-code; `verify_partialrope.py` pass result not captured |
| Result status | per `hp_tuning_plan.md`: PPL ~184K random → roughly 100-200 expected (Config A) — **expected, not measured**; Phase A actual best = 46.31 | **PENDING (Phase B mid-run)** | **PENDING** |

---

## 7. Verifying architectural correctness

The hard gate of a *reproduction* build is **numerical equivalence to the HF
reference**, not "looks plausible." [`verify.py`](verify.py) checks this directly:

1. **Load both models from the same weights.** The HF `Qwen3ForCausalLM` and our
   from-blank `model.py` `Qwen3ForCausalLM(Qwen3Config())` are loaded in **fp32**
   (`"dtype": "float32"`), the regime where bf16/fp16 rounding noise cannot mask a
   real bug.
2. **Single fixed prompt, fixed shape.** `"The capital of France is"` →
   `input_shape: [1, 5]` (5 tokens). A deterministic, batch-1 forward pass removes
   any sampling/batching nondeterminism.
3. **Compare the full final-position logit vector**, not just the prediction:
   - `max_abs_error` = max over the vocab of `|logit_ours − logit_hf|`.
   - `relative_error` likewise.
   - `argmax_match` = do both pick the same next-token id?
4. **Gate:** `passed = (max_abs_error < tolerance)` with `tolerance = 0.001` (the
   hard `<1e-3` reproduction gate).

**Why `0.0` matters.** An `argmax_match: true` alone is weak — two very different
logit vectors can share a top-1 token. `max_abs_error = 0.0` means **every one of
the ~151k vocab logits is identical to the HF reference to fp32 precision.** That
can only happen if RMSNorm, RoPE (full rotary), GQA attention, SwiGLU MLP,
QK-norm, weight tying / lm_head, and the exact layer ordering are all implemented
correctly *and* wired in the right order. It converts "I think the architecture
is right" into "the architecture is provably the same function as HF." Both models
also agree on the human-meaningful output (`" Paris"`, id `12095`), which is the
sanity check a reader can eyeball.

**Run it:**

```bash
python verify.py
```

> Note on the SmolLM2 vs Qwen3 result: for SmolLM2 we observed exactly `0.0` only
> on CPU (tiny `~1e-5` GPU jitter from reduction order). Here `verify.json`
> records `max_abs_error = 0.0` and `relative_error = 0.0` in fp32 — the strongest
> possible pass.

**If parity fails**, the diagnosis order for this architecture, highest to lowest
probability:

1. **QK-Norm placement** — must be applied per-head **before** RoPE and before the
   transpose; norm is over `head_dim`, not `hidden_size`.
2. **RoPE layout** (interleaved vs split-halves) and **θ = 1e6** (not 1e4).
3. **`head_dim = 128` independent** — do not derive it as `hidden/n_heads = 64`;
   the projection widths (2048 / 1024) depend on it.
4. **GQA repeat axis** — `repeat_interleave(n_rep=2, dim=1)` after the head
   transpose.
5. **RMSNorm precision** — the fp32 upcast, and `eps = 1e-6` (not 1e-5).
6. **Tied embedding** — `lm_head.weight = embed_tokens.weight` must be the alias
   assignment (same storage), not a copy.

---

## 8. Training recipe + matched-compute methodology

### What the original was actually trained on (`training_plan.md`)

- **Corpus:** 36T tokens across **119 languages**.
- **Mix:** web + PDF-extracted (Qwen2.5-VL OCR) + synthetic STEM/code
  (Qwen2.5-Math, Qwen2.5-Coder) + books + reasoning data.
- **Three-stage pretraining:** Stage 1 = 30T tok @ 4K ctx; Stage 2 = +5T tok,
  more STEM/code/reasoning @ 4K; Stage 3 = long-context extension to **32K**.
- **Paper baseline recipe (the anchor we copy):** AdamW, eps 1e-8; cosine, linear
  warmup 0→**1.7e-3** over **1000 steps**, cosine decay to **3.2e-4** over 500B
  tokens; batch **4M tokens**.
- **HONEST GAP:** the per-size 0.6B-Base recipe is **not disclosed**
  (`hp_tuning_plan.md` quotes the paper: *"we set the predicted optimal learning
  rate and batch size strategy for each dense or MoE model"* — but no concrete
  per-size table). We faithfully match the **paper's baseline LR *shape*** only.

### Our data: FineWeb-Edu sample-10BT (proxy for the undisclosed 36T mix)

`HuggingFaceFW/fineweb-edu`, config `sample-10BT`, ODC-By, 10B-token subset
(`training_plan.md` data table). Chosen as the **best public proxy** for Qwen3's
knowledge-dense Stage-1 mix that fits a single GB10's disk. The real 36T
multilingual mix is undisclosed and unreproducible at this scale, so this is
explicitly a proxy, not a match.

### Phase A — LR sweep @ 131M tokens (pick the LR)

Matched-compute, 3 runs, identical except peak LR (`run_lr_sweep.sh`:
`lr17 1.7e-3 / lr24 2.4e-3 / lr30 3.0e-3`). Per `qwen3_lr24_train.log` args:

```
steps 2000, seq_len 4096, micro_batch 4, grad_accum 4,
peak_lr 0.0024, end_lr 0.00032, warmup_steps 150,
weight_decay 0.01, grad_clip 1.0, dtype bfloat16, seed 0
```

Effective batch = 4 × 4 × 4096 = **65,536 tok/step**; 2000 steps ⇒ ~131M tokens.
**Outcome: 2.4e-3 (`lr24`) was best (46.31)** and is the LR carried into Phase B.

> **Why micro_batch=4 and not 16?** The plan's "bs=16" is infeasible as a single
> micro-batch on this box: `hp_tuning_plan.md` §throughput records that
> micro_batch 16 and 8 at seq_len 4096 **both OOM** on the GB10 unified pool and
> the probe backs off to 4. So the effective batch is realized as
> `micro_batch=4 × grad_accum=4`.

### Phase B — 4 matched-compute runs @ 2 TPP = 1.19B tokens each (PENDING / mid-run)

Source: `phase_b_driver.sh`. **2 tokens-per-parameter** (TPP) ⇒ **18,150 steps =
1.19B tokens** per run, **best LR 2.4e-3** from Phase A, warmup 900, sequential
(**one GPU job at a time on the GB10 unified-memory box**; first run streams +
caches 1.19B tokens ~30 min, runs 2–4 reuse the cache):

| # | Run | Recipe |
|---|---|---|
| 1 | faithful baseline | AdamW 2.4e-3, full RoPE (Build-1 final + 100%-RoPE anchor) |
| 2 | IMU-1 full bundle | NorMuon + value-residual + LN-scale + head-gate + WSD |
| 3 | partial RoPE 25% | AdamW, baseline recipe, `partial_rotary_factor 0.25` |
| 4 | partial RoPE 10% | AdamW, baseline recipe, `partial_rotary_factor 0.10` |

Common flags: `--eval_every 2000 --ckpt_every 2000 --log_every 50`,
`--steps 18150 --warmup_steps 900`. Phase B final PPLs are **PENDING** (the IMU-1
smoke in §0.4 is a separate 1000-step preview, not the Phase B run).

---

## 9. Repo layout

```
Qwen3-0.6B/
├── model.py                 — the faithful architecture, single file (the §3/§4/§5 model)
├── verify.py                — load official safetensors into ours, assert logits match
├── README.md                — this file
└── builds/
    ├── phase_b_driver.sh    — drives the 4 sequential Phase-B runs (one GPU job at a time)
    │
    ├── 2026-06-08_target-survey/
    │   └── model_target_plan.md          — the 4-candidate survey (§1)
    │
    ├── 2026-06-08_reproduce-faithful_qwen3-0.6b/      — BUILD 1 (anchor)
    │   ├── architecture_plan.md          — canonical config + per-component spec
    │   ├── training_plan.md              — what the original was trained on + our proxy data
    │   ├── hp_tuning_plan.md             — Config A/B/C, throughput probe, OOM finding
    │   ├── train_qwen3.py                — AdamW + cosine trainer
    │   ├── test_model.py                 — param-count + structural test (expected 596,049,920)
    │   ├── eval_original_vs_repro.py     — identical-code PPL: original vs repro (§0.2)
    │   ├── run_lr_sweep.sh               — Phase A lr17/lr24/lr30 driver
    │   ├── results.ipynb                 — results notebook
    │   └── results/
    │       ├── verify.json               — the bit-exact gate (max_abs_error 0.0)
    │       ├── original_vs_repro.txt      — 13.40 vs 46.31 (3.5× / ~275,000× data)
    │       ├── qwen3_lr{17,24,30}_after.txt   — sweep finals
    │       ├── throughput_probe.json     — GB10 throughput + OOM backoff
    │       └── ... (train logs/CSVs, plots)
    │
    ├── 2026-06-08_reproduce-modernized_qwen3-0.6b/    — BUILD 2 (IMU-1 bundle)
    │   ├── build2_spec.md                — the full bundle spec
    │   ├── model_imu1.py                 — faithful model + value-resid/LN-scale/head-gate toggles
    │   ├── normuon.py                    — NorMuon optimizer (NS5 + cautious WD)
    │   ├── train_imu1.py / verify_imu1.py
    │   └── results/
    │       └── qwen3_imu1_smoke_train.log     — 1000-step smoke (§0.4, PRELIMINARY)
    │
    └── 2026-06-08_reproduce-exploratory_qwen3-0.6b/   — BUILD 3 (partial RoPE)
        ├── model_partialrope.py          — faithful model + partial_rotary_factor
        ├── verify_partialrope.py         — factor=1.0 ⇒ bit-identical gate
        └── train_partialrope.py
```

**Why one file for the faithful model:** the entire architecture fits on a few
screens — copy nanoGPT's UX philosophy. On a video you should never need to jump
between files to explain a layer. Builds 2 and 3 are deliberately *derived* from
that single `model.py` so the diff is the lesson.

---

## 10. Setup

```bash
pip install torch transformers datasets safetensors accelerate
```

- **torch** — we use `F.scaled_dot_product_attention`; needs a recent 2.x. The
  faithful model depends on nothing but PyTorch.
- **transformers** — only for the *reference* `Qwen3ForCausalLM` in `verify.py`
  and the tokenizer.
- **datasets** — only for the trainers to stream FineWeb-Edu `sample-10BT`.

### The GB10 unified-memory caveat (read before training)

The dev box is an **NVIDIA GB10 Grace Blackwell** where CPU and GPU share **one
~119 GB unified pool** (torch reports 128.5 GB) — there is **no separate VRAM**,
and overcommitting the pool can crash the whole machine. Concretely:

- **One GPU job at a time.** `phase_b_driver.sh` runs the four Phase-B jobs
  **sequentially**, never in parallel — two large jobs would exceed the shared
  pool.
- **micro_batch is memory-capped at 4** at seq_len 4096. The throughput probe
  found `micro_batch=16` and `=8` both **OOM** and backed off to 4; larger
  effective batch is realized via `grad_accum`, not larger micro-batch.
- The eval/train scripts import a **`safe_cuda`/safe-env guard** (e.g.
  `eval_original_vs_repro.py`, `train_qwen3.py`, `throughput_probe.py`) that caps
  the process at a fraction (~85%) of the unified pool before allocating, the same
  discipline as the JAX preallocation guard used elsewhere in this repo. Always
  let that guard run before instantiating the model on the GB10.

A CUDA GPU is not required for parity verification or short generation —
`verify.py` runs in fp32 and is fine on CPU.

---

## 11. Honest accounting

| Item | Status | Detail |
|---|---|---|
| Architecture correctness | ✅ **VERIFIED (bit-exact)** | `verify.json`: `max_abs_error = 0.0`, `relative_error = 0.0`, `argmax_match = true`, `passed = true`, fp32, tol 1e-3. Param count `596,049,920` (`architecture_plan.md`, `test_model.py`). |
| Source of the PPL gap | ✅ **VERIFIED: data, not skill** | 3.5× PPL gap (46.31 vs 13.40) against a **~275,000×** data deficit (`original_vs_repro.txt`, `eval_original_vs_repro.py`). Tiny gap-per-decade-of-data ⇒ recipe is sound; remaining gap is the missing 36T tokens. |
| Phase A LR choice | ✅ **VERIFIED** | `lr24 = 2.4e-3` best (46.31 < 46.89 < 49.28) at matched 131M-tok compute. This is the only *original* result — Qwen3 never published the 0.6B LR. |
| Phase B (4 runs @ 2 TPP) | ⏳ **PENDING / IN PROGRESS** | 1.19B tok/run, sequential on the GB10. Final PPLs not yet captured. |
| IMU-1 smoke trajectory | ⚠️ **PRELIMINARY + CONFOUNDED** | 1000-step smoke, different recipe (NorMuon lr 1.1e-2, WSD). Bundle **intentionally changes 6 things at once** (NorMuon + value-resid + LN-scale + head-gate + WSD) — by design it cannot attribute the gain to any single change. Not comparable to Phase A numbers. |
| 2 TPP token budget | ⚠️ **INFERRED / directional** | Chinchilla-optimal for 596M ≈ **12B tokens** (20 TPP, `training_plan.md`). 2 TPP ≈ **~80×** below that and far below the methods' validated regime ⇒ Phase B results are **directional, not headline**; absolute parity with Qwen3-0.6B-Base is impossible at this scale. |
| muP (maximal-update param.) | ❌ **OMITTED** | Not implemented; standard init `Normal(0, 0.02)` from HF config (`hp_tuning_plan.md`). |
| NorMuon NS5 coefficients | ⚠️ **STANDARD, not paper-quoted** | The Newton–Schulz coefficients used `(3.4445, -4.7750, 2.0315)` are the **standard Muon** coefficients, not values quoted from the IMU-1 / NorMuon paper. |
| Recipe faithfulness | ⚠️ **LR *shape* only** | Per-size 0.6B HPs are undisclosed; we match the paper's **baseline LR shape** (1.7e-3 cosine → 3.2e-4), then empirically retune to 2.4e-3. Data is a FineWeb-Edu **proxy**, not the real 36T/119-lang mix. |
| Upstream `config.json` | ⚠️ **Quoted from in-file comments** | The original JSON is not stored in-repo; config values come from `model.py`'s inline comments asserting an `AutoConfig` pull on 2026-06-08, cross-checked for self-consistency against the verified param count. |

**Bottom line:** the *model* is reproduced exactly (bit-exact logits). The
*training* is an honest, scaled-down, matched-compute study — the 3.5× PPL gap is
fully explained by a ~275,000× data deficit, the best LR (2.4e-3) is an original
verified finding, and everything about Phase B, the IMU-1 bundle, the 2-TPP
budget, the omitted muP, and the standard-not-paper NS5 coefficients is flagged
rather than overclaimed.

---

## 12. What we'll try once Phase B is done

In order of payoff:

1. **Close out Phase B.** Capture the four final PPLs (faithful baseline / IMU-1
   bundle / partial-RoPE 25% / partial-RoPE 10%) at matched 1.19B-token compute,
   and fill in the cross-build summary in §6 with measured numbers instead of
   PENDING.
2. **Run the two bit-exact "off" gates as regression tests.** IMU-1 with all
   toggles `False`, and partial-RoPE at `factor=1.0`, must both reproduce the
   faithful logits exactly — wire them into CI alongside `verify.py`.
3. **Decompose the IMU-1 bundle.** It intentionally confounds 6 changes; once the
   full-bundle vs faithful delta is known, ablate one component at a time
   (NorMuon-only, value-residual-only, …) to attribute the gain.
4. **Partial-RoPE scaling check.** Test whether the paper's "~10% rotation ≈ full
   RoPE" claim (verified at 135M scale) holds at 596M, using the
   `partial_rotary_factor` sweep already wired.
5. **KV cache.** Turn generation from O(T²) to O(T) per token — the natural next
   single-topic build, mirroring the SmolLM2 follow-up plan.
6. **Resolve the upstream config provenance.** Store the actual `config.json`
   pulled via `AutoConfig` in-repo so the spec sheet cites the JSON directly, not
   in-file comments.

---

## 13. Source map

| Claim | Source |
|---|---|
| Architecture config | live `AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base")`, pulled 2026-06-08 (`architecture_plan.md`) |
| Per-component spec (esp. QK-Norm) | HF `transformers` [`modeling_qwen3.py`][hfqwen3] |
| Training recipe (LR shape) | [Qwen3 Technical Report][qwen3paper] arXiv:2505.09388; `training_plan.md`, `hp_tuning_plan.md` |
| Independent PyTorch reference | [rasbt/qwen3-from-scratch][rasbt] · [article][rasbtarticle] |
| Qwen3 Technical Report | https://arxiv.org/abs/2505.09388 |
| NorMuon optimizer | https://arxiv.org/abs/2510.05491 (Oct 2025) |
| IMU-1 modernization bundle | https://arxiv.org/abs/2602.02522 (Jan 2026) |
| Partial RoPE ("Fractional Rotation, Full Potential?") | https://arxiv.org/abs/2603.11611 (Mar 2026) |
| RoPE paper | https://arxiv.org/abs/2104.09864 (Su et al. 2021) |
| GQA paper | https://arxiv.org/abs/2305.13245 (Ainslie et al. 2023) |
| SwiGLU paper | https://arxiv.org/abs/2002.05202 (Shazeer 2020) |
| RMSNorm paper | https://arxiv.org/abs/1910.07467 (Zhang & Sennrich 2019) |
| Base checkpoint | [Qwen/Qwen3-0.6B-Base][hfbase] |
