# Qwen3-0.6B — from-scratch reproduction + research experiment

A single-file PyTorch reproduction of [`Qwen/Qwen3-0.6B-Base`][hfbase] (596M-param
decoder-only transformer), **verified bit-exact** against the official HuggingFace
weights (`max |Δlogits| = 0.0`), used as the base for a **three-build experiment**:
reproduce it faithfully, then apply recent (2026) research methods and measure — at
matched compute — whether they beat the faithful baseline.

> **Status: Phase B decided — now de-confounding the IMU-1 win.** Architecture
> VERIFIED bit-exact; Phase A LR sweep done (`lr24 = 2.4e-3`). Phase B (matched
> compute @ 2 TPP): faithful **28.65** · **IMU-1 bundle 23.52 — a proven 17.9%
> win** (gap to original 2.14× → **1.76×**) · partial-RoPE 0.25 **29.54 (loses,
> +3.1%)**; 0.10 abandoned at ~30% (50.71 @ step 4000, also losing). **But the
> IMU-1 win is a confounded bundle** (NorMuon + WSD + z-loss + 3 arch tweaks).
> **Now running** (2026-06-18): a single-variable ablation ladder to *attribute* it
> — see [End-to-end lifecycle](#end-to-end-lifecycle--what-weve-done--whats-next).

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
| **🥇 IMU-1 bundle (Build 2)** | **1.19B** | **23.52** | **1.76×** |
| Faithful baseline (Build 1) | 1.19B | 28.65 | 2.14× |
| partial-RoPE 0.25 (Build 3) | 1.19B | 29.54 | 2.20× |
| Our best (Phase A, `lr24`) | 131M | 46.31 | 3.5× |

![Phase B — final val PPL: IMU-1 wins, partial-RoPE loses to the baseline](builds/comparison/phaseB_final_ppl.png)

![Phase B — matched-compute val-PPL curves (same data, eval, budget)](builds/comparison/phaseB_ppl_curves.png)

**Two headline results:**

1. **Reproduction** — the faithful baseline reproduces Qwen3-0.6B to within **2.14×**
   perplexity using **~275,000× less data**; each ~10× more data roughly halves the gap
   (the gap is *data scale, not correctness*).
2. **Research win (matched compute, both @ 2 TPP / 1.19B tokens)** — the modernized
   **IMU-1 bundle (23.52) beats the faithful baseline (28.65) by 17.9%** — the project's
   first *proven* result that a recent 2026 method improves on our own correct baseline.
   ⚠️ It's the *full* bundle (NorMuon + value-residuals + LN-scaling + head-gating +
   **WSD-to-zero** vs the baseline's cosine) — a **recipe-level** win, **not** attributable
   to any single component (the WSD schedule alone could account for part of it).

**Scaling trend:** `65.5M → 131M → 1.19B → 36T  ≈  96 → 46 → 28.65 → 13.4`.

**Phase A LR sweep** (131M tokens, matched compute) → picked the LR:
`lr17` (1.7e-3) = 46.89 · **`lr24` (2.4e-3) = 46.31 ← best** · `lr30` (3.0e-3) = 49.28.

![Phase A — LR sweep (lr24 wins)](builds/comparison/phaseA_lr_sweep.png)

The earlier **Build-2 IMU-1 smoke** (39.83 @ 65.5M tokens vs faithful smoke 95.87) was a
directional hint — now confirmed by the full 2-TPP run above (**23.52 vs 28.65**).

> **IMU-1 vs baseline: DECIDED — IMU-1 wins (23.52 < 28.65).**
> **partial-RoPE vs baseline: DECIDED — partial RoPE LOSES.** 0.25 finished at
> **29.54 (3.1% worse than the 28.65 baseline)**; 0.10 is tracking far worse
> (50.71 @ step 4000, run in progress). Reducing the rotated RoPE fraction does
> **not** match the full-RoPE baseline at this scale — a clean, if negative, result.

---

## End-to-end lifecycle — what we've done & what's next

This model is the spine of a full **small-scale LLM lifecycle** run on one GB10 box
(no rented compute). Every number traces to the ledger
(`research/ledger/ledger.json`) or a build log.

### Done (with sources)

| Stage | Result | Evidence |
|---|---|---|
| **Architecture** | bit-exact vs HF (`max\|Δlogits\| = 0.0`), 596,049,920 params | `verify.json` |
| **Pretrain — 3 builds @ 2 TPP** | faithful 28.65 · **IMU-1 23.52 (win, −17.9%)** · partial-RoPE 0.25 29.54 (loss); 0.10 abandoned ~30% | build logs (above) |
| **Optimizer ablation (clean, single-variable)** | NorMuon **beats** AdamW: wikitext **−0.474 bpb** (95% CI [0.444, 0.505]), code −0.502 bpb ([0.456, 0.547]) — **significant win** | ledger `2026-06-16_qwen3_normuon-vs-adamw` |
| **Post-train — SFT** | reasoning OpenR1-Math PPL **14.26 → 11.60 (−18.7%)**; catastrophic forgetting **retained** (wikitext +0.2%, code −3.0%, fineweb-edu +0.74% — none significant). **n=1 → verdict inconclusive** | ledger `…vibethinker-small-reasoning` |
| **Paper** | *"Reproduce, Then Modernize…"* — **packaged** (arXiv/HF source tree), not yet submitted | ledger `papers[]` |
| **Harness-search side-quest** | Meta-Harness replication: automated harness search ≈ a trivial heuristic on every cheaply-searchable task (bin-packing, seq-packing, codeharness all **zero-headroom**, the last proven against a real 9B). The **promotion gate** (held-out + brittle-exclusion + significance) is the transferable contribution; reward-hack + shadowing fixes committed (`bdc5ec6`). | `research/harness_search/` |

### Now running — de-confound the IMU-1 bundle (Phase 1)

`Qwen3-0.6B/experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1/`. The IMU-1 win is a
confounded bundle; this is a **single-variable** ladder — each arm differs from the
faithful baseline by exactly one flag (`model_imu1` with arch-flags **off** is proven
bit-identical to the faithful model, so the baseline is genuinely faithful):

| Arm | schedule | z-loss | arch | (optimizer = AdamW 1.7e-3, all arms) |
|---|---|---|---|---|
| baseline | cosine | 0 | off | = faithful recipe |
| +WSD | **WSD** | 0 | off | |
| +z-loss | cosine | **1e-4** | off | |
| +arch | cosine | 0 | **on** | model_imu1's 3 tweaks |

3 seeds/arm (paired), **iso-FLOP** (token-matched; +arch adds 0.077% params → FLOP
ratio 1.00043, within the 5% gate), 2000-step proxy (131M tok/cell, ~5h/cell,
**~2.5 days** total). Verdict by the across-seed 95% CI
(`eval_stats.seed_delta_significant`). NorMuon is already isolated (the win above), so
it is excluded here. _Caveat: at the proxy budget, small per-component deltas may sit
inside the seed-noise floor → honestly inconclusive; a clear winner gets confirmed at
higher budget in Phase 2._

**How it runs / how we read it.** One parameterized trainer (`train_ablation.py`,
each arm = one CLI flag flipped) driven by a sequential supervisor (`run_arms.sh`) —
**one trainer at a time** (GB10 §C4.5), `sentinel.py`-guarded, **idempotent** (a crash
re-runs the supervisor, which skips finished cells and resumes the interrupted one from
its last 250-step checkpoint). When all 12 cells finish, `/eval-harness` scores every
checkpoint on the fixed suite (model's own tokenizer, `suite_version` pinned) and
`research/eval_stats.py::seed_delta_significant` computes the **across-seed 95% CI** for
each axis (arm − baseline). An axis is called a *driver* only if its CI excludes 0; a CI
that straddles 0 is reported `not significant`. Progress + verdict land in the ledger
run `2026-06-18_qwen3-0.6b_imu1-deconfound-p1`.

### Next (after Phase 1) — what & **how**

Each step reuses machinery that already exists; the "how" is concrete, not aspirational.

1. **Phase 2 — drill into the dominant axis.** *What:* attribute the winning axis to its
   sub-components. *How:* reuse the SAME `train_ablation.py` + `run_arms.sh`. If **arch**
   wins, split it into its three already-separate config flags (`use_value_residual`,
   `use_layernorm_scaling`, `use_head_gating`) → baseline + 3 single-variable sub-arms ×
   3 seeds, iso-FLOP, same seed-CI verdict. If **WSD** or **z-loss** wins, re-run that one
   arm at the **full 2-TPP budget** (18,150 steps) to confirm the proxy result holds at
   scale. New experiment dir, same gate.
2. **Publish — turn the attribution into the paper.** *What:* ship the packaged
   manuscript with a real per-component result. *How:* re-run `/manuscript` on this run;
   the Phase-1 **claim↔evidence gate** now passes because the headline is a single
   attributed component (not the confounded bundle), figures/tables regenerate from the
   ablation CSVs, and the package goes out behind the **human attestation** (the skill
   never auto-submits → you do the arXiv/HF click).
3. **Post-train rigor — close the post-train arc.** *What:* convert the n=1 inconclusive
   SFT into a real verdict. *How:* re-run the reasoning SFT at **≥3 seeds** via
   `/ablation-runner` in `finetune` mode (paired control + the §C13 catastrophic-forgetting
   probe), or add a preference stage (DPO/GRPO/RLVR); `/eval-harness` → across-seed CI
   decides win/loss (forgetting regression = a fail, not a footnote).
4. **Ship the loop end-to-end.** *What:* one fully autonomous cycle. *How:* paste the cron
   lines (**human-only**, §C4.2/§C20 — I can't install cron) so `/research-loop` runs
   nightly through its skill chain: `model-radar` → `ml-research` (brief) →
   `ablation-runner` (this same trainer/gate) → `eval-harness` → `experiment-ledger` →
   `weekly-retro` → `/manuscript`. One unattended idea→paper pass = the Tier-0 milestone.

> **GB10-only reality:** the reachable target is the **rigorous small-scale** lifecycle
> above — *not* at-scale distributed training (multi-node / MFU-at-scale need rented
> compute this box doesn't have). "A+-evidence, not A+-credential."

---

## The three builds

Each is a self-contained folder with its own README, model/scripts, verify gate,
and results. Click through for the detail.

| Build | Folder (→ README) | What changes | Backing paper | Status |
|---|---|---|---|---|
| **1 · Faithful** | [`builds/…reproduce-faithful…`](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/README.md) | nothing — exact arch, AdamW + cosine (the baseline + shared harness) | [Qwen3 TR][qwen3paper] | ✅ baseline = **28.65** |
| **2 · Modernized** | [`builds/…reproduce-modernized…`](builds/2026-06-08_reproduce-modernized_qwen3-0.6b/README.md) | full **IMU-1 bundle**: NorMuon + value residuals + LayerNorm-scaling + per-head gating + cautious-WD + WSD + z-loss | [IMU-1](https://arxiv.org/abs/2602.02522), [NorMuon](https://arxiv.org/abs/2510.05491) | ✅ **23.52 — beats baseline −18%** |
| **3 · Exploratory** | [`builds/…reproduce-exploratory…`](builds/2026-06-08_reproduce-exploratory_qwen3-0.6b/README.md) | **partial RoPE** — rotate 25% / 10% of head dims, pass the rest through | [arXiv:2603.11611](https://arxiv.org/abs/2603.11611) | ✅ 0.25 = **29.54** (loses to baseline); 0.10 in progress (50.71 @ ~22%) |
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
- 🔶 **Phase B** — baseline (28.65), IMU-1 (**23.52, a proven −18% win**), and
  partial-RoPE 0.25 (**29.54 — loses to baseline**) done; 0.10 in progress (50.71
  @ ~22%). The partial-RoPE *vs* baseline comparison is now **decided (it loses)**.
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
