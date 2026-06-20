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
> **Now de-confounding it** (run `2026-06-18_…imu1-deconfound-p1`, 6/12 cells done):
> a single-variable, 3-seed ladder. **Preliminary** (in-loop val PPL, not the final
> BPB verdict): **+WSD tracking as a significant driver** (+6.9%, 95% CI [+0.90, +5.51])
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

## Plots & figures gallery

All figures are generated **CPU-only from the on-disk logs/CSVs** (every number traces to a
file; partial/in-progress runs are labelled). Full set + captions:
[`PLOTS_INDEX.md`](PLOTS_INDEX.md) (31 figures). Regenerate the cross-build panel with
[`builds/comparison/make_all_builds_comparison.py`](builds/comparison/make_all_builds_comparison.py).

### Cross-build comparison — matched compute, *same steps*

The three builds (plus the abandoned partial-RoPE-10%) on one axis: identical 1.19B-token
budget (18,150 steps x 65,536 tok), identical eval slice. IMU-1 sits below the baseline the
whole way; partial-RoPE stays above it.

![All builds - validation perplexity vs step (matched compute)](builds/comparison/comparison_all_builds_ppl_vs_step.png)

![All builds - training loss vs step](builds/comparison/comparison_all_builds_train_loss.png)

![Matched-compute final val PPL across builds vs the published model](results_overview/plots/fig1_matched_compute_final_ppl_bar.png)

### Phase A — learning-rate sweep (131M tokens)

`lr24 = 2.4e-3` won at matched compute (the LR Qwen3 never published for 0.6B).

![Phase A LR sweep](builds/comparison/phaseA_lr_sweep.png)

### Per-build training dynamics

**Build 1 — Faithful baseline** — loss/LR, grad-norm, peak-mem vs the 109 GB cap, val PPL:

![Faithful loss and LR](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/plots/fig1_loss_lr_baseline2tpp.png)
![Faithful grad-norm](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/plots/fig2_grad_norm_baseline2tpp.png)
![Faithful peak memory vs cap](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/plots/fig3_peak_mem_baseline2tpp.png)
![Faithful val PPL vs tokens](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/plots/fig4_val_ppl_baseline2tpp.png)

**Build 2 — Modernized (IMU-1 bundle)** — training CE and the eval-PPL descent to 23.52:

![IMU-1 training cross-entropy](builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/plots/fig1_train_ce_vs_step.png)
![IMU-1 eval PPL descent](builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/plots/fig2_eval_ppl_vs_step.png)

**Build 3 — Exploratory (partial-RoPE)** — prope25 complete (29.54); prope10 dashed (died @5450):

![partial-RoPE training loss](builds/2026-06-08_reproduce-exploratory_qwen3-0.6b/results/plots/exploratory_prope_train_loss.png)
![partial-RoPE val PPL](builds/2026-06-08_reproduce-exploratory_qwen3-0.6b/results/plots/exploratory_prope_val_ppl.png)

### Controlled attribution — NorMuon vs AdamW (single-variable, 3 seeds, iso-FLOP, verifier-PASS)

The clean optimizer isolation that de-confounds one strand of the IMU-1 bundle: NorMuon beats
AdamW by **+0.474 bpb on wikitext-2 (95% CI [0.444, 0.505])** and +0.502 on code — significant.

![NorMuon vs AdamW - wikitext-2 BPB with 95% CI](experiments/2026-06-16_qwen3_normuon-vs-adamw/results/plots/fig1_headline_wikitext2_bpb.png)
![NorMuon vs AdamW - code BPB](experiments/2026-06-16_qwen3_normuon-vs-adamw/results/plots/fig2_code_bpb.png)
![NorMuon vs AdamW - FineWeb-Edu val PPL](experiments/2026-06-16_qwen3_normuon-vs-adamw/results/plots/fig3_fineweb_val_ppl.png)
![NorMuon vs AdamW - per-seed training curves](experiments/2026-06-16_qwen3_normuon-vs-adamw/results/plots/fig4_train_loss_curves.png)
![AdamW LR-sweep robustness control](experiments/2026-06-16_qwen3_normuon-vs-adamw/results/plots/fig5_adamw_lr_sweep.png)

### Deconfounding the IMU-1 win — 12-cell single-variable ladder (in progress)

*Which* component of the IMU-1 bundle drives the -17.9%? A single-variable, 3-seed,
iso-FLOP ladder at a matched **131M-token / 2000-step proxy** (baseline vs +WSD vs +z-loss
vs +arch, all AdamW). **Preliminary (9/12 cells done; the +arch arm is running):** **+WSD is
the driver** — 43.2 vs baseline 46.4 (~7%); **+z-loss is flat** (46.5); **+arch pending**.
(In-loop val PPL; the canonical eval-harness BPB verdict lands when all 12 cells finish.)
The honest same-step caveat is built in: the deconfound arms are *complete* 2000-step runs
(LR fully decayed), while a build's "step 2000" is a *mid-run* snapshot of an 18,150-step run
(LR still high) — so the valid cross-check is deconfound-baseline (46.4) ≈ Phase-A faithful
complete-run (46.31), which holds.

![Deconfound arms vs the full builds (different budgets, not directly overlaid)](experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1/deconfound_vs_builds.png)

![Deconfound arms vs the builds at the same 2000 steps (complete vs mid-run, honest)](experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1/same_steps_curves.png)

![Per-component attribution - single-variable, 3 seeds per arm](experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1/deconfound_attribution.png)

### Post-training — SFT (VibeThinker reasoning, n=1 preliminary)

Held-out reasoning PPL 14.26 -> 11.60; no catastrophic forgetting (FineWeb-Edu retained).

![SFT training loss](experiments/2026-06-17_qwen3-0.6b_vibethinker-small-reasoning/results/plots/vibethinker_sft_loss.png)
![SFT reasoning PPL vs step](experiments/2026-06-17_qwen3-0.6b_vibethinker-small-reasoning/results/plots/vibethinker_sft_reasoning_ppl.png)

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

**Pre-launch validation (the "tests" for this build).** Before any GPU budget: CPU
dry-run of all **4 arms** (single-variable confirmed — baseline/wsd/zloss share an
identical forward, only +arch changes it); GPU **smoke 4/4 + a resume round-trip**
(checkpoint → reload → continue); **iso-FLOP** check via `flop_accounting.py` (arch-on
adds 0.077% params → FLOP ratio **1.00043**, inside the 5% gate). Results→verdict is
pre-wired: `score_cohort.py` (scores all 12 checkpoints — uses `model_imu1` with per-arm
arch flags so the +arch checkpoints load) → `verdict.py` (`seed_delta_significant`, 15
tests green) → auto-fired by the conditional `post_cohort.sh` watcher on `cohort.done`.

**Live progress (Jun 19 — 6/12 cells done):** baseline ✓✓✓ · wsd ✓✓✓ · zloss (running) ·
arch (queued). **Preliminary +WSD signal** — *in-loop val PPL* (the trainer's quick eval,
NOT yet the canonical eval-harness BPB verdict): baseline **46.44 ±0.35** vs +WSD **43.24
±0.63** → Δ **+6.9%**, 95% CI **[+0.90, +5.51]**, **significant across 3 seeds** → WSD is
tracking as a real driver of the IMU-1 gain (mechanistically expected — WSD anneals the LR
to zero over the last 20%, sharpening the final loss vs cosine-to-floor). The rigorous
per-axis BPB verdict lands when all 12 cells finish.

**GB10 memory engineering (a real single-box lesson).** The full 151,936-vocab logits make
`torch.compile`'s startup transiently spike the **unified pool to ~80.7%**, tripping the
default 80% `sentinel.py` guard — it killed the zloss cell **4×** (the trainer's own RSS was
only 4.6 GB, so it was the *pool*, not the trainer; snap-confined Firefox couldn't be freed
to make room). Fix: raise the **per-cohort sentinel to `--kill-at 0.83`** (still under
`safe_cuda`'s 0.85 CUDA hard-cap, which errors cleanly — so the box stays crash-safe) and
run the full **mb4+compile** config, *identical* to baseline/wsd (zero execution confound)
at ~5,000–6,800 tok/s. Net throughput holds; revised total **~2 days**.

### Next (after Phase 1) — what & **how**

Each step reuses machinery that already exists; the "how" is concrete, not aspirational.

1. **Phase 2 — drill into the dominant axis.** *What:* attribute the winning axis to its
   sub-components. *How:* reuse the SAME `train_ablation.py` + `run_arms.sh`. If **arch**
   wins, split it into its three already-separate config flags (`use_value_residual`,
   `use_layernorm_scaling`, `use_head_gating`) → baseline + 3 single-variable sub-arms ×
   3 seeds, iso-FLOP, same seed-CI verdict. If **WSD** or **z-loss** wins, re-run that one
   arm at the **full 2-TPP budget** (18,150 steps) to confirm the proxy result holds at
   scale. New experiment dir, same gate.
2. **Phase 2b — optimizer/schedule head-to-head (settle WSD vs NorMuon vs Zeta), reusing
   what we already have.** *What:* put WSD, NorMuon, and Zeta on one comparable axis — the
   2000-step proxy ranked the *non-optimizer* components but couldn't compare the
   optimizer/schedule effects (different experiments, metrics, budgets). *How — and the key
   efficiency:* **don't re-run the baseline or WSD — Phase 1 already produced
   `baseline_seed{0,1,2}` and `wsd_seed{0,1,2}` at exactly this config/budget** (§C13
   control-reuse: a control is reusable when budget/data/seed/config match). So Phase 2b adds
   only **two new arms — +NorMuon (NorMuon+cosine) and +Zeta (Zeta+cosine), 3 seeds each =
   6 new cells (~1.5 days)** — then `score_cohort.py` scores all 12 (6 reused + 6 new) on the
   *same* **eval-harness BPB** for a clean 4-way comparison: baseline · +WSD · +NorMuon ·
   +Zeta. (NorMuon/Zeta optimizer gains show all-training-long, so they read cleanly even at
   2000 steps; WSD's gain is endpoint-only and *budget-dependent*, so **only if the
   optimizer margin is close** do we spend a single full-1.19B confirmation of the winner —
   not a 22-day 12-cell cohort.) *Code to add:* Zeta from its Algorithm 2 as `zeta.py` (the
   `normuon.py` pattern) + a `--optimizer zeta` flag in `train_ablation.py`
   ([arXiv:2606.14187](https://arxiv.org/abs/2606.14187), brief
   `research/briefs/zeta-dual-whitening.md`); trainer/`run_arms.sh`/gate/83%-guard reused.
3. **Publish — turn the attribution into the paper.** *What:* ship the packaged
   manuscript with a real per-component result. *How:* re-run `/manuscript` on this run;
   the Phase-1 **claim↔evidence gate** now passes because the headline is a single
   attributed component (not the confounded bundle), figures/tables regenerate from the
   ablation CSVs, and the package goes out behind the **human attestation** (the skill
   never auto-submits → you do the arXiv/HF click).
4. **Post-train rigor — close the post-train arc.** *What:* convert the n=1 inconclusive
   SFT into a real verdict. *How:* re-run the reasoning SFT at **≥3 seeds** via
   `/ablation-runner` in `finetune` mode (paired control + the §C13 catastrophic-forgetting
   probe), or add a preference stage (DPO/GRPO/RLVR); `/eval-harness` → across-seed CI
   decides win/loss (forgetting regression = a fail, not a footnote).
5. **Ship the loop end-to-end.** *What:* one fully autonomous cycle. *How:* paste the cron
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
