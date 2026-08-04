# BuildFromScratch

From-scratch language-model reproductions — each built single-file from a blank
editor, verified **bit-exact in fp32 on CPU** against the official HuggingFace weights, then
carried forward through a multi-stage research lifecycle (pretraining-era
architecture/optimizer/data studies → post-training) where every cross-run claim
is held to multi-seed CIs, iso-FLOP matching, and a held-out noise floor.

| Path | What it is |
|---|---|
| [`SmolLM2-134(base)/`](SmolLM2-134(base)/) | Single-file PyTorch reproduction of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (134,515,008 params), verified **bit-exact in fp32 on CPU** vs the official weights (`max \|Δlogits\| = 0.0`; on GPU it is **not** bit-exact — final-logits max 4.72e-05, per-layer hidden-state max 1.95e-03 at layer 14 — see `SmolLM2-134(base)/results/comparison_with_hf.md`). Includes from-scratch training, continued pretraining on TinyStories, multi-axis parity diagnostics, in-domain vs OOD eval, and an `lm-evaluation-harness` wrapper. |
| [`Qwen3-0.6B/`](Qwen3-0.6B/) | Single-file PyTorch reproduction of [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), verified **bit-exact in fp32 on CPU** (`max \|Δlogits\| = 0.0`, single 5-token prompt; no GPU parity check exists for this model), then a **full research lifecycle** on top (architecture / optimizer / data / post-training, below). See [its README](Qwen3-0.6B/README.md). |

> The repo is driven by a set of **local-only** Claude Code skills (an ML-research
> loop whose scanning, briefing, data-prep and scoring stages run autonomously,
> while a human triggers every GPU run) and a FastAPI agent-harness showcase;
> those are kept local and are **not** committed. What is committed is the model
> code, the verify gates, and the experiment results below.

## Research lifecycle — Qwen3-0.6B

The reproduction is the floor, not the finish line. From the bit-exact faithful
build, the project runs controlled, single-variable studies across the model
lifecycle. Every headline carries BPB on ≥2 corpora, across-seed CIs, iso-FLOP
matching, and a held-out noise floor — and is rewritten to exactly what the
evidence supports.

**1 · Reproduction (done).** Faithful Qwen3-0.6B trained from scratch reaches val PPL
**28.65** at 1.19B tokens, against **13.40** for the released Base (our own eval of it)
using **~30,000× less data** (1.19B vs 36T tokens). The implied **2.14×** ratio is
**cross-cache** — 28.65 was scored on the 1.19B run's val tail, 13.40 on the 131M run's
tail, and no same-cache score for the released model exists on disk — so read it as
indicative, not as a measured gap. The earlier 131M-token probe *is* same-cache: 46.31
vs 13.40 = **3.46×** with ~275,000× less data.

**2 · Architecture + optimizer — the "IMU-1" study (done).** A three-build
experiment (faithful baseline / modernized *IMU-1* bundle / exploratory
*partial-RoPE*) applies recent 2026 papers at matched compute. The modernized
**IMU-1 bundle beat the faithful baseline by 17.9%** (FineWeb-Edu PPL **23.52 vs
28.65**); exploratory partial-RoPE **lost** (0.25 = 29.54; the 0.10 variant died
incomplete). A two-phase, single-variable, **3-seed, iso-FLOP de-confound** then
**attributed** the win to **NorMuon** + the IMU-1 architecture modules
(value-residual / layernorm-scaling / head-gating — each individually significant
on canonical BPB), with learning-rate schedule and z-loss **not** significant.
The **NorMuon axis** of that attribution was isolated at a **42M-token** budget (the
schedule / z-loss / architecture axes ran at **131M tokens per cell**), and that NorMuon
strand **fades with budget** — the ladder in §7 returns `null`: the wikitext gap falls
0.474 → 0.072, and while every rung stays individually significant, the OLS-fitted edge at
the top rung (0.0297) lands inside the 0.0368 noise floor. On code the gap plateaus at
+0.177 and its fitted edge (0.1255 vs a 0.0463 floor) is still resolved.

**3 · Data composition (done).** A pretraining data-mix curve found a **50/50
mix** to be best-of-both — it keeps English while capturing ~84% of the code win.

**4 · Mid-training (done, 2026-07-01).** The stage was fully planned (§C27) with
two arms. A step-0 effective-context diagnostic on the base **failed the gate** —
the faithful base collapses at its own trained window (passkey accuracy **0.08 at
length 4096**), so the **context-extension** arm is held **propose-only**. The
**anneal** arm ran on-box: a low-LR 1-sqrt cooldown (2.5e-4 → 2.5e-5, ~150M tok ≈
13% of pretrain) from the faithful base on the **premium 50/50 mix** vs an
**iso-token FineWeb-Edu control**, 3 seeds each. The matched-decay control ate the
"LR-decay + more tokens" confound (control ≈ base, +0.005 code BPB) — and the
residual **data effect is real: code BPB improves +0.272 vs control (CI [+0.264,
+0.279], significant, n=3)**, ~46% of the constant-LR data-mix effect retained
under the anneal. English also moved slightly (+0.016, significant — the plan had
pre-registered a null there). A same-night RULER passkey ladder (6 rungs ×
40 paired samples, all 7 checkpoints) completed the §C25 battery: **no rung
regresses**, both arms fix the base's anomalous trained-length dip (passkey
0.03 → 0.40 at 4096), and the mix anneal **quadruples short-rung retrieval**
(0.20 → 0.82 at 512). Verdict: **win** (§C25 HARD-complete + §C18
single-variable/iso-FLOP confound check recorded in the ledger). The
post-training SFT below predates this stage and ran on the *faithful* base.

**5 · Post-training — SFT vs iso-FLOP control (done, 2026-06-30).** The faithful
base was fine-tuned on ~125M math-reasoning tokens with **response-masked SFT**
(3 seeds) against a **`--no_mask` continued-pretrain control** (3 seeds) on the
*same data, budget, and config* — isolating one variable: the masking.

- Held-out reasoning PPL improves **~18% over base** (14.13 → **11.57**), and the
  fine-tune shows **no catastrophic forgetting** (wikitext-2, code, and FineWeb-Edu
  all retained within the noise floor).
- **But response-masking does *not* separate from the iso-FLOP control**
  (SFT 11.573 vs control 11.582; masked +0.009 *sig* / full-sequence −0.006 *n.s.*
  → **directional, not a win**). The reasoning gain comes from the *extra
  in-domain tokens*, which the control gets too — not from the masking.
- A naive **in-loop** metric had reported a spurious **0.68-PPL "win"**; it was an
  eval-token confound (SFT scored response-only, the control scored all-tokens).
  Re-scoring both arms on **one fixed held-out response-masked set** collapsed the
  gap to **~0.01** — a clean negative result on the attribution question, and the
  reason the iso-FLOP control existed.

**6 · Reasoning RL — RLVR / GRPO (done, 2026-07-05).** A §C27 method-research pass
**pre-registered a null**: RLVR only *sharpens* solutions a base can already sample,
and at ~1% native math accuracy there is nothing to sharpen. We built the missing
**decision metric** (`math-acc-v1`: exact-match **pass@1** with Wilson CI + **pass@k**
with the Chen-2021 estimator, on decontaminated GSM8K + MATH-500) and tested it.

- **Phase-1 go/no-go** — the SFT'd base is **near the floor** (GSM8K pass@1 **1.1%** /
  pass@8 7%; MATH-500-L1–3 pass@1 **1.5%** / pass@8 10%), barely clearing the
  pre-registered GO threshold.
- **Phase-2 GRPO** (Dr.GRPO, 300 steps) — training reward stayed **flat at ~0.9%
  correct with no learning trend** (first-50 mean 0.0091 vs last-50 0.0080); gradient
  did flow (~13.5/16 groups kept by DAPO dynamic sampling, mostly on the format
  reward), but correctness never moved. The **iso-compute rejection-sampling
  control** collected **352** verifier-correct completions (~0.9% of 38,400
  rollouts) — too few for its masked-SFT pass to separate from the floor
  (nominally above on pass@8, CIs overlap); a **random-reward control** trained
  at ~0.5 mean reward (proving the pipeline works).
- **Verdict: GRPO beats neither the SFT floor nor the random-reward gate** (pass@1
  Wilson CIs overlap on both corpora) → **directional null, exactly as predicted**.
  The reasoning capability lives in SFT/distillation, not RL at this scale — the gate
  saved a multi-seed cohort before it was spent.

**7 · Scaling persistence of the NorMuon win (first 10 cells done 2026-07-12; the two 420M seed-2 cells 2026-07-25/26; re-scored at n=3 2026-07-28).** Study #2 isolated
**NorMuon** (at 42M tokens) as one strand of the IMU-1 win — note its de-confound arms were
all AdamW, so no iso-budget NorMuon-vs-architecture comparison exists. This ladder asks
whether NorMuon's **+0.474 wikitext BPB** edge over AdamW **persists or converges with budget**. At fixed
N=596M it sweeps the token budget — 42M (reused) + **168M** + **420M**, each
×{NorMuon,AdamW}×**3 seeds** — varying only `--steps`. All **12 newly-trained** cells
completed (the six 42M cells are reused from `2026-06-16_qwen3_normuon-vs-adamw`, so 18 are
scored in total); re-scored at n=3 on 2026-07-28 after the third 420M seed (one cell per
arm) landed.

- **The gap shrinks with budget, and both corpora agree.** wikitext-2
  (AdamW − NorMuon, BPB): **+0.474** [+0.443, +0.505] at 42M → **+0.126**
  [+0.089, +0.163] at 168M → **+0.072** [+0.055, +0.088] at 420M. code_py:
  **+0.502** [+0.456, +0.547] → **+0.176** [+0.137, +0.215] → **+0.177**
  [+0.131, +0.223]. OLS over log10(tokens) gives slope **−0.417** (r² 0.92) on
  wikitext and **−0.342** (r² 0.84) on code → **CONVERGES** on both.
- **Verdict: `null`** (ledger `2026-07-05_qwen3-0.6b_scaling-persistence`) — an
  early-training speedup that converges away. Note the verdict was *also* capped by
  construction: the §C25 `scaling` HARD battery is incomplete (no `log_rmse_r2`,
  `holdout_extrapolation_pctdev`, `bootstrap_forecast_ci`), so `win` was unreachable
  regardless — though the CONVERGES trend independently maps to `null` anyway.
- **What is resolved, and what isn't.** The *slope* is resolved on both corpora; the
  *edge at the top rung* is resolved on **code** but not on **wikitext**. All six
  rungs remain **nominally significant** (every CI excludes 0) — "falls within the
  noise floor" refers to the *OLS-fitted* edge at the top rung, which on wikitext is
  0.0297 against a 0.0368 floor (`edge_resolved: false`), while on code it is 0.1255
  against 0.0463 (`edge_resolved: true`). The code_py gap also does not shrink
  **monotonically**: it falls 0.502 → 0.176, then ticks up to 0.177 — flat between the
  top two rungs, a **plateau**, even though the 3-point OLS still scores `CONVERGES`
  (a negative slope carried largely by the 42M point). A disclosed **inherited confound**
  cuts the same way: both learning rates were tuned at the 42M horizon and never
  re-tuned per budget, so part of the fade may be a mis-tuned-LR artifact rather than
  true convergence. Earning more needs an 840M rung and a per-horizon LR check. This
  is a **budget** null at fixed N=596M — it says nothing about larger N.
- Read it as: NorMuon looks like an **early-training speedup that converges away** —
  exactly what IMU-1's own Limitation #3 warned it might be. Numbers:
  `experiments/2026-07-05_qwen3-0.6b_scaling-persistence/verdict.json`.

> **GB10 thermal-survival note (2026-07-08→10).** The 420M rungs surfaced a hardware
> reality: under sustained load in warm ambient the unified Grace-Blackwell package
> **overheats to 92–94 °C with the GPU thermal-throttling**, which had been silently
> hard-locking the whole box mid-run (a ~16–18 h rung could never finish, restarting
> from step 0 each crash). The run is now **crash-survivable**: `train_ablation.py`
> checkpoints full training state (model + both optimizers + step + RNG, atomic+fsync)
> every 100 steps and **auto-resumes** from it; `sentinel.py watch` gained a **thermal
> kill-switch** (SIGTERM at ≥90 °C or a hardware throttle flag — which also *prevents*
> the hard-locks by shedding load before the box wedges); a dense `thermal_log.py`
> records the full temperature envelope; and `run_ladder.sh` loops-until-done behind a
> cool-down gate, backed by an `@reboot` auto-resume (`boot_resume.sh`). Net: the
> multi-day ladder now survives each thermal event by losing ≤~15 min (one checkpoint
> interval) instead of a whole rung. **It worked:** a 2026-07-09 hot spell had left the
> ladder stuck at 4/10 cells missing with no net progress, and the four 420M rungs then
> finished 2026-07-10→12 *through* repeated thermal kills. Cooling the box's ambient is the high-leverage
> throughput fix — cooling *time* after a kill is only ~10–30 s; it's the ~3 min reheat
> to 90 °C under warm ambient that throttles daytime throughput.

Each study lives under `Qwen3-0.6B/experiments/<YYYY-MM-DD>_<model>_<slug>/` with
its methodology (`c5_evidence.json`), results (`verdict.json` /
`reasoning_verdict.json`), and a per-run record in `research/ledger/runs/`.

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

The Qwen3-0.6B reproduction has the same shape — `cd Qwen3-0.6B && python verify.py`
runs its bit-exact parity gate. See each subproject's `README.md` for the full
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
│   ├── generate.py                # CLI sampler
│   ├── tests/test_parity.py       # pytest gate
│   ├── scripts/                   # lm-eval wrapper + checkpoint export
│   ├── proof.ipynb / results.ipynb
│   └── README.md                  # the long-form architecture/recipe doc
├── Qwen3-0.6B/                    # the Qwen3-0.6B reproduction + lifecycle studies
│   ├── model.py                  # the architecture, one file — verified bit-exact vs HF
│   ├── verify.py                 # parity gate
│   ├── builds/                   # faithful / modernized (IMU-1) / exploratory (partial-RoPE)
│   ├── experiments/              # single-variable studies (arch / optimizer / data / post-training)
│   └── README.md                 # the long-form study writeup
├── safe_cuda.py                   # GB10 unified-memory guard (caps the CUDA process)
├── sentinel.py                    # resource watchdog — preflight / watch (memory + thermal kill-switch) / liveness
├── jax_safe_env.py                # JAX preallocation guard for the shared-memory box
├── mfu_meter.py                   # MFU / HFU + achieved-TFLOPS (honest, GB10 peak flagged estimated)
└── flop_accounting.py             # FLOP-per-token (6N + 12·L·H·Q·T) — feeds the iso-FLOP gate
```
