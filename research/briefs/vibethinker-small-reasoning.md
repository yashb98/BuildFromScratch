# Brief: VibeThinker-3B — Spectrum-to-Signal verifiable-reasoning post-training (vibethinker-small-reasoning)
- researched: 2026-06-17 · by: ml-research · fetch_level: fulltext-arxiv-html
- paper_date: 2026-06-15 (cutoff_3m this run: 2026-03-17)
- modality: text-lm · verdict: runnable-now
- objective: finetune (§C13) · taste_score: 6.0 (§C15.2)

## Sources (all fetch-verified this run, §C3)
| url | what it is | accessed | replication status |
|---|---|---|---|
| https://export.arxiv.org/api/query?id_list=2606.16140 | arXiv API metadata (published 2026-06-15T02:57:19Z, cs.AI/cs.CL, v1) | 2026-06-17 | primary source |
| https://arxiv.org/html/2606.16140 | full paper HTML — recipe, stages, limitations, hypothesis | 2026-06-17 | claim, unreplicated (1 day old) |
| https://arxiv.org/abs/2606.16140 | abstract / authors (WeiboAI team) / categories | 2026-06-17 | primary source |
| https://huggingface.co/api/models/WeiboAI/VibeThinker-3B | model weights exist, ungated, sha 51e5928c3cc79ad954fc7a66cc17aa91be7581d7 | 2026-06-17 | weights released (not independently re-evaluated) |
| https://news.ycombinator.com/item?id=45910410 | HN thread on predecessor VibeThinker-1.5B (14 comments) | 2026-06-17 | independently discussed, not re-measured |
| https://news.ycombinator.com/item?id=48562111 | HN thread on the 3B (3 comments, 1-day-old) | 2026-06-17 | independently discussed, not re-measured |
| WebSearch "VibeThinker-3B … reproduction" | landscape: HF papers, GitHub WeiboAI/VibeThinker, VentureBeat | 2026-06-17 | no independent repro found |
| WebSearch "VibeThinker 1.5B … contamination/general knowledge" | predecessor critique + the team's contamination defense (AIME25/HMMT25 post-date the base) | 2026-06-17 | mixed: defense + practitioner skepticism |

Note: VentureBeat "Why Weibo's tiny VibeThinker-3B has the AI world arguing over benchmarks again" surfaced in search results but the page returned HTTP 429 on two WebFetch attempts this run — recorded as a not-fetched lead, NOT cited as evidence.

## What it changes
VibeThinker-3B is a post-training *recipe*, not an architecture change: starting from a dense base (the paper uses Qwen2.5-Coder-3B), it applies the "Spectrum-to-Signal" pipeline — (1) curriculum-based SFT (broad-coverage distillation of multi-path reasoning traces, then a hard-sample second stage), (2) multi-domain RL with verifiable rewards (their MGPO/GRPO-family algorithm over Math→Code→STEM, rewards from answer-checking and sandbox code execution), and (3) offline self-distillation on the model's own high-"learning-potential" trajectories, plus an instruct-RL pass. The thesis (Parametric Compression-Coverage Hypothesis) is that *verifiable* reasoning is a compressible "signal" that a tiny model can hold, even though broad knowledge ("spectrum") needs many parameters — so a sub-1B base should be able to absorb the reasoning gain. For our purposes the importable core is: **curriculum SFT on reasoning traces → GRPO with a verifiable (correctness) reward**, which maps directly onto the repo's `research/posttrain_losses.py` (SFT CE + GRPO group-advantage + clipped surrogate + k3 KL).

## Taxonomy (§C12 — axes touched)
architecture: decoder-only · training-stage: reasoning (RLVR/GRPO) on a base checkpoint · modality: text · specialization: code+math · size-band: SLM/tiny (our base is 596M) · context/position: RoPE θ=1e6 / GQA (inherited unchanged from the Qwen3-0.6B base) · openness: fully-open (weights+code on HF/GitHub).

## Objective (§C13)
- type: finetune
- base checkpoint: `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/checkpoint_qwen3_baseline2tpp.pt` (verified on disk this run; 596,049,920 params, FineWeb-Edu val PPL 28.65, vocab 151,936, seq_len 4096, PyTorch). NOTE: the paper's base is Qwen2.5-Coder-3B (a *code-pretrained, 5× larger* model); our base is a knowledge-dense FineWeb-Edu reproduction ~10× *under*-trained (2.14× PPL gap to real Qwen3-0.6B). This is a major extrapolation gap, recorded in Research-taste and Failure modes.
- adaptation method: curriculum **SFT** (reasoning-trace distillation, masked-completion CE) → **GRPO** with a verifiable correctness reward (math: answer-check; code: sandbox pass/fail). Both losses already implemented + unit-tested in `research/posttrain_losses.py`; the GPU launch routes through `/post-train` → `/ablation-runner` (§C5/§C11). Self-distillation/instruct-RL stages are out of scope for a first run (kept as later phases).
- catastrophic-forgetting probe (§C10, must not regress): **FineWeb-Edu held-out PPL with the model's own Qwen3 tokenizer** (the base's 28.65 number, same `eval_original_vs_repro.py` 300k-token slice) — the general-competence metric the paper itself admits is at risk ("knowledge-intensive benchmarks still expose a clear gap"; GPQA-Diamond lags). This becomes a required `metrics` field on the eventual `finetune` run.

## Exact recipe
All values from `arxiv.org/html/2606.16140` (fulltext) unless flagged. Paper trains a 3B base; our base is 596M, so transferred values are `inferred`.

| Hyperparameter | Value | Flag | Provenance |
|---|---|---|---|
| Base model (paper) | Qwen2.5-Coder-3B base | reported | "Qwen2.5-Coder-3B base, a compact 3B dense foundation model" |
| Base model (ours) | faithful Qwen3-0.6B repro ckpt (596M) | inferred | repo-fit substitution (Phase 5) |
| SFT stage-1 optimizer | not stated (AdamW assumed) | inferred | paper omits; modern-LLM + repo convention |
| SFT stage-1 batch size | global batch 128 | reported | "global batch size of 128" |
| SFT stage-1 peak LR | 5e-5, cosine → 8e-8, 5% linear warmup | reported | "initial learning rate to 5×10⁻⁵ … cosine annealing … 8×10⁻⁸ … 5% linear warmup" |
| SFT stage-1 epochs | 5 | reported | "trained for 5 epochs" |
| SFT stage-2 (hard) | +2 epochs, same HPs, on hard subset | reported | "additional 2 epochs … exact hyperparameter configuration from the first stage" |
| SFT hard-sample filter | reasoning trace ≥5K tokens; error-rate ≥0.75 (ref = VibeThinker-1.5B) | reported | data-filtering description |
| SFT data hygiene | n-gram filter, LLM query assessment, answer-check + sandbox-exec verification, eval-set de-contam | reported | quality-control section |
| RL algorithm | MGPO (MaxEnt-Guided Policy Optimization, GRPO family) | reported | "MaxEnt-Guided Policy Optimization (MGPO) retained from 1.5B work" |
| RL context window | single 64K long-context | reported | "single 64K long-context window" |
| RL domain order | Math → Code → STEM (sequential) → Instruct RL | reported | training-sequence description |
| RL reward (math/code) | binary correctness (answer-check / sandbox pass) | reported | "correctness binary signals" |
| Long2Short reward λ | 0.2 (max redistribution magnitude, zero-sum) | reported | "λ = 0.2 controlling maximum redistribution magnitude" |
| RL clip ε / KL β / steps / samples | not reported | reported-as-absent | "Not stated: clipping coefficient ε … number of training steps … KL penalty coefficients" |
| Self-distillation | length-normalized NLL "learning-potential" score; pick mid-high band per domain length bucket | reported | stage-3 description |
| Instruct RL | rule-based validators + rubric reward models | reported | stage-4 description |
| Eval sampling | T=1.0, top-p=0.95, top-k=-1; 64 gens/math problem | reported | evaluation protocol |
| Total compute / dataset sizes | not reported | reported-as-absent | "no information on total training compute … dataset sizes … samples" |

## Recommended budget (scaled)
- paper budget + where it's stated: SFT = 5 epochs broad + 2 epochs hard (token/sample counts **not reported**); RL step/sample counts **not reported**; total compute **not reported** (a real transparency gap). The only firm SFT anchors are batch=128, LR 5e-5→8e-8 cosine, 5%/-warmup.
- scaling reasoning: with no paper token count, scale by epochs over the *prepared* dataset, not a token target. **SFT (inferred):** with the repo's measured ~7,300 tok/s, a ~50–150M-token reasoning-trace SFT (≈ a few epochs over a ~20–40M-token curated trace set, e.g. an OpenR1-Math subset) is a 2–6 h job — a cheap, low-blast-radius first arm. **GRPO (inferred):** budget by *rollouts*, not pretrain tokens; a first GRPO arm of ~2k–5k prompts × group size 8–16 × a few epochs is the right exploratory size at our scale (the paper's 64K context is **not** affordable here — use 2k–4k completion length, an explicit downscale flagged inferred). LR for GRPO ~1e-6 (inferred, standard small-model RLVR).
- TOKEN_BUDGET per model (source attached — §C5.2):
  - **Qwen3-0.6B (target):** SFT arm ≈ 100M tokens over the prepared trace shards (inferred: epochs × dataset size, paper gives only batch/LR/epochs); GRPO arm budgeted as ~3k prompts × 8 samples × ≤4k tokens ≈ exploratory, ETA computed by ablation-runner's probe. First run = **SFT only** (cheapest path to a measurable signal); GRPO is a queued follow-on.
  - **SmolLM2-134M:** out of scope for the first run — only the Qwen3 base is named here; a SmolLM2 reasoning post-train is a separate future candidate.

## Framework / runtime fit (§C14)
- recommendation: **pytorch** — the base checkpoint is PyTorch (§C14(a), inherited), the repo's `posttrain_losses.py` is PyTorch, and you cannot cheaply port an existing checkpoint. `jax_vs_pytorch_tradeoffs.md` (read this run) confirms JAX is **not meaningfully faster** on a single GB10 for a sub-1B transformer (±10–20%; measured PyTorch baseline ≈ 7,300 tok/s, Qwen3 mb4@4096) and carries an XLA-preallocation memory-safety tax — no reason to switch.
- portability / kernel flags: GRPO rollout *generation* (autoregressive decode) is the hot path; a served/batched generation backend (vLLM via `/serving-bench`) would speed rollouts, but that is an optimization opportunity to flag, **not** an unattended kernel write (§C14, §C5). The paper's MGPO is a GRPO variant — we implement the standard GRPO surrogate in `posttrain_losses.py`, treating the MaxEnt/Long2Short reward shaping as an optional, separately-flagged delta (avoid bundling per §C18).

## Baseline + win condition
- baseline checkpoint: `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/checkpoint_qwen3_baseline2tpp.pt` (provenance: faithful-build README + parent Qwen3-0.6B README read this run — val PPL 28.65, 596,049,920 params).
- win = a **TWO-part test (§C13)**: (i) a reasoning-task gain that clears the §C17 noise floor with seeds ≥ 3 and a paired-difference CI excluding 0 — measured via `/eval-harness` (target metric: an in-distribution math/code accuracy probe on a held-out split prepared with dataset-forge hygiene, e.g. GSM8K-style exact-match or a small MATH subset; the base's near-zero reasoning accuracy is the floor), **AND** (ii) the catastrophic-forgetting probe (FineWeb-Edu PPL under the Qwen3 tokenizer) does **not** regress beyond the noise floor from 28.65. A reasoning gain bought with PPL regression is a `loss` (§C13). Both numbers are required `metrics` on the `finetune` run.

## Research-taste verdict (§C15.2)
- taste_score: 6.0 · axes: mechanism **strong** (curriculum-SFT→RLVR is the most-validated post-training recipe of 2025–26; verifiable rewards are honest, non-gameable signals; the compression-coverage thesis is coherent) · evidence **mixed** (clean contamination defense — AIME25/HMMT25 post-date the base — but a real transparency gap: no compute/dataset/sample counts, no base-vs-tuned general-benchmark table, and the bundle confounds SFT+RL+self-distill+instruct-RL, which §C18 forbids running as one arm) · reproduction **none yet** (paper 1 day old; predecessor 1.5B was widely replicable as weights but practitioners report a benchmark-vs-utility gap and narrow specialization) · scaling-to-our-scale **risky** (paper's base is a 3B *code*-pretrained model; ours is a 596M FineWeb-Edu repro ~10× under-trained — the "compressible reasoning core" claim is least tested exactly at our size/under-training) · ROI **good IF data exists** (the recipe is cheap and the SFT-first arm is low-cost/low-blast-radius; the gain is plausible because RLVR reliably lifts in-distribution reasoning even on small bases) · simplicity/blast-radius **good** (decompose into single-variable arms: SFT-only first, GRPO second — `posttrain_losses.py` already exists; no canonical-file edits) · safety **good** (a finetune in an `experiments/` dir; reversible; memory fits — §C1).
- One paragraph: This is a high-mechanism, low-novelty recipe — curriculum SFT then GRPO with verifiable rewards is the dominant reasoning-post-training playbook, and that *raises* my confidence in the direction even though the paper itself is buzz-heavy ("beats Gemini 3 Pro") in a narrow band it admits doesn't transfer to knowledge tasks. I weight mechanism + the SFT-first ROI up, and weight the missing-budget transparency, the confounded bundle (§C18), and the 3B-code-base → 596M-FineWeb-base extrapolation down. The decisive blocker is data: there is **no verifiable-reward dataset on disk**, so this cannot be `runnable-now`. Decomposed to an SFT-only first arm on a forged reasoning-trace set, with the FineWeb-PPL forgetting guard, it is a worthwhile mid-priority experiment — hence 6.0, above pure-buzz candidates but below a clean, data-ready single-variable ablation.

## Reproductions & criticism
- The 3B paper is **1 day old (2026-06-15)** — **no independent reproduction found as of 2026-06-17** (queries: title + "reproduction results"; "VibeThinker-3B … small language model reproduction"; HN Algolia "VibeThinker" → only WeiboAI-origin and discussion posts, ≤3 comments on the 3B). Recorded as a finding: zero independent re-measurement, which pushes the first budget small.
- Predecessor **VibeThinker-1.5B** (Nov 2025, WeiboAI, also fully open) is the track record: the team's contamination defense is genuine (AIME25/HMMT25 released after the Qwen2.5-Math base, so the math gains are not pure leakage). But HN practitioners report (i) narrow specialization — "specifically trained on maths," general coding/explanation "break completely," repetition loops; (ii) a benchmark-vs-real-utility gap. For the 3B, HN testers note it "reliably writes working Python" but "takes shortcuts / writes wrong steps" and found 0 security bugs. Net: the *reasoning-on-verifiable-tasks* gain is credible; the *general-competence* claims are where to be skeptical — which is exactly what the forgetting probe guards.

## Failure modes & abort criteria
All measurable from artifacts ablation-runner/eval-harness already produce:
- **Forgetting (the primary risk):** FineWeb-Edu PPL regresses > noise floor above 28.65 for 2 consecutive logged evals → abort (a reasoning gain bought with general regression is a `loss`, §C13).
- **RL instability:** grad-norm > 5× the SFT run's median, or NaN/Inf at any step → instant abort (GRPO + tiny under-trained base is init/LR-sensitive).
- **Reward hacking / KL blow-up:** k3 KL to the reference policy grows monotonically while train reward rises but the held-out reasoning probe does not → abort (policy is gaming the verifier, not reasoning).
- **No early signal:** after the first 25% of the SFT/GRPO budget, the in-distribution reasoning probe delta is below the §C10 noise floor → abort (recipe not transferring at 596M under-trained scale, the stated extrapolation risk).
- **Throughput:** rollout tok/s degraded > 30% vs the run's own probe (§C5.3) → flag overhead (RL generation overhead beyond plan).

## GB10 feasibility (§C1)
- Memory (this run's arithmetic, 596M params): params bf16 1.19 GB + grads bf16 1.19 GB + AdamW fp32 m+v 4.77 GB + frozen reference policy (bf16, needed for GRPO ratio/KL) 1.19 GB = **~8.3 GB** weights/state. Activations dominate; the base pretrain measured **52.4 GB peak** at mb4×seq4096 with chunked CE — well under the 60%-of-119 GB ≈ **71 GB** plan cap. GRPO uses shorter completions (≤4k, downscaled from the paper's 64K), so it fits; the §C5.3 measured probe (ablation-runner) is the launch authority, not this estimate.
- vocab 151,936 > 64k → **chunked cross-entropy is mandatory** (§C1; the exact allocation that crashed the box on 2026-06-08). The repo trainer already does this.
- modality fit: text-LM, matches the repo baselines — no modality gap.
- aarch64 deps: no extra CUDA-only x86 library required — SFT+GRPO run on the in-repo `research/posttrain_losses.py` with the installed stack (torch 2.11+cu130, transformers 5.8.0, both importable this run); no `trl`/`verl` dependency (not installed, not needed). A sandbox code-verifier (for code-RL rewards) is the one extra component — pure-Python/subprocess, no x86 wheel issue — but code-RL is a later phase, not the first SFT arm.

## Verdict
**runnable-now** (flipped 2026-06-17, research-loop S6 — SFT dataset now prepared on disk; see Addendum). Originally **needs-dataset** — the recipe, base checkpoint, adaptation method (SFT→GRPO via `posttrain_losses.py`), forgetting probe, budget, and memory fit are all concrete and on-box, but there is **no reasoning-trace SFT set or verifiable-reward GRPO prompt set prepared on disk** (`research/datasets/` is empty). dataset-forge must produce, for the Qwen3-0.6B base: (a) an SFT shard set of curated math/code reasoning traces (candidate source: `open-r1/OpenR1-Math-220k`, HF http 200, ungated, sha e4e141ec9dea9f8326f4d347be56105859b2bd68 — verified this run) with response-masking + a hygienic held-out reasoning eval split; and (b) for the queued GRPO follow-on, a prompt+verifiable-answer set (candidate `agentica-org/DeepScaleR-Preview-Dataset`, http 200, ungated, sha b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29; eval anchor `HuggingFaceH4/MATH-500`, http 200, ungated, sha 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be) with a checkable reward function. First run should be **SFT-only** (single variable, §C18); GRPO is a separate queued arm once SFT clears the forgetting probe.

## Addendum (2026-06-17, research-loop S6)
Verdict flipped `needs-dataset` → `runnable-now`. The SFT dataset is now prepared on disk:
- **dataset:** `research/datasets/math-reasoning-openr1-math-220k/` — 125,000,592 train tokens (35,512 verified OpenR1-Math-220k reasoning traces, Qwen3 tokenizer, response-masked via per-sample `prompt_len`), held-out reasoning eval (66 docs / 232,918 tok, 0 leakage) + the §C13 FineWeb-Edu forgetting probe (202 docs / 167,242 tok).
- **forge ledger run:** `2026-06-17_qwen3-0.6b_openr1-math-220k` (type=dataset-prep, done); HF sha `e4e141ec9dea9f8326f4d347be56105859b2bd68`.
- **first run:** SFT-only (~100–125M tokens, single variable §C18) on `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/checkpoint_qwen3_baseline2tpp.pt`; win-metric = held-out reasoning gain clearing the §C17 noise floor (seeds≥3, paired CI excludes 0) AND FineWeb-Edu PPL must not regress past 28.65 beyond floor (§C13). GRPO remains a separate queued arm.
