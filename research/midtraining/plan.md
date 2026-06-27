<!-- Generated 2026-06-27 by workflow wf_a4d31c41-80a (midtraining-recipe-research): 3 personas recency-first + adversarially verified. Methods fetch-verified; flagged items dropped. -->

# Mid-Training Plan — Qwen3-0.6B (596,049,920 params, 1.19B-token faithful base)

**Chair synthesis of three verified positions + their adversarial checks. Every method below survived the red-team; flagged/contested items are dropped or demoted to propose-only and called out in §5.**

Base checkpoint (single-variable, clean): `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/checkpoint_qwen3_baseline2tpp.pt` (seq 4096, cosine, AdamW peak 1.7e-3, ended ~3.2e-4 region).

---

## 1. The honest verdict up front

At **596M params / 1.19B tokens** (badly undertrained, ~2 TPP), mid-training is **mostly a methodology demonstration, with exactly one near-non-fakeable capability axis and exactly one possibly-attributable scalar win.** Build it so it cannot oversell:

- **Context-extension (4K→8K) is the capability headline — but GATED.** It is the *only* axis where the base provably cannot already do the thing: it literally never saw a position > 4096. A genuine 4K→8K effective-context jump on the RULER ladder is hard to fake. **BUT** all three positions hand-waved one fact: an undertrained 1.19B-token base may not even *use* its full 4096 window. **So context-extension is worth the GPU-day only if a cheap diagnostic (§3, step 0) shows the base's effective context actually reaches ~4096 before collapsing.** If it collapses earlier, extension is moot → demote to propose-only. Even when it passes, the honest win is *"the window genuinely extended" (retrieval rungs rise)* — **NOT** *"good at long context."* NoLiMa indirection / multi-hop will stay near floor; a 596M/1.19B base lacks the latent retrieval+reasoning capacity to reason over 8K.

- **Anneal is mechanistically real but near-null on anything broad.** At ~1.2B tokens, data-source differences sit inside the noise floor (codelion: 31.42–32.38 across 7 datasets), and our premium mix injects **no** math/code/SFT/instruct data — unlike MiniCPM (whose C-Eval jump came from injecting instruct/code into the decay) or Llama-3-8B (whose +24% GSM8k came from a capable base with surfacable latent skills). Our base is the opposite failure mode: undertrained **and** low-capability. **The single place a real, attributable signal might clear the floor is matched-domain CODE BPB** — anchored on-disk: the 50/50 mix gave **+0.5904 BPB on `code_py`, CI [+0.5532, +0.6277], significant, n=3** vs FineWeb-Edu, while English wikitext2 was **+0.0161, CI [−0.0005, +0.0327], NOT significant** (`Qwen3-0.6B/experiments/2026-06-26_qwen3-0.6b_data-mix-composition/verdict.json`). **Caveat baked in:** that +0.59 was *full continued-train at normal LR*, not a low-LR WSD anneal — the anneal version will be **weaker**. Report the anneal **only** as matched-domain BPB vs its iso-token control, with a 3-seed CI that must exclude the control. **Never** as a downstream win.

**Worth the GPU-day:** context-extension (capability the base lacks), conditional on the diagnostic. **Near-null but cheap and on-disk-grounded:** the anneal — run it, but cap the claim to code BPB. **The single biggest way this stage fakes a result:** confounding the WSD LR-decay loss drop (which happens on *any* data — MiniCPM Fig 6) with "premium data seen last." The matched-decay control in §2 is non-negotiable.

---

## 2. The anneal experiment

**Goal:** isolate "premium 50/50 mix seen last under a low-LR cooldown" from "the LR decay itself + more tokens." Mechanism demo on matched-domain BPB; honest null expected on everything broad.

**Base:** fork the faithful `checkpoint_qwen3_baseline2tpp.pt` for **both** arms (clean single variable). Shape unchanged: **seq 4096, micro_batch 4** (proven, ~6,800 tok/s, zero new OOM surface), AdamW(0.9, 0.95), wd 0.01, grad_clip 1.0.

**LR schedule (WSD cooldown, reconciled with the adversarial corrections):**
- Start at **~2–3e-4** (≈ the base's cosine end-LR / ~10–15% of the 1.7e-3 peak). **No re-warm in the headline arm** (re-warming to 0.3× peak adds a re-warm-vs-not confound; keep it as an *optional* second arm only).
- **1-sqrt or linear** decay shape (cooldown-dynamics 2508.01483 supports both; the exact "lowered-linear-0.7" name is search-surfaced — do **not** pin it).
- **Decay to a small non-zero floor (~5–10% of start LR), not exactly 0.** This honors the LR-decay-wastes-data finding (2511.18903: premium data stranded under a *collapsed* LR is partly wasted) at ~zero cost, instead of blindly adopting decay-to-0.
- **Blend the premium 50/50 mix across the entire decay window** — do not hold it for a last-N-token cliff (same paper's concern).
- **EMA/Polyak-average the last ~3–5 anneal checkpoints** (free; endorsed by both Llama-3 and 2511.18903's averaging fix).

**Token budget:** **~150–200M tokens (~13–17% of the 1.19B pretrain)** — squarely in WSD's 10–20% cooldown band; MiniCPM reports ~10% sufficient. **Do NOT justify this via a Llama-3 absolute-token floor** — that number is contested between the verifiers (40M vs 40B) and is dropped. The mid-training-survey per-model percentages (MiniCPM ~2% / SmolLM2 ~9% / Qwen3 ~17%) are PDF-pending and internally inconsistent — do not pin them either.

**Data — treatment vs control (the attribution-critical pair):**
- **Treatment:** the verified premium **50/50 FineWeb-Edu + dclm-edu** mix (best-of-both: +84% of dclm's code gain, English retained). Optionally retain ~70% curated / ~30% premium-upweight per the survey's "retain a pretraining subset" rule — but keep the *contrast vs control* identical.
- **Control (MANDATORY, iso-token / iso-FLOP):** identical fork, identical 1-sqrt-to-floor schedule, identical ~150–200M-token budget, identical 3 seeds — but on **baseline FineWeb-Edu only**. Headline = `treatment_BPB − control_BPB`. **The control eats most of the apparent gain** if it's just "more training"; only the residual is a data effect.
- **Free no-anneal floor (§C13 reuse):** reuse the already-eval'd `baseline2tpp` checkpoint — do **not** re-run it. ⚠️ Do **NOT** reuse the data-mix study's `checkpoint_mix_seed*.pt` as the anneal control — those are *full continued-train at normal LR*, a different LR regime; they serve only as a constant-LR upper-bound reference for the code-BPB effect.

**Seeds / gate:** **3 seeds per arm** (§C17). Reuse `verdict.py` + `score_cohort.py` from the data-mix experiment dir. Welch-t; the treatment−control CI must **exclude the control** to claim a data effect.

**Eval (reuse §C25 mid-training row — `research/eval/per_stage_eval_batteries.md:25`, do NOT re-research):** anneal-gain vs matched control on **BPB (wikitext2 + code_py)**; text-lm-v3 downstream **reported honestly as expected-null**; **short-context non-regression** on the original 4096 battery; per-position loss tail. Pre-register that English-BPB and downstream sitting **inside** the 3-seed CI is an **accepted honest null at this scale**, not a failure.

**Honest expected outcome:** small, possibly-significant win on **code BPB only** (and weaker than the +0.59 constant-LR anchor); null on English BPB and downstream. The "~2–6% relative BPB" figure from one position is an **un-cited hypothesis, not a literature number** — do not report it as evidence.

---

## 3. The context-extension experiment

**Method (the single verified-current choice for our regime):** **continued-pretrain AT length, RoPE θ unchanged.** No PI / NTK / YaRN / LongRoPE2. Verified in code: `model.py:45` `rope_theta = 1_000_000.0`, `model.py:44` `max_position_embeddings = 40_960`, and the RoPE cache is prebuilt fp32 to 40,960 positions (`model.py:78-82, 222-225`). At a 2–4× target the model has **enormous unused rotational headroom** — the bottleneck is *never having seen a position > 4096*, not RoPE OOD. PI/NTK are corrections for θ=1e4 models (superseded for this use-case); YaRN/LongRoPE2 are 8–32× machinery (overkill, propose-only). The current-best minimal recipe is **ProLong (2410.02660)**: continue-pretrain directly at the longer seq on long documents; "training beyond the eval length boosts long-context performance."

**Step 0 — diagnostic GATE (eval-only, no training):** build the RULER-ladder loader (⚠️ **this loader does not exist yet** — it's a §C25 build-backlog item: `per_stage_eval_batteries.md:120`) with **center rungs ABOVE 4096** (2K / 4K / 6K / 8K — a 1K–8K ladder "tests nothing" since the model trained at 4096). Score the existing base. **If effective context reaches ~4096 then collapses → proceed. If it collapses before 4096 → context-extension is moot → propose-only.** This is the fact all three positions assumed.

**Target length & memory math (corrected — this is where two of three positions had the mechanism wrong):**
- With `is_causal=True` / `attention_mask=None` (`model.py:162-166`), SDPA dispatches the **flash / mem-efficient backend → attention memory is O(seq), not O(seq²)**. O(seq²) is **COMPUTE/FLOPs**, not memory. The activation ceiling is set by **tokens-per-microbatch**, and the proven ceiling is **mb4 × seq4096 = 16,384 tokens** (~52–57 GB observed).
- Therefore **seq 8192 @ mb2** and **seq 16384 @ mb1** both = **16,384 tokens/microbatch = the proven footprint → both should FIT.** Use grad-accum for global batch. **Target 8192 (solid); 16384 is a feasible stretch** (slower). **Refuse 32K+** (= 32,768-token footprint → needs activation checkpointing; fp32 logits ≈ the ~40 GB that hard-crashed the box 2026-06-08).
- Throughput falls to **~4,300–5,500 tok/s** at 8K–16K from the O(seq²) attention compute (not a memory blowup).
- ⚠️ **MANDATORY backend smoke-test before committing GPU-days:** that O(seq) memory story assumes SDPA actually selects flash/mem-efficient on the **GB10 Grace-Blackwell ARM** arch — *assumed, not verified*. Print the dispatched backend at seq 8192 first. If it falls back to math, 16K gets tight.

**Three code-level fixes that MUST land before any >4K run (verified against `model.py`):**
1. **bf16-RoPE corruption (ACTIVE risk — `per_stage_eval_batteries.md:25` flags it too).** The cache is built fp32 (`model.py:78-80`) but `cos/sin` are downcast to `x.dtype` (bf16 in training) at `model.py:230-231`, and the rotation `(q*cos)+(rotate_half(q)*sin)` runs in that dtype (`model.py:94-95`) → bf16 rotation = AnchorAttention's first-token failure mode, material > 4K. **Fix:** do the rotation in fp32, cast q/k back to bf16 after.
2. **Doc-masking must NOT use a dense mask.** Intra-document (cross-doc) masking helps **both** short and long context (ProLong, paper body) and is worth turning ON. **BUT** passing a dense `(S,S)` `attn_mask` sets `is_causal=False` (`model.py:166`) → kills the flash path → math backend materializes O(seq²) (~8.6 GB per 16-head batch at 16384) → instant OOM. **Implement via FlashAttention-2 varlen or `flex_attention` block-diagonal** (what ProLong itself uses), never a dense SDPA mask. ⚠️ Also smoke-test `flex_attention` works on this PyTorch build + Blackwell/ARM (it's newish).
3. **Chunked CE is mandatory at extended length** (safe_cuda, §C1). Plain `F.cross_entropy` at `model.py:269` over vocab 151,936: fp32 logits at seq16384/mb1 ≈ 9.96 GB + grad. Chunk it.
4. **Stamp θ + any scale factor into checkpoint metadata + the ledger run entry.** RoPE is a non-persistent buffer rebuilt from `cfg` at init (`model.py:224`) — a resumed run silently rebuilds the cache at whatever θ is loaded and will "look fine while being wrong" if θ isn't recorded.

**Data & budget:** **~200–300M tokens** at seq 8192. **25–30% long documents** (concatenated code repos + books + arXiv — ProLong's best long sources) **interleaved with 70–75% short-context premium replay** to protect short-context (literature says ≤2B tokens suffice to extend, so this is ample at our scale). Doc-masking ON via FA2-varlen/flex per fix #2.

**Eval (reuse §C25):** the **RULER ladder** (center rungs above 4096) for **effective context length (ECL: longest rung ≥ 0.85× short-ctx)**; **NoLiMa indirection** (expected near-floor — report honestly); **short-context NON-regression** on the 4096 battery (the win gate — LongRoPE2/Phi-3 warn naive extension can drop MMLU ~7.6 pts); **per-position loss tail**. **Controls:** (a) the un-extended base on the same ladder = the zero-line (it should collapse past its usable window); (b) an iso-token seq-4096 continuation = isolates "trained-at-length" from "more training." **Gate on the ladder + indirection + non-regression — NOT single-needle NIAH** (which would overstate).

---

## 4. Sequencing + cost

**One GPU job at a time. Full §C5 on every launch:** sentinel preflight → smoke-test (1 step, exit 0) → iso-FLOP/iso-token matched arms → `sentinel.py watch` kill-switch armed beside the run → evidence written to the ledger **before** launch → `safe_cuda.guard(0.85)` armed → chunked CE.

| # | Step | Type | GPU-days | Gate / note |
|---|------|------|----------|-------------|
| 0 | **RULER-ladder diagnostic** on the existing base (build the loader first) | eval-only, on-box | ~0.3 | **GATES step 2.** Resolves "does the base use 4096?" + sets the short-ctx non-regression reference. No training. |
| 1 | **Anneal A/B**: treatment (premium 50/50) + control (FineWeb-Edu), WSD decay, 3 seeds each (6 runs × ~150–200M tok @ ~6,800 tok/s, seq4096/mb4) | train, on-box | ~1.9 (incl. evals) | Runs regardless of step 0. Proven shape, zero new OOM. Safe first launch to validate the §C5 pipeline. While it trains, build the bf16-RoPE fix + FA2-varlen/flex + backend smoke-test for step 2. |
| 2 | **Context-extension**, 1-seed directional: CPT seq 8192, θ=1e6 unchanged, FA2-varlen doc-mask, chunked CE, bf16-RoPE fix (~200–300M tok @ ~4,500 tok/s) + ladder/NoLiMa eval | train, on-box | ~1.5–2.0 | **Only if step 0 passes.** Then 3-seed **only if** seed-0 clears short-ctx non-regression **and** ECL longest rung ≥ 0.85× short. |

**Order rationale:** step 0 is eval-only and gates step 2, so it goes first. Anneal (step 1) is the cheaper, lower-risk, on-disk-grounded run on the proven shape — launch it next to exercise §C5 while the riskier context-extension plumbing (three code fixes + two smoke-tests) is built. Context-extension is the capability *headline* but the riskier *engineering*, and it's gated — so it runs last.

**Cost totals (on-box, serialized on one GPU):**
- **Recommended (anneal 3-seed + ctx-ext 1-seed + diagnostics): ~3.5–4.0 GPU-days ≈ 4–5 calendar days.**
- **Full 3-seed both arms: ~6–7 GPU-days ≈ 6–7 calendar days.**
- **Budget-forced single pick:** do **context-extension** (the clear non-null capability) if step 0 passes; run the anneal as a single iso-token A/B pair (~0.7 GPU-day), reported as **directional matched-domain code-BPB only.**

**On-box vs propose-only / rented:**
- **On-box (all of the above):** every training + eval run. No rented compute this stage.
- **Propose-only (§C20 — do NOT build here):** WSM checkpoint-merging (2507.17634, not fetched); YaRN / LongRoPE2 large-extension machinery; any 32K+ target; PolicyLong on-policy long-data synthesis (2604.07809); any multi-scale × multi-data × multi-seed grid that would need a rented cluster.

---

## 5. Pinned facts + every method carried

**Pinned code/data facts (verified against on-disk files this session):**
- `model.py:45` `rope_theta=1_000_000.0`; `model.py:44` `max_position_embeddings=40_960`; RoPE cache fp32-built to 40,960 (`model.py:78-82`), non-persistent buffer rebuilt from cfg (`model.py:224`). → no RoPE rescaling needed at 2–4×; θ must be stamped into ckpt metadata.
- `model.py:162-166` SDPA, `is_causal=(attention_mask is None)` → causal path = O(seq) memory (flash/mem-eff); dense mask → O(seq²) math backend.
- bf16-RoPE rotation: cos/sin downcast to bf16 at `model.py:230-231`, rotation in `model.py:94-95` → fp32-rotation fix required > 4K.
- Data-mix anchor (`.../2026-06-26_qwen3-0.6b_data-mix-composition/verdict.json`, n=3): **mix code_py +0.5904 BPB CI[+0.5532,+0.6277] significant**; **mix wikitext2 +0.0161 CI[−0.0005,+0.0327] not significant** (vs FineWeb-Edu baseline, iso-token **full continued-train, not anneal**).
- §C25 mid-training eval row exists (`per_stage_eval_batteries.md:25`); **RULER/NoLiMa/MRCR loaders + ECL + mid-training run-type are still build-backlog** (`:120`) — the eval *spec* is done, the *loaders* are not.

**Methods carried (all survived the adversarial check):**

| Method | Role | Source | Date | Status |
|---|---|---|---|---|
| MiniCPM / WSD (warmup-stable-decay, decay-stage HQ data) | anneal schedule/shape | 2404.06395 | 2024-04 | current-best ✓ (C-Eval 40.0→52.6 exact digits **not re-fetched** — use mechanism, not the number) |
| Cooldown-dynamics of WSD | decay shape (sqrt/linear, last 10–20%) | 2508.01483 (TMLR) | 2025-08 | usable ✓ (exact "lowered-linear-0.7" name + "8–12% insensitivity" = **search-surfaced, do not pin**) |
| Llama-3 herd (anneal scale verdict) | scale evidence: 8B **+24.0% GSM8k / +6.4% MATH**, 405B **negligible** | 2407.21783 | 2024-07 | usable ✓ for the **scale verdict only**; ⚠️ **anneal token budget DROPPED** — "40M" vs "40B" is contested between verifiers, not used as a budget floor |
| LR-decay-wastes-best-data (Luo et al., curriculum pretraining) | honesty anchor: don't strand premium data under collapsed LR; decay to non-zero floor + averaging | 2511.18903 | 2025-11 (v3) | usable ✓ — **CORRECTIONS:** it's **1.5B params / 30B tokens** (the "32B" tag was wrong); full title "...in **Curriculum-based** LLM Pretraining"; finding is curriculum-specific (+1.64% avg); ⚠️ **"penalty larger for small models" = FABRICATED attribution, DROPPED** |
| ProLong — How to Train Long-Context LMs (Effectively) | context-ext recipe: CPT-at-length, code+books, intra-doc masking (helps short+long), FA2-varlen | 2410.02660 (ACL 2025) | 2024-10 | current-best ✓ |
| bf16-RoPE / AnchorAttention | the fp32-rotation fix | 2411.13476 | 2024-11 | usable ✓ |
| Mid-Training survey (Mo et al.) | "retain pretraining subset" rule | 2510.06826 | 2025-10 | usable ✓; ⚠️ **per-model anneal-% (2/9/17%) PDF-pending + internally inconsistent — do not pin** |
| codelion "Scaling Pedagogical Pre-training to 10B" | small-scale within-noise floor (31.42–32.38) | HF blog | 2025–2026 | usable ✓ (secondary source — directional) |
| YaRN | contingency only if ever >32K | 2309.00071 | 2023-09 | usable but **NOT NEEDED at 2–4×** |
| LongRoPE2 | reference for large-extension SOTA | 2502.20082 (ICML 2025) | 2025-02 | real ✓ but **OVERKILL / propose-only**; RULER 82.03/73.40/49.39 digits = **secondary, do not rely** |
| SmolLM2 | anneal reference | 2502.02737 | 2025-02 | usable; ⚠️ **58/24/14/4 anneal split = secondary, DO NOT rely** |
| Olmo 3 | mid-training reference | 2512.13961 | 2025-12 | usable ✓ (exists) |

**Dropped / demoted (build must NOT rely on these):** PI / NTK-aware scaling (superseded for θ=1e6 at 2–4×); θ-bump 1e6→2–4e6 for 8K (unnecessary — no RoPE OOD at our headroom; keep only as a contingency if pushing toward 32K); WSM checkpoint-merging 2507.17634 (not fetched → propose-only); PolicyLong 2604.07809 (out-of-scope → propose-only); the "~2–6% relative BPB anneal gain" (un-cited hypothesis, not a literature value); dense-`(S,S)` doc-mask on SDPA (forbidden — OOM); any single-needle-NIAH "context works at 8K" claim (gate on the RULER ladder + NoLiMa + non-regression instead).
