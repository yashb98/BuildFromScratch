<!-- §C27 stage plan — Generated 2026-06-27 by workflow wf_fb376816-3a4 (inline pattern). 3 personas recency-first + adversarially verified. Stage VERIFIED = post-training. -->

research_synthesized_on 2026-06-27 · base verified on disk · chair synthesis of {Post-training RS, Frontier ME, MLOps} + adversarial pass

# Qwen3-0.6B Post-Training: ONE Buildable Plan

**Scope.** Add lifecycle Ch.9–11 (SFT → preference → RLVR) to the **faithful** Qwen3-0.6B base (596M params, README PPL 28.65), pretrained from scratch on ~1.19B FineWeb-Edu (non-math) tokens. One GB10 box, ~119 GB unified CPU+GPU pool, one GPU job at a time. Use the faithful checkpoint (not IMU-1) for clean single-variable attribution. All three personas and both adversarial reviews **converge**; what follows drops every flagged item and flags every un-measured one.

---

## 1. HONEST VERDICT (per stage, at 596M / 1.19B tokens)

| Stage | Real gain or near-null? | Worth the GPU-day? | What it actually is |
|---|---|---|---|
| **SFT** | **REAL** (modest, in-domain) — but it is **distillation/imitation** of teacher reasoning traces, not emergent reasoning | **YES — the only stage that buys usable behavior** | Continued-pretrain + instruction-format acquisition on traces the base has literally never seen |
| **Preference** (DPO/ORPO/SimPO/KTO) | **NEAR-NULL** capability | No (run **cheap, once**, for rigor) | Style/length/format re-ranking on a barely-instructable base; methodology-completeness checkbox |
| **RLVR** (GRPO→Dr.GRPO) | **NEAR-NULL → TRAP** | No as a capability bet; **YES as a controlled negative result** | A 1.19B-token non-math base has ~no latent reasoning to *elicit*; any apparent bump is most likely format-shaping/noise |

**Ranking of GPU-day value: SFT ≫ preference ≳ RLVR.** The post-training arc here is **primarily a methodology/rigor demonstration with exactly one genuine capability lever (SFT-as-distillation).** Preference and RLVR must be framed as method/rigor demos, never capability wins.

**The honesty anchor (verified on disk, not a claim):** the existing n=1 SFT's eval-harness §C10 reasoning delta is **−19.2% (PPL 17.60→14.22 on `openr1_math_220k_heldout`)** but the single-run noise floor is **25.2%**, so it is currently `significant=false, "not significant"`. The whole job is to convert this n=1 into a ≥3-seed **paired CI that excludes 0** — until then SFT's win is *inconclusive*, not established.

**Why RLVR is a trap, not a bet (both papers verified):** *elicits-not-expands* (2504.13837) shows RL only re-samples paths already in the base and the base **wins at large pass@k**; *Spurious-Rewards* (2506.10947) shows random/format rewards lifted Qwen2.5-Math-7B (**+21.4 MATH-500, random reward** — verified) *only* by amplifying that model's **latent code-reasoning**, and the effect **does not transfer to weaker/non-math bases (Llama3/OLMo2)**. Our FineWeb-Edu base lacks those priors, so even the spurious channel is probably absent. Decision rule is baked in below.

---

## 2. THE EXPERIMENTS (matched control arm is non-negotiable, per stage)

All losses import the **unit-tested `research/posttrain_losses.py`** (verified present): `masked_sft_nll`, `dpo_loss`, `group_normalized_advantages`, `grpo_clipped_objective`, `kl_penalty_k3`, `grpo_loss`, `label_smoothed_nll`. Executor = `/post-train` → launches only via `/ablation-runner` (§C5/§C11). Every script starts `import safe_cuda; safe_cuda.guard(0.85)` + chunked large-vocab CE (vocab **151,936** > 64k). Evals = the **already-researched §C25 battery** (do NOT re-research).

### 2A. SFT — upgrade the n=1 to a ≥3-seed verdict (PRIORITY)

**Recipe (reuse the proven n=1 config — keep variables minimal):** response-masked CE (`masked_sft_nll`, prompt tokens → −100; keep — prompt-token loss is objective misspecification); **peak LR 5e-5 cosine→8e-8, ~5% warmup (warmup=11), global_batch 128 (micro_batch 4 × grad_accum 32), AdamW(0.9,0.95) wd 0.01 (zero wd on embeddings, OLMo2), grad-clip 1.0, bf16, label-smoothing 0**; 1–3 epochs over the OpenR1-Math-220k teacher-distilled set (so this SFT *is* distillation). **Throughput upgrade (cheap):** the n=1 used EOS-separated flat shards — switch to **sample packing with block-diagonal / intra-document attention masking** (current-best small-model practice). NEFTune is **SKIPPED** for the clean attribution run (it boosts conversational *format*, not capability) — keep it only as a later separate single-variable arm.

**≥3-seed plan:** re-run **3 seeds** varying data-shuffle seed + init-noise seed → **paired CI excluding 0** against the 25.2% noise floor. Score every seed with §C25: BPB on ≥2 corpora, the **forgetting probe**, **IFEval prompt-strict (programmatic)**, and **add a GSM8K exact-match accuracy probe** (current evidence is in-domain PPL on the teacher's own distribution — the weakest possible "capability" signal).

**Matched controls (two, both mandatory):**
1. **Frozen faithful base**, identical eval/seeds — already on disk, **free**.
2. **ISO-FLOP / matched continued-pretrain control** — continue-pretrain the base on the **same 124.78M-token budget, same data, UNMASKED, no instruction template** → attributes any reasoning gain to **SFT-format vs just-more-tokens**.

**Already done — do NOT re-plan (adversarial correction):** the **FineWeb-Edu forgetting probe already exists** in `eval/brief_probes_results.json` (base 24.55 → target 24.73, **+0.74%, retained, not significant**, floor 33.9%). Note this is base-vs-target *on the same corpus*, **not** "vs 28.65" (28.65 is the README headline on a different corpus — do not conflate). Reuse this probe; just add it to each new seed.

### 2B. Preference — ONE cheap rigor demo (do not multi-seed until SFT is locked)

**Pick: ORPO** (single monolithic stage, **reference-free, no separate SFT** → removes the frozen-reference copy from the pool; tested 125M–7B so 596M is in-band). Alternatives: **SimPO** (ref-free, length-normalized avg-logprob reward) or **DPO** (`dpo_loss`, frozen ref ~2.4 GB fp32 — trivial here; cache ref-logprobs to disk once → ~1× model at train time). **No reward model needed** for any of these — they need preference *pairs* (UltraFeedback-style) or binary labels. **Data gap to resolve:** at 0.6B, on-policy-sample-then-judge yields weak signal; source pairs explicitly.

**Matched control = the SFT checkpoint** (preference must beat **SFT**, not the base). Score: **held-out preference-accuracy Wilson CI + length-controlled win-rate** (kills the DPO/SimPO length-gaming confound) **+ reward-hacking probe + KL-to-ref frontier**.

### 2C. RLVR — FIX the executor first, then run a CONTROL-ARMED screen

**FIX, do not keep (verified on disk):** `group_normalized_advantages` computes `(r − mean)/(pstdev + eps)` = **std-biased GRPO** (over-weights low-variance/easy questions; per-response length normalization inflates length). **Switch to Dr.GRPO** (2503.20783): **subtract group mean only, do NOT divide by std, drop per-response length normalization** (constant token-loss normalizer). One-line change to default the RLVR path. Add **DAPO dynamic sampling** (drop all-correct/all-wrong groups) — load-bearing here because a weak base produces mostly **all-wrong** groups whose advantages collapse to zero. KL via `kl_penalty_k3`, β small (0.0–0.001). GSPO (sequence-level ratio) is **usable-not-needed** (MoE/large; our model is dense 596M); DAPO clip-higher add-ons were **32B-tuned** (optional).

**On-box GSM8K recipe:** group G=8–16, ~64–128 prompts/step, exact-match on boxed answer, max_new_tokens ~512, clip 0.2, ~300 steps for the screen.

**THE LOAD-BEARING CONTROL ARMS (spurious-reward):** run **three identical arms — (i) real verifiable reward, (ii) RANDOM reward, (iii) FORMAT-only reward** — same base, same prompts, same group size; **the reward function is the ONLY variable.** Plus **pass@k-vs-pass@1 divergence vs frozen base** (Chen-2021 unbiased pass@k; pass@1↑ but pass@256↓ ⇒ *elicits-not-expands* confirmed) **+ verifier-honesty/IPT**. Significance via `eval_metrics.py` (McNemar/Wilson on pass@1).
**DECISION RULE:** if **random ≈ real**, the verdict is **"trap demonstrated,"** NOT "capability gained." Only if real **significantly beats both** controls does RLVR earn a ≥3-seed confirm.

---

## 3. SEQUENCING · GPU-DAYS · ON-BOX vs PROPOSE-ONLY

**Sequence:** (1) SFT ≥3-seed + iso-FLOP control → **lock the one real lever** → (2) ORPO/SimPO single demo arm on top of SFT → (3) **fix std-biased GRPO → Dr.GRPO + DAPO dynamic sampling**, then the 3-arm RLVR screen. Do not invest RLVR multi-seed unless the screen shows real > both controls.

| Stage | On-box? | GPU-days | Confidence |
|---|---|---|---|
| **SFT** 3 seeds + iso-FLOP control | **YES — proven** (measured: ~7,000–7,459 tok/s, peak **52.4 GB** < 71.4 GB cap, **297.3 min = 4.95 h/seed** on disk) | ~**0.85–1.5** (base control free) | **measured** |
| **Preference** (ORPO/SimPO, 1 demo arm, light multi-seed) | **YES** — ref-free or cached-ref, memory-trivial in 119 GB | ~**0.4–0.5** | solid |
| **RLVR** screen: 3 arms (real+random+format), 1 seed, ~300 steps | **FEASIBLE-PENDING** (see flags) | ~**3** (screen); ~**9–12** only if it earns a ≥3-seed confirm (unlikely) | **UNMEASURED** |
| **Full screening pass** | | ~**4.5 GPU-days** | mixed |

**On-box vs propose-only (the real engineering risk):**
- SFT + preference: **on-box, no §C20 trigger.** Memory is never the binding constraint for a 0.6B in a 119 GB pool.
- **RLVR is on-box-FEASIBLE but PENDING VALIDATION, not proven.** Three un-measured/un-demonstrated risks, all flagged honestly:
  1. **GB10 platform risk (UNVERIFIED on this box):** on-box rollouts rest on **vLLM colocate + sleep()-L2** to free vLLM weights+KV during the optimizer step. That mechanism is verified on **x86/CUDA**, **not demonstrated on GB10 (aarch64 + Blackwell)**. If immature, RLVR **degrades to propose-only (§C20)** — and that would change the headline. **No rollout engine is currently wired on-box.**
  2. **Rollout-throughput cost is UNMEASURED:** the GPU-day figures use **training** throughput (~6,800–7,400 tok/s) as a proxy for **autoregressive generation** throughput — different regimes. The ~3-GPU-day screen is an estimate, optimistically framed.
  3. **Zero-advantage collapse under-costs the screen:** at ~0–2% base GSM8K accuracy, **~85% of G=8 groups are all-wrong** → dynamic sampling discards them → you need **~6–7× oversampling** to fill a batch, which the ~3-day estimate does not budget. Either the screen costs materially more, **or it produces almost no learnable batches — which is itself the negative result.**
- **Systems guards (mandatory):** cap vLLM `gpu_memory_utilization` ~0.3–0.4 (colocate reserves a fixed chunk upfront — trainer+ref+vLLM can exceed the shared pool and **hard-crash the whole box**), `safe_cuda.guard(0.85)`, chunked CE, one-GPU-job sentinel (§C4.5), serialize rollout→score→update.

---

## 4. PINNED FACTS + METHOD LEDGER

**Pinned, verified on disk (2026-06-27):**
- Executor `group_normalized_advantages` = `(r − mean)/(pstdev + eps)` → **std-biased GRPO** (confirmed). KL primitive is **`kl_penalty_k3`** (not `kl_penalty_k`).
- n=1 SFT: OpenR1-Math-220k, **124.78M tok, 238 steps, peak_lr 5e-5→8e-8 cosine, warmup 11, global_batch 128 (micro_batch 4 × grad_accum 32)**, peak **52.4 GB**, ~7,000–7,459 tok/s, **297.3 min wall-clock**.
- Reasoning (eval-harness §C10, `openr1_math_220k_heldout`): **17.60→14.22, −19.2%, floor 25.2%, NOT significant.** Training-log internal eval (different split): 14.26→11.60, −18.7%. **n=1 verdict = inconclusive.**
- Forgetting (FineWeb-Edu, §C13): **24.55→24.73, +0.74%, retained, not significant** — **already done.**

**Corrections applied (dropped per adversarial):**
- Dropped the **llm-stats blog** as the citation for SFT response-masking/packing (substantiation mismatch — the blog does not discuss it); real support is **SmolLM2/OLMo2** practice (their specific arXiv IDs are ⚠UNVERIFIED here — technique itself is standard/current-best).
- Dropped the **"+13.8 MATH-500 format-reward"** figure (⚠UNVERIFIED; only the **+21.4 random-reward** number is confirmed).
- Dropped **"4.65 h/seed measured"** (that was a derived ETA from a different run) → use measured **4.95 h**.
- Dropped Dr.GRPO **"COLM 2025"** venue (⚠UNVERIFIED).
- Corrected "add FineWeb-Edu forgetting probe" — **it already exists**; and corrected "vs 28.65" framing (probe is same-corpus base-vs-target).

**Method ledger (every method fetch-verified unless flagged):**

| Method | arXiv | Date | Status | Use here |
|---|---|---|---|---|
| SFT response-masked CE + sample-packing w/ intra-doc attn mask | — (SmolLM2/OLMo2 ⚠UNVERIFIED source IDs) | 2025–26 | current-best | **SFT baseline** |
| NEFTune (noisy embeddings) | 2310.05914 | 2023-10 | usable | **SKIP** (format not capability; later separate arm) |
| DPO (closed-form, frozen ref) | 2305.18290 | 2023-05 | usable | preference fallback |
| SimPO (ref-free, length-normalized) | 2405.14734 | 2024-05 | usable | preference alt (+6.4 AlpacaEval2/+7.5 Arena-Hard on **8–9B instruct** ⚠number partly UNVERIFIED, near-null transfer at 0.6B) |
| KTO (binary feedback) | ⚠UNVERIFIED ID | 2024-02 | usable | preference alt (no pairs) |
| **ORPO (single-stage, ref-free)** | 2403.07691 | 2024-03 | **current-best operational** | **preference pick** (memory win, tested 125M–7B) |
| GRPO (DeepSeekMath, std-biased) | 2402.03300 | 2024-02 | **executor's current — FIX** | replace normalizer |
| **Dr.GRPO (remove /std + length norm)** | 2503.20783 | 2025-03 | **current-best fix** | **RLVR objective** |
| DAPO (clip-higher, dynamic sampling, token-level, drop-KL) | 2503.14476 | 2025-03 | usable (32B-tuned) | **dynamic sampling** only |
| GSPO (sequence-level ratio, Qwen3) | 2507.18071 | 2025-07 | usable-not-needed | skip (MoE/large) |
| **Spurious Rewards (random/format control mandate)** | 2506.10947 | 2025-06 (rev 2026-02) | **current-best** | **RLVR control arms** |
| **elicits-not-expands (pass@k crossover)** | 2504.13837 | 2025-04 | **current-best** | RLVR pass@k check |
| Limits of Difficulty Scaling (sub-1B GRPO diminishing returns) | 2604.06298 | 2026-04 | usable (post-cutoff, verified real) | scale-honesty evidence |
| TRL co-located vLLM + sleep()-L2 rollout engine | — | 2025-09 | usable | RLVR rollouts (⚠UNVERIFIED on GB10) |

**Bottom line for the build:** ship **SFT ≥3-seed + iso-FLOP control** (the one real, on-box, measured lever) and the **Dr.GRPO executor fix** regardless. Run preference (ORPO) and the RLVR 3-arm screen **as rigor/negative-result demos with their controls baked in**, expecting near-null/trap — and treat on-box RLVR as **feasible-pending GB10 vLLM-colocate validation**, falling back to propose-only (§C20) if the rollout engine doesn't stand up on aarch64/Blackwell.
