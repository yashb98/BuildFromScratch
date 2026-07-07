<!-- §C27 per-stage method-research plan for the `rlvr` (reasoning-RL) lifecycle stage.
Produced by the stage-method-research workflow (run wf_5cf3b0d7-5f1), 3 personas
(RS / frontier-eng / MLOps) recency-first + adversarially verified, chair-synthesized.
Adversarial verdicts: rs=needs-fixes, fe=solid, mlops=needs-fixes — the chair below DROPPED
every fabricated/on-disk-claim the skeptics caught and reconciled the Dr.GRPO↔Lite-PPO conflict.
This is the §C27.1 "no plan, no run" gate for GRPO's first run. researched_on 2026-07-01. -->

Verified on disk. Confirming the adversarial flags: `research/datasets/` holds only `math-reasoning-openr1-math-220k` (the SFT set) + `data-selection-dclm-edu` — no GSM8K, MATH-500, or DeepScaleR; eval-harness `suite.md` has no boxed/exact-match/pass@k (only text-lm-v2/v3 PPL+MC); GSM8K exists only in an external project's HF cache (`/home/yashb98/projects/qwen-distill/hf_cache/.../openai--gsm8k`), not prepared in-repo. The 6 SFT+ctrl checkpoints and all GRPO loss primitives exist. Note `spurious_reward_control_gate` is a REQUIRED §C25 rlvr item. Synthesis below.

---

# RLVR stage plan — Qwen3-0.6B faithful build (§C27 method-research, researched_on 2026-07-01)

## 1. HONEST verdict up front

**At 0.6B on our ~10×-undertrained FineWeb-Edu→SFT base, RLVR is a methodology/negative-result artifact, not a reasoning win.** RLVR (GRPO family) only *sharpens* solutions the policy can already sample; it does not add capability (Yue 2504.13837; RLVR-vs-Distillation 2505.14216). Our SFT'd base has near-zero native math and an unmeasured — probably near-floor — pass@k in the eval band, so there is little for policy gradient to concentrate probability onto. The most likely *true* outcome is a null-or-tiny attributable Dr.GRPO gain that the control battery should correctly **refuse to call a win** — exactly the honest shape of our SFT-masking result (directional, Δ0.0092 PPL).

**Where the reasoning actually lives:** the SFT/distillation step, not RL. Every literature "RL win" at small scale started from a *math-pretrained or R1-distilled* base (DeepScaleR-1.5B, Tina) — none is a raw-0.6B counter-example. And because our base shares the **Qwen tokenizer/lineage**, Spurious Rewards (2506.10947) is directly on-point: Qwen-family models post large MATH gains even from *random* rewards via prior elicitation — so any headline not beaten down by a random-reward control is probably not reasoning.

**Worth the GPU-day:**
- **Measuring the SFT'd checkpoints' pass@1 / pass@k baseline** — the go/no-go gate. If pass@k≈0 in the eval band, GRPO is null-by-construction and we stop (cheap, decisive).
- **ONE exploratory Dr.GRPO seed + its iso-compute RFT control + the required random-reward gate** — to *measure the delta* with rigor. Format/extractor-reward shaping (making outputs parseable) is the one plausibly-real point or two.

**Near-null / do NOT spend GPU on:** GRPO on hard MATH-500/AIME (nothing to sharpen), long-context RL (8k→24k), DAPO's full system, GSPO (MoE-oriented), multi-domain RL, and any headline "reasoning win" attributed to RL alone. These are propose-only-if-ever, off-box.

**Blocking caveat (why this is not runnable today):** the mandatory decision metric — held-out GSM8K/MATH-500 exact-match **pass@1 + pass@k** — **does not exist**: no boxed/exact-match/pass@k capability in eval-harness, and neither the GRPO prompt set nor the eval sets are prepared on disk. These are prerequisites, counted below.

---

## 2. The experiment + matched controls

**Executor route:** `/post-train grpo Qwen3-0.6B vibethinker-small-reasoning` → §C5.0 smoke test → `/ablation-runner` (the §C11 launch monopoly, full §C5 bounded-auto-run) → `/eval-harness`. Loss = the already-unit-tested `research/posttrain_losses.py` primitives. Ledger run `type=finetune`, `lifecycle_stage=rlvr`, technique slug `vibethinker-small-reasoning`.

**Policy init + frozen reference:** the SFT'd `checkpoint_sft_seed{0,1,2}.pt` (NOT the pretrain base — RL on the raw base is hopeless). Reference = a frozen copy of the same SFT checkpoint (KL anchor).

**Algorithm — Dr.GRPO on the built primitives, three fetch-justified deltas (conflict resolved):**
1. **Drop per-group std** (Dr.GRPO 2503.20783): add a flag to `group_normalized_advantages` to mean-center only and divide by a **fixed constant** (not `/(std+eps)`), removing the difficulty-weighting bias.
2. **Token-level global-mean loss aggregation, no per-sequence-length division** (Dr.GRPO's length-bias fix; **and** Lite PPO 2508.08221, which recommends token-level aggregation — "particularly effective for base models"). *This is the corrected reading: both papers point to token-level + dropping per-group std, so the two "current-best" deltas do **not** conflict once Lite PPO is characterized correctly (its recipe is group-mean centering + batch-level std + token-level loss, NOT the sequence-level/std-norm inversion in the raw draft).*
3. **DAPO dynamic sampling** (2503.14476): before each update drop prompts whose G samples are all-correct or all-wrong (zero advantage) — essential at 0.6B where most groups are all-wrong.
- **KL:** keep the k3 KL toward the frozen SFT reference, β≈1e-3–1e-2 (**coefficient INFERRED**, not paper-pinned), as a leash against verifier gaming; β=0 only as an ablation if entropy stays stable.
- **Clip:** symmetric 0.2 baseline; DAPO clip-higher (0.2/0.28) added **only if** `grpo_health` shows entropy collapse.
- **Hyperparams (INFERRED, small-model RLVR):** G=8, prompts/step 16–32, completion cap ≤1–2k tokens (NOT DeepScaleR's 8k–24k), LR≈1e-6 AdamW, ~300–500 updates, 3 seeds.

**Reward:** binary exact-match on the extracted `\boxed{}` final answer via a **SymPy-normalized equivalence checker** (NOT string / first-number match) + small format reward for a parseable answer. Verifiable-only, no neural RM. Extractor **pinned + versioned**; adversarially fuzzed (permissive-vs-strict differential test, measure false-positive rate) as a **pre-train diagnostic** — the `extractor_pinned` + `verifier_honesty_ipt` §C25 items — BEFORE any GPU spend.

**Arms (all paired by seed, ≥3 seeds):**
| Arm | What | Isolates |
|---|---|---|
| **A — RL** | Dr.GRPO from `checkpoint_sft_seed_i` | the RL effect |
| **B — matched control (primary)** | iso-compute rejection-sampling SFT (RFT/STaR/best-of-n): same checkpoint, same rollout budget (G×prompts), same pinned verifier, keep verifier-correct completions, SFT via `masked_sft_nll`, no policy gradient | whether on-policy advantage beats best-of-n SFT at iso-generation-compute (Yue predicts ≈parity at 0.6B) |
| **C — random-reward gate (REQUIRED §C25)** | identical pipeline, rewards randomized (and/or format-only) | if C ≈ A, the "gain" is prior elicitation (Spurious Rewards), not reasoning → refuse the win |
| **Floor** | frozen `checkpoint_sft_seed_i` | absolute baseline |
| **Ablations (§C18, single-variable)** | no-KL (β=0); Dr.GRPO length-fix on vs off | stability / length attribution |

**Decision metric (win definition):** held-out **GSM8K + MATH-500 exact-match pass@1 (Wilson CI) AND pass@k (k=8,16, Chen-2021 estimator)** — pass@k exposes the Yue signature: a real win raises pass@1 *without collapsing pass@k below the SFT floor*; a pass@1 bump with flat/narrowed pass@k is pure sampling-efficiency, not capability. ≥3 seeds, **paired-difference CI excluding 0** (§C17), iso-FLOP ≤5% (§C18). Plus §C13 forgetting probe (wikitext/code PPL vs SFT base, no regress past the noise floor). **PPL cannot separate arms** — the SFT cohort proved it (masked 11.573 vs 11.582) — so exact-match accuracy is the *only* decision metric.

---

## 3. Sequencing, GPU-days, on-box vs propose-only

**Phase 0 — prerequisites, CPU only, on-box (~0 GPU-days, real calendar time). BLOCKS everything:**
- **P1 `/dataset-forge` GRPO prompt+answer set:** GSM8K-train (pull from the on-box HF cache `openai/gsm8k`, prepare in-repo) + a MATH level-1–3 band (the easy band where the SFT'd 0.6B has non-zero pass@k so groups carry gradient). Decontaminate (13-gram + embedding) vs the eval sets AND the SFT set.
- **P2 `/dataset-forge` decision-metric eval sets:** GSM8K-test + MATH-500 (`HuggingFaceH4/MATH-500`), held out, decontaminated. *(Neither is on disk today.)*
- **P3 BUILD math-accuracy in `/eval-harness`:** boxed extraction + SymPy exact-match + pass@1 (Wilson) + pass@k (Chen-2021, k=8,16). Version-stamp a new suite (e.g. `math-acc-v1`). *(Does not exist — the run is inconclusive-by-construction without it.)*
- **P4 pin + fuzz the extractor/verifier** (`extractor_pinned`, `verifier_honesty_ipt`).

**Phase 1 — go/no-go, on-box GPU (~0.25–0.5 GPU-day):** measure the SFT'd checkpoints' pass@1 + pass@k baseline on the eval band (pass@k=16 ⇒ 16 completions/eval-prompt — a real generation cost, folded in here). **If pass@k≈0 → STOP**, write the directional/null result; RL cannot help.

**Phase 2 — exploratory delta, on-box GPU (~2–4 GPU-days):** ONE Dr.GRPO seed (~1–2 GPU-day, rollout-generation-dominated) + ONE iso-compute RFT control seed (~0.5–1) + the random-reward gate arm (~0.5). Go/no-go: does the Dr.GRPO delta over RFT clear the noise floor at iso-compute, and does the random-reward arm stay *below* it?

**On-box budget total ≈ 3–5 GPU-days** (Phase 1+2) once Phase 0 lands. *(Honest at this small-probe config: 16 prompts × G=8 × ~300–500 steps × ~1k gen. The GPU-day figure inflates 3–5× if the config is pushed to the aggressive end (64 prompts × G=16 × 800 steps × 2k gen ≈ 1.6B generated tokens) because decode is bandwidth-bound — keep the config small.)*

**Phase 3 — PROPOSE-ONLY / OFF-BOX via `/remote-launcher` (human authorizes spend), ONLY if Phase 2 clears the floor:** the full 3-seed paired-CI cohort (3 Dr.GRPO + 3 RFT + random-reward + no-KL/no-length-fix ablations). Rent to parallelize seeds → CI in ~1–2 days; on one GB10 serialized it is ~6–10 GPU-days / ~1.5–2.5 weeks (§C4.5, one GPU job at a time).

**Split summary:** ON-BOX = Phase 0 (CPU forge + eval build) + Phase 1 (SFT baseline) + Phase 2 (1 exploratory GRPO + RFT + random-reward). OFF-BOX/propose-only = Phase 3 multi-seed CI cohort. **NEVER on-box/at-scale:** DeepScaleR-scale or long-context (8k→24k) RL, multi-domain RL, a new from-scratch build. The GB10 is the control plane, not the RL scale box.

**Box-safety (non-negotiable):** `import safe_cuda` before torch + `safe_cuda.guard(0.85)`; `sentinel watch --pid <PID> --kill-at 0.83`; chunked CE for V=151,936 (§C1, the 2026-06-08 crash pattern); one GPU job at a time; `setsid nohup` detached launch capturing PID; §C5.0 smoke test (1 GRPO step, 4 prompts×G=4, 128-tok cap: rollout finite, reward∈{0,1}, advantages finite, k3-KL≥0, ckpt save→reload, exit 0) → write `c5_evidence.json` + ledger entry BEFORE launch.

---

## 4. Pinned facts + methods (verified source+date; UNVERIFIED flagged)

**Pinned on-disk facts (cross-checked this pass):**
- Model: 596,049,920 params (155,582,464 embed + 440,467,456 non-embed, tied); D=1024; 28 layers; 16 Q / 8 KV heads (GQA); head_dim 128; V=151,936; bf16.
- Checkpoints present: `Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/checkpoint_{sft,ctrl}_seed{0,1,2}.pt` (~3.58 GB each).
- Loss primitives present in `research/posttrain_losses.py`: `grpo_loss`, `grpo_clipped_objective`, `group_normalized_advantages`, `kl_penalty_k3`, `dpo_loss`, `label_smoothed_nll`, `masked_sft_nll`.
- SFT verdict (`reasoning_verdict.json`): held-out reasoning-set **masked PPL** base 14.13 → SFT 11.573 / ctrl 11.582; Δ(SFT−ctrl)=0.0092 masked (sig) but full-seq disagrees → **directional**. Forgetting (FineWeb-Edu PPL): base 21.495 / SFT 21.652 / ctrl 21.660. *(The "base wikitext 14.13" phrasing in the task maps on disk to `reasoning_ppl_masked` on the OpenR1-Math held-out split, not wikitext — corrected here.)*
- §C25 rlvr required battery (`research/eval_completeness.py`): `pass1_wilson_ci`, `passk_chen2021`, `spurious_reward_control_gate`, `verifier_honesty_ipt`, `extractor_pinned`, `grpo_health`, `seed_ci`. HARD-incomplete caps the verdict to `directional`.
- Datasets present: `research/datasets/{math-reasoning-openr1-math-220k, data-selection-dclm-edu}` only. GSM8K only in external `/home/yashb98/projects/qwen-distill/hf_cache` (not in-repo/decontaminated). **MATH-500 and DeepScaleR: absent — DROPPED from the plan (were falsely claimed "on disk").**
- eval-harness: `text-lm-v2` ACTIVE, `text-lm-v3` STAGED (LAMBADA+ARC-e+HellaSwag+Winogrande). **No math exact-match/pass@k — DROPPED the "/eval-harness exact-match as if live" claim; it must be built (P3).**

**Memory / throughput (verified arithmetic):**
- State: policy bf16 1.19 GB + grads 1.19 + AdamW fp32 m/v ~4.8 GB (~7.1 GB with an fp32 master copy) + frozen ref bf16 1.19 ≈ **8–11 GB** — trivial in the 119 GB pool.
- KV cache: 2×28×8×128×2 B ≈ **0.115 MB/token/seq**. **Cap total concurrent sequences × gen-len**: at a 1024-tok cap, ≤~130 sequences keeps KV <~15 GB. **The rollout batch is `prompts × G` TOTAL sequences (~16 prompts × G=8 ≈ 128), NOT `64–128 prompts × G`** — the latter (512–1024 seqs ⇒ 58–115 GB KV) is box-crash territory; micro-batch generation, storing completed rollouts as token-ids.
- The **52.4 GB peak @ mb4×4096 is a PRETRAIN activation measurement**, not GRPO; GRPO's own activation peak (policy fwd/bwd + ref fwd over ≤1–2k completions) is an unmeasured §C5.3-probe unknown — directionally lower (shorter seqs) but **measure before trusting the ~71 GB cap**.
- **Rollout generation is the binding constraint, not memory.** Autoregressive decode is memory-bandwidth-bound; GB10 unified LPDDR5X ~273 GB/s (vs A100 ~2 TB/s) → aggregate decode lands in the low-thousands tok/s. The measured ~7,300 tok/s is **training fwd+bwd** (compute-bound), NOT decode. **vLLM/SGLang on aarch64 sm_121 is UNVERIFIED** (bundled torch kernels reportedly stop at sm_120; community sm_121 build unvalidated) — **default to HF `generate` static-batching with a KV cap; treat vLLM as an optimization to VERIFY, not assume.**

**Methods carried (source · date · status):**

| Method | arXiv | Date | Status | Use here |
|---|---|---|---|---|
| **Dr.GRPO** — Understanding R1-Zero-Like Training (length + group-std bias fix) | 2503.20783 | 2025-03-26 | **current-best ✓** | **the algorithm** — drop per-group std, token-level loss |
| **DAPO** — clip-higher + dynamic sampling + token-level loss | 2503.14476 | 2025-03-18 | usable ✓ | borrow **dynamic sampling**; clip-higher only on entropy collapse |
| **Lite PPO** — "Tricks or Traps? A Deep Dive into RL for LLM Reasoning" | 2508.08221 | 2025-08 | current-best ✓ (corrected) | recipe = group-mean centering + **batch-level std** + **token-level loss**; small KL. *(Raw draft's "keep group-std / seq-level loss" was an inversion — FIXED. KL≈0.02 coefficient INFERRED, not paper-pinned.)* |
| **GRPO** — DeepSeekMath (group-relative advantage, critic-free) | 2402.03300 | 2024-02-05 | usable ✓ | family baseline; carries length + group-std bias |
| **Yue et al.** — "Does RLVR incentivize reasoning beyond the base model?" | 2504.13837 | 2025-04-18 | usable ✓ | effect-size anchor: sharpen-not-expand, pass@k narrows, ~6 algos equal |
| **RLVR-vs-Distillation** (Kim et al.) | 2505.14216 | 2025-05 | usable ✓ | anchor: RL +pass@1 but pass@k flat; distillation lifts both. *(exact 97.2→97.0 / +12.2 pts UNVERIFIED; pattern confirmed)* |
| **Spurious Rewards** — random rewards match GT on Qwen-family via prior elicitation | 2506.10947 | 2025-06-12 (v1) | usable ✓ | mandates the **random-reward control gate**; directly relevant (Qwen lineage) |
| **REINFORCE++** — global-batch advantage norm | 2501.03262 | 2025-01-04 | usable ✓ | simpler alternative to Dr.GRPO's std fix (fallback) |
| **RLOO** — "Back to Basics" (REINFORCE leave-one-out) | 2402.14740 | 2024-02-22 | usable ✓ | simplest critic-free variant (fallback) |
| **GSPO** — sequence-level importance ratio (Qwen) | 2507.18071 | 2025-07-24 (v1) | usable, low-relevance | MoE-oriented; skip for dense 0.6B. *(raw draft's 2025-07-29 corrected)* |
| **Tina** — Tiny Reasoning via LoRA-GRPO | 2504.15777 | 2025-04-22 | usable ✓ | NOT a counter-example — base is R1-Distill-1.5B (already distilled) |
| **DeepScaleR-1.5B-Preview** — GRPO + iterative context 8k→24k | (HF, 2502) | 2025-02-10 | usable ✓ | scale anchor: **~28.9→43.1 AIME** *(raw draft's 22.9→43 corrected)*; began from a distilled base on 32×A100 — does NOT transfer to our 0.6B |
| **VibeThinker** Spectrum-to-Signal (curriculum-SFT→verifiable RL→self-distill) | 2606.16140 | 2026-06-15 | **⚠ UNVERIFIED** (post-cutoff; brief-sourced `research/briefs/vibethinker-small-reasoning.md`) | repo anchor; importable core = SFT-then-GRPO with verifiable reward |
| Verifier-fuzzing / differential-fuzz-before-training (raw draft "2606.01066") | — | 2026 | **⚠ UNVERIFIED / DROPPED as a citation** (post-cutoff, single-source, conflated with Spurious Rewards) | the *practice* (pin + fuzz the extractor) is kept via the on-disk `extractor_pinned`/`verifier_honesty_ipt` battery, not the unverified paper |

**Dropped as fabricated/infeasible (adversarial):** "DeepScaleR-Preview on disk" and "MATH-500 on disk" (absent → must be forged, P2); "GSM8K/MATH-L1-3 GRPO prompts prepared" (absent → P1); "/eval-harness exact-match live" (absent → P3); "curriculum-SFT GSM8K 2%→20–30%" (unmeasured, in tension with the directional SFT verdict → replaced by the Phase-1 measured go/no-go); the Lite PPO normalization/loss-level inversion; and the Dr.GRPO↔Lite PPO "conflict" (dissolved by the corrected Lite PPO reading — both drop per-group std and use token-level loss).

---

## 6. Verification addendum (2026-07-01, adversarial re-check — workflow wf_be2a962d-bee)

Six fetch-verification agents re-checked every load-bearing claim against live sources (107 fetches).
**The §1 verdict and the Phase-1 go/no-go design SURVIVE.** Corrections + additions of record:

**Falsified as worded → corrected:** "no raw/non-distilled sub-1B RLVR win exists" is too strong.
**SimpleRL-Zoo (2503.18892)** ran zero-RL GRPO on the RAW Qwen2.5-0.5B base: GSM8K 36.7→49.5,
MATH500 15.8→34.4 (first-party GitHub results table). **SuperNova (2604.08477)** GRPO-trains
Qwen3-0.6B (the real 36T-token one) with gains persisting to k=128. **Corrected claim:** every
small-scale RLVR win starts from a base with SUBSTANTIAL latent task capability (trillion-token,
math-enriched pretraining — Qwen2.5-0.5B already scored 36.7 GSM8K before RL). No win exists on a
base with ~nothing to sharpen; our 1.19B-token 596M base is predicted to be exactly that — which is
what Phase 1 measures instead of assumes. (Note both counter-example bases are Qwen-lineage, so the
Spurious-Rewards elicitation caveat applies to THEM too.)

**Verified verbatim (quotes on file):** Dr.GRPO's two biases + fix (2503.20783; TRL `loss_type="dr_grpo"`,
verl `norm_adv_by_std_in_grpo=False` — adopted in both frameworks); DAPO dynamic-sampling
(over-sample, drop all-correct/all-wrong groups) + clip-higher vs entropy collapse (2503.14476);
Lite PPO = group-mean centering + batch-level std + token-level loss, no clip-higher, no KL
(2508.08221); Yue pass@k crossover, 6 algorithms ≈equal, distillation expands (2504.13837);
Spurious Rewards 21.4-pt random-reward MATH gain on Qwen2.5-Math-7B, family-specific, paper itself
recommends random/format-reward controls (2506.10947).

**Config-relevant nuance:** Lite PPO finds token-level loss best for BASE models but sequence-level
better for ALIGNED models — our policy init is the SFT'd checkpoint, so the loss-aggregation choice
should be treated as an explicit ablation flag, not assumed (2508.08221).

**New sources to carry (postdate the plan's sweep):**
- **ScaleRL** 2510.13786 (2025-10, >400k GPU-hours): normalization/aggregation choices affect
  EFFICIENCY not asymptote → do not over-tune variant choice; CISPO recipe reference.
- **CoT-Pass@K** 2506.14245 (ICLR 2026): plain pass@k credits lucky guessing → hedge the win
  definition; consider CoT-validity scoring if a win ever needs defending.
- **Spurious Rewards Paradox** 2601.11061 (2026-01): mechanistic follow-up; strengthens the arm-C
  gate, adds a prompt-coherence diagnostic.
- **Learning from Less** 2604.18381 (2026-04): SLM RLVR under low data/compute; mixed-difficulty
  prompt sets — relevant to P1 band selection.
- **Curriculum-RL beyond base** 2606.22317 (2026-06): boundary-aware curriculum counter-evidence to
  strict sharpen-not-expand; soften the absolute phrasing, keep the measured gate.

**Infra re-check:** vLLM on GB10 (aarch64 sm_121) still broken for prebuilt wheels (vllm#36821 OPEN,
2026-03; PyTorch binaries stop at sm_120) — the HF-`generate`-with-KV-cap default stands.

**Prereq status at addendum time:** P2 DONE (`research/datasets/math-eval-v1`, GSM8K-test 1319 +
MATH-500 500, pinned revisions, cross-eval 0 flagged, **vs-SFT decontam DONE 2026-07-01: 0 flagged
vs 35,924 OpenR1 problems, 1.14M 13-grams**); P3 DONE (`research/eval_math_acc.py`, 34 tests
passing); P4 DONE (extractor pinned `math-acc-v1` + `verifier_false_positive_rate`); P1 (GRPO
training prompt set) NOT built — only needed if Phase 1 passes. **Phase 1 is UNBLOCKED.**
