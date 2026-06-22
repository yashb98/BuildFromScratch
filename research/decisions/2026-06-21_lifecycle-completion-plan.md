# Lifecycle-completion plan — Qwen3-0.6B (from 2026-06-21)

How to read this: it is a **scaffold, not a commitment**. The chapter order, the gates, the
budgets, and the if-then branches are fixed now. The *technique each chapter actually uses* is
NOT — it is produced by `/research-loop` + the eval-harness/seed-CI scrutiny as each prior result
lands. See §"What can and can't be planned now" at the bottom.

GB10 reality: **one GPU job at a time**, ~1.9 days per 2-TPP run, ~5h per 2000-step proxy cell,
~631M tok/day. Two stages (kernel roofline, distributed-scaling) **cannot run on this box** — rented,
behind the human-$ gate, and overlap the on-box line once authorized.

## Dated serial schedule (rigorous path)

| # | Stage (chapter) | Work | Days | Est. window | Gate / branch at the boundary |
|---|---|---|---|---|---|
| 1 | **Phase 2 arch drill (Ch.3)** | 3 sub-flags × 3 seeds proxy + BPB score | ~2 | Jun 21 → 23 | which sub-flag's CI excludes 0 → **names the paper's mechanism**; all null → arch is "bundle-only" |
| 2 | **Data arm (Ch.2)** | `/dataset-forge` prep (CPU, overlaps) + 2–3 seeds @2TPP, OOD-BPB | ~4 | Jun 23 → 27 | **BRANCH:** big in-budget win → consider rented in-regime confirm ($ gate); null → record + move on |
| 3 | **Mid-training (Ch.7)** | anneal on premium data @low LR + RoPE context-extension | ~1.5 | Jun 27 → 29 | context-len eval must not regress short-context |
| 4 | **Serving export (Ch.14) — pulled forward** | vLLM registration shim for our `model.py` | ~1 | Jun 29 → 30 | **unblocks GRPO rollouts** (see §sequencing) |
| 5 | **Post-training (Ch.9–11)** | SFT ≥3-seed + control → DPO → GRPO/RLVR demo | ~5–6 | Jun 30 → Jul 6 | each gated on the **catastrophic-forgetting probe** + preference-acc/reward-margin; forgetting regression = fail |
| 6 | **Serving-bench + SLOs (Ch.14)** | `/serving-bench` batching/KV/`--quant fp8` + `/observability-slo` | ~1.5 | Jul 6 → 8 | a throughput win with PPL regression beyond noise = **LOSS** |
| 7 | **Safety (Ch.12)** | `/safeguards-eval` + red-team passes (methodology demo) | ~1 | Jul 8 → 9 | — |
| 8 | **Interpretability (Ch.13)** | SAE/probing demo — **no skill yet; build-or-skip** | ~1–2 | Jul 9 → 11 | gap: ad-hoc or propose-only |
| 9 | **Publish (Ch.15)** | `/manuscript` — clean attribution headline | ~1 | Jul 11 → 12 | claim↔evidence audit + **human submit click** (never auto) |
| R | **Kernel roofline (Ch.5.7/14.8)** | one Triton kernel + oracle + roofline | ~1 | when authorized | **RENTED + $ gate**; overlaps on-box line |
| R | **Distributed-scaling (Ch.5)** | FSDP/TP scaling table | ~1–2 | when authorized | **RENTED-only — GB10 can't**; overlaps |

**Rigorous finish ≈ Jul 12–18, 2026 (~3.5–4 weeks).** Demo-grade (1 seed where allowed, skip interp,
fp8-only) compresses to **~Jul 4–6 (~2 weeks).** Critical path is **data arm → post-training**, not
the cheap eval/serving/safety stages.

## Sequencing optimization (why the order ≠ the naive lifecycle order)

The serving export (vLLM shim) is **pulled forward** before post-training, because **GRPO/RLVR (Ch.11)
needs a separate rollout/inference engine** (vLLM/SGLath) — the *same* export. Building it once at
step 4 unblocks the heaviest post-train stage *and* the Ch.14 serving stage. This kind of cross-stage
dependency is exactly what planning-the-scaffold-now buys you — without pre-committing any technique.

## What can and can't be planned now

**LOCK NOW (the scaffold):** chapter order · the gates (noise-floor CI, iso-FLOP, forgetting probe,
claim↔evidence, cost) · the if-then branches · budgets · skill ownership · this dated skeleton.

**CANNOT LOCK (produced per-chapter by the loop + RS scrutiny):** the specific technique/config each
stage uses (conditioned on the prior stage's measured result) · whether a stage wins or is null (you
must be willing to believe a null and re-route) · whether to escalate to a rented in-regime run.

**Proof from this very project:** Phase 1's preliminary in-loop signal said *WSD* was the driver
(−6.9%, "significant"). The canonical BPB verdict said **arch** is the driver and **WSD is null**. Had
we pre-committed "Phase 2 = WSD full-budget confirm," we'd have burned 1.5–8 days confirming a
non-effect. The per-result scrutiny (BPB-before-believing-the-proxy) is what re-routed it. That is
§1.7 of the lifecycle doc: *"the hard part is not running the ladder — it's having the discipline to
believe a null result."*

## How often RS scrutiny actually fires

- **Automated gate at EVERY result (cheap, no human):** eval-harness BPB + across-seed CI + forgetting
  probe → win/loss/inconclusive. Built into `/eval-harness` + the verdict gate; fires at every node.
- **Full RS contradiction (expensive, human-in-loop) ONLY at branch points:** a surprising/borderline
  result, or a next bet that costs rented $. Expect **~2–4 of these across the whole lifecycle**
  (e.g. "data won big — escalate to rented?"; "post-train: which of SFT/DPO/GRPO to spend on?"),
  **not one per chapter.** The cheap chapters (serving-bench, safety) have deterministic gates and
  just run.
