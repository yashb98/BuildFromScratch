# Propose-only run spec — Zeta optimizer as a *tuned-baseline replication*

- status: **propose-only** — behind the human cost gate (§C20). NOT scheduled. NOT on-cron.
- technique: `zeta-dual-whitening` (ledger `briefed`, taste 7.5) · paper arXiv 2606.14187
- brief: `research/briefs/zeta-dual-whitening.md`
- governing critique: arXiv 2509.02046 "Fantastic Pretraining Optimizers and Where to Find Them" (fetch-verified)
- author: panel synthesis 2026-06-21 · supersedes the "fund Zeta at proxy budget" option the Phase-2 contradiction refused

---

## 1. The frame (why this is interesting, and why the obvious version is not)

The boring framing — "is Zeta a better optimizer?" — is a **bad bet for us**: we already hold a
proven, isolated, 3-seed win in the *same matrix-optimizer family* (NorMuon: **−0.4743 bpb wikitext,
CI [0.444, 0.505]; −0.502 code, CI [0.456, 0.547]**, run `2026-06-16_qwen3_normuon-vs-adamw`). A
second family member is redundant, and 2509.02046 shows the family's edge over a *properly tuned*
AdamW decays from 1.4× @ 0.1B → **1.1× @ 1.2B** — our 596M model sits in that fade zone.

The **interesting** framing is a *replication of the 2509.02046 critique at our scale*:

> **Q:** Does a freshly-published matrix optimizer's claimed win survive a **properly LR-tuned
> AdamW baseline** at 0.6B — or is it another fake win manufactured by an under-tuned baseline?

That meta-result (does optimizer-paper hype hold up under a tuned baseline) is more publishable
than the optimizer itself, and it is the *only* frame in which spending on Zeta buys us something
the field cares about.

## 2. Hypotheses

- **H1 (replication):** With each optimizer's LR individually tuned at matched budget, Zeta ≈ NorMuon
  (both ~0.4–0.5 bpb over a tuned AdamW), i.e. the family wins but Zeta is not distinguishable from
  the incumbent. *(prior: most likely)*
- **H2 (Zeta wins):** Zeta's dual whitening beats NorMuon by a CI-separable margin at 0.6B.
- **H3 (critique confirmed):** Against a *properly tuned* AdamW, the gap shrinks toward the 1.1×
  that 2509.02046 predicts — the paper's headline was inflated by an under-tuned baseline.

A clean answer to **any** of these is a result. The thing we refuse is an *un-attributable* number.

## 3. Arms & controls (reuse maximally, §C13)

| Arm | optimizer | schedule | LR | exists on disk? |
|---|---|---|---|---|
| AdamW (tuned) | AdamW | cosine | **swept** (don't assume 1.7e-3 is its best) | baseline cells exist; LR re-tune is new |
| NorMuon | NorMuon | cosine | swept | 640-step ablation cells exist; full-budget does NOT |
| **Zeta** | Zeta | cosine | swept around 9e-4 | nothing — fully new |

- Zeta config (from brief, paper §4 / Alg. 2): β₁=0.95, β₂=0.99, Newton–Schulz **K=5**, NS coeffs
  (3.4445, −4.7750, 2.0315), RMS scale 0.2·√(mn)/‖U‖_F, decoupled wd=0.1, ε=1e-8 (inferred),
  **2D matrices → Zeta; biases/LN/embeds → AdamW** (identical param-split to our verified NorMuon),
  **no custom kernel**.
- **Single-variable discipline:** every arm differs only in the optimizer + its tuned LR. Same data,
  same seq, same tok/step, same eval-harness BPB metric.

## 4. The anti-confound that makes or breaks it — per-optimizer LR tuning

This is the **load-bearing protocol**, not optional. The paper tunes each optimizer's LR
individually on Qwen3-0.6B; transplanting a single LR (9e-4) untuned is *exactly* the trap
2509.02046 documents. So before any verdict run, each arm gets a short LR sweep and the **best LR is
chosen by held-out BPB**, then the verdict run uses that LR. A null with an untuned LR is worthless.

## 5. Two budget regimes — pick by how much trust you need

| Regime | tok/arm | TPP | what it buys | trust |
|---|---|---|---|---|
| **A — on-box directional hint** | 1.19B (18,150 steps) | 2 | sign of Zeta vs NorMuon vs AdamW *at our scale* | directional only — still **~17× below** the paper's 35 TPP |
| **B — rented publishable verdict** | ~20.9B (paper §4) | ~35 | the real, in-regime, 3-seed claim | publishable |

Regime A is the *most* the GB10 can do; it is honestly a hint, not a verdict, because 2 TPP is far
below the regime where the claim is meaningful. Regime B is the actual spec the human-$ gate exists
for.

## 6. Day & cost estimate — **highest (worst-case) days to a HINT**

Throughput, all conservative (→ highest days): Zeta **5,000 tok/s** (NorMuon measured 5,150; Zeta's
extra coordinate-whitening makes it ≥ that cost), NorMuon 5,150, AdamW 6,900. One run at a time
(serial). +~10% wall-clock for compile/eval/checkpoint/queue, plus slack for one sentinel-restart.
tok/step 65,536. AdamW full-budget baseline (28.65) **already exists → 0 days**.

### Regime A — worst-case on-box hint (1 seed, $0 cash)
| Step | what | math | days |
|---|---|---|---|
| 1 | Zeta LR sweep — 3 LRs × 1 seed × 2,000 steps (131M tok) | 3 × 131M / 5,000 × 1.1 | **~1.0** |
| 2 | Zeta @ full 2 TPP, best LR, 1 seed | 1.19B / 5,000 × 1.1 | **~3.0** |
| 3 | NorMuon @ full 2 TPP, 1 seed (matched control — does not exist yet) | 1.19B / 5,150 × 1.1 | **~2.9** |
| 4 | eval-harness BPB on all 3 + reused AdamW baseline | cheap | **~0.1** |
| | **TOTAL (serial)** | | **~7 days; ceiling ~8 with crash/queue slack** |

> **→ Highest number of days to get the hint: ≈ 8 days on the GB10, $0 cash (opportunity cost only).**
> That buys a 1-seed, LR-tuned, matched 3-way (Zeta · NorMuon · AdamW) at full on-box budget —
> directional at *our* scale, not the paper's regime.

**Cheaper-but-weaker option (~1 day):** reuse the existing 640-step NorMuon+AdamW 3-seed cells, add
Zeta 3×640 steps + a tiny LR sweep. This is the **budget-confounded proxy the contradiction
refused** — it gives a sign in <1 day but ~80× below regime, and 2509.02046 says that sign can flip
at higher budget. A fast hint you should not trust.

**Shortcut on the worst case (~4 days):** drop step 3 (the matched NorMuon-full run) and read Zeta-full
against the existing AdamW baseline (28.65) + the confounded IMU-1 bundle (23.52) as loose anchors.
Cheaper, but the Zeta-vs-NorMuon comparison is then cross-budget — a looser hint.

### Regime B — rented publishable verdict (3 seeds, 35 TPP, tuned)
Wall-clock depends on cluster shape (a 0.6B model is small → high throughput on multi-GPU). Order of
magnitude on an 8×H100 node (~250k tok/s assumed): ~24 h/arm-seed; 3 arms × 3 seeds + sweeps ≈ a few
days wall-clock, **real $** (provider/region/instance pinned by `/remote-launcher`, hard cost cap,
managed-spot). This is the only path to a number that goes in a paper. **Human authorizes the spend.**

## 7. Gates / kill criteria

1. **LR sweep is mandatory** before any verdict run — pick LR by held-out BPB. No untuned-LR result.
2. **Regime A is labelled a hint, never a verdict** — 2 TPP is below regime; do not let an A-result
   reach `/manuscript` as a Zeta claim.
3. **Kill if un-attributable:** if a Zeta seed NaNs/diverges, or the LR sweep shows no stable LR,
   stop — do not report.
4. **Stop-after-hint rule:** if Regime A shows Zeta clearly *inside* NorMuon's noise band, do **not**
   escalate to rented Regime B — H1 is answered, bank NorMuon, move on. Escalate to B only if A shows
   a Zeta-over-NorMuon margin worth a publishable, in-regime confirmation.
5. **Standing on-box refusals:** no second concurrent training (unified-memory overcommit hard-crashes
   the box); no proxy-budget Zeta as a cross-optimizer *claim*; rented spend = human gate.

## 8. What this does NOT do (honesty)

- Even Regime A (full on-box budget) is **~17× below** the paper's 35 TPP — a directional hint at our
  scale, not a replication of the paper's regime.
- It does not change the Phase-2 priority order: BPB-score the de-confound cohort → WSD full-budget
  confirm → data arm all rank **above** this. Zeta is a *fund-if-authorized* side bet, not the
  critical path. See `research/decisions/2026-06-20_phase2-technique-contradiction.md`.

---
*Verified figures: NorMuon win + throughput from ledger `2026-06-16_qwen3_normuon-vs-adamw`; baseline
28.65 = `checkpoint_qwen3_baseline2tpp.pt` (n=1); Zeta recipe from `research/briefs/zeta-dual-whitening.md`
(paper §4 / Alg. 2). ⚠ Zeta peak-LR 9e-4 / 35 TPP are reported in the brief but flagged `inferred`/un-re-verified against paper text — confirm before the rented run.*
