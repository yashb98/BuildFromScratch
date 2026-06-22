<!-- Generated 2026-06-20 by a 4-phase adversarial workflow (wf_65659f7a-6d2):
     3 persona positions (Research Scientist / Frontier Model Engineer / MLOps Engineer)
     -> cross-examination -> red-team of 5 canonical bets x 3 lenses -> chair synthesis.
     22 subagents, ~963k tokens. Every OUR-runs number pinned to on-disk evidence. -->

> **Disk claims re-verified by hand after the run (2026-06-20):**
> (1) full-budget 2-TPP baseline is **n=1** — only `checkpoint_qwen3_baseline2tpp.pt` exists (= 28.65);
>     the 3-seed baselines are the 2000-step *proxy*, a different budget+metric.
> (2) the +WSD proxy arm is **single-variable** — log header `schedule=wsd optimizer=adamw zloss=0.0 | vr=False ln=False hg=False`, ~6,660 tok/s (AdamW rate).
> (3) arch seed0 **live** at step 500/2000, val PPL 92.17, PID 541295, 5,940 tok/s @ 61.1 GB.
> ⚠ arxiv ids `2602.06797`, `2508.01483`, `2507.17634` are NOT grounded in-repo → treat as UNVERIFIED, do not cite without fetching.

# Phase-2 Decision Memo — Qwen3-0.6B
**Chair's final call · 2026-06-20 · numbers verified off disk**

## 1. The call

Run Phase 2 in this order. **(0) Let the in-flight arch arm finish** (~1 day, ~$0, already queued — seed0 is at step 500/2000, confirmed on disk) **and the moment all three Phase-1 proxy arms exist, score the ENTIRE de-confound cohort — baseline, +WSD, +z-loss, +arch — through eval-harness BPB.** This is the cheapest, strictly-dominant first move: it costs essentially zero new GPU time and converts our preliminary in-loop val-PPL reads into the *canonical* metric, which is the only thing the manuscript gate accepts. **(1) Then, gated on that BPB result, run the WSD full-budget confirm:** AdamW + WSD-to-zero ONLY (no NorMuon, no arch flags), at 2 TPP / 18,150 steps, scored on eval-harness BPB. The arm runs at the **AdamW rate (~6,900 tok/s, ≈1.5 days)** — verified from `train_wsd_seed0.log`, *not* the 1.9-day arch-tax rate — checkpointed every ~2000 steps under sentinel `--kill-at 0.83`. **(2) Then pivot the next marginal GB10-day to a token-matched DATA arm**, because the gap to 13.40 is provably data, not method. **Do NOT spend on a Zeta optimizer arm** — neither at the 2000-step proxy (un-attributable by construction, per `2509.02046`) nor at full budget on-box (≈1.5 days for a ~1.1x-class effect we already hold a stronger, isolated version of in NorMuon). Zeta stays propose-only behind the human cost gate.

## 2. Why — reading the trajectory

The results trail tells one clean story:

| Stage | Metric | Reading |
|---|---|---|
| Faithful baseline @ 2 TPP | 28.65 val PPL (n=1, full budget) | reference; gap **2.14x** to the published 13.40 |
| IMU-1 modernized bundle @ 2 TPP | 23.52 (−17.9%) | **a win, but confounds ~6 changes + WSD-to-zero** — not a single claim |
| NorMuon vs AdamW (3 seeds, BPB) | −0.4743 bpb wikitext, CI [0.444, 0.505]; −0.502 code, CI [0.456, 0.547] | **PROVEN, isolated optimizer win** — the only fully-gated result we own |
| +WSD proxy (3 seeds, in-loop PPL) | −6.9%, significant across seeds | **preliminary** — proxy metric, proxy budget; not yet BPB |
| +z-loss proxy (3 seeds, in-loop PPL) | +0.1%, null | **null on PPL** — it's a stability/logit knob, not a loss lever |
| +arch proxy | seed0 @ step 500 (val PPL 92.17, descending), seed1/2 queued | **UNSCORED** |

The −17.9% bundle **decomposes**, preliminarily, into **an optimizer win (NorMuon, proven on BPB) + a schedule win (WSD, proxy-only) + a null (z-loss) + an unknown (arch)**. That is the entire publishable thesis: turn one confounded headline into named, individually-validated components.

**The bitter-lesson read is the strategic spine.** The baseline gap to 13.40 shrank from 3.5x → 2.14x **purely by going 131M → 1.19B tokens, with zero architecture or optimizer change.** We trained on ~275,000x less data than the published model. Every optimizer/schedule lever in this menu is a **second-order tweak on a first-order data deficit** — and the surfaced literature converges hard on this: `2509.02046` (verified) shows matrix-optimizer speedup over a *tuned* AdamW collapses from 1.4x @ 0.1B to **1.1x @ 1.2B** (our 596M model sits squarely in that decay zone); SmolLM2 `2502.02737` (verified) attributes small-LM gains to multi-stage data curation with architecture/optimizer secondary. **Method-tweaking is near its diminishing-returns floor at our scale; tokens/data is where the anchor actually lives.**

So the portfolio splits cleanly into two jobs: **(a) finish the attribution** (cheap, gating, unblocks the paper — do it first), and **(b) move the anchor** (data, the real lever — do it next). Buying a *third* optimizer does neither.

## 3. Where the three lenses AGREE — and where they genuinely CONFLICT

**Unanimous agreement** (high signal — three independent lenses converged):
- **WSD full-budget confirm is rank-1** among new training runs. It is the one experiment that closes BOTH translations the rest of the menu silently assumes away: proxy val-PPL → canonical BPB, *and* 2000-step → 18,150-step (where the cosine baseline finally enters its own long decay).
- **Refuse the IMU-1 bundle re-run as a "result"** (confounded), **refuse a proxy-budget Zeta** as a cross-optimizer comparison, **refuse any new from-scratch build** (propose-only on this box), **refuse re-running any §C13 control**, and **refuse a second n=1 SFT**.
- **The WSD arm must be plain AdamW + WSD-to-zero.** This is a *correctness* condition, not a speed footnote — inheriting NorMuon would re-confound schedule with optimizer. **Verified on disk: it already is AdamW** (`train_wsd_seed0.log`).
- **Score on eval-harness BPB, never publish in-loop val PPL** as the verdict.

**The genuine, unresolved conflict is the COST and SEQUENCING of the WSD confirm — and the Frontier Engineer is factually right:**

1. **Is the full-budget baseline reusable as a multi-seed control?** The RS priced WSD-confirm at "~5.7 days most-likely (reuse baseline)." **I verified disk: there is exactly ONE full-budget baseline checkpoint (`checkpoint_qwen3_baseline2tpp.pt`), single-seed = 28.65.** The 3-seed baselines that exist are the **2000-step proxy** baselines — a different budget AND metric. So a clean **3-seed** WSD-vs-baseline CI at 18,150 steps requires **3 NEW full-budget baseline seeds** (the ~11.4-day branch), OR accepts comparing 3 WSD seeds against an n=1 baseline (which cannot produce the "CI-crosses-zero" kill test the RS's own gate demands). **The RS's headline cost quietly imported a control that does not exist at the needed seed×budget cell.** This is the real contradiction, and it materially weakens "WSD-full-budget" against the cheaper alternatives.

2. **MLOps resolves the cost conflict correctly:** the decisive unknown is **binary and directional** — *does the proxy WSD edge survive the cosine baseline's own cooldown?* A **single** full-budget WSD arm vs the existing n=1 full-budget baseline answers that for ~1.5 days. **Buy seeds 2–3 only if seed-1 lands inside the noise floor** — staged, not up-front. This dissolves the 5.7–11.4-day pre-commit into one ~1.5-day directional run.

3. **Evidentiary hygiene (both rebuttals land):** the schedule-budget-dependence claim floated through these positions on **`2602.06797`** — **which I grepped and is NOT on disk.** Neither is `2508.01483` or `2507.17634`. The verified WSD grounding in our repo is **MiniCPM `2404.06395`** + **D2Z `2507.09846`** (both confirmed on disk). Under the verified-accuracy mandate, the budget-dependence justification must cite those, or be flagged UNVERIFIED. The *physics* (WSD's edge may shrink once cosine cools) is sound and well-supported by the in-repo MiniCPM/D2Z sources — but the fabricated id must not reach any artifact.

**Where they don't really conflict (and the red-team agrees):** what comes *second*. RS conceded data is the highest-EV gap-closer; Frontier ranks data #2; MLOps moved data above any 4th optimizer. The only one defending a Zeta spend is Frontier (as a 640-step parity sign-check) — and MLOps flags this **fatal**, correctly: it spends compute on a scale-fragile number the cited `2509.02046` exists to prevent.

## 4. Ranked Phase-2 portfolio (ordered by EV-per-GB10-day)

| # | Bet | What to run | Expected gain vs 13.40 anchor | GB10-days | $ if rented | Decision gate / kill | Venue |
|---|---|---|---|---|---|---|---|
| **0** | **BPB-score the de-confound cohort** | Finish arch seed0/1/2 (queued), then eval-harness BPB on baseline·+WSD·+z-loss·+arch — **zero new training** | None (attribution only). Turns proxy reads into canonical verdict; unblocks the paper | **~1** (arch finish) **+ ~0** (eval) | ~$0 | n/a — pure information. Kill nothing; this *sets* every downstream gate | **on-box tonight** |
| **1** | **WSD full-budget confirm** | **1 seed**, AdamW+WSD-to-zero ONLY, 2 TPP/18,150 steps, ckpt every 2000, vs existing n=1 baseline (28.65); BPB at endpoint | Knowledge-decisive. If holds: baseline 28.65 → plausibly ~26–27 band, gap ~1.95x. **No anchor-closing claimed** | **~1.5** (AdamW rate ~6,900 tok/s, verified — *not* 1.9) | ~$90–140 (1× H100-class, ~36h) | **Add seeds 2–3 ONLY if seed-1 lands inside the noise floor.** Kill the SCALED claim if 3-seed BPB CI crosses 0 → retire schedule axis. Kill the RUN if sentinel hits 0.83 or tok/s drops >20% | **on-box tonight** |
| **2** | **Token-matched DATA A/B** | webgraphmix vs FineWeb-Edu at **identical tokens**, ≥3 seeds, same optimizer/schedule; judge ONLY on **OOD** BPB (wikitext+code), NOT in-dist FineWeb val PPL | **Highest gradient on the anchor** of any arm (literature: multiplicative, not 1.1x). Honest negative also publishable | **~2–2.5** (incl. **/dataset-forge prep+decontam+disk-floor** critical path) | ~$150–320 | Pre-register: claim is "data-selection delta at fixed tokens," **NOT** "closing to 13.40." Read `2510.00866` (UNVERIFIED) first — filter gains may be smaller than advertised. Kill if no BPB AND no downstream signal | **on-box** (after #1) — prep is the gate, not tonight |
| **3** | **Arch sub-flag drill** (vr / ln-scaling / head-gating) | 3 flags × 3 seeds × 2000-step proxy | Small; "free-param" tweaks, literature reports modest | **~0 now (gated)**; ~1.9 if it fires | ~$200–400 | **CONDITIONAL on #0:** drill ONLY if arch-bundle BPB CI excludes 0 AND aggregate delta >~5 PPL (so each sub-flag can clear the ~1.7-PPL floor). Else mark arch null, skip | **on-box, conditional** |
| **4** | **Manuscript** | /manuscript once headline = single attributed component on BPB | None (artifact) | ~0 (CPU) | $0 | **Gate: do NOT write until #0+#1 land.** Lead = NorMuon (isolated) + WSD-at-budget verdict. Never the −17.9% bundle | **on-box, after #1** |
| **5** | **SFT re-run ≥3 seeds** | SFT + paired control + forgetting probe, ≥3 seeds | None on pretrain anchor | ~1–2.5 | ~$70–180 | Power-check the n=1 spread first; skip if 3 seeds can't resolve forgetting deltas above floor | **on-box, low priority** |
| **—** | **Zeta optimizer arm** | — | ~1.1x-class, scale-fragile | — | — | **REFUSED on-box.** Propose-only: full-budget, LR-tuned, 3-seed **rented** run behind human cost gate, or not at all | **needs human-$ gate** |
| **—** | **New from-scratch builds, at-scale MFU** | ternary-mamba, hybrid-attn, scaling report, etc. | — | — | — | **REFUSED** — propose-only / physically impossible on single unified-memory box | **needs human-$ gate** |

## 5. The honest "is more research/training worth it" verdict

- **Optimizer axis — DIMINISHING RETURNS / STOP.** We already hold the only fully-gated win here (NorMuon, −0.47/−0.50 bpb, CIs excluding zero, verified in `cohort_bpb.json`). `2509.02046` (verified) says a *second* matrix optimizer buys a ~1.1x effect at our scale that erodes further with budget, with rankings that flip mid-training. **The next marginal GB10-day on a new optimizer is a waste.** Payoff: forgettable. Bank NorMuon, move on.

- **Schedule axis (WSD) — WORTH EXACTLY ONE RUN, then decide.** This is the single highest-value *attribution* day because the −6.9% is **proxy-only** and **budget-confounded in WSD's favor** (more cumulative peak-LR time than cosine gets at 2000 steps). One ~1.5-day full-budget BPB run resolves whether it survives. Worth it **once**; if the BPB CI crosses zero at full budget, the schedule axis is **dead** — do not drill further. Payoff: a clean publishable component *or* a clean "proxy mis-estimates schedule gains" methodological negative. Both are A+-evidence.

- **Arch axis — CONDITIONAL, default-skip.** Preliminary signal points to optimizer+schedule carrying the bundle and z-loss null; arch is most likely small. **Finishing the in-flight arm is ~free and mandatory** (it's already running). The 9-cell drill is **only** worth it if the arch bundle clears the BPB floor with enough headroom that each sub-flag can resolve. **Default expectation: skip the drill.** Payoff if it fires: a named mechanistic component (head-gating is the best-supported a priori, though `2505.06708` is UNVERIFIED on disk).

- **Data / tokens axis — THE REAL LEVER, worth the most days.** This is where the 2.14x gap lives (verified: gap shrank 3.5x→2.14x purely from tokens). Literature is multiplicative here (SmolLM2 verified; BeyondWeb `2508.10975` claims up to 7.7x — UNVERIFIED). **The marginal GB10-day after attribution is best spent here, not on a 4th optimizer.** Caveat: it has a real prep/decontam critical path (a leaked eval doc silently fakes a BPB win), and the OOD-canonical-metric may *under*-credit a real in-distribution data gain — so judge it ≥3 seeds on OOD BPB and pre-register it as a data-*selection* result, not an anchor-closing claim. Payoff: the largest honest move toward 13.40 available on this box.

- **Post-train (SFT/DPO/GRPO) — NOT NOW.** The existing n=1 SFT is INCONCLUSIVE; repeating at n=1 adds nothing. Only worth it at ≥3 seeds with a paired control, and only after the pretrain attribution+data story lands. Lower EV than data this cycle. Payoff: converts INCONCLUSIVE → a signed forgetting result — real but not gating.

**Bottom line on time & money:** Tonight costs **~$0** (on-box, opportunity-only). The whole attribution-closeout + WSD confirm is **~2.5 GB10-days serial**. The only thing that would cost real **$** is a rented Zeta arm or at-scale work — and the evidence says **don't**. The highest-EV *dollar*, if/when you rent, is a **data/token run**, not a third optimizer.

## 6. Decision gates (if-this-then-that, so the plan adapts to the in-flight Phase-1 result)

1. **WHEN arch seed0/1/2 finish (~1 day):** immediately run eval-harness BPB on the full cohort → write `cohort_bpb.json` / `verdict.json` for the de-confound experiment. This is the trigger for everything below. *(Until this lands, every "WSD −6.9%" / "z-loss null" number is in-loop val PPL — a proxy, NOT the canonical verdict.)*

2. **IF +WSD BPB delta vs baseline CI excludes 0** → launch the **WSD full-budget confirm** (bet #1), 1 seed, AdamW+WSD-only. **ELSE (WSD null on canonical BPB)** → the proxy PPL win did not survive the metric jump; **do NOT burn the full-budget day** — record "proxy PPL over-credited WSD" and route the day to the **data arm** (#2).

3. **AT the full-budget WSD endpoint:** **IF** 1-seed BPB clearly beats the n=1 baseline beyond the eval noise floor → directional confirm; **add seeds 2–3** only to get a publishable CI. **IF** it lands inside the noise floor → **declare WSD a no-op at 2 TPP, retire the schedule axis** (do not buy more seeds to chase noise). **IF** a WSD seed NaNs/diverges (decay-to-zero instability) → kill that seed, flag.

4. **IF +arch BPB CI excludes 0 AND aggregate arch delta is large enough that each of 3 sub-flags can clear the ~1.7-PPL floor** → fund the 3-flag drill (bet #3), dropping any sub-flag whose single-variable CI crosses 0. **ELSE** → mark arch **null at this budget** (like z-loss), skip the drill, reallocate to data.

5. **IF (NorMuon already-proven) AND (WSD confirmed at full budget on BPB)** → /manuscript may proceed, headline = NorMuon (isolated) + WSD (confirmed schedule win). **IF WSD did NOT survive** → /manuscript proceeds with headline = NorMuon win + "WSD does not survive at budget" as a methodological finding. **Either is clean; never headline the −17.9% bundle.**

6. **Standing refusals (no gate can flip these on-box):** no second concurrent training (unified-memory overcommit hard-crashes the box — serial only); no re-running any §C13 control; no proxy-budget Zeta as a cross-optimizer claim; no new from-scratch build; no at-scale/MFU claim on the GB10. Zeta and any rented run require the **human cost gate**.

---
**Hygiene flags for whoever drafts the paper:** `2602.06797`, `2508.01483`, `2507.17634` are **NOT on disk** — do not cite. Verified WSD grounding = MiniCPM **`2404.06395`** + D2Z **`2507.09846`**. `2509.02046` (optimizer critique) and SmolLM2 `2502.02737` are fetch-verified. BeyondWeb `2508.10975`, the data-filter-illusion `2510.00866`, gated-attention `2505.06708`, WSM `2507.17634`, and the LayerNorm-Scaling id `2502.05795` are **UNVERIFIED** — flag before citing.

**Verified-on-disk this session:** WSD arm = AdamW @ ~6,642–6,909 tok/s (`train_wsd_seed0.log`); arch arm = 5,940 tok/s @ 61.1 GB, seed0 step 500/2000 val PPL 92.17 (`train_arch_seed0.log`); NorMuon BPB win real (`cohort_bpb.json`: NorMuon wikitext bpb ~1.62–1.65 vs AdamW ~2.10); **full-budget baseline is single-seed** (one `checkpoint_qwen3_baseline2tpp.pt` = 28.65) — so a 3-seed full-budget WSD CI needs new baseline seeds OR accepts an n=1 baseline comparison.
