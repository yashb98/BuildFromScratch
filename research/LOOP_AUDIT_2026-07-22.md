# Research-loop deep audit — 2026-07-22

Source: 8-agent read-only workflow (wf_641090e2-442) + first-hand session record.
74 evidence-anchored weaknesses. The 2 mining agents + synthesizer hit the session
limit; synthesis + pain-mining done by the main loop from the 8 reports + lived history.

## Scorecard (what this system actually is, as evidenced)

A single-box (GB10) ML-research state machine with 27 skills and a §C1–§C27 contract
spine. In 6 weeks it produced **24 ledger runs, ~15–19 GPU-days, 2 wins** (1 stage-win
in 13 stages), and the flagship optimizer "win" is now measured to **converge away**.
Its real, differentiated product is **rigor** — it caught an eval-token confound, a
token-shuffle bug, an implausible verdict, a resume-budget iso-FLOP violation, a
silent ledger-write failure, and pre-registered the RLVR null. The "autonomous nightly
loop" has closed **one** CPU-light iteration (launch declined); **cron was never armed**;
nearly every headline came from human-driven sessions.

## Top weaknesses, ranked (merged across the 8 reports)

1. **Autonomy is inert.** No BuildFromScratch cron installed (nightly/liveness/@reboot);
   a *foreign* project (forge-loop) now owns the crontab. A reboot tonight strands the
   live 15-cell ladder until a human intervenes. weekly-retro never ran → the §C15.3.8
   calibration loop never closed → every score is unfalsified self-assessment.
2. **Specified path ≠ actual path.** /ablation-runner ran ~2× while ~24 runs landed
   hand-driven; "no brief, no run" is not operative (11/24 runs technique_slug=null);
   the flagship win has no c5_evidence; out-of-loop launches are the dominant mode and
   the contracts define no protocol for them (the current live run mutates loop_state
   while skipping the digest/provenance machinery).
3. **Truth store has zero off-box durability.** ledger.json is git-untracked; 17/24
   runs point at detail_md files that don't exist; 60+ ad-hoc --set keys (eval metrics
   stored outside metrics{}); techniques strand in non-selectable states so next-best is
   actively wrong; working branch is 6 commits ahead of origin, 58 dirty paths, and
   gitignored evidence lives on one disk (already destroyed once via branch switch).
4. **Eval starvation + monopoly-in-name-only.** score_arch_ladder.py doesn't exist (the
   LIVE run's scoring hook is a silent no-op); the continuous driver leaves no GPU-free
   window so no suite number for ~6 days; the "eval-harness is the ONLY comparable source"
   monopoly is honored in stamp but not code — 6 hand-rolled score_cohort.py copies;
   text-lm-v2/v3 governance contradiction; §C25 HARD registry is ahead of its tooling so
   several stages are structurally capped at "directional" regardless of result quality.
5. **"directional" compresses opposite realities** — genuine nulls (sft-masking, grpo)
   and big *significant* effects capped only by one missing HARD item (data-mix +0.59
   code BPB, significant). A ledger reader can't tell "found nothing" from "found
   something big, one gate short."
6. **Safety debt on the newest layer.** The thermal-kill path (added because the box
   hard-locks from heat) has ZERO tests; thermal_log.py is NOT running beside the live
   run; safe_cuda / jax_safe_env / cron_runner / liveness_cron / the arch-ladder driver
   are all untested; kdump/panic still disabled so every future hard-lock is trace-less.
7. **Intake is 100% trend-following and stale.** idea-selection §C15.3 (bandit/cascade/
   red-team) has never executed (it's "sort, not bandit"); taxonomy_gap has never
   produced a candidate; pulse ran 3× in 40 days; dedup is exact-slug only; the candidate
   pile grows monotonically with no aging/kill policy.
8. **Win ceiling.** Wins correlate with huge effects + cheap batteries (data-stage,
   on-box passkey); optimizer/architecture effects at this budget are small or fade, and
   several stages are capped by structurally expensive HARD items — so on-box ablations
   of the same shape are near-guaranteed "directional." The one big open question
   (NorMuon at scale, ~17–34h) is unresolved while 141h goes to the hybrid ladder.
9. **Publication drift.** arXiv tarball is stale vs the 2026-07-20 rebuilt sections;
   qwen3-study state.json says phase 3 while Phase-6 artifacts exist; the normuon paper
   is orphaned and its headline may not survive (converges); both master strategy docs
   open with leaked LLM meta-commentary.
10. **Contract drift.** S10 has 3 contradictory specs (one names a nonexistent skill);
    §C27.6 S4 stage-plan gate unimplemented (the 141h arch stage has no plan.md); pinned
    schemas lag the code; §C13 objective vocab can't express most §C25 lifecycle stages.

Full per-area weakness list with file:line evidence: see workflow journal
(subagents/workflows/wf_641090e2-442/journal.jsonl) — 8 result lines, one per area.

## Decision plan — 7 batches × 4 questions (28 total)

1. North star & what "best" means
2. Automation & autonomy level (cron, gates, unattended rights)
3. Contract-vs-reality operating model (out-of-loop runs, launcher, c5)
4. Research intake & selection policy
5. Execution & compute (phase-2, Muon, off-box, NorMuon-at-scale)
6. Evaluation rigor & economics (suite, seeds, scoring scheduling)
7. Durability, engineering & publication (git policy, test debt, arXiv)

Answers get folded into a follow-up implementation plan.

## Decisions (recorded as answered)

### Batch 1 — North star
1. Optimize to be: **the rigor factory** (de-confounding/verification harness).
2. Primary 1–2mo deliverable: **published papers** (arXiv qwen3 study + hybrid-SSM study).
3. Next GPU-week: **finish NorMuon-at-scale** (3rd 420M seed + 840M rung; rescues the paper).
4. Verdict vocabulary: **split it** (null / promising-capped / win — stop compressing opposites).

### Batch 2 — Automation & autonomy
1. Cron: **recovery crons only** (@reboot boot_resume + */30 liveness; protects the live ladder, no auto-launch).
2. Launch rights: **propose-only GPU, auto CPU** (loop briefs/preps/scores autonomously; human triggers every GPU run).
3. Box sharing: **shared-box lock/handshake** both BFS and forge-loop honor.
4. Calibration: **wire outcomes→calibration** + real retro cadence (make scores falsifiable).

### Batch 3 — Contract vs reality
1. Out-of-loop runs: **codify the manual path** (adopted-run protocol: c5 + ledger entry + digest stub + loop_state).
2. Brief gate: **required for new techniques only** (not re-runs/controls/seeds/scaling rungs).
3. c5 evidence: **validated schema + pre-launch lint** (refuse launch on incomplete evidence).
4. Ledger timing: **entry-at-launch wins; fix CLAUDE.md** (can't have a verdict before the run).

### Batch 4 — Research intake & selection
1. Intake: **your taste, trend as input** (pulse/radar surface options; you pick the questions).
2. Candidates: **auto-expire unbriefed** after N weeks (out of next-best; reversible).
3. Scan cadence: **weekly** (not daily).
4. Dedup: **arXiv-id + fuzzy-title** (not slug-only).

### Batch 5 — Execution & compute
1. Muon: **accept AdamW as the ladder baseline** (note deviation; don't restart 5 done cells).
2. Mixer confound: **cheap per-arm LR probe** before any mixer-type claim (de-confound first).
3. Off-box: **defer, stay on-box** (build remote only when a specific run needs it).
4. Phase-2 seeds: **seed-up only after de-confounding**, only arms that still separate.

### Batch 6 — Evaluation rigor & economics
1. Eval timing: **score between rungs** (pause driver after each rung, BPB-score its cells, resume).
2. Scorer: **one real eval-harness impl** (delete the 6 hand-rolled copies; fix the suite stamp).
3. §C25 tooling: **build only for stages you'll publish** (pretraining/architecture HARD items).
4. Noise floor: **fixed reference-model floor** (not self-floor on undertrained checkpoints).

### Batch 7 — Durability, engineering & publication
1. Git policy: **track the durable record** (ledger.json + runs/*.md + briefs + digests; only ckpts/logs/data ignored).
2. Test debt: **safety-killers first** (thermal-kill, safe_cuda/jax guards, driver concurrency/PID).
3. arXiv paper: **rebuild the package + fix state.json, then user submits** (I refresh, human uploads).
4. Crash forensics: **kdump/panic fix at next safe reboot** (crashkernel=512M, panic=10).

All 28 answered. Implementation roadmap: research/LOOP_UPGRADE_PLAN_2026-07-22.md.
