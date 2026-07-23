# Research-loop upgrade plan — 2026-07-22

Derived from the 28-decision audit (`LOOP_AUDIT_2026-07-22.md`). One-line identity:
**a rigor factory that ships papers, runs its safety/prep autonomously, and keeps a human on every GPU-spend trigger.** All work below is CPU-only and safe beside the live ladder unless marked [GPU] or [HUMAN].

## Tier 0 — Immediate, cheap, protects live work

1. **Print recovery cron lines** for the user to paste [HUMAN paste, §C4.2]:
   `@reboot bash <root>/research/boot_resume.sh` and `*/30 * * * * <root>/research/liveness_cron.sh`.
   → the live 141h ladder currently has NO reboot protection.
2. **Track the durable record in git** (batch7 Q1): un-ignore ledger.json + runs/*.md + briefs/ + digests/ (keep ckpts/*.pkl, *.pt, *.log, datasets ignored). Commit. Closes the #3 risk (one-disk truth store; a branch switch destroyed a ledger.json once). Commit BEFORE any future branch switch.
3. **Rebuild the arXiv package** (batch7 Q3): refresh tarball/.bbl/state.json to match the 2026-07-20 sections, re-run the honesty scrub. [HUMAN does the actual arXiv upload.]

## Tier 1 — The rigor engine (papers depend on it)

4. **Wire ladder scoring** (batch6 Q1): write the missing `score_arch_ladder.py`; implement "score between rungs" — after each rung the driver pauses, eval-harness BPB-scores its 5 cells, then resumes. Backfill: BPB-score the 42M rung (and 85M when done) at the next rung gap. [GPU, brief]
5. **Per-arm LR probe** (batch5 Q2): small LR sweep at 42M per mixer to de-confound the SSM-vs-attention gap before it's a claim. The 42M rung's mixer finding stays "directional + LR-confounded" until this runs. [GPU]
6. **One real eval-harness** (batch6 Q2): make eval-harness the single scorer; delete the 6 hand-rolled score_cohort.py copies; fix the suite-version stamp so it guarantees what ran.
7. **Fixed reference-model noise floor** (batch6 Q4): replace self-floor with a stable reference checkpoint so CIs are meaningful.
8. **Split the verdict vocabulary** (batch1 Q4): add null / promising-capped / win to the ledger schema; relabel the 9 existing "directional" runs.
9. **NorMuon-at-scale** (batch1 Q3) [GPU, HUMAN-gated]: 3rd 420M seed (~17h) + 840M rung (~34h) to resolve the flagship open question and rescue the orphaned paper. SEQUENCING (one point needing your ok): finish the current hybrid 85M rung, then insert NorMuon-at-scale, then resume the hybrid 150M rung — respects "NorMuon next GPU-week" without wasting the running ladder.

## Tier 2 — Contract & hygiene reconciliation (CPU-only)

10. **Codify the out-of-loop path** (batch3 Q1): an "adopted run" protocol — any manual launch must write c5 + a ledger entry + a digest stub + register in loop_state.
11. **Brief gate = new techniques only** (batch3 Q2); **c5 validated schema + pre-launch lint** (batch3 Q3); **fix CLAUDE.md verdict-timing** to entry-at-launch (batch3 Q4).
12. **Ledger integrity**: backfill/stop-stamping the 17 missing detail_md; move eval metrics into metrics{}; un-strand techniques stuck in non-selectable states.
13. **Candidate auto-expire** (batch4 Q2) + **arXiv-id/fuzzy dedup** (batch4 Q4).
14. **Wire outcomes→calibration** + a weekly retro (batch2 Q4).
15. **Reframe docs**: "autonomous AI scientist" → "rigor factory" (batch1 Q1); scrub the leaked LLM meta-commentary from both strategy docs.

## Tier 3 — Safety & durability engineering

16. **Safety-killer tests first** (batch7 Q2): thermal-kill parsing/debounce, safe_cuda/jax_safe_env guards, arch-ladder driver concurrency/PID. Then the recovery chain.
17. **Shared-box lock/handshake** with forge-loop (batch2 Q3): one cross-project GPU lock both honor.
18. **loop_state.py fsync parity** with ledger.py (recovery-critical file lacks the parent-dir fsync).
19. **kdump/panic fix** at next safe reboot (batch7 Q4) [HUMAN, sudo].

## Tier 4 — Standing policy (no build, just adopt)

20. Weekly scan cadence (batch4 Q3). Taste-driven intake, trend as input (batch4 Q1).
21. Propose-only GPU / auto CPU (batch2 Q2). Defer off-box; stop maintaining remote as live surface (batch5 Q3).
22. Accept AdamW as the hybrid ladder baseline; note the ARCHITECTURE.md deviation (batch5 Q1).

## The one sequencing question for you
Item 9: insert NorMuon-at-scale after the current 85M rung (pausing the hybrid ladder ~34h), or let the whole hybrid ladder finish first? Your batch-1 answer said "NorMuon next GPU-week," which implies the former — confirm when you want it.

## Execution log — 2026-07-22 (this session)

DONE (committed + pushed):
- Tier 0 #2 track the record (0409620) · #3 arXiv package rebuild (66f2953).
- Tier 1 #8 verdict vocabulary split + relabel 4 clear runs (79a514f).
- Tier 1 #4 score_arch_ladder.py + loud driver hook (e389793).
- Tier 1 #7 fixed-reference noise floor + stale-v3-status correction (b9ce4e3).
- Tier 0 #1 recovery cron lines PRINTED for the user to paste (cron_logs/ created) — awaiting paste.

CORRECTED FRAMING:
- #6 "one real eval-harness": the canonical PRIMITIVES already exist + are tested
  (eval_metrics.bits_per_byte, eval_stats.subsample_noise_floor / seed_delta_significant /
  noise_floor). The 6 score_cohort copies differ only in the thin per-model windowed-CE
  score LOOP (legitimate experiment-isolation glue). So #6 is smaller than the audit framed;
  the real integrity gaps were the noise-floor basis (#7, DONE) and the suite stamp (DONE).
  Remaining #6 work (optional): extract the shared score-loop into one importable helper.

BLOCKED (GPU + human trigger, per propose-only):
- #5 per-arm LR probe (de-confounds the 42M mixer finding) — run at a rung gap.
- #9 NorMuon-at-scale — run at a rung gap; sequencing still to confirm.
- score_arch_ladder.py GPU end-to-end validation — first rung gap.

GOVERNANCE TODO flagged, not decided unilaterally:
- Flip the v3 core PPL/BPB section ACTIVE vs keep v3 downstream-only (suite.md).
- Relabel the 5 still-'directional' runs after resolving the data-run verdict.json vs README contradiction.

## Execution log — 2026-07-23 (continued)
- Tier 2 #12 (partial): un-stranded 4 finished techniques so next-best is correct again (7852728).
- Tier 3 #16 (safety-killers first): thermal-kill path + safe_cuda + jax_safe_env now tested
  (test_guards.py new; test_sentinel.py +6 thermal tests). Full gate 433 passed (8ab1d2b).
- All pushed to origin/harden-research-loop.
Remaining Tier 2/3 (each needs a bit of your steer): #10 adopted-run protocol, #11 c5 schema+lint /
  CLAUDE.md verdict-timing, #13 dedup, #14 calibration wiring, #15 doc reframe + meta-scrub,
  #17 shared-box lock, #18 loop_state fsync, #19 kdump [HUMAN]. GPU items #5/#9 still await a rung gap + go.

## Execution log — 2026-07-23 (Tier 2 pass)
- #13 dedup hardened: arXiv-id + fuzzy-title (dcf01e0). Verified vs live ledger; 36 tests.
- #11a c5_validate.py: machine-checkable §C5 pre-launch lint (dd9661a). Both live HybridSSM c5 files PASS 7/7.
- #11b CLAUDE.md verdict-timing contradiction corrected (entry-at-launch; verdict written at finish). [local file]
Remaining Tier 2: #10 adopted-run protocol [.claude/, bigger], #14 calibration wiring [bigger], #15 doc reframe + meta-scrub [local docs].

## Execution log — 2026-07-23 (incident recovery + Tier 1/2/3 sweep)

INCIDENT: the hand-launched arch ladder (driver pid 1808201) is DEAD. Timeline reconstructed from
logs + `last -x`: sentinel's thermal killer fired twice (14:16:46Z gpu85/soc92, 14:21:37Z gpu86/soc91),
then the **box hard-locked at 15:24:38 BST** (boot -1 ends there, no panic/OOM trace — the documented
kdump-less lockup). 7/15 cells done (42M rung complete + ssm_base_85M + swa128_85M). Safety system worked;
the ladder DRIVER's gate did not. Root cause was NOT a metric mismatch (driver + sentinel agree ~69C) —
it was a structural cool-down gate: 1-sample accept, 30-min fall-through that LAUNCHED anyway, discarded
return value, and COOL_C=70 sitting at the box's own idle floor. Verified via a 2-workflow recon (5 read-only
audits + adversarial cross-check) then an 8-lane fix workflow (each diff adversarially reviewed).

DONE (committed):
- **Tier 3 cooldown fix** (#16-adjacent): run_arch_ladder.sh cool_down now requires 6 sustained sub-COOL_C
  samples (3 min dwell), DEFERS on the bounded fall-through (return 1 → engages the hot-spell backoff), and
  honours its return value at the call site. COOL_C 70→58 (= sentinel KILL 90 − measured +31C idle→load
  transient − 1). Kept the reviewer's 63–65 suggestion OUT: 65+31=96 > 90 would re-cross the kill line.
- **Tier 1 #4 scorer FIX**: score_arch_ladder.py resolved checkpoints under LDIR but the trainer writes them
  under BUILD → it scored 0/7 cells and exited 0 (the exact silent no-op it was meant to kill). Fixed the
  path, made it fail-loud on 0-scored-with-.done, added a pure-path --smoke check. (GPU end-to-end still pending.)
- **sentinel marker race**: kill marker now written BEFORE the SIGTERM→SIGKILL grace loop (only 1 of 2 kills
  left a marker before). + regression test pinning the ordering.
- **Tier 1 #8 verdict vocab**: eval_completeness.gate_verdict emits null/promising, never `directional`.
  Fixed a floor BYPASS (disallowed-sole-signal only fired at len(present)==1, so valppl-n1 + any 2nd item —
  even a figure — reached promising/win); now floors to inconclusive whenever no admissible signal remains.
- **Tier 1 #18 loop_state durability**: parent-dir fsync + advisory lock + mode preservation to match ledger.py;
  new register() setter for the flat in-flight fields.
- **Tier 2 #12 ledger integrity**: 17 dangling detail_md nulled, 33 stray eval keys migrated into metrics{},
  unknown-run-key hygiene warning added; cross-lane provenance keys (c5_lint, adopted, evidence_path,
  reconciled) added to RUN_ADDITIVE_KEYS.
- **Tier 2 #10 adopted-run protocol**: research/adopt_run.py (adopt + reconcile, dry-run by default) + tests.
  Standalone tool for now — wiring into ablation-runner/liveness-cron is the follow-up.
- **Tier 2 #14 capture half**: research/calibration_pairs.py (read-only join, honestly prints n=0 today). No
  ECE computed — 0 realised pairs.
- **Tier 3 #16 safety tests**: trainer_alive() argv[0] guard + recovery-chain (exit-4 routing, flock) tested.
- **Tier 1 #15 doc reframe**: "autonomous nightly loop" → "rigor factory / propose-only GPU" across CLAUDE.md,
  AGENTS.md, README.md + strategy-doc meta-commentary scrub. (Most are gitignored; on-disk only.)
- **RNG-checkpoint fix** (user decision): train_hybrid.py now saves/restores the PRNG key so a resumed cell
  continues its exact data-window stream (round-trip verified bit-exact). Discarded the swa128_nope_85M ckpt
  (renamed .discarded_rng_confound_20260723) so it reruns clean — per the user's "fix RNG then rerun" choice.
- **RECONCILED the dead run** (sanctioned CLI only): arch-ladder + orphan s1-pretrain runs → `crashed`;
  technique hybrid-attention-rethink queued→`running` (E-lane wrongly made it a fresh next-best launch);
  loop_state in-flight pointer cleared via register(); stale sentinel_kill marker archived. `sentinel liveness`
  now exits 0; next-best no longer surfaces an in-flight technique. Full suite 537 passed / 1 skipped; fsck clean.

USER DECISIONS (GPU work — propose-only, human-triggered; NOT launched this session):
- Ladder: restart as soon as fixes land (rely on the dwell gate, no overnight window).
- Resume: fix RNG checkpointing then RERUN swa128_nope_85M clean (done: code + ckpt discarded).
- NorMuon-at-scale (#9): INSERT NOW, before the 85M rung resumes (the ladder is stopped, cheapest switch point).
  → SEQUENCING TENSION between "restart ladder ASAP" and "NorMuon first": resolved as NorMuon-at-scale is next
    in the GPU queue, ladder stays live/restart-ready and resumes right after. Both await the human GPU trigger.

STILL OPEN: #5 per-arm LR probe [GPU], #9 NorMuon-at-scale launch prep [GPU/human], #17 shared-box lock,
#19 kdump/panic GRUB fix [HUMAN sudo — the box hard-locked AGAIN today, still undiagnosable], recovery-cron
paste [HUMAN], wiring adopt_run/calibration_pairs into their callers.
