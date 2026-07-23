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
