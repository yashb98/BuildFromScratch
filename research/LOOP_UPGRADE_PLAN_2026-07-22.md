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
