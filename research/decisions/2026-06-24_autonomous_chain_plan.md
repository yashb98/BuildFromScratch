# Autonomous on-box chain — auto-advance, pause only at hard gates

**Directive (2026-06-24):** when one step finishes, the next starts automatically — no waiting on
the user between on-box steps. Mechanism: each step runs detached + a completion watcher
(`run_in_background`) that re-invokes me; on re-invocation I execute the next step under the §C5
bounded-auto-run contract (preflight → smoke → iso-FLOP → ledger-before-launch → detached →
sentinel) and arm the next watcher. No step is launched without §C5 passing.

## The chain (each arrow = an automatic handoff on completion)

1. **Data-arm materialization** (running, watcher `bd5my3ol5`)
   → write dataset card + ledger `dataset-prep` entry.
2. **Data-selection A/B** — `/ablation-runner` cohort: **control = FineWeb-Edu** (the existing stream)
   vs **treatment = dclm-edu** (the new shards), matched ~150M tokens, 3 seeds each, single-variable
   (only the data slice changes), §C5 + iso-FLOP. Trainer variant reads the prepared shards.
   → on done: score **OOD-BPB (text-lm-v2) + downstream (text-lm-v3) + §C25**, across-seed-CI verdict.
3. **Verdict → README + commit.** Use the better-or-equal data slice for everything downstream.
4. **Mid-training** (anneal on the winning data @ low LR + RoPE context-extension), §C5.
   → score (RULER short-ctx non-regression + BPB), verdict, commit.
5. **Post-training — SFT (≥3 seeds + paired control + forgetting probe)**, then **GRPO** (on-box GSM8K
   scale: pass@1 + spurious-reward control arms), §C5 each.
   → score (the §C13 + §C25 post-train battery), verdict, commit.
6. **Serving-bench (on-box)** — vLLM export shim + p50/p99 + `--quant fp8` quality-vs-floor.
   → record, commit.
7. **/manuscript** — only once a stage headline is §C25-complete + significant.

## HARD PAUSE points (I STOP and surface to you — never auto-spend/auto-decide these)

- **Any rented / off-box step (§C20):** the multi-scale × cross-data arch grid, distributed-scaling
  table, kernel roofline, dangerous-capability uplift, any LLM-as-judge pass. These cost real money
  and need your authorization — I queue a proposal and STOP.
- **A genuine branch point / surprise:** a result that invalidates the plan, a verdict that flips a
  prior claim, or a §C25 `incomplete-eval` that can't be completed on-box. I report + ask.
- **Box-safety trip:** any sentinel kill / preflight FAIL that isn't the benign "one job running"
  state. I diagnose, don't blindly relaunch (≤2 auto-resumes, §C5 tail).
- **A judged `loss`** that would otherwise auto-launch the next dependent step → I report the loss
  and the re-plan, not silently proceed.

## Bounds (so it can't run away)
- One GPU job at a time (§C4.5); each launch is §C5-gated; sentinel-guarded; ledger-recorded.
- You can interrupt at any time; the chain is resumable (each step's state is in the ledger +
  its experiment dir).
- The chain ENDS at /manuscript (lifecycle demonstrated) or at the first hard-pause gate.
