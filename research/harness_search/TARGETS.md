# harness-search TARGETS — the honest "every single thing" map

You asked to apply this to *everything*. The experiment (2026-06-18) and the paper's
own cost model say it can't go everywhere. A component is **searchable** iff it has:
(a) a **cheap, repeatable** evaluation (≈60 candidate runs feasible), (b) a **clean
scalar/Pareto reward**, and (c) **search + ≥3 held-out splits** so the gate can refuse
overfit/brittle winners. Below, every component is tagged honestly.

## ✅ APPLICABLE NOW — cheap, deterministic reward, runs today (no GPU, no LLM)

| Target | Reward | Status |
|---|---|---|
| **bin-packing** (reference) | fill efficiency | **done + validated** (+5.3 pts vs hand-design, significant) |
| **`from-scratch-build` SEQUENCE/DOCUMENT PACKING** — how variable-length docs are packed into fixed-length training batches to minimize PADDING waste | packing efficiency (real-tokens / total-tokens) over a fixed tokenized sample | plugin TODO — **deterministic, cheap, no training/GPU needed**; bin-packing is literally its proof-of-concept. Better packing = less wasted compute per step. The ONE searchable slice of the build harness. |
| **ledger `next-best` / `idea-selection` ranking heuristic** | ranking quality (NDCG/Kendall-τ) vs a held-out true ordering (from past runs' real win/loss, or a synthetic) | plugin TODO — deterministic, high-value |
| **`eval_metrics` contamination params** (n-gram `n`, threshold) | detection F1 on a labeled contaminated/clean set | plugin TODO — deterministic |
| **eval noise-floor / significance config** | calibration: false-positive rate on simulated NULL A/B pairs | plugin TODO — deterministic via simulation |

## 🟡 APPLICABLE WITH CHEAP INFERENCE — needs forward passes (an idle box, a small CPU model, or an API), NOT training

| Target | Reward | Blocker |
|---|---|---|
| **Claude's CODING / AGENT harness** (the prompts, context-management, tool-use logic around Claude-when-writing-code — incl. THIS harness-searcher's own proposer) | coding-task pass rate on a fixed benchmark (SWE-bench / TerminalBench-style) | **SCAFFOLDED** (`tasks/codeharness/`): the deterministic reward machinery — 10-task benchmark + sandboxed runner + scorer — is **built + tested (9 tests green)**; only the agent-in-the-loop SEARCH is staged (needs the model + an idle box; see its README). **This is the paper's TerminalBench-2 experiment** (#1 among Haiku-4.5 agents) — where TRACES MATTER MOST. Highest-value 🟡 target. |
| **eval-harness prompt / few-shot / format config** for a capability benchmark | accuracy on a labeled set | needs model inference (blocked while the trainer holds the GPU) |
| **`community-pulse` / `ml-research` retrieval & query/source selection** | relevance@k vs a labeled relevance set | needs a labeled set + network/inference |
| **agentic orchestration / `§C-spawn` prompt template** | downstream task completion | needs an LLM-in-the-loop reward |

These are where **traces likely DO matter** (unlike easy bin-packing): the failure
modes are non-obvious, exactly the regime where arXiv 2603.28052's full-trace edge shows.

## ❌ NOT APPLICABLE — and the reason is load-bearing, not laziness

| Component | Why it can't be searched |
|---|---|
| **`/ablation-runner`, `/from-scratch-build`, `/post-train` TRAINING loop** | eval = a multi-hour GPU run. ~60 candidate evals × hours = infeasible. The paradigm's own cost model forbids it — and our experiment confirmed cheap-eval is the precondition. (Exception: the build's DATA-PACKING harness IS cheap-eval — see ✅ above.) |
| **`from-scratch-build` numerical-VERIFY gate** (`max_abs_error < 1e-3` vs the HF reference) | a CORRECTNESS contract, not a reward — there is no "more optimal" than bit-exact. You satisfy it, you don't search it. |
| **the model architecture / `model.py`** | optimizing it = architecture search = you must TRAIN each variant to score it → GPU-expensive → out by the same cost model. (Meta-Harness wraps a FIXED model; from-scratch-build has no fixed model to wrap.) |
| **`sentinel.py`, `safe_cuda`, crash guards** | correctness *contract*, not an optimization reward. You **test** a kill-switch (the suite + mutation tests), you don't search it. Searching safety code is how you get a fast-but-fatal kill-switch. |
| **`ledger.py` atomic write / schema validation** | correctness, not reward (same as above). |
| **"make the digest better", "improve research taste", "better proposals"** | **no clean scalar reward.** Without a measurable reward there is nothing to optimize against — searching here would be optimizing a number you made up. |
| **the model weights themselves** | that's training, not the harness. The paper explicitly defers co-evolving harness+weights to future work. |
| **the WHOLE LLM lifecycle, end-to-end, as ONE search target** | the end-to-end reward is a slow, noisy, GPU-bound research outcome (one run = days + a win/loss verdict) — you cannot run ~60 cheap end-to-end iterations. **Decompose it:** search the cheap-eval COMPONENTS (above), and the pipeline improves piecewise. The paper never searches "all of ML research at once" — it searches individual task harnesses. |

## The honest bottom line
"Every single thing" resolves to: **the 4 deterministic-reward components (runnable now),
3 inference-reward components (need an idle box/model), and a hard NO on the GPU training
loop, the safety code, and anything without a reward.** Implementing the framework + gate
makes all of the ✅/🟡 row searchable as soon as each gets its small plugin; the ❌ row
stays hand-built-and-tested **by design**, and our own experiment is the evidence for why.
