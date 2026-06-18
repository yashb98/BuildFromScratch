# codeharness — the coding-agent harness-search plugin

Searches the **harness around Claude-when-writing-code** (prompt / context /
post-processing strategy), the paper's flagship target (arXiv 2603.28052,
TerminalBench-2). The model is FROZEN; only the harness is evolved.

## What's built + tested NOW (CPU, no model)
- `benchmark.py` — 10 HumanEval-style tasks, fixed SEARCH (5) / HELDOUT (5) split.
- `runner.py` — **sandboxed** (`subprocess` + timeout + temp cwd) test-runner +
  `score(solve_fn, tasks) -> (pass_rate, valid_fraction)`. The deterministic reward.
- `baseline_harness.py` — the incumbent harness (zero-shot prompt) with the agent
  call **injected** (`make_solve(agent_fn, harness)`), so the reward is testable
  without the model.
- `research/tests/test_codeharness.py` — 9 tests (reference passes, wrong fails,
  infinite-loop times out, scorer + harness wiring). All green.

## What's STAGED (needs the model + an idle box)
The agent-in-the-loop SEARCH. The reward needs the agent to actually WRITE code, so:

1. Provide a real `agent_fn(prompt) -> text` (a Claude/model call).
2. **Proposer loop** (a Workflow, like the bin-packing run): each iteration a
   proposer reads prior harness configs + their scores + the FULL trace (which
   tasks failed and the stderr), writes an improved harness (new system prompt /
   few-shot from solved tasks / a self-check-and-repair loop / better fence
   parsing) and scores it via `runner.score(make_solve(agent_fn, harness),
   benchmark.SEARCH_TASKS)`.
3. **Gate**: `framework.select_and_promote(scorer, candidates, incumbent,
   search_seed, heldout_seeds, direction="higher_is_better")` on
   `benchmark.HELDOUT_TASKS`, with seeds = agent SAMPLING seeds (temperature) so
   the held-out variance is real. Promote ONLY on a significant held-out pass-rate
   win — the exact gate that, in our bin-packing run, refused a brittle winner.

## Security (do before any unattended/agent-written-code search)
`runner._run` executes candidate code with only `subprocess` + timeout — fine for
our reference solutions. For code an agent writes against untrusted tasks, isolate
`_run` in **Docker with no network** (the paper containerizes TerminalBench
execution). Wire that at the `_run` boundary first.

## Why this is the highest-value target
Coding failures are NON-OBVIOUS, so this is where execution TRACES beat scalar
scores most (the paper's qualitative win was a coding proposer reading traces to
isolate a confounded regression). Unlike bin-packing — where a strong agent didn't
need traces — here they should matter.
