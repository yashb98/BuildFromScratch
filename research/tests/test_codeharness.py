"""Tests for the coding-agent-harness plugin's DETERMINISTIC reward machinery
(research/harness_search/tasks/codeharness/{benchmark,runner,baseline_harness}.py).
The runner that executes candidate code + hidden tests in a sandbox is the
testable-now core; only the agent that PRODUCES the code is staged for the box.

Module names (benchmark/runner/baseline_harness) are unique, so importing them in
the shared suite is clash-free.
"""
import sys
from pathlib import Path

CH = Path(__file__).resolve().parents[1] / "harness_search" / "tasks" / "codeharness"
sys.path.insert(0, str(CH))

import benchmark as bm          # noqa: E402
import runner as rn            # noqa: E402
import baseline_harness as bh  # noqa: E402


# ---- the sandbox runner: reference passes, wrong fails, infinite-loop times out

def test_reference_solution_passes():
    t = bm.BY_ID["fib"]
    assert rn.run_solution(t["reference"], t) is True


def test_wrong_solution_fails():
    t = bm.BY_ID["fib"]
    assert rn.run_solution("def fib(n):\n    return 123", t) is False


def test_syntax_error_fails():
    t = bm.BY_ID["add"]
    assert rn.run_solution("def add(a, b)\n    return a+b", t) is False  # missing colon


def test_infinite_loop_times_out():
    t = bm.BY_ID["add"]
    assert rn.run_solution("def add(a,b):\n    while True: pass", t, timeout=2) is False


# ---- reward-hack guards: the candidate must NOT be able to self-report a pass.
# The pass signal is the trusted driver's tests-keyed sentinel, not the candidate's
# exit code, so an early-exit-before-tests or no-entry-point candidate scores 0.

def test_early_exit_before_tests_is_not_a_pass():
    t = bm.BY_ID["add"]
    assert rn.run_solution("import sys\nsys.exit(0)", t) is False        # SystemExit
    assert rn.run_solution("import os\nos._exit(0)", t) is False         # hard exit, code 0
    assert rn.run_solution("raise SystemExit(0)", t) is False            # raised SystemExit


def test_no_entry_point_is_not_a_pass():
    t = bm.BY_ID["add"]
    assert rn.run_solution("x = 1  # never defines add()", t) is False


def test_exit_zero_harness_scores_zero_not_promotable():
    # end-to-end: a harness emitting sys.exit(0) for every task scores 0.0 pass-rate
    # (the framework gate would refuse it), NOT a self-reported perfect 1.0.
    rate, _ = rn.score(lambda t: "import sys\nsys.exit(0)", bm.HELDOUT_TASKS)
    assert rate == 0.0


# ---- the scorer: reference -> 1.0, broken -> 0.0, valid_fraction tracks runnable

def test_score_reference_is_perfect():
    rate, valid = rn.score(lambda t: t["reference"], bm.SEARCH_TASKS)
    assert rate == 1.0 and valid == 1.0


def test_score_broken_is_zero_but_valid():
    # returns a runnable-but-wrong stub for every task: valid (code ran) but 0 pass.
    solve = lambda t: f"def {t['entry_point']}(*a, **k):\n    return None"
    rate, valid = rn.score(solve, bm.HELDOUT_TASKS)
    assert rate == 0.0 and valid == 1.0


def test_score_crashing_harness_is_invalid():
    # a harness that raises on a task -> that task is neither valid nor passed.
    def solve(t):
        raise RuntimeError("harness failed")
    rate, valid = rn.score(solve, bm.SEARCH_TASKS)
    assert rate == 0.0 and valid == 0.0


# ---- the harness wiring (build_prompt -> agent_fn -> strip_fences -> code)

def test_make_solve_with_fake_agent_passes():
    # fake agent returns the reference solution wrapped in markdown fences;
    # strip_fences must recover runnable code that then passes the tests.
    def fake_agent(prompt):
        ep = prompt.split("def ", 1)[1].split("(", 1)[0].strip()  # entry point from prompt
        return f"```python\n{bm.BY_ID[ep]['reference']}\n```"
    solve = bh.make_solve(fake_agent)
    rate, valid = rn.score(solve, bm.SEARCH_TASKS)
    assert rate == 1.0 and valid == 1.0


def test_strip_fences_variants():
    assert bh.strip_fences("```python\ndef f(): pass\n```") == "def f(): pass"
    assert bh.strip_fences("def f(): pass") == "def f(): pass"
    assert bh.strip_fences("```\nx=1\n```") == "x=1"
