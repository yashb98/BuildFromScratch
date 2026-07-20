"""Two harnesses for the headroom check, differing ONLY in whether they use
execution TRACES:

  no_repair   – one shot: build prompt -> agent_fn -> strip fences. (scalar-blind)
  self_repair – build prompt -> agent_fn -> RUN the public tests -> on failure feed
                the actual traceback back to the model and regenerate (<= max_repairs
                rounds). It consumes the execution trace; the no-repair harness can't.

Both return CODE that is then graded by the FIXED runner.run_solution on the HIDDEN
tests (which neither harness ever sees). If self_repair's hidden pass rate beats
no_repair's, traces carry signal on this benchmark -> real headroom for a search.
The repair loop runs only the task's PUBLIC tests (not secret), so reading their
failures is fair; the grade stays hack-proof because hidden grading uses the
sentinel runner, not this loop.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_harness as bh  # noqa: E402  (strip_fences + build_prompt + INCUMBENT)

TIMEOUT_S = 10
MAX_REPAIRS = 2


def run_public(code, task):
    """Run `code` against the task's PUBLIC tests; return (passed, trace_tail). The
    trace_tail is the last chunk of stderr (the failing assert + traceback) that the
    self-repair harness feeds back to the model."""
    program = (code or "") + "\n\n" + task["public_tests"] + "\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "c.py"
        f.write_text(program)
        try:
            p = subprocess.run([sys.executable, str(f)], cwd=d, timeout=TIMEOUT_S,
                               capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return False, "timeout: the code did not finish on the public tests"
        except Exception as e:  # pragma: no cover
            return False, f"spawn error: {type(e).__name__}: {e}"
    if p.returncode == 0:
        return True, ""
    return False, ((p.stderr or "").strip()[-800:] or f"exit {p.returncode}")


def _repair_prompt(task, code, trace):
    return (f"Your Python solution failed a test.\n\n"
            f"Problem:\n{task['prompt']}\n\n"
            f"Your code:\n{code}\n\n"
            f"Running the public tests produced this error:\n{trace}\n\n"
            f"Return ONLY the corrected function definition — no prose, no markdown fences.")


def make_no_repair_solve(agent_fn):
    def solve(task):
        return bh.strip_fences(agent_fn(bh.build_prompt(task, bh.INCUMBENT)))
    return solve


def make_self_repair_solve(agent_fn, max_repairs=MAX_REPAIRS):
    def solve(task):
        code = bh.strip_fences(agent_fn(bh.build_prompt(task, bh.INCUMBENT)))
        for _ in range(max_repairs):
            passed, trace = run_public(code, task)
            if passed:
                break
            code = bh.strip_fences(agent_fn(_repair_prompt(task, code, trace)))
        return code
    return solve
