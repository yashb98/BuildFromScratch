"""Deterministic SANDBOXED reward machinery for the coding-agent harness target.
Given candidate code for a task, run it + the task's hidden tests in a SUBPROCESS
(fresh interpreter, hard timeout, temp dir) and return pass/fail. Then `score`
turns a harness's per-task code into a pass-rate over a split.

The candidate CANNOT self-report a pass (this is the oracle contract the packing
evaluate.py also obeys). A trusted driver — fed to the interpreter over STDIN, so
the pass sentinel lives only in interpreter memory, not in any file or in
/proc/self/cmdline the candidate could read — loads the candidate as a module,
REJECTS any attempt to exit during its own definition, REQUIRES the entry_point to
be callable, runs the hidden tests, and prints a tests-keyed sentinel ONLY after
every assert held. A pass needs exit 0 AND that sentinel on stdout, so a candidate
that early-exits (sys.exit / os._exit / raise SystemExit) before the tests, or
never defines the entry_point, can no longer score a free pass. (Before this
guard, `passed = returncode==0` let `import sys; sys.exit(0)` self-report 1.0 — a
reward-hack an optimizing harness would have found.)

Security note: candidate code is still EXECUTED. subprocess + timeout + a temp cwd
is a BASIC guard (fine for our own reference solutions and tests) and the sentinel
defeats trivial reward-hacks, but it is NOT a security/memory sandbox: a malicious
candidate could read interpreter memory, or a memory-bomb (e.g. [0]*10**12) could
overcommit the GB10 unified-memory pool and crash the box. Before any UNATTENDED
agent-written-code search, run `_run` inside Docker with `--network=none`, a
read-only mount, and CPU/memory limits (the paper containerizes TerminalBench
execution). Wire that at the `_run` boundary. The bulletproof upgrade for the
self-report contract is a secret-expected-output grader (the candidate computes
outputs in a child it controls; the trusted parent holds the expected values it
never sees) — adopt it when the real search is wired.

Pure-CPU, no model. The runner is fully unit-testable now (test_codeharness.py);
only the harness that PRODUCES the code needs the agent.
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_S = 10


def _driver(entry_point, tests, sentinel):
    """The TRUSTED grader program (run via stdin, never written to a file). It
    loads the candidate module, rejects an exit during definition, requires a
    callable entry_point, runs the hidden tests in the candidate's namespace, and
    prints `sentinel` iff every assert held."""
    return (
        "import sys, importlib.util\n"
        "_spec = importlib.util.spec_from_file_location('cand', 'cand.py')\n"
        "_m = importlib.util.module_from_spec(_spec)\n"
        "try:\n"
        "    _spec.loader.exec_module(_m)\n"            # SystemExit is a BaseException -> caught
        "except BaseException as _e:\n"
        "    sys.stderr.write('reject:exit-during-define:%s\\n' % type(_e).__name__); sys.exit(11)\n"
        "_fn = getattr(_m, " + repr(entry_point) + ", None)\n"
        "if not callable(_fn):\n"
        "    sys.stderr.write('reject:no-entry-point\\n'); sys.exit(12)\n"
        "try:\n"
        "    exec(compile(" + repr(tests) + ", '<tests>', 'exec'), _m.__dict__)\n"
        "except BaseException as _e:\n"
        "    sys.stderr.write('fail:%s\\n' % type(_e).__name__); sys.exit(1)\n"
        "print(" + repr(sentinel) + ")\n"               # only reached iff all asserts passed
    )


def _run(cand_code, driver_code, timeout=TIMEOUT_S):
    """Execution boundary: write the candidate to a temp dir and run the trusted
    driver (over stdin) against it in a subprocess. Returns (returncode, stdout).
    Wrap THIS call in Docker (--network=none, read-only, mem-limited) before any
    unattended agent-written-code search."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "cand.py").write_text((cand_code or "") + "\n")
        try:
            p = subprocess.run([sys.executable, "-"], input=driver_code, cwd=d,
                               timeout=timeout, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return None, ""                              # treated as not-passed
        except Exception as e:  # pragma: no cover - defensive
            return None, f"spawn: {type(e).__name__}: {e}"
    return p.returncode, (p.stdout or "")


def run_solution(code, task, timeout=TIMEOUT_S):
    """True iff `code` defines task['entry_point'] AND passes every hidden test —
    decided by the trusted driver's sentinel, never by the candidate's exit code.
    The sentinel is keyed on the (hidden) tests, so a static candidate that never
    receives the tests cannot hardcode it."""
    ep, tests = task["entry_point"], task["tests"]
    sentinel = "__CODEHARNESS_PASS__" + hashlib.sha256(
        (ep + "\x00" + tests).encode()).hexdigest()[:24]
    rc, out = _run(code, _driver(ep, tests, sentinel), timeout)
    return rc == 0 and sentinel in out


def score(solve_fn, tasks, timeout=TIMEOUT_S):
    """Run a harness over a set of tasks. `solve_fn(task) -> code_string` is the
    harness (in production it drives the agent; here it can be any callable).
    Returns (pass_rate, valid_fraction): pass_rate = fraction of tasks solved;
    valid_fraction = fraction where the harness returned runnable code at all
    (a harness that errors/empties out on a task is 'invalid' there -> the
    framework's brittleness gate will exclude it)."""
    passed, valid = 0, 0
    for t in tasks:
        try:
            code = solve_fn(t)
        except Exception:
            continue  # harness crashed on this task -> not valid, not passed
        if not isinstance(code, str) or not code.strip():
            continue
        valid += 1
        if run_solution(code, t, timeout):
            passed += 1
    n = len(tasks)
    return (passed / n if n else 0.0), (valid / n if n else 0.0)
