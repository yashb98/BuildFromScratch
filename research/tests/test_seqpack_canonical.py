"""Regression tests for the CANONICAL sequence-packing target
(research/harness_search/tasks/seqpack/). The headline guard is shadow-resistance:
this exact component has twice been silently broken by a bare `import task`
resolving to the bin-packing root task.py (the false-green incident, then a
regression in this rewrite). test_oracle_is_shadow_resistant pins the fix —
poison sys.modules['task'] with the bin-packing task, then assert the seqpack
oracle STILL resolves its own task and scores the baseline correctly. Stdlib-only,
no GPU, no model."""
import importlib.util
import sys
from pathlib import Path

SP = Path(__file__).resolve().parents[1] / "harness_search" / "tasks" / "seqpack"
HS = Path(__file__).resolve().parents[1] / "harness_search"
BASELINE = str(SP / "baseline.py")


def _load(name, path):
    """Load a module by EXPLICIT path under a unique name (the test must not itself
    fall into the shadowing trap it is guarding against)."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


task = _load("seqpack_task_test", SP / "task.py")
ev = _load("seqpack_evaluate_test", SP / "evaluate.py")


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_baseline_valid_and_sub_perfect():
    res, _ = ev.evaluate(BASELINE, "heldout")
    assert res["valid_fraction"] == 1.0
    assert 0.0 < res["score"] < 1.0
    assert res["n"] == task.N_INSTANCES        # 30 — proves the LOCAL task, not bin-packing's 40


def test_oracle_is_shadow_resistant():
    # Poison sys.modules['task'] with the bin-packing root task (no SEQ_LEN), exactly
    # as an in-process search loop that imported the framework/root first would. The
    # oracle must STILL bind its own seqpack task by explicit path.
    sys.path.insert(0, str(HS))
    import task as bp                            # noqa: E402 — bin-packing root task.py
    assert not hasattr(bp, "SEQ_LEN")           # confirms the poison is the wrong task
    fresh = _load("seqpack_evaluate_poisoned", SP / "evaluate.py")
    assert getattr(fresh.task, "SEQ_LEN", None) == 2048
    res, _ = fresh.evaluate(BASELINE, "heldout")
    assert res["n"] == 30 and res["valid_fraction"] == 1.0 and res["score"] == 0.97465


def test_dropping_a_doc_is_invalid_scores_zero(tmp_path):
    cheat = _write(tmp_path, "cheat.py",
                   "def pack(d, L):\n"
                   "    seqs=[]\n"
                   "    for n in d[:-1]:\n"
                   "        for s in seqs:\n"
                   "            if sum(s)+n<=L: s.append(n); break\n"
                   "        else: seqs.append([n])\n"
                   "    return seqs\n")
    res, _ = ev.evaluate(cheat, "heldout")
    assert res["valid_fraction"] == 0.0 and res["score"] == 0.0


def test_overflow_is_invalid(tmp_path):
    over = _write(tmp_path, "over.py", "def pack(d, L):\n    return [list(d)]\n")
    res, _ = ev.evaluate(over, "heldout")
    assert res["score"] == 0.0


def test_scoring_is_deterministic():
    a, _ = ev.evaluate(BASELINE, "search", seed=9)
    b, _ = ev.evaluate(BASELINE, "search", seed=9)
    assert a["score"] == b["score"]
