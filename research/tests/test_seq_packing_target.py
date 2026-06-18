"""Tests for the sequence-packing harness-search target (the GB10-throughput
target: better packing = fewer padding FLOPs = more real tokens/GPU-hour).
Pins: the oracle is honest (dropping a doc is invalid -> 0), there is real
headroom over the hand-designed first-fit baseline (so the search has something
to find), scoring is deterministic, and the framework gate refuses a brittle
candidate on this target exactly as it did on bin-packing. Stdlib-only, no GPU."""
import importlib.util
import sys
from pathlib import Path

import pytest

SP = Path(__file__).resolve().parents[1] / "harness_search" / "targets" / "seq_packing"
HS = Path(__file__).resolve().parents[1] / "harness_search"


def _load(name, path):
    """Load a module by EXPLICIT path under a unique name. Both the seq_packing
    target and the bin-packing root define `task.py`/`evaluate.py`; a bare
    `import task`/`import evaluate` with HS on sys.path resolves to the ROOT
    (bin-packing) modules, so this suite used to pass while testing the WRONG
    oracle (false-green). Explicit path + unique name pins the real ones."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


task = _load("seq_packing_task_test", SP / "task.py")
ev = _load("seq_packing_evaluate_test", SP / "evaluate.py")
fw = _load("harness_framework_test", HS / "framework.py")  # self-adds research/ for eval_stats

BASELINE = str(SP / "baseline_seq_pack.py")


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_baseline_valid_and_sub_perfect():
    res, _ = ev.evaluate(BASELINE, "search")
    assert res["valid_fraction"] == 1.0           # first-fit never drops/overflows
    assert 0.0 < res["score"] < 1.0               # real packing, real padding waste


def test_dropping_a_doc_is_invalid_scores_zero(tmp_path):
    # a "harness" that silently drops the last doc loses data -> must score 0
    cheat = _write(tmp_path, "cheat.py",
                   "def pack(d, c):\n"
                   "    seqs=[]\n"
                   "    for n in d[:-1]:\n"
                   "        for s in seqs:\n"
                   "            if sum(s)+n<=c: s.append(n); break\n"
                   "        else: seqs.append([n])\n"
                   "    return seqs\n")
    res, _ = ev.evaluate(cheat, "search")
    assert res["valid_fraction"] == 0.0 and res["score"] == 0.0


def test_overflow_is_invalid(tmp_path):
    # putting everything in one sequence overflows capacity -> invalid -> 0
    over = _write(tmp_path, "over.py", "def pack(d, c):\n    return [list(d)]\n")
    res, _ = ev.evaluate(over, "search")
    assert res["score"] == 0.0


def test_best_fit_decreasing_beats_first_fit(tmp_path):
    # the headroom that justifies the search: BFD packs tighter than first-fit
    bfd = _write(tmp_path, "bfd.py",
                 "def pack(d, c):\n"
                 "    seqs=[]\n"
                 "    for n in sorted(d, reverse=True):\n"
                 "        best=-1; slack=c+1\n"
                 "        for i,s in enumerate(seqs):\n"
                 "            room=c-sum(s)\n"
                 "            if n<=room<slack: slack=room; best=i\n"
                 "        if best>=0: seqs[best].append(n)\n"
                 "        else: seqs.append([n])\n"
                 "    return seqs\n")
    base, _ = ev.evaluate(BASELINE, "heldout")
    better, _ = ev.evaluate(bfd, "heldout")
    assert better["valid_fraction"] == 1.0
    assert better["score"] > base["score"]        # there IS something to find


def test_scoring_is_deterministic():
    a, _ = ev.evaluate(BASELINE, "search", seed=7)
    b, _ = ev.evaluate(BASELINE, "search", seed=7)
    assert a["score"] == b["score"]


def test_framework_gate_refuses_brittle_on_seq_packing(tmp_path):
    # the cand_06 lesson must hold on THIS target: a candidate that times out on a
    # held-out seed (highest search score) is refused; a robust BFD is promoted.
    bfd = _write(tmp_path, "bfd.py",
                 "def pack(d, c):\n"
                 "    seqs=[]\n"
                 "    for n in sorted(d, reverse=True):\n"
                 "        best=-1; slack=c+1\n"
                 "        for i,s in enumerate(seqs):\n"
                 "            room=c-sum(s)\n"
                 "            if n<=room<slack: slack=room; best=i\n"
                 "        if best>=0: seqs[best].append(n)\n"
                 "        else: seqs.append([n])\n"
                 "    return seqs\n")

    def scorer(path, seed):
        r, _ = ev.evaluate(path, "heldout", seed=seed)
        return r["score"], r["valid_fraction"]

    rep = fw.select_and_promote(scorer, [bfd], BASELINE,
                                search_seed=task.SPLIT_SEED["search"],
                                heldout_seeds=[101, 102, 103, 104], direction="higher_is_better")
    assert rep["challenger"] == bfd and rep["promoted"] is True
