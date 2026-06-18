#!/usr/bin/env python3
"""Deterministic ORACLE for the sequence-packing harness-search target. Mirrors
the bin-packing evaluate.py contract exactly so the framework's scorer works
unchanged: imports a candidate `pack(doc_lengths, capacity)`, runs it on a
split's instances under a hard timeout + exception guard (a broken/slow candidate
scores 0, never hangs the loop), validates every packing, and writes
<harness>.score.json + <harness>.trace.txt. The proposer agents NEVER score
themselves — this is the single source of truth.

Usage:  python evaluate.py <harness.py> [--split search|heldout] [--seed N] [--no-trace]
"""
import importlib.util
import json
import signal
import sys
from pathlib import Path

# Bind THIS target's task by EXPLICIT path under a UNIQUE module name. A bare
# `import task` resolves to whichever task.py is first on sys.path — when more
# than one harness-search target is importable in the same process (e.g. an
# in-process test driver with harness_search/ on the path) it silently picks up
# a SIBLING target's task.py. That shadowing made the seq_packing tests
# false-green: they passed while exercising the bin-packing oracle. A unique
# spec name + explicit file path makes the oracle's task binding immune to
# sys.path ordering and to the sys.modules cache.
_task_spec = importlib.util.spec_from_file_location(
    "seq_packing_task", Path(__file__).resolve().parent / "task.py")
task = importlib.util.module_from_spec(_task_spec)
_task_spec.loader.exec_module(task)  # noqa: E402  (the LOCAL seq_packing task)

TIMEOUT_S = 20


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _load_pack(path):
    spec = importlib.util.spec_from_file_location("candidate_seq_pack", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "pack"):
        raise AttributeError("harness defines no pack(doc_lengths, capacity)")
    return mod.pack


def evaluate(path, split, seed=None):
    s = task.SPLIT_SEED[split] if seed is None else seed
    insts = task.make_instances(s)
    try:
        pack = _load_pack(path)
    except Exception as e:
        return {"score": 0.0, "split": split, "valid_fraction": 0.0,
                "n": len(insts), "error": f"load: {type(e).__name__}: {e}"}, ""

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(TIMEOUT_S)
    effs, valid = [], 0
    try:
        for docs in insts:
            seqs = pack(list(docs), task.CAPACITY)
            if task.validate(docs, seqs, task.CAPACITY):
                effs.append(task.efficiency(docs, seqs))
                valid += 1
            else:
                effs.append(0.0)
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        return {"score": 0.0, "split": split, "valid_fraction": valid / len(insts),
                "n": len(insts), "error": f"run: {type(e).__name__}: {e}"}, ""

    score = sum(effs) / len(effs)
    trace_lines = [f"score={score:.5f} valid_fraction={valid/len(insts):.3f} "
                   f"(efficiency = real_tokens/(n_seq*{task.CAPACITY}); 1.0 = no padding)"]
    for k, docs in enumerate(insts[:3]):
        seqs = pack(list(docs), task.CAPACITY)
        rows = sorted(((task.CAPACITY - sum(b), b) for b in seqs if b), reverse=True)
        trace_lines.append(
            f"instance {k}: {len(rows)} sequences; most-padded -> "
            + "; ".join(f"[pad={w} fill={sum(b)} docs={len(b)}]" for w, b in rows[:6]))
    return {"score": round(score, 5), "split": split,
            "valid_fraction": valid / len(insts), "n": len(insts)}, "\n".join(trace_lines)


def main(argv):
    if len(argv) < 2:
        print("usage: evaluate.py <harness.py> [--split search|heldout] [--seed N] [--no-trace]")
        return 2
    path, split, seed, no_trace, i = argv[1], "search", None, False, 2
    while i < len(argv):
        if argv[i] == "--split":
            split = argv[i + 1]; i += 2
        elif argv[i] == "--seed":
            seed = int(argv[i + 1]); i += 2
        elif argv[i] == "--no-trace":
            no_trace = True; i += 1
        else:
            i += 1
    res, trace = evaluate(path, split, seed)
    Path(str(path) + ".score.json").write_text(json.dumps(res, indent=2) + "\n")
    if not no_trace:
        Path(str(path) + ".trace.txt").write_text(trace + "\n")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
