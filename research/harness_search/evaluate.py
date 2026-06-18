#!/usr/bin/env python3
"""Deterministic ORACLE for the harness-search experiment. Imports a candidate
harness file, runs its pack() on a split's instances under a hard timeout +
exception guard (a broken/slow candidate scores 0, never hangs or crashes the
loop), validates every packing, and writes the score + an execution TRACE next
to the candidate. The proposer agents NEVER score themselves — this script is the
single source of truth, exactly so the search can't be gamed.

Usage:  python evaluate.py <harness.py> [--split search|heldout] [--seed N]
Writes: <harness.py>.score.json   and   <harness.py>.trace.txt
"""
import importlib.util
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task  # noqa: E402

TIMEOUT_S = 20


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _load_pack(path):
    spec = importlib.util.spec_from_file_location("candidate_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "pack"):
        raise AttributeError("harness defines no pack(items, capacity)")
    return mod.pack


def evaluate(path, split, seed=None):
    s = task.SPLIT_SEED[split] if seed is None else seed
    insts = task.make_instances(s)
    try:
        pack = _load_pack(path)
    except Exception as e:  # import/syntax error -> 0
        return {"score": 0.0, "split": split, "valid_fraction": 0.0,
                "n": len(insts), "error": f"load: {type(e).__name__}: {e}"}, ""

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(TIMEOUT_S)
    effs, valid = [], 0
    try:
        for items in insts:
            bins = pack(list(items), task.CAPACITY)
            if task.validate(items, bins, task.CAPACITY):
                effs.append(task.efficiency(items, bins))
                valid += 1
            else:
                effs.append(0.0)
        signal.alarm(0)
    except Exception as e:  # timeout or runtime error -> 0
        signal.alarm(0)
        return {"score": 0.0, "split": split, "valid_fraction": valid / len(insts),
                "n": len(insts), "error": f"run: {type(e).__name__}: {e}"}, ""

    score = sum(effs) / len(effs)
    # TRACE: for a few instances, the bins sorted by WASTE — this is the raw
    # diagnostic the full-traces arm gets and the scores-only arm does not.
    trace_lines = [f"score={score:.5f} valid_fraction={valid/len(insts):.3f} "
                   f"(efficiency = sum(items)/bins; 1.0 = perfect)"]
    for k, items in enumerate(insts[:3]):
        bins = pack(list(items), task.CAPACITY)
        rows = sorted(((task.CAPACITY - sum(b), b) for b in bins if b), reverse=True)
        trace_lines.append(
            f"instance {k}: {len(rows)} bins; most-wasted -> "
            + "; ".join(f"[waste={w:.3f} fill={sum(b):.3f} items={len(b)}]"
                        for w, b in rows[:6]))
    return {"score": round(score, 5), "split": split,
            "valid_fraction": valid / len(insts), "n": len(insts)}, "\n".join(trace_lines)


def main(argv):
    if len(argv) < 2:
        print("usage: evaluate.py <harness.py> [--split search|heldout] [--seed N]")
        return 2
    path = argv[1]
    split = "search"
    seed = None
    no_trace = False
    i = 2
    while i < len(argv):
        if argv[i] == "--split":
            split = argv[i + 1]
            i += 2
        elif argv[i] == "--seed":
            seed = int(argv[i + 1])
            i += 2
        elif argv[i] == "--no-trace":   # scores-only ablation: emit NO trace file
            no_trace = True
            i += 1
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
