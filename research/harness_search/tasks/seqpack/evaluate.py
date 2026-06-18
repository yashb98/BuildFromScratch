#!/usr/bin/env python3
"""Deterministic ORACLE for the sequence-packing harness-search target. Imports a
candidate packer, runs pack(docs, seq_len) on a split's instances under a hard
timeout + exception guard (broken/slow -> score 0, never hangs), validates every
packing, and writes the efficiency score + a padding trace next to the candidate.
Single source of truth; proposer agents never self-report.

Usage:  python evaluate.py <harness.py> [--split search|heldout] [--seed N] [--no-trace]
"""
import importlib.util
import json
import signal
import sys
from pathlib import Path

# Bind THIS target's task by EXPLICIT path under a UNIQUE module name. A bare
# `import task` resolves to whichever task.py is first on sys.path / already in
# sys.modules — when harness_search/ (which has its own bin-packing task.py) is
# importable in-process, it silently shadows this one and every instance errors
# on the missing SEQ_LEN, scoring 0.0. Explicit path + unique name is immune to
# both sys.path ordering and the sys.modules cache (the false-green incident).
_task_spec = importlib.util.spec_from_file_location(
    "seqpack_task", Path(__file__).resolve().parent / "task.py")
task = importlib.util.module_from_spec(_task_spec)
_task_spec.loader.exec_module(task)  # noqa: E402  (the LOCAL seqpack task)

TIMEOUT_S = 20


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _load_pack(path):
    spec = importlib.util.spec_from_file_location("candidate_packer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "pack"):
        raise AttributeError("harness defines no pack(docs, seq_len)")
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
            seqs = pack(list(docs), task.SEQ_LEN)
            if task.validate(docs, seqs, task.SEQ_LEN):
                effs.append(task.efficiency(docs, seqs, task.SEQ_LEN))
                valid += 1
            else:
                effs.append(0.0)
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        return {"score": 0.0, "split": split, "valid_fraction": valid / len(insts),
                "n": len(insts), "error": f"run: {type(e).__name__}: {e}"}, ""

    score = sum(effs) / len(effs)
    trace = [f"score={score:.5f} valid_fraction={valid/len(insts):.3f} "
             f"(efficiency = real_tokens/(n_seqs*{task.SEQ_LEN}); 1.0 = no padding; "
             f"(1-score) = wasted FLOPs)"]
    for k, docs in enumerate(insts[:3]):
        seqs = pack(list(docs), task.SEQ_LEN)
        rows = sorted(((task.SEQ_LEN - sum(sq), sq) for sq in seqs if sq), reverse=True)
        trace.append(
            f"instance {k}: {len(rows)} sequences; most-PADDED -> "
            + "; ".join(f"[pad={p} fill={sum(sq)} docs={len(sq)}]" for p, sq in rows[:6]))
    return {"score": round(score, 5), "split": split,
            "valid_fraction": valid / len(insts), "n": len(insts)}, "\n".join(trace)


def main(argv):
    if len(argv) < 2:
        print("usage: evaluate.py <harness.py> [--split search|heldout] [--seed N] [--no-trace]")
        return 2
    path, split, seed, no_trace = argv[1], "search", None, False
    i = 2
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
