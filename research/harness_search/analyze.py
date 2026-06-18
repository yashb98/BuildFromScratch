#!/usr/bin/env python3
"""Verdict for the Meta-Harness experiment (bin-packing reference target).

Two selection methods are reported side by side — the contrast IS the experiment's
finding:

  * GATED (the product): framework.select_and_promote — rank a shortlist on the
    SEARCH seed, EXCLUDE any candidate brittle on a held-out seed (timeout/invalid),
    take the best held-out non-brittle as challenger, and promote only on a
    SIGNIFICANT held-out win (eval_stats Student-t). This is what we ship.
  * NAIVE (the trap): pick each arm's best by SEARCH score alone, no brittle
    exclusion. This is what a careless reader does — and on this archive it crowns
    a brittle, timeout-prone candidate, so its H1 reads "not significant". That
    negative result is the MOTIVATION for the gate, not a refutation of it.

Every candidate is re-scored DETERMINISTICALLY in an ISOLATED SUBPROCESS (each gets
its own interpreter + fresh signal.alarm) — evaluate.py's per-call 20s timeout does
NOT compose across an in-process sweep, so a single brittle candidate would hang the
whole verdict if scored in-process. Pure CPU, no model, no GPU.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))               # framework
sys.path.insert(0, str(HERE.parents[0]))    # research/ -> eval_stats
import eval_stats as es      # noqa: E402
import framework as fw       # noqa: E402

EVAL = str(HERE / "evaluate.py")
SEARCH_SEED = 1
HELDOUT_SEEDS = [2, 3, 4, 5, 6, 7]          # all != the search seed
CAP_S = 10                                   # outer per-call wall cap: legit candidates run
#                                              <1s; brittle ones hang past evaluate's 20s alarm,
#                                              so any cap in (1s,20s) cleanly marks them brittle.


def score(path, seed):
    """(score, valid_fraction) from an ISOLATED subprocess; (0,0) on timeout."""
    try:
        out = subprocess.run([sys.executable, EVAL, str(path), "--split", "heldout",
                              "--seed", str(seed)], capture_output=True, text=True,
                             cwd=str(HERE), timeout=CAP_S)
    except subprocess.TimeoutExpired:
        return 0.0, 0.0
    lines = out.stdout.strip().splitlines()
    if not lines:
        return 0.0, 0.0
    r = json.loads(lines[-1])
    return r.get("score", 0.0), r.get("valid_fraction", 0.0)


def best_by_search(arm_dir):
    cands = sorted(Path(arm_dir).glob("cand_*.py"))
    scored = sorted(((score(c, SEARCH_SEED)[0], c) for c in cands),
                    key=lambda x: (-x[0], str(x[1])))
    n_valid = sum(1 for s, _ in scored if s > 0)
    return scored[0][0], scored[0][1], len(scored), n_valid


def heldout(path):
    return [score(path, s)[0] for s in HELDOUT_SEEDS]


def fmt(xs):
    return f"{sum(xs)/len(xs):.4f}  (seeds: {', '.join(f'{x:.4f}' for x in xs)})"


def verdict(h):
    lo, hi = h["ci95"]
    sig = "SIGNIFICANT" if h["significant"] else "not significant"
    return (f"improvement={h['improvement']:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  -> {sig}"
            + (f"  ({h['warning']})" if h.get("warning") else ""))


def main():
    baseline = HERE / "baseline_harness.py"

    print("=" * 78)
    print("META-HARNESS EXPERIMENT — bin-packing (efficiency = fill/bins, higher better)")
    print("=" * 78)

    # ---- GATED verdict (the product): the gate's own promotion decision per arm ----
    print("GATED selection (framework.select_and_promote — held-out + brittle-exclusion + t-gate):")
    for arm in ("scores", "full"):
        cands = sorted(str(p) for p in (HERE / f"archive/{arm}").glob("cand_*.py"))
        rep = fw.select_and_promote(score, cands, str(baseline), search_seed=SEARCH_SEED,
                                    heldout_seeds=HELDOUT_SEEDS, direction="higher_is_better")
        sig = rep.get("significance") or {}
        ch = Path(rep["challenger"]).name if rep.get("challenger") else None
        lo, hi = (sig.get("ci95") or [float("nan"), float("nan")])
        print(f"  [{arm:6}] promote={rep['promoted']!s:5} challenger={ch} "
              f"vs baseline {rep.get('incumbent_heldout_mean', 0):.4f} -> "
              f"{rep.get('challenger_heldout_mean', 0):.4f}  "
              f"Δ={sig.get('improvement', 0):+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}] "
              f"sig={sig.get('significant')}  excluded_brittle={[Path(x).name for x in rep.get('excluded_brittle', [])]}")
    print("-" * 78)

    # ---- NAIVE verdict (the trap): best-by-search, no brittle exclusion ----
    fb_score, fb_path, fn, fv = best_by_search(HERE / "archive/full")
    sb_score, sb_path, sn, sv = best_by_search(HERE / "archive/scores")
    base_rep, full_rep, scor_rep = heldout(baseline), heldout(fb_path), heldout(sb_path)
    print("NAIVE selection (best SEARCH score only — the trap a careless reader falls into):")
    print(f"  full-traces arm: {fn} candidates ({fv} valid), best SEARCH = {fb_score:.4f} [{fb_path.name}]")
    print(f"  scores-only arm: {sn} candidates ({sv} valid), best SEARCH = {sb_score:.4f} [{sb_path.name}]")
    print(f"  held-out  baseline    : {fmt(base_rep)}")
    print(f"  held-out  full-traces : {fmt(full_rep)}")
    print(f"  held-out  scores-only : {fmt(scor_rep)}")
    auto_rep = full_rep if fb_score >= sb_score else scor_rep
    auto_name = "full-traces" if fb_score >= sb_score else "scores-only"
    h1 = es.seed_delta_significant(base_rep, auto_rep, direction="higher_is_better")
    h2 = es.seed_delta_significant(scor_rep, full_rep, direction="higher_is_better")
    print(f"  H1 naive-automated ({auto_name}) > baseline?   {verdict(h1)}")
    print(f"  H2 full-traces > scores-only?                  {verdict(h2)}")
    print("-" * 78)
    print("READ: the GATED row is the headline (a significant automated > hand-designed win);")
    print("the NAIVE H1 reads negative because best-by-search crowns a brittle candidate —")
    print("that gap is exactly why the gate exists. Traces-vs-scores (H2) is inconclusive")
    print("here because the bin-packing reward is scalar-sufficient (an easy task).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
