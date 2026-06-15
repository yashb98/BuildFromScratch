#!/usr/bin/env python3
"""research/ci/eval_gate.py — the eval-regression deploy gate (§C23.4).

Reads an eval-harness suite_results.json and a baseline, and FAILS (exit 1) if a
headline metric has regressed beyond the noise floor. This is the merge/deploy
authority for serving checkpoints: a release is blocked unless it is at least as
good as the baseline, within the same comparability law the loop uses elsewhere
(§C10). Stdlib-only, CPU-only, no model load — runs in CI on any Linux.

It does NOT re-derive the noise floor; it reuses the floor recorded by
eval-harness (or a --floor-abs passed by the caller), so the gate and the
research verdict speak the same language.

Metric direction: perplexity / loss are lower-is-better (a regression is an
INCREASE); accuracy-style metrics are higher-is-better. Pass --higher-better for
the latter.

suite_results.json shape (only the read fields shown):
    {"suite_version": "...", "metrics": {"<name>": {"mean": <f>} | <f>, ...}}
Both a bare number and a {"mean": ...} object are accepted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _metric_value(metrics: dict, name: str):
    if name not in metrics:
        raise KeyError(f"metric {name!r} not in suite_results (have {sorted(metrics)})")
    v = metrics[name]
    if isinstance(v, dict):
        if "mean" not in v:
            raise KeyError(f"metric {name!r} object has no 'mean'")
        return float(v["mean"])
    return float(v)


def check(candidate: dict, baseline: dict, metric: str, floor_abs: float,
          higher_better: bool = False) -> dict:
    """Return a verdict dict; verdict in {pass, regress}. A regression is a move
    in the worse direction by MORE than floor_abs (a within-floor move is noise
    and passes)."""
    cand = _metric_value(candidate.get("metrics", {}), metric)
    base = _metric_value(baseline.get("metrics", {}), metric)
    # delta in the "worse" direction
    worse = (base - cand) if higher_better else (cand - base)
    regressed = worse > floor_abs
    return {
        "metric": metric,
        "candidate": cand,
        "baseline": base,
        "delta": cand - base,
        "floor_abs": floor_abs,
        "higher_better": higher_better,
        "verdict": "regress" if regressed else "pass",
        "suite_version": candidate.get("suite_version"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", type=Path, required=True,
                    help="suite_results.json for the release candidate")
    ap.add_argument("--baseline", type=Path, required=True,
                    help="suite_results.json for the current/known-good baseline")
    ap.add_argument("--metric", required=True, help="headline metric name")
    ap.add_argument("--floor-abs", type=float, required=True,
                    help="noise floor (from eval-harness); a worse move within "
                         "this is noise and passes")
    ap.add_argument("--higher-better", action="store_true",
                    help="metric is higher-is-better (default: lower-is-better, "
                         "e.g. perplexity/loss)")
    a = ap.parse_args(argv)
    try:
        cand = json.loads(a.candidate.read_text())
        base = json.loads(a.baseline.read_text())
        v = check(cand, base, a.metric, a.floor_abs, a.higher_better)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2
    # PR-comment-style summary on stdout
    arrow = "↑" if v["delta"] > 0 else ("↓" if v["delta"] < 0 else "→")
    print(f"[eval-gate] {a.metric}: {v['baseline']:.4f} {arrow} {v['candidate']:.4f} "
          f"(Δ={v['delta']:+.4f}, floor=±{a.floor_abs:.4f}) → {v['verdict'].upper()}")
    print(json.dumps(v, indent=2))
    return 1 if v["verdict"] == "regress" else 0


if __name__ == "__main__":
    sys.exit(main())
