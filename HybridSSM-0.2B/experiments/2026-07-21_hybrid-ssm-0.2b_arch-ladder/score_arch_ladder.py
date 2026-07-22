#!/usr/bin/env python3
"""score_arch_ladder.py — the arch-ladder's completion/rung scorer.

Fixes the audit finding that run_arch_ladder.sh's scoring hook was a silent no-op
(this file did not exist). It REUSES the proven JAX suite functions from the build
dir's eval_suite_jax.py (load_model / ppl / load_corpora) — it does NOT re-implement
scoring (the 6-copies-of-score_cohort anti-pattern the audit flagged). It patches the
per-arm config (mixer / attn_every / nope / checkpoint) onto that module per cell.

Metric: val PPL with the model's own Qwen3 tokenizer on the same wikitext-2 + code
corpora the suite pins (suite_version text-lm-v2). Because every cell shares the
tokenizer, corpora, and window settings, PPL is directly COMPARABLE across arms —
this is the valid cross-arm ranking metric. It is NOT stamped as BPB: the audit
flagged that BPB was being stamped on text-lm-v2 which does not define it, and a
cross-STUDY BPB must come from the SINGLE consolidated eval-harness (upgrade-plan
item 6), not a hand-rolled copy here. So this scorer reports PPL cross-arm + the
emergence-speed curve, and defers a BPB number to eval-harness.

Modes:
  --smoke : CPU-only structural self-test of the pure aggregation/curve math (no GPU,
            no model load) — safe to run beside a live trainer.
  (default): GPU — score every .done cell in cells.json. §C4.5: run ONLY when no
            trainer is live (the driver calls this at a rung gap / at ladder end).

GPU-VALIDATION-PENDING: the per-cell scoring path (load_model+ppl) is proven (it scored
the pilot), but this orchestrator's end-to-end run has not yet executed on GPU — it will
at the first rung gap. Until then treat the emitted numbers as unproduced.
"""
from __future__ import annotations
import json
import math
import pathlib
import sys

LDIR = pathlib.Path(__file__).resolve().parent
BUILD = LDIR.parent / "2026-07-19_hybrid-ssm-0.2b_build"
ROOT = LDIR.parents[2]
sys.path.insert(0, str(ROOT / "research"))


def load_cells():
    return json.loads((LDIR / "cells.json").read_text())


def emergence_curve(per_arm_rung_ppl: dict) -> dict:
    """per_arm_rung_ppl[arm] = {tokens: ppl}. Fit ppl vs log10(tokens) per arm (OLS)
    and report the slope (emergence speed) + the arm-minus-base gap at each rung.
    Pure math — the part this file unit-tests on CPU."""
    curves = {}
    base = per_arm_rung_ppl.get("ssm_base", {})
    for arm, pts in per_arm_rung_ppl.items():
        xs = [math.log10(t) for t in sorted(pts)]
        ys = [pts[t] for t in sorted(pts)]
        slope = intercept = None
        if len(xs) >= 2:
            n = len(xs); sx = sum(xs); sy = sum(ys)
            sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
            denom = n * sxx - sx * sx
            if denom != 0:
                slope = (n * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / n
        gap_vs_base = {t: round(pts[t] - base[t], 4) for t in sorted(pts) if t in base}
        curves[arm] = {"points": {str(t): round(pts[t], 4) for t in sorted(pts)},
                       "ppl_vs_log10tok_slope": (round(slope, 4) if slope is not None else None),
                       "intercept": (round(intercept, 4) if intercept is not None else None),
                       "gap_vs_base": gap_vs_base}
    return curves


def _smoke() -> int:
    # synthetic: ssm improves fastest, swa lags — assert the curve math is sane.
    fake = {"ssm_base": {42_000_000: 4.7, 85_000_000: 4.2, 150_000_000: 3.9},
            "swa128":   {48_000_000: 5.7, 96_000_000: 5.3, 170_000_000: 5.0}}
    c = emergence_curve(fake)
    assert c["ssm_base"]["ppl_vs_log10tok_slope"] < 0, "ppl must fall with tokens"
    assert c["swa128"]["ppl_vs_log10tok_slope"] < 0
    assert c["ssm_base"]["gap_vs_base"][42_000_000] == 0.0
    # base has no gap key for swa's token budgets (different rung tokens) -> ok
    assert set(c) == {"ssm_base", "swa128"}
    print("SMOKE PASS: emergence_curve math sane (slopes negative, base gap 0)")
    return 0


def _score_all() -> int:
    """GPU path — score every .done cell. §C4.5: caller guarantees no live trainer."""
    sys.path.insert(0, str(BUILD))
    import importlib
    esj = importlib.import_module("eval_suite_jax")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(esj.TOKENIZER_REPO)

    cells = load_cells()
    per_arm_rung = {}          # arm -> {tokens: {corpus: ppl}}
    scored = []
    for cell in cells["cells"]:
        done = LDIR / f"{cell['id']}.done"
        ckpt = LDIR / f"checkpoint_{cell['id']}.pkl"
        if not done.exists() or not ckpt.exists():
            continue
        # patch the suite module's per-arm config, then reuse its proven loader/scorer
        esj.MIXER, esj.ATTN_EVERY, esj.NOPE = cell["mixer"], cell["attn_every"], cell["nope"]
        esj.TARGET_CKPT = str(ckpt)
        params, forward = esj.load_model()
        corpora, _ = esj.load_corpora(tok)
        row = {"cell": cell["id"], "arm": cell["arm"], "tokens": cell["tokens"], "ppl": {}}
        for name, c in corpora.items():
            v, n = esj.ppl(params, forward, tok, c["text"])
            row["ppl"][name] = round(v, 4)
        scored.append(row)
        per_arm_rung.setdefault(cell["arm"], {}).setdefault(cell["tokens"], row["ppl"])
        print(f"[scored] {cell['id']}: {row['ppl']}", flush=True)

    # per-corpus emergence curve (use the headline corpus present in most rows)
    corpus = "wikitext2_val"
    per_arm_ppl = {arm: {tok_n: cor.get(corpus) for tok_n, cor in rungs.items() if corpus in cor}
                   for arm, rungs in per_arm_rung.items()}
    per_arm_ppl = {a: p for a, p in per_arm_ppl.items() if p}
    out = {"suite_version": "text-lm-v2", "metric": f"val PPL ({corpus}), own tokenizer, n=1/cell",
           "comparability": "cross-arm PPL only (same tokenizer/corpora/windows); "
                            "cross-study BPB deferred to the consolidated eval-harness",
           "cells_scored": len(scored), "rows": scored,
           "emergence_curve": emergence_curve(per_arm_ppl),
           "caveat": "n=1 per cell -> DIRECTIONAL (§C17); mixer-type gaps carry the "
                     "LR-not-retuned-per-arm confound (see rung_42M_comparison.md)."}
    (LDIR / "arch_ladder_scores.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[done] wrote arch_ladder_scores.json ({len(scored)} cells)")
    return 0


def main() -> int:
    if "--smoke" in sys.argv:
        return _smoke()
    return _score_all()


if __name__ == "__main__":
    raise SystemExit(main())
