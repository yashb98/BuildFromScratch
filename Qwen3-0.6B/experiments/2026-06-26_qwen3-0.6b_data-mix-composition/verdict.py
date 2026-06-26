#!/usr/bin/env python3
"""Per-axis across-seed verdict for the IMU-1 de-confounding ladder, from
cohort_bpb.json via the tested eval_stats.seed_delta_significant (Welch-t CI).

Each axis (WSD / z-loss / arch) is the TREATMENT vs the faithful baseline (control);
lower BPB is better. An axis is a DRIVER iff its 95% CI excludes 0 in the improving
direction. The headline is which single component(s) actually move the metric — the
attribution the paper's Limitations name as the missing experiment.
"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research"))
from eval_stats import seed_delta_significant, mean_std_sem  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
d = json.loads((HERE / "cohort_bpb.json").read_text())
cells = d["cells"]
SEEDS = (0, 1, 2)
HEADLINE_CORPUS = "wikitext2_val"   # the cross-tokenizer-comparable metric


def arm_vals(arm, corpus, key="bpb"):
    return [cells[f"{arm}_seed{s}"][corpus][key] for s in SEEDS]


out = {"suite_version": d["suite_version"], "headline_corpus": HEADLINE_CORPUS, "axes": {}}
drivers = []
for axis in ("dclm", "mix"):
    out["axes"][axis] = {}
    for corpus in ("wikitext2_val", "code_py"):
        base = arm_vals("fineweb", corpus)
        treat = arm_vals(axis, corpus)
        res = seed_delta_significant(base, treat, direction="lower_is_better")
        bm, _, bsem = mean_std_sem(base)
        tm, _, tsem = mean_std_sem(treat)
        out["axes"][axis][corpus] = {
            "baseline_bpb": base, "baseline_mean": bm, "baseline_sem": bsem,
            "treatment_bpb": treat, "treatment_mean": tm, "treatment_sem": tsem,
            "improvement_bpb": res["improvement"], "ci95": res["ci95"],
            "significant": res["significant"], "warning": res["warning"],
        }
    hl = out["axes"][axis][HEADLINE_CORPUS]
    if hl["significant"] and hl["improvement_bpb"] > 0:
        drivers.append(axis)

out["drivers"] = drivers
out["overall_verdict"] = (
    "attributed" if drivers else "recipe-level (no single axis clears the seed-noise floor)")

print("=" * 74)
print(f"IMU-1 DE-CONFOUND — per-axis across-seed verdict (headline: {HEADLINE_CORPUS}, lower BPB better)")
print("=" * 74)
for axis, by in out["axes"].items():
    r = by[HEADLINE_CORPUS]
    flag = "  <-- DRIVER" if axis in drivers else ""
    print(f"\n[+{axis}] baseline seeds {[round(x,4) for x in r['baseline_bpb']]} "
          f"mean={r['baseline_mean']:.4f}±{r['baseline_sem']:.4f}")
    print(f"        +{axis}   seeds {[round(x,4) for x in r['treatment_bpb']]} "
          f"mean={r['treatment_mean']:.4f}±{r['treatment_sem']:.4f}")
    print(f"        Δ(baseline−+{axis}) = {r['improvement_bpb']:+.4f} bpb  "
          f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
          f"significant={r['significant']}{flag}")
    if r["warning"]:
        print(f"        WARNING: {r['warning']}")
print("\n" + "=" * 74)
print(f"DRIVERS: {drivers or 'none'}  ->  {out['overall_verdict']}")
print("=" * 74)
(HERE / "verdict.json").write_text(json.dumps(out, indent=2))
