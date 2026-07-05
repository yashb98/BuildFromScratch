#!/usr/bin/env python3
"""Compute the across-seed verdict for the NorMuon-vs-AdamW ablation from
cohort_bpb.json, using the tested eval_stats.seed_delta_significant (Welch-t CI).
NorMuon is the TREATMENT, AdamW the BASELINE; lower BPB is better."""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research"))
from eval_stats import seed_delta_significant, mean_std_sem  # noqa: E402

RES = pathlib.Path(__file__).resolve().parent / "results"
d = json.loads((RES / "cohort_bpb.json").read_text())
cells = d["cells"]
SEEDS = (0, 1, 2)


def arm_vals(arm, corpus, key):
    return [cells[f"{arm}_seed{s}"][corpus][key] for s in SEEDS]


out = {"suite_version": d["suite_version"], "by_corpus": {}}
for corpus in ("wikitext2_val", "code_py"):
    adamw = arm_vals("adamw", corpus, "bpb")
    normuon = arm_vals("normuon", corpus, "bpb")
    res = seed_delta_significant(adamw, normuon, direction="lower_is_better")
    am, asd, asem = mean_std_sem(adamw)
    nm, nsd, nsem = mean_std_sem(normuon)
    out["by_corpus"][corpus] = {
        "adamw_bpb": adamw, "adamw_mean": am, "adamw_sem": asem,
        "normuon_bpb": normuon, "normuon_mean": nm, "normuon_sem": nsem,
        "improvement_bpb": res["improvement"], "ci95": res["ci95"],
        "significant": res["significant"], "warning": res["warning"],
        "df": res["df"], "n": [res["n_baseline"], res["n_treatment"]],
    }

# headline verdict on wikitext BPB (the cross-tokenizer-comparable metric)
hl = out["by_corpus"]["wikitext2_val"]
if hl["significant"] and hl["improvement_bpb"] > 0:
    verdict = "win"          # NorMuon significantly lower BPB; CI disjoint from 0
elif hl["significant"] and hl["improvement_bpb"] < 0:
    verdict = "loss"
else:
    verdict = "inconclusive"
out["headline_corpus"] = "wikitext2_val"
out["verdict"] = verdict

print("=" * 70)
for corpus, r in out["by_corpus"].items():
    print(f"\n[{corpus}]  (lower BPB = better)")
    print(f"  AdamW   seeds: {[round(x,4) for x in r['adamw_bpb']]}  "
          f"mean={r['adamw_mean']:.4f} ±{r['adamw_sem']:.4f}")
    print(f"  NorMuon seeds: {[round(x,4) for x in r['normuon_bpb']]}  "
          f"mean={r['normuon_mean']:.4f} ±{r['normuon_sem']:.4f}")
    print(f"  improvement (AdamW-NorMuon) = {r['improvement_bpb']:+.4f} bpb  "
          f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
          f"significant={r['significant']}")
    if r["warning"]:
        print(f"  WARNING: {r['warning']}")
print("\n" + "=" * 70)
print(f"HEADLINE VERDICT (wikitext BPB, across-seed CI): {verdict.upper()}")
print("=" * 70)
(RES / "verdict.json").write_text(json.dumps(out, indent=2))
