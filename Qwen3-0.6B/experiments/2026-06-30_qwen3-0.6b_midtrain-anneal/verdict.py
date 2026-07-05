#!/usr/bin/env python3
"""Across-seed verdict for the mid-training ANNEAL cohort, from cohort_bpb.json via
the tested eval_stats.seed_delta_significant (Welch-t CI).

Pre-registration (c5_evidence.json + research/midtraining/plan.md §2):
  * HEADLINE = treatment−control BPB on code_py (matched-domain, the only place the
    plan expects signal). Treatment = mix (premium 50/50 cooldown); control =
    fineweb (iso-token FineWeb-Edu-only cooldown). CI must exclude 0 to claim a
    data effect — the control eats the LR-decay/more-tokens gain.
  * wikitext2_val is SECONDARY with a pre-registered expected honest null.
  * Short-context NON-regression (§C25 mid-training row): each anneal arm vs the
    un-annealed base on both corpora; regression = the arm's whole one-sample 95%
    t-CI sits above the base BPB.
  * §C25 gate: effective_context_length_ruler was NOT run on the anneal
    checkpoints (loader is build-backlog) -> verdict capped at `directional`
    regardless of significance. Pre-registered; not a failure.
"""
import json, math, pathlib, statistics, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # BuildFromScratch/
sys.path.insert(0, str(ROOT / "research"))
from eval_stats import seed_delta_significant, mean_std_sem  # noqa: E402
from eval_completeness import gate_verdict  # noqa: E402
from eval_metrics import accuracy_wilson_ci  # noqa: E402

d = json.loads((HERE / "cohort_bpb.json").read_text())
cells = d["cells"]
SEEDS = (0, 1, 2)
HEADLINE_CORPUS = "code_py"          # pre-registered (plan §2); wikitext2 = expected null
CORPORA = ("code_py", "wikitext2_val")
T95_DF2 = 4.302652729911275          # two-sided 95% Student-t, df = n_seeds - 1 = 2

# Constant-LR upper-bound anchor (full continued-train, NOT an anneal — context only).
ANCHOR = json.loads((ROOT / "Qwen3-0.6B" / "experiments" /
                     "2026-06-26_qwen3-0.6b_data-mix-composition" / "verdict.json").read_text())


def arm_vals(arm, corpus, key="bpb"):
    return [cells[f"{arm}_seed{s}"][corpus][key] for s in SEEDS]


out = {"suite_version": d["suite_version"], "headline_corpus": HEADLINE_CORPUS,
       "treatment": "mix", "control": "fineweb", "corpora": {}}

# 1) treatment vs iso-token control, per corpus (the attribution-critical pair)
for corpus in CORPORA:
    base = arm_vals("fineweb", corpus)
    treat = arm_vals("mix", corpus)
    res = seed_delta_significant(base, treat, direction="lower_is_better")
    bm, _, bsem = mean_std_sem(base)
    tm, _, tsem = mean_std_sem(treat)
    out["corpora"][corpus] = {
        "control_bpb": base, "control_mean": bm, "control_sem": bsem,
        "treatment_bpb": treat, "treatment_mean": tm, "treatment_sem": tsem,
        "improvement_bpb": res["improvement"], "ci95": res["ci95"],
        "significant": res["significant"], "warning": res["warning"],
    }

# 2) short-context non-regression vs the un-annealed base (§C25 mid-training row)
out["short_ctx_non_regression"] = {}
for arm in ("mix", "fineweb"):
    for corpus in CORPORA:
        vals = arm_vals(arm, corpus)
        m, _, sem = mean_std_sem(vals)
        base_bpb = cells["base"][corpus]["bpb"]
        lo, hi = m - T95_DF2 * sem, m + T95_DF2 * sem
        regressed = lo > base_bpb        # whole 95% CI worse than base
        out["short_ctx_non_regression"][f"{arm}/{corpus}"] = {
            "base_bpb": base_bpb, "arm_mean": m, "arm_ci95": [lo, hi],
            "delta_vs_base": base_bpb - m,   # positive = anneal improved on base
            "regressed": regressed,
        }
all_non_regressed = not any(v["regressed"] for v in out["short_ctx_non_regression"].values())

# 2b) RULER-style effective-context ladder (run_ecl_ladder.py), when present —
#     supplies the §C25 `effective_context_length_ruler` HARD item. Ladder regression
#     check: at rungs <= trained_len, the treatment arm's POOLED (3-seed) Wilson CI
#     must not sit entirely below the base's CI.
ecl_ok = False
ecl_path = HERE / "ecl_ladder.json"
if ecl_path.exists():
    ecl = json.loads(ecl_path.read_text())
    ec = ecl["cells"]
    n_rung = ecl["n_per_rung"]
    ladder_regressed = []
    arm_summary = {}
    for arm in ("mix", "fineweb"):
        per_rung = {}
        for L in map(str, ecl["lengths"]):
            k = sum(round(ec[f"{arm}_seed{s}"]["per_rung_accuracy"][L] * n_rung)
                    for s in (0, 1, 2))
            acc, lo, hi = accuracy_wilson_ci([1] * k + [0] * (3 * n_rung - k))
            per_rung[L] = {"acc": acc, "wilson95": [lo, hi]}
            if int(L) <= ecl["trained_len"]:
                b_lo, b_hi = ec["base"]["per_rung_wilson95"][L]
                if hi < b_lo:
                    ladder_regressed.append(f"{arm}@{L}")
        arm_summary[arm] = {
            "per_rung_pooled": per_rung,
            "ecl_rel085": [ec[f"{arm}_seed{s}"]["ecl_rel085_shortctx"] for s in (0, 1, 2)],
            "ecl_abs05": [ec[f"{arm}_seed{s}"]["ecl_abs05"] for s in (0, 1, 2)],
        }
    out["ecl_ladder"] = {
        "suite": ecl["suite"], "n_per_rung_per_cell": n_rung,
        "base": {"per_rung_accuracy": ec["base"]["per_rung_accuracy"],
                 "per_rung_wilson95": ec["base"]["per_rung_wilson95"],
                 "ecl_rel085": ec["base"]["ecl_rel085_shortctx"],
                 "ecl_abs05": ec["base"]["ecl_abs05"]},
        "arms": arm_summary,
        "ladder_regressed_cells": ladder_regressed,
    }
    ecl_ok = not ladder_regressed

# 3) significance verdict on the pre-registered headline, then the §C25 cap
hl = out["corpora"][HEADLINE_CORPUS]
if hl["significant"] and hl["improvement_bpb"] > 0:
    sig = "win"
elif hl["significant"] and hl["improvement_bpb"] < 0:
    sig = "loss"
else:
    sig = "inconclusive"
present = ["anneal_gain_vs_iso_token_control"]
if all_non_regressed and (ecl_ok or not ecl_path.exists()):
    present.append("short_ctx_non_regression")
if ecl_path.exists() and ecl_ok:
    present.append("effective_context_length_ruler")
gate = gate_verdict("mid-training", present, sig, conditions={"anneal_claim"})
out["significance_verdict"] = sig
out["c25_gate"] = gate
out["final_verdict"] = gate["verdict"]

# 4) context anchor (constant-LR full continued-train from the data-mix study)
anchor_code = ANCHOR["axes"]["mix"]["code_py"]
out["constant_lr_anchor"] = {
    "source": "Qwen3-0.6B/experiments/2026-06-26_qwen3-0.6b_data-mix-composition/verdict.json",
    "regime": "full continued-train at normal LR (NOT an anneal) — upper-bound reference only",
    "code_py_improvement_bpb": anchor_code["improvement_bpb"],
    "code_py_ci95": anchor_code["ci95"],
}

print("=" * 74)
print("MID-TRAINING ANNEAL — mix (premium 50/50) vs iso-token fineweb control")
print(f"(headline: {HEADLINE_CORPUS}, lower BPB better; wikitext2 = pre-registered expected null)")
print("=" * 74)
for corpus in CORPORA:
    r = out["corpora"][corpus]
    tag = "HEADLINE" if corpus == HEADLINE_CORPUS else "secondary"
    print(f"\n[{corpus}] ({tag})")
    print(f"  control  (fineweb) seeds {[round(x, 4) for x in r['control_bpb']]} "
          f"mean={r['control_mean']:.4f}±{r['control_sem']:.4f}")
    print(f"  treatment(mix)     seeds {[round(x, 4) for x in r['treatment_bpb']]} "
          f"mean={r['treatment_mean']:.4f}±{r['treatment_sem']:.4f}")
    print(f"  Δ(control−treatment) = {r['improvement_bpb']:+.4f} bpb  "
          f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  significant={r['significant']}")
    if r["warning"]:
        print(f"  WARNING: {r['warning']}")
print(f"\nshort-context non-regression vs base (base = no-anneal floor):")
for k, v in out["short_ctx_non_regression"].items():
    print(f"  {k:22s} base={v['base_bpb']:.4f} arm={v['arm_mean']:.4f} "
          f"Δvs_base={v['delta_vs_base']:+.4f}  regressed={v['regressed']}")
if "ecl_ladder" in out:
    el = out["ecl_ladder"]
    print(f"\neffective-context ladder ({el['suite']}, {el['n_per_rung_per_cell']}/rung/cell):")
    base_acc = el["base"]["per_rung_accuracy"]
    print(f"  base   : " + " ".join(f"{L}:{a:.2f}" for L, a in base_acc.items())
          + f"  ECL rel={el['base']['ecl_rel085']} abs={el['base']['ecl_abs05']}")
    for arm, s in el["arms"].items():
        accs = " ".join(f"{L}:{v['acc']:.2f}" for L, v in s["per_rung_pooled"].items())
        print(f"  {arm:7s}: {accs}  ECL rel={s['ecl_rel085']} abs={s['ecl_abs05']}")
    print(f"  ladder regression (arm CI below base CI at rungs <= trained_len): "
          f"{el['ladder_regressed_cells'] or 'none'}")
print(f"\nconstant-LR anchor (context only): code_py {anchor_code['improvement_bpb']:+.4f} "
      f"CI [{anchor_code['ci95'][0]:+.4f}, {anchor_code['ci95'][1]:+.4f}] — different LR regime")
print("\n" + "=" * 74)
print(f"significance: {sig}  |  §C25 gate: {gate['verdict']}  ({gate['why']})")
print("=" * 74)
(HERE / "verdict.json").write_text(json.dumps(out, indent=2))
print(f"-> {HERE / 'verdict.json'}")
