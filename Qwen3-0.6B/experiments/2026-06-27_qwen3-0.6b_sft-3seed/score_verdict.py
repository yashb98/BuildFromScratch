"""Post-cohort SFT verdict (post-training plan §2A). Reads each cell's FINAL held-out reasoning
PPL from its train log + the (shared) frozen-base baseline, then computes the across-seed 95% CI:
  - SFT vs CONTROL  (does response-masking beat just-more-tokens? the attribution headline)
  - SFT vs BASE     (is there a reasoning gain at all, vs the 25.2% n=1 noise floor)
Uses the unit-tested research/eval_stats.seed_delta_significant (Welch-t). Writes verdict.json + a figure."""
from __future__ import annotations
import re, json, glob, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "research"))
from eval_stats import seed_delta_significant, mean_std_sem  # noqa


def final_ppl(cell):
    """last 'reasoning PPL=' (the final eval at end of training) from the cell's log."""
    for p in (HERE / "results" / f"{cell}.log", HERE / f"run_{cell}.log"):
        if p.exists():
            m = re.findall(r"reasoning PPL=([\d.]+)", p.read_text())
            if m:
                return float(m[-1])
    return None


def base_ppl():
    for p in glob.glob(str(HERE / "results" / "*.log")) + glob.glob(str(HERE / "run_*.log")):
        m = re.findall(r"baseline held-out reasoning PPL=([\d.]+)", pathlib.Path(p).read_text())
        if m:
            return float(m[0])
    return None


def main():
    sft = [final_ppl(f"sft_seed{s}") for s in (0, 1, 2)]
    ctrl = [final_ppl(f"ctrl_seed{s}") for s in (0, 1, 2)]
    base = base_ppl()
    out = {"sft_final_ppl": sft, "ctrl_final_ppl": ctrl, "base_ppl": base}
    if None in sft:
        out["status"] = "incomplete — missing SFT cells"; (HERE / "verdict.json").write_text(json.dumps(out, indent=2)); print(out); return 0
    # lower PPL is better -> improvement = control_mean - sft_mean (positive = SFT better)
    if None not in ctrl:
        r = seed_delta_significant(ctrl, sft, direction="lower_is_better")
        out["sft_vs_control"] = {"improvement_ppl": r["improvement"], "ci95": r["ci95"],
                                 "significant": r["significant"], "warning": r["warning"]}
    if base is not None:
        rb = seed_delta_significant([base, base, base], sft, direction="lower_is_better")
        out["sft_vs_base"] = {"improvement_ppl": rb["improvement"], "ci95": rb["ci95"],
                              "significant": rb["significant"],
                              "note": "base is a single frozen number repeated x3 -> CI reflects SFT seed-variance only (a sanity check, not a true paired test)"}
    sm, _, _ = mean_std_sem(sft)
    out["headline"] = (f"SFT reasoning PPL {sm:.2f} (3 seeds) vs base {base:.2f}"
                       + (f", control {mean_std_sem(ctrl)[0]:.2f}" if None not in ctrl else "")
                       + f"; SFT-vs-control significant={out.get('sft_vs_control',{}).get('significant')}")
    out["overall_verdict"] = ("win — SFT beats the iso-FLOP control (CI excludes 0)"
                              if out.get("sft_vs_control", {}).get("significant") else
                              "inconclusive — SFT gain not separable from just-more-tokens at this budget")
    (HERE / "verdict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # figure
    try:
        sys.path.insert(0, str(ROOT / "research")); import eval_plots as P
        labels = ["base", "control\n(no_mask)", "SFT\n(masked)"]
        vals = [base, mean_std_sem(ctrl)[0] if None not in ctrl else base, sm]
        P.bar_with_ci(labels, vals, title="SFT 3-seed: reasoning PPL vs base & iso-FLOP control",
                      ylabel="held-out reasoning PPL (lower=better)", out=HERE / "plots" / "sft_verdict.png",
                      zero_line=False)
    except Exception as e:
        print("figure skipped:", str(e)[:60])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
