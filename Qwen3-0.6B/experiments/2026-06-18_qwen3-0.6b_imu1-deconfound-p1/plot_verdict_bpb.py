#!/usr/bin/env python3
"""Canonical per-axis BPB attribution plot for the IMU-1 de-confound (Phase 1).

CPU only, no GPU, no torch. Reads ONLY verdict.json (the eval-harness BPB verdict
written by verdict.py after all 12 cells finished) — NOT the in-loop proxy PPL.
Every bar is the held-out BPB improvement (baseline - treatment, higher = better)
with its across-seed 95% CI; a bar is called a DRIVER only if its CI excludes 0
(verdict.json `significant`). Title cites the source + suite_version. Re-run any time.

Output (this dir):
  deconfound_bpb_verdict.png/.pdf  — per-axis BPB improvement + 95% CI, wikitext2 & code
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
V = json.loads((HERE / "verdict.json").read_text())

AXES = ["wsd", "zloss", "arch"]
LABEL = {"wsd": "+WSD\n(schedule)", "zloss": "+z-loss", "arch": "+arch\n(value-resid /\nLN-scale / head-gate)"}
CORPORA = [("wikitext2_val", "WikiText-2 (headline)"), ("code_py", "code (codeparrot)")]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linestyle": "--"})

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
for ax, (corpus, ctitle) in zip(axs, CORPORA):
    for i, axis in enumerate(AXES):
        d = V["axes"][axis][corpus]
        imp = d["improvement_bpb"]
        lo, hi = d["ci95"]
        sig = d["significant"]
        color = "#2ca02c" if sig else "#999999"
        yerr = [[imp - lo], [hi - imp]]
        ax.bar(i, imp, width=0.6, color=color, alpha=0.65,
               edgecolor="k", lw=0.6, zorder=2)
        ax.errorbar(i, imp, yerr=yerr, fmt="none", ecolor="k",
                    capsize=5, lw=1.3, zorder=3)
        tag = "SIGNIFICANT" if sig else "not sig."
        ax.annotate(f"{imp:+.3f}\nCI[{lo:+.3f},{hi:+.3f}]\n{tag}",
                    (i, hi), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.5,
                    color=("#1a7a1a" if sig else "#666"))
    ax.axhline(0, color="#444", lw=1)
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels([LABEL[a] for a in AXES], fontsize=8)
    ax.set_ylabel("BPB improvement vs faithful baseline\n(baseline − treatment; higher = better)")
    ax.set_title(ctitle, fontsize=10)
    ax.margins(y=0.22)

fig.suptitle(
    "IMU-1 de-confound — canonical per-axis attribution (eval-harness BPB, 3 seeds/arm, 95% CI)\n"
    f"only the architecture axis is significant  →  drivers={V['drivers']}, verdict='{V['overall_verdict']}'   "
    f"(source: verdict.json, suite_version={V['suite_version']})",
    fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"deconfound_bpb_verdict.{ext}", dpi=200)
plt.close(fig)

# echo the exact plotted numbers for cross-checking
print(f"source: {HERE/'verdict.json'}  suite_version={V['suite_version']}  drivers={V['drivers']}  verdict={V['overall_verdict']}")
for axis in AXES:
    for corpus, _ in CORPORA:
        d = V["axes"][axis][corpus]
        print(f"  {axis:7s} {corpus:14s} Δbpb={d['improvement_bpb']:+.4f}  CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]  significant={d['significant']}")
print("wrote: deconfound_bpb_verdict.png/.pdf")
