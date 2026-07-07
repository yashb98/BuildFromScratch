#!/usr/bin/env python3
"""fig_ecl_ladder: passkey-retrieval accuracy vs context rung (ECL ladder).

Reads ONLY from the mid-train anneal verdict file; no plotted number is
hardcoded here (annotation constants: trained context length marker,
axis labels).

Source:
  Qwen3-0.6B/experiments/2026-06-30_qwen3-0.6b_midtrain-anneal/verdict.json
    -> key "ecl_ladder"
       base : per_rung_accuracy + per_rung_wilson95           (n=40/rung)
       arms : mix / fineweb per_rung_pooled {acc, wilson95}   (pooled n=120/rung)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = (
    "/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/experiments/"
    "2026-06-30_qwen3-0.6b_midtrain-anneal/verdict.json"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Annotation-only constant (figure spec: mark the trained context length).
TRAINED_LEN = 4096

with open(SRC) as f:
    verdict = json.load(f)
ladder = verdict["ecl_ladder"]

# ---- parse base ------------------------------------------------------------
base_acc_map = ladder["base"]["per_rung_accuracy"]
base_ci_map = ladder["base"]["per_rung_wilson95"]
rungs = sorted(int(r) for r in base_acc_map)
base_acc = [base_acc_map[str(r)] for r in rungs]
base_lo = [base_ci_map[str(r)][0] for r in rungs]
base_hi = [base_ci_map[str(r)][1] for r in rungs]

# ---- parse arms (pooled across seed checkpoints) ---------------------------
def arm_series(name):
    pooled = ladder["arms"][name]["per_rung_pooled"]
    acc = [pooled[str(r)]["acc"] for r in rungs]
    lo = [pooled[str(r)]["wilson95"][0] for r in rungs]
    hi = [pooled[str(r)]["wilson95"][1] for r in rungs]
    return acc, lo, hi

mix_acc, mix_lo, mix_hi = arm_series("mix")
fw_acc, fw_lo, fw_hi = arm_series("fineweb")

# pooled-n and per-cell-n for the legend, derived from the file itself
n_cell = ladder["n_per_rung_per_cell"]
n_seeds = len(ladder["arms"]["mix"]["ecl_rel085"])
n_pooled = n_cell * n_seeds

# ---- style -----------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
    }
)

# Okabe–Ito colorblind-safe palette
C_BASE = "#000000"  # black
C_FW = "#0072B2"    # blue
C_MIX = "#D55E00"   # vermillion

fig, ax = plt.subplots(figsize=(3.4, 2.6))  # single column

series = [
    (f"base (pre-anneal, n={n_cell}/rung)", base_acc, base_lo, base_hi, C_BASE, "o", "--"),
    (f"fineweb anneal (pooled n={n_pooled})", fw_acc, fw_lo, fw_hi, C_FW, "s", "-"),
    (f"mix anneal (pooled n={n_pooled})", mix_acc, mix_lo, mix_hi, C_MIX, "^", "-"),
]
for label, acc, lo, hi, color, marker, ls in series:
    ax.plot(rungs, acc, marker=marker, ms=4, lw=1.4, ls=ls, color=color,
            label=label, zorder=3)
    ax.fill_between(rungs, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)

ax.axvline(TRAINED_LEN, color="0.45", ls=(0, (4, 3)), lw=1.0, zorder=1)
ax.text(TRAINED_LEN, 1.01, "trained len", ha="center", va="bottom",
        fontsize=9, color="0.35")

ax.set_xscale("log", base=2)
ax.set_xticks(rungs)
ax.set_xticklabels([str(r) for r in rungs])
ax.minorticks_off()
ax.set_xlabel("Passkey context length (tokens)")
ax.set_ylabel("Retrieval accuracy")
ax.set_ylim(0, 1.0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", frameon=False, handlelength=1.8,
          borderaxespad=0.0)

fig.tight_layout()
pdf = os.path.join(OUT_DIR, "fig_ecl_ladder.pdf")
png = os.path.join(OUT_DIR, "fig_ecl_ladder.png")
fig.savefig(pdf)
fig.savefig(png, dpi=200)
print("wrote", pdf)
print("wrote", png)
