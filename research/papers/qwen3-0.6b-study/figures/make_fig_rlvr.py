#!/usr/bin/env python3
"""fig_rlvr: two-panel RLVR figure for the Qwen3-0.6B study.

Panel A: pass@1 per arm (sft/grpo/random/rft) on gsm8k + math500_l13 with
         Wilson 95% CIs where available (grpo, sft); pass@8 as hollow markers.
         Source: .../2026-07-02_qwen3-0.6b_grpo-phase2/verdict.json (comparison[]).
Panel B: GRPO mean_reward per step (light) + 20-step rolling mean (bold).
         Source: .../2026-07-02_qwen3-0.6b_grpo-phase2/health_grpo_seed0.jsonl.

All plotted numbers are read from the two source files; nothing is hardcoded
except axis labels and styling.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

EXP_DIR = (
    "/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/experiments/"
    "2026-07-02_qwen3-0.6b_grpo-phase2"
)
VERDICT_PATH = os.path.join(EXP_DIR, "verdict.json")
HEALTH_PATH = os.path.join(EXP_DIR, "health_grpo_seed0.jsonl")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- load data
with open(VERDICT_PATH) as f:
    verdict = json.load(f)
comparison = verdict["comparison"]

steps, rewards = [], []
with open(HEALTH_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        steps.append(rec["step"])
        rewards.append(rec["mean_reward"])
steps = np.asarray(steps, dtype=float)
rewards = np.asarray(rewards, dtype=float)
order = np.argsort(steps)
steps, rewards = steps[order], rewards[order]

# ------------------------------------------------------------------- style
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)

# Okabe-Ito colorblind-safe palette
ARMS = ["sft", "grpo", "random", "rft"]
ARM_LABELS = {"sft": "SFT", "grpo": "GRPO", "random": "Random-reward", "rft": "RFT"}
ARM_COLORS = {
    "sft": "#0072B2",  # blue
    "grpo": "#D55E00",  # vermillion
    "random": "#009E73", # bluish green
    "rft": "#CC79A7",  # reddish purple
}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 2.7))

# ------------------------------------------------------------------ Panel A
group_centers = np.arange(len(comparison), dtype=float)
offsets = np.linspace(-0.27, 0.27, len(ARMS))
set_labels = []

for gi, row in enumerate(comparison):
    set_labels.append(row["set"])
    for ai, arm in enumerate(ARMS):
        x = group_centers[gi] + offsets[ai]
        c = ARM_COLORS[arm]
        p1 = row[f"{arm}_pass1"]
        p8 = row[f"{arm}_pass8"]
        ci_key = f"{arm}_pass1_ci"
        if ci_key in row:  # Wilson CI available (grpo, sft only)
            lo, hi = row[ci_key]
            yerr = np.array([[p1 - lo], [hi - p1]])
            axA.errorbar(
                x, p1, yerr=yerr, fmt="o", color=c, ms=5,
                capsize=2.5, elinewidth=1.0, capthick=1.0, zorder=3,
            )
        else:
            axA.plot(x, p1, "o", color=c, ms=5, zorder=3)
        # pass@8 as hollow marker
        axA.plot(
            x, p8, "s", mfc="none", mec=c, mew=1.2, ms=5.5, zorder=3,
        )

axA.set_xticks(group_centers)
axA.set_xticklabels(set_labels)
axA.set_ylabel("accuracy")
axA.set_xlim(-0.6, len(comparison) - 0.4)
axA.set_ylim(bottom=0)
axA.yaxis.grid(True, lw=0.4, alpha=0.4)
axA.set_axisbelow(True)

arm_handles = [
    Line2D([], [], marker="o", ls="none", color=ARM_COLORS[a], ms=5,
           label=ARM_LABELS[a])
    for a in ARMS
]
shape_handles = [
    Line2D([], [], marker="o", ls="none", color="0.3", ms=5, label="pass@1"),
    Line2D([], [], marker="s", ls="none", mfc="none", mec="0.3", mew=1.2,
           ms=5.5, label="pass@8"),
]
leg1 = axA.legend(handles=arm_handles, loc="upper left", frameon=False,
                  handletextpad=0.3, borderaxespad=0.2)
axA.add_artist(leg1)
axA.legend(handles=shape_handles, loc="upper left", frameon=False,
           bbox_to_anchor=(0.42, 1.0), handletextpad=0.3, borderaxespad=0.2)
axA.text(0.02, -0.28, "(a)", transform=axA.transAxes, fontsize=9,
         fontweight="bold")

# ------------------------------------------------------------------ Panel B
axB.plot(steps, rewards, color="#D55E00", lw=0.7, alpha=0.35,
         label="per step")
W = 20
if len(rewards) >= W:
    roll = np.convolve(rewards, np.ones(W) / W, mode="valid")
    roll_x = steps[W - 1:]
    axB.plot(roll_x, roll, color="#D55E00", lw=1.8,
             label=f"{W}-step rolling mean")

axB.set_xlabel("GRPO step")
axB.set_ylabel("mean reward")
axB.set_ylim(0, 0.06)
axB.set_xlim(steps.min(), steps.max())
axB.yaxis.grid(True, lw=0.4, alpha=0.4)
axB.set_axisbelow(True)
axB.legend(loc="upper right", frameon=False, handletextpad=0.5,
           borderaxespad=0.2)

# flat-trend annotation, computed from the data itself
first50 = rewards[:50].mean()
last50 = rewards[-50:].mean()
overall = rewards.mean()
axB.annotate(
    (f"flat: mean {overall:.4f}\n"
     f"(first-50 {first50:.4f} vs last-50 {last50:.4f})"),
    xy=(float(np.median(steps)), overall),
    xytext=(0.30, 0.62), textcoords="axes fraction",
    fontsize=9, ha="left", va="center",
    arrowprops=dict(arrowstyle="-", lw=0.7, color="0.4"),
)
axB.text(0.02, -0.28, "(b)", transform=axB.transAxes, fontsize=9,
         fontweight="bold")

fig.tight_layout()
pdf_path = os.path.join(OUT_DIR, "fig_rlvr.pdf")
png_path = os.path.join(OUT_DIR, "fig_rlvr.png")
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=200)
print(f"wrote {pdf_path}")
print(f"wrote {png_path}")
print(f"panelB: n_steps={len(steps)} overall={overall:.4f} "
      f"first50={first50:.4f} last50={last50:.4f}")
