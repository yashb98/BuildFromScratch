#!/usr/bin/env python3
"""fig_sft_confound: two-panel figure of the SFT eval-token confound.

Panel A -- the in-loop ADVISORY comparison from
  Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/verdict.json
  (sft_final_ppl vs ctrl_final_ppl per seed). The two arms are scored on
  DIFFERENT token sets (SFT: response tokens only; control: all tokens),
  so the apparent 0.68-PPL gap is not a comparison at all.

Panel B -- the CORRECTED comparison from reasoning_verdict.json in the same
  directory: both arms re-scored on ONE fixed held-out response-masked set
  (reasoning_ppl_masked), with the base checkpoint as a reference line.
  The gap collapses to ~0.009 PPL.

Every plotted number is read from the two JSON files at run time; nothing
numeric is hardcoded except axis labels / annotation wording.
CPU-only: matplotlib + json, no torch, no model loads.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/yashb98/Downloads/BuildFromScratch"
EXP = os.path.join(REPO, "Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed")

SRC_A = os.path.join(EXP, "verdict.json")
SRC_B = os.path.join(EXP, "reasoning_verdict.json")

with open(SRC_A) as f:
    va = json.load(f)
with open(SRC_B) as f:
    vb = json.load(f)

# Okabe-Ito colorblind-safe palette (house style)
C_SFT = "#0072B2"    # blue
C_CTRL = "#E69F00"   # orange
C_BASE = "#009E73"   # bluish green (reference line)
C_ZERO = "#666666"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)

# ------------------------------------------------------------------ data
# Panel A: in-loop advisory numbers (per seed).
sft_a = va["sft_final_ppl"]
ctrl_a = va["ctrl_final_ppl"]
gap_a = va["sft_vs_control"]["improvement_ppl"]
assert "CONFOUND" in va["sft_vs_control"], "expected the advisory confound note"

# Panel B: corrected numbers on the fixed held-out response-masked set.
seeds = [0, 1, 2]
metric = vb["headline_metric"]           # "reasoning_ppl_masked"
sft_b = [vb["arms"][f"sft_seed{s}"][metric] for s in seeds]
ctrl_b = [vb["arms"][f"ctrl_seed{s}"][metric] for s in seeds]
base_b = vb["arms"]["base"][metric]
gap_b = vb["sft_vs_control"]["improvement_ppl"]

mean = lambda xs: sum(xs) / len(xs)

# ------------------------------------------------------------------ layout
# Left: panel A (full height). Right: panel B with a broken y-axis so the
# base reference line (~14.1) and the ~0.01-wide sft/ctrl band both show.
fig = plt.figure(figsize=(6.5, 2.9))
gs = gridspec.GridSpec(
    2, 2, width_ratios=[1, 1], height_ratios=[1, 3], hspace=0.12, wspace=0.42,
    left=0.10, right=0.98, bottom=0.17, top=0.94,
)
axA = fig.add_subplot(gs[:, 0])
axB_top = fig.add_subplot(gs[0, 1])
axB_bot = fig.add_subplot(gs[1, 1])


def strip(ax, ys_sft, ys_ctrl, label=True):
    """Per-seed points for the two arms, plus a short dashed mean line each."""
    ax.plot(
        seeds, ys_sft, "o", ms=5, color=C_SFT, ls="none", zorder=3,
        label="SFT (masked loss)" if label else None,
    )
    ax.plot(
        seeds, ys_ctrl, "s", ms=5, color=C_CTRL, ls="none", zorder=3,
        label="control (no mask)" if label else None,
    )
    for ys, c in ((ys_sft, C_SFT), (ys_ctrl, C_CTRL)):
        ax.hlines(mean(ys), -0.35, 2.35, color=c, lw=0.9, ls="--", alpha=0.6, zorder=2)


def gap_arrow(ax, x, y_lo, y_hi, text, dx_text=0.08):
    ax.annotate(
        "", xy=(x, y_hi), xytext=(x, y_lo),
        arrowprops=dict(arrowstyle="<->", color="black", lw=0.9),
    )
    ax.text(x + dx_text, (y_lo + y_hi) / 2.0, text, fontsize=8, va="center", ha="left")


# ------------------------------------------------------------------ Panel A
strip(axA, sft_a, ctrl_a)
gap_arrow(axA, 2.55, mean(sft_a), mean(ctrl_a), "gap\n%.2f" % gap_a)
axA.set_xlim(-0.5, 3.1)
axA.set_xticks(seeds)
axA.set_xlabel("seed")
axA.set_ylabel("perplexity (in-loop eval)")
axA.text(
    0.03, 0.62,
    "SFT scored on response tokens only,\ncontrol on ALL tokens:\nNOT comparable (advisory)",
    transform=axA.transAxes, fontsize=7.5, va="top", ha="left", style="italic",
)
axA.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.78))
axA.text(
    0.02, 0.98, "A", transform=axA.transAxes, fontsize=11, fontweight="bold", va="top"
)

# ------------------------------------------------------------------ Panel B
# Top segment: base reference line only.
axB_top.axhline(base_b, color=C_BASE, lw=1.2, ls="-", zorder=2)
axB_top.text(
    2.35, base_b, "base %.3f" % base_b, fontsize=8, color=C_BASE,
    va="bottom", ha="right",
)
pad_top = 0.12
axB_top.set_ylim(base_b - pad_top, base_b + pad_top)
axB_top.set_yticks([round(base_b, 1)])
axB_top.set_xlim(-0.5, 3.1)
axB_top.set_xticks([])
axB_top.spines["bottom"].set_visible(False)
axB_top.text(
    0.02, 0.95, "B", transform=axB_top.transAxes, fontsize=11,
    fontweight="bold", va="top",
)

# Bottom segment: the two arms, per seed, on the fixed held-out masked set.
strip(axB_bot, sft_b, ctrl_b, label=False)
gap_arrow(axB_bot, 2.55, mean(sft_b), mean(ctrl_b), "gap\n%.3f" % gap_b)
lo = min(sft_b + ctrl_b)
hi = max(sft_b + ctrl_b)
pad_bot = 0.35 * (hi - lo)
axB_bot.set_ylim(lo - pad_bot, hi + pad_bot)
axB_bot.set_xlim(-0.5, 3.1)
axB_bot.set_xticks(seeds)
axB_bot.set_xlabel("seed")
axB_bot.set_ylabel("PPL (fixed held-out, masked)")
axB_bot.yaxis.set_label_coords(-0.26, 0.66)
axB_bot.text(
    0.03, 0.10, "same eval for both arms", transform=axB_bot.transAxes,
    fontsize=7.5, va="bottom", ha="left", style="italic",
)

# Axis-break marks between the two right-hand segments.
d = 0.5
break_kw = dict(
    marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
    color="k", mec="k", mew=1, clip_on=False,
)
axB_top.plot([0], [0], transform=axB_top.transAxes, **break_kw)
axB_bot.plot([0], [1], transform=axB_bot.transAxes, **break_kw)

out_pdf = os.path.join(HERE, "fig_sft_confound.pdf")
out_png = os.path.join(HERE, "fig_sft_confound.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print("wrote", out_pdf)
print("wrote", out_png)
