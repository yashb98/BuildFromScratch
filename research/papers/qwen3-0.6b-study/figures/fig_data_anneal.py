#!/usr/bin/env python3
"""fig_data_anneal: two-panel figure.

Panel A -- data-composition sweep (dclm / mix vs fineweb baseline), from
  Qwen3-0.6B/experiments/2026-06-26_qwen3-0.6b_data-mix-composition/verdict.json
Panel B -- mid-training anneal (mix treatment vs fineweb iso-token control), from
  Qwen3-0.6B/experiments/2026-06-30_qwen3-0.6b_midtrain-anneal/verdict.json

Every plotted number is read from the two verdict.json files at run time;
nothing numeric is hardcoded except axis labels / legend text.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/yashb98/Downloads/BuildFromScratch"

SRC_A = os.path.join(
    REPO,
    "Qwen3-0.6B/experiments/2026-06-26_qwen3-0.6b_data-mix-composition/verdict.json",
)
SRC_B = os.path.join(
    REPO,
    "Qwen3-0.6B/experiments/2026-06-30_qwen3-0.6b_midtrain-anneal/verdict.json",
)

with open(SRC_A) as f:
    va = json.load(f)
with open(SRC_B) as f:
    vb = json.load(f)

CORPORA = ["code_py", "wikitext2_val"]
CORPUS_LABEL = {"code_py": "code_py", "wikitext2_val": "wikitext2_val"}

# Okabe-Ito colorblind-safe palette
C_DCLM = "#0072B2"   # blue
C_MIX = "#E69F00"    # orange
C_REF_MIX = "#E69F00"
C_REF_CTRL = "#009E73"  # bluish green
C_ZERO = "#666666"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def asym_err(mean, ci):
    """Return ([lower_err], [upper_err]) for one bar from mean + [lo, hi] CI."""
    lo, hi = ci
    return [mean - lo], [hi - mean]


def sig_mark(ax, x, mean, ci, significant):
    """Star (significant) or 'n.s.' above the CI whisker."""
    top = max(ci[1], mean, 0.0)
    txt = "*" if significant else "n.s."
    ax.annotate(
        txt,
        xy=(x, top),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )


fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)

# ------------------------------------------------------------------ Panel A
axes_names = ["dclm", "mix"]
colors = {"dclm": C_DCLM, "mix": C_MIX}
width = 0.32
for j, axis_name in enumerate(axes_names):
    xs, means, lows, highs, sigs = [], [], [], [], []
    for i, corpus in enumerate(CORPORA):
        cell = va["axes"][axis_name][corpus]
        x = i + (j - 0.5) * width
        xs.append(x)
        means.append(cell["improvement_bpb"])
        lo, hi = asym_err(cell["improvement_bpb"], cell["ci95"])
        lows.append(lo[0])
        highs.append(hi[0])
        sigs.append((x, cell["improvement_bpb"], cell["ci95"], cell["significant"]))
    axA.bar(
        xs,
        means,
        width=width,
        color=colors[axis_name],
        yerr=[lows, highs],
        capsize=3,
        error_kw={"lw": 1.0},
        label=axis_name,
        zorder=3,
    )
    for s in sigs:
        sig_mark(axA, *s)

axA.axhline(0.0, color=C_ZERO, lw=0.8, zorder=1)
axA.set_xticks(range(len(CORPORA)))
axA.set_xticklabels([CORPUS_LABEL[c] for c in CORPORA])
axA.set_ylabel(r"$\Delta$BPB vs fineweb baseline (bits/byte)")
axA.legend(title=None, frameon=False, loc="upper right")
axA.text(
    0.02, 0.98, "A", transform=axA.transAxes, fontsize=11, fontweight="bold", va="top"
)

# ------------------------------------------------------------------ Panel B
treatment_name = vb["treatment"]   # e.g. "mix"
control_name = vb["control"]       # e.g. "fineweb"

xs, means, lows, highs, sigs = [], [], [], [], []
for i, corpus in enumerate(CORPORA):
    cell = vb["corpora"][corpus]
    xs.append(i)
    means.append(cell["improvement_bpb"])
    lo, hi = asym_err(cell["improvement_bpb"], cell["ci95"])
    lows.append(lo[0])
    highs.append(hi[0])
    sigs.append((i, cell["improvement_bpb"], cell["ci95"], cell["significant"]))

axB.bar(
    xs,
    means,
    width=0.45,
    color=C_MIX,
    yerr=[lows, highs],
    capsize=3,
    error_kw={"lw": 1.0},
    label=f"{treatment_name} anneal vs {control_name} control",
    zorder=3,
)
for s in sigs:
    sig_mark(axB, *s)

# Light reference marks: delta_vs_base from short_ctx_non_regression, both arms.
ref = vb["short_ctx_non_regression"]
ref_off = 0.33
for i, corpus in enumerate(CORPORA):
    d_treat = ref[f"{treatment_name}/{corpus}"]["delta_vs_base"]
    d_ctrl = ref[f"{control_name}/{corpus}"]["delta_vs_base"]
    axB.plot(
        [i + ref_off],
        [d_treat],
        marker="_",
        ms=11,
        mew=1.6,
        color=C_REF_MIX,
        alpha=0.55,
        ls="none",
        zorder=4,
        label=f"{treatment_name} $\\Delta$ vs pre-anneal base" if i == 0 else None,
    )
    axB.plot(
        [i + ref_off],
        [d_ctrl],
        marker="_",
        ms=11,
        mew=1.6,
        color=C_REF_CTRL,
        alpha=0.55,
        ls="none",
        zorder=4,
        label=f"{control_name} $\\Delta$ vs pre-anneal base" if i == 0 else None,
    )

axB.axhline(0.0, color=C_ZERO, lw=0.8, zorder=1)
axB.set_xticks(range(len(CORPORA)))
axB.set_xticklabels([CORPUS_LABEL[c] for c in CORPORA])
axB.legend(frameon=False, loc="upper right")
axB.text(
    0.02, 0.98, "B", transform=axB.transAxes, fontsize=11, fontweight="bold", va="top"
)

fig.tight_layout()
out_pdf = os.path.join(HERE, "fig_data_anneal.pdf")
out_png = os.path.join(HERE, "fig_data_anneal.png")
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=200)
print("wrote", out_pdf)
print("wrote", out_png)
