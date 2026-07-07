#!/usr/bin/env python3
"""fig_deconfound: two-panel horizontal bar chart of the IMU-1 de-confound.

Panel A: wsd / zloss / arch axes from the 2026-06-18 imu1-deconfound-p1 verdict.
Panel B: value-residual / layernorm-scaling / head-gating from the 2026-06-21
arch-subdrill-p2 verdict.

All plotted numbers are parsed from the two verdict.json files below; nothing
is hardcoded except axis labels. Output: fig_deconfound.pdf (+ .png preview).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/yashb98/Downloads/BuildFromScratch"

SRC_A = os.path.join(
    REPO, "Qwen3-0.6B/experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1/verdict.json"
)
SRC_B = os.path.join(
    REPO, "Qwen3-0.6B/experiments/2026-06-21_qwen3-0.6b_arch-subdrill-p2/verdict.json"
)

CORPUS = "wikitext2_val"

# Okabe-Ito colorblind-safe palette
COLOR_SIG = "#0072B2"   # blue: significant
COLOR_NS = "#BBBBBB"    # grey: non-significant (also hatched)

PANEL_A_AXES = [("wsd", "wsd"), ("zloss", "zloss"), ("arch", "arch")]
PANEL_B_AXES = [
    ("vr", "value-residual"),
    ("ln", "layernorm-scaling"),
    ("hg", "head-gating"),
]


def load_axis_rows(path, axis_keys):
    """Read improvement_bpb + ci95 + significance for each axis from verdict.json."""
    with open(path) as f:
        verdict = json.load(f)
    rows = []
    for key, label in axis_keys:
        entry = verdict["axes"][key][CORPUS]
        imp = entry["improvement_bpb"]
        lo, hi = entry["ci95"]
        rows.append(
            {
                "label": label,
                "improvement": imp,
                "err_lo": imp - lo,
                "err_hi": hi - imp,
                "significant": bool(entry["significant"]),
            }
        )
    return rows, verdict.get("suite_version")


def draw_panel(ax, rows, panel_tag):
    ys = range(len(rows))
    for y, row in zip(ys, rows):
        if row["significant"]:
            ax.barh(
                y,
                row["improvement"],
                height=0.6,
                color=COLOR_SIG,
                edgecolor="black",
                linewidth=0.6,
            )
        else:
            ax.barh(
                y,
                row["improvement"],
                height=0.6,
                color=COLOR_NS,
                edgecolor="black",
                linewidth=0.6,
                hatch="///",
            )
        ax.errorbar(
            row["improvement"],
            y,
            xerr=[[row["err_lo"]], [row["err_hi"]]],
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
        )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel(r"$\Delta$BPB vs. baseline (WikiText-2)", fontsize=9)
    ax.text(
        0.02,
        1.02,
        panel_tag,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    rows_a, suite_a = load_axis_rows(SRC_A, PANEL_A_AXES)
    rows_b, suite_b = load_axis_rows(SRC_B, PANEL_B_AXES)
    assert suite_a == suite_b == "text-lm-v2", (suite_a, suite_b)

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.7, 2.2))
    draw_panel(ax_a, rows_a, "A")
    draw_panel(ax_b, rows_b, "B")

    # Shared legend for fill semantics (significant vs. not)
    sig_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor=COLOR_SIG, edgecolor="black", linewidth=0.6
    )
    ns_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor=COLOR_NS, edgecolor="black", linewidth=0.6, hatch="///"
    )
    fig.legend(
        [sig_patch, ns_patch],
        ["significant (95% CI excludes 0)", "not significant"],
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout()
    pdf_path = os.path.join(HERE, "fig_deconfound.pdf")
    png_path = os.path.join(HERE, "fig_deconfound.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print("wrote", pdf_path)
    print("wrote", png_path)


if __name__ == "__main__":
    main()
