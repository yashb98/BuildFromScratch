#!/usr/bin/env python3
"""
Cross-build comparison figures for the Qwen3-0.6B builds.

CPU-ONLY, matplotlib Agg, NO GPU, NO model loads, NO running-process interaction.
Every number below is parsed/transcribed from the real on-disk result files cited
in each figure's title/caption (verified this run). No extrapolation; partial /
in-progress runs are labelled. Each figure -> PNG (300 dpi) + PDF.

Source files (verified this run):
  Faithful final:   builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_after.txt
                    -> "val PPL: 185810.49 -> 28.65"
  Faithful traj:    builds/.../faithful.../results/qwen3_baseline2tpp_train.log  ([eval @ N] lines)
  IMU-1 final/traj: builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log
                    -> "[eval @ 18000] val PPL=23.52"
  pRoPE-25 final:   builds/2026-06-08_reproduce-exploratory_qwen3-0.6b/results/qwen3_prope25_2tpp_train.log
                    -> "DONE  final val PPL=29.54"
  Original ref:     builds/.../faithful.../results/original_vs_repro.txt
                    -> "ORIGINAL  Qwen3-0.6B-Base (36T tok)   val PPL =   13.400"
  Matched budget:   all three 2-TPP logs report tok/step=65,536, steps=18,150,
                    token_budget=1,189,478,400 (1.19B).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

OUT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
})

TOK_PER_STEP = 65_536          # verified identical across the 3 runs
BUDGET = 1_189_478_400         # 18,150 * 65,536, verified in all 3 logs
BUDGET_B = BUDGET / 1e9        # 1.189...

# ---------------------------------------------------------------------------
# Verified scalar endpoints (matched-compute, 1.19B tokens)
# ---------------------------------------------------------------------------
ORIGINAL_PPL = 13.40          # Qwen3-0.6B-Base, original_vs_repro.txt
FINAL = {
    # label                       ppl     gap_x   source-file (basename)
    "Faithful":               (28.65, 2.14, "qwen3_baseline2tpp_after.txt"),
    "IMU-1\n(Modernized)":    (23.52, 1.76, "qwen3_imu1_2tpp_train.log (eval@18000)"),
    "partial-RoPE-25%\n(Exploratory)": (29.54, 2.21, "qwen3_prope25_2tpp_train.log (DONE)"),
}

# ---------------------------------------------------------------------------
# Verified eval trajectories (step -> val PPL), parsed this run from the logs.
# step is converted to tokens via TOK_PER_STEP (65,536), verified.
# ---------------------------------------------------------------------------
# Faithful: qwen3_baseline2tpp_train.log [eval @ N] + AFTER 28.65 @ step 18150
FAITHFUL_TRAJ = [
    (2000, 60.10), (4000, 45.06), (6000, 39.44), (8000, 35.71),
    (10000, 33.04), (12000, 30.93), (14000, 29.59), (16000, 28.93),
    (18000, 28.66), (18150, 28.65),   # 18150 = AFTER eval (after.txt)
]
# IMU-1: qwen3_imu1_2tpp_train.log [eval @ N]
IMU1_TRAJ = [
    (2000, 44.38), (4000, 35.62), (6000, 32.54), (8000, 30.41),
    (10000, 29.14), (12000, 28.43), (14000, 27.40), (16000, 24.86),
    (18000, 23.52),
]


def save(fig, name):
    png = os.path.join(PLOTS, name + ".png")
    pdf = os.path.join(PLOTS, name + ".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# ===========================================================================
# FIGURE 1 — matched-compute final val-PPL bar
# ===========================================================================
def fig_bar():
    labels = list(FINAL.keys())
    ppls = [FINAL[k][0] for k in labels]
    gaps = [FINAL[k][1] for k in labels]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    x = range(len(labels))
    bars = ax.bar(x, ppls, color=colors, width=0.6, edgecolor="black", linewidth=0.6, zorder=3)

    # original reference line
    ax.axhline(ORIGINAL_PPL, color="black", linestyle="--", linewidth=1.4, zorder=2,
               label=f"Published Qwen3-0.6B-Base = {ORIGINAL_PPL:.2f} (36T tok)")

    for xi, (b, ppl, gap) in enumerate(zip(bars, ppls, gaps)):
        ax.text(xi, ppl + 0.5, f"{ppl:.2f}\n({gap:.2f}x orig)",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Final validation perplexity (lower is better)")
    ax.set_ylim(0, max(ppls) * 1.18)
    ax.set_title(
        "Qwen3-0.6B from-scratch: matched-compute (1.19B-token) final val-PPL\n"
        "Faithful=28.65 (after.txt) | IMU-1=23.52 (imu1 log eval@18000) | "
        "pRoPE-25%=29.54 (prope25 log DONE)\n"
        "ref: original_vs_repro.txt; all three @ 18,150 steps x 65,536 tok = 1,189,478,400 tokens",
        fontsize=8.6)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, "fig1_matched_compute_final_ppl_bar")


# ===========================================================================
# FIGURE 2 — eval val-PPL vs tokens: faithful vs IMU-1 (log y)
# ===========================================================================
def fig_traj():
    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    fx = [s * TOK_PER_STEP / 1e9 for s, _ in FAITHFUL_TRAJ]
    fy = [p for _, p in FAITHFUL_TRAJ]
    ix = [s * TOK_PER_STEP / 1e9 for s, _ in IMU1_TRAJ]
    iy = [p for _, p in IMU1_TRAJ]

    ax.plot(fx, fy, marker="o", color="#4C72B0", linewidth=1.6, markersize=5,
            label="Faithful (baseline2tpp) -> 28.65", zorder=3)
    ax.plot(ix, iy, marker="s", color="#55A868", linewidth=1.6, markersize=5,
            label="IMU-1 / NorMuon (imu1_2tpp) -> 23.52", zorder=3)

    ax.axhline(ORIGINAL_PPL, color="black", linestyle="--", linewidth=1.2, zorder=2,
               label=f"Published Qwen3-0.6B-Base = {ORIGINAL_PPL:.2f}")

    # annotate endpoints
    ax.annotate(f"{fy[-1]:.2f}", (fx[-1], fy[-1]), textcoords="offset points",
                xytext=(4, 6), fontsize=9, color="#4C72B0", fontweight="bold")
    ax.annotate(f"{iy[-1]:.2f}", (ix[-1], iy[-1]), textcoords="offset points",
                xytext=(4, -12), fontsize=9, color="#55A868", fontweight="bold")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.set_yticks([15, 20, 25, 30, 40, 50, 60])
    ax.set_xlabel("Tokens seen (billions; step x 65,536 / 1e9)")
    ax.set_ylabel("Validation perplexity (log scale, lower is better)")
    ax.set_xlim(0, BUDGET_B * 1.02)
    ax.set_title(
        "Eval val-PPL vs tokens: Faithful vs IMU-1 (both COMPLETE @ 1.19B tok)\n"
        "src: qwen3_baseline2tpp_train.log (eval lines + after.txt) ; "
        "qwen3_imu1_2tpp_train.log (eval lines)\n"
        "step->tokens via tok/step=65,536 (verified in both logs); init PPL ~185k omitted (off-scale)",
        fontsize=8.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    return save(fig, "fig2_eval_ppl_vs_tokens_faithful_vs_imu1")


if __name__ == "__main__":
    print("=== Qwen3-0.6B cross-build overview plots ===")
    print(f"matched budget: {BUDGET:,} tokens = {BUDGET_B:.6f}B (18,150 x 65,536)")
    print(f"original reference PPL: {ORIGINAL_PPL}")
    print()

    p1 = fig_bar()
    print("FIG1 matched-compute final val-PPL bar:")
    for k, (ppl, gap, src) in FINAL.items():
        print(f"   {k.replace(chr(10),' '):34s} PPL={ppl:6.2f}  gap={gap:.2f}x  src={src}")
    print(f"   reference line: {ORIGINAL_PPL} (original_vs_repro.txt)")
    print(f"   -> {p1[0]}\n   -> {p1[1]}\n")

    p2 = fig_traj()
    print("FIG2 eval val-PPL vs tokens (Faithful vs IMU-1):")
    print("   FAITHFUL (step, tokenB, PPL):")
    for s, p in FAITHFUL_TRAJ:
        print(f"      step {s:>5}  {s*TOK_PER_STEP/1e9:.4f}B  PPL={p}")
    print("   IMU-1 (step, tokenB, PPL):")
    for s, p in IMU1_TRAJ:
        print(f"      step {s:>5}  {s*TOK_PER_STEP/1e9:.4f}B  PPL={p}")
    print(f"   -> {p2[0]}\n   -> {p2[1]}\n")
    print("DONE")
