#!/usr/bin/env python3
"""
Plot generator for the EXPLORATORY (partial-RoPE) Qwen3-0.6B build.

HARD RULES enforced here:
  - CPU-only, NO GPU, NO model loads, NO torch import.
  - matplotlib Agg backend (no display).
  - Plots ONLY numbers parsed from real on-disk log files this run.
  - Partial/incomplete run (prope10) is LABELED incomplete; no extrapolation.
  - Every figure cites its exact source file in title + caption.

Sources (parsed live):
  prope25 (COMPLETED): results/qwen3_prope25_2tpp_train.log
  prope10 (INCOMPLETE, died at step 5450): results/qwen3_prope10_2tpp_train.log
"""
import os
import re
import sys

# Force no GPU visibility for any transitive lib; we import zero ML libs anyway.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib
matplotlib.use("Agg")  # headless, no display
import matplotlib.pyplot as plt

RESULTS_DIR = "/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds/2026-06-08_reproduce-exploratory_qwen3-0.6b/results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PROPE25_LOG = os.path.join(RESULTS_DIR, "qwen3_prope25_2tpp_train.log")
PROPE10_LOG = os.path.join(RESULTS_DIR, "qwen3_prope10_2tpp_train.log")

# regex for step lines:  "step    250/18150  loss 6.0284 ..."
STEP_RE = re.compile(r"step\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)")
# regex for eval lines:  "[eval @ 2000] val PPL=62.33"
EVAL_RE = re.compile(r"\[eval @ (\d+)\]\s+val PPL=([0-9.]+)")
# regex for final DONE:  "DONE  final val PPL=29.54"
DONE_RE = re.compile(r"DONE\s+final val PPL=([0-9.]+)")


def parse_log(path):
    """Return (steps, losses, eval_steps, eval_ppls, total_steps, final_done_ppl, last_step)."""
    steps, losses = [], []
    eval_steps, eval_ppls = [], []
    total_steps = None
    final_done_ppl = None
    with open(path, "r") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                s = int(m.group(1))
                total_steps = int(m.group(2))
                steps.append(s)
                losses.append(float(m.group(3)))
            me = EVAL_RE.search(line)
            if me:
                eval_steps.append(int(me.group(1)))
                eval_ppls.append(float(me.group(2)))
            md = DONE_RE.search(line)
            if md:
                final_done_ppl = float(md.group(1))
    last_step = steps[-1] if steps else None
    return steps, losses, eval_steps, eval_ppls, total_steps, final_done_ppl, last_step


def main():
    # --- Parse both logs from disk ---
    (s25, l25, es25, ep25, tot25, done25, last25) = parse_log(PROPE25_LOG)
    (s10, l10, es10, ep10, tot10, done10, last10) = parse_log(PROPE10_LOG)

    # The DONE eval is the final-step val PPL; append it as the terminal eval point.
    if done25 is not None and last25 is not None:
        es25_full = es25 + [last25]
        ep25_full = ep25 + [done25]
    else:
        es25_full, ep25_full = es25, ep25

    # --- Echo the exact data points plotted (for cross-check) ---
    print("=" * 78)
    print("EXACT DATA POINTS PLOTTED (parsed live from on-disk logs)")
    print("=" * 78)
    print(f"\n[prope25] source: {PROPE25_LOG}")
    print(f"  status: COMPLETED (DONE) | total_steps={tot25} | last_step={last25} | final_val_PPL(DONE)={done25}")
    print(f"  training-loss points: n={len(s25)}  (step, loss) every 50 steps from {s25[0]} to {s25[-1]}")
    print("  eval (val-PPL) points [from [eval @ N] lines + terminal DONE]:")
    for st, pp in zip(es25_full, ep25_full):
        tag = "  <- DONE final" if (done25 is not None and st == last25 and pp == done25) else ""
        print(f"    step {st:>6} : val PPL = {pp}{tag}")

    print(f"\n[prope10] source: {PROPE10_LOG}")
    print(f"  status: INCOMPLETE -- trainer DIED at step {last10}/{tot10} (no DONE line)")
    print(f"  training-loss points: n={len(s10)}  (step, loss) every 50 steps from {s10[0]} to {s10[-1]}")
    print("  eval (val-PPL) points [from [eval @ N] lines only; NO terminal DONE]:")
    for st, pp in zip(es10, ep10):
        print(f"    step {st:>6} : val PPL = {pp}")
    print("=" * 78)

    # --- Style: clean serif ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })

    C25 = "#1f4e79"  # deep blue, completed
    C10 = "#c0392b"  # red, incomplete

    # =========================================================
    # FIGURE 1: training loss vs step (both runs overlaid)
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(s25, l25, color=C25, lw=1.3, alpha=0.9,
             label=f"prope25 (rotary_dim=32, factor=0.25) -- COMPLETED {tot25} steps")
    ax1.plot(s10, l10, color=C10, lw=1.5, ls="--", alpha=0.9,
             label=f"prope10 (rotary_dim=12, factor=0.10) -- INCOMPLETE, died @ step {last10}")
    # mark the death point
    ax1.scatter([s10[-1]], [l10[-1]], color=C10, s=55, zorder=5,
                marker="X", label=f"prope10 death point (step {last10}, loss {l10[-1]})")

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Training loss (cross-entropy)")
    ax1.set_title("Qwen3-0.6B EXPLORATORY (partial-RoPE): training loss vs step\n"
                  "prope10 is INCOMPLETE -- died at step "
                  f"{last10}/{tot10} (no extrapolation)",
                  fontsize=12)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax1.set_xlim(left=0)
    cap1 = ("Source: results/qwen3_prope25_2tpp_train.log (COMPLETED, DONE) and "
            "results/qwen3_prope10_2tpp_train.log (INCOMPLETE, died @ step "
            f"{last10}/{tot10}). Per-50-step loss parsed directly from logs; no smoothing, no extrapolation.")
    fig1.text(0.5, 0.005, cap1, ha="center", va="bottom", fontsize=7.0,
              wrap=True, color="#444444")
    fig1.tight_layout(rect=[0, 0.045, 1, 1])
    f1_png = os.path.join(PLOTS_DIR, "exploratory_prope_train_loss.png")
    f1_pdf = os.path.join(PLOTS_DIR, "exploratory_prope_train_loss.pdf")
    fig1.savefig(f1_png, dpi=300)
    fig1.savefig(f1_pdf)
    plt.close(fig1)

    # =========================================================
    # FIGURE 2: eval val-PPL vs step (log y, both runs)
    # =========================================================
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    ax2.plot(es25_full, ep25_full, color=C25, lw=1.4, marker="o", ms=5,
             label=f"prope25 (factor=0.25) -- COMPLETED, final PPL={done25}")
    # annotate prope25 final
    ax2.annotate(f"final PPL={done25}\n(DONE @ step {last25})",
                 xy=(last25, done25), xytext=(last25 * 0.62, done25 * 1.9),
                 fontsize=8.5, color=C25,
                 arrowprops=dict(arrowstyle="->", color=C25, lw=1.0))

    ax2.plot(es10, ep10, color=C10, lw=1.6, ls="--", marker="s", ms=6,
             label=f"prope10 (factor=0.10) -- INCOMPLETE (died @ step {last10}; only 2 evals)")
    ax2.annotate(f"last eval PPL={ep10[-1]}\n(@ step {es10[-1]}; run died @ {last10})",
                 xy=(es10[-1], ep10[-1]), xytext=(es10[-1] * 1.15, ep10[-1] * 1.25),
                 fontsize=8.5, color=C10,
                 arrowprops=dict(arrowstyle="->", color=C10, lw=1.0))

    ax2.set_yscale("log")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Validation perplexity (log scale)")
    ax2.set_title("Qwen3-0.6B EXPLORATORY (partial-RoPE): eval val-PPL vs step (log y)\n"
                  "prope10 is INCOMPLETE -- died at step "
                  f"{last10}/{tot10}, only 2 eval points (no extrapolation)",
                  fontsize=12)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2.set_xlim(left=0)
    cap2 = ("Source: results/qwen3_prope25_2tpp_train.log (COMPLETED; eval @ 2k..18k + DONE final PPL=29.54) "
            "and results/qwen3_prope10_2tpp_train.log (INCOMPLETE, died @ step "
            f"{last10}/{tot10}; only eval @ 2000 & 4000 exist). val-PPL parsed directly; no extrapolation.")
    fig2.text(0.5, 0.005, cap2, ha="center", va="bottom", fontsize=7.0,
              wrap=True, color="#444444")
    fig2.tight_layout(rect=[0, 0.045, 1, 1])
    f2_png = os.path.join(PLOTS_DIR, "exploratory_prope_val_ppl.png")
    f2_pdf = os.path.join(PLOTS_DIR, "exploratory_prope_val_ppl.pdf")
    fig2.savefig(f2_png, dpi=300)
    fig2.savefig(f2_pdf)
    plt.close(fig2)

    print("\nWROTE:")
    for p in [f1_png, f1_pdf, f2_png, f2_pdf]:
        print(f"  {p}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    sys.exit(main())
