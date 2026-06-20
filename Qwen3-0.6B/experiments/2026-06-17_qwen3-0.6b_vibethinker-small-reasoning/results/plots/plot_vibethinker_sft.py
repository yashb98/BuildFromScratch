#!/usr/bin/env python3
"""
Re-runnable plotter for the VibeThinker SFT post-train of Qwen3-0.6B.

HARD RULES honored here:
  * CPU-only. matplotlib 'Agg' backend forced BEFORE pyplot import. No GPU, no
    CUDA, no model loads, no torch import. This script never touches the running
    trainer (PID in results/trainer.pid owns the GPU; we only read text files).
  * Plots ONLY numbers parsed from the on-disk driver log of THIS run.
  * Every figure title cites its exact source file.
  * The run is IN PROGRESS -> every title is labelled so; no extrapolation, no
    fabricated points. We plot only the discrete steps that appear in the log.

Re-run it any time as the run progresses; it re-parses the log and re-renders.

Usage:
    python3 plot_vibethinker_sft.py
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # CPU-only, headless. MUST precede pyplot import.
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- paths (absolute) -------------------------------------------------------
EXP_DIR = "/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/experiments/2026-06-17_qwen3-0.6b_vibethinker-small-reasoning"
DRIVER_LOG = os.path.join(EXP_DIR, "results", "vibethinker_sft_driver.log")
OUT_DIR = os.path.join(EXP_DIR, "results", "plots")
SRC_NAME = "results/vibethinker_sft_driver.log"  # short label for captions

TOTAL_STEPS = 238
BASE_PPL_FLOOR = 14.262  # pre-SFT base reasoning-PPL floor, verified from log line 8
AS_OF = "2026-06-18"

# --- clean serif style ------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.fontsize": 9,
    "figure.dpi": 120,
})

# --- regexes for the driver-log lines --------------------------------------
# step line: "[time] [vibethinker_sft] step N/238 loss X lr .. |grad| .. mem .. tok/s .. tok .. ETA .."
STEP_RE = re.compile(
    r"\bstep\s+(\d+)\s*/\s*\d+\s+loss\s+([0-9]*\.?[0-9]+)"
)
# eval line: "[eval @ N] reasoning PPL=X (Δ vs base ..)"
EVAL_RE = re.compile(
    r"\[eval\s*@\s*(\d+)\]\s+reasoning\s+PPL\s*=\s*([0-9]*\.?[0-9]+)"
)


def parse_log(path):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: driver log not found: {path}")
    steps, losses = [], []
    eval_steps, eval_ppls = [], []
    with open(path, "r") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                continue
            e = EVAL_RE.search(line)
            if e:
                eval_steps.append(int(e.group(1)))
                eval_ppls.append(float(e.group(2)))
    # keep sorted/dedup by step (last value wins) for robustness on re-runs
    def _dedup(xs, ys):
        d = {}
        for x, y in zip(xs, ys):
            d[x] = y
        xs2 = sorted(d)
        return xs2, [d[x] for x in xs2]
    steps, losses = _dedup(steps, losses)
    eval_steps, eval_ppls = _dedup(eval_steps, eval_ppls)
    return steps, losses, eval_steps, eval_ppls


RUN_DONE = False  # set in main() from the driver log's DONE line

def in_progress_suffix(latest_step):
    if RUN_DONE:
        return f"(COMPLETE — {TOTAL_STEPS}/{TOTAL_STEPS} steps; run finished {AS_OF})"
    return (f"(IN PROGRESS — step {latest_step}/{TOTAL_STEPS} as of "
            f"{AS_OF}, run not yet complete)")


def plot_loss(steps, losses, latest_step):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(steps, losses, marker="o", ms=4.5, lw=1.4, color="#1f4e79",
            label="training loss (logged steps)")
    ax.set_xlabel("training step (of %d)" % TOTAL_STEPS)
    ax.set_ylabel("SFT cross-entropy loss")
    ax.set_title(
        "VibeThinker SFT — training loss vs step\n"
        + in_progress_suffix(latest_step)
        + f"\nsource: {SRC_NAME}",
        fontsize=10,
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"vibethinker_sft_loss.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ppl(eval_steps, eval_ppls, latest_step):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.axhline(BASE_PPL_FLOOR, color="#a00000", lw=1.4, ls="--",
               label=f"pre-SFT base floor = {BASE_PPL_FLOOR:.3f}")
    if eval_steps:
        ax.plot(eval_steps, eval_ppls, marker="s", ms=6, lw=1.6,
                color="#1f7a1f", label="reasoning val-PPL (logged evals)")
        # annotate each point with its value
        for x, y in zip(eval_steps, eval_ppls):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("training step (of %d)" % TOTAL_STEPS)
    ax.set_ylabel("held-out reasoning perplexity (PPL)")
    ax.set_title(
        "VibeThinker SFT — reasoning val-PPL vs step\n"
        + in_progress_suffix(latest_step)
        + f"\nsource: {SRC_NAME}",
        fontsize=10,
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"vibethinker_sft_reasoning_ppl.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(steps, losses, eval_steps, eval_ppls, latest_step):
    lines = []
    lines.append("# VibeThinker SFT — plots\n")
    status = (f"COMPLETE — {TOTAL_STEPS}/{TOTAL_STEPS} steps, finished {AS_OF}" if RUN_DONE
              else f"IN PROGRESS — latest logged step {latest_step}/{TOTAL_STEPS} as of {AS_OF}")
    lines.append(f"_Run {status}. No extrapolation; only discrete logged points are drawn._\n")
    lines.append(f"All data parsed from `{SRC_NAME}` by `plots/plot_vibethinker_sft.py` "
                 "(CPU-only, matplotlib Agg, no GPU, no model loads). Re-run that script "
                 "to refresh as the run progresses.\n")
    lines.append("## Figures\n")
    lines.append("- **vibethinker_sft_loss.{png,pdf}** — SFT training cross-entropy "
                 f"loss vs step (logged steps {steps[0]}–{steps[-1]}). "
                 f"Source: `{SRC_NAME}`.")
    lines.append("- **vibethinker_sft_reasoning_ppl.{png,pdf}** — held-out reasoning "
                 "perplexity vs step at eval checkpoints, with the pre-SFT base floor "
                 f"{BASE_PPL_FLOOR:.3f} as a dashed reference line. Source: `{SRC_NAME}`.")
    lines.append("")
    lines.append("Each figure is saved as PNG (300 dpi) and PDF.\n")
    with open(os.path.join(OUT_DIR, "README.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    global RUN_DONE
    RUN_DONE = "DONE" in open(DRIVER_LOG).read()
    steps, losses, eval_steps, eval_ppls = parse_log(DRIVER_LOG)
    if not steps:
        sys.exit("ERROR: no step lines parsed from driver log.")
    latest_step = max(steps + eval_steps) if eval_steps else max(steps)

    plot_loss(steps, losses, latest_step)
    plot_ppl(eval_steps, eval_ppls, latest_step)
    write_readme(steps, losses, eval_steps, eval_ppls, latest_step)

    # --- print exact plotted data for cross-checking ---
    print("=" * 64)
    print(f"SOURCE (parsed): {DRIVER_LOG}")
    print(f"latest logged step: {latest_step}/{TOTAL_STEPS}  (run IN PROGRESS)")
    print("=" * 64)
    print("\nFIGURE 1 -- training loss vs step (driver-log step lines):")
    print(f"  {'step':>5}  {'loss':>10}")
    for s, l in zip(steps, losses):
        print(f"  {s:>5}  {l:>10.4f}")
    print(f"  [n={len(steps)} points]")
    print("\nFIGURE 2 -- reasoning val-PPL vs step (driver-log eval lines):")
    print(f"  base floor (horizontal ref line): PPL = {BASE_PPL_FLOOR:.3f}")
    print(f"  {'step':>5}  {'PPL':>10}")
    for s, p in zip(eval_steps, eval_ppls):
        print(f"  {s:>5}  {p:>10.3f}")
    print(f"  [n={len(eval_steps)} eval points]")
    print("\nWrote:")
    for f in ("vibethinker_sft_loss.png", "vibethinker_sft_loss.pdf",
              "vibethinker_sft_reasoning_ppl.png", "vibethinker_sft_reasoning_ppl.pdf",
              "README.md"):
        print(f"  {os.path.join(OUT_DIR, f)}")


if __name__ == "__main__":
    main()
