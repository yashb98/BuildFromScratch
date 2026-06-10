"""
Visual results for the Qwen3-0.6B from-scratch smoke run.

Reads:
    results/qwen3_train.csv   (step, loss, lr, grad_norm, peak_mem_gb, tok_seen)
    results/qwen3_train.log   (baseline / mid-training eval / after val PPL)

Writes PNGs under results/plots/ and a combined dashboard. Mirrors the plot
conventions in SmolLM2-134(base)/_build_notebook.py (palette, dpi=110).

    python make_plots.py
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ap = argparse.ArgumentParser()
_ap.add_argument("--run_name", default="", help="tag matching train_qwen3.py --run_name (reads qwen3_<tag>train.csv)")
_ARGS = _ap.parse_args()
_TAG = f"{_ARGS.run_name}_" if _ARGS.run_name else ""

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / ("plots" if not _ARGS.run_name else f"plots_{_ARGS.run_name}")
PLOTS.mkdir(parents=True, exist_ok=True)

LOSS_C, LR_C, GRAD_C, MEM_C, PPL_C = "#e76f51", "#2a9d8f", "#d97706", "#6a4c93", "#264653"
TOTAL_GB, CAP_GB = 119.7, 109.0  # GB10 unified pool; safe_cuda 0.85 cap


def ema(series, alpha=0.1):
    out, m = [], None
    for v in series:
        m = v if m is None else alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def parse_log_ppls(log_path: pathlib.Path):
    """Return (baseline_ppl, [(step, ppl)...], after_ppl) parsed from the log."""
    base, evals, after = None, [], None
    if not log_path.exists():
        return base, evals, after
    text = log_path.read_text()
    m = re.search(r"baseline val PPL=([\d.]+)", text)
    if m:
        base = float(m.group(1))
    for m in re.finditer(r"\[eval @ (\d+)\] val PPL=([\d.]+)", text):
        evals.append((int(m.group(1)), float(m.group(2))))
    m = re.search(r"AFTER val PPL=([\d.]+)", text)
    if m:
        after = float(m.group(1))
    return base, evals, after


def main():
    csv_path = RESULTS / f"qwen3_{_TAG}train.csv"
    if not csv_path.exists():
        print(f"No {csv_path} yet — run training first.")
        return 1
    df = pd.read_csv(csv_path)
    base_ppl, evals, after_ppl = parse_log_ppls(RESULTS / f"qwen3_{_TAG}train.log")
    print(f"Loaded {len(df)} steps; baseline_ppl={base_ppl}, evals={len(evals)}, after_ppl={after_ppl}")

    # 1) Loss curve (raw + EMA) with LR on a twin axis ----------------------
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["step"], df["loss"], color=LOSS_C, alpha=0.35, lw=1, label="loss (raw)")
    ax1.plot(df["step"], ema(df["loss"]), color=LOSS_C, lw=2, label="loss (EMA)")
    ax1.set_xlabel("optimizer step"); ax1.set_ylabel("train loss (nats/token)", color=LOSS_C)
    ax1.tick_params(axis="y", labelcolor=LOSS_C)
    ax1.axhline(math.log(151936), ls=":", color="gray", lw=1,
                label=f"ln(vocab)={math.log(151936):.2f} (random-init)")
    ax2 = ax1.twinx()
    ax2.plot(df["step"], df["lr"], color=LR_C, lw=1.4, alpha=0.8, label="lr")
    ax2.set_ylabel("learning rate", color=LR_C); ax2.tick_params(axis="y", labelcolor=LR_C)
    ax1.set_title("Qwen3-0.6B from-scratch — training loss & LR")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "loss_curve.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 2) LR schedule --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["step"], df["lr"], color=LR_C, lw=2)
    ax.set_xlabel("step"); ax.set_ylabel("learning rate"); ax.set_title("Cosine LR schedule (1.7e-3 → 3.2e-4)")
    fig.tight_layout(); fig.savefig(PLOTS / "lr_schedule.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 3) Grad norm ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["step"], df["grad_norm"], color=GRAD_C, lw=1, alpha=0.6)
    ax.plot(df["step"], ema(df["grad_norm"]), color=GRAD_C, lw=2)
    ax.axhline(1.0, ls="--", color="gray", lw=1, label="clip=1.0")
    ax.set_xlabel("step"); ax.set_ylabel("grad norm"); ax.set_title("Gradient norm"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "grad_norm.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 4) Peak memory vs the unified-pool cap (visualizes the crash guard) ----
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["step"], df["peak_mem_gb"], color=MEM_C, lw=2, label="peak CUDA mem")
    ax.axhline(CAP_GB, ls="--", color="#e63946", lw=1.5, label=f"safe_cuda cap ({CAP_GB:.0f} GB)")
    ax.axhline(TOTAL_GB, ls=":", color="black", lw=1, label=f"unified pool ({TOTAL_GB:.0f} GB)")
    ax.set_ylim(0, TOTAL_GB * 1.05)
    ax.set_xlabel("step"); ax.set_ylabel("GB"); ax.set_title("Peak memory vs unified-pool cap"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "peak_mem.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 5) Validation PPL: baseline → mid-evals → after (log scale) -----------
    if base_ppl or evals or after_ppl:
        fig, ax = plt.subplots(figsize=(9, 4))
        xs, ys = [], []
        if base_ppl:
            xs.append(0); ys.append(base_ppl)
        xs += [s for s, _ in evals]; ys += [p for _, p in evals]
        if after_ppl:
            xs.append(int(df["step"].iloc[-1])); ys.append(after_ppl)
        ax.plot(xs, ys, marker="o", color=PPL_C, lw=2)
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("val perplexity (log)")
        ax.set_title("FineWeb-Edu validation PPL")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
        fig.tight_layout(); fig.savefig(PLOTS / "val_ppl.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 6) Combined dashboard -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(df["step"], df["loss"], color=LOSS_C, alpha=0.3, lw=1)
    axes[0, 0].plot(df["step"], ema(df["loss"]), color=LOSS_C, lw=2)
    axes[0, 0].set_title("train loss"); axes[0, 0].set_xlabel("step")
    axes[0, 1].plot(df["step"], df["lr"], color=LR_C, lw=2); axes[0, 1].set_title("learning rate"); axes[0, 1].set_xlabel("step")
    axes[1, 0].plot(df["step"], ema(df["grad_norm"]), color=GRAD_C, lw=2)
    axes[1, 0].axhline(1.0, ls="--", color="gray", lw=1); axes[1, 0].set_title("grad norm (EMA)"); axes[1, 0].set_xlabel("step")
    axes[1, 1].plot(df["step"], df["peak_mem_gb"], color=MEM_C, lw=2)
    axes[1, 1].axhline(CAP_GB, ls="--", color="#e63946", lw=1.5)
    axes[1, 1].axhline(TOTAL_GB, ls=":", color="black", lw=1)
    axes[1, 1].set_ylim(0, TOTAL_GB * 1.05); axes[1, 1].set_title("peak mem vs cap"); axes[1, 1].set_xlabel("step")
    fig.suptitle("Qwen3-0.6B from-scratch smoke run — dashboard", fontsize=14)
    fig.tight_layout(); fig.savefig(PLOTS / "dashboard.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # Summary table ---------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  steps: {len(df)}  | tokens: {df['tok_seen'].iloc[-1]:,}")
    print(f"  loss: {df['loss'].iloc[0]:.3f} (start) -> {df['loss'].iloc[-1]:.3f} (end), "
          f"min {df['loss'].min():.3f}")
    print(f"  peak mem: max {df['peak_mem_gb'].max():.1f} GB (cap {CAP_GB} GB, pool {TOTAL_GB} GB)")
    if base_ppl and after_ppl:
        print(f"  val PPL: {base_ppl:,.1f} -> {after_ppl:,.1f}  "
              f"({100*(base_ppl-after_ppl)/base_ppl:+.1f}%)")
    print(f"\nWrote {len(list(PLOTS.glob('*.png')))} plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
