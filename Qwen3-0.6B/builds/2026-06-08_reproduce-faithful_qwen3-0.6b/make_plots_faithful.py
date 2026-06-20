"""
Publication plots for the FAITHFUL Qwen3-0.6B from-scratch build.

CPU-ONLY, matplotlib Agg, no GPU, no model loads. Plots ONLY numbers parsed
from real on-disk files in results/. Every figure title/caption cites its
exact source file. Partial/short runs are LABELLED as such (no extrapolation).

Outputs (each as 300-dpi PNG **and** PDF) into results/plots/:
    fig1_loss_lr_<run>      training loss (raw+EMA) + LR schedule twin-axis
    fig2_grad_norm_<run>    grad-norm vs step (raw + EMA, clip line)
    fig3_peak_mem_<run>     peak CUDA mem vs step + 109 GB safe_cuda cap line
    fig4_val_ppl_<run>      eval val-PPL vs tokens (log y)
    fig5_phaseA_lr_sweep    Phase-A LR-sweep final val-PPL bar chart

Sources (relative to results/):
    qwen3_baseline2tpp_train.csv   step,loss,lr,grad_norm,peak_mem_gb,tok_seen
    qwen3_baseline2tpp_train.log   '[eval @ N] val PPL=X' lines + AFTER line
    qwen3_lr{17,24,30}_train.csv    Phase-A LR-sweep training traces
    qwen3_lr{17,24,30}_after.txt    Phase-A final 'val PPL: ... -> X'

    python make_plots_faithful.py
"""
from __future__ import annotations

import pathlib
import re

import matplotlib
matplotlib.use("Agg")  # headless, CPU-only — no display, no GPU
import matplotlib.pyplot as plt
import pandas as pd

# ---- clean serif style -----------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.fontsize": 9,
    "figure.dpi": 110,
})

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

LOSS_C, LR_C, GRAD_C, MEM_C, PPL_C = "#e76f51", "#2a9d8f", "#d97706", "#6a4c93", "#264653"
TOTAL_GB, CAP_GB = 119.7, 109.0  # GB10 unified pool; safe_cuda 0.85 cap

# The baseline run that owns the headline numbers.
BASE = "baseline2tpp"
BASE_CSV = RESULTS / f"qwen3_{BASE}_train.csv"
BASE_LOG = RESULTS / f"qwen3_{BASE}_train.log"

# Phase-A LR-sweep runs (short: 2000 steps each, ~131M tokens).
SWEEP = [
    ("lr17", "1.7e-3", "Qwen3 paper anchor"),
    ("lr24", "2.4e-3", "midpoint (chosen)"),
    ("lr30", "3.0e-3", "SmolLM2 anchor"),
]


def ema(series, alpha=0.05):
    out, m = [], None
    for v in series:
        m = v if m is None else alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def save(fig, stem):
    """Save a figure as 300-dpi PNG and PDF."""
    png, pdf = PLOTS / f"{stem}.png", PLOTS / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def parse_log_ppls(log_path: pathlib.Path):
    """Return (baseline_ppl, [(step, ppl)...], after_ppl) from a train log."""
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


def parse_after_ppl(after_path: pathlib.Path):
    """Return final val PPL from an after.txt 'val PPL: A -> B' line."""
    if not after_path.exists():
        return None
    m = re.search(r"val PPL:\s*[\d.]+\s*->\s*([\d.]+)", after_path.read_text())
    return float(m.group(1)) if m else None


def main():
    printed = {}  # stem -> list of (label, value) plotted, for cross-check
    df = pd.read_csv(BASE_CSV)
    base_ppl, evals, after_ppl = parse_log_ppls(BASE_LOG)
    final_step = int(df["step"].iloc[-1])
    final_tok = int(df["tok_seen"].iloc[-1])
    src_csv = BASE_CSV.name
    src_log = BASE_LOG.name

    # =====================================================================
    # FIG 1 — training loss (raw + EMA) + LR schedule on a twin axis
    # =====================================================================
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["step"], df["loss"], color=LOSS_C, alpha=0.30, lw=0.8, label="train loss (raw)")
    ax1.plot(df["step"], ema(df["loss"]), color=LOSS_C, lw=2.0, label="train loss (EMA α=0.05)")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("train loss (nats/token)", color=LOSS_C)
    ax1.tick_params(axis="y", labelcolor=LOSS_C)
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(df["step"], df["lr"], color=LR_C, lw=1.6, alpha=0.85, label="learning rate")
    ax2.set_ylabel("learning rate", color=LR_C)
    ax2.tick_params(axis="y", labelcolor=LR_C)
    ax1.set_title(
        "Qwen3-0.6B FAITHFUL — train loss & cosine LR (COMPLETE run, "
        f"{final_step:,} steps / {final_tok/1e9:.2f}B tok)\n"
        f"source: results/{src_csv}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    save(fig, f"fig1_loss_lr_{BASE}")
    printed[f"fig1_loss_lr_{BASE}"] = [
        ("loss[step1]", round(float(df["loss"].iloc[0]), 4)),
        ("loss[last]", round(float(df["loss"].iloc[-1]), 4)),
        ("loss[min]", round(float(df["loss"].min()), 4)),
        ("lr[peak]", float(df["lr"].max())),
        ("lr[end]", float(df["lr"].iloc[-1])),
        ("n_rows", len(df)),
    ]

    # =====================================================================
    # FIG 2 — gradient norm vs step
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(df["step"], df["grad_norm"], color=GRAD_C, lw=0.7, alpha=0.45, label="grad norm (raw)")
    ax.plot(df["step"], ema(df["grad_norm"]), color=GRAD_C, lw=2.0, label="grad norm (EMA α=0.05)")
    ax.axhline(1.0, ls="--", color="gray", lw=1.2, label="grad clip = 1.0")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("gradient L2 norm")
    ax.set_title(
        "Qwen3-0.6B FAITHFUL — gradient norm (COMPLETE run)\n"
        f"source: results/{src_csv}")
    ax.legend(loc="upper right")
    save(fig, f"fig2_grad_norm_{BASE}")
    printed[f"fig2_grad_norm_{BASE}"] = [
        ("grad[step1]", round(float(df["grad_norm"].iloc[0]), 4)),
        ("grad[last]", round(float(df["grad_norm"].iloc[-1]), 4)),
        ("grad[max]", round(float(df["grad_norm"].max()), 4)),
        ("grad[min]", round(float(df["grad_norm"].min()), 4)),
        ("grad[mean]", round(float(df["grad_norm"].mean()), 4)),
    ]

    # =====================================================================
    # FIG 3 — peak memory vs step + 109 GB safe_cuda cap line
    # =====================================================================
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(df["step"], df["peak_mem_gb"], color=MEM_C, lw=2.0, label="peak CUDA mem (logged)")
    ax.axhline(CAP_GB, ls="--", color="#e63946", lw=1.6, label=f"safe_cuda cap = {CAP_GB:.0f} GB")
    ax.axhline(TOTAL_GB, ls=":", color="black", lw=1.2, label=f"unified pool = {TOTAL_GB:.0f} GB")
    ax.set_ylim(0, TOTAL_GB * 1.05)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("memory (GB)")
    ax.set_title(
        "Qwen3-0.6B FAITHFUL — peak memory vs unified-pool cap (COMPLETE run)\n"
        f"source: results/{src_csv}")
    ax.legend(loc="center right")
    save(fig, f"fig3_peak_mem_{BASE}")
    printed[f"fig3_peak_mem_{BASE}"] = [
        ("mem[step1]", round(float(df["peak_mem_gb"].iloc[0]), 4)),
        ("mem[max]", round(float(df["peak_mem_gb"].max()), 4)),
        ("mem[last]", round(float(df["peak_mem_gb"].iloc[-1]), 4)),
        ("cap_GB", CAP_GB),
        ("pool_GB", TOTAL_GB),
    ]

    # =====================================================================
    # FIG 4 — eval val-PPL vs TOKENS (log y)
    # =====================================================================
    # Map each eval step -> tokens seen at that step (from the CSV).
    step_to_tok = dict(zip(df["step"], df["tok_seen"]))
    pts = []  # (tokens, ppl, label)
    for s, p in evals:
        tok = step_to_tok.get(s)
        if tok is None:
            # nearest logged step at/just before s
            sub = df[df["step"] <= s]
            tok = int(sub["tok_seen"].iloc[-1]) if len(sub) else None
        if tok is not None:
            pts.append((int(tok), p, f"@{s}"))
    if after_ppl is not None:
        pts.append((final_tok, after_ppl, "AFTER"))
    pts.sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs = [t / 1e9 for t, _, _ in pts]
    ys = [p for _, p, _ in pts]
    ax.plot(xs, ys, marker="o", color=PPL_C, lw=2.0, label="val PPL (mid-train evals + AFTER)")
    ax.set_yscale("log")
    ax.set_xlabel("tokens seen (billions)")
    ax.set_ylabel("validation perplexity (log scale)")
    ax.set_title(
        "Qwen3-0.6B FAITHFUL — validation PPL vs tokens (COMPLETE run)\n"
        f"sources: results/{src_log} (eval lines), results/{src_csv} (step->tokens)")
    for (tok, p, lab) in pts:
        ax.annotate(f"{p:.1f}", (tok / 1e9, p),
                    textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    ax.legend(loc="upper right")
    save(fig, f"fig4_val_ppl_{BASE}")
    printed[f"fig4_val_ppl_{BASE}"] = [
        (f"{lab} tok={tok:,}", round(p, 2)) for (tok, p, lab) in pts]
    if base_ppl is not None:
        printed[f"fig4_val_ppl_{BASE}"].insert(
            0, ("baseline@init (NOT plotted, off-scale)", round(base_ppl, 2)))

    # =====================================================================
    # FIG 5 — Phase-A LR-sweep final val-PPL bar chart (SHORT runs)
    # =====================================================================
    bars = []  # (label, ppl, peak_lr, steps, max_tok)
    for name, lr_str, note in SWEEP:
        after_p = parse_after_ppl(RESULTS / f"qwen3_{name}_after.txt")
        sdf = pd.read_csv(RESULTS / f"qwen3_{name}_train.csv")
        bars.append((name, after_p, lr_str, len(sdf), int(sdf["tok_seen"].max()), note))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [f"{n}\n(peak LR {lr})\n{note}" for n, _, lr, _, _, note in bars]
    vals = [p for _, p, _, _, _, _ in bars]
    colors = ["#8d99ae", "#2a9d8f", "#8d99ae"]  # highlight chosen lr24
    barobj = ax.bar(labels, vals, color=colors, edgecolor="black", width=0.6)
    ax.bar_label(barobj, fmt="%.2f", padding=3, fontsize=10)
    best_i = min(range(len(vals)), key=lambda i: vals[i])
    ax.set_ylabel("final validation PPL (lower is better)")
    ax.set_ylim(0, max(vals) * 1.15)
    # steps/tokens are identical across the sweep; state once
    s_steps = bars[0][3]
    s_tok = bars[0][4]
    ax.set_title(
        "Qwen3-0.6B FAITHFUL — Phase-A LR sweep final val PPL\n"
        f"SHORT matched-compute runs: {s_steps:,} steps / {s_tok/1e6:.0f}M tok each "
        f"(NOT the 2-TPP baseline)\n"
        "sources: results/qwen3_lr{17,24,30}_after.txt")
    ax.annotate(f"best: {bars[best_i][0]} = {vals[best_i]:.2f}",
                xy=(best_i, vals[best_i]), xytext=(best_i, vals[best_i] * 0.55),
                ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#2a9d8f"))
    save(fig, "fig5_phaseA_lr_sweep")
    printed["fig5_phaseA_lr_sweep"] = [
        (f"{n} (peak LR {lr}, {st:,} steps, {tk/1e6:.0f}M tok)", round(p, 2))
        for n, p, lr, st, tk, _ in bars]

    # =====================================================================
    # Print every plotted data point for cross-check
    # =====================================================================
    print("=" * 72)
    print("PLOTTED DATA POINTS (cross-check against on-disk source files)")
    print("=" * 72)
    for stem, items in printed.items():
        print(f"\n[{stem}]")
        for lab, val in items:
            print(f"    {lab:<42} {val}")
    print("\n" + "=" * 72)
    print(f"Wrote {len(printed)} figures (PNG @300dpi + PDF) to {PLOTS}")
    print("=" * 72)
    return printed, df, evals, after_ppl, base_ppl, bars, final_step, final_tok


if __name__ == "__main__":
    main()
