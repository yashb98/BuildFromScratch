#!/usr/bin/env python3
"""make_figures.py — figures for the qwen3-imu1-matched-compute paper.

Every figure is plotted from the ORIGINAL result files on disk (no hardcoded
numbers). Run from anywhere; paths are absolute. CPU only, matplotlib Agg.
Outputs PNG (300 dpi) + PDF (vector, for the LaTeX build) into this directory,
and prints the extracted endpoints so they can be cross-checked against
evidence_manifest.json.

Sources:
  faithful CSV : .../reproduce-faithful.../results/qwen3_baseline2tpp_train.csv  (step,loss,...)
  faithful log : .../reproduce-faithful.../results/qwen3_baseline2tpp_train.log  ([eval @ N] val PPL=..)
  imu    log   : .../reproduce-modernized.../results/qwen3_imu1_2tpp_train.log    (step lines + eval lines)
"""
import re
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/yashb98/Downloads/BuildFromScratch")
FAITH = REPO / "Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results"
MOD = REPO / "Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results"
OUT = Path(__file__).resolve().parent
TOK_PER_STEP = 65_536

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "figure.dpi": 300,
})
FAITH_C = "#444444"   # baseline: neutral grey
IMU_C = "#1f6fb4"     # modernized: blue


def parse_evals(logpath):
    """[(step, ppl)] from '[eval @ N] val PPL=X' lines."""
    pat = re.compile(r"\[eval @ (\d+)\] val PPL=([\d.]+)")
    out = []
    for line in Path(logpath).read_text().splitlines():
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def parse_log_ce(logpath):
    """[(step, ce)] from 'step N/T ce X' lines (IMU log, every 50 steps)."""
    pat = re.compile(r"step\s+(\d+)/\d+\s+ce\s+([\d.]+)")
    out = []
    for line in Path(logpath).read_text().splitlines():
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def parse_csv_loss(csvpath):
    """[(step, loss)] from the faithful per-step CSV."""
    out = []
    with open(csvpath) as f:
        for row in csv.DictReader(f):
            out.append((int(row["step"]), float(row["loss"])))
    return out


def smooth(ys, k=51):
    """Centered moving average; k odd. Keeps length via edge clamping."""
    if len(ys) < k:
        return ys
    half = k // 2
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


# ---- gather ----
faith_eval = parse_evals(FAITH / "qwen3_baseline2tpp_train.log")
imu_eval = parse_evals(MOD / "qwen3_imu1_2tpp_train.log")
faith_loss = parse_csv_loss(FAITH / "qwen3_baseline2tpp_train.csv")
imu_ce = parse_log_ce(MOD / "qwen3_imu1_2tpp_train.log")

print("faithful eval points:", faith_eval)
print("imu eval points:", imu_eval)
print(f"faithful CSV rows: {len(faith_loss)} (final step {faith_loss[-1][0]})")
print(f"imu log CE points: {len(imu_ce)} (final step {imu_ce[-1][0]})")

# ---- Figure 1: validation perplexity vs tokens (the headline) ----
fig, ax = plt.subplots(figsize=(5.0, 3.4))
fx = [s * TOK_PER_STEP / 1e9 for s, _ in faith_eval]
fy = [p for _, p in faith_eval]
ix = [s * TOK_PER_STEP / 1e9 for s, _ in imu_eval]
iy = [p for _, p in imu_eval]
ax.plot(fx, fy, "o-", color=FAITH_C, ms=3.5, lw=1.3, label="Faithful baseline (AdamW + cosine)")
ax.plot(ix, iy, "s-", color=IMU_C, ms=3.5, lw=1.3, label="Modernized (IMU-1 bundle + NorMuon, WSD)")
ax.set_yscale("log")
ax.set_xlabel("Training tokens (billions)")
ax.set_ylabel("Validation perplexity (log scale)")
ax.set_title("Validation perplexity at matched compute")
# annotate the matched-compute endpoints
ax.annotate(f"{fy[-1]:.2f}", (fx[-1], fy[-1]), textcoords="offset points",
            xytext=(4, 6), color=FAITH_C, fontsize=8)
ax.annotate(f"{iy[-1]:.2f}", (ix[-1], iy[-1]), textcoords="offset points",
            xytext=(4, -10), color=IMU_C, fontsize=8)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig1_val_ppl.png")
fig.savefig(OUT / "fig1_val_ppl.pdf")
plt.close(fig)

# ---- Figure 2: training cross-entropy vs step (dynamics) ----
fig, ax = plt.subplots(figsize=(5.0, 3.4))
fs = [s for s, _ in faith_loss]
fl = smooth([l for _, l in faith_loss], 101)
isx = [s for s, _ in imu_ce]
il = smooth([c for _, c in imu_ce], 11)
ax.plot(fs, fl, color=FAITH_C, lw=1.0, label="Faithful baseline (per-step, smoothed)")
ax.plot(isx, il, color=IMU_C, lw=1.2, label="Modernized (every 50 steps, smoothed)")
ax.set_xlabel("Optimizer step")
ax.set_ylabel("Training cross-entropy (nats)")
ax.set_title("Training loss")
ax.set_ylim(2.5, 6.0)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig2_train_loss.png")
fig.savefig(OUT / "fig2_train_loss.pdf")
plt.close(fig)

print("\nWROTE:", sorted(p.name for p in OUT.glob("fig*.p*")))
print("ENDPOINTS for cross-check -> faithful final eval %.2f @ %.3fB tok ; imu final eval %.2f @ %.3fB tok"
      % (fy[-1], fx[-1], iy[-1], ix[-1]))
