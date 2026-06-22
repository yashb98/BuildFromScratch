#!/usr/bin/env python3
"""Step-aligned cross-build comparison for the Qwen3-0.6B three-build experiment.

CPU-only (matplotlib Agg), no GPU, no torch. Reads the four 2-TPP training logs
and overlays them on ONE axis so the builds are compared at the SAME steps /
matched compute (18,150 steps x 65,536 tok = 1.19B tokens). Only numbers parsed
from the logs this run are plotted; the incomplete partial-RoPE-0.10 run (died
at step 5450) is drawn dashed and labelled. Prints the extracted endpoints for
cross-checking. Re-run any time.
"""
import re, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path("/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds")
OUT = B / "comparison"
TPS = 65_536
EVAL = re.compile(r"\[eval @ (\d+)\] val PPL=([\d.]+)")

# (label, log, color, style, train-loss field, complete?)
BUILDS = [
    ("Faithful baseline (AdamW + cosine)", B/"2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.log", "#444444", "o-", "csv", True),
    ("Modernized (IMU-1 bundle + NorMuon, WSD)", B/"2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log", "#1f6fb4", "s-", "ce", True),
    ("Exploratory partial-RoPE 25%", B/"2026-06-08_reproduce-exploratory_qwen3-0.6b/results/qwen3_prope25_2tpp_train.log", "#b03030", "^-", "loss", True),
    ("Exploratory partial-RoPE 10% (died @5450, partial)", B/"2026-06-08_reproduce-exploratory_qwen3-0.6b/results/qwen3_prope10_2tpp_train.log", "#d08010", "x--", "loss", False),
]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linestyle": "--", "figure.dpi": 300})


def evals(log):
    out = []
    for ln in Path(log).read_text().splitlines():
        m = EVAL.search(ln)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def train_loss(log, field):
    pts = []
    if field == "csv":
        csvp = Path(str(log).replace("_train.log", "_train.csv"))
        if csvp.exists():
            with open(csvp) as f:
                for r in csv.DictReader(f):
                    pts.append((int(r["step"]), float(r["loss"])))
            return pts
        field = "loss"
    pat = re.compile(rf"step\s+(\d+)/\d+\s+{field}\s+([\d.]+)")
    for ln in Path(log).read_text().splitlines():
        m = pat.search(ln)
        if m:
            pts.append((int(m.group(1)), float(m.group(2))))
    return pts


def smooth(ys, k=101):
    if len(ys) < k:
        return ys
    h = k // 2
    return [sum(ys[max(0, i-h):min(len(ys), i+h+1)]) / (min(len(ys), i+h+1)-max(0, i-h)) for i in range(len(ys))]


# ---- Figure 1: eval val-PPL vs step (the matched-compute comparison) ----
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for label, log, c, st, _f, complete in BUILDS:
    ev = evals(log)
    if not ev:
        continue
    xs = [s for s, _ in ev]; ys = [p for _, p in ev]
    ax.plot(xs, ys, st, color=c, ms=4, lw=1.4, label=label)
    ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(5, 0), color=c, fontsize=8, va="center")
    print(f"{label[:26]:26s} eval: {ev}")
ax.set_yscale("log")
ax.set_xlabel("training step  (of 18,150;  x 65,536 = tokens, matched 1.19B-token budget)")
ax.set_ylabel("held-out FineWeb-Edu val perplexity (log scale)")
ax.set_title("Three-build comparison at matched compute: validation perplexity vs step")
ax.legend(frameon=True, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT/"comparison_all_builds_ppl_vs_step.png")
fig.savefig(OUT/"comparison_all_builds_ppl_vs_step.pdf")
plt.close(fig)

# ---- Figure 2: training loss vs step (all builds) ----
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for label, log, c, st, field, complete in BUILDS:
    pts = train_loss(log, field)
    if not pts:
        continue
    xs = [s for s, _ in pts]; ys = smooth([l for _, l in pts], 101 if len(pts) > 300 else 11)
    ax.plot(xs, ys, "--" if not complete else "-", color=c, lw=1.1, label=label)
ax.set_xlabel("training step  (of 18,150)")
ax.set_ylabel("training cross-entropy (nats, smoothed)")
ax.set_ylim(2.4, 5.0)
ax.set_title("Three-build comparison: training loss vs step")
ax.legend(frameon=True, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT/"comparison_all_builds_train_loss.png")
fig.savefig(OUT/"comparison_all_builds_train_loss.pdf")
plt.close(fig)

print("\nWROTE:", *sorted(p.name for p in OUT.glob("comparison_all_builds_*.p*")))
