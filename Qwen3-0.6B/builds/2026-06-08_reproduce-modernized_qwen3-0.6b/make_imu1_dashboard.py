#!/usr/bin/env python3
"""make_imu1_dashboard.py — Build-2 (IMU-1 / NorMuon bundle) results dashboard,
parsed straight from the training log so every plotted point traces to a log line
(brutal-accuracy rule). CPU-only, matplotlib, no model load.

Panels: (1) val PPL vs step, IMU-1 vs the faithful baseline (the headline);
(2) CE training loss; (3) LR schedule (the WSD-to-zero decay); (4) grad-norm;
(5) z-loss; (6) throughput (tok/s). Panels 2-6 cover the steps present in the log.
Output: results/plots/qwen3_imu1_2tpp_dashboard.png
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
IMU1_LOG = HERE / "results/qwen3_imu1_2tpp_train.log"
BASE_LOG = HERE / "../2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.log"
OUT = HERE / "results/plots/qwen3_imu1_2tpp_dashboard.png"

STEP_RE = re.compile(
    r"step\s+(\d+)/18150\s+ce\s+([\d.]+)\s+z\s+([\d.]+)\s+lr\s+([\d.eE+-]+)"
    r"\s+\|grad\|\s+([\d.]+)\s+mem\s+([\d.]+)GB\s+tok/s\s+([\d,]+)")
EVAL_RE = re.compile(r"\[eval @ (\d+)\] val PPL=([\d.]+)")


def parse_steps(log):
    rows = {"step": [], "ce": [], "z": [], "lr": [], "grad": [], "mem": [], "toks": []}
    for line in log.read_text().splitlines():
        m = STEP_RE.search(line)
        if m:
            rows["step"].append(int(m.group(1)))
            rows["ce"].append(float(m.group(2)))
            rows["z"].append(float(m.group(3)))
            rows["lr"].append(float(m.group(4)))
            rows["grad"].append(float(m.group(5)))
            rows["mem"].append(float(m.group(6)))
            rows["toks"].append(int(m.group(7).replace(",", "")))
    return rows


def parse_evals(log):
    s, p = [], []
    for line in log.read_text().splitlines():
        m = EVAL_RE.search(line)
        if m:
            s.append(int(m.group(1)))
            p.append(float(m.group(2)))
    return s, p


r = parse_steps(IMU1_LOG)
ie_s, ie_p = parse_evals(IMU1_LOG)
be_s, be_p = parse_evals(BASE_LOG)
print(f"IMU-1: {len(r['step'])} metric rows (step {min(r['step'])}-{max(r['step'])}), "
      f"{len(ie_p)} evals, final PPL {ie_p[-1] if ie_p else '?'}")
print(f"baseline evals: {len(be_p)} (final {be_p[-1] if be_p else '?'})")

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Qwen3-0.6B Build 2 — IMU-1 (NorMuon bundle) @ 2 TPP — final val PPL 23.52 "
             "(beats the 28.65 baseline by 17.9%)", fontsize=13, fontweight="bold")

# 1) val PPL vs baseline (headline)
ax[0, 0].plot(be_s, be_p, "o-", ms=4, color="#1f77b4", label=f"faithful baseline → {be_p[-1]:.2f}")
ax[0, 0].plot(ie_s, ie_p, "o-", ms=4, color="#2ca02c", label=f"IMU-1 (NorMuon) → {ie_p[-1]:.2f}")
ax[0, 0].axhline(13.40, color="black", ls=":", lw=1, label="original (36T tok) = 13.40")
ax[0, 0].set_yscale("log"); ax[0, 0].set_title("val PPL — IMU-1 vs baseline (the result)")
ax[0, 0].set_xlabel("step"); ax[0, 0].set_ylabel("val PPL (log)")
ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True, which="both", alpha=0.3)

panels = [
    ((0, 1), "ce", "CE training loss", "#d62728", False),
    ((0, 2), "lr", "LR schedule (WSD decay to 0)", "#9467bd", False),
    ((1, 0), "grad", "grad-norm |grad|", "#8c564b", False),
    ((1, 1), "z", "z-loss (logit-norm reg.)", "#e377c2", False),
    ((1, 2), "toks", "throughput (tok/s)", "#17becf", False),
]
for (i, j), key, title, color, _ in panels:
    ax[i, j].plot(r["step"], r[key], color=color, lw=1.2)
    ax[i, j].set_title(title); ax[i, j].set_xlabel("step")
    ax[i, j].grid(True, alpha=0.3)
ax[0, 2].set_yscale("log")

fig.text(0.5, 0.005, f"metric panels cover the logged step range "
         f"{min(r['step'])}-{max(r['step'])}; PPL panel is the full 9-eval run. "
         f"Parsed from results/qwen3_imu1_2tpp_train.log.", ha="center", fontsize=8)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(OUT, dpi=120)
print(f"Wrote {OUT}")
