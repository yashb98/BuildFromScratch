#!/usr/bin/env python3
"""make_phase_plots.py — regenerate the Phase-A / Phase-B result figures straight
from the training logs, so every number in a plot traces to an on-disk log line
(brutal-accuracy rule). CPU-only, matplotlib, no model load.

Outputs into builds/comparison/:
  phaseB_ppl_curves.png   val-PPL vs step, all four Phase-B runs + original line
  phaseB_final_ppl.png    final/latest val-PPL bar (original + 4 runs)
  phaseA_lr_sweep.png     Phase-A LR sweep final PPL (lr17/lr24/lr30)
Prints the parsed series so the values can be eyeballed against the logs.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
B = ROOT / "builds"
OUT = B / "comparison"
OUT.mkdir(exist_ok=True)

FA = B / "2026-06-08_reproduce-faithful_qwen3-0.6b/results"
MO = B / "2026-06-08_reproduce-modernized_qwen3-0.6b/results"
EX = B / "2026-06-08_reproduce-exploratory_qwen3-0.6b/results"
RUNS = [
    # (name, train log, after.txt canonical-final or None, colour)
    ("faithful baseline", FA / "qwen3_baseline2tpp_train.log", FA / "qwen3_baseline2tpp_after.txt", "#1f77b4"),
    ("IMU-1 (NorMuon bundle)", MO / "qwen3_imu1_2tpp_train.log", None, "#2ca02c"),
    ("partial-RoPE 0.25", EX / "qwen3_prope25_2tpp_train.log", EX / "qwen3_prope25_2tpp_after.txt", "#ff7f0e"),
    ("partial-RoPE 0.10", EX / "qwen3_prope10_2tpp_train.log", None, "#d62728"),
]
ORIGINAL_PPL = 13.40  # Qwen3-0.6B-Base, 36T tok — from results/original_vs_repro.txt

EVAL_RE = re.compile(r"\[eval @ (\d+)\] val PPL=([\d.]+)")
FINAL_RE = re.compile(r"final val PPL=([\d.]+)")
AFTER_RE = re.compile(r"->\s*([\d.]+)")


def parse(log: Path, after: Path = None):
    """Return (steps, ppls, final_ppl, done). Prefer the after.txt canonical
    final (the post-run eval the READMEs quote); else a DONE line; else the last
    in-loop eval. done = a canonical/DONE result exists (not a mid-run snapshot)."""
    if not log.exists():
        return [], [], None, False
    steps, ppls = [], []
    text = log.read_text()
    for line in text.splitlines():
        m = EVAL_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            ppls.append(float(m.group(2)))
    if after and after.exists():
        final = float(AFTER_RE.search(after.read_text()).group(1))
        return steps, ppls, final, True
    fm = FINAL_RE.search(text)
    final = float(fm.group(1)) if fm else (ppls[-1] if ppls else None)
    done = (fm is not None) or ("DONE" in text)
    return steps, ppls, final, done


def parse_lr_sweep():
    f = B / "2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_vs_repro.txt"
    out = {}
    for line in f.read_text().splitlines():
        m = re.search(r"lr(\d+) .*val PPL =\s*([\d.]+)", line)
        if m:
            out[f"lr{m.group(1)}"] = float(m.group(2))
    return out


# -------- parse + report (verification) --------
parsed = {}
print("=== Phase-B parsed series (verify vs logs) ===")
for name, log, after, color in RUNS:
    steps, ppls, final, done = parse(log, after)
    parsed[name] = (steps, ppls, final, done, color)
    tag = "DONE" if done else f"in-progress @ step {steps[-1] if steps else '?'}"
    print(f"  {name:24s} final/latest PPL={final}  ({len(ppls)} evals, {tag})")
lr = parse_lr_sweep()
print(f"=== Phase-A LR sweep: {lr}")

# -------- Fig 1: PPL curves --------
plt.figure(figsize=(9, 5.5))
for name, (steps, ppls, final, done, color) in parsed.items():
    if not steps:
        continue
    label = f"{name} → {final:.2f}" + ("" if done else " (in progress)")
    plt.plot(steps, ppls, marker="o", ms=3, color=color, label=label,
             linestyle="-" if done else "--")
plt.axhline(ORIGINAL_PPL, color="black", ls=":", lw=1.2,
            label=f"original Qwen3-0.6B (36T tok) = {ORIGINAL_PPL}")
plt.yscale("log")
plt.xlabel("training step (18,150 = 2 tokens/param)")
plt.ylabel("held-out val PPL (log scale)")
plt.title("Qwen3-0.6B Phase B — matched-compute val PPL (same data, eval, budget)")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "phaseB_ppl_curves.png", dpi=130)
plt.close()

# -------- Fig 2: final PPL bar --------
bars = [("original\n(36T tok)", ORIGINAL_PPL, "#7f7f7f")]
for name, (steps, ppls, final, done, color) in parsed.items():
    if final is not None:
        lbl = name.replace(" (", "\n(") + ("" if done else "\n(in prog.)")
        bars.append((lbl, final, color))
plt.figure(figsize=(9, 5.5))
xs = range(len(bars))
plt.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars])
for x, (_, v, _) in zip(xs, bars):
    plt.text(x, v + 0.6, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
plt.xticks(list(xs), [b[0] for b in bars], fontsize=8)
plt.ylabel("val PPL (lower = better)")
plt.title("Qwen3-0.6B — final/latest val PPL (IMU-1 wins; partial-RoPE loses to baseline)")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "phaseB_final_ppl.png", dpi=130)
plt.close()

# -------- Fig 3: Phase-A LR sweep --------
if lr:
    order = sorted(lr.items(), key=lambda kv: kv[0])
    plt.figure(figsize=(6, 4.5))
    names = [k for k, _ in order]
    vals = [v for _, v in order]
    best = min(range(len(vals)), key=lambda i: vals[i])
    colors = ["#2ca02c" if i == best else "#1f77b4" for i in range(len(vals))]
    plt.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    plt.ylabel("val PPL @ 131M tok")
    plt.title("Phase A — LR sweep (lr24 = 2.4e-3 wins)")
    plt.ylim(min(vals) - 1, max(vals) + 1.5)
    plt.tight_layout()
    plt.savefig(OUT / "phaseA_lr_sweep.png", dpi=130)
    plt.close()

print(f"\nWrote: {sorted(p.name for p in OUT.glob('phase*.png'))}")
