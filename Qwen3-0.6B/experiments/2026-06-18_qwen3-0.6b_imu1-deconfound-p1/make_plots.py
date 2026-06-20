#!/usr/bin/env python3
"""Plots for the IMU-1 de-confound cohort + comparison to the earlier Phase-B builds.
CPU only, reads the run logs (in-loop val PPL @ 500/1000/1500/2000 per cell) and the
build result CSVs. Regenerate any time — handles incomplete arms (arch still running).

Outputs (in this dir):
  deconfound_ppl_curves.png   per-arm in-loop val-PPL vs step (mean over seeds, range band)
  deconfound_attribution.png  per-axis final PPL: which single component drives the gain
  deconfound_vs_builds.png    de-confound (131M proxy) beside Phase-B builds (1.19B full)
HONEST: in-loop val PPL (the trainer's quick eval), NOT the canonical eval-harness BPB.
The de-confound budget (2000 steps = 131M tok) equals Phase A, not Phase B's 1.19B.
"""
import re, sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "research"))
from eval_stats import seed_delta_significant, mean_std_sem  # noqa: E402

ARMS = ["baseline", "wsd", "zloss", "arch"]
LABEL = {"baseline": "baseline (faithful)", "wsd": "+WSD", "zloss": "+z-loss", "arch": "+arch (model_imu1)"}
COLOR = {"baseline": "#444", "wsd": "#1f77b4", "zloss": "#d62728", "arch": "#2ca02c"}
STEPS = [500, 1000, 1500, 2000]


def curve(cell):
    f = HERE / f"run_{cell}.log"
    if not f.exists():
        return {}
    pts = dict(re.findall(r"eval @ (\d+)\] val PPL=([0-9.]+)", f.read_text()))
    return {int(s): float(p) for s, p in pts.items()}


def final(cell):
    c = curve(cell)
    return c.get(2000) if (HERE / f"arm_{cell}.done").exists() and 2000 in c else None


def arm_curves(arm):
    return [curve(f"{arm}_seed{s}") for s in (0, 1, 2)]


def arm_finals(arm):
    return [v for v in (final(f"{arm}_seed{s}") for s in (0, 1, 2)) if v is not None]


# ---- Fig 1: per-arm val-PPL vs step (mean over seeds) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for arm in ARMS:
    cs = arm_curves(arm)
    xs = [s for s in STEPS if all(s in c for c in cs)]
    if not xs:
        continue
    means = [sum(c[s] for c in cs) / len(cs) for s in xs]
    lo = [min(c[s] for c in cs) for s in xs]
    hi = [max(c[s] for c in cs) for s in xs]
    ax.plot(xs, means, "-o", color=COLOR[arm], label=LABEL[arm], lw=2)
    ax.fill_between(xs, lo, hi, color=COLOR[arm], alpha=0.15)
ax.set(xlabel="step (of 2000; 65,536 tok/step)", ylabel="in-loop val PPL (FineWeb-Edu)",
       title="IMU-1 de-confound — per-arm convergence (3-seed mean, range band)")
ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE / "deconfound_ppl_curves.png", dpi=130); plt.close(fig)

# ---- Fig 2: per-axis attribution (final PPL bars + seed dots + CI) ----
base = arm_finals("baseline")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xpos = range(len(ARMS))
for i, arm in enumerate(ARMS):
    vals = arm_finals(arm)
    if not vals:
        ax.text(i, base and mean_std_sem(base)[0] or 46, "running…", ha="center", color="gray"); continue
    m = mean_std_sem(vals)[0]
    ax.bar(i, m, color=COLOR[arm], alpha=0.55, width=0.6)
    ax.scatter([i] * len(vals), vals, color=COLOR[arm], zorder=3, s=30, edgecolor="k", lw=0.5)
    if arm != "baseline" and base and len(vals) == 3:
        r = seed_delta_significant(base, vals, direction="lower_is_better")
        tag = "DRIVER" if (r["significant"] and r["improvement"] > 0) else ("hurts" if r["significant"] else "n.s.")
        ax.annotate(f"Δ{r['improvement']:+.2f}\nCI[{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}]\n{tag}",
                    (i, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8,
                    color=("green" if tag == "DRIVER" else "gray"))
if base:
    ax.axhline(mean_std_sem(base)[0], color="#444", ls="--", lw=1, alpha=0.6)
ax.set_xticks(list(xpos)); ax.set_xticklabels([LABEL[a] for a in ARMS], rotation=12, fontsize=9)
ax.set(ylabel="final in-loop val PPL (lower=better)",
       title="Which single IMU-1 component drives the gain? (131M tok, 3 seeds)")
ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
fig.savefig(HERE / "deconfound_attribution.png", dpi=130); plt.close(fig)

# ---- Fig 3: de-confound (131M proxy) vs Phase-B builds (1.19B full) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
# left: de-confound finals
for i, arm in enumerate(ARMS):
    vals = arm_finals(arm)
    if vals:
        a1.bar(i, mean_std_sem(vals)[0], color=COLOR[arm], alpha=0.6)
        a1.scatter([i] * len(vals), vals, color=COLOR[arm], s=25, zorder=3, edgecolor="k", lw=0.4)
    else:
        a1.bar(i, 0, color=COLOR[arm], alpha=0.2)
a1.set_xticks(range(len(ARMS))); a1.set_xticklabels([LABEL[a].split(" (")[0] for a in ARMS], rotation=12, fontsize=8)
a1.set(ylabel="val PPL", title="De-confound — single-variable\n(131M tok / 2 TPP-proxy, 3 seeds)")
a1.grid(alpha=0.3, axis="y")
# right: Phase-B builds (known finals from build READMEs/logs, full 1.19B)
builds = {"faithful": 28.65, "IMU-1 bundle": 23.52, "partial-RoPE .25": 29.54}
bc = {"faithful": "#444", "IMU-1 bundle": "#9467bd", "partial-RoPE .25": "#8c564b"}
a2.bar(range(len(builds)), list(builds.values()), color=[bc[k] for k in builds], alpha=0.7)
for i, (k, v) in enumerate(builds.items()):
    a2.text(i, v + 0.2, f"{v}", ha="center", fontsize=9)
a2.set_xticks(range(len(builds))); a2.set_xticklabels(list(builds), rotation=12, fontsize=8)
a2.set(ylabel="val PPL", title="Phase-B builds — full bundle\n(1.19B tok / 2 TPP, single run)")
a2.grid(alpha=0.3, axis="y")
fig.suptitle("IMU-1: the de-confound isolates WHICH component (left); the full builds show the bundle's total effect (right)\n"
             "different budgets — NOT directly overlaid; in-loop val PPL", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "deconfound_vs_builds.png", dpi=130); plt.close(fig)

print("wrote: deconfound_ppl_curves.png, deconfound_attribution.png, deconfound_vs_builds.png")
print("arms complete:", {a: len(arm_finals(a)) for a in ARMS})
