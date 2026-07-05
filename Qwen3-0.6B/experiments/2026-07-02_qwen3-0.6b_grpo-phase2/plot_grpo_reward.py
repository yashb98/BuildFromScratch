#!/usr/bin/env python3
"""GRPO reward-trajectory plot for the Qwen3-0.6B RLVR Phase-2 seed-0 run. Pure CPU (no torch).
Top: GRPO vs the random-reward control (shows the pipeline delivers reward when a signal exists).
Bottom: GRPO zoomed — the reward never trends up = nothing to sharpen at ~1% base accuracy (the null)."""
import json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

G = pathlib.Path(__file__).parent
load = lambda f: [json.loads(l) for l in open(G / f)]
g, r = load("health_grpo_seed0.jsonl"), load("health_random_seed0.jsonl")
gs, gr = [x["step"] for x in g], [x["mean_reward"] for x in g]
rs, rr = [x["step"] for x in r], [x["mean_reward"] for x in r]

def roll(y, w=20):
    y = np.asarray(y, float)
    return np.arange(w, len(y) + 1), np.convolve(y, np.ones(w) / w, mode="valid")
grx, grm = roll(gr)
g_mean = float(np.mean(gr))

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1]})

# top: contrast
ax0.plot(rs, rr, color="#7f8c8d", lw=1.4, ls="--", alpha=.85, label="random-reward control (~0.5)")
ax0.scatter(gs, gr, s=10, alpha=.35, color="#c0392b")
ax0.plot(grx, grm, color="#c0392b", lw=2.3, label="GRPO (20-step rolling mean)")
ax0.set_ylim(-.03, .62); ax0.set_ylabel("mean batch reward\n(fraction verifier-correct)")
ax0.set_title("GRPO reward trajectory — Qwen3-0.6B RLVR Phase-2 (seed 0, 300 steps)\n"
              "random rewards flow at ~0.5 → pipeline works; GRPO stays pinned near 0 → the predicted null",
              fontsize=11)
ax0.grid(alpha=.3); ax0.legend(loc="center right")
ax0.annotate("pipeline delivers reward\nwhen a signal exists", xy=(210, rr[209]), xytext=(150, .30),
             fontsize=9, color="#555", arrowprops=dict(arrowstyle="->", color="#999"))

# bottom: GRPO zoomed
ax1.scatter(gs, gr, s=10, alpha=.35, color="#c0392b", label="GRPO per-step reward")
ax1.plot(grx, grm, color="#c0392b", lw=2.3, label="GRPO 20-step rolling mean")
ax1.axhline(g_mean, color="#c0392b", ls=":", lw=1.2, alpha=.8)
ax1.set_ylim(-.004, .05); ax1.set_xlabel("GRPO update step"); ax1.set_ylabel("mean batch reward (zoom)")
ax1.grid(alpha=.3); ax1.legend(loc="upper left")
ax1.annotate(f"flat: mean={g_mean:.4f}, max={max(gr):.3f} over 300 steps\n"
             "no upward trend → no gradient signal to learn from\n"
             "(most rollout groups are all-wrong → zero advantage)",
             xy=(150, g_mean), xytext=(60, .032), fontsize=9, color="#c0392b",
             arrowprops=dict(arrowstyle="->", color="#c0392b"))

plt.tight_layout()
out = G / "grpo_reward_trajectory.png"
plt.savefig(out, dpi=130)
print("saved", out, "| GRPO mean %.4f max %.4f | random mean %.3f" % (g_mean, max(gr), float(np.mean(rr))))
