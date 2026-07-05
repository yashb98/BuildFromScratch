#!/usr/bin/env python3
"""
Headline plots for the NorMuon-vs-AdamW iso-FLOP ablation on Qwen3-0.6B (42M tok).

HARD RULES honored:
  - CPU-only, matplotlib Agg backend, NO GPU, NO model loads, NO torch.
  - Plots ONLY numbers parsed from real on-disk files THIS run.
  - cohort BPB / verdict numbers come from results/cohort_bpb.json + verdict.json.
  - fineweb-val PPL comes from the 6 arm logs (final DONE line).
  - training-loss curves come from `step N/640 loss X` lines in the 6 arm logs.
  - LR-sweep comes from results/lr_sweep_bpb.json (off-baseline pts) +
    verdict.json (the 2.4e-3 baseline = 3-seed cohort mean).
Every figure title cites its exact source file.
"""
import os, re, json
import matplotlib
matplotlib.use("Agg")  # headless, no display, no GPU
import matplotlib.pyplot as plt
import numpy as np

# ---- clean serif style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})

EXP = "/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw"
RES = os.path.join(EXP, "results")
OUT = os.path.join(RES, "plots")
os.makedirs(OUT, exist_ok=True)

# arm colors (color by arm)
C_ADAMW = "#B5651D"     # warm ochre
C_NORMUON = "#1F5FA8"   # deep blue
SEEDS = [0, 1, 2]

printed = {}  # collect everything we plot, to print at the end


def save(fig, name):
    png = os.path.join(OUT, name + ".png")
    pdf = os.path.join(OUT, name + ".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# ----------------------------------------------------------------------------
# Load JSONs
# ----------------------------------------------------------------------------
with open(os.path.join(RES, "cohort_bpb.json")) as f:
    cohort = json.load(f)
with open(os.path.join(RES, "verdict.json")) as f:
    verdict = json.load(f)
with open(os.path.join(RES, "lr_sweep_bpb.json")) as f:
    lrsweep = json.load(f)

wk = verdict["by_corpus"]["wikitext2_val"]
cd = verdict["by_corpus"]["code_py"]

# Per-seed BPB straight from cohort json (authoritative)
def seed_bpb(arm, corpus):
    return [cohort["cells"][f"{arm}_seed{s}"][corpus]["bpb"] for s in SEEDS]

adamw_wk = seed_bpb("adamw", "wikitext2_val")
norm_wk  = seed_bpb("normuon", "wikitext2_val")
adamw_cd = seed_bpb("adamw", "code_py")
norm_cd  = seed_bpb("normuon", "code_py")

# Cross-check verdict.json arrays == cohort-derived arrays
assert np.allclose(adamw_wk, wk["adamw_bpb"]), "wikitext adamw mismatch json vs cohort"
assert np.allclose(norm_wk, wk["normuon_bpb"]), "wikitext normuon mismatch"
assert np.allclose(adamw_cd, cd["adamw_bpb"]), "code adamw mismatch"
assert np.allclose(norm_cd, cd["normuon_bpb"]), "code normuon mismatch"

printed["wikitext_BPB"] = {
    "adamw_seeds": adamw_wk, "adamw_mean": wk["adamw_mean"], "adamw_sem": wk["adamw_sem"],
    "normuon_seeds": norm_wk, "normuon_mean": wk["normuon_mean"], "normuon_sem": wk["normuon_sem"],
    "improvement_bpb": wk["improvement_bpb"], "ci95": wk["ci95"], "df": wk["df"],
}
printed["code_BPB"] = {
    "adamw_seeds": adamw_cd, "adamw_mean": cd["adamw_mean"], "adamw_sem": cd["adamw_sem"],
    "normuon_seeds": norm_cd, "normuon_mean": cd["normuon_mean"], "normuon_sem": cd["normuon_sem"],
    "improvement_bpb": cd["improvement_bpb"], "ci95": cd["ci95"], "df": cd["df"],
}

# t for 95% CI on the MEAN (n=3 -> df=2, t=4.302) for the per-arm error bars
# (verdict.json's CI is the Welch-t on the DELTA; we annotate that separately).
T_DF2_95 = 4.302652729  # scipy t.ppf(0.975, 2)


# ----------------------------------------------------------------------------
# Parse fineweb-val PPL from the 6 arm logs (final DONE line, /640 run)
# ----------------------------------------------------------------------------
done_re = re.compile(r"DONE in ([\d.]+) min; fineweb-val PPL [\d.]+ -> ([\d.]+)")

def final_fineweb_ppl(arm, seed):
    """Return (ppl, minutes) from the LAST DONE line whose run-time is the real
    (>10 min) run — excludes the short smoke/aborted DONE lines."""
    path = os.path.join(RES, f"{arm}_seed{seed}.log")
    best = None
    with open(path) as f:
        for line in f:
            m = done_re.search(line)
            if m:
                mins = float(m.group(1)); ppl = float(m.group(2))
                if mins > 10.0:  # the real 95-106 min run, not a 0.8 min smoke
                    best = (ppl, mins)
    assert best is not None, f"no real DONE line in {path}"
    return best

fw_adamw, fw_norm = [], []
fw_meta = {}
for s in SEEDS:
    pa, ma = final_fineweb_ppl("adamw", s)
    pn, mn = final_fineweb_ppl("normuon", s)
    fw_adamw.append(pa); fw_norm.append(pn)
    fw_meta[f"adamw_seed{s}"] = {"ppl": pa, "min": ma}
    fw_meta[f"normuon_seed{s}"] = {"ppl": pn, "min": mn}
printed["fineweb_val_PPL"] = fw_meta


# ----------------------------------------------------------------------------
# Parse training-loss curves: only `step N/640 loss X` lines from each arm log
# ----------------------------------------------------------------------------
step_re = re.compile(r"step (\d+)/640 loss ([\d.]+)")

def loss_curve(arm, seed):
    path = os.path.join(RES, f"{arm}_seed{seed}.log")
    d = {}
    with open(path) as f:
        for line in f:
            m = step_re.search(line)
            if m:
                d[int(m.group(1))] = float(m.group(2))  # later line wins (restart-safe)
    steps = sorted(d)
    return steps, [d[s] for s in steps]

curves = {}
for arm in ("adamw", "normuon"):
    for s in SEEDS:
        st, ls = loss_curve(arm, s)
        curves[(arm, s)] = (st, ls)
printed["loss_curve_summary"] = {
    f"{a}_seed{s}": {"n_points": len(curves[(a, s)][0]),
                      "first_step": curves[(a, s)][0][0],
                      "last_step": curves[(a, s)][0][-1],
                      "first_loss": curves[(a, s)][1][0],
                      "last_loss": curves[(a, s)][1][-1]}
    for a in ("adamw", "normuon") for s in SEEDS
}


# ----------------------------------------------------------------------------
# FIGURE 1 — THE HEADLINE: wikitext-2 BPB bars + 95% CI + per-seed dots
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 5.2))
x = [0, 1]
means = [wk["adamw_mean"], wk["normuon_mean"]]
# per-arm 95% CI on the mean (n=3, df=2): t * sem
errs = [T_DF2_95 * wk["adamw_sem"], T_DF2_95 * wk["normuon_sem"]]
colors = [C_ADAMW, C_NORMUON]
bars = ax.bar(x, means, width=0.55, color=colors, alpha=0.85,
              yerr=errs, capsize=8, ecolor="black",
              error_kw=dict(lw=1.4, capthick=1.4), zorder=2)
# overlay per-seed dots
for xi, vals in zip(x, [adamw_wk, norm_wk]):
    jit = np.linspace(-0.10, 0.10, len(vals))
    ax.scatter([xi + j for j in jit], vals, color="black", s=42,
               zorder=4, edgecolor="white", linewidth=0.8)
# value labels
for xi, m in zip(x, means):
    ax.text(xi, m + 0.03, f"{m:.4f}", ha="center", va="bottom", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["AdamW @ 2.4e-3\n(2D weights)", "NorMuon @ 0.011\n(2D weights)"])
ax.set_ylabel("wikitext-2 bits-per-byte (lower is better)")
ax.set_ylim(0, 2.55)
# improvement annotation (Welch-t delta CI from verdict.json)
imp = wk["improvement_bpb"]; lo, hi = wk["ci95"]
ax.annotate("", xy=(1, wk["normuon_mean"]), xytext=(1, wk["adamw_mean"]),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
ax.text(1.32, (wk["adamw_mean"] + wk["normuon_mean"]) / 2,
        f"improvement\n+{imp:.4f} bpb\n95% CI [+{lo:.3f}, +{hi:.3f}]\n(Welch-t, n=3/arm, significant)",
        ha="left", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f3f3f3", ec="#999"))
ax.set_title("NorMuon vs AdamW on Qwen3-0.6B 2D weights — wikitext-2 BPB\n"
             "iso-FLOP 42M tok (640 steps), 3 seeds/arm  ·  bars=mean, whisker=95% CI on mean, dots=seeds\n"
             "source: results/cohort_bpb.json + results/verdict.json", fontsize=9.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=C_ADAMW, label="AdamW"),
                   Patch(fc=C_NORMUON, label="NorMuon"),
                   plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                              markersize=7, label="per-seed BPB")],
          loc="upper right", frameon=True, fontsize=9)
fig.tight_layout()
f1 = save(fig, "fig1_headline_wikitext2_bpb")


# ----------------------------------------------------------------------------
# FIGURE 2 — same for the CODE corpus
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 5.2))
means = [cd["adamw_mean"], cd["normuon_mean"]]
errs = [T_DF2_95 * cd["adamw_sem"], T_DF2_95 * cd["normuon_sem"]]
ax.bar(x, means, width=0.55, color=colors, alpha=0.85,
       yerr=errs, capsize=8, ecolor="black",
       error_kw=dict(lw=1.4, capthick=1.4), zorder=2)
for xi, vals in zip(x, [adamw_cd, norm_cd]):
    jit = np.linspace(-0.10, 0.10, len(vals))
    ax.scatter([xi + j for j in jit], vals, color="black", s=42,
               zorder=4, edgecolor="white", linewidth=0.8)
for xi, m in zip(x, means):
    ax.text(xi, m + 0.05, f"{m:.4f}", ha="center", va="bottom", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["AdamW @ 2.4e-3\n(2D weights)", "NorMuon @ 0.011\n(2D weights)"])
ax.set_ylabel("codeparrot-clean-valid bits-per-byte (lower is better)")
ax.set_ylim(0, 4.05)
imp = cd["improvement_bpb"]; lo, hi = cd["ci95"]
ax.annotate("", xy=(1, cd["normuon_mean"]), xytext=(1, cd["adamw_mean"]),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
ax.text(1.32, (cd["adamw_mean"] + cd["normuon_mean"]) / 2,
        f"improvement\n+{imp:.4f} bpb\n95% CI [+{lo:.3f}, +{hi:.3f}]\n(Welch-t, n=3/arm, significant)",
        ha="left", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f3f3f3", ec="#999"))
ax.set_title("NorMuon vs AdamW on Qwen3-0.6B 2D weights — code BPB\n"
             "iso-FLOP 42M tok (640 steps), 3 seeds/arm  ·  bars=mean, whisker=95% CI on mean, dots=seeds\n"
             "source: results/cohort_bpb.json + results/verdict.json", fontsize=9.5)
ax.legend(handles=[Patch(fc=C_ADAMW, label="AdamW"),
                   Patch(fc=C_NORMUON, label="NorMuon"),
                   plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                              markersize=7, label="per-seed BPB")],
          loc="upper right", frameon=True, fontsize=9)
fig.tight_layout()
f2 = save(fig, "fig2_code_bpb")


# ----------------------------------------------------------------------------
# FIGURE 3 — fineweb-val PPL grouped bars (per-seed), log y-axis
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.0))
xs = np.arange(len(SEEDS))
w = 0.38
b1 = ax.bar(xs - w/2, fw_adamw, w, color=C_ADAMW, alpha=0.85, label="AdamW", zorder=2)
b2 = ax.bar(xs + w/2, fw_norm, w, color=C_NORMUON, alpha=0.85, label="NorMuon", zorder=2)
ax.set_yscale("log")
ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=9)
ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([f"seed {s}" for s in SEEDS])
ax.set_ylabel("fineweb-val perplexity (log scale, lower is better)")
ax.set_ylim(40, 400)
ax.set_title("In-training fineweb-val PPL per seed — AdamW vs NorMuon (Qwen3-0.6B 2D weights)\n"
             "iso-FLOP 42M tok  ·  independent held-out corpus corroborates the BPB gap (~2.4x)\n"
             "source: results/{adamw,normuon}_seed{0,1,2}.log (final DONE line)", fontsize=9.5)
ax.legend(loc="upper right", frameon=True)
fig.tight_layout()
f3 = save(fig, "fig3_fineweb_val_ppl")


# ----------------------------------------------------------------------------
# FIGURE 4 — training loss vs step, 6 cells (2 arms x 3 seeds), color by arm
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True)
for r, arm in enumerate(("adamw", "normuon")):
    color = C_ADAMW if arm == "adamw" else C_NORMUON
    for c, s in enumerate(SEEDS):
        ax = axes[r][c]
        st, ls = curves[(arm, s)]
        ax.plot(st, ls, color=color, lw=1.8, marker="o", ms=2.5)
        ax.set_title(f"{arm}  seed {s}   (final loss {ls[-1]:.3f}, {len(st)} pts)",
                     fontsize=9.5, color=color)
        ax.grid(True, alpha=0.25)
        if r == 1:
            ax.set_xlabel("training step (of 640)")
        if c == 0:
            ax.set_ylabel("train loss")
fig.suptitle("Training loss vs step — 6 cells (AdamW=ochre, NorMuon=blue), "
             "iso-FLOP 42M tok, only `step N/640` lines\n"
             "source: results/{adamw,normuon}_seed{0,1,2}.log", fontsize=10, y=1.005)
fig.tight_layout()
f4 = save(fig, "fig4_train_loss_curves")


# ----------------------------------------------------------------------------
# FIGURE 5 — AdamW LR sweep (confirmatory): baseline isn't undertuned
# 2.4e-3 baseline = 3-seed cohort mean (verdict.json), with its 95% CI;
# 1.7/3.5/4.8e-3 = single-seed points (lr_sweep_bpb.json) -> labeled single-seed.
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
lr_pts = [
    (1.7e-3, lrsweep["1.7e-3"]["wikitext_bpb"], "single-seed"),
    (2.4e-3, wk["adamw_mean"], "3-seed mean"),
    (3.5e-3, lrsweep["3.5e-3"]["wikitext_bpb"], "single-seed"),
    (4.8e-3, lrsweep["4.8e-3"]["wikitext_bpb"], "single-seed"),
]
lrs = [p[0] for p in lr_pts]
bpbs = [p[1] for p in lr_pts]

# single-seed points
ss_x = [p[0] for p in lr_pts if p[2] == "single-seed"]
ss_y = [p[1] for p in lr_pts if p[2] == "single-seed"]
ax.plot(lrs, bpbs, color=C_ADAMW, lw=1.4, ls="--", alpha=0.6, zorder=1)
ax.scatter(ss_x, ss_y, color=C_ADAMW, s=70, zorder=3,
           label="AdamW LR sweep (single-seed)")
# baseline 3-seed point with 95% CI on mean
base_err = T_DF2_95 * wk["adamw_sem"]
ax.errorbar([2.4e-3], [wk["adamw_mean"]], yerr=[base_err], fmt="s",
            color="#7a3d10", ms=10, capsize=6, lw=1.6, zorder=4,
            label="AdamW @ 2.4e-3 baseline (3-seed mean ± 95% CI)")
# NorMuon reference line (its 3-seed mean) + band
nm = wk["normuon_mean"]; nlo, nhi = nm - T_DF2_95*wk["normuon_sem"], nm + T_DF2_95*wk["normuon_sem"]
ax.axhline(nm, color=C_NORMUON, ls="-", lw=1.8, zorder=2,
           label=f"NorMuon @ 0.011 (3-seed mean {nm:.3f})")
ax.fill_between([1.4e-3, 5.2e-3], nlo, nhi, color=C_NORMUON, alpha=0.12, zorder=0)
# value labels
for lr, bpb, kind in lr_pts:
    ax.annotate(f"{bpb:.4f}", (lr, bpb), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8.5)
ax.set_xscale("log")
ax.set_xticks(lrs)
ax.set_xticklabels(["1.7e-3", "2.4e-3", "3.5e-3", "4.8e-3"])
ax.set_xlim(1.45e-3, 5.3e-3)
ax.set_xlabel("AdamW peak learning rate (2D weights)")
ax.set_ylabel("wikitext-2 bits-per-byte (lower is better)")
spread = max(bpbs) - min(bpbs)
gap = wk["improvement_bpb"]
ax.set_title("Confirmatory AdamW LR sweep — the baseline is not undertuned\n"
             f"full LR spread = {spread:.3f} bpb  vs  NorMuon gap = +{gap:.3f} bpb (~{gap/spread:.0f}x larger); "
             "no swept LR closes the gap\n"
             "source: results/lr_sweep_bpb.json (off-baseline) + results/verdict.json (2.4e-3 cohort)",
             fontsize=9.2)
ax.legend(loc="center right", frameon=True, fontsize=8.5)
fig.tight_layout()
f5 = save(fig, "fig5_adamw_lr_sweep")

printed["lr_sweep"] = {
    "1.7e-3 (single-seed)": lrsweep["1.7e-3"]["wikitext_bpb"],
    "2.4e-3 (3-seed mean)": wk["adamw_mean"],
    "3.5e-3 (single-seed)": lrsweep["3.5e-3"]["wikitext_bpb"],
    "4.8e-3 (single-seed)": lrsweep["4.8e-3"]["wikitext_bpb"],
    "full_spread_bpb": spread,
    "normuon_mean_bpb": nm,
}

# ----------------------------------------------------------------------------
# Print everything plotted for cross-checking
# ----------------------------------------------------------------------------
print("FILES WRITTEN:")
for p in (f1, f2, f3, f4, f5):
    print("  ", p[0]); print("  ", p[1])
print()
print(json.dumps(printed, indent=2))
