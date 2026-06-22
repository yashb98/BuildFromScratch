#!/usr/bin/env python3
"""Same-BUDGET (131M tok / step 2000) comparison: the de-confound arms vs the earlier
Phase-B builds — done HONESTLY, because the two are NOT on the same footing:

  * de-confound arms = COMPLETE 2000-step runs (LR schedule sized to 2000 -> fully
    decayed by step 2000).
  * build "eval @ 2000" = a MID-RUN snapshot of an 18,150-step run (LR still high,
    not decayed). So a build's step-2000 PPL is HIGHER than a complete short run.

Valid cross-check: de-confound baseline (complete) should match Phase-A faithful
(complete, 46.31). The build snapshots are shown in their OWN group with the caveat.
In-loop val PPL throughout (not eval-harness BPB). CPU only.
"""
import re, sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
BUILDS = HERE.parents[1] / "builds"
sys.path.insert(0, str(HERE.parents[2] / "research"))
from eval_stats import mean_std_sem  # noqa: E402

def dc_final(arm):
    vals = [v for v in ([_ppl(f"{arm}_seed{s}") for s in (0,1,2)]) if v]
    return mean_std_sem(vals)[0] if len(vals)==3 else (vals[0] if vals else None)
def _ppl(cell):
    f=HERE/f"run_{cell}.log"
    if not (HERE/f"arm_{cell}.done").exists() or not f.exists(): return None
    m=re.findall(r"eval @ 2000\] val PPL=([0-9.]+)", f.read_text()); return float(m[-1]) if m else None
def build_curve(glob):
    logs=list(BUILDS.glob(glob))
    if not logs: return {}
    pts=re.findall(r"eval @ (\d+)\] val PPL=([0-9.]+)", logs[0].read_text())
    return {int(s):float(p) for s,p in pts}

# de-confound complete-run finals (131M, LR decayed)
dc = {a: dc_final(a) for a in ("baseline","wsd","zloss","arch")}
PHASE_A_FAITHFUL = 46.31   # README: Phase A lr24 complete 131M run (the valid cross-check)

# build curves (full 18,150) + their step-2000 mid-run snapshots
bf = build_curve("*reproduce-faithful*/results/*baseline2tpp*train.log")
bm = build_curve("*reproduce-modernized*/results/*imu1_2tpp*train.log")

# ---- Fig A: build curves with the de-confound complete-run points overlaid at step 2000 ----
fig, ax = plt.subplots(figsize=(8,5))
if bf: ax.plot(sorted(bf), [bf[s] for s in sorted(bf)], "-", color="#444", lw=1.8, label="build: faithful (18,150-step run)")
if bm: ax.plot(sorted(bm), [bm[s] for s in sorted(bm)], "-", color="#9467bd", lw=1.8, label="build: full IMU-1 bundle (18,150-step run)")
ax.axvline(2000, color="gray", ls=":", lw=1)
# de-confound complete-run points at step 2000 (square markers = different condition)
mk={"baseline":"#444","wsd":"#1f77b4","zloss":"#d62728","arch":"#2ca02c"}
for a,v in dc.items():
    if v: ax.scatter(2000, v, marker="s", s=70, color=mk[a], edgecolor="k", zorder=5,
                     label=f"de-confound {a} (COMPLETE 2000-step): {v:.1f}")
ax.scatter(2000, PHASE_A_FAITHFUL, marker="*", s=180, color="#444", edgecolor="k", zorder=6,
           label=f"Phase-A faithful complete-run cross-check: {PHASE_A_FAITHFUL}")
ax.set(xlabel="step (65,536 tok/step)", ylabel="in-loop val PPL", yscale="log",
       title="Same-budget (131M tok) — de-confound COMPLETE runs (squares) vs build MID-RUN snapshots (lines @ dotted line)\n"
             "NOT the same LR state at step 2000: build LR still high, de-confound LR fully decayed")
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/"same_steps_curves.png", dpi=130); plt.close(fig)

# ---- Fig B: grouped bars at step 2000, two clearly-separated conditions ----
fig, ax = plt.subplots(figsize=(9,5))
grpA = [("faithful", bf.get(2000)), ("full IMU-1", bm.get(2000))]            # mid-run snapshots
grpB = [("baseline", dc["baseline"]), ("+WSD", dc["wsd"]), ("+z-loss", dc["zloss"]), ("+arch", dc["arch"])]  # complete runs
x=0; xticks=[]; xlab=[]
for name,v in grpA:
    if v: ax.bar(x, v, color="#9467bd" if "IMU" in name else "#444", alpha=0.55)
    if v: ax.text(x, v+0.5, f"{v:.1f}", ha="center", fontsize=8)
    xticks.append(x); xlab.append(name); x+=1
x+=0.8
for name,v in grpB:
    c={"baseline":"#444","+WSD":"#1f77b4","+z-loss":"#d62728","+arch":"#2ca02c"}[name]
    if v: ax.bar(x, v, color=c, alpha=0.7)
    if v: ax.text(x, v+0.5, f"{v:.1f}", ha="center", fontsize=8)
    xticks.append(x); xlab.append(name); x+=1
ax.axhline(PHASE_A_FAITHFUL, color="#444", ls="--", lw=1, alpha=0.6)
ax.text(x-0.5, PHASE_A_FAITHFUL+0.3, f"Phase-A faithful complete = {PHASE_A_FAITHFUL}", fontsize=7.5, ha="right", color="#444")
ax.set_xticks(xticks); ax.set_xticklabels(xlab, fontsize=9)
ax.set(ylabel="in-loop val PPL @ 131M tok (lower=better)",
       title="Same budget (131M tok), but two conditions:\nLEFT = build MID-RUN snapshots (LR high)   |   RIGHT = de-confound COMPLETE runs (LR decayed)")
ax.annotate("", xy=(1.6,0), xytext=(1.6,max(filter(None,[bf.get(2000),bm.get(2000)]+list(dc.values())))*0.9),
            arrowprops=dict(arrowstyle="-", ls=":", color="gray"))
ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
fig.savefig(HERE/"same_steps_bars.png", dpi=130); plt.close(fig)
print("wrote same_steps_curves.png, same_steps_bars.png")
print("de-confound finals:", {k:(round(v,2) if v else None) for k,v in dc.items()})
print("build @2000 (mid-run): faithful", bf.get(2000), "| full IMU-1", bm.get(2000))
