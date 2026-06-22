#!/usr/bin/env python3
"""
Plot generator for the MODERNIZED (IMU-1 / NorMuon bundle) Qwen3-0.6B build.

HARD RULES enforced here:
  - CPU-only, matplotlib Agg backend, NO GPU, NO model loads.
  - Plot ONLY numbers parsed from real on-disk log files this run.
  - Cite the exact source file in every figure caption/title.
  - No extrapolation, no fabricated points. Partial runs are LABELED.

Source files (parsed, not loaded as models):
  results/qwen3_imu1_2tpp_train.log    -- the 2-tokens-per-param main run
  results/qwen3_imu1_smoke_train.log   -- the short smoke run

Step lines:  '[hh:mm:ss] step N/TOTAL ce X z .. lr .. |grad| .. mem .. tok/s ..'
Eval lines:  '[hh:mm:ss]   [eval @ N] val PPL=X'
"""
import os
import re
import matplotlib
matplotlib.use("Agg")  # headless, no display, no GPU
import matplotlib.pyplot as plt

# ---- clean serif style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "figure.dpi": 100,
})

RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
OUT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_2TPP = os.path.join(RESULTS_DIR, "qwen3_imu1_2tpp_train.log")
LOG_SMOKE = os.path.join(RESULTS_DIR, "qwen3_imu1_smoke_train.log")

# Filenames as shown in captions (relative to build dir, for provenance)
SRC_2TPP = "results/qwen3_imu1_2tpp_train.log"
SRC_SMOKE = "results/qwen3_imu1_smoke_train.log"

STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+ce\s+([\d.]+)\s+z\s+([\d.eE+-]+)\s+"
    r"lr\s+([\d.eE+-]+)\s+\|grad\|\s+([\d.eE+-]+)"
)
EVAL_RE = re.compile(r"\[eval @ (\d+)\]\s+val PPL=([\d.]+)")
DONE_RE = re.compile(r"\]\s*DONE\s*$")


def parse_log(path):
    """Return dict with steps, ce, grad, lr, eval points, total_steps, done flag."""
    steps, ce, grad, lr = [], [], [], []
    eval_steps, eval_ppl = [], []
    total_steps = None
    done = False
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                total_steps = int(m.group(2))
                ce.append(float(m.group(3)))
                lr.append(float(m.group(5)))
                grad.append(float(m.group(6)))
                continue
            e = EVAL_RE.search(line)
            if e:
                eval_steps.append(int(e.group(1)))
                eval_ppl.append(float(e.group(2)))
                continue
            if DONE_RE.search(line):
                done = True
    return {
        "steps": steps, "ce": ce, "grad": grad, "lr": lr,
        "eval_steps": eval_steps, "eval_ppl": eval_ppl,
        "total_steps": total_steps, "done": done,
        "last_step": steps[-1] if steps else None,
    }


def ema(xs, alpha=0.1):
    """Exponential moving average for smoothing (no point fabrication; same x's)."""
    out = []
    s = None
    for x in xs:
        s = x if s is None else (alpha * x + (1 - alpha) * s)
        out.append(s)
    return out


def save(fig, basename):
    png = os.path.join(OUT_DIR, basename + ".png")
    pdf = os.path.join(OUT_DIR, basename + ".pdf")
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def status_str(d):
    if d["done"]:
        return "COMPLETE (DONE)"
    return "IN PROGRESS / PARTIAL"


# ============================ PARSE ============================
main = parse_log(LOG_2TPP)
smoke = parse_log(LOG_SMOKE)

main_done = main["done"] and main["last_step"] == main["total_steps"]
smoke_done = smoke["done"] and smoke["last_step"] == smoke["total_steps"]

main_label = (
    f"COMPLETE: reached step {main['last_step']}/{main['total_steps']} (DONE)"
    if main_done else
    f"PARTIAL: last step {main['last_step']}/{main['total_steps']} (no DONE)"
)
smoke_label = (
    f"COMPLETE: reached step {smoke['last_step']}/{smoke['total_steps']} (DONE)"
    if smoke_done else
    f"PARTIAL: last step {smoke['last_step']}/{smoke['total_steps']} (no DONE)"
)

# ===================== FIG 1: training CE vs step =====================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(main["steps"], main["ce"], color="#9ecae1", lw=0.8, alpha=0.7,
        label="raw ce (per logged step)")
ax.plot(main["steps"], ema(main["ce"], 0.1), color="#08519c", lw=1.8,
        label="smoothed ce (EMA alpha=0.1)")
ax.set_xlabel("training step")
ax.set_ylabel("cross-entropy (nats/token)")
ax.set_title(
    "Qwen3-0.6B MODERNIZED (IMU-1/NorMuon): training cross-entropy vs step\n"
    f"source: {SRC_2TPP}  |  {main_label}"
)
ax.legend(loc="upper right")
p1 = save(fig, "fig1_train_ce_vs_step")

# ===================== FIG 2: eval val-PPL vs step (log y) =====================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(main["eval_steps"], main["eval_ppl"], "o-", color="#08519c", lw=1.8,
        markersize=6, label="val PPL (eval @ N)")
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("validation perplexity (log scale)")
# annotate verified endpoints
ax.annotate(f"{main['eval_ppl'][0]:.2f} @ {main['eval_steps'][0]}",
            (main["eval_steps"][0], main["eval_ppl"][0]),
            textcoords="offset points", xytext=(8, 8), fontsize=9)
ax.annotate(f"{main['eval_ppl'][-1]:.2f} @ {main['eval_steps'][-1]}",
            (main["eval_steps"][-1], main["eval_ppl"][-1]),
            textcoords="offset points", xytext=(-70, -16), fontsize=9)
ax.set_title(
    "Qwen3-0.6B MODERNIZED (IMU-1/NorMuon): eval validation PPL vs step\n"
    f"source: {SRC_2TPP}  |  {main_label}"
)
ax.legend(loc="upper right")
p2 = save(fig, "fig2_eval_ppl_vs_step")

# ===================== FIG 3: grad-norm vs step =====================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(main["steps"], main["grad"], color="#fdae6b", lw=0.8, alpha=0.7,
        label="raw |grad| (per logged step)")
ax.plot(main["steps"], ema(main["grad"], 0.1), color="#a63603", lw=1.8,
        label="smoothed |grad| (EMA alpha=0.1)")
ax.set_xlabel("training step")
ax.set_ylabel("gradient norm |grad|")
ax.set_title(
    "Qwen3-0.6B MODERNIZED (IMU-1/NorMuon): gradient norm vs step\n"
    f"source: {SRC_2TPP}  |  {main_label}"
)
ax.legend(loc="upper right")
p3 = save(fig, "fig3_grad_norm_vs_step")

# ===================== FIG 4: smoke-run eval descent =====================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(smoke["eval_steps"], smoke["eval_ppl"], "s-", color="#54278f", lw=1.8,
        markersize=7, label="smoke val PPL (eval @ N)")
for x, y in zip(smoke["eval_steps"], smoke["eval_ppl"]):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                xytext=(6, 6), fontsize=9)
ax.set_xlabel("training step")
ax.set_ylabel("validation perplexity")
ax.set_title(
    "Qwen3-0.6B MODERNIZED (IMU-1/NorMuon): SMOKE-run eval descent\n"
    f"source: {SRC_SMOKE}  |  {smoke_label}"
)
ax.legend(loc="upper right")
p4 = save(fig, "fig4_smoke_eval_descent")

# ============================ REPORT ============================
print("=" * 70)
print("PARSED DATA POINTS (cross-check against source logs)")
print("=" * 70)
print(f"\n[MAIN 2TPP] source: {SRC_2TPP}")
print(f"  status: {status_str(main)}  last_step={main['last_step']}/{main['total_steps']}")
print(f"  step lines parsed: {len(main['steps'])}")
print(f"  FIG1 ce: first={main['ce'][0]} @step{main['steps'][0]}  "
      f"last={main['ce'][-1]} @step{main['steps'][-1]}  "
      f"min={min(main['ce'])}")
print(f"  FIG3 |grad|: first={main['grad'][0]} @step{main['steps'][0]}  "
      f"last={main['grad'][-1]} @step{main['steps'][-1]}  "
      f"max={max(main['grad'])}")
print(f"  FIG2 eval val PPL ({len(main['eval_steps'])} points):")
for s, p in zip(main["eval_steps"], main["eval_ppl"]):
    print(f"    step {s:>6}  PPL={p}")

print(f"\n[SMOKE] source: {SRC_SMOKE}")
print(f"  status: {status_str(smoke)}  last_step={smoke['last_step']}/{smoke['total_steps']}")
print(f"  step lines parsed: {len(smoke['steps'])}")
print(f"  FIG4 eval val PPL ({len(smoke['eval_steps'])} points):")
for s, p in zip(smoke["eval_steps"], smoke["eval_ppl"]):
    print(f"    step {s:>6}  PPL={p}")

print("\nOUTPUT FILES:")
for png, pdf in [p1, p2, p3, p4]:
    print(f"  {png}")
    print(f"  {pdf}")
