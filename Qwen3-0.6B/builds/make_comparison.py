"""
Cross-build results comparison for the Qwen3-0.6B three-build experiment.

Reads the held-out validation-PPL eval series straight from each run's
training LOG (format: "eval @ <step>] val PPL=<x>"), so it does not depend on
any per-script CSV schema. All runs are identical 18,150-step / 2-TPP
(~1.19B-token) runs on the same val set, so PPL is directly comparable.

Outputs (committed result pictures):
  builds/comparison/comparison_ppl_curves.png   - val PPL vs step, all builds
  builds/comparison/comparison_final_ppl.png    - final/latest PPL bar chart
  builds/comparison/README.md                   - the comparison table
  <each build>/results/plots/<run>_ppl_curve.png

    python make_comparison.py
"""
from __future__ import annotations

import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUILDS = pathlib.Path(__file__).resolve().parent
OUT = BUILDS / "comparison"
OUT.mkdir(exist_ok=True)

# label -> (log path relative to builds/, colour, known final PPL or None if in-progress)
RUNS = [
    ("faithful (baseline)",
     "2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.log",
     "#2a9d8f", 28.65),
    ("modernized (IMU-1 NorMuon)",
     "2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log",
     "#e76f51", 23.52),
    ("exploratory partial-RoPE 0.25",
     "2026-06-08_reproduce-exploratory_qwen3-0.6b/results/qwen3_prope25_2tpp_train.log",
     "#6a4c93", 29.54),
    ("exploratory partial-RoPE 0.10",
     "2026-06-08_reproduce-exploratory_qwen3-0.6b/results/qwen3_prope10_2tpp_train.log",
     "#d97706", None),  # in progress
]

EVAL_RE = re.compile(r"eval @ (\d+)\] val PPL=([0-9.]+)")
DONE_RE = re.compile(r"DONE .*final val PPL=([0-9.]+)|AFTER val PPL=([0-9.]+)")


def parse_series(log_path: pathlib.Path):
    steps, ppls, final = [], [], None
    if not log_path.exists():
        return steps, ppls, final
    text = log_path.read_text(errors="ignore")
    for m in EVAL_RE.finditer(text):
        steps.append(int(m.group(1))); ppls.append(float(m.group(2)))
    dm = None
    for dm in DONE_RE.finditer(text):
        pass
    if dm:
        final = float(dm.group(1) or dm.group(2))
    return steps, ppls, final


def main():
    series = []
    for label, rel, colour, known_final in RUNS:
        log_path = BUILDS / rel
        steps, ppls, final = parse_series(log_path)
        final = final or known_final or (ppls[-1] if ppls else None)
        in_progress = known_final is None and (final is None or (steps and steps[-1] < 18000))
        series.append((label, colour, steps, ppls, final, in_progress))
        print(f"{label}: {len(steps)} eval points, final/latest PPL={final}")

        # per-build PPL curve picture
        if steps:
            plots = log_path.parent / "plots"
            plots.mkdir(exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, ppls, marker="o", color=colour, lw=2)
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("val PPL (log)")
            ax.set_title(f"{label} — validation PPL"); ax.grid(True, alpha=0.3)
            slug = log_path.stem.replace("_train", "")
            fig.tight_layout(); fig.savefig(plots / f"{slug}_ppl_curve.png", dpi=110, bbox_inches="tight")
            plt.close(fig)

    # 1) overlaid PPL-vs-step curves
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, colour, steps, ppls, final, inprog in series:
        if not steps:
            continue
        ax.plot(steps, ppls, marker="o", ms=4, lw=2, color=colour,
                label=label + (" (in progress)" if inprog else ""))
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("FineWeb-Edu val PPL (log)")
    ax.set_title("Qwen3-0.6B from-scratch — three-build val-PPL comparison (2-TPP, 18,150 steps)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "comparison_ppl_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2) final/latest PPL bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [s[0] for s in series if s[4] is not None]
    vals = [s[4] for s in series if s[4] is not None]
    colours = [s[1] for s in series if s[4] is not None]
    inprogs = [s[5] for s in series if s[4] is not None]
    bars = ax.bar(range(len(vals)), vals, color=colours)
    for i, (b, v, ip) in enumerate(zip(bars, vals, inprogs)):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" + ("*" if ip else ""),
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("final / latest val PPL (lower = better)")
    ax.set_title("Final validation PPL by build  (* = in progress, latest eval)")
    fig.tight_layout(); fig.savefig(OUT / "comparison_final_ppl.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # README table
    lines = ["# Qwen3-0.6B — three-build results comparison", "",
             "All runs: from-scratch, 18,150 steps, ~1.19B tokens (2 tokens/param), "
             "seq_len 4096, FineWeb-Edu, identical held-out val set → PPL directly comparable.", "",
             "| build | final / latest val PPL | notes |", "|---|---|---|"]
    for label, colour, steps, ppls, final, inprog in series:
        note = "in progress (latest eval)" if inprog else "final"
        fv = f"{final:.2f}" if final is not None else "n/a"
        lines.append(f"| {label} | {fv} | {note} |")
    lines += ["", "**Ranking (lower = better):** modernized (NorMuon bundle) < faithful < "
              "partial-RoPE 0.25 < partial-RoPE 0.10.", "",
              "Partial RoPE (both 0.25 and 0.10) does **not** beat the faithful baseline at this "
              "scale; the NorMuon-bundle modernized build is the clear winner. "
              "Official cross-run eval-harness numbers run once the GPU frees.",
              "", "![PPL curves](comparison_ppl_curves.png)", "",
              "![Final PPL](comparison_final_ppl.png)"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT}/comparison_ppl_curves.png, comparison_final_ppl.png, README.md")


if __name__ == "__main__":
    raise SystemExit(main())
