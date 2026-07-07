#!/usr/bin/env python3
"""fig_ppl_curves: in-loop val PPL vs consumed tokens, faithful vs IMU-1.

Parses the "[eval @ N] val PPL=X" lines from the two 2 TPP training logs
(identical 1,189,478,400-token budget, 18,150 steps at 65,536 tok/step) and
plots both curves on a log-y axis. For the faithful run the log also carries
a final end-of-training eval line "AFTER val PPL=28.65" at step 18150, which
is appended as the curve's terminal point; the IMU-1 log has no AFTER line,
so its curve ends at its last in-loop eval ([eval @ 18000] val PPL=23.52).

All plotted numbers are parsed from the two logs below; nothing is hardcoded
except the tokens-per-step constant (verified against both log headers) and
the expected endpoint values, which are asserted, not injected.
Output: fig_ppl_curves.pdf (+ .png preview).
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/yashb98/Downloads/BuildFromScratch"

SRC_FAITHFUL = os.path.join(
    REPO,
    "Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/"
    "results/qwen3_baseline2tpp_train.log",
)
SRC_IMU1 = os.path.join(
    REPO,
    "Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/"
    "results/qwen3_imu1_2tpp_train.log",
)

TOK_PER_STEP = 65_536  # both logs: "tok/step=65,536  steps=18,150"
TOTAL_STEPS = 18_150   # -> 1,189,478,400-token budget for both arms

EVAL_RE = re.compile(r"\[eval @ (\d+)\] val PPL=([0-9.]+)")
AFTER_RE = re.compile(r"AFTER val PPL=([0-9.]+)")

# Okabe-Ito colorblind-safe palette (house style, cf. fig_deconfound.py)
COLOR_FAITHFUL = "#0072B2"  # blue
COLOR_IMU1 = "#D55E00"      # vermillion


def parse_curve(path, take_after_at_step=None):
    """Return (steps, ppls) from the [eval @ N] lines of one training log.

    If take_after_at_step is given, the log's final "AFTER val PPL=X"
    end-of-training eval is appended at that step.
    """
    steps, ppls = [], []
    after = None
    with open(path) as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                ppls.append(float(m.group(2)))
                continue
            m = AFTER_RE.search(line)
            if m:
                after = float(m.group(1))
    if take_after_at_step is not None:
        assert after is not None, "no AFTER line in %s" % path
        steps.append(take_after_at_step)
        ppls.append(after)
    assert steps == sorted(steps), "eval steps out of order in %s" % path
    return steps, ppls


def main():
    steps_f, ppl_f = parse_curve(SRC_FAITHFUL, take_after_at_step=TOTAL_STEPS)
    steps_m, ppl_m = parse_curve(SRC_IMU1)

    # Endpoint guards: values must come out of the logs exactly.
    assert ppl_f[-1] == 28.65, "faithful endpoint %r != 28.65" % ppl_f[-1]
    assert ppl_m[-1] == 23.52, "IMU-1 endpoint %r != 23.52" % ppl_m[-1]
    assert steps_f[-1] == TOTAL_STEPS and steps_m[-1] == 18_000

    tok_f = [s * TOK_PER_STEP / 1e9 for s in steps_f]
    tok_m = [s * TOK_PER_STEP / 1e9 for s in steps_m]

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 2.5))

    ax.plot(
        tok_f,
        ppl_f,
        color=COLOR_FAITHFUL,
        linewidth=1.6,
        marker="o",
        markersize=3.5,
        label="faithful (AdamW)",
    )
    ax.plot(
        tok_m,
        ppl_m,
        color=COLOR_IMU1,
        linewidth=1.6,
        linestyle="--",
        marker="s",
        markersize=3.5,
        label="modernized (IMU-1)",
    )

    # Direct endpoint labels (the two headline numbers).
    ax.annotate(
        "%.2f" % ppl_f[-1],
        xy=(tok_f[-1], ppl_f[-1]),
        xytext=(4, 3),
        textcoords="offset points",
        fontsize=8,
        color="#333333",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "%.2f" % ppl_m[-1],
        xy=(tok_m[-1], ppl_m[-1]),
        xytext=(4, -3),
        textcoords="offset points",
        fontsize=8,
        color="#333333",
        ha="left",
        va="top",
    )

    ax.set_yscale("log")
    ax.set_yticks([20, 30, 40, 50, 60])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % v))
    ax.minorticks_off()
    ax.set_xlabel("tokens consumed (billions)", fontsize=9)
    ax.set_ylabel("val PPL (log scale)", fontsize=9)
    ax.set_xlim(0, 1.25)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    pdf_path = os.path.join(HERE, "fig_ppl_curves.pdf")
    png_path = os.path.join(HERE, "fig_ppl_curves.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print("wrote", pdf_path)
    print("wrote", png_path)
    print("faithful:  %d evals, endpoint PPL=%.2f @ %.3fB tok" % (len(ppl_f), ppl_f[-1], tok_f[-1]))
    print("IMU-1:     %d evals, endpoint PPL=%.2f @ %.3fB tok" % (len(ppl_m), ppl_m[-1], tok_m[-1]))


if __name__ == "__main__":
    main()
