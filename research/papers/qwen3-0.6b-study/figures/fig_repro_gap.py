#!/usr/bin/env python3
"""fig_repro_gap: val-PPL vs training tokens (log-log).

All plotted numbers are parsed from the two source results files below —
nothing is hardcoded except axis labels and styling.

Sources:
  Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_vs_repro.txt
  Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_after.txt
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/yashb98/Downloads/BuildFromScratch")
RESULTS = REPO / "Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results"
SRC_ORIG = RESULTS / "original_vs_repro.txt"
SRC_PHASEB = RESULTS / "qwen3_baseline2tpp_after.txt"
OUTDIR = Path(__file__).resolve().parent


def parse_tok_count(s: str) -> float:
    """'36T' -> 36e12, '131M' -> 131e6, '1,189,478,400' -> 1.1894784e9."""
    s = s.replace(",", "").strip()
    mult = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    if s[-1].upper() in mult:
        return float(s[:-1]) * mult[s[-1].upper()]
    return float(s)


# --- Parse original_vs_repro.txt: original point + Phase-A best (min PPL) ---
orig_text = SRC_ORIG.read_text()

m = re.search(
    r"ORIGINAL\s+Qwen3-0\.6B-Base\s+\((\S+)\s+tok\)\s+val PPL\s*=\s*([\d.]+)",
    orig_text,
)
assert m, f"original line not found in {SRC_ORIG}"
orig_tokens = parse_tok_count(m.group(1))
orig_ppl = float(m.group(2))

repro_rows = re.findall(
    r"REPRO\s+(\S+)\s+\((\S+)\s+tok, from scratch\)\s+val PPL\s*=\s*([\d.]+)",
    orig_text,
)
assert repro_rows, f"no REPRO rows found in {SRC_ORIG}"
best_name, best_tok_s, best_ppl = min(repro_rows, key=lambda r: float(r[2]))
phaseA_tokens = parse_tok_count(best_tok_s)
phaseA_ppl = float(best_ppl)

# --- Parse qwen3_baseline2tpp_after.txt: Phase-B point ---
pb_text = SRC_PHASEB.read_text()
m = re.search(r"after\s+([\d,]+)\s+tokens", pb_text)
assert m, f"token count not found in {SRC_PHASEB}"
phaseB_tokens = parse_tok_count(m.group(1))
m = re.search(r"val PPL:\s*[\d.]+\s*->\s*([\d.]+)", pb_text)
assert m, f"final val PPL not found in {SRC_PHASEB}"
phaseB_ppl = float(m.group(1))

# --- Gap ratios (computed, not hardcoded) ---
gapA = phaseA_ppl / orig_ppl
gapB = phaseB_ppl / orig_ppl

# --- Plot (Okabe-Ito colorblind-safe palette) ---
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10})
fig, ax = plt.subplots(figsize=(3.4, 2.7))

ax.plot(
    [phaseA_tokens, phaseB_tokens],
    [phaseA_ppl, phaseB_ppl],
    color="0.6", lw=1.0, ls="--", zorder=1,
)
ax.scatter(
    [phaseA_tokens], [phaseA_ppl],
    marker="o", s=45, color=C_BLUE, zorder=3,
    label=f"Phase-A best ({best_name})",
)
ax.scatter(
    [phaseB_tokens], [phaseB_ppl],
    marker="s", s=45, color=C_ORANGE, zorder=3,
    label="Phase-B (2 TPP)",
)
ax.scatter(
    [orig_tokens], [orig_ppl],
    marker="*", s=170, color=C_GREEN, zorder=3,
    label="Qwen3-0.6B-Base (original)",
)

ax.annotate(
    f"{gapA:.1f}$\\times$ gap",
    xy=(phaseA_tokens, phaseA_ppl), xytext=(4, 7),
    textcoords="offset points", fontsize=9, color=C_BLUE,
)
ax.annotate(
    f"{gapB:.2f}$\\times$ gap",
    xy=(phaseB_tokens, phaseB_ppl), xytext=(4, 7),
    textcoords="offset points", fontsize=9, color=C_ORANGE,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Training tokens")
ax.set_ylabel("Validation perplexity")
ax.grid(True, which="both", lw=0.3, alpha=0.4)
ax.legend(fontsize=9, frameon=False, loc="upper right", handletextpad=0.4)

fig.tight_layout()
fig.savefig(OUTDIR / "fig_repro_gap.pdf")
fig.savefig(OUTDIR / "fig_repro_gap.png", dpi=200)

print(f"orig:   {orig_tokens:.4g} tok, PPL {orig_ppl}")
print(f"phaseA: {phaseA_tokens:.4g} tok, PPL {phaseA_ppl} ({best_name}), gap {gapA:.2f}x")
print(f"phaseB: {phaseB_tokens:.4g} tok, PPL {phaseB_ppl}, gap {gapB:.2f}x")
print(f"wrote {OUTDIR / 'fig_repro_gap.pdf'}")
