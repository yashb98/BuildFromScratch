# VibeThinker SFT — plots

_Run COMPLETE — 238/238 steps, finished 2026-06-18. No extrapolation; only discrete logged points are drawn._

All data parsed from `results/vibethinker_sft_driver.log` by `plots/plot_vibethinker_sft.py` (CPU-only, matplotlib Agg, no GPU, no model loads). Re-run that script to refresh as the run progresses.

## Figures

- **vibethinker_sft_loss.{png,pdf}** — SFT training cross-entropy loss vs step (logged steps 10–230). Source: `results/vibethinker_sft_driver.log`.
- **vibethinker_sft_reasoning_ppl.{png,pdf}** — held-out reasoning perplexity vs step at eval checkpoints, with the pre-SFT base floor 14.262 as a dashed reference line. Source: `results/vibethinker_sft_driver.log`.

Each figure is saved as PNG (300 dpi) and PDF.

