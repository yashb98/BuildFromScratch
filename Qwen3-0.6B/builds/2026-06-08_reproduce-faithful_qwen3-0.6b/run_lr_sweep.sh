#!/usr/bin/env bash
# Build 1 — matched-compute learning-rate sweep (Qwen3-0.6B from scratch)
# Three runs, identical except peak LR. 2,000 steps = ~131M tokens each.
#   A: 1.7e-3  (Qwen3 paper baseline anchor)
#   B: 2.4e-3  (midpoint)
#   C: 3.0e-3  (SmolLM2 paper anchor)
# Shared fixed knobs: warmup 150, end_lr 3.2e-4, cosine, AdamW(0.9,0.95),
# wd 0.01, clip 1.0, bf16, seq 4096, micro_batch 4 x grad_accum 4, seed 0.
set -euo pipefail
cd "$(dirname "$0")"
COMMON="--steps 2000 --warmup_steps 150 --end_lr 3.2e-4 --seq_len 4096 \
  --micro_batch 4 --grad_accum 4 --weight_decay 0.01 --grad_clip 1.0 \
  --mem_fraction 0.85 --seed 0 --eval_every 250 --ckpt_every 500"

echo "=== [$(date)] SWEEP START ==="
for cfg in "lr17 1.7e-3" "lr24 2.4e-3" "lr30 3.0e-3"; do
  set -- $cfg
  name=$1; lr=$2
  echo "=== [$(date)] RUN $name  peak_lr=$lr ==="
  python train_qwen3.py $COMMON --peak_lr "$lr" --run_name "$name"
  echo "=== [$(date)] RUN $name DONE ==="
done
echo "=== [$(date)] SWEEP COMPLETE ==="
