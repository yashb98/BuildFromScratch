#!/usr/bin/env bash
# Phase B — 4 matched-compute runs @ 2 TPP (18,150 steps = 1.19B tokens each).
# Sequential (one GPU job at a time on the GB10). Best LR from Phase A = 2.4e-3.
#   1. faithful baseline   (AdamW, full RoPE)            <- also Build 1 final + 100%-RoPE anchor
#   2. IMU-1 full bundle    (NorMuon + value-resid + LN-scale + head-gate + WSD)
#   3. partial RoPE 25%     (AdamW, same recipe as baseline)
#   4. partial RoPE 10%     (AdamW, same recipe as baseline)
# First run streams+caches 1.19B tokens (~30 min); runs 2-4 reuse the cache.
set -uo pipefail
B=/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds
FAITHFUL=$B/2026-06-08_reproduce-faithful_qwen3-0.6b
MOD=$B/2026-06-08_reproduce-modernized_qwen3-0.6b
EXP=$B/2026-06-08_reproduce-exploratory_qwen3-0.6b
S=18150; W=900; COMMON="--eval_every 2000 --ckpt_every 2000 --log_every 50"

echo "=== PHASE B START $(date) ==="

echo "=== [1/4] faithful baseline (AdamW 2.4e-3) $(date) ==="
cd "$FAITHFUL" && python train_qwen3.py --steps $S --peak_lr 2.4e-3 --end_lr 3.2e-4 \
  --warmup_steps $W $COMMON --run_name baseline2tpp
echo "=== [1/4] DONE $(date) ==="

echo "=== [2/4] IMU-1 full bundle (NorMuon) $(date) ==="
cd "$MOD" && python train_imu1.py --steps $S --warmup_steps $W $COMMON --run_name imu1_2tpp
echo "=== [2/4] DONE $(date) ==="

echo "=== [3/4] partial RoPE 25% $(date) ==="
cd "$EXP" && python train_partialrope.py --steps $S --partial_rotary_factor 0.25 \
  --warmup_steps $W $COMMON --run_name prope25_2tpp
echo "=== [3/4] DONE $(date) ==="

echo "=== [4/4] partial RoPE 10% $(date) ==="
cd "$EXP" && python train_partialrope.py --steps $S --partial_rotary_factor 0.10 \
  --warmup_steps $W $COMMON --run_name prope10_2tpp
echo "=== [4/4] DONE $(date) ==="

echo "=== PHASE B COMPLETE $(date) ==="
