#!/usr/bin/env bash
# Wait for the Phase A sweep to fully finish (no train_qwen3 process), then run
# the original-vs-reproduction PPL comparison. Keeps the GB10 at ONE GPU job:
# it only launches the eval AFTER the last sweep run has exited.
set -uo pipefail
cd "$(dirname "$0")"
echo "[$(date)] waiting for Phase A sweep to finish before original-eval..."
while pgrep -f "train_qwen3.py --steps 2000" >/dev/null; do
  sleep 60
done
sleep 15   # let the last process release CUDA memory
echo "[$(date)] sweep done — running original_vs_repro eval"
python eval_original_vs_repro.py
echo "[$(date)] original-eval complete"
