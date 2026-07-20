#!/usr/bin/env bash
# Pause the anneal at its step-1500 checkpoint -> run the GB10 CCE benchmark -> ALWAYS resume the
# anneal (trap EXIT) from that checkpoint. One GPU job at a time (§C4.5); resume is guaranteed.
set -u
ROOT=/home/yashb98/Downloads/BuildFromScratch
EXP=$ROOT/Qwen3-0.6B/experiments/2026-06-30_qwen3-0.6b_midtrain-anneal
KDIR=$ROOT/research/kernel
RUNLOG=$KDIR/gb10_cce_run.log

resume_anneal() {
  rm -f "$KDIR/gb10_cce_running"
  cd "$EXP"
  if ! pgrep -f "$EXP/run_arms.sh" >/dev/null 2>&1 && [ ! -f cohort.done ]; then
    setsid nohup bash run_arms.sh >> run_arms.log 2>&1 &
    echo "[$(date '+%T')] ANNEAL RESUMED — run_arms.sh relaunched (--resume mix_seed0 from ckpt)" >> "$RUNLOG"
  fi
}
trap resume_anneal EXIT   # guarantee the anneal comes back, success/fail/timeout

{
echo "===== $(date '+%F %T') GB10 CCE benchmark orchestration ====="
echo "[1/4] pip install liger + cce (--no-deps; runs while the anneal trains)"
python3 -m pip install -q --no-deps liger-kernel "git+https://github.com/apple/ml-cross-entropy.git" 2>&1 | tail -3

echo "[2/4] waiting for the step-1500 checkpoint (so resume loses ~0 progress)..."
for _ in $(seq 1 180); do
  grep -q "ckpt @ 1500" "$EXP/run_mix_seed0.log" 2>/dev/null && { echo "  -> ckpt @ 1500 saved"; break; }
  sleep 10
done

echo "[3/4] pausing the anneal"
touch "$KDIR/gb10_cce_running"
pkill -TERM -f "$EXP/run_arms.sh" 2>/dev/null
pkill -TERM -f "train_anneal.py"  2>/dev/null
pkill -f "sentinel.py watch"      2>/dev/null
sleep 8
echo "  GPU util now: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null) (should be ~0)"

echo "[4/4] running the GB10 CCE benchmark (CCE should now fit — Blackwell shared memory)"
cd "$KDIR"; timeout 1500 python3 gb10_cce_bench.py
touch "$KDIR/gb10_cce.done"
echo "[$(date '+%T')] benchmark finished -> EXIT trap will resume the anneal"
} >> "$RUNLOG" 2>&1
