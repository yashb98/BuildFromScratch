#!/usr/bin/env bash
# Data-selection A/B — treatment cells only (control = Phase-1 FineWeb-Edu baselines, §C13 reused).
# treatment = dclm-edu (the prepared shards) vs control = FineWeb-Edu, SAME model/arch/optimizer/
# schedule/budget/seed — ONLY the training data differs (single-variable; iso-FLOP by construction).
# 1 arm × 3 seeds = 3 NEW cells, 2000-step proxy (~131M tok consumed of the 150M slice, ~5h/cell).
# One trainer at a time (§C4.5); idempotent (.done markers + --resume); sentinel --kill-at 0.83.
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$EXP/../../.." && pwd)"; cd "$EXP"
PY=python3; STEPS=2000; WARMUP=100
CHILD=""; trap '[ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 143' TERM INT
for seed in 0 1 2; do
  cell="treatment_seed${seed}"
  [ -f "arm_${cell}.done" ] && { echo "[skip] $cell"; continue; }
  for _ in $(seq 1 24); do a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 5; done
  ckpt="checkpoint_${cell}.pt"; resume=""; [ -f "$ckpt" ] && resume="--resume $ckpt"
  echo "===== CELL $cell ($(date '+%H:%M:%S')) $resume ====="
  $PY train_dataarm.py --data dclm --arch faithful --schedule cosine --zloss 0 --optimizer adamw \
      --seed "$seed" --steps "$STEPS" --warmup_steps "$WARMUP" --eval_every 500 --ckpt_every 250 \
      --ckpt_path "$ckpt" --run_name "$cell" $resume >> "run_${cell}.log" 2>&1 &
  CHILD=$!
  setsid nohup $PY "$REPO/sentinel.py" watch --pid "$CHILD" --kill-at 0.83 --log "$EXP/sentinel.log" >/dev/null 2>&1 &
  wait "$CHILD"; rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then touch "arm_${cell}.done"; echo "[done] $cell"
  else echo "[FAIL] $cell rc=$rc"; exit "$rc"; fi
done
echo "ALL 3 TREATMENT CELLS COMPLETE ($(date '+%H:%M:%S'))"; touch cohort.done
