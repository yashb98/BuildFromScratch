#!/usr/bin/env bash
# RLVR Phase-2 cohort supervisor (plan §3 Phase 2, ACCEPTED 2026-07-02): serializes
# grpo_seed0 -> random_seed0 (spurious gate) -> rft_seed0 (iso-generation control), one
# GPU job at a time (§C4.5). Idempotent: .done markers + --resume from step-25-multiple
# checkpoints. sentinel watch --kill-at 0.83 armed beside every arm (§C6).
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$EXP/../../.." && pwd)"; cd "$EXP"
PY=python3
CHILD=""; trap '[ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 143' TERM INT

run_arm () {  # $1 = arm name
  local arm="$1" name="${1}_seed0" ckpt="checkpoint_${1}_seed0.pt"
  [ -f "arm_${name}.done" ] && { echo "[skip] $name"; return 0; }
  for _ in $(seq 1 24); do a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 5; done
  local resume=""
  [ -f "$ckpt" ] && resume="--resume $ckpt"
  echo "===== ARM $name ($(date '+%F %H:%M:%S')) ${resume:-fresh} ====="
  $PY train_grpo.py --arm "$arm" --steps 300 --seed 0 $resume \
      >> "run_${name}.log" 2>&1 &
  CHILD=$!
  setsid nohup $PY "$REPO/sentinel.py" watch --pid "$CHILD" --kill-at 0.83 --log "$EXP/sentinel.log" >/dev/null 2>&1 &
  wait "$CHILD"; local rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then touch "arm_${name}.done"; echo "[done] $name"
  else echo "[FAIL] $name rc=$rc"; exit "$rc"; fi
}

run_arm grpo
run_arm random
run_arm rft
echo "ALL 3 ARMS COMPLETE ($(date '+%F %H:%M:%S'))"; touch cohort.done
