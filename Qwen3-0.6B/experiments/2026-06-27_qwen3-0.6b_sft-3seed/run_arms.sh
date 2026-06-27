#!/usr/bin/env bash
# SFT 3-seed verdict + iso-FLOP control (post-training plan §2A). Converts the n=1 SFT
# (inconclusive, floor 25.2%) into a >=3-seed paired-CI verdict.
#   sft_seed{0,1,2}  = response-masked SFT (the treatment)
#   ctrl_seed{0,1,2} = --no_mask continued-pretrain on the SAME data/budget (iso-FLOP control:
#                      attributes any reasoning gain to SFT-format vs just-more-math-tokens)
# Faithful base loaded strict=True by sft_train.py. Proven config (LR 5e-5 cosine, mb4 x ga32,
# ~125M tok, compile ON ~7,000-7,459 tok/s, peak 52.4 GB). One trainer at a time (§C4.5);
# idempotent (.done + --resume); sentinel --kill-at 0.83. ~5h/cell, 6 cells ~1.25 days.
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$EXP/../../.." && pwd)"; cd "$EXP"
PY=python3
CHILD=""; trap '[ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 143' TERM INT

run_cell () {  # $1=arm flag ("" masked | "--no_mask"), $2=name
  local flag="$1" cell="$2" ckpt="checkpoint_${2}.pt"
  [ -f "arm_${cell}.done" ] && { echo "[skip] $cell"; return 0; }
  for _ in $(seq 1 24); do a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 5; done
  local resume=""; [ -f "$ckpt" ] && resume="--resume $ckpt"
  echo "===== CELL $cell ($(date '+%H:%M:%S')) $resume ====="
  local seed="${cell##*seed}"
  $PY sft_train.py $flag --seed "$seed" --micro_batch 4 --grad_accum 32 \
      --eval_every 999999 --ckpt_every 100 --ckpt_path "$ckpt" --run_name "$cell" $resume \
      >> "run_${cell}.log" 2>&1 &
  CHILD=$!
  setsid nohup $PY "$REPO/sentinel.py" watch --pid "$CHILD" --kill-at 0.83 --log "$EXP/sentinel.log" >/dev/null 2>&1 &
  wait "$CHILD"; local rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then touch "arm_${cell}.done"; echo "[done] $cell"
  else echo "[FAIL] $cell rc=$rc"; exit "$rc"; fi
}

for seed in 0 1 2; do run_cell ""          "sft_seed${seed}";  done
for seed in 0 1 2; do run_cell "--no_mask" "ctrl_seed${seed}"; done
echo "ALL 6 CELLS COMPLETE ($(date '+%H:%M:%S'))"; touch cohort.done
