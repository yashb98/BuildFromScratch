#!/usr/bin/env bash
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"; cd "$EXP"; LOG="$EXP/post_cohort.log"
echo "[$(date '+%F %T')] FIXED scorer armed" >> "$LOG"
dead=0
while [ ! -f "$EXP/cohort.done" ]; do
  if pgrep -f "$EXP/run_arms.sh" >/dev/null || pgrep -f "sft_train.py" >/dev/null; then dead=0
  else dead=$((dead+1)); [ $dead -ge 3 ] && { echo "[$(date '+%F %T')] sustained-dead, no cohort.done -> needs_resume" >> "$LOG"; touch "$EXP/needs_resume"; exit 2; }; fi
  sleep 120
done
echo "[$(date '+%F %T')] cohort.done -> scoring" >> "$LOG"
for t in 1 2 3 4; do python3 "$EXP/score_verdict.py" >> "$LOG" 2>&1 && { touch "$EXP/pipeline.done"; exit 0; }; sleep $((t*120)); done
touch "$EXP/pipeline_error"
