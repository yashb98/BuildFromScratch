#!/usr/bin/env bash
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"; cd "$EXP"
LOG="$EXP/post_cohort.log"
echo "[$(date '+%F %T')] watcher armed" >> "$LOG"
while [ ! -f "$EXP/cohort.done" ]; do
  pgrep -f "$EXP/run_arms.sh" >/dev/null || { [ -f "$EXP/cohort.done" ] || { echo "[$(date '+%F %T')] supervisor gone, no cohort.done -> needs_resume" >> "$LOG"; touch "$EXP/needs_resume"; exit 2; }; }
  sleep 300
done
echo "[$(date '+%F %T')] cohort.done -> scoring" >> "$LOG"
python3 "$EXP/score_verdict.py" >> "$LOG" 2>&1 && touch "$EXP/pipeline.done" || touch "$EXP/pipeline_error"
