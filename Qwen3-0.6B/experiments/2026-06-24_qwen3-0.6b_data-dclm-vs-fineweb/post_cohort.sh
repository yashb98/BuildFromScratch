#!/usr/bin/env bash
# CONDITIONAL trigger (NOT cron). Watches for the cohort-completion CONDITION and
# fires the deterministic results->verdict chain automatically, on the box, with no
# agent and no clock. Two outcomes are signalled by marker files the agent waits on:
#   pipeline.done  -> cohort finished + scored + per-axis verdict computed (do manuscript/next)
#   needs_resume   -> supervisor died WITHOUT cohort.done (re-run run_arms.sh to resume)
# Idempotent: exits immediately if pipeline.done already exists.
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../../.." && pwd)"
cd "$EXP"
LOG="$EXP/post_cohort.log"
echo "[$(date '+%F %T')] watcher armed (waiting for cohort.done)" >> "$LOG"

[ -f "$EXP/pipeline.done" ] && { echo "[$(date '+%F %T')] pipeline.done already present; exit" >> "$LOG"; exit 0; }

# 1) wait for the CONDITION; also detect a crash (supervisor gone, no cohort.done)
while [ ! -f "$EXP/cohort.done" ]; do
  if ! pgrep -f "run_arms.sh" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] run_arms.sh not running and no cohort.done -> needs_resume" >> "$LOG"
    touch "$EXP/needs_resume"; exit 2
  fi
  sleep 300
done
echo "[$(date '+%F %T')] cohort.done detected -> scoring 12 cells" >> "$LOG"

# 2) GPU is free now (cohort just finished) — preflight, then the deterministic chain
python3 "$REPO/sentinel.py" preflight >> "$LOG" 2>&1
python3 "$EXP/score_cohort.py" >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] score_cohort FAILED" >> "$LOG"; touch "$EXP/pipeline_error"; exit 1; }
python3 "$EXP/verdict.py"       >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] verdict FAILED" >> "$LOG"; touch "$EXP/pipeline_error"; exit 1; }

echo "[$(date '+%F %T')] PIPELINE DONE -> verdict.json ready; signalling pipeline.done" >> "$LOG"
touch "$EXP/pipeline.done"
