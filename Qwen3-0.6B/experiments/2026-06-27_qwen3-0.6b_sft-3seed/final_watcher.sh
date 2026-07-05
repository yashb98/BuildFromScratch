#!/usr/bin/env bash
# Post-cohort watcher for the SFT 3-seed + iso-FLOP control cohort. Waits for run_arms.sh to
# touch cohort.done, then runs BOTH verdicts one-at-a-time (GPU is free once training is done):
#   1. score_verdict.py   - advisory in-loop number (train-set proxy; flags the eval CONFOUND)
#   2. reasoning_eval.py  - the REAL held-out, uniformly-masked SFT-vs-control verdict (§C10-style)
# Sustained-dead (no run_arms.sh / no sft_train.py / still no cohort.done) -> needs_resume marker,
# so a second reboot mid-control is caught for /research-loop resume. Idempotent + retrying.
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"; cd "$EXP"; LOG="$EXP/post_cohort.log"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # reasoning_eval reconstructs prompt_len offline
echo "[$(date '+%F %T')] final_watcher armed (advisory score_verdict + corrected held-out reasoning_eval)" >> "$LOG"
dead=0
while [ ! -f "$EXP/cohort.done" ]; do
  if pgrep -f "$EXP/run_arms.sh" >/dev/null || pgrep -f "sft_train.py" >/dev/null; then dead=0
  else dead=$((dead+1)); [ "$dead" -ge 3 ] && { echo "[$(date '+%F %T')] sustained-dead, no cohort.done -> needs_resume" >> "$LOG"; touch "$EXP/needs_resume"; exit 2; }; fi
  sleep 120
done
echo "[$(date '+%F %T')] cohort.done -> [1/2] advisory in-loop scorer" >> "$LOG"
for t in 1 2 3; do python3 "$EXP/score_verdict.py" >> "$LOG" 2>&1 && break; sleep $((t*60)); done
echo "[$(date '+%F %T')] -> [2/2] corrected HELD-OUT reasoning verdict (GPU)" >> "$LOG"
for t in 1 2 3; do
  python3 "$EXP/reasoning_eval.py" >> "$LOG" 2>&1 \
    && { touch "$EXP/pipeline.done"; echo "[$(date '+%F %T')] DONE -> reasoning_verdict.json" >> "$LOG"; exit 0; }
  sleep $((t*120))
done
touch "$EXP/pipeline_error"; echo "[$(date '+%F %T')] reasoning_eval failed after retries" >> "$LOG"
