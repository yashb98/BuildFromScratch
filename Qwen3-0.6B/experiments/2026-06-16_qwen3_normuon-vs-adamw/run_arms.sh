#!/usr/bin/env bash
# 6-cell ablation supervisor: NorMuon vs AdamW, 3 seeds each, run SEQUENTIALLY on
# the single GB10 (parallel cells would double unified-memory pressure -> crash).
# Each cell: skip if its .done marker exists (resumable), one retry on failure,
# log to results/. A crashed cell does NOT abort the cohort. The python child PID
# is written to results/<cell>.pid so a watchdog can target the real trainer.
#
# Usage:  bash run_arms.sh <STEPS>
set -u
STEPS="${1:?usage: run_arms.sh <STEPS>}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$DIR/results"; mkdir -p "$RES"
PROG="$DIR/train_ablation.py"
SUP_LOG="$RES/supervisor.log"

log(){ echo "[$(date -Is)] $*" | tee -a "$SUP_LOG"; }

run_cell(){  # $1=arm $2=seed
  local arm="$1" seed="$2" cell="${1}_seed${2}"
  local done="$RES/${cell}.done" ck="$RES/checkpoint_${cell}.pt"
  if [ -f "$done" ] && [ -f "$ck" ]; then log "SKIP $cell (already done)"; return 0; fi
  for attempt in 1 2; do
    log "RUN  $cell  attempt $attempt  (steps=$STEPS)"
    python3 "$PROG" --optimizer "$arm" --seed "$seed" --steps "$STEPS" \
        >>"$RES/${cell}.out" 2>&1 &
    local pid=$!
    echo "$pid" > "$RES/${cell}.pid"
    wait "$pid"; local rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$ck" ]; then
      touch "$done"; log "DONE $cell (rc=0)"; return 0
    fi
    log "FAIL $cell attempt $attempt (rc=$rc)"
  done
  log "GIVEUP $cell after 2 attempts"; return 1
}

log "==== ablation cohort start: steps=$STEPS, 6 cells (2 arms x 3 seeds) ===="
t0=$(date +%s)
ok=0
for arm in adamw normuon; do
  for seed in 0 1 2; do
    run_cell "$arm" "$seed" && ok=$((ok+1))
  done
done
dt=$(( ($(date +%s) - t0) / 60 ))
log "==== cohort complete: $ok/6 cells ok in ${dt} min ===="
touch "$RES/COHORT_COMPLETE"
