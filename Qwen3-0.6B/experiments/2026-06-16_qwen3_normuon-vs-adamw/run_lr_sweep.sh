#!/usr/bin/env bash
# Confirmatory AdamW LR sweep at the EXACT ablation config (42M tok, wd=0.1, 2D
# AdamW split, seed 0, 640 steps) — to check whether our 2.4e-3 (already run with
# 3 seeds as adamw_seed0/1/2) is the best AdamW LR for THIS budget+wd, killing the
# 'undertuned baseline' confound. Single seed per LR is enough to find the best.
# Sequential on the single GB10 (parallel would over-commit -> crash).
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$DIR/results"; mkdir -p "$RES"
PROG="$DIR/train_ablation.py"
SUP_LOG="$RES/lr_sweep_supervisor.log"
log(){ echo "[$(date -Is)] $*" | tee -a "$SUP_LOG"; }

# LR -> tag  (2.4e-3 already covered by adamw_seed0/1/2; sweep the others)
declare -A SWEEP=( [1.7e-3]=adamw_lr17_seed0 [3.5e-3]=adamw_lr35_seed0 [4.8e-3]=adamw_lr48_seed0 )

run_cell(){  # $1=lr $2=tag
  local lr="$1" tag="$2" done="$RES/${2}.done" ck="$RES/checkpoint_${2}.pt"
  if [ -f "$done" ] && [ -f "$ck" ]; then log "SKIP $tag (done)"; return 0; fi
  for attempt in 1 2; do
    log "RUN  $tag  lr=$lr  attempt $attempt"
    python3 "$PROG" --optimizer adamw --seed 0 --steps 640 --peak_lr "$lr" --tag "$tag" \
        >>"$RES/${tag}.out" 2>&1 &
    local pid=$!; echo "$pid" > "$RES/${tag}.pid"; wait "$pid"; local rc=$?
    # a NaN-divergence at high LR still saves a ckpt+exits 0 (informative: bad LR);
    # only a real crash (rc!=0 AND no ckpt) is a retryable failure.
    if [ -f "$ck" ]; then touch "$done"; log "DONE $tag (rc=$rc, ckpt present)"; return 0; fi
    log "FAIL $tag attempt $attempt (rc=$rc, no ckpt)"
  done
  log "GIVEUP $tag"; return 1
}

log "==== AdamW confirmatory LR sweep start: 3 LRs x 1 seed @ 42M tok, wd=0.1 ===="
t0=$(date +%s); ok=0
for lr in 1.7e-3 3.5e-3 4.8e-3; do
  run_cell "$lr" "${SWEEP[$lr]}" && ok=$((ok+1))
done
log "==== sweep complete: $ok/3 ok in $(( ($(date +%s)-t0)/60 )) min ===="
touch "$RES/LR_SWEEP_COMPLETE"
