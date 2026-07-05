#!/usr/bin/env bash
# Sequential (arm × seed) cohort supervisor — Phase-2 ARCH SUB-COMPONENT DRILL.
# Splits the Phase-1-attributed "arch" axis into its 3 single-variable flags:
#   vr = use_value_residual | ln = use_layernorm_scaling | hg = use_head_gating
# 3 arms × 3 seeds = 9 NEW cells, 2000-step proxy (131M tok/cell, ~5h/cell, ~1.9d).
# Baseline is REUSED from Phase-1 (§C13 control-reuse: checkpoint_baseline_seed{0,1,2}.pt
# are symlinks to the de-confound dir) — NOT retrained here.
# ONE trainer at a time (§C4.5). Idempotent: a cell with its arm_<cell>.done marker is
# skipped; a partial cell resumes from its checkpoint (every 250 steps), so resume_cmd
# is just a re-run of THIS script. Per cell: sentinel.py on the TRAINER pid, --kill-at
# 0.83 (the Phase-1-verified guard for the mb4+compile unified-pool spike, under the
# 0.85 safe_cuda hard cap). Exits 0 ONLY after every cell completes -> clean exit = done.
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../../.." && pwd)"
cd "$EXP"
PY=python3
STEPS=2000; WARMUP=100

declare -A ARMS=(
  [vr]="--arch faithful --value_residual 1"
  [ln]="--arch faithful --layernorm_scaling 1"
  [hg]="--arch faithful --head_gating 1"
)
SEEDS="0 1 2"

CHILD=""
trap '[ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 143' TERM INT

for arm in vr ln hg; do
  for seed in $SEEDS; do
    cell="${arm}_seed${seed}"
    if [ -f "arm_${cell}.done" ]; then echo "[skip] $cell (done)"; continue; fi
    # inter-cell SETTLE: let the previous cell's unified-pool memory release before this
    # one compiles, so the brief overlap can't trip the sentinel guard.
    for _ in $(seq 1 24); do
      a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 5
    done
    echo "  [settle] MemAvailable $(free -g | awk '/Mem:/{print $7}')G before $cell"
    ckpt="checkpoint_${cell}.pt"
    resume=""
    [ -f "$ckpt" ] && resume="--resume $ckpt"
    echo "===== CELL $cell  ($(date '+%H:%M:%S')) $resume ====="
    # mb4 + COMPILE (defaults): IDENTICAL config to the Phase-1 arms (zero confound),
    # AdamW 1.7e-3 cosine. Single flag flipped per arm = single-variable (§C18).
    $PY train_subdrill.py ${ARMS[$arm]} --optimizer adamw --seed "$seed" \
        --steps "$STEPS" --warmup_steps "$WARMUP" --eval_every 500 --ckpt_every 250 \
        --ckpt_path "$ckpt" --run_name "$cell" $resume >> "run_${cell}.log" 2>&1 &
    CHILD=$!
    setsid nohup $PY "$REPO/sentinel.py" watch --pid "$CHILD" --kill-at 0.83 \
        --log "$EXP/sentinel.log" >/dev/null 2>&1 &
    wait "$CHILD"; rc=$?
    if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then
      touch "arm_${cell}.done"; echo "[done] $cell"
    else
      echo "[FAIL] $cell rc=$rc — stopping cohort (resume = re-run this script)"; exit "$rc"
    fi
  done
done
echo "ALL 9 CELLS COMPLETE ($(date '+%H:%M:%S'))"
touch cohort.done
