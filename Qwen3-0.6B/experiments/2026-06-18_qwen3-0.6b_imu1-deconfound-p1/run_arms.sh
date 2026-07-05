#!/usr/bin/env bash
# Sequential (arm × seed) cohort supervisor for the IMU-1 de-confound Phase 1.
# ONE trainer at a time (§C4.5). 4 arms × 3 seeds = 12 cells, 2000-step proxy
# (131M tok/cell, ~5h/cell, ~2.5 days total). Idempotent: a cell with its
# arm_<cell>.done marker is skipped; a partial cell resumes from its checkpoint
# (saved every 250 steps), so `resume_cmd` is just a re-run of THIS script.
# Per cell: arms sentinel.py on the TRAINER pid (§C6 default kill threshold).
# Exits 0 ONLY after every cell completes -> a clean exit = whole cohort done.
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../../.." && pwd)"
cd "$EXP"
PY=python3
STEPS=2000; WARMUP=100

declare -A ARMS=(
  [baseline]="--arch faithful --schedule cosine --zloss 0"
  [wsd]="--arch faithful --schedule wsd --zloss 0"
  [zloss]="--arch faithful --schedule cosine --zloss 1e-4"
  [arch]="--arch imu1 --schedule cosine --zloss 0"
)
SEEDS="0 1 2"

CHILD=""
trap '[ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 143' TERM INT

for arm in baseline wsd zloss arch; do
  for seed in $SEEDS; do
    cell="${arm}_seed${seed}"
    if [ -f "arm_${cell}.done" ]; then echo "[skip] $cell (done)"; continue; fi
    # inter-cell SETTLE: wait for the previous cell's unified-pool memory to release
    # before this cell compiles, so the brief overlap can't trip the 80% sentinel guard.
    for _ in $(seq 1 24); do
      a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 5
    done
    echo "  [settle] MemAvailable $(free -g | awk '/Mem:/{print $7}')G before $cell"
    ckpt="checkpoint_${cell}.pt"
    resume=""
    [ -f "$ckpt" ] && resume="--resume $ckpt"
    echo "===== CELL $cell  ($(date '+%H:%M:%S')) $resume ====="
    # compile ON (default) for the build's ~7,440 tok/s / ~52GB profile
    # --no_compile: the model's torch.compile startup transiently spikes the unified
    # pool to ~82GB (151936-vocab logits) and trips the 80% sentinel; no-compile runs
    # at ~75GB steady (smoke-verified). compile preserves training numerics, so the
    # baseline/wsd (compiled) vs zloss/arch (no-compile) difference is FP-reordering
    # only — below the seed-noise floor. Recorded as a known caveat in the verdict.
    # mb4 + COMPILE (defaults): the original ~6,800 tok/s config, IDENTICAL to baseline/wsd
    # (zero confound). Its ~82GB compile spike peaks the pool at ~80.5%; the sentinel below
    # is raised to --kill-at 0.83 (user directive) so this clears the guard, while
    # safe_cuda still hard-caps CUDA at 0.85 as the clean-error backstop.
    $PY train_ablation.py ${ARMS[$arm]} --optimizer adamw --seed "$seed" \
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
echo "ALL 12 CELLS COMPLETE ($(date '+%H:%M:%S'))"
touch cohort.done
