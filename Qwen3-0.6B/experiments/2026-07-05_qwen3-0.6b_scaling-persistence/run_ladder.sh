#!/usr/bin/env bash
# Scaling persistence ladder driver (research/scaling/plan.md). Closes IMU-1 RESULT.md Limitation #3:
# does NorMuon's +0.474 BPB win PERSIST or vanish with budget? Reuses the IMU-1 train_ablation.py at
# fixed N=596M, sweeps the token budget. 3-budget CORE first (42M reused + 168M + 420M ~= 4 GPU-days);
# 840M is a later extension (the trainer has no mid-run resume, so cells are kept <=~27h; .done markers
# give cell-level resumability across the box's occasional hard-locks).
#
# ARMED: waits for the GPU to free (RLVR verdict.json written = rft done + cohort scored) before ANY
# launch. One GPU job at a time (§C4.5). Smoke-first (§C5.0). sentinel watch --kill-at 0.80 per cell.
set -u
ROOT=/home/yashb98/Downloads/BuildFromScratch
IMU1=$ROOT/Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw
LDIR=$ROOT/Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence
RLVR_VERDICT=$ROOT/Qwen3-0.6B/experiments/2026-07-02_qwen3-0.6b_grpo-phase2/verdict.json
TRAIN=$IMU1/train_ablation.py
RESULTS=$IMU1/results
LOG=$LDIR/run_ladder.log
PY=python3
mkdir -p "$LDIR"

# CORE cells: "<steps> <arm> <seed>". 168M=2564 steps (3 seeds/arm), 420M=6409 steps (2 seeds/arm).
CELLS=(
 "2564 adamw 0" "2564 normuon 0" "2564 adamw 1" "2564 normuon 1" "2564 adamw 2" "2564 normuon 2"
 "6409 adamw 0" "6409 normuon 0" "6409 adamw 1" "6409 normuon 1"
)

run_cell () {
  local steps=$1 arm=$2 seed=$3
  local budgetM=$(( steps * 65536 / 1000000 ))
  local tag=persist_${budgetM}M_${arm}_s${seed}
  [ -f "$LDIR/${tag}.done" ] && { echo "[$(date '+%T')] [skip] $tag"; return 0; }
  # pool headroom (unified memory shared; wait for >=60 GB free)
  for _ in $(seq 1 90); do a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 10; done
  echo "[$(date '+%F %T')] START $tag steps=$steps (~$(( steps * 65536 / 1000000 ))M tok)"
  $PY "$TRAIN" --optimizer "$arm" --seed "$seed" --steps "$steps" --tag "$tag" \
      >> "$RESULTS/${tag}.out" 2>&1 &
  local tpid=$!
  $PY "$ROOT/sentinel.py" watch --pid "$tpid" --kill-at 0.80 --log "$LDIR/sentinel_${tag}.log" \
      >/dev/null 2>&1 &
  local spid=$!
  wait "$tpid"; local rc=$?
  kill "$spid" 2>/dev/null
  if [ $rc -eq 0 ] && [ -f "$RESULTS/checkpoint_${tag}.pt" ]; then
    touch "$LDIR/${tag}.done"; echo "[$(date '+%T')] [done] $tag"
  else
    echo "[$(date '+%T')] [FAIL rc=$rc] $tag — will retry on next driver pass"
  fi
}

{
echo "===== $(date '+%F %T') scaling-persistence ladder armed (waiting for GPU to free) ====="
# 1) wait for the RLVR chapter to close (verdict.json written = GPU free), up to ~24h
for _ in $(seq 1 8640); do [ -f "$RLVR_VERDICT" ] && { echo "[$(date '+%T')] RLVR verdict seen — GPU freeing"; break; }; sleep 10; done
[ -f "$RLVR_VERDICT" ] || { echo "TIMEOUT waiting for RLVR verdict (>24h) — abort"; exit 1; }
# 2) wait until no GPU trainer/scorer is live
for _ in $(seq 1 360); do pgrep -f "train_grpo.py|run_phase1_passk.py|train_ablation.py" >/dev/null 2>&1 || break; sleep 20; done
sleep 5
# 3) preflight (§C6)
$PY "$ROOT/sentinel.py" preflight || { echo "preflight FAIL — abort"; exit 1; }
# 4) SMOKE (§C5.0): 1 step, no compile, confirms model build + data + step + checkpoint save
echo "[$(date '+%T')] smoke: train_ablation.py --steps 1"
$PY "$TRAIN" --optimizer adamw --seed 0 --steps 1 --no_compile --tag smoke_ladder >> "$LDIR/smoke.log" 2>&1
if [ $? -ne 0 ] || [ ! -f "$RESULTS/checkpoint_smoke_ladder.pt" ]; then
  echo "SMOKE FAILED — abort (see $LDIR/smoke.log)"; exit 1
fi
rm -f "$RESULTS/checkpoint_smoke_ladder.pt"; echo "[$(date '+%T')] smoke OK"
# 5) train the core cells (one at a time; .done markers make this idempotent/resumable)
for c in "${CELLS[@]}"; do run_cell $c; done
touch "$LDIR/ladder.done"
echo "===== $(date '+%F %T') CORE LADDER COMPLETE — $LDIR/ladder.done ====="
# 6) score if a scorer is present (armed separately, like the RLVR autoscore)
[ -f "$LDIR/score_ladder.py" ] && $PY "$LDIR/score_ladder.py" >> "$LOG" 2>&1
} >> "$LOG" 2>&1
