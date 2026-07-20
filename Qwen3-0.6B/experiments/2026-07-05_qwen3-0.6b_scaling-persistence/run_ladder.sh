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
COOL_C=70   # 2026-07-08: user heat-crash hypothesis — cool the box below this (C) before each cell
mkdir -p "$LDIR"

# Hottest of the GPU die + all ACPI SoC zones, whole deg C (empty if unreadable). No sudo.
hottest_c () {
  local g z zc max=""
  g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  [ -n "$g" ] && max=$g
  for z in /sys/class/thermal/thermal_zone*/temp; do
    [ -r "$z" ] || continue
    zc=$(( $(cat "$z" 2>/dev/null || echo 0) / 1000 ))
    { [ -z "$max" ] || [ "$zc" -gt "$max" ]; } && max=$zc
  done
  echo "$max"
}

# Bounded, fail-open cool-down: wait up to ~30 min for the box to drop below COOL_C before
# launching a cell (implements the heat-crash mitigation). Unreadable temp => proceed.
cool_down () {
  local tag=$1 h
  for _ in $(seq 1 60); do
    h=$(hottest_c)
    [ -z "$h" ] && { echo "[$(date '+%T')] [cooldown] $tag: temp unreadable, proceeding"; return 0; }
    [ "$h" -lt "$COOL_C" ] && { echo "[$(date '+%T')] [cooldown] $tag: ${h}C < ${COOL_C}C, launching"; return 0; }
    echo "[$(date '+%T')] [cooldown] $tag: ${h}C >= ${COOL_C}C, waiting 30s for cool-down"
    sleep 30
  done
  echo "[$(date '+%T')] [cooldown] $tag: still >= ${COOL_C}C after 30min — launching anyway (bounded)"
}

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
  cool_down "$tag"   # 2026-07-08: don't launch onto a hot box (heat-crash mitigation)
  # pool headroom (unified memory shared; wait for >=60 GB free)
  for _ in $(seq 1 90); do a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 10; done
  echo "[$(date '+%F %T')] START $tag steps=$steps (~$(( steps * 65536 / 1000000 ))M tok)"
  $PY "$TRAIN" --optimizer "$arm" --seed "$seed" --steps "$steps" --tag "$tag" --resume_every 100 \
      >> "$RESULTS/${tag}.out" 2>&1 &
  local tpid=$!
  $PY "$ROOT/sentinel.py" watch --pid "$tpid" --kill-at 0.80 --log "$LDIR/sentinel_${tag}.log" \
      >/dev/null 2>&1 &
  local spid=$!
  wait "$tpid"; local rc=$?
  kill "$spid" 2>/dev/null
  if [ $rc -eq 0 ] && [ -f "$RESULTS/checkpoint_${tag}.pt" ]; then
    touch "$LDIR/${tag}.done"; echo "[$(date '+%T')] [done] $tag"; return 0
  else
    echo "[$(date '+%T')] [FAIL rc=$rc] $tag — will retry on next driver pass"; return 1
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
# 4) SMOKE (§C5.0): 1 step, no compile, confirms model build + data + step + checkpoint save.
# Watched by sentinel (§C6) like the cells — it is an unattended GPU launch.
echo "[$(date '+%T')] smoke: train_ablation.py --steps 1"
$PY "$TRAIN" --optimizer adamw --seed 0 --steps 1 --no_compile --tag smoke_ladder >> "$LDIR/smoke.log" 2>&1 &
smpid=$!
$PY "$ROOT/sentinel.py" watch --pid "$smpid" --kill-at 0.80 --log "$LDIR/sentinel_smoke_ladder.log" >/dev/null 2>&1 &
smwatch=$!
wait "$smpid"; smrc=$?
kill "$smwatch" 2>/dev/null
if [ $smrc -ne 0 ] || [ ! -f "$RESULTS/checkpoint_smoke_ladder.pt" ]; then
  echo "SMOKE FAILED — abort (see $LDIR/smoke.log)"; exit 1
fi
rm -f "$RESULTS/checkpoint_smoke_ladder.pt"; echo "[$(date '+%T')] smoke OK"
# 5) train the core cells (one at a time; .done markers make this idempotent/resumable).
# 2026-07-09 fix: LOOP over passes so a thermally-killed cell is RE-ATTEMPTED (resuming from its
# checkpoint via --resume_every) instead of the driver quitting after ONE pass (the 07-09 stall:
# midday heat SIGTERM'd all 4 cells in a row, then the driver exited and nothing ran for ~17h).
# Each pass skips .done cells instantly and cools down before every (re)attempt. Bounded by MAXPASS.
pass=0; MAXPASS=100; hot_backoff=0; prev_missing=999
while : ; do
  fails=0
  for c in "${CELLS[@]}"; do run_cell $c || fails=$((fails+1)); done
  missing=0
  for c in "${CELLS[@]}"; do
    set -- $c; s=$1; arm=$2; seed=$3; bM=$(( s * 65536 / 1000000 ))
    [ -f "$LDIR/persist_${bM}M_${arm}_s${seed}.done" ] || missing=$((missing+1))
  done
  [ "$missing" -eq 0 ] && break
  pass=$((pass+1))
  [ "$pass" -ge "$MAXPASS" ] && { echo "[$(date '+%T')] MAXPASS=$MAXPASS reached, $missing cells incomplete — stopping (resume ckpts preserved; re-run to continue)"; break; }
  # HOT-SPELL BACKOFF (2026-07-10): if a whole pass FAILED with NO cell completing (missing didn't
  # drop) and there were failures, the box is too hot to make progress — during peak-heat spikes a
  # cell reheats to 90C and thermal-dies in <20 steps, thrashing the GPU at ~0 net progress. Wait an
  # escalating (5→30 min cap) cool-off for a cooler window instead of re-attempting every ~3 min.
  # Any completion (missing drops) resets the backoff so cool hours run at full speed.
  if [ "$fails" -gt 0 ] && [ "$missing" -ge "$prev_missing" ]; then
    hot_backoff=$(( hot_backoff + 1 )); wait_s=$(( hot_backoff * 300 )); [ "$wait_s" -gt 1800 ] && wait_s=1800
    echo "[$(date '+%F %T')] pass $pass: $missing/${#CELLS[@]} incomplete, no progress ($fails failed — box likely too hot) — backing off ${wait_s}s for a cooler window"
    sleep "$wait_s"
  else
    hot_backoff=0
    echo "[$(date '+%F %T')] pass $pass: $missing/${#CELLS[@]} cells incomplete ($fails failed) — re-attempting (cells resume from checkpoint)"
  fi
  prev_missing=$missing
done
# 5b) COMPLETION GATE (2026-07-07 fix): only mark the ladder complete + score when EVERY cell
# has its .done marker (run_cell writes .done only on rc==0 AND a saved checkpoint). Stops a
# failed cell from producing a FALSE 'CORE LADDER COMPLETE' + a scorer/ledger run over an
# incomplete ladder (the 09:39 bug: all 4x420M failed on a DNS blip yet ladder.done was set).
missing=0
for c in "${CELLS[@]}"; do
  set -- $c; s=$1; arm=$2; seed=$3; bM=$(( s * 65536 / 1000000 ))
  [ -f "$LDIR/persist_${bM}M_${arm}_s${seed}.done" ] || missing=$((missing+1))
done
if [ "$missing" -eq 0 ]; then
  touch "$LDIR/ladder.done"
  echo "===== $(date '+%F %T') CORE LADDER COMPLETE — $LDIR/ladder.done ====="
  # 6) score only when the ladder is genuinely complete (all cells .done)
  [ -f "$LDIR/score_ladder.py" ] && $PY "$LDIR/score_ladder.py" >> "$LOG" 2>&1
else
  rm -f "$LDIR/ladder.done"
  echo "===== $(date '+%F %T') LADDER INCOMPLETE — $missing/${#CELLS[@]} cells missing .done ($fails failed this pass); NOT scoring, ladder.done cleared ====="
  exit 1
fi
} >> "$LOG" 2>&1
