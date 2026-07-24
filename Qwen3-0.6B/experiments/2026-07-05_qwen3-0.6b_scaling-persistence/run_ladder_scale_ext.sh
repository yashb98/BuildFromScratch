#!/usr/bin/env bash
# NorMuon-at-scale extension driver — upgrade-plan #9 (research/LOOP_UPGRADE_PLAN_2026-07-22.md).
#
# Extends the CORE scaling-persistence ladder (run_ladder.sh, CORE = 42M+168M+420M, COMPLETE) with
# the two rungs that resolve the flagship open question "does NorMuon's +0.474 BPB win over AdamW
# PERSIST or converge with token budget?":
#   - 420M 3rd seed  (s2 pair): takes the current top rung from n=2 -> n=3 paired seeds, so the 420M
#                               gap finally carries a real across-seed CI (§C17) instead of directional.
#   - 840M 1st seed  (s0 pair): a new higher-budget point to extend the gap-vs-budget trend.
# Cells are ordered 420M-s2 FIRST so the CI-completing half lands before the (much longer) 840M rung.
#
# Same fixed N=596M trainer (train_ablation.py), same results dir, same .done markers as CORE, so any
# already-complete cell is skipped instantly. train_ablation.py has mid-run resume (--resume_every), so
# a thermally-killed cell resumes from its checkpoint on the next pass (the CORE header's "no mid-run
# resume" note is stale — the crash-survivable upgrade landed 2026-07-08).
#
# SAFETY (2026-07-23): the cool-down gate is the FIXED version. The CORE run_ladder.sh (and the
# HybridSSM arch ladder modelled on it) shipped a cool-down that accepted ONE sub-threshold sample,
# LAUNCHED ANYWAY after 30 min, and discarded its own return value — which let a cell relaunch onto a
# heat-soaked box ~60s after a thermal kill, reheat past 90C, and re-kill. That thrash preceded the
# 2026-07-23 15:24 BST hard-lock. This driver requires SUSTAINED cool (dwell), DEFERS on the bounded
# fall-through, and honours the return value so the hot-spell backoff actually engages.
# One GPU job at a time (§C4.5). Smoke-first (§C5.0). sentinel watch --kill-at 0.80 per cell (§C6).
set -u
ROOT=/home/yashb98/Downloads/BuildFromScratch
IMU1=$ROOT/Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw
LDIR=$ROOT/Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence
TRAIN=$IMU1/train_ablation.py
RESULTS=$IMU1/results
LOG=$LDIR/run_ladder_scale_ext.log
PY=python3

# COOL_C (2026-07-23, corrected): cool_down compares COOL_C to hottest_c = max(GPU die, ALL ACPI SoC
# zones). The SoC zones idle at ~64-68C on this box, so a threshold BELOW that floor (an earlier 58C
# try) can never be satisfied and defers every cell forever. sentinel WARNs at 82C and KILLs at 90C
# (3 consecutive). 72C sits just above the ~64-68C idle floor (so a sustained-cool dwell is reachable),
# 10C under WARN and 18C under KILL. The proven CORE run_ladder.sh ran the whole 420M ladder at 70C;
# 72C matches that with dwell headroom. The REAL fix for the 2026-07-23 thrash is the sustained dwell +
# DEFER below (a heat-soaked box dipping through 72C mid-cooldown can't satisfy 6 consecutive samples),
# NOT a low threshold. Overridable, but keep it above the idle floor and well under WARN.
COOL_C="${LADDER_COOL_C:-72}"
COOL_DWELL="${LADDER_COOL_DWELL:-6}"   # consecutive sub-COOL_C samples required (6 x 30s = 3 min sustained)
COOL_MAX="${LADDER_COOL_MAX:-30}"      # bounded wait: 30 samples = 15 min, then DEFER (never "launch anyway")

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

# SUSTAINED-cool gate. Requires COOL_DWELL consecutive samples strictly below COOL_C (the streak resets
# on any hot sample), so a box that is merely dipping between load spikes cannot pass. Returns 0 only on
# a real sustained-cool window; returns 1 if still hot after the bounded wait (the caller DEFERS the cell
# to the next pass, where the hot-spell backoff engages). Unreadable temp => proceed (never fabricate heat).
cool_down () {
  local tag=$1 h streak=0 i=0
  while [ "$i" -lt "$COOL_MAX" ]; do
    h=$(hottest_c)
    [ -z "$h" ] && { echo "[$(date '+%T')] [cooldown] $tag: temp unreadable, proceeding"; return 0; }
    if [ "$h" -lt "$COOL_C" ]; then
      streak=$(( streak + 1 ))
      [ "$streak" -ge "$COOL_DWELL" ] && { echo "[$(date '+%T')] [cooldown] $tag: ${h}C < ${COOL_C}C sustained ${streak}x, launching"; return 0; }
    else
      [ "$streak" -ne 0 ] && echo "[$(date '+%T')] [cooldown] $tag: ${h}C >= ${COOL_C}C, streak reset"
      streak=0
    fi
    sleep 30; i=$(( i + 1 ))
  done
  echo "[$(date '+%T')] [cooldown] $tag: box did not hold < ${COOL_C}C for ${COOL_DWELL} samples within $(( COOL_MAX / 2 )) min — DEFER to next pass"
  return 1
}

# ---- Proactive thermal GOVERNOR (2026-07-23, user-requested) -----------------------------------------
# A SOFT layer BELOW sentinel's 90C hard kill. While a cell trains, this watches hottest_c (GPU die + all
# SoC zones). At/above PAUSE_C it SIGSTOPs the trainer — the process freezes, the GPU goes idle and the box
# cools, with ZERO lost progress (in-memory state is preserved; no checkpoint/restart). It then rechecks
# every GOV_PAUSE_INTERVAL (3 min, per the user) and SIGCONTs once the box drops below RESUME_C. This keeps
# the box out of the 90C zone that preceded the 2026-07-23 hard-lock while making continuous forward
# progress. Checkpoints (--resume_every 100) stay as the backstop for an actual hard-lock.
# PAUSE_C=80 (not the user's literal 85): live 2026-07-23 test showed the 596M load + concurrent GPU work
# heats the box ~21C/min, so with a 30s sample an 85C threshold OVERSHOT to 90-91C — right at sentinel's
# 90C hard-kill / the hard-lock line. 80C + a 10s sample catches it at ~82-83C, keeping a real ~7C margin
# below 90. RESUME_C=75 with the 3-min cool check (user spec) still deep-cools to ~58C each pause, so the
# run window stays ~60s (measured ~25% duty). Raise LADDER_PAUSE_C back to 85 only if the box runs cooler.
# 2026-07-24 RETUNE for "make it work under permanent heavy concurrent load": the user keeps a dozen
# CPU/GPU processes running, so the shared die barely cools; the old 3-min cooldown check left the trainer
# idle 85-90% of the time (it resumed long after the die was already cool enough). This is now a RESPONSIVE
# duty-cycle governor: pause at 80C, resume the moment it dips to 78C, rechecking every 5s — so it holds the
# die in a tight 78-80C band and runs as much of the time as the box's cooling allows (~3x the old duty),
# while staying 10C under sentinel's 90C kill. Safety is unchanged (PAUSE_C + sentinel backstop); only the
# resume latency shrank. Widen LADDER_RESUME_C down / LADDER_GOV_PAUSE up to trade duty for a cooler die.
# 2026-07-24 measured: an 80/78 band gave 50% duty but PEAKED 88C (only 2C under the 90C kill) — the
# ~8C overshoot from thermal inertia + concurrent-load spikes is too much on a box that hard-locked at
# 92C today. Pulled the band down to 76/72 and sped run-sampling to 3s: peak now ~82-84C (a ~7C margin),
# still ~40-45% duty because the die cools fast to ~50C when idle so a 4C band cycles quickly.
PAUSE_C="${LADDER_PAUSE_C:-76}"                 # SIGSTOP at/above this — leaves room for the ~6-8C overshoot
RESUME_C="${LADDER_RESUME_C:-72}"               # SIGCONT at this (4C band; cooling is fast so windows stay long)
GOV_RUN_INTERVAL="${LADDER_GOV_RUN:-3}"         # sample every 3s while running — catch 76C before it overshoots far
GOV_PAUSE_INTERVAL="${LADDER_GOV_PAUSE:-5}"     # recheck every 5s while cooling — resume promptly

thermal_governor () {
  local pid=$1 tag=$2 paused=0 h
  while kill -0 "$pid" 2>/dev/null; do
    h=$(hottest_c)
    if [ -n "$h" ]; then
      if [ "$paused" -eq 0 ]; then
        if [ "$h" -ge "$PAUSE_C" ] && kill -STOP "$pid" 2>/dev/null; then
          paused=1
          echo "[$(date '+%T')] [gov] $tag: ${h}C >= ${PAUSE_C}C -> PAUSED (SIGSTOP) to cool"
        fi
      else
        if [ "$h" -lt "$RESUME_C" ] && kill -CONT "$pid" 2>/dev/null; then
          paused=0
          echo "[$(date '+%T')] [gov] $tag: ${h}C < ${RESUME_C}C -> RESUMED (SIGCONT)"
        else
          echo "[$(date '+%T')] [gov] $tag: ${h}C >= ${RESUME_C}C, still cooling (recheck in $((GOV_PAUSE_INTERVAL/60))min)"
        fi
      fi
    fi
    if [ "$paused" -eq 1 ]; then sleep "$GOV_PAUSE_INTERVAL"; else sleep "$GOV_RUN_INTERVAL"; fi
  done
  kill -CONT "$pid" 2>/dev/null   # never leave a now-exited trainer in a stopped state
}

# EXTENSION cells: "<steps> <arm> <seed>". 6409 steps = 420M tok (~17.5 h/cell measured), 12818 = 840M (~35 h).
# 2026-07-23 LAUNCH SCOPE = the 420M-s2 pair ONLY (user-authorized): completes the n=3 across-seed CI at
# the current top rung (~35 h), then STOPS. The 840M s0 pair is DEFERRED behind the firmware/kdump fix —
# the box hard-locked today under a lighter load, so the heaviest rung waits for the durable fix + a reboot.
# To run the 840M rung later, uncomment its line below and re-launch (done cells are skipped).
CELLS=(
 "6409 adamw 2" "6409 normuon 2"
 # "12818 adamw 0" "12818 normuon 0"   # 840M — DEFERRED (enable after enable_kdump_gb10.sh + OTA 7.5.0 + reboot)
)

# Is a REAL foreign python trainer alive? Require argv[0] to be a python interpreter (a bare
# `pgrep -f train_*.py` self-matches greps/monitors/this driver). Never count ourselves. Returns
# 0 (true) only if some OTHER process is a genuine python trainer.
trainer_alive () {
  local p exe
  for p in $(pgrep -f 'train_[A-Za-z0-9_]*\.py' 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    exe=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)
    case "$(basename "${exe:-none}")" in python*) return 0 ;; esac
  done
  return 1
}

run_cell () {
  local steps=$1 arm=$2 seed=$3
  local budgetM=$(( steps * 65536 / 1000000 ))
  local tag=persist_${budgetM}M_${arm}_s${seed}
  [ -f "$LDIR/${tag}.done" ] && { echo "[$(date '+%T')] [skip] $tag"; return 0; }
  # §C4.5: never two trainers — a foreign trainer means someone else owns the GPU, wait it out.
  if trainer_alive; then
    echo "[$(date '+%T')] [wait] $tag: another python trainer is alive, deferring this pass"; return 1
  fi
  if ! cool_down "$tag"; then echo "[$(date '+%T')] [defer] $tag: box too hot"; return 1; fi
  # pool headroom (unified memory shared; wait for >=60 GB free — a 596M cell runs at ~51.5 GB)
  for _ in $(seq 1 90); do a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 10; done
  echo "[$(date '+%F %T')] START $tag steps=$steps (~${budgetM}M tok)"
  # --resume_every 100 (not 200): the 596M load hits sentinel's 90C thermal kill in ~28 min on this box
  # (2026-07-23), close to the step-200 checkpoint interval — so checkpoint every 100 steps (~17 min) to
  # guarantee forward progress survives each thermal kill, even if the box still runs warm post-firmware-fix.
  $PY "$TRAIN" --optimizer "$arm" --seed "$seed" --steps "$steps" --tag "$tag" --resume_every 100 \
      >> "$RESULTS/${tag}.out" 2>&1 &
  local tpid=$!
  $PY "$ROOT/sentinel.py" watch --pid "$tpid" --kill-at 0.80 --log "$LDIR/sentinel_${tag}.log" \
      >/dev/null 2>&1 &
  local spid=$!
  # Proactive thermal governor: SIGSTOP/SIGCONT the trainer to hold it under PAUSE_C (soft, below the
  # sentinel 90C hard kill). Runs beside sentinel; sentinel remains the memory + last-resort thermal guard.
  thermal_governor "$tpid" "$tag" >> "$LDIR/gov_${tag}.log" 2>&1 &
  local gpid=$!
  wait "$tpid"; local rc=$?
  kill -CONT "$tpid" 2>/dev/null    # if the trainer exited while SIGSTOP'd, don't leave it stopped
  kill "$gpid" 2>/dev/null          # stop the governor
  kill "$spid" 2>/dev/null          # stop the sentinel watcher
  if [ $rc -eq 0 ] && [ -f "$RESULTS/checkpoint_${tag}.pt" ]; then
    touch "$LDIR/${tag}.done"; echo "[$(date '+%T')] [done] $tag"; return 0
  else
    echo "[$(date '+%T')] [FAIL rc=$rc] $tag — will retry on next pass (resumes from checkpoint)"; return 1
  fi
}

{
echo "===== $(date '+%F %T') NorMuon-at-scale extension driver start (pid $$) ====="
# 1) no GPU trainer live
for _ in $(seq 1 360); do pgrep -f "train_grpo.py|run_phase1_passk.py|train_ablation.py|train_hybrid.py" >/dev/null 2>&1 || break; sleep 20; done
sleep 5
# 2) preflight (§C6)
$PY "$ROOT/sentinel.py" preflight || { echo "preflight FAIL — abort"; exit 1; }
# 3) SMOKE (§C5.0): 1 step, no compile — confirms model build + data + step + checkpoint save.
echo "[$(date '+%T')] smoke: train_ablation.py --steps 1"
$PY "$TRAIN" --optimizer adamw --seed 0 --steps 1 --no_compile --tag smoke_scale_ext >> "$LDIR/smoke_scale_ext.log" 2>&1 &
smpid=$!
$PY "$ROOT/sentinel.py" watch --pid "$smpid" --kill-at 0.80 --log "$LDIR/sentinel_smoke_scale_ext.log" >/dev/null 2>&1 &
smwatch=$!
wait "$smpid"; smrc=$?
kill "$smwatch" 2>/dev/null
if [ $smrc -ne 0 ] || [ ! -f "$RESULTS/checkpoint_smoke_scale_ext.pt" ]; then
  echo "SMOKE FAILED — abort (see $LDIR/smoke_scale_ext.log)"; exit 1
fi
rm -f "$RESULTS/checkpoint_smoke_scale_ext.pt"; echo "[$(date '+%T')] smoke OK"
# 4) train the extension cells, one at a time; loop passes so a thermally-killed cell is re-attempted
# (resuming from checkpoint) instead of the driver quitting; hot-spell backoff for peak-heat windows.
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
  if [ "$fails" -gt 0 ] && [ "$missing" -ge "$prev_missing" ]; then
    hot_backoff=$(( hot_backoff + 1 )); wait_s=$(( hot_backoff * 300 )); [ "$wait_s" -gt 1800 ] && wait_s=1800
    echo "[$(date '+%F %T')] pass $pass: $missing/${#CELLS[@]} incomplete, no progress ($fails failed — box likely too hot) — backing off ${wait_s}s"
    sleep "$wait_s"
  else
    hot_backoff=0
    echo "[$(date '+%F %T')] pass $pass: $missing/${#CELLS[@]} cells incomplete ($fails failed) — re-attempting"
  fi
  prev_missing=$missing
done
# 5) completion gate: only claim done when EVERY extension cell has its marker
missing=0
for c in "${CELLS[@]}"; do
  set -- $c; s=$1; arm=$2; seed=$3; bM=$(( s * 65536 / 1000000 ))
  [ -f "$LDIR/persist_${bM}M_${arm}_s${seed}.done" ] || { missing=$((missing+1)); echo "  incomplete: persist_${bM}M_${arm}_s${seed}"; }
done
if [ "$missing" -eq 0 ]; then
  touch "$LDIR/ladder_scale_ext.done"
  echo "===== $(date '+%F %T') NORMUON-AT-SCALE EXTENSION COMPLETE — $LDIR/ladder_scale_ext.done ====="
  echo "[$(date '+%T')] next: eval-harness BPB-score the new cells + scaling_ladder.fit_gap_trend over 168M/420M(n=3)/840M"
else
  echo "===== $(date '+%F %T') EXTENSION INCOMPLETE — $missing/${#CELLS[@]} cells missing .done ====="
  exit 1
fi
} >> "$LOG" 2>&1
