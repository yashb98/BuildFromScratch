#!/usr/bin/env bash
# HybridSSM-0.2B architecture ladder — continuous driver.
#
# Runs every queued cell in cells.json ONE AT A TIME (§C4.5) until all are done, so the
# ladder keeps going across thermal kills, crashes and reboots without a human restarting
# it. Modelled on the proven Qwen3 run_ladder.sh: .done markers make it idempotent, a
# cool-down gate keeps it off a hot box, each cell gets its own sentinel watcher, a
# loop-until-done pass structure re-attempts killed cells (they resume from checkpoint),
# and an escalating hot-spell backoff stops it thrashing at zero net progress.
#
# Cells are ordered CHEAP RUNG FIRST, so the 42M rung finishes in ~22 h and already yields
# a complete 5-arm architecture comparison; partial completion is still a result.
set -uo pipefail

ROOT="${BFS_ROOT:-/home/yashb98/Downloads/BuildFromScratch}"
LDIR="$ROOT/HybridSSM-0.2B/experiments/2026-07-21_hybrid-ssm-0.2b_arch-ladder"
BUILD="$ROOT/HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build"
DATA="$ROOT/Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/tokcache_170034304_300000_seed0_Qwen3-0.6B-Base.pt"
LOG="$LDIR/run_arch_ladder.log"
PY=python3
# Cool-down gate (see cool_down() for the 2026-07-23 post-mortem that set these).
# COOL_C=72 (CORRECTED 2026-07-23 evening): cool_down compares to hottest_c = max(GPU die, ALL ACPI
# SoC zones), and the SoC zones idle at ~64-68C on this box — so an earlier 58C try was BELOW the idle
# floor and deadlocked every launch (the dwell could never be satisfied). sentinel WARNs at 82C, KILLs
# at 90C (3 consecutive). 72C is just above the idle floor (dwell reachable), 10C under WARN, 18C under
# KILL; the proven CORE run_ladder.sh ran the 420M ladder at 70C. The +31C "transient" that argued for
# 58C was measured on a HEAT-SOAKED box mid-cooldown after a kill (66C dipping from 90C), not a cool
# one — which is exactly what the DWELL (6 sustained samples) rejects. The dwell + DEFER is the real
# fix for the thrash; the threshold just has to be reachable AND under WARN.
COOL_C="${LADDER_COOL_C:-72}"     # don't launch onto a box hotter than this (sustained, via dwell)
COOL_DWELL="${LADDER_COOL_DWELL:-6}"    # consecutive cool samples required (6 x 30s = 3 min)
COOL_MAX="${LADDER_COOL_MAX:-30}"       # bound: 30 x 30s = 15 min, then DEFER (never launch)
MAXPASS="${LADDER_MAXPASS:-100}"
SEQ=2048; BATCH=4; LR=3e-3; WARMUP=200

exec >> "$LOG" 2>&1
echo "===== $(date '+%F %T') driver start (pid $$) ====="

# Hottest of GPU die + all ACPI SoC zones, whole deg C (empty if unreadable). No sudo.
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

# Cool-down gate. Returns 0 = safe to launch, 1 = still hot -> DEFER this cell.
#
# 2026-07-23 post-mortem (the 15:16-15:24 BST thrash that ended in a hard lock). The old
# gate accepted a SINGLE sample below 70C, and when its 30-min bound expired it LAUNCHED
# ANYWAY; its return value was discarded at the call site, so nothing could have stopped
# it either way. What that produced, from run_arch_ladder.log + the sentinel logs (sentinel
# stamps UTC, the driver stamps BST = UTC+1):
#   14:16:46Z  sentinel thermal-kills swa128_nope_85M_s0 at gpu 85C / soc 92C
#   15:17:47   driver relaunches attn1to3_85M_s0 on ONE 66C sample, 60s after that kill
#   14:18:37Z  that arm is already at soc 97C (a +31C idle->load rise in 50s)
#   14:21:37Z  sentinel thermal-kills it too
#   15:22:38   driver relaunches fullattn_85M_s0 on ONE 69C sample, 61s after THAT kill
#   15:24:38   box hard-locks (no trace: crashkernel=0M, see the GB10 lockup memory note)
# Three launches, two thermal kills and one hard lock inside eight minutes.
#
# Two structural fixes:
#  (a) DWELL — require COOL_DWELL consecutive samples strictly under COOL_C. One cool
#      sample a minute after a 92C kill is a sensor dip, not a cool box: this box sheds to
#      ~66C in ~60s and then PLATEAUS there, so a single-sample 70C gate had ~1C of
#      headroom and waved through a box still full of heat.
#  (b) DEFER, never launch-anyway — the bound now returns 1 so the caller drops the cell to
#      the next pass and the hot-spell backoff at the end of the pass loop engages, instead
#      of the driver walking down the cell list igniting one arm after another.
# Unreadable temp still fails OPEN (proceed): fabricating heat is worse than not measuring
# it, and sentinel.py takes the same stance (gpu_thermal/hottest_soc_c return None rather
# than a fake high reading).
cool_down () {
  local tag=$1 h streak=0
  for _ in $(seq 1 "$COOL_MAX"); do
    h=$(hottest_c)
    [ -z "$h" ] && { echo "[$(date '+%T')] [cooldown] $tag: temp unreadable, proceeding"; return 0; }
    if [ "$h" -lt "$COOL_C" ]; then
      streak=$((streak+1))
      if [ "$streak" -ge "$COOL_DWELL" ]; then
        echo "[$(date '+%T')] [cooldown] $tag: ${h}C < ${COOL_C}C for $streak consecutive samples, launching"
        return 0
      fi
      echo "[$(date '+%T')] [cooldown] $tag: ${h}C < ${COOL_C}C, dwell $streak/${COOL_DWELL}"
    else
      echo "[$(date '+%T')] [cooldown] $tag: ${h}C >= ${COOL_C}C, waiting 30s (dwell reset from $streak)"
      streak=0
    fi
    sleep 30
  done
  echo "[$(date '+%T')] [cooldown] $tag: no ${COOL_DWELL}-sample cool spell in $((COOL_MAX * 30 / 60))min (last ${h}C) — DEFERRING"
  return 1
}

# Is a REAL trainer alive? A bare `pgrep -f 'train_*.py'` is not good enough: it matches any
# process whose cmdline merely MENTIONS a trainer — a grep, an editor, a monitoring command,
# another agent session. That false positive deferred all 15 cells on the first launch attempt
# (2026-07-21 22:03) while sentinel's own preflight, which filters probes, correctly reported
# trainers=none on the very same second. So: require argv[0] to be a python interpreter, which
# no shell wrapper or pgrep can satisfy, and never count ourselves.
trainer_alive () {
  local p exe
  for p in $(pgrep -f 'train_[A-Za-z0-9_]*\.py' 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    exe=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)
    case "$(basename "${exe:-none}")" in python*) return 0 ;; esac
  done
  return 1
}

# cells.json -> "id tokens steps mixer attn_every nope" lines, queue order preserved.
cells () { $PY - "$LDIR/cells.json" <<'PYEOF'
import json, sys
for c in json.load(open(sys.argv[1]))["cells"]:
    print(c["id"], c["tokens"], c["steps"], c["mixer"], c["attn_every"], int(c["nope"]))
PYEOF
}

run_cell () {
  local id=$1 tokens=$2 steps=$3 mixer=$4 every=$5 nope=$6
  [ -f "$LDIR/${id}.done" ] && { echo "[$(date '+%T')] [skip] $id"; return 0; }

  # §C4.5: never two trainers. A foreign trainer means someone else owns the GPU — wait it out.
  if trainer_alive; then
    echo "[$(date '+%T')] [wait] $id: another trainer is alive, deferring this pass"; return 1
  fi
  # HONOUR the gate. Until 2026-07-23 this call discarded cool_down's return value, so even
  # a gate that wanted to refuse could not: the driver launched regardless. A deferral must
  # cost this cell the pass, so the hot-spell backoff below sees "failed, no progress".
  if ! cool_down "$id"; then
    echo "[$(date '+%T')] [defer] $id: no sustained cool window — deferring this pass"; return 1
  fi
  # unified pool headroom (shared CPU+GPU memory): wait for >= 60 GB available
  for _ in $(seq 1 90); do a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 10; done

  local nopeflag=""; [ "$nope" = "1" ] && nopeflag="--nope_on_full"
  echo "[$(date '+%F %T')] START $id  tokens=$tokens steps=$steps mixer=$mixer attn_every=$every nope=$nope"
  ( cd "$BUILD" && $PY train_hybrid.py --data "$DATA" --seq $SEQ --batch $BATCH \
      --tokens "$tokens" --lr $LR --warmup $WARMUP --mixer "$mixer" --attn_every "$every" $nopeflag \
      --ckpt "checkpoint_${id}.pkl" --resume "checkpoint_${id}.pkl" \
      --ckpt_every 200 --eval_every 400 --done_marker "$LDIR/${id}.done" \
      >> "$LDIR/${id}.log" 2>&1 ) &
  local tpid=$!
  # sentinel on the PYTHON trainer, not this subshell; no --kill-at so it takes the §C6 default
  local spid=""
  sleep 20
  local realpid; realpid=$(pgrep -n -f "python[0-9.]* train_hybrid\.py .*${id}" || echo "$tpid")
  $PY "$ROOT/sentinel.py" watch --pid "$realpid" --log "$LDIR/sentinel_${id}.log" >/dev/null 2>&1 &
  spid=$!
  wait "$tpid"; local rc=$?
  kill "$spid" 2>/dev/null

  if [ $rc -eq 0 ] && [ -f "$LDIR/${id}.done" ]; then
    echo "[$(date '+%T')] [done] $id"; return 0
  fi
  rm -f "$LDIR/${id}.done"        # never trust a marker from a failed cell
  echo "[$(date '+%T')] [FAIL rc=$rc] $id — will retry next pass (resumes from checkpoint)"
  return 1
}

# ---- preflight once, then loop passes until every cell has its marker -------------------
$PY "$ROOT/sentinel.py" preflight || { echo "preflight FAIL — abort"; exit 1; }

pass=0; hot_backoff=0; prev_missing=99999
while : ; do
  fails=0
  while read -r id tokens steps mixer every nope; do
    [ -z "${id:-}" ] && continue
    run_cell "$id" "$tokens" "$steps" "$mixer" "$every" "$nope" || fails=$((fails+1))
  done < <(cells)

  missing=0
  while read -r id _; do [ -f "$LDIR/${id}.done" ] || missing=$((missing+1)); done < <(cells)
  [ "$missing" -eq 0 ] && break

  pass=$((pass+1))
  [ "$pass" -ge "$MAXPASS" ] && { echo "[$(date '+%T')] MAXPASS=$MAXPASS, $missing incomplete — stopping (checkpoints preserved; re-run to continue)"; break; }

  # HOT-SPELL BACKOFF: a whole pass with failures and NO cell completing means the box is too
  # hot to make progress — wait for a cooler window (5→30 min cap) instead of thrashing.
  if [ "$fails" -gt 0 ] && [ "$missing" -ge "$prev_missing" ]; then
    hot_backoff=$((hot_backoff+1)); wait_s=$((hot_backoff*300)); [ "$wait_s" -gt 1800 ] && wait_s=1800
    echo "[$(date '+%F %T')] pass $pass: $missing incomplete, no progress ($fails failed) — backing off ${wait_s}s"
    sleep "$wait_s"
  else
    hot_backoff=0
    echo "[$(date '+%F %T')] pass $pass: $missing incomplete ($fails failed) — re-attempting"
  fi
  prev_missing=$missing
done

# ---- completion gate: only claim done when EVERY cell has its marker --------------------
missing=0
while read -r id _; do [ -f "$LDIR/${id}.done" ] || { missing=$((missing+1)); echo "  incomplete: $id"; }; done < <(cells)
if [ "$missing" -eq 0 ]; then
  touch "$LDIR/ladder.done"
  echo "===== $(date '+%F %T') ARCH LADDER COMPLETE — $LDIR/ladder.done ====="
  # Score at ladder end. LOUD on absence/failure — the audit found this hook was a
  # silent no-op (the scorer did not exist), so a completed ladder produced NO numbers
  # and nothing said so. No trainer is live here (all cells done), so §C4.5 permits it.
  if [ -f "$LDIR/score_arch_ladder.py" ]; then
    echo "[$(date '+%T')] scoring ladder..."
    if $PY "$LDIR/score_arch_ladder.py"; then
      echo "[$(date '+%T')] scoring done -> arch_ladder_scores.json"
    else
      echo "[$(date '+%T')] !! SCORING FAILED (rc=$?) — cells are trained but UNSCORED; run score_arch_ladder.py by hand"
    fi
  else
    echo "[$(date '+%T')] !! NO SCORER (score_arch_ladder.py absent) — ladder COMPLETE but UNSCORED"
  fi
else
  echo "===== $(date '+%F %T') LADDER INCOMPLETE — $missing cells missing .done ====="
fi
