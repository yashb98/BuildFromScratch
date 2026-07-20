#!/usr/bin/env bash
# research/boot_resume.sh — GENERIC @reboot cross-reboot recovery for the research loop.
#
# The GB10 silently hard-locks under sustained training load every ~5-10h (thermal, no
# kernel trace). In-process resume (ablation-runner Phase 0) only fires when the loop is
# re-invoked; after a HARD-LOCK REBOOT nothing re-invokes it. This hook — fired ONCE per
# boot from crontab `@reboot` — re-adopts the one §C5-approved in-flight run recorded in
# research/loop_state.json and relaunches it from its checkpoint. It is the promotion of
# the ladder-local boot_resume.sh (the stack that delivered the 420M sweep) into a generic,
# loop-state-driven hook the sanctioned launcher (/ablation-runner) can rely on.
#
# It NEVER initiates a NEW run (that is S4 selection + the full §C5 contract). It only
# resumes a run the launcher already approved and recorded. Fully guarded so it can never
# double-launch (§C4.5), never resume past the §C5 auto-resume cap (no resume-loop that
# re-crashes the box), and never blindly resume after a sentinel kill.
#
# Guards (hardened 2026-07-19 after an adversarial review):
#   - MUTUAL EXCLUSION: holds an flock on the SHARED /tmp/research-loop.lock (the same lock
#     cron_runner.sh uses), so duplicate @reboot lines AND the liveness->/research-loop
#     resume agent path can never both relaunch — exactly one recovery runs per boot.
#   - cd's into the repo root first: @reboot cron runs with cwd=$HOME, but resume_cmd is
#     repo-relative, so without this the trainer silently never launches.
#   - Settles (BOOT_SETTLE_SECS, default 90s) BEFORE preflight, so a post-reboot GPU/driver
#     storm does not fail the health gate and forfeit the once-per-boot recovery.
#   - Refuses on a sentinel_kill marker (a memory/thermal kill is not auto-resumable at the
#     same config — ablation-runner Phase 0 discipline).
#   - Shares the §C5 auto_resumes cap via the tested `loop_state.py record-resume`.
#   - Arms the sentinel on the PYTHON trainer child, never the run_arms.sh/run_ladder.sh
#     wrapper (sentinel signals one pid; watching the wrapper orphans the trainer on SIGKILL).
#
# The agent NEVER auto-installs this (repo policy §C4.2). /ablation-runner PRINTS the
# `@reboot` crontab line at launch; the user pastes it once. It is idempotent — a second
# identical line is harmless (the flock lets only one instance act), and it self-nops once
# loop_state has no in-flight run.
#
# Usage:
#   bash research/boot_resume.sh              # real: guard -> resume the in-flight run
#   bash research/boot_resume.sh --dry-run    # print the DECISION only; no lock, no launch, no cap write
# Test hooks (never set in production): BFS_ROOT, BOOT_SETTLE_SECS, BOOT_LOCK.
#
# Exit: 0 = handled (resumed, or a clean reason not to); 1 = preflight failed / cannot cd.
set -u

# @reboot cron runs with a bare PATH — make python3 / pgrep / flock / the miniforge
# interpreter resolve, or the hook dies before it can resume. Box-specific (single GB10).
export PATH="/opt/miniforge3/bin:$HOME/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT="${BFS_ROOT:-/home/yashb98/Downloads/BuildFromScratch}"
STATE="$ROOT/research/loop_state.json"
MARKER="$STATE.sentinel_kill"                 # sentinel.py writes this on a kill (§C6)
RECOVERY="$ROOT/research/recovery"
BOOTLOG="$RECOVERY/boot_resume.log"
CAP=2                                          # §C5 shared auto-resume cap (== loop_state.MAX_AUTO_RESUMES)
SETTLE="${BOOT_SETTLE_SECS:-90}"               # seconds to settle after boot before the health gate
LOCK="${BOOT_LOCK:-/tmp/research-loop.lock}"   # SHARED with cron_runner.sh -> excludes the agent-resume path too
# GUARD pattern (broad, fail-safe): any experiment trainer (train_<slug>.py) + the wrappers.
# Deliberately matches ablation-runner's per-run train_<SLUG>.py, not a static basename list.
# Env-overridable (BOOT_TRAINER_RE) for hermetic tests only — never set in production.
TRAINER_RE="${BOOT_TRAINER_RE:-train_[A-Za-z0-9_]*\.py|run_arms\.sh|run_ladder\.sh}"
# PID-CAPTURE pattern (precise): the PYTHON trainer child, so the sentinel is armed on the
# memory-holding process, never the bash -lc / run_arms.sh wrapper. Env-overridable for tests.
PIDCAP_RE="${BOOT_PIDCAP_RE:-python[0-9.]* .*train_[A-Za-z0-9_]*\.py}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

mkdir -p "$RECOVERY"
# @reboot cwd is $HOME; resume_cmd (and run_arms.sh's own paths) are repo-relative.
cd "$ROOT" || { echo "DECISION=cannot-cd"; echo "$(date '+%F %T') [boot_resume] cannot cd to $ROOT" >>"$BOOTLOG" 2>/dev/null; exit 1; }

log(){ echo "$(date '+%F %T') [boot_resume] $*" | tee -a "$BOOTLOG" >&2; }
decide(){ echo "DECISION=$1"; log "decision=$1 ${2:-}"; }
preflight_ok(){ python3 "$ROOT/sentinel.py" preflight >/dev/null 2>&1; }

# --- read the recovery pointer from loop_state.json (the ONE source of truth) -----------
# Uses loop_state.py's fail-open load, so a truncated/corrupt state (a lock mid-write)
# yields the canonical schema instead of crashing this hook.
read_state(){ PYTHONPATH="$ROOT/research" python3 - "$STATE" <<'PY'
import sys
try:
    import loop_state as ls
    st = ls.load(sys.argv[1])
except Exception:
    st = {}
def g(k, d=""):
    v = st.get(k, d)
    return "" if v is None else str(v)
# \x1f (unit separator): a NON-whitespace delimiter, so bash `read` cannot collapse
# empty leading fields (a tab would — in_flight_run=None then reads as the next field).
print("\x1f".join([g("in_flight_run"), g("resume_cmd"),
                   g("auto_resumes", "0"), g("train_pid")]))
PY
}

IFS=$'\x1f' read -r RUN_ID RESUME_CMD AUTO_RESUMES TRAIN_PID < <(read_state)
AUTO_RESUMES="${AUTO_RESUMES:-0}"

# --- PURE-STATE GUARDS (identical in dry-run and real; evaluated before any side effect) ---
if [ -z "$RUN_ID" ] || [ -z "$RESUME_CMD" ]; then
  decide nothing-to-resume "(no in_flight_run / resume_cmd in loop_state)"; exit 0
fi
if [ -e "$MARKER" ]; then
  decide refuse-sentinel-kill "(marker $MARKER present — a memory/thermal kill is NOT auto-resumable at the same config; human must inspect + clear the marker)"
  exit 0
fi
if pgrep -f "$TRAINER_RE" >/dev/null 2>&1; then
  decide already-running "(a trainer is alive — no double-launch, §C4.5)"; exit 0
fi
if [ "$AUTO_RESUMES" -ge "$CAP" ] 2>/dev/null; then
  decide cap-reached "(auto_resumes=$AUTO_RESUMES >= $CAP — refusing; never loop-crash the box, §C5)"
  exit 0
fi

if [ "$DRY" -eq 1 ]; then
  # Dry-run runs preflight WITHOUT the settle (fast, read-only) purely to report the decision.
  preflight_ok || { decide preflight-fail "(sentinel preflight != 0)"; exit 1; }
  decide would-resume "(RUN_ID=$RUN_ID auto_resumes=$AUTO_RESUMES -> would relaunch resume_cmd)"
  exit 0
fi

# --- REAL RESUME --------------------------------------------------------------------------
# MUTUAL EXCLUSION: hold the shared lock for the whole critical section (settle -> preflight
# -> record-resume -> spawn). A duplicate @reboot boot_resume, or a liveness-triggered
# cron_runner (which flocks the same file), cannot also relaunch. Released when this exits.
exec 9>"$LOCK" 2>/dev/null || { decide lock-unavailable "(cannot open $LOCK)"; exit 0; }
if ! flock -n 9; then
  decide already-locked "(another recovery holds $LOCK — no double-launch, §C4.5)"; exit 0
fi

[ "$SETTLE" -gt 0 ] 2>/dev/null && sleep "$SETTLE"   # settle BEFORE the health gate so the GPU driver is up

# Re-probe the guards that can change during the settle window (a kill marker appearing, or
# a trainer coming up via another path). Now holding the lock, so this is race-free.
if pgrep -f "$TRAINER_RE" >/dev/null 2>&1; then
  decide already-running "(trainer appeared during settle — no double-launch)"; exit 0
fi
[ -e "$MARKER" ] && { decide refuse-sentinel-kill "(marker appeared during settle)"; exit 0; }
preflight_ok || { decide preflight-fail "(sentinel preflight != 0 after settle — not launching)"; exit 1; }

# Account this recovery through the TESTED atomic cap counter (the ONE shared counter);
# `crashed` means the cap is now reached — refuse without launching.
DEC=$(PYTHONPATH="$ROOT/research" python3 "$ROOT/research/loop_state.py" \
        --path "$STATE" record-resume --ts "$(date -Is)" 2>/dev/null \
        | python3 -c 'import sys,json; print(json.load(sys.stdin).get("decision","crashed"))' 2>/dev/null)
if [ "$DEC" != "resume" ]; then
  decide cap-reached "(record-resume returned '$DEC' — auto-resume budget exhausted, §C5)"; exit 0
fi

# Dense thermal forensics beside the run (pure observability; sentinel.py owns the kill).
pgrep -f 'thermal_log.py' >/dev/null 2>&1 || \
  setsid nohup python3 "$ROOT/research/thermal_log.py" --interval 10 \
    --out "$RECOVERY/${RUN_ID}.thermal.log" >> "$RECOVERY/${RUN_ID}.thermal.stdout" 2>&1 < /dev/null &

# Relaunch the recorded resume_cmd DETACHED (§C5 launch form), from cwd=$ROOT so its
# repo-relative paths resolve. resume_cmd is idempotent: a multi-arm wrapper skips completed
# arms; a single-arm trainer restores from --resume.
RESUME_LOG="$RECOVERY/${RUN_ID}.resume.log"
echo "===== $(date '+%F %T') @reboot auto-resume: $RUN_ID (auto_resumes now accounted) =====" >> "$RESUME_LOG"
setsid nohup bash -lc "$RESUME_CMD" >> "$RESUME_LOG" 2>&1 < /dev/null &
sleep 5

# Find the actual PYTHON trainer child (newest match), NOT the bash -lc / run_arms.sh wrapper
# — the sentinel must watch the memory-holding process or a SIGKILL orphans the real trainer.
NEWPID="$(pgrep -n -f "$PIDCAP_RE" | head -1)"
if [ -z "$NEWPID" ]; then
  decide relaunch-unconfirmed "(resume_cmd launched but no python trainer found after 5s — check $RESUME_LOG)"
  exit 0
fi

# Re-arm the sentinel on the trainer pid UNLESS a wrapper already self-armed one
# (multi-arm run_arms.sh arms a per-arm watcher — Phase 5.2). Never double-watch.
if ! pgrep -f 'sentinel.py watch' >/dev/null 2>&1; then
  setsid nohup python3 "$ROOT/sentinel.py" watch --pid "$NEWPID" \
    --log "$RECOVERY/${RUN_ID}.sentinel.log" >/dev/null 2>&1 < /dev/null &
  log "armed sentinel watch on trainer pid $NEWPID"
else
  log "sentinel watch already alive (wrapper self-armed) — not double-arming"
fi

# Write the new trainer pid back through loop_state.py (the only sanctioned writer) so the
# next Phase 0 adopt sees the live pid. record-resume already updated auto_resumes.
PYTHONPATH="$ROOT/research" python3 - "$STATE" "$NEWPID" <<'PY' 2>/dev/null || true
import sys, loop_state as ls
p, pid = sys.argv[1], int(sys.argv[2])
st = ls.load(p); st["train_pid"] = pid; ls.save(p, st)
PY

decide resumed "(RUN_ID=$RUN_ID new train_pid=$NEWPID auto_resumes=$((AUTO_RESUMES+1)) log=$RESUME_LOG)"
exit 0
