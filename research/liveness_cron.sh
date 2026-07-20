#!/usr/bin/env bash
# Lightweight recovery trigger for the research loop (contracts §C6).
# Used by TWO crons:
#   */30 * * * *   liveness_cron.sh        -> catch a silently-dead run in <=30 min
#   @reboot        liveness_cron.sh 180    -> auto-resume an in-flight run after a crash
#
# Cheap by design: it only runs `sentinel.py liveness` (a PID check, no Claude
# session). It launches a session ONLY when an in-flight run's PID is dead
# (sentinel exit 4), and then defers to cron_runner.sh "/research-loop resume",
# whose flock prevents overlap with the nightly fire and whose S1 logic does
# the full preflight + §C5 (incl. the §C5.0 smoke test) before any relaunch.
set -u

REPO=/home/yashb98/Downloads/BuildFromScratch
SETTLE="${1:-0}"   # seconds to wait before probing (180 on @reboot, 0 otherwise)
LOG_DIR="$REPO/research/cron_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/liveness.log"

[ "$SETTLE" -gt 0 ] 2>/dev/null && sleep "$SETTLE"

python3 "$REPO/sentinel.py" liveness >>"$LOG" 2>&1
rc=$?
if [ "$rc" -eq 4 ]; then
    echo "$(date -Is) liveness rc=4 (dead in-flight run) -> /research-loop resume" >>"$LOG"
    "$REPO/research/cron_runner.sh" "/research-loop resume"
else
    # rc 0 = idle or running fine; anything else = probe glitch (fail open).
    echo "$(date -Is) liveness rc=$rc -> no action" >>"$LOG"
fi
