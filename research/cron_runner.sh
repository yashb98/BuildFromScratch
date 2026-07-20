#!/usr/bin/env bash
# Headless runner for the autonomous research loop.
# Usage: cron_runner.sh "/research-loop"   (or "/weekly-retro")
# Installed via crontab; logs to research/cron_logs/. A flock guard skips the
# night if the previous session is still running. The 6h timeout is a backstop
# against a hung session — training runs are launched detached (nohup) by
# ablation-runner and survive this session exiting.
set -u

REPO=/home/yashb98/Downloads/BuildFromScratch
CLAUDE=/home/yashb98/.local/bin/claude
PROMPT="${1:-/research-loop}"
LOG_DIR="$REPO/research/cron_logs"
LOCK=/tmp/research-loop.lock

mkdir -p "$LOG_DIR"
SLUG=$(echo "$PROMPT" | tr -cd '[:alnum:]-' | cut -c1-32)
LOG="$LOG_DIR/$(date +%Y-%m-%d_%H%M)_${SLUG}.log"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date -Is) previous loop session still running — skipping this fire" >>"$LOG"
    exit 0
fi

cd "$REPO" || exit 1
echo "$(date -Is) starting: $PROMPT" >>"$LOG"
timeout --signal=TERM 6h "$CLAUDE" -p "$PROMPT" --dangerously-skip-permissions >>"$LOG" 2>&1
echo "$(date -Is) finished, exit=$?" >>"$LOG"

# keep the last 60 logs
ls -1t "$LOG_DIR" | tail -n +61 | while read -r f; do rm -f "$LOG_DIR/$f"; done
