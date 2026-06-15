#!/usr/bin/env bash
# Durable second copy of the loop's memory (§C8). git history (pushed to the
# GitHub remote) is the PRIMARY off-box backup now that research/ is tracked;
# this script adds a dated local snapshot so an accidental mid-session `rm` or
# a bad write is recoverable without a commit. Validates the file parses through
# ledger.py before snapshotting, so a torn ledger is never archived as "good".
# Wire into cron (daily) if you want automatic rotation; keeps the last 14.
set -euo pipefail
cd "$(dirname "$0")"

LEDGER="ledger/ledger.json"
DEST="ledger/backups"
KEEP=14

[ -f "$LEDGER" ] || { echo "[backup] no ledger at $LEDGER" >&2; exit 1; }

# Refuse to archive a ledger that does not load/validate (§C8 guard).
python3 ledger/ledger.py status --ledger "$LEDGER" >/dev/null

mkdir -p "$DEST"
STAMP="$(date +%Y-%m-%d)"
cp "$LEDGER" "$DEST/ledger-$STAMP.json"
echo "[backup] wrote $DEST/ledger-$STAMP.json"

# rotate: keep newest $KEEP
ls -1t "$DEST"/ledger-*.json 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old; do
  rm -f "$old"; echo "[backup] rotated out $old"
done
