#!/usr/bin/env bash
# research/bootstrap.sh — Tier-0 "activate the substrate" helper.
#
# Makes the autonomous loop real: fixes the cron_runner exec bit, stages the
# trackable substrate for git, and prints the three crontab lines for the human
# to paste. It NEVER installs cron itself (crontab is human-only, §C4.2/§C20),
# NEVER pushes to a remote, and NEVER launches GPU work or the loop. CPU-only;
# safe to run while a trainer is active.
#
# Usage:
#   research/bootstrap.sh            # show this help (no side effects)
#   research/bootstrap.sh --apply    # chmod cron_runner.sh + git add the substrate
#   research/bootstrap.sh --cron     # print the crontab lines to paste yourself
#   research/bootstrap.sh --verify   # check exec bits + tests, report; exit 1 if not ready
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

cron_lines() {
  cat <<EOF
# --- research-loop cron (paste into \`crontab -e\`; adjust the time as you like) ---
0 2 * * *      cd $REPO && bash research/cron_runner.sh   >> $REPO/research/cron_logs/loop.log 2>&1
*/30 * * * *   cd $REPO && bash research/liveness_cron.sh >> $REPO/research/cron_logs/liveness.log 2>&1
@reboot        sleep 180 && cd $REPO && bash research/liveness_cron.sh >> $REPO/research/cron_logs/liveness.log 2>&1
EOF
}

case "${1:-help}" in
  --apply)
    echo "[bootstrap] chmod +x research/cron_runner.sh"
    chmod +x "$HERE/cron_runner.sh"
    echo "[bootstrap] staging trackable substrate (research/ + root safety code + CI)"
    # .claude/ is gitignored by design ('kept local only'); the SKILLS are NOT
    # staged here. If you want them tracked, that is a deliberate choice:
    #   git add -f .claude/skills/      # (overrides .gitignore)
    git -C "$REPO" add research/ sentinel.py safe_cuda.py jax_safe_env.py \
        flop_accounting.py mfu_meter.py .github/ 2>/dev/null || true
    echo "[bootstrap] staged. Review with 'git status', then commit (push is yours)."
    echo "[bootstrap] NOTE: .claude/skills are gitignored — see comment above to track them."
    ;;
  --cron)
    cron_lines
    ;;
  --verify)
    ok=0
    if [ -x "$HERE/cron_runner.sh" ]; then echo "[ok] cron_runner.sh executable"
    else echo "[MISSING] cron_runner.sh not executable — run --apply"; ok=1; fi
    if crontab -l 2>/dev/null | grep -q "research/cron_runner.sh"; then
      echo "[ok] research-loop cron installed"
    else echo "[MISSING] research-loop cron not installed — run --cron and paste"; ok=1; fi
    echo "[bootstrap] running the test gate..."
    if bash "$HERE/run_tests.sh" >/dev/null 2>&1; then echo "[ok] tests green"
    else echo "[MISSING] tests not green — run research/run_tests.sh"; ok=1; fi
    exit "$ok"
    ;;
  help|--help|-h|"")
    sed -n '2,18p' "${BASH_SOURCE[0]}"
    ;;
  *)
    echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
esac
