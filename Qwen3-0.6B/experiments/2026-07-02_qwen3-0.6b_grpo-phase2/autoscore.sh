#!/usr/bin/env bash
# Fire-on-cohort.done auto-scorer for the Phase-2 RLVR cohort. Waits for all 3 arms to finish,
# waits for the GPU to free (one job at a time §C4.5), preflights, scores the GRPO + random-reward
# checkpoints on the SAME paired band as Phase-1 (so they compare directly to the SFT floor), then
# builds verdict.json + updates the ledger. RFT was a 0-completion no-op (checkpoint == SFT floor)
# so it is not re-scored. safe_cuda inside the runner; this script launches nothing unguarded.
set -u
ROOT=/home/yashb98/Downloads/BuildFromScratch
G=$ROOT/Qwen3-0.6B/experiments/2026-07-02_qwen3-0.6b_grpo-phase2
P1=$ROOT/Qwen3-0.6B/experiments/2026-07-01_qwen3-0.6b_rlvr-phase1-passk
LOG=$G/autoscore.log
PY=python3

{
echo "===== $(date '+%F %T') phase-2 auto-scorer armed (waiting for cohort.done) ====="
for _ in $(seq 1 21600); do [ -f "$G/cohort.done" ] && { echo "[$(date '+%T')] cohort.done seen"; break; }; sleep 10; done
[ -f "$G/cohort.done" ] || { echo "TIMEOUT waiting for cohort.done (>60h) — abort, score manually"; exit 1; }

echo "[$(date '+%T')] waiting for the GPU to free (no train_grpo.py)"
for _ in $(seq 1 180); do pgrep -f "train_grpo.py" >/dev/null 2>&1 || break; sleep 10; done
sleep 5

echo "[$(date '+%T')] preflight"
$PY "$ROOT/sentinel.py" preflight || { echo "preflight FAIL — abort"; exit 1; }

cd "$P1" || exit 1
for a_rm in grpo random rft; do
  ck="$G/checkpoint_${a_rm}_seed0.pt"
  if [ -f "$ck" ]; then
    echo "[$(date '+%T')] scoring arm=$a_rm ($ck)"
    $PY run_phase1_passk.py --ckpt "$ck" --label "$a_rm" --out "$G/passk_${a_rm}.json" \
      && echo "[$(date '+%T')] arm=$a_rm scored" || echo "[$(date '+%T')] arm=$a_rm scoring FAILED"
  else
    echo "[$(date '+%T')] arm=$a_rm: no checkpoint at $ck — skip"
  fi
done

echo "[$(date '+%T')] building verdict"
cd "$G" || exit 1
$PY score_verdict.py
echo "===== $(date '+%F %T') auto-scorer DONE — see $G/verdict.json ====="
} >> "$LOG" 2>&1
