#!/usr/bin/env bash
# Mid-training ANNEAL cohort (research/midtraining/plan.md §2). The mid-training stage's
# on-box run: a low-LR 1-sqrt cooldown FROM the faithful base on the premium 50/50 mix
# (treatment) vs an ISO-TOKEN fineweb-only control — isolating "premium data seen last under
# a cooldown" from "the LR decay + more tokens." Headline = treatment_BPB - control_BPB on
# matched-domain code_py (the only place the plan expects signal); honest null on English/
# downstream. 6 cells: mix_seed{0,1,2} (treatment) + fineweb_seed{0,1,2} (iso-token control),
# interleaved so the seed0 pair lands first. ~2300 steps x 65,536 tok = ~150M tok/cell
# (~13% of the 1.19B pretrain, WSD band), ~5.6h/cell @ compile, 6 cells ~1.4 days.
# Each cell: --init_from base on first run, --resume on restart (night survival). One trainer
# at a time (§C4.5); idempotent (.done + --resume); sentinel --kill-at 0.83.
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$EXP/../../.." && pwd)"; cd "$EXP"
BASE="$REPO/Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/checkpoint_qwen3_baseline2tpp.pt"
PY=python3
STEPS=2300; WARMUP=46; PEAK=2.5e-4; END=2.5e-5   # 1-sqrt cooldown 2.5e-4 -> 2.5e-5 (10% floor)
CHILD=""; trap '[ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 143' TERM INT

run_cell () {  # $1=data (mix|fineweb), $2=cell name
  local data="$1" cell="$2" ckpt="checkpoint_${2}.pt"
  [ -f "arm_${cell}.done" ] && { echo "[skip] $cell"; return 0; }
  for _ in $(seq 1 24); do a=$(free -g | awk '/Mem:/{print $7}'); [ "${a:-0}" -ge 60 ] && break; sleep 5; done
  local init="" resume=""
  if [ -f "$ckpt" ]; then resume="--resume $ckpt"; else init="--init_from $BASE"; fi
  echo "===== CELL $cell ($(date '+%H:%M:%S')) ${resume:-$init} ====="
  local seed="${cell##*seed}"
  $PY train_anneal.py --schedule cooldown --data "$data" --seed "$seed" \
      --steps "$STEPS" --warmup_steps "$WARMUP" --peak_lr "$PEAK" --end_lr "$END" \
      --micro_batch 4 --grad_accum 4 --eval_every 0 --ckpt_every 500 \
      --ckpt_path "$ckpt" --run_name "$cell" $init $resume \
      >> "run_${cell}.log" 2>&1 &
  CHILD=$!
  setsid nohup $PY "$REPO/sentinel.py" watch --pid "$CHILD" --kill-at 0.83 --log "$EXP/sentinel.log" >/dev/null 2>&1 &
  wait "$CHILD"; local rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then touch "arm_${cell}.done"; echo "[done] $cell"
  else echo "[FAIL] $cell rc=$rc"; exit "$rc"; fi
}

for seed in 0 1 2; do
  run_cell mix     "mix_seed${seed}"
  run_cell fineweb "fineweb_seed${seed}"
done
echo "ALL 6 CELLS COMPLETE ($(date '+%H:%M:%S'))"; touch cohort.done
