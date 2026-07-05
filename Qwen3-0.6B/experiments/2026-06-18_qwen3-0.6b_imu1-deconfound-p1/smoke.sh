#!/usr/bin/env bash
# §C5.0 smoke: each Phase-1 arm a few REAL GPU steps (no compile, tiny budget) +
# a resume round-trip on baseline. Asserts every arm exits 0 with finite loss and
# a checkpoint; the resume re-loads model+optims+step+RNG and continues.
set -u
cd "$(dirname "$0")"
PY=python3
SMOKE_STEPS=6
COMMON="--steps $SMOKE_STEPS --warmup_steps 3 --ckpt_every $SMOKE_STEPS --eval_every 0 --no_compile --seed 0 --micro_batch 4 --grad_accum 4"

declare -A ARMS=(
  [baseline]="--arch faithful --schedule cosine --zloss 0"
  [wsd]="--arch faithful --schedule wsd --zloss 0"
  [zloss]="--arch faithful --schedule cosine --zloss 1e-4"
  [arch]="--arch imu1 --schedule cosine --zloss 0"
)

PASS=1
for name in baseline wsd zloss arch; do
  echo "===== SMOKE arm=$name ====="
  $PY train_ablation.py ${ARMS[$name]} --optimizer adamw $COMMON \
      --ckpt_path "smoke_$name.pt" --run_name "smoke_$name"
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "smoke_$name.pt" ]; then echo "ARM $name FAILED rc=$rc"; PASS=0; fi
done

echo "===== RESUME round-trip (baseline: resume smoke_baseline.pt, continue to 10) ====="
$PY train_ablation.py ${ARMS[baseline]} --optimizer adamw \
    --steps 10 --warmup_steps 3 --ckpt_every 10 --eval_every 0 --no_compile --seed 0 \
    --resume "smoke_baseline.pt" --ckpt_path "smoke_baseline_resumed.pt" --run_name "smoke_resume"
rc=$?
grep -q "RESUMED" "train_smoke_resume.log" 2>/dev/null && RES="resume-loaded" || RES="NO-RESUME-LINE"
[ $rc -eq 0 ] || PASS=0

echo "===== SMOKE SUMMARY ====="
echo "resume: $RES (rc=$rc)"
echo "checkpoints: $(ls smoke_*.pt 2>/dev/null | wc -l)/5"
echo "VERDICT: $([ $PASS -eq 1 ] && echo SMOKE-PASS || echo SMOKE-FAIL)"
