#!/usr/bin/env bash
# Run lm-evaluation-harness on the official SmolLM2-135M baseline AND on a local
# checkpoint. Produces JSON + a markdown summary under results/lm_eval/.
#
# Prerequisites:
#   pip install lm-eval                      # https://github.com/EleutherAI/lm-evaluation-harness
#   (already have transformers + datasets installed for the rest of this repo)
#
# Usage:
#   # Eval base model only
#   bash scripts/run_lm_eval.sh
#
#   # Eval base + your trained checkpoint
#   bash scripts/run_lm_eval.sh checkpoint_tinystories.pt
#
# The task set below mirrors what SmolLM2's model card reports so you can
# diff against the published numbers (HellaSwag, ARC easy/challenge, PIQA,
# MMLU, WinoGrande, CommonsenseQA, OpenBookQA).
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
BASE_REPO="HuggingFaceTB/SmolLM2-135M"
TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,commonsense_qa,openbookqa,mmlu"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
OUT_DIR="results/lm_eval"

mkdir -p "$OUT_DIR"

# ── Sanity: is lm-eval installed? ──────────────────────────────────────────
if ! command -v lm_eval >/dev/null 2>&1; then
  echo "lm_eval not found. Install with:  pip install lm-eval" >&2
  exit 1
fi

# ── 1. Base model ──────────────────────────────────────────────────────────
echo "==> Evaluating BASE model: $BASE_REPO"
lm_eval \
  --model hf \
  --model_args "pretrained=$BASE_REPO,dtype=bfloat16" \
  --tasks "$TASKS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_fewshot "$NUM_FEWSHOT" \
  --output_path "$OUT_DIR/base"

# ── 2. Optional: trained checkpoint ────────────────────────────────────────
if [[ "${1:-}" != "" ]]; then
  CKPT="$1"
  EXPORT_DIR="$OUT_DIR/exported"

  if [[ ! -f "$CKPT" ]]; then
    echo "Checkpoint not found: $CKPT" >&2
    exit 1
  fi

  echo "==> Exporting $CKPT to HF format at $EXPORT_DIR ..."
  python scripts/export_to_hf.py "$CKPT" "$EXPORT_DIR"

  echo "==> Evaluating TRAINED checkpoint: $EXPORT_DIR"
  lm_eval \
    --model hf \
    --model_args "pretrained=$EXPORT_DIR,dtype=bfloat16" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot "$NUM_FEWSHOT" \
    --output_path "$OUT_DIR/trained"
fi

echo
echo "Done. JSON results are under $OUT_DIR/."
echo "Compare against the SmolLM2-135M model card to validate the harness setup."
