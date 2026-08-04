# SmolLM2-135M reproduction — results catalog

Every file here is produced live by `../results.ipynb` (or by re-running
`../verify.py` / `../model_full.py`) — with two documented exceptions: the GPU-side
numbers in `comparison_with_hf.md` have no machine-written backing JSON on disk, and
the min-loss / tokenization values below are read out of `loss_curve.csv` and the
executed `results.ipynb` rather than `summary.json`.

## Headline numbers (most also in `summary.json`)

| Metric | Value |
|---|---|
| Unique parameter count | **134,515,008** (target match ✓) |
| `lm_head` tied to `embed_tokens` | True |
| `max │Δlogits│` vs HuggingFace `LlamaForCausalLM` (fp32, **CPU**) | **0.000e+00** — on GPU 4.72e-05 final-logits / 1.95e-03 per-layer, see `comparison_with_hf.md` |
| Deterministic argmax for `"The capital of France is"` | `' the'` (logit 14.023) |
| Runner-up token | `' Paris'` (logit 12.997, rank #2) |
| Tokenization of `"The capital of France is"` | `[504, 3575, 282, 4649, 314]` |
| Perplexity on wikitext-2 val (ours) | **15.371** |
| Perplexity on wikitext-2 val (HF) | **15.371** (Δ ≈ 1e-6) |
| Demo from-scratch loss (start → end, 150 steps) | 11.254 → **6.288** (min 6.039 @ step 140) |
| Baseline ln(vocab) = ln(49152) | 10.803 |

## Files

### Top-level summaries
- `summary.json` — single-shot digest of every claim the notebook proves.
- `param_count.log` — output of `python3 model.py` (param count + random-init forward).
- `parity.log` — output of `python3 verify.py` (architecture parity gate).
- `training_recipe_resolved.json` — verified training hyperparameters from
  `huggingface/smollm/text/pretraining/smollm2/config_smollm2_135M.yaml`
  (warmup=2000, weight_decay=0.01, clip_grad=1.0, seq_len=2048, global_batch=512,
  total_steps=2,000,000). Resolves the ⚠️ INFERRED rows in README §11.

### Inference & language behaviour
- `topk_predictions.json` — top-10 tokens, probabilities, and raw logits for
  five canonical prompts (capital of France, Pythagorean theorem, Once upon a
  time, def quicksort, Gravity).
- `generations.txt` — 4 prompts × 3 temperatures (greedy / 0.4 / 0.9) using the
  official weights loaded into our class.
- `perplexity.json` — sliding-window CE perplexity on wikitext-2-raw-v1
  validation, 62,403 scored targets (= 31,743 distinct positions, ~1.97x
  double-counted by the overlapping window), seq=1024 stride=512. Ours vs HF
  agree to 5 decimal places (Δ = 8.7 × 10⁻⁷; the 6th differs: 15.370989 vs
  15.370990).

### Visualizations
- `plots/rope_tables.png` — RoPE cos/sin tables, 256 positions × 64-dim head.
  Demonstrates the split-halves layout (`config.json: rope_interleaved=false`,
  θ=100k).
- `plots/residual_norms.png` — mean L2 norm of the residual stream at the
  output of every block (0 = embeddings, 30 = pre-final-RMSNorm, 31 = after
  final RMSNorm).
- `plots/wsd_schedule.png` — WSD schedule shape for three (total_steps,
  warmup_steps) combinations. Confirms warmup → stable → linear decay.
- `plots/loss_curve.png` — 150-step from-scratch training mini-run on a
  wikitext-2 slice with the nanotron-canonical recipe
  (AdamW(0.9, 0.95), peak LR 3e-3, weight_decay=0.01, clip_grad=1.0,
  WSD(warmup=20, decay=20%)). Loss + LR overlaid.
- `attention/layer_00.png`, `layer_14.png`, `layer_29.png` — softmax-normalized
  attention weights for all 9 query heads at the first, middle, and last
  layer of the 30-layer stack, for the sample
  `"The quick brown fox jumps over the lazy dog because it was hungry."`.

### Training mini-run
- `loss_curve.csv` — per-step (step, loss, lr) for the 150-step demo.
- `../checkpoint.pt` — saved state_dict + config + losses + lrs (~538 MB).

## How to re-generate everything

```bash
# from /home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)/
python3 _build_notebook.py       # writes results.ipynb (28 cells, no outputs)
jupyter nbconvert --to notebook \
    --execute results.ipynb \
    --output results.ipynb \
    --ExecutePreprocessor.timeout=2400
# total runtime ~3 minutes (perplexity & training are the slow cells)
```

## What this notebook does NOT cover (next-step list, see ../README.md §12)

1. KV cache in `model.generate()` — currently O(T²) per token.
2. Multi-GPU / FSDP training (single-process only).
3. Activation checkpointing for longer sequence lengths.
4. Cross-document attention masking in `train.py`'s packer.
5. Port to Qwen3 0.6B (adds QK-norm, different head config).
