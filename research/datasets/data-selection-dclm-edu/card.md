# Dataset card — dclm-edu treatment slice (data-selection A/B)

**Prepared:** 2026-06-24 · for Qwen3-0.6B · `prepare_dclm_edu.py`

## Source
- **id:** `HuggingFaceTB/dclm-edu` · **revision (pinned):** `dbad8ad71224482740cd9c9d353591adbf62fe04`
- **license:** cc-by-4.0 · **gated:** no
- Live HF API listing verified 2026-06-22 (recon); shards fetched 2026-06-24.

## What & why
The **treatment** slice for a fixed-token **data-selection A/B**: dclm-edu (a DCLM + educational-classifier
filter, *different lineage* from FineWeb-Edu) **vs** FineWeb-Edu (the control the model already trains on),
at identical tokens, judged on **OOD-BPB** (wikitext2 + code). The bitter-lesson lever toward the 13.40
anchor (the reproduction gap is data, not method). *Note:* the recon's lead candidate Ultra-FineWeb-L3
turned out to be **synthetic** data (`-QA-Synthetic`/`-Multi-Style-Synthetic`) — a different hypothesis —
so dclm-edu (a genuine filter) was chosen as the cleaner treatment.

## Prep method (why not streaming)
HF *streaming* of dclm-edu's multi-GB parquet shards is network-throttled to ~20 tok/s. So:
**direct `hf_hub_download` of parquet shards (~55 MB/s) → local Qwen-tokenize (~563k tok/s)**. Model's
own tokenizer `Qwen/Qwen3-0.6B-Base` (vocab 151,936 → **uint32** shards). `trust_remote_code` never used.

## Stats (`stats.json`)
- **train: 150,107,480 tokens** (4 shards, `shard_0000{0,1,2}.bin` = 50M each + remainder) · **eval: 748,210 tokens**
- docs processed: 110,000 · **docs dropped (13-gram decontam): 0**
- decontam: 13-gram Jaccard > 0.8 of each TRAIN doc vs the standing OOD eval corpora (wikitext2_val +
  code_py, 259,822 13-grams) → drop, so the A/B's OOD-BPB judgment can't be contaminated by benchmark text.
- doc-level held-out eval split (~0.5%); verified uint32, max token id 151,643 < vocab 151,936.
- wall-clock: 267 s.

## Consumers
- A/B training: `Qwen3-0.6B/experiments/2026-06-24_qwen3-0.6b_data-dclm-vs-fineweb/` (treatment arm reads
  these shards via `train_dataarm.py --data dclm`; control = Phase-1 FineWeb-Edu baselines, §C13 reused).
- Scored by `/eval-harness` on OOD-BPB + text-lm-v3 downstream; never re-scored here (forge prepares only).
