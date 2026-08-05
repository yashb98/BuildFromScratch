# Content dossier — SmolLM2-135M from scratch + TinyStories continued pretraining

Every number below is verified from a file or a live model run in this project — no
illustrative values. Source-of-truth pointer next to each one.

---

## Headline numbers (one-liners for IG stories)

| Number | What it is | Source |
|---|---|---|
| **134,515,008** | Unique parameters in our model (exact match to target) | `results/param_count.log` |
| **0.000e+00** | `max \|Δlogits\|` between ours and HuggingFace (CPU fp32) | `results/parity.log` |
| **15.371 = 15.371** | wikitext-2 val PPL — ours = HF (Δ ≈ 9 × 10⁻⁷) | `results/perplexity.json` |
| **6.8945 → 3.7900** | TinyStories-val PPL, before → after 100M-token continued pretraining | `results/tinystories_after.txt` |
| **−45.0%** | TinyStories PPL improvement | derived above |
| **116.1 min** | Wall-clock for the 100M-token TinyStories run on NVIDIA GB10 | `results/tinystories_train.log` |
| **24,414 steps** | Total steps at seq_len 1024, batch 4 | `results/tinystories_train.csv` |
| **0.9088** | Best single-step training loss (step 22,353, deep in WSD decay) | `results/tinystories_train.csv` |

---

## The two achievements

### 1. Architecture parity — "I rebuilt SmolLM2-135M and it matches HuggingFace exactly"

- Param count: **134,515,008** unique, with `lm_head.weight` aliased to
  `embed_tokens.weight` (tied embeddings).
- Final-logits parity vs HF `LlamaForCausalLM` on fp32 CPU: **max\|Δ\| = 0.0**
  (test prompt: `"The capital of France is"`).
- Argmax next-token agreement: HF predicts `' the'`, ours predicts `' the'`.
  (The popular intuition would be `' Paris'`; the actual model puts Paris at rank 2.)
- wikitext-2-raw-v1 validation perplexity, sliding window seq=1024 / stride=512
  over 62,403 target tokens: **ours 15.370989 vs HF 15.370990** (Δ ≈ 9 × 10⁻⁷,
  pure fp32 noise floor).
- Tokenization of `"The capital of France is"` → `[504, 3575, 282, 4649, 314]`
  (5 BPE pieces).

**Architecture spec the parity confirms** (all from `config.json`, line-for-line):
30 decoder layers · hidden 576 · intermediate 1536 · 9 Q heads / 3 KV heads (GQA
3:1) · head_dim 64 · vocab 49,152 · RoPE θ=100,000 (split-halves) · RMSNorm ·
SwiGLU · tied embeddings · no biases anywhere.

### 2. Continued pretraining — "Then I trained it on TinyStories and dropped perplexity 45%"

- 100,000,000 tokens of `roneneldan/TinyStories`, starting from official
  SmolLM2-135M safetensors loaded into our class.
- Recipe (verified against nanotron defaults + paper §6): AdamW(0.9, 0.95),
  ε 1e-8, peak LR **3e-4** (10× lower than from-scratch's 3e-3), WSD with
  warmup 200 and decay last 20% (decay starts at step 19,531), weight_decay
  **0.01** (on 2D params only), grad_clip 1.0, bf16, seq_len 1024, micro_batch 4.
- **Validation PPL: 6.8945 → 3.7900 (−45.0%)** on 199,485 TinyStories-val
  target tokens.
- Wall-clock: **116.1 minutes** on NVIDIA GB10. Sustained throughput
  ~14,300 tok/s through the second hour.
- Bucket-mean training loss (1000-step buckets): 1.586 (first) → **1.3138** (last bucket = (24000, 24414], 414 rows).
- Best single-batch loss: **0.9088** at step 22,353 (deep in the decay phase).

---

## Before vs After — TinyStories generations (same seed, temp 0.7, top-k 40)

### Prompt: `"Once upon a time, there was a little"`

**BEFORE:** Once upon a time, there was a little girl named Lily. Lily loved to
play and sing with her friends. One day, she heard that she could become a
doctor and help people who were sick. But she didn't know much about medicine
or how to take care of herself…

**AFTER:** Once upon a time, there was a little boy named Timmy. Timmy loved
to go on adventures with his family. One day, they went on a walk in the
woods. Timmy saw a big tree and he asked his family, *"Can I climb the tree?"*
His parents said, *"No, it's too high. But we can climb it together."* Timmy
was very happy and wanted to climb the tree. He started to climb…

*What changed:* dialogue with quotes, simpler vocab, full narrative arc with a
problem-then-resolution beat — classic TinyStories signature.

### Prompt: `"The brave little mouse"`

**BEFORE:** The brave little mouse that runs the way I have been told to run,
and who is always at the ready, to get me out of trouble, and give me back my
hat and coat, to make me happy and to help me to live like a man, and to make
me understand that I am safe and happy and comfortable in my new home, and
that I shall have plenty of **plenty of plenty of plenty of**…

**AFTER:** The brave little mouse was so excited that he couldn't wait to find
out what the old man was up to. Once upon a time, there was a big, big bear.
He was very strong and brave. He lived in a forest with lots of trees. One
day, the bear decided to go for a walk…

*What changed:* base model degenerates into a repetition loop. Trained model
exits cleanly, even rolls a fresh `"Once upon a time"` — proof it's internalised
EOS-as-story-break.

### Prompt: `"In a faraway forest,"`

**BEFORE:** In a faraway forest, a wise old owl named Ollie looked out of his
tree and saw a young girl who seemed lost and sad. **Ollie asked the owl** why
she was so sad…

**AFTER:** In a faraway forest, there was a small, beautiful bird who loved to
whistle. One day, she was whistling in the forest when she heard a noise coming
from a nearby tree. The bird knew she had to be brave and follow the noise.
She flew down to the tree and pecked at the trunk. She heard a voice coming
from the trunk. *"Hello. What are you doing"…*

*What changed:* base has a self-reference bug (Ollie *is* the owl). Trained
gives a clean character setup with dialogue.

---

## Top-5 next-token for `"The capital of France is"` (verified)

| Rank | Token | Probability | Logit |
|---:|---|---:|---:|
| 1 | `' the'` | 0.2617 | 14.023 |
| 2 | `' Paris'` | 0.0938 | 12.997 |
| 3 | `' located'` | 0.0731 | 12.747 |
| 4 | `' called'` | 0.0439 | 12.237 |
| 5 | `' a'` | 0.0392 | 12.125 |

Source: `results/topk_predictions.json`. This is the catch the parity test
makes obvious — the popular "of course it says Paris" intuition is wrong; the
deterministic argmax wants to complete with `' the'` because the model has seen
many `"The capital of France is the city of Paris"` constructions.

---

## All visuals available for posting

```
results/plots/loss_curve.png              demo from-scratch run (150 steps)
results/plots/wsd_schedule.png            WSD shape × 3 configs
results/plots/rope_tables.png             RoPE cos/sin, 256 × 64
results/plots/residual_norms.png          per-layer residual L2 norm
results/plots/tinystories_loss_curve.png  THE training run (24,414 steps, with bucket-mean + LR + decay marker)
results/attention/layer_00.png            9 Q heads, layer 0
results/attention/layer_14.png            9 Q heads, layer 14 (mid-stack)
results/attention/layer_29.png            9 Q heads, layer 29 (final)
```

All currently default-landscape matplotlib output. For IG (1080×1350 or 1080×1920)
they will need a re-export pass.

---

## All data artifacts (16 catalog files + 5 TinyStories files)

```
results/summary.json                      11-line digest
results/perplexity.json                   ours vs HF, 62,403 tokens
results/topk_predictions.json             top-10 × 5 prompts
results/generations.txt                   4 prompts × 3 temperatures
results/training_recipe_resolved.json     paper-resolved hyperparameters
results/param_count.log                   `python3 model.py`
results/parity.log                        `python3 verify.py` (max|Δ|=0)
results/loss_curve.csv                    150-step demo (step, loss, lr)
results/comparison_with_hf.md             6 cross-checks vs HF
results/tinystories_before.txt            BASE PPL + 3 samples
results/tinystories_after.txt             TRAINED PPL + 3 samples
results/tinystories_train.csv             24,414 rows (step, loss, lr, tok_seen)
results/tinystories_train.log             52 KB stdout log
results/tinystories_summary.md            long-form write-up (prior run)
checkpoint.pt                             538 MB — demo-run state_dict
checkpoint_tinystories.pt                 269 MB — TinyStories-trained bf16 state_dict
```

---

## Suggested IG carousel — 10 slides, all text verified

1. **Hook.** *"I rebuilt SmolLM2-135M from scratch and matched HuggingFace to zero."*
2. **The number.** *134,515,008 parameters. lm_head tied to embed_tokens.*
3. **The spec.** *30 layers · hidden 576 · 9 Q / 3 KV heads · RoPE θ=100,000 · vocab 49,152.*
4. **The code.** Crop of `model.py` forward pass (~20 lines).
5. **The proof.** Terminal screenshot: `max |Δlogits| = 0.000e+00` (CPU fp32). Output of `python3 verify.py`.
6. **Attention.** `results/attention/layer_14.png`.
7. **Then I trained it.** *100M tokens TinyStories · 116 min on GB10 · PPL 6.8945 → 3.7900 (−45%).*
8. **Loss curve.** `results/plots/tinystories_loss_curve.png`.
9. **Before / After.** The mouse prompt — base loops on "plenty of plenty of", trained writes a clean story with dialogue.
10. **CTA.** *Code + write-up in bio. Full walkthrough on YouTube.*

## Suggested IG story tiles — 7 tiles, save as a Highlight

1. Cover: `134,515,008` over a dark background. *Swipe → carousel*.
2. `max \|Δlogits\| = 0` · HF parity verified.
3. `15.371 = 15.371` · wikitext-2 PPL, ours vs HF, Δ ≈ 9 × 10⁻⁷.
4. `6.8945 → 3.7900` · TinyStories PPL, −45%.
5. `116.1 min on GB10` · 100M tokens · WSD schedule.
6. 5-second GIF of the Forward Pass Cinema (`architecture_docs.html`) scrubbing layers.
7. Link sticker → YouTube long-form.

## YouTube chapter outline (uses the same data, expanded)

1. **Hook** (30 s) — the parity claim. Show `verify.py` running live.
2. **Why SmolLM2-135M** (1 min) — see README §1.
3. **Architecture spec sheet** (3 min) — README §3.
4. **`model.py` walkthrough** (10 min) — README §7, component by component.
5. **The parity gate** (3 min) — `verify.py` + the topk-table reveal that Paris is #2.
6. **Continued pretraining on TinyStories** (5 min) — `train_tinystories.py` recipe + before/after samples.
7. **Loss curve + decay-phase callout** (3 min) — `tinystories_loss_curve.png`.
8. **What's matched vs inferred** (2 min) — README §11.
9. **What's next** (1 min) — KV cache, Qwen3 port.

---

## Cross-checks you can paste under any claim (for receipts)

- Parity zero: `cat results/parity.log`
- Param count: `cat results/param_count.log`
- PPL equality: `cat results/perplexity.json`
- TinyStories before/after: `cat results/tinystories_{before,after}.txt`
- Recipe used: `cat results/training_recipe_resolved.json`
- Top-k Paris-is-#2: `cat results/topk_predictions.json | jq '."The capital of France is"[:3]'`
