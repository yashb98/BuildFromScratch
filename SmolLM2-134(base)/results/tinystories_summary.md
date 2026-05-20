# TinyStories continued pretraining — final wrap-up

## Headline

**SmolLM2-135M, 100M tokens of continued pretraining on roneneldan/TinyStories.**

| Metric | Before | After | Δ |
|---|---|---|---|
| TinyStories-val perplexity | **6.8945** | **3.7893** | **−45.0%** |
| TinyStories-val ln(PPL) | 1.931 | 1.332 | −0.599 nats |
| Best training-batch loss | 1.220 (step 0–1000) | **0.910** (step ~22500) | −0.310 nats |
| Bucket-mean training loss | 1.586 (step 0–1000) | 1.313 (step 24–25k) | −0.273 nats |

Wall clock: **137.3 minutes** on NVIDIA GB10, bf16. (Original estimate: 135 min — off by 2 min, ~1.5%.)

## Side-by-side qualitative comparison

Same three prompts, same sampling (temperature 0.7, top-k 40, deterministic seed). The shift is clearly in TinyStories direction — simpler vocab, dialogue, clear narrative beats.

### Prompt: `"Once upon a time, there was a little"`

**Before:**
> Once upon a time, there was a little girl named Lily. Lily loved to play and sing with her friends. One day, she heard that she could become

**After:**
> Once upon a time, there was a little boy named Timmy. Timmy loved to go on adventures with his family. One day, they went on a walk in the woods. Timmy saw a big tree and he asked his family, "Can I climb the tree?" His parents said, "No, it's too high. But we can climb it together." Timmy was very happy and wanted to climb the tree. He started to climb

After-version has: full narrative arc, dialogue with quotes, a problem-then-resolution beat — classic TinyStories signature. Before version stalled mid-clause.

### Prompt: `"The brave little mouse"`

**Before:**
> The brave little mouse that runs the way I have been told to run, and who is always at the ready, to get me out of trouble, and give me back

**After:**
> The brave little mouse was so excited that he couldn't wait to find out what the old man was up to.Once upon a time, there was a big, big bear. He was very sad because he had no friends. One day, he saw a small rabbit. The bear said, "I am so sorry, I don't know how to make friends. Please come join me for dinner."

Before is a runaway compound sentence in adult voice. After commits to the kid-story register and even rolls a fresh "Once upon a time" — a signal the model has internalized that EOS = story-end.

### Prompt: `"In a faraway forest,"`

**Before:**
> In a faraway forest, a wise old owl named Ollie looked out of his tree and saw a young girl who seemed lost and sad. Ollie asked the owl wh

**After:**
> In a faraway forest, there was a small fairy named Lila. Lila was a very kind fairy who loved to help others. One day, Lila was playing with her friend, a tiny rabbit named Sam. "Hi, Sam! Do you want to play with me?" asked Lila. Sam looked at her and said, "Yes, thank you! But I am lost. Can you help me

Before contains a self-reference bug ("Ollie asked the owl" — Ollie *is* the owl). After is a clean character-meets-friend, dialogue-driven setup.

## Training trajectory (per-1000-step bucket means)

| Bucket | Mean loss | Min loss | LR phase |
|---|---|---|---|
| 0–1000 | 1.586 | 1.220 | warmup (0→3e-4) |
| 1000–2000 | 1.482 | 1.195 | stable |
| 5000–6000 | 1.404 | 1.092 | stable |
| 10000–11000 | 1.374 | 1.044 | stable |
| 15000–16000 | 1.348 | 1.014 | stable |
| 18000–19000 | 1.339 | 0.968 | stable (last) |
| 19000–20000 | 1.339 | 1.012 | decay begins (step 19531) |
| 21000–22000 | 1.324 | 0.981 | decay |
| 22000–23000 | **1.312** | **0.910** | decay |
| 24000–25000 | 1.313 | 0.982 | decay ends |

The decay phase delivered an additional **−0.03 nats** on top of the stable plateau. Best single-batch loss across the run: **0.910** at step ~22,500.

## Recipe (verified against nanotron canonical defaults)

| Field | Value | Source |
|---|---|---|
| Init | official SmolLM2-135M safetensors (continued, not from-scratch) | HuggingFaceTB/SmolLM2-135M |
| Tokens trained | 100,000,000 | this run |
| Sequence length | 1024 | this run |
| Micro batch | 4 | this run |
| Grad accumulation | 1 | this run |
| Total steps | 24,414 | computed |
| Optimizer | AdamW(0.9, 0.95), eps 1e-8 | paper §4.1 |
| Peak LR | **3e-4** (10× lower than from-scratch's 3e-3 — standard for continued pretraining) | this run |
| Schedule | WSD: warmup 200, decay last 20% (steps 19,531 → 24,414) | paper §6 |
| Weight decay | **0.01** (on 2D params only) | nanotron config_smollm2_135M.yaml |
| Gradient clip | 1.0 | nanotron config |
| Precision | bf16 | model card |

## Throughput characterization (GB10)

| Phase | tok/s | Notes |
|---|---|---|
| Stable, first hour | 14,500–14,900 | matches benchmark |
| Stable, second hour | 12,100–12,500 | mild ~17% drift |
| Mean across full run | **12,150** | (100M tokens / 137.3 min) |

The drift is not thermal (max temp 72 °C, well below limit). Most likely shared-machine effect (5 users on the box); reproducible benchmarks under load would need exclusive access.

## Files this run produced

```
checkpoint_tinystories.pt              269 MB — bf16 state_dict + metadata
results/tinystories_train.log           52 KB — full stdout log
results/tinystories_train.csv          1.1 MB — 24,414 rows (step,loss,lr,tok)
results/tinystories_before.txt         1.3 KB — baseline samples + PPL
results/tinystories_after.txt          1.3 KB — trained-model samples + PPL
results/plots/tinystories_loss_curve.png       composite loss+lr plot
```

## Interpretation

The 45% perplexity reduction (6.89 → 3.79) on TinyStories-val after 100M tokens
is consistent with what continued pretraining gives you when the *target
distribution is a strict subset of the original pretrain distribution*: the model
re-allocates probability mass toward the in-domain register (simple vocab,
character-driven dialogue, short clean sentences) and away from the broader
internet's stylistic noise.

This is a **style-shift**, not a knowledge addition: the base SmolLM2-135M
already "knew how to write stories"; what it didn't have was the strong prior
that *every output should be one*. After 100M tokens it does. The qualitative
samples are more diagnostic than the perplexity number — and they show the
shift clearly.

## What would push this further

1. **More tokens**: full epoch over TinyStories (~400M tokens) would likely drop
   PPL another 0.3–0.5. ~6 hours additional GB10 time at observed throughput.
2. **Lower peak LR** (1e-4 instead of 3e-4): less catastrophic-forgetting risk
   if you also care about preserving general-text quality. We didn't measure
   wikitext-2 PPL post-training but it almost certainly got worse — that's the
   tradeoff continued pretraining always makes.
3. **Mix in 10–20% of the original pretraining mixture** to prevent forgetting
   on out-of-domain text. Standard practice for production continued pretraining.
