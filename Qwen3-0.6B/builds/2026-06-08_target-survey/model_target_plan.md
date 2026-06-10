# Model Target Plan — Qwen ~0.5B class

- **Date:** 2026-06-08
- **Recency cutoff (≤6mo):** 2025-12-08
- **Build mode(s) planned:** Three sequential builds — reproduce-faithful, reproduce-modernized, reproduce-exploratory
- **Family preference:** Qwen (latest, open-weights, text-only decoder)
- **Size band:** ~0.5B params (sub-billion)
- **Survey breadth:** 4 candidate Qwen models in the band

## Comparison

| Target | Family | Params (exact) | Hidden / Layers / Heads (Q/KV) / Head dim | FFN dim | Vocab | Max ctx | RoPE θ | Tied emb | License | HF repo | Reproducibility difficulty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-0.6B** | qwen3 | **596M** (text-only) | 1024 / 28 / 16 / 8 / 128 (GQA 2:1) | 3072 (SwiGLU) | 151,936 | 40,960 | 1e6 | yes | Apache 2.0 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | **LOW** — clean text decoder; published from-scratch reference (Raschka) |
| Qwen2.5-0.5B | qwen2 | ~494M (text-only) | 896 / 24 / 14 / 2 / 64 (GQA 7:1) | 4864 (SwiGLU) | 151,936 | 32,768 | 1e6 | yes | Apache 2.0 | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | LOW — older but well-trodden |
| Qwen2-0.5B | qwen2 | ~494M (text-only) | 896 / 24 / 14 / 2 / 64 | 4864 (SwiGLU) | 151,936 | 131,072 | 1e6 | yes | Apache 2.0 | [Qwen/Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) | LOW — superseded by Qwen2.5; no reason to pick |
| ~~Qwen3.5-0.8B~~ | qwen3_5 (multimodal) | ~0.9B incl. ViT | 1024 / 24 / 8 / 2 / 256, hybrid linear+full attn, MTP head, partial RoPE, MRoPE | 3584 | 248,320 | 262,144 | 1e7 | yes | Apache 2.0 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | **OUT OF SCOPE** — vision encoder + linear/full attn hybrid + MTP + MRoPE; not a text-only decoder |

Confirmed from runtime `AutoConfig.from_pretrained` pulls — figures above are not hand-quoted, they come from the live config.json on each repo.

---

### Qwen3-0.6B (recommended)

- **HF model card:** https://huggingface.co/Qwen/Qwen3-0.6B (base: https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- **Paper:** Qwen3 Technical Report, arXiv:2505.09388 (May 2025) — https://arxiv.org/abs/2505.09388
- **Architecture summary:** decoder-only transformer; 28 layers × 1024 hidden; GQA with 16 query heads / 8 KV heads / head_dim 128; SwiGLU FFN (intermediate 3072); RMSNorm pre-norm (eps 1e-6); RoPE θ=10^6, max_position_embeddings 40,960; tied input/output embeddings; vocab 151,936 (Qwen tokenizer BBPE).
- **Training corpus (disclosed):** ~36T tokens, 119 languages. Exact source mixture not fully disclosed (typical for SOTA frontier releases).
- **Reported numbers:** Qwen team's blog states Qwen3-0.6B surpasses Qwen2.5-instruct on math, code, and commonsense reasoning. Specific MMLU/GSM8K/HumanEval numbers vary by source and post-training stage; the base checkpoint reproduction will compare against the open base weights via PPL deltas.
- **Community references — verified:**
  - [Sebastian Raschka — *Understanding and Implementing Qwen3 From Scratch*](https://magazine.sebastianraschka.com/p/qwen3-from-scratch) (article + code)
  - [rasbt/qwen3-from-scratch](https://huggingface.co/rasbt/qwen3-from-scratch) — pure-PyTorch standalone notebook for the 0.6B with KV-cache variants
  - [rasbt/LLMs-from-scratch ch05/11_qwen3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3) — book code
  - [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py) — reasoning extension
- **Known gotchas:** RoPE θ=10^6 (not the 10^4 GPT-style default — easy mistake); QK norm is applied per-head inside attention in Qwen3 (new vs Qwen2); tied embeddings with no separate lm_head bias; vocab_size pad mismatch is common.
- **Why it fits:** newest Qwen family with a sub-billion text-only base, has an Apache 2.0 license, exists as a stable open base checkpoint (no thinking-mode prompt tags needed for base PPL eval), and has a high-quality community PyTorch reference to triangulate verify-gate failures against.

### Qwen2.5-0.5B (runner-up)

- **HF:** https://huggingface.co/Qwen/Qwen2.5-0.5B
- **Why it's runner-up:** smaller, more aggressive GQA ratio (7:1 KV reduction), more proven recipe with public reproductions. But older (Sep 2024) — fails the "latest Qwen" criterion you specified.

### Qwen2-0.5B (skipped)

- Same Qwen2 arch as 2.5. No reason to pick over 2.5.

### Qwen3.5-0.8B (excluded — out of scope)

- The 3.5 0.8B is a multimodal model: separate vision tower (12-layer ViT, hidden 768, patch 16) + text decoder with **hybrid linear/full attention** (3 linear-attention layers between every full-attention layer), MRoPE multi-axis rotary, partial rotary factor 0.25, attention-output-gating, and a Multi-Token Prediction head.
- Not a clean text-only transformer; implementing it from scratch would mean writing a vision encoder, Mamba/linear-attention layers, MTP heads, and MRoPE — three full builds of that is not realistic.
- Flagged here so the "pick the newest Qwen" instinct doesn't quietly drift us into a multimodal/SSM reproduction we don't want.

---

## Three-build slot allocation (proposed)

| Build | Folder | Mode | What's varied |
|---|---|---|---|
| 1 | `builds/2026-06-08_reproduce-faithful_qwen3-0.6b/` | Faithful | Match HF config exactly; published-paper recipe; verify gate <1e-3 max-err vs HF weights |
| 2 | `builds/2026-06-08_reproduce-modernized_qwen3-0.6b/` | Modernized | Same arch; current best-practice recipe (fineweb-edu, modern WSD); verify gate on unchanged components |
| 3 | `builds/2026-06-08_reproduce-exploratory_qwen3-0.6b/` | Exploratory | Arch as start, swap 1–2 components (candidates: NoPE vs RoPE, GQA→MQA, RMSNorm→DyT, etc. — picked in Phase 3); smoke-test only |

Each build gets its own architecture_plan / training_plan / hp_tuning_plan / train script / eval / notebooks.

---

## Recommendation

**Qwen3-0.6B** for all three builds. Strongest combination of: newest text-only Qwen, ~0.5B size, GQA 2:1 (interesting verify target but not exotic), high-quality community reference for triangulation, and Apache 2.0.

## Open questions for the user

1. **Base vs Instruct:** I recommend reproducing the **base** checkpoint (`Qwen/Qwen3-0.6B-Base`), not the post-trained instruct (`Qwen/Qwen3-0.6B`). Base PPL on held-out text is the clean reproduction signal; instruct adds thinking-mode prompt tokens that complicate eval. Confirm OK.
2. **Verify tolerance:** default <1e-3 max-abs-err in fp32 for the faithful build. Acceptable, or do you want stricter?
3. **Exploratory swap (Build 3):** I'll propose 2–3 component swap options in Phase 3 (e.g., RoPE→NoPE, GQA→MQA, RMSNorm→DyT/SLA, attention bias toggle). Any preference now, or decide then?
