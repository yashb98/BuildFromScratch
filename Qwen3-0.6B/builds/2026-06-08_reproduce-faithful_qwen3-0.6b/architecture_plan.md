# Architecture Plan — Qwen3-0.6B, reproduce-faithful

- **Target:** `Qwen/Qwen3-0.6B-Base` (596M text-only decoder)
- **Build mode:** reproduce-faithful
- **Date:** 2026-06-08
- **Verify tolerance:** max-abs-err < 1e-3 in fp32 (hard gate in Phase 5)
- **Sources:**
  - HF model card: https://huggingface.co/Qwen/Qwen3-0.6B-Base
  - Tech report: https://arxiv.org/abs/2505.09388 (Qwen3, May 2025)
  - Reference impl (HF): https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
  - Reference impl (from-scratch): https://huggingface.co/rasbt/qwen3-from-scratch  ·  https://magazine.sebastianraschka.com/p/qwen3-from-scratch
- **Live config.json (pulled via `AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base")` on 2026-06-08):**
  hidden_size 1024 · num_hidden_layers 28 · num_attention_heads 16 · num_key_value_heads 8 · head_dim 128 · intermediate_size 3072 · vocab_size 151,936 · max_position_embeddings 40,960 · rope_theta 1,000,000 · rms_norm_eps 1e-6 · tie_word_embeddings true · attention_bias false · hidden_act silu

## Component-by-component

| Component | Faithful spec (from HF config + modeling_qwen3.py) | Decision for this build | Rationale | Source |
|---|---|---|---|---|
| Decoder layout | Pre-norm decoder-only transformer; `embed_tokens → 28 × Block → final RMSNorm → lm_head (tied)` | Same | Faithful | HF config + modeling_qwen3 |
| Embedding | `nn.Embedding(151936, 1024)`, tied to lm_head | Same | Faithful | config: `tie_word_embeddings=true` |
| Per-block norms | `input_layernorm`, `post_attention_layernorm` — both `Qwen3RMSNorm(1024, eps=1e-6)` | Same | Faithful | modeling_qwen3.Qwen3DecoderLayer |
| RMSNorm impl | x → cast fp32 → x * rsqrt(mean(x²) + eps) → multiply by weight → cast back | Same (mirrors `RMSNorm` in our SmolLM2 model_full.py) | Numerically stable, matches HF | HF Qwen3RMSNorm |
| Attention type | GQA, 16 query heads, 8 KV heads, `n_rep = 2` | Same | Faithful | config: 16/8 |
| Head dim | **128 (explicit, NOT hidden/n_heads)** — so q_proj out = 16×128 = 2048, k_proj/v_proj out = 8×128 = 1024 | Same | Faithful — critical: `head_dim ≠ hidden_size // num_heads` here (hidden/n_heads=64; head_dim=128) | config: `head_dim=128` |
| Projection biases | `attention_bias=false` ⇒ no bias on q/k/v/o_proj | No biases | Faithful | config |
| **QK-Norm** | `q_norm = Qwen3RMSNorm(head_dim=128, eps=1e-6)`, `k_norm = Qwen3RMSNorm(head_dim=128, eps=1e-6)`. Applied as: `q = q_norm(q_proj(x).view(B,T,16,128))` then `transpose(1,2)`. RoPE applied AFTER QK-Norm. | Same | **This is the #1 architectural difference vs Llama/Qwen2/SmolLM2 — must implement exactly or verify gate fails.** The norm weight is a single (128,) vector broadcast across heads. | modeling_qwen3.Qwen3Attention.__init__ + forward (verified from HF source) |
| RoPE | Standard rotate_half layout (NOT interleaved), `theta = 1e6`, applied to Q and K after QK-Norm | Same | Faithful — note θ differs from Llama (1e4) and SmolLM2-v2 (1e5) | config: `rope_theta=1000000` |
| RoPE cache | Precompute cos/sin tables up to `max_position_embeddings=40960`, register as non-persistent buffers | Same (mirrors SmolLM2 pattern) | Same trick keeps state_dict clean | mirror of our SmolLM2 model_full.py |
| Sliding window | `use_sliding_window=false`, `sliding_window=null` → all layers full-attention (`layer_types: 28× "full_attention"`) | Full attention every layer | Faithful — Qwen3 dense base does not use sliding window at this size | config |
| Attention impl | `F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)`; explicit `repeat_interleave(n_rep=2)` on K and V to match Q heads | Same | Mirrors SmolLM2 pattern | mirror + HF |
| FFN | SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`. gate/up/down all bias-free; intermediate 3072. | Same | Faithful | config: `hidden_act="silu"`, intermediate_size=3072 |
| Final norm | `Qwen3RMSNorm(1024, eps=1e-6)` | Same | Faithful | modeling_qwen3.Qwen3Model.norm |
| LM head | `nn.Linear(1024, 151936, bias=False)`, weight tied to `embed_tokens.weight` | Same | Faithful | config: `tie_word_embeddings=true` |
| Init scheme | HF default `initializer_range=0.02` — `Normal(0, 0.02)` for Linear and Embedding weights | Same | Faithful | config: `initializer_range=0.02` |
| Dropout | `attention_dropout=0.0`, no other dropout | None | Faithful | config |
| Vocab size | 151,936 (Qwen tokenizer BBPE, vocab is padded to multiple-of-128 from the true ~151,669) | Same | Faithful | config |
| Position embeddings | None separately; positional info is entirely from RoPE | RoPE only | Faithful | standard |

## Param count sanity check (forward calc, fp32 unique params)

| Block component | Per-block params | × 28 blocks |
|---|---:|---:|
| q_proj (1024→2048) | 2,097,152 | 58,720,256 |
| k_proj (1024→1024) | 1,048,576 | 29,360,128 |
| v_proj (1024→1024) | 1,048,576 | 29,360,128 |
| o_proj (2048→1024) | 2,097,152 | 58,720,256 |
| q_norm (head_dim=128) | 128 | 3,584 |
| k_norm (head_dim=128) | 128 | 3,584 |
| input_layernorm (1024) | 1,024 | 28,672 |
| post_attention_layernorm (1024) | 1,024 | 28,672 |
| gate_proj (1024→3072) | 3,145,728 | 88,080,384 |
| up_proj (1024→3072) | 3,145,728 | 88,080,384 |
| down_proj (3072→1024) | 3,145,728 | 88,080,384 |
| **Block subtotal** | **15,730,944** | **440,466,432** |

| Non-block | Params |
|---|---:|
| embed_tokens / lm_head (tied — counted once) (151936×1024) | 155,582,464 |
| final norm (1024) | 1,024 |
| **Total** | **596,049,920** |

≈ **596M**, matches the published "0.6B" branding (the Qwen team rounds to 0.6B; widely-cited exact figure is 596M).

## Class layout (mirrors `model_full.py` conventions)

```
Qwen3Config (dataclass)
RMSNorm(hidden_size, eps)
_build_rope_cache, _rotate_half, _apply_rope
Qwen3Attention(cfg)           # adds q_norm, k_norm vs SmolLM2 Attention
Qwen3MLP(cfg)                 # identical to SmolLM2 MLP (SwiGLU)
Qwen3Block(cfg)               # identical pre-norm structure to SmolLM2 Block
Qwen3Model(cfg)               # .embed_tokens, .layers, .norm, rope buffers
Qwen3ForCausalLM(cfg)         # .model, .lm_head (tied), generate()
num_params(model)
```

State-dict key names will match HF Qwen3 exactly (`model.embed_tokens.weight`, `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `model.layers.{i}.self_attn.{q,k}_norm.weight`, `model.layers.{i}.mlp.{gate,up,down}_proj.weight`, `model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`, `model.norm.weight`) so `verify.py` can `load_state_dict(strict=False)` with only `lm_head.weight` allowed missing (tied).

## Verify gate (Phase 5)

- **Method:** load `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype=torch.float32)`, copy state dict into ours, run identical input through both, compute `max |Δlogits|`.
- **Tolerance:** `< 1e-3` in fp32.
- **Tie-breaker check:** assert `argmax` of last-token logits matches between the two models.
- **Fixed input:** `tokenizer("The capital of France is", return_tensors="pt").input_ids` (same string the SmolLM2 verify used — keeps the pattern).

## Open questions

1. **`generate()` decoding defaults:** mirror SmolLM2 (`temperature=0.8, top_k=50, max_new_tokens=64`)? Or set base-model-friendly defaults (greedy / lower temp)?
2. **Compile-friendliness for QK-Norm:** explicit RMSNorm call inside attention adds two non-fused ops. For training speed we may want to enable `torch.compile` (kept optional in the SmolLM2 `train.py`). No decision needed at the arch stage — flagged for Phase 9.
3. **Layer-types tensor in config:** HF config has `layer_types` enumerated for all 28 layers, all "full_attention". Should we mirror this as a config field (forward-compatible with future sliding-window builds) or hardcode? Recommend: hardcode for the faithful build (simpler); add the field in the exploratory build if relevant.

## Approval needed

If approved, Phase 4 writes:
- `Qwen3-0.6B/model.py` (single-file impl mirroring `model_full.py` layout)
- `Qwen3-0.6B/verify.py` (HF-weight loader)
- `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/test_model.py`
- `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/verify_run.py`

…and runs the test suite + verify before requesting Phase 6 approval.
