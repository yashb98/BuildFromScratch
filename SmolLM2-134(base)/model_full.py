"""
SmolLM2-135M reproduction — single-file PyTorch implementation.

Architecture is LlamaForCausalLM as shipped in:
    https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json

Every architectural choice is justified inline with the canonical source
(config field, paper section, or HF transformers reference).

Parameter naming mirrors transformers.models.llama exactly, so the official
SmolLM2-135M safetensors load into this module via `load_state_dict` with no
key remapping. See verify.py for the parity test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config — every default matches config.json at the SmolLM2-135M repo HEAD.
# Source: HuggingFaceTB/SmolLM2-135M/config.json (fetched in spec sheet step).
# ---------------------------------------------------------------------------
@dataclass
class SmolLM2Config:
    vocab_size: int = 49152                       # config.json: vocab_size
    hidden_size: int = 576                        # config.json: hidden_size
    intermediate_size: int = 1536                 # config.json: intermediate_size
    num_hidden_layers: int = 30                   # config.json: num_hidden_layers
    num_attention_heads: int = 9                  # config.json: num_attention_heads
    num_key_value_heads: int = 3                  # config.json: num_key_value_heads (GQA: 9 Q / 3 KV)
    max_position_embeddings: int = 8192           # config.json: max_position_embeddings
    rope_theta: float = 100_000.0                 # config.json: rope_theta (note: v2 uses 100k, v1 was 10k)
    rms_norm_eps: float = 1e-5                    # config.json: rms_norm_eps
    initializer_range: float = 1.0 / math.sqrt(576)  # config.json: initializer_range = 0.041666... = 1/sqrt(576)
    tie_word_embeddings: bool = True              # config.json: tie_word_embeddings
    attention_bias: bool = False                  # config.json: attention_bias
    attention_dropout: float = 0.0                # config.json: attention_dropout
    # hidden_act = "silu" → SwiGLU(silu(gate) * up), per HF LlamaMLP. Hardcoded below.

    @property
    def head_dim(self) -> int:
        # Llama convention: head_dim = hidden_size // num_attention_heads.
        # 576 / 9 = 64.
        return self.hidden_size // self.num_attention_heads


# ---------------------------------------------------------------------------
# RMSNorm
# Reference: Zhang & Sennrich 2019; used in Llama since v1.
# HF impl: transformers/models/llama/modeling_llama.py :: LlamaRMSNorm
# Note the .pow(2).mean(-1) vs .mean(.pow(2)) — same math, different memory pattern.
# We compute in fp32 (HF does too) for numerical stability, then cast back.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


# ---------------------------------------------------------------------------
# RoPE — rotary position embeddings, Su et al. 2021 (arXiv:2104.09864).
# We precompute cos/sin tables up to max_position_embeddings and slice per call.
#
# Layout: rope_interleaved=false in config.json. This is the HF convention:
# split the head_dim in half along the LAST dim and apply rotate_half, i.e.
#   x rotated = concat(-x[..., d/2:], x[..., :d/2])
# This is *not* the GPT-NeoX/EleutherAI interleaved pair layout. Getting this
# wrong is the #1 cause of weight-loading silently passing but outputs being
# garbage, so this is the convention to verify against HF in verify.py.
# ---------------------------------------------------------------------------
def _build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    # inv_freq shape: (head_dim/2,)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)   # (seq_len, head_dim) — duplicated for rotate_half
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Splits last dim in half and rotates. Matches HF Llama's rotate_half.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # cos/sin: (seq_len, head_dim). q/k: (B, n_heads, seq_len, head_dim).
    # Broadcasting: unsqueeze to (1, 1, seq_len, head_dim).
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# Q heads = 9, KV heads = 3 → each KV head is shared by 3 Q heads.
# Reference: Ainslie et al. 2023 (arXiv:2305.13245).
# No biases on q_proj/k_proj/v_proj/o_proj (config: attention_bias=false).
# Use F.scaled_dot_product_attention so we get flash attention on supported HW.
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # 3 — each KV repeated this many times

        # Projection sizes:
        #   Q: hidden_size → n_heads * head_dim         (576 → 576)
        #   K: hidden_size → n_kv_heads * head_dim      (576 → 192)
        #   V: hidden_size → n_kv_heads * head_dim      (576 → 192)
        #   O: hidden_size → hidden_size                (576 → 576)
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)        # (B, nH, T, D)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)     # (B, nKV, T, D)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)     # (B, nKV, T, D)

        # RoPE applied to Q and K only (not V), as in standard Llama.
        q, k = _apply_rope(q, k, cos, sin)

        # GQA: repeat KV heads to match Q heads. SDPA in recent PyTorch supports
        # enable_gqa=True, but explicit repeat keeps this version-agnostic and
        # makes the math transparent for teaching.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (B, nH, T, D)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Causal masked SDPA. dropout_p=0.0 per config.
        # is_causal=True is correct only when there's no extra padding mask;
        # for batched padded inputs you'd pass attn_mask explicitly.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None),
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# Reference: Shazeer 2020 (arXiv:2002.05202); standard Llama MLP.
#   y = down( silu(gate(x)) * up(x) )
# No biases (Llama convention; HF LlamaMLP defaults bias=False).
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer block — pre-norm, residual around attention and MLP.
# Naming (input_layernorm, post_attention_layernorm, self_attn, mlp) mirrors
# HF Llama exactly so safetensors load without key remapping.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, attention_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# Full model. HF's structure:
#   LlamaForCausalLM
#     .model: LlamaModel
#         .embed_tokens
#         .layers: ModuleList[LlamaDecoderLayer]
#         .norm
#     .lm_head        # shares weight with embed_tokens when tie_word_embeddings=True
# We mirror this so state dicts are drop-in compatible.
# ---------------------------------------------------------------------------
class SmolLM2Model(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        # Buffer-cached RoPE tables. Registered as non-persistent so they don't
        # appear in state_dict (avoids spurious "missing keys" when loading HF
        # weights — HF doesn't ship rope tables).
        cos, sin = _build_rope_cache(cfg.max_position_embeddings, cfg.head_dim,
                                     cfg.rope_theta, device='cpu', dtype=torch.float32)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = input_ids.shape[1]
        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[:T].to(x.dtype)
        sin = self.rope_sin[:T].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
        return self.norm(x)


class SmolLM2ForCausalLM(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.cfg = cfg
        self.model = SmolLM2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            # Weight tying: lm_head.weight IS embed_tokens.weight (same storage).
            # config.json: tie_word_embeddings = true.
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        # Normal init with std = initializer_range, per config.json
        # (= 1/sqrt(hidden_size) = 0.04167 for 135M). HF's PreTrainedModel
        # uses the same scheme for Llama.
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        # RMSNorm.weight is initialized to ones in its __init__; nothing to do.

    def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None):
        hidden = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Standard causal LM shift: predict token t+1 from positions ≤ t.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int | None = 50) -> torch.Tensor:
        """Greedy/top-k sampling. No KV cache — recomputes the prefix each step.
        Fine for the educational starter; swap in a KV cache for production."""
        self.eval()
        for _ in range(max_new_tokens):
            # Truncate context to max_position_embeddings.
            ctx = input_ids[:, -self.cfg.max_position_embeddings:]
            logits = self.forward(ctx)["logits"][:, -1, :]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def num_params(model: nn.Module, only_trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_trainable))


if __name__ == "__main__":
    cfg = SmolLM2Config()
    m = SmolLM2ForCausalLM(cfg)
    # Tied embeddings: lm_head.weight is the same tensor as embed_tokens.weight,
    # so .parameters() yields it once, giving the true unique param count.
    print(f"Unique params: {num_params(m):,}")
    print(f"Expected:      ~134,515,008  (135M-branded)")
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(x, labels=x)
    print(f"logits: {tuple(out['logits'].shape)},  loss: {out['loss'].item():.3f}")
