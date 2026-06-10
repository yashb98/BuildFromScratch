"""
Qwen3-0.6B — EXPLORATORY build (Build 3): partial RoPE.

Derived from the verified faithful model.py; the ONLY change is partial RoPE
(rotate `partial_rotary_factor` of head_dim, pass the rest through — see the
Qwen3Config flag and _apply_rope below). With factor=1.0 it is bit-identical to
the faithful build (verified in verify_partialrope.py). Backing paper:
"Fractional Rotation, Full Potential?" (arXiv:2603.11611).

Architecture is Qwen3ForCausalLM as shipped in:
    https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json

Every architectural choice is justified inline with the canonical source
(config field, paper section, or HF transformers reference).

Parameter naming mirrors transformers.models.qwen3 exactly, so the official
Qwen3-0.6B-Base safetensors load into this module via `load_state_dict` with no
key remapping. See verify.py for the parity test.

Differences from the SmolLM2/Llama family this codebase already reproduces:
    1. QK-Norm: per-head RMSNorm on Q and K before RoPE (Qwen3-specific).
    2. head_dim is an independent config field (128 here), NOT hidden/n_heads
       (1024/16 = 64). Q/K/V projections size off head_dim, not hidden.
    3. RoPE theta = 1e6 (vs 1e4 Llama, 1e5 SmolLM2-v2).
    4. RMSNorm eps = 1e-6 (vs 1e-5 SmolLM2).
    5. Initializer std = 0.02 fixed (vs 1/sqrt(hidden) SmolLM2).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config — every default matches config.json at the Qwen3-0.6B-Base repo HEAD
# (pulled via AutoConfig.from_pretrained on 2026-06-08).
# ---------------------------------------------------------------------------
@dataclass
class Qwen3Config:
    vocab_size: int = 151_936                     # config.json: vocab_size
    hidden_size: int = 1024                       # config.json: hidden_size
    intermediate_size: int = 3072                 # config.json: intermediate_size
    num_hidden_layers: int = 28                   # config.json: num_hidden_layers
    num_attention_heads: int = 16                 # config.json: num_attention_heads
    num_key_value_heads: int = 8                  # config.json: num_key_value_heads (GQA 16/8 = 2:1)
    head_dim: int = 128                           # config.json: head_dim — INDEPENDENT field, not hidden/n_heads
    max_position_embeddings: int = 40_960         # config.json: max_position_embeddings
    rope_theta: float = 1_000_000.0               # config.json: rope_theta
    # EXPLORATORY SWAP (Build 3): partial RoPE — rotate only this fraction of head_dim,
    # pass the rest through un-rotated. factor=1.0 == faithful baseline (full RoPE).
    # Backed by arXiv 2603.11611 "Fractional Rotation, Full Potential?" (Mar 2026):
    # ~10% rotation reaches convergence comparable to full RoPE at 135M scale.
    partial_rotary_factor: float = 1.0            # 1.0=baseline; sweep 0.25, ~0.10
    rms_norm_eps: float = 1e-6                    # config.json: rms_norm_eps
    initializer_range: float = 0.02               # config.json: initializer_range
    tie_word_embeddings: bool = True              # config.json: tie_word_embeddings
    attention_bias: bool = False                  # config.json: attention_bias
    attention_dropout: float = 0.0                # config.json: attention_dropout
    # hidden_act = "silu" → SwiGLU(silu(gate) * up). Hardcoded in MLP below.


# ---------------------------------------------------------------------------
# RMSNorm — Zhang & Sennrich 2019; standard Llama-family pre-norm.
# fp32 internals match HF Qwen3RMSNorm for numerical stability.
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
# RoPE — Su et al. 2021. rotate_half layout (HF / Llama convention),
# NOT GPT-NeoX interleaved. theta = 1e6.
#
# PARTIAL RoPE (Build 3): the cache is built over `rotary_dim` (= head_dim * factor)
# dimensions, following the standard HF partial_rotary_factor convention used by
# GPT-NeoX/Phi/StableLM — inv_freq divides by rotary_dim, NOT head_dim, so the
# frequency band spans the same range over fewer dims. When rotary_dim == head_dim
# this is bit-identical to the faithful full-RoPE cache.
# ---------------------------------------------------------------------------
def _build_rope_cache(seq_len: int, rotary_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)             # (seq_len, rotary_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)      # (seq_len, rotary_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q/k: (B, n_heads, T, head_dim). cos/sin: (T, rotary_dim) → broadcast to (1,1,T,rd).
    # Partial RoPE: rotate the first `rotary_dim` dims, pass the rest through unchanged.
    # rotary_dim == head_dim → q_pass is empty and this equals full RoPE bit-for-bit.
    rd = cos.shape[-1]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_out = torch.cat([q_rot, q_pass], dim=-1)
    k_out = torch.cat([k_rot, k_pass], dim=-1)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Qwen3 Grouped-Query Attention with QK-Norm.
#
# Order (mirrors transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward):
#   1. project x → q/k/v
#   2. view to (B, T, n_*_heads, head_dim)
#   3. q_norm / k_norm — RMSNorm over the head_dim, weight broadcast across heads
#   4. transpose to (B, n_*_heads, T, head_dim)
#   5. apply RoPE to q and k (after QK-Norm — Qwen3 specific)
#   6. repeat_interleave KV to match Q heads (GQA)
#   7. SDPA (causal) → o_proj
#
# QK-Norm is the #1 difference vs Llama/Qwen2/SmolLM2. Each of q_norm and
# k_norm is a single RMSNorm with weight shape (head_dim,), shared across heads.
# Reference: HF Qwen3Attention.__init__:
#     self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
#     self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads          # 16
        self.n_kv_heads = cfg.num_key_value_heads       # 8
        self.head_dim = cfg.head_dim                    # 128 (independent of hidden/n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads    # 2

        # Projection sizes (note: q/o use n_heads*head_dim = 2048; k/v use n_kv_heads*head_dim = 1024):
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)

        # QK-Norm: RMSNorm over head_dim, applied to Q and K before RoPE.
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        # Project, then view to (B, T, n_*_heads, head_dim) — QK-Norm operates here.
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # QK-Norm before transpose (matches HF: norm.weight broadcasts across head dim).
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, n_*_heads, T, head_dim) for RoPE + SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE applied to Q and K (not V), AFTER QK-Norm.
        q, k = _apply_rope(q, k, cos, sin)

        # GQA: repeat KV heads to match Q heads.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None),
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU FFN — Shazeer 2020. Identical to the Llama/SmolLM2 form.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer block — pre-norm, residual around attention and MLP.
# Submodule names (input_layernorm, post_attention_layernorm, self_attn, mlp)
# mirror HF Qwen3DecoderLayer exactly so state dicts load without key remap.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: Qwen3Config):
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
# Full model. HF Qwen3 structure:
#   Qwen3ForCausalLM
#     .model: Qwen3Model
#         .embed_tokens
#         .layers: ModuleList[Qwen3DecoderLayer]
#         .norm
#     .lm_head    # tied to embed_tokens when tie_word_embeddings=True
# ---------------------------------------------------------------------------
class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        # Partial RoPE: rotated-dim count must be even (rotate_half chunks in 2).
        rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
        rotary_dim -= rotary_dim % 2
        assert 2 <= rotary_dim <= cfg.head_dim, f"rotary_dim {rotary_dim} out of range"
        self.rotary_dim = rotary_dim
        cos, sin = _build_rope_cache(cfg.max_position_embeddings, rotary_dim,
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


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            # lm_head.weight aliases embed_tokens.weight (same storage).
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        # HF Qwen3 default: Normal(0, initializer_range=0.02) for Linear and Embedding weights.
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        # RMSNorm.weight initialized to ones in its __init__.

    def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None):
        hidden = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
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
        Defaults mirror SmolLM2's generate() so the harness stays consistent."""
        self.eval()
        for _ in range(max_new_tokens):
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
    cfg = Qwen3Config()
    m = Qwen3ForCausalLM(cfg)
    print(f"Unique params: {num_params(m):,}")
    print(f"Expected:      ~596,049,920  (596M-branded, '0.6B')")
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(x, labels=x)
    print(f"logits: {tuple(out['logits'].shape)},  loss: {out['loss'].item():.3f}")
