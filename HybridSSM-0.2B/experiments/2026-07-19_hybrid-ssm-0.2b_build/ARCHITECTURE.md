# HybridSSM-0.2B — architecture design (novel from-scratch build, JAX/Flax)

**Purpose:** study the attention-vs-efficient-mixer composition (brief `hybrid-attention-rethink`,
arXiv 2606.15378) on a single GB10, on this repo's evidence standard. Novel design → **no bit-exact
oracle**; the verify gate is numerical cross-check vs an independent reference at ~1e-2 (§C14/JAX).
Framework **JAX/Flax** (user-confirmed 2026-07-19; installed + verified, `associative_scan` runs).

## Config (base arm)

| field | value | why |
|---|---|---|
| d_model | 768 | ~146M non-embed at L=24 (paper S4 104M < this < S5 477M) |
| n_layers | 24 | 1:1 interleave → 12 full-attention + 12 efficient-mixer |
| layer pattern | `[full, eff, full, eff, …]` | paper's 1:1 main setting (1:3 ≈ same val loss is a later arm) |
| full-attn | GQA n_heads=12, n_kv=4, head_dim=64, RoPE (θ=1e4) | Qwen3-family attention, shrunk |
| efficient mixer | **Mamba-2-style selective SSM** via `associative_scan` (diagonal linear recurrence + input/gate proj) | JAX-native scan; the SWA-128 and GatedDeltaNet variants are ablation arms |
| MLP | SwiGLU, intermediate 2048 | Qwen3 recipe |
| norm | RMSNorm (eps 1e-6), pre-norm | Qwen3 recipe |
| vocab / tokenizer | **151,936 (Qwen3-0.6B-Base)** | reuses the validated text-lm-v2 data + eval pipeline; BPB-comparable to the 596M study. Report NON-EMBED params (paper convention). |
| tied embedding | yes | 117M embed counted once |
| CE | **chunked** over vocab (152k > 64k, §C1) | never materialize (N,152k) logits — the box-crash vector |
| seq_len (pretrain) | **4096** (probe decides; 16K is the paper's, memory-tight here) | emergence-speed + relative-hybrid comparison is visible at 4K; long-context/NoPE finding needs a later ctx-extension arm |
| optimizer | Muon (2D weights) + AdamW (1D/embed) — JAX port of `normuon.py` | paper uses Muon; repo has the PyTorch impl to port + cross-check |
| precision | bf16 compute, fp32 master/optimizer state | Qwen3 recipe |

## The toggles that make it a STUDY (single-variable ablation matrix)

Each is one flag on the base arm; iso-FLOP where the flag changes params (≤5%, §C18); ≥3 seeds; BPB
CIs on wikitext-2 + code (text-lm-v2) + a long-context retrieval probe (RULER-NIAH-style):

1. **mixer type** on the efficient layers: `ssm` (Mamba-2) vs `swa128` (sliding-window attn, w=128) vs
   `full` (all-attention control = the dense baseline, comparable to the 596M study) vs `none` (all-SSM).
2. **attention fraction / placement**: 1:1 vs 1:3 (one full-attn per three efficient) — the "how much
   attention does a hybrid need" curve.
3. **NoPE-on-full-attention** (the paper's headline design knob): RoPE vs NoPE on the full-attn layers of
   the SWA-128 hybrid — predicted long-context gain, ~zero short-context cost.
4. **token-budget ladder** (the emergence-speed instrument, mirroring the scaling-persistence study): score
   each hybrid at increasing budgets → does the efficient-mixer choice affect emergence SPEED but converge?

Headline object = the **emergence-speed curve** (quality vs tokens per hybrid) + a box-scale validation of
NoPE-on-full-attn. This is the same *shape* as "the disappearing win" — a coherent next chapter.

## Files (novel design → its own folder, not canonical)

- `model.py` — the JAX/Flax hybrid model (implemented from blank).
- `ssm.py` — the selective-SSM mixer (associative_scan) + the SWA mixer.
- `muon_jax.py` — JAX Muon (ported + cross-checked vs `normuon.py`).
- `verify.py` — numerical cross-check vs an independent reference (~1e-2) + shape/dtype grid.
- `train.py` — training loop (chunked CE, safe guards, ckpt/resume for the recovery chain).

## Guards (§C1)

`import jax_safe_env` BEFORE `import jax` (PREALLOCATE=false, MEM_FRACTION=0.5). Chunked CE for the 152k
vocab. sentinel preflight before any GPU work; sentinel watch + the hardened `boot_resume.sh` recovery
chain + thermal kill beside any unattended trainer.
