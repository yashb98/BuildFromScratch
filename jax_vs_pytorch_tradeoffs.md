# JAX vs PyTorch for from-scratch builds on this box

Written 2026-06-11 to inform the **next** from-scratch model's framework choice. Context: the
Qwen3-0.6B build went PyTorch by inheritance (mirrored SmolLM2) and was locked in by the
bit-exact verify gate — the JAX request was never captured in any plan doc or code. This file
exists so the next decision is made deliberately, not silently.

## TL;DR

- On a **single GB10 (one GPU, unified memory)**, JAX is **not meaningfully faster** than
  `torch.compile` for a sub-1B transformer. Expect rough parity (±10–20%), not an order of
  magnitude. JAX's real speed edge is **TPUs** and **large-scale SPMD sharding** — neither applies here.
- The real cost of choosing JAX here is **the verify gate**, not throughput.
- JAX carries a **memory-safety tax** on this box (XLA preallocation) that PyTorch does not.

## The decisive factor: the verify gate

The from-scratch-build hard gate is `max_abs_error < 1e-3` against the HuggingFace reference.
The way it's achieved (Phase 5):

```python
hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
load_official_weights_into_ours(hf.state_dict(), our_model)   # copy torch tensors directly
assert max_abs_error(our_logits, hf_logits) < 1e-3            # Qwen3 hit 0.0 (bit-exact)
```

This is **intrinsically a PyTorch operation**. A JAX/Flax model cannot consume an HF torch
`state_dict` directly — you'd transpose/rename params into a JAX pytree, and cross-framework
numerics (different matmul/softmax/RoPE kernels) mean you will **not** reproduce `0.0`.

| | PyTorch | JAX / Flax |
|---|---|---|
| Load HF weights | direct `state_dict` copy | manual pytree port (transpose, rename) |
| Realistic verify tolerance | **bit-exact, ~0.0** | numerical port, **~1e-2** |
| What "verified" means | provably identical to HF | "close enough," judgment call |

For a **reproduction**, this is a genuine downgrade in proof strength. For a **novel design**
(no HF reference exists), there's no bit-exact gate to lose — so JAX costs nothing here, making
novel design the natural place to switch frameworks.

## Speed — honest picture on this hardware

- Current Qwen3 PyTorch build: **~7,300 tok/s** with `torch.compile`, micro_batch=4 @ seq 4096 (measured).
- PyTorch on NVIDIA is heavily tuned: cuDNN, cuBLAS, FlashAttention, fused optimizers.
- JAX/XLA wins on **fusion of many small ops** and on **TPU**; on a single NVIDIA GPU a standard
  transformer lands within ~10–20% either way. No reliable win to expect here.
- `torch.compile` is itself a compiler (TorchInductor→Triton), so the fair comparison is
  compiled-vs-compiled, not JAX-jit-vs-PyTorch-eager.

## Memory safety on the unified-memory GB10

- **PyTorch:** `import safe_cuda; safe_cuda.guard(0.85)` caps the process and turns overflow into a
  catchable `torch.OutOfMemoryError`. (The 2026-06-08 crash was a PyTorch fp32-logits over-alloc.)
- **JAX:** XLA preallocates **~75% of "GPU" memory by default** = ~90 GB grabbed at startup on this
  box → hard crash. Must set `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `MEM_FRACTION=0.5` *before*
  `import jax` — that's what `jax_safe_env.py` enforces. Non-prealloc can also cost some throughput.

## What JAX genuinely buys

- `vmap`/`grad`/`jvp` functional transforms — elegant for research with exotic batching/jacobians.
- Clean SPMD sharding (`shard_map`/`pjit`) — pays off at multi-device/TPU scale (not this box).
- TPU portability if you ever move off the GB10.
- Learning JAX itself / a JAX-native reference implementation.

## Recommendation

- **Reproductions → PyTorch.** The bit-exact gate is the whole point and only PyTorch delivers it.
- **Next model in JAX → do it as a NOVEL design** (or accept ~1e-2 numerical-port verify for a
  reproduction). That honors the JAX request where it costs the least and where JAX's strengths
  (functional transforms, your own from-scratch JAX stack) actually show up.
- Either way: **the framework is now an explicit Phase 1 approval gate** in the from-scratch-build
  skill, so it's never inherited silently again.
