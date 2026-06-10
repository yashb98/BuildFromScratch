"""
Architecture parity test for Qwen3-0.6B.

Loads the official Qwen3-0.6B-Base safetensors into our Qwen3ForCausalLM,
runs the same input through both our model and HF's Qwen3ForCausalLM, and
asserts the logits match to fp32 numerical tolerance.

This is the "architecture is correct" gate per the project rules: do not
move to training until this passes.

    python verify.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Qwen3ForCausalLM, Qwen3Config


REPO = "Qwen/Qwen3-0.6B-Base"


def load_official_weights_into_ours(ours: Qwen3ForCausalLM, hf_state_dict: dict):
    """
    HF Qwen3 state dict keys (sampled):
        model.embed_tokens.weight
        model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        model.layers.{i}.self_attn.{q,k}_norm.weight                # NEW vs Llama/Qwen2
        model.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight                                              # absent when tied

    Our submodule names mirror these one-for-one (model.py was written to do
    so), so load_state_dict is direct. Tied lm_head: HF omits lm_head.weight
    from the saved state dict; we do the same aliasing in __init__, so the
    missing key for lm_head.weight is expected and filtered.
    """
    missing, unexpected = ours.load_state_dict(hf_state_dict, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"Unexpected missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")


@torch.no_grad()
def main():
    print(f"Loading official {REPO} ...")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf_model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
    hf_model.eval()

    print("Building our model and copying weights ...")
    ours = Qwen3ForCausalLM(Qwen3Config())
    load_official_weights_into_ours(ours, hf_model.state_dict())
    ours.eval()

    # Same prompt, same dtype, same device.
    text = "The capital of France is"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    hf_out = hf_model(input_ids).logits          # (1, T, V)
    our_out = ours(input_ids)["logits"]

    max_abs = (hf_out - our_out).abs().max().item()
    rel = max_abs / hf_out.abs().max().item()
    print(f"max |Δlogits| = {max_abs:.3e}")
    print(f"relative      = {rel:.3e}")

    # Tolerance: fp32 weights and fp32 math → ~1e-5 numerical noise from
    # non-associative reductions. Anything > 1e-3 likely means an arch
    # mismatch (most often RoPE convention, QK-Norm placement, or head_dim).
    assert max_abs < 1e-3, f"Outputs diverge: {max_abs}. Architecture mismatch."

    # Argmax agreement is the loosest possible reproducibility check.
    hf_next = hf_out[0, -1].argmax().item()
    our_next = our_out[0, -1].argmax().item()
    print(f"HF next token : {tokenizer.decode([hf_next])!r}")
    print(f"Ours next     : {tokenizer.decode([our_next])!r}")
    assert hf_next == our_next, "Next-token disagreement"

    print("\n✓ Architecture parity verified.")


if __name__ == "__main__":
    main()
