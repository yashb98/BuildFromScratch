"""
Architecture parity test.

Loads the official SmolLM2-135M safetensors into our SmolLM2ForCausalLM, runs
the same input through both our model and HF's LlamaForCausalLM, and asserts
the logits match to bf16 numerical tolerance.

This is the "architecture is correct" gate per the project rules: do not move
to training until this passes.

    python verify.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_full import SmolLM2ForCausalLM, SmolLM2Config


REPO = "HuggingFaceTB/SmolLM2-135M"


def load_official_weights_into_ours(ours: SmolLM2ForCausalLM, hf_state_dict: dict):
    """
    HF Llama state dict keys (sampled):
        model.embed_tokens.weight
        model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        model.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight                  # absent when tie_word_embeddings=True

    Our module names mirror these one-for-one, so load_state_dict is direct.
    The only caveat is the tied lm_head: when HF saves with tie_word_embeddings=True,
    `lm_head.weight` is omitted from the state dict because it aliases embed_tokens.
    In our module we do the same aliasing in __init__, so missing-key strictness
    must allow lm_head.weight to be absent.
    """
    # Filter strict=False to ignore the absent lm_head.weight (it's tied).
    missing, unexpected = ours.load_state_dict(hf_state_dict, strict=False)
    # We expect lm_head.weight to be "missing" (tied), and nothing unexpected.
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"Unexpected missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")


@torch.no_grad()
def main():
    print(f"Loading official {REPO} ...")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf_model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32)
    hf_model.eval()

    print("Building our model and copying weights ...")
    ours = SmolLM2ForCausalLM(SmolLM2Config())
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

    # Tolerance: with fp32 weights and fp32 math we should see ~1e-5 numerical
    # noise from non-associative float reductions. Anything > 1e-3 likely means
    # an architectural mismatch (most often RoPE convention or norm placement).
    assert max_abs < 1e-3, f"Outputs diverge: {max_abs}. Architecture mismatch."

    # Also confirm next-token argmax agrees — the loosest possible check.
    hf_next = hf_out[0, -1].argmax().item()
    our_next = our_out[0, -1].argmax().item()
    print(f"HF next token : {tokenizer.decode([hf_next])!r}")
    print(f"Ours next     : {tokenizer.decode([our_next])!r}")
    assert hf_next == our_next, "Next-token disagreement"

    print("\n✓ Architecture parity verified.")


if __name__ == "__main__":
    main()
