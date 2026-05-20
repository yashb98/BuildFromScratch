"""
Sample from SmolLM2-135M using our reproduction loaded with official weights.

    python generate.py "Once upon a time"
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_full import SmolLM2ForCausalLM, SmolLM2Config
from verify import load_official_weights_into_ours, REPO


def main(prompt: str, max_new_tokens: int = 64):
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32)

    model = SmolLM2ForCausalLM(SmolLM2Config())
    load_official_weights_into_ours(model, hf.state_dict())
    del hf
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Gravity is"
    main(prompt)
