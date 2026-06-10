"""
Phase-5 verify runner — Qwen3-0.6B-Base, fp32, hard gate at max_abs_err < 1e-3.

Loads the official Qwen/Qwen3-0.6B-Base safetensors into our Qwen3ForCausalLM
via load_official_weights_into_ours (from sibling verify.py), runs an identical
prompt through both models, computes max |Δlogits|, and writes the result to
results/verify.json. Asserts the tolerance gate.

    python verify_run.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make Qwen3-0.6B/ (model.py, verify.py) importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import Qwen3Config, Qwen3ForCausalLM                       # noqa: E402
from verify import REPO, load_official_weights_into_ours              # noqa: E402

BUILD_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BUILD_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_JSON = RESULTS_DIR / "verify.json"

TOLERANCE = 1e-3
PROMPT = "The capital of France is"


@torch.no_grad()
def main() -> int:
    t0 = time.time()
    print(f"Loading {REPO} in fp32 (this downloads ~1.2GB on first run) ...")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf_model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
    hf_model.eval()
    print(f"  HF model load: {time.time()-t0:.1f}s")

    t1 = time.time()
    print("Building ours and copying weights ...")
    ours = Qwen3ForCausalLM(Qwen3Config())
    load_official_weights_into_ours(ours, hf_model.state_dict())
    ours.eval()
    print(f"  ours build+copy: {time.time()-t1:.1f}s")

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    t2 = time.time()
    hf_out = hf_model(input_ids).logits
    print(f"  HF forward: {time.time()-t2:.1f}s")

    t3 = time.time()
    our_out = ours(input_ids)["logits"]
    print(f"  ours forward: {time.time()-t3:.1f}s")

    diff = (hf_out - our_out).abs()
    max_abs = diff.max().item()
    rel = max_abs / hf_out.abs().max().item()
    hf_next = hf_out[0, -1].argmax().item()
    our_next = our_out[0, -1].argmax().item()

    result = {
        "repo": REPO,
        "prompt": PROMPT,
        "dtype": "float32",
        "tolerance": TOLERANCE,
        "max_abs_error": max_abs,
        "relative_error": rel,
        "hf_next_token_id": hf_next,
        "our_next_token_id": our_next,
        "hf_next_token_text": tokenizer.decode([hf_next]),
        "our_next_token_text": tokenizer.decode([our_next]),
        "argmax_match": hf_next == our_next,
        "passed": max_abs < TOLERANCE and hf_next == our_next,
        "input_shape": list(input_ids.shape),
        "total_seconds": round(time.time() - t0, 1),
    }

    print("\n=== Verify gate ===")
    print(f"max |Δlogits| = {max_abs:.3e}  (tolerance {TOLERANCE})")
    print(f"relative      = {rel:.3e}")
    print(f"HF next token : {result['hf_next_token_text']!r}  (id {hf_next})")
    print(f"Ours next     : {result['our_next_token_text']!r}  (id {our_next})")
    print(f"argmax match  : {result['argmax_match']}")

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT_JSON}")

    if not result["passed"]:
        print("\n✗ Verify gate FAILED.")
        return 1

    print("\n✓ Verify gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
