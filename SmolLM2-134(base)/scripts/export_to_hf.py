"""
Convert a train.py / train_tinystories.py checkpoint into a HuggingFace
LlamaForCausalLM directory so it can be loaded by lm-evaluation-harness or any
other tool that expects a transformers-style model dir.

Why this is needed: train_tinystories.py saves a plain `torch.save({...})`
dict (model state + recipe + RNG state). lm-eval-harness expects
`from_pretrained(<dir>)`, which means it needs `config.json` + `*.safetensors`
+ tokenizer files.

Usage:
    python scripts/export_to_hf.py <checkpoint.pt> <output_dir>

Example:
    python scripts/export_to_hf.py checkpoint_tinystories.pt hf_export/tinystories
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_full import SmolLM2ForCausalLM, SmolLM2Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to .pt checkpoint")
    ap.add_argument("output_dir", help="directory to write the HF-format model into")
    ap.add_argument("--repo", default="HuggingFaceTB/SmolLM2-135M",
                    help="HF repo to pull the matching LlamaConfig + tokenizer from")
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint {args.checkpoint}...")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ours_sd = ck["model"]

    # Load our model and verify shapes match by round-tripping through SmolLM2ForCausalLM.
    ours = SmolLM2ForCausalLM(SmolLM2Config())
    missing, unexpected = ours.load_state_dict(ours_sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(f"checkpoint key mismatch: missing={missing} unexpected={unexpected}")

    # Build a HF LlamaForCausalLM with the official config + tokenizer, then
    # copy our weights in (the key names already match HF's by construction).
    print(f"Loading {args.repo} config + tokenizer...")
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
    cfg = AutoConfig.from_pretrained(args.repo)
    hf = LlamaForCausalLM(cfg)
    missing, unexpected = hf.load_state_dict(ours_sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(f"HF load mismatch: missing={missing} unexpected={unexpected}")

    print(f"Saving HF-format model to {out}...")
    hf.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.repo)
    tok.save_pretrained(out)

    # Also save the training recipe alongside so the eval is traceable.
    if "training_recipe" in ck:
        import json
        with open(out / "training_recipe.json", "w") as f:
            json.dump({
                "training_recipe": ck["training_recipe"],
                "step": ck.get("step"),
                "tok_seen": ck.get("tok_seen"),
                "baseline_ppl": ck.get("baseline_ppl"),
                "trained_ppl": ck.get("trained_ppl"),
                "source_checkpoint": args.checkpoint,
            }, f, indent=2, default=str)
        print(f"Wrote {out / 'training_recipe.json'}")

    print(f"Done. Load with: AutoModelForCausalLM.from_pretrained({str(out)!r})")


if __name__ == "__main__":
    main()
