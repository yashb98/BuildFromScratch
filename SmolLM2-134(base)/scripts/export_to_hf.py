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
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="dtype to save in. bf16 (default) matches the training dtype and "
                         "the base model's published torch_dtype; fp32 upcasts and doubles the size")
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

    # LlamaForCausalLM is built in fp32, so load_state_dict upcast our bf16
    # tensors. Cast back before saving unless fp32 was explicitly requested --
    # otherwise the export is 2x the size for zero extra information.
    if args.dtype == "bf16":
        hf = hf.to(torch.bfloat16)
    print(f"Saving HF-format model to {out} in {next(hf.parameters()).dtype}...")
    hf.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.repo)
    tok.save_pretrained(out)

    # Save the provenance metadata alongside so the eval is traceable. Not every
    # checkpoint carries a training_recipe (train_tinystories.py does not), so
    # write whichever of these keys the checkpoint actually has.
    import json
    meta = {k: ck[k] for k in ("training_recipe", "step", "tok_seen",
                               "baseline_ppl", "trained_ppl") if k in ck}
    if meta:
        meta["source_checkpoint"] = args.checkpoint
        meta["saved_dtype"] = str(next(hf.parameters()).dtype)
        with open(out / "training_recipe.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"Wrote {out / 'training_recipe.json'} ({', '.join(meta)})")

    print(f"Done. Load with: AutoModelForCausalLM.from_pretrained({str(out)!r})")


if __name__ == "__main__":
    main()
