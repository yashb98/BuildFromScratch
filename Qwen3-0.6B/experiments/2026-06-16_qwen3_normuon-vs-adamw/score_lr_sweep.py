#!/usr/bin/env python3
"""Score the 3 confirmatory AdamW LR-sweep checkpoints on text-lm-v2 BPB and
compare to the existing 2.4e-3 (3-seed) AdamW point and the NorMuon point."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score_cohort as sc   # reuse load_corpora + score + model load
import torch
from model import Qwen3Config, Qwen3ForCausalLM

RES = pathlib.Path(__file__).resolve().parent / "results"
NEW = {"1.7e-3": "adamw_lr17_seed0", "3.5e-3": "adamw_lr35_seed0", "4.8e-3": "adamw_lr48_seed0"}

sc.safe_cuda.guard(0.85)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
corpora = sc.load_corpora(tok)
out = {}
for lr, tag in NEW.items():
    sd = torch.load(RES / f"checkpoint_{tag}.pt", map_location="cpu", weights_only=False)["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m = Qwen3ForCausalLM(Qwen3Config()).to(device=sc.DEVICE, dtype=sc.DTYPE).eval()
    m.load_state_dict(sd, strict=True)
    wt = sc.score(m, corpora["wikitext2_val"], tok)["bpb"]
    cd = sc.score(m, corpora["code_py"], tok)["bpb"]
    out[lr] = {"wikitext_bpb": wt, "code_bpb": cd}
    del m; torch.cuda.empty_cache()
    print(f"  AdamW lr={lr}: wikitext bpb={wt:.4f}  code bpb={cd:.4f}")
(RES / "lr_sweep_bpb.json").write_text(json.dumps(out, indent=2))
print("\n--- reference (from the cohort) ---")
print("  AdamW lr=2.4e-3 (3 seeds): wikitext bpb=2.1098  <-- our baseline point")
print("  NorMuon       (3 seeds): wikitext bpb=1.6355  <-- the treatment")
