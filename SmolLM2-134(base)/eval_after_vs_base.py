"""
Honest accounting of "did continued pretraining make the model better?"

In domain (TinyStories) it clearly did. But continued pretraining at LR 3e-4
with no replay almost certainly degrades general-text performance. We measure
*both* sides so the answer is verifiable, not asserted.

Outputs results/tinystories_vs_base.md and ...json.
"""
from __future__ import annotations

import json
import math
import pathlib
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_full import SmolLM2ForCausalLM, SmolLM2Config
from verify import REPO, load_official_weights_into_ours

RESULTS = pathlib.Path("results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(REPO)

# --- load both models ----------------------------------------------------
print("Loading BASE (official) model...")
hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
base = SmolLM2ForCausalLM(SmolLM2Config())
load_official_weights_into_ours(base, hf.state_dict())
base = base.to(device=device, dtype=dtype).eval()
del hf

print("Loading TRAINED (TinyStories continued-pretrained) model...")
trained = SmolLM2ForCausalLM(SmolLM2Config())
ckpt = torch.load("checkpoint_tinystories.pt", map_location="cpu", weights_only=False)
trained.load_state_dict(ckpt["model"])
trained = trained.to(device=device, dtype=dtype).eval()


# --- perplexity helper ---------------------------------------------------
@torch.no_grad()
def ppl(model, text, seq=1024, stride=512, max_windows=200):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    nlls, n = [], 0
    for i, begin in enumerate(range(0, len(ids) - seq, stride)):
        if i >= max_windows: break
        chunk = ids[begin:begin+seq].unsqueeze(0).to(device)
        logits = model(chunk)["logits"][..., :-1, :].float()
        labels = chunk[..., 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), reduction="sum")
        nlls.append(loss.item()); n += labels.numel()
    return math.exp(sum(nlls)/n), n


# --- 1. TinyStories validation (the domain we trained on) ----------------
print("\n[1/3] TinyStories validation PPL...")
ts_val = load_dataset("roneneldan/TinyStories", split="validation")
ts_text = "\n\n".join(ex["text"] for ex in ts_val if ex["text"].strip())
ts_text = ts_text[:200_000 * 5]
t = time.time(); base_ts, n_ts = ppl(base, ts_text); print(f"   base    {base_ts:.4f}  ({n_ts:,} tok, {time.time()-t:.1f}s)")
t = time.time(); train_ts, _    = ppl(trained, ts_text); print(f"   trained {train_ts:.4f}  ({time.time()-t:.1f}s)")

# --- 2. wikitext-2 validation (general text, was 15.371 for both before) -
print("\n[2/3] wikitext-2-raw-v1 validation PPL  (general text, OOD now)...")
wk = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
wk_text = "\n\n".join(ex["text"] for ex in wk if ex["text"].strip())
t = time.time(); base_wk, n_wk = ppl(base, wk_text); print(f"   base    {base_wk:.4f}  ({n_wk:,} tok, {time.time()-t:.1f}s)")
t = time.time(); train_wk, _    = ppl(trained, wk_text); print(f"   trained {train_wk:.4f}  ({time.time()-t:.1f}s)")

# --- 3. Code (also OOD) --------------------------------------------------
print("\n[3/3] Python-code PPL (codeparrot/github-code-clean Python sample)...")
try:
    code_ds = load_dataset("codeparrot/github-code-clean", "Python-mit",
                           split="train", streaming=True)
    code_text = ""
    for ex in code_ds:
        code_text += ex["code"] + "\n\n"
        if len(code_text) > 500_000: break
except Exception as e:
    # Fallback: use any python-ish corpus from disk
    print(f"   (couldn't load codeparrot ds: {e}; falling back to model.py × 30)")
    code_text = open("model.py").read() * 30

t = time.time(); base_code, n_code = ppl(base, code_text); print(f"   base    {base_code:.4f}  ({n_code:,} tok, {time.time()-t:.1f}s)")
t = time.time(); train_code, _    = ppl(trained, code_text); print(f"   trained {train_code:.4f}  ({time.time()-t:.1f}s)")


# --- 4. Generation comparison on non-story prompts -----------------------
@torch.no_grad()
def gen(m, prompt, n=60, t=0.7, k=40):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = m.generate(ids, max_new_tokens=n, temperature=t, top_k=k)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("\nGeneration on NON-STORY prompts (does the trained model still know things?)")
ood_prompts = [
    "def fibonacci(n):",
    "The Pythagorean theorem states",
    "The capital of France is",
    "In Python, a list comprehension is",
]
ood_results = []
torch.manual_seed(42)
for p in ood_prompts:
    b = gen(base, p)
    torch.manual_seed(42)
    t = gen(trained, p)
    ood_results.append({"prompt": p, "base": b, "trained": t})
    print(f"\n  -- {p!r}")
    print(f"  BASE   : {b[:200]}")
    print(f"  TRAINED: {t[:200]}")


# --- Save results --------------------------------------------------------
summary = {
    "perplexity": {
        "tinystories_val": {"base": base_ts, "trained": train_ts,
                             "delta_pct": 100*(train_ts-base_ts)/base_ts,
                             "n_tokens": n_ts},
        "wikitext2_val":   {"base": base_wk, "trained": train_wk,
                             "delta_pct": 100*(train_wk-base_wk)/base_wk,
                             "n_tokens": n_wk},
        "python_code":     {"base": base_code, "trained": train_code,
                             "delta_pct": 100*(train_code-base_code)/base_code,
                             "n_tokens": n_code},
    },
    "generation": ood_results,
}
with open(RESULTS / "tinystories_vs_base.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}\nFinal verdict:")
print(f"  TinyStories (in-domain):  {base_ts:.2f} -> {train_ts:.2f}  "
      f"({100*(train_ts-base_ts)/base_ts:+.1f}%)")
print(f"  wikitext-2 (general):     {base_wk:.2f} -> {train_wk:.2f}  "
      f"({100*(train_wk-base_wk)/base_wk:+.1f}%)")
print(f"  Python code (OOD):        {base_code:.2f} -> {train_code:.2f}  "
      f"({100*(train_code-base_code)/base_code:+.1f}%)")
print(f"\nSaved {RESULTS}/tinystories_vs_base.json")
