"""
Head-to-head comparison: our SmolLM2ForCausalLM vs HuggingFaceTB/SmolLM2-135M.

Tests:
  1. Final logits parity (max |Δ|, top-k overlap)
  2. Per-layer hidden-state parity (catches subtle layer-internal divergence)
  3. Greedy generation token-by-token agreement
  4. Top-k probability mass agreement on a wide prompt set
  5. Long-context behavior (positional encoding sanity at 512 tokens)
  6. Same-seed sampling determinism

All findings written to results/comparison_with_hf.{md,json}.
"""
from __future__ import annotations

import json
import math
import pathlib
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_full import SmolLM2ForCausalLM, SmolLM2Config
from verify import REPO, load_official_weights_into_ours

RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)


def banner(s):
    print(f"\n{'='*70}\n{s}\n{'='*70}")


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    banner("Loading models")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).to(device).eval()
    ours = SmolLM2ForCausalLM(SmolLM2Config())
    load_official_weights_into_ours(ours, hf.state_dict())
    ours = ours.to(device).eval()
    print("HF + ours loaded on", device)

    findings = {}

    # ──────────────────────────────────────────────────────────────────────
    banner("1. Final-logit parity on short prompt")
    text = "The capital of France is"
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        h_log = hf(ids).logits
        o_log = ours(ids)["logits"]
    delta = (h_log - o_log).abs()
    findings["short_prompt"] = {
        "prompt": text,
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
        "max_rel_delta": float(delta.max() / h_log.abs().max()),
        "argmax_match": int(h_log[0, -1].argmax()) == int(o_log[0, -1].argmax()),
    }
    print(json.dumps(findings["short_prompt"], indent=2))

    # ──────────────────────────────────────────────────────────────────────
    banner("2. Per-layer hidden-state parity")
    # We register hooks on both models' decoder-layer outputs and compare.
    hf_states, our_states = [], []

    def hf_hook(_, __, out):
        # HF LlamaDecoderLayer returns a tuple (hidden, ...)
        hf_states.append(out[0] if isinstance(out, tuple) else out)

    def our_hook(_, __, out):
        our_states.append(out)

    hf_handles = [l.register_forward_hook(hf_hook) for l in hf.model.layers]
    our_handles = [l.register_forward_hook(our_hook) for l in ours.model.layers]
    with torch.no_grad():
        hf(ids)
        ours(ids)
    for h in hf_handles + our_handles:
        h.remove()

    per_layer = []
    for i, (a, b) in enumerate(zip(hf_states, our_states)):
        diff = (a - b).abs()
        per_layer.append({
            "layer": i,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rel": float(diff.max() / a.abs().max()),
        })
    print(f"  Showing first/middle/last layer:")
    for i in [0, 14, 29]:
        print(f"   layer {per_layer[i]['layer']:2d}  max|Δ|={per_layer[i]['max_abs']:.3e}  "
              f"mean|Δ|={per_layer[i]['mean_abs']:.3e}  rel={per_layer[i]['rel']:.3e}")
    findings["per_layer"] = per_layer

    # ──────────────────────────────────────────────────────────────────────
    banner("3. Greedy generation token-by-token (no sampling = deterministic)")
    test_prompts = [
        "The capital of France is",
        "Once upon a time",
        "def fibonacci(n):",
        "In a world where",
        "The mitochondria is",
    ]
    gen_compare = []
    for p in test_prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        N = 24

        # HF greedy
        with torch.no_grad():
            hf_out = hf.generate(ids, max_new_tokens=N, do_sample=False,
                                 temperature=1.0, top_k=None,
                                 pad_token_id=tokenizer.eos_token_id)
        # Ours greedy (temperature=0 triggers argmax in our code)
        with torch.no_grad():
            ours_out = ours.generate(ids, max_new_tokens=N, temperature=0.0, top_k=None)

        hf_tokens = hf_out[0].tolist()
        our_tokens = ours_out[0].tolist()
        match_len = sum(1 for a, b in zip(hf_tokens, our_tokens) if a == b)
        full_match = hf_tokens == our_tokens

        gen_compare.append({
            "prompt": p,
            "hf":   tokenizer.decode(hf_tokens, skip_special_tokens=True),
            "ours": tokenizer.decode(our_tokens, skip_special_tokens=True),
            "tokens_matched": match_len,
            "tokens_total":   len(hf_tokens),
            "exact": full_match,
        })
        print(f"  {p!r}  →  exact={full_match}  matched={match_len}/{len(hf_tokens)}")
        if not full_match:
            for i, (a, b) in enumerate(zip(hf_tokens, our_tokens)):
                if a != b:
                    print(f"      first divergence at index {i}: "
                          f"HF={a!r}/{tokenizer.decode([a])!r}  Ours={b!r}/{tokenizer.decode([b])!r}")
                    break
    findings["greedy_generation"] = gen_compare

    # ──────────────────────────────────────────────────────────────────────
    banner("4. Top-10 probability mass agreement")
    top_k_check = []
    for p in test_prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            h = F.softmax(hf(ids).logits[0, -1].float(), dim=-1)
            o = F.softmax(ours(ids)["logits"][0, -1].float(), dim=-1)
        h_top = torch.topk(h, 10)
        o_top = torch.topk(o, 10)
        hf_set = set(h_top.indices.tolist())
        our_set = set(o_top.indices.tolist())
        overlap = len(hf_set & our_set)
        max_prob_delta = (h_top.values - o_top.values).abs().max().item()
        top_k_check.append({
            "prompt": p,
            "top10_overlap": overlap,
            "max_prob_delta": max_prob_delta,
            "hf_top1_idx": int(h_top.indices[0]),
            "ours_top1_idx": int(o_top.indices[0]),
            "hf_top1_token": tokenizer.decode([int(h_top.indices[0])]),
            "ours_top1_token": tokenizer.decode([int(o_top.indices[0])]),
        })
        print(f"  {p!r:<40}  top10 overlap={overlap}/10  max|Δp|={max_prob_delta:.3e}")
    findings["topk_agreement"] = top_k_check

    # ──────────────────────────────────────────────────────────────────────
    banner("5. Long-context (RoPE sanity at 512 tokens)")
    # Build a 512-token input and compare last-position logits.
    long_text = ("In the field of language modeling, " * 60)[:2000]
    ids = tokenizer(long_text, return_tensors="pt", truncation=True,
                    max_length=512).input_ids.to(device)
    print(f"  input length: {ids.shape[1]} tokens")
    with torch.no_grad():
        h_log = hf(ids).logits[0, -1].float()
        o_log = ours(ids)["logits"][0, -1].float()
    long_delta = (h_log - o_log).abs()
    findings["long_context"] = {
        "input_len": int(ids.shape[1]),
        "max_abs_delta": float(long_delta.max()),
        "argmax_match": int(h_log.argmax()) == int(o_log.argmax()),
        "hf_top1": tokenizer.decode([int(h_log.argmax())]),
        "ours_top1": tokenizer.decode([int(o_log.argmax())]),
    }
    print(json.dumps(findings["long_context"], indent=2))

    # ──────────────────────────────────────────────────────────────────────
    banner("6. Same-seed sampling determinism")
    # Note: HF sampling and ours use different RNG paths, so we don't expect
    # identical samples, but we *do* expect the distributions to be the same.
    # Here we just sample many tokens and check the empirical top-1 frequency
    # matches the analytic top-1 probability.
    p = "The cat sat on the"
    ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        log = ours(ids)["logits"][0, -1].float()
        probs = F.softmax(log, dim=-1)
        top1_prob = float(probs.max())
        top1_id = int(probs.argmax())

    N_SAMPLES = 2000
    torch.manual_seed(42)
    samples = []
    with torch.no_grad():
        for _ in range(N_SAMPLES):
            log = ours(ids)["logits"][0, -1].float()
            samples.append(int(torch.multinomial(F.softmax(log, dim=-1), 1)))
    from collections import Counter
    c = Counter(samples)
    emp_top1_freq = c[top1_id] / N_SAMPLES
    findings["sampling"] = {
        "prompt": p,
        "analytic_top1_prob": top1_prob,
        "empirical_top1_freq": emp_top1_freq,
        "delta": abs(top1_prob - emp_top1_freq),
        "top1_token": tokenizer.decode([top1_id]),
        "samples": N_SAMPLES,
    }
    print(json.dumps(findings["sampling"], indent=2))

    # ──────────────────────────────────────────────────────────────────────
    # Summary table
    banner("SUMMARY")
    print(f"{'Check':<45}  Result")
    print("-" * 70)
    ok_short = findings["short_prompt"]["max_abs_delta"] < 1e-3
    print(f"{'1. Final logits parity (short prompt)':<45}  "
          f"max|Δ|={findings['short_prompt']['max_abs_delta']:.3e}  "
          f"{'✓' if ok_short else '✗'}")
    max_layer = max(l["max_abs"] for l in findings["per_layer"])
    ok_layer = max_layer < 1e-3
    print(f"{'2. Per-layer hidden-state parity':<45}  "
          f"max|Δ|={max_layer:.3e}  {'✓' if ok_layer else '✗'}")
    n_match = sum(1 for g in findings["greedy_generation"] if g["exact"])
    print(f"{'3. Greedy generation full-match':<45}  "
          f"{n_match}/{len(findings['greedy_generation'])} prompts exact")
    perfect_topk = sum(1 for t in findings["topk_agreement"] if t["top10_overlap"] == 10)
    print(f"{'4. Top-10 set overlap (perfect=10/10)':<45}  "
          f"{perfect_topk}/{len(findings['topk_agreement'])} prompts perfect")
    ok_long = findings["long_context"]["max_abs_delta"] < 1e-3 and \
              findings["long_context"]["argmax_match"]
    print(f"{'5. Long-context (512 tok)':<45}  "
          f"max|Δ|={findings['long_context']['max_abs_delta']:.3e}  "
          f"{'✓' if ok_long else '✗'}")
    ok_sample = findings["sampling"]["delta"] < 0.05
    print(f"{'6. Sampling matches analytic top-1 prob':<45}  "
          f"|Δ|={findings['sampling']['delta']:.3f}  "
          f"{'✓' if ok_sample else '✗'}")

    with open(RESULTS / "comparison_with_hf.json", "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\nSaved {RESULTS}/comparison_with_hf.json")


if __name__ == "__main__":
    main()
