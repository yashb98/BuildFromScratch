"""Original-vs-reproduction perplexity comparison.

Measures, with IDENTICAL eval code on the IDENTICAL val slice:
  - the published Qwen/Qwen3-0.6B-Base (trained on 36T tokens) — "the original"
  - our from-scratch sweep checkpoints lr17 / lr24 / lr30 (131M tokens each)

This is the true reproduction gap. val tokens come from the same cache the sweep
trained/evaluated against, so the numbers are directly comparable to the sweep logs.
"""
import sys, pathlib, math, time
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]; MODEL_DIR = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(MODEL_DIR)); sys.path.insert(0, str(HERE))

import safe_cuda
from model import Qwen3Config, Qwen3ForCausalLM
from train_qwen3 import chunked_cross_entropy, REPO

SEQ_LEN, MAX_WINDOWS = 4096, 50
CACHE = HERE / "results" / "tokcache_133072000_300000.pt"
OUT = HERE / "results" / "original_vs_repro.txt"


@torch.no_grad()
def eval_ppl(model, val, device):
    model.eval()
    nlls, n = 0.0, 0
    for begin in range(0, min(len(val) - SEQ_LEN, MAX_WINDOWS * SEQ_LEN), SEQ_LEN):
        ids = val[begin:begin + SEQ_LEN + 1].unsqueeze(0).to(device)
        out = model(input_ids=ids[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        loss = chunked_cross_entropy(logits, ids[:, 1:]) * (ids.size(1) - 1)
        nlls += loss.item(); n += ids.size(1) - 1
    return math.exp(nlls / max(1, n)), n


def main():
    safe_cuda.guard(0.85)
    device = torch.device("cuda")
    val = torch.load(CACHE)["val"]
    lines = [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Original vs reproduction — val={len(val):,} tokens, "
             f"{MAX_WINDOWS} windows x {SEQ_LEN}"]

    # --- the original published model ---
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(device)
    ppl_orig, n = eval_ppl(hf, val, device)
    lines.append(f"ORIGINAL  Qwen3-0.6B-Base (36T tok)   val PPL = {ppl_orig:8.3f}  ({n:,} tok, {time.time()-t0:.0f}s)")
    del hf; torch.cuda.empty_cache()

    # --- our from-scratch sweep checkpoints ---
    repro = {}
    for name in ("lr17", "lr24", "lr30"):
        ckpt = HERE / f"checkpoint_qwen3_{name}.pt"
        if not ckpt.exists():
            lines.append(f"REPRO     {name}                        (checkpoint missing — skipped)")
            continue
        m = Qwen3ForCausalLM(Qwen3Config()).to(device=device, dtype=torch.bfloat16)
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        ppl, _ = eval_ppl(m, val, device)
        repro[name] = ppl
        lines.append(f"REPRO     {name} (131M tok, from scratch) val PPL = {ppl:8.3f}  gap x{ppl/ppl_orig:6.1f}")
        del m; torch.cuda.empty_cache()

    if repro:
        best = min(repro, key=repro.get)
        lines.append(f"\nBest reproduction: {best} = {repro[best]:.3f}  vs original {ppl_orig:.3f}  "
                     f"=> {repro[best]/ppl_orig:.1f}x higher PPL (expected: 36T vs 131M tokens = ~275,000x less data)")

    report = "\n".join(lines)
    print(report)
    OUT.write_text(report + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
