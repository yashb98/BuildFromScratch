#!/usr/bin/env python3
"""HEADROOM PROBE for the codeharness target — the decisive cheap experiment before
committing GPU to a full harness search.

Question: does HARNESS quality move the held-out pass rate at all on this benchmark
with a REAL frozen model? If the hand-designed baseline and a deliberately-naive
harness both score ~1.0 (or both ~0.0), there is no signal to search — codeharness
is the packing trap again, and the honest move is harder tasks / a weaker model. If
the naive harness scores well BELOW baseline, extraction/prompt headroom is real and
a search (+ traces) is worth running.

Frozen model = Qwen3.5-9B (HF cache), greedy/deterministic so the whole probe is
reproducible. The runner is the FIXED (reward-hack-closed) oracle. GB10-only, no
rented compute; safe_cuda guards the unified-memory pool.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))                 # safe_cuda at repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))  # benchmark, runner, baseline_harness

import safe_cuda                              # noqa: E402
safe_cuda.guard(0.85)

import torch                                  # noqa: E402
from transformers import AutoTokenizer        # noqa: E402
import benchmark as bm                        # noqa: E402
import runner as rn                           # noqa: E402
import baseline_harness as bh                 # noqa: E402

MODEL = "Qwen/Qwen3.5-9B"
MAX_NEW = 256


def _load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = None
    errs = []
    from transformers import AutoModelForCausalLM
    for loader_name, loader in [("AutoModelForCausalLM", AutoModelForCausalLM)]:
        try:
            model = loader.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
            print(f"loaded via {loader_name}", flush=True)
            break
        except Exception as e:
            errs.append(f"{loader_name}: {type(e).__name__}: {e}")
    if model is None:
        # VLM fallback: load the text-to-text / image-text-to-text head, use text path only
        for loader_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            try:
                import transformers
                loader = getattr(transformers, loader_name)
                model = loader.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
                print(f"loaded via {loader_name}", flush=True)
                break
            except Exception as e:
                errs.append(f"{loader_name}: {type(e).__name__}: {e}")
    if model is None:
        raise RuntimeError("could not load model:\n" + "\n".join(errs))
    model.eval()
    return tok, model


def make_agent_fn(tok, model):
    @torch.no_grad()
    def agent_fn(prompt):
        # the harness already built the full instruction; wrap as a single user turn.
        # enable_thinking=False — else Qwen3.5's reasoning block eats the whole token
        # budget before any code is emitted (the diagnostic showed this = a false 0/10).
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = out[0][enc["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)
    return agent_fn


def main():
    t0 = time.time()
    tok, model = _load()
    agent_fn = make_agent_fn(tok, model)
    print(f"model ready in {time.time()-t0:.0f}s\n", flush=True)

    baseline_solve = bh.make_solve(agent_fn, bh.INCUMBENT)       # build_prompt + strip_fences
    # deliberately-naive harness: same prompt, NO fence stripping (raw model text as code)
    def naive_solve(task):
        return agent_fn(bh.build_prompt(task, bh.INCUMBENT))

    rows = []
    for t in bm.TASKS:
        raw = agent_fn(bh.build_prompt(t, bh.INCUMBENT))
        base_code = bh.strip_fences(raw)
        base_pass = rn.run_solution(base_code, t)
        naive_pass = rn.run_solution(raw, t)
        fenced = raw.lstrip().startswith("```")
        rows.append((t["id"], base_pass, naive_pass, fenced, raw))
        print(f"  {t['id']:14} baseline={'P' if base_pass else 'F'} "
              f"naive={'P' if naive_pass else 'F'} fenced={fenced}", flush=True)

    held = [r for r in rows if r[0] in {x['id'] for x in bm.HELDOUT_TASKS}]
    def rate(rs, i): return sum(1 for r in rs if r[i]) / len(rs)
    print("\n--- HEAD-ROOM ---", flush=True)
    print(f"ALL  (n={len(rows)}): baseline={rate(rows,1):.2f}  naive={rate(rows,2):.2f}", flush=True)
    print(f"HELD (n={len(held)}): baseline={rate(held,1):.2f}  naive={rate(held,2):.2f}", flush=True)
    print(f"fenced outputs: {sum(1 for r in rows if r[3])}/{len(rows)}", flush=True)
    spread = rate(rows, 1) - rate(rows, 2)
    print(f"\nHEADROOM (baseline - naive) = {spread:+.2f} -> "
          + ("REAL extraction headroom; a search can find it." if spread >= 0.2
             else "thin; baseline≈naive — likely the easy-task trap (need harder tasks / weaker model)."),
          flush=True)
    # show one fenced raw output so the failure mode is concrete
    for rid, bp, npass, fenced, raw in rows:
        if fenced:
            print(f"\nexample raw output [{rid}] (first 240 chars):\n{raw[:240]!r}", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
