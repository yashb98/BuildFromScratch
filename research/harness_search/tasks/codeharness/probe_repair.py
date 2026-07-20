#!/usr/bin/env python3
"""HEADROOM CHECK on the HARD tasks: does a TRACE-USING self-repair harness beat a
no-repair (scalar-blind) one with the real frozen model? If yes, execution traces
carry signal here -> a full Meta-Harness search (trace-using vs scalar proposer) is
justified. If no, codeharness on GB10 with this model is a dead end and we stop.

Frozen model = Qwen/Qwen3.5-9B, greedy, enable_thinking=False (else the reasoning
block eats the budget). Grading is the FIXED hack-proof runner on HIDDEN tests.
GB10-only, safe_cuda-guarded.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import safe_cuda                              # noqa: E402
safe_cuda.guard(0.85)

import torch                                  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
import hard_benchmark as hb                   # noqa: E402
import runner as rn                           # noqa: E402
import repair_harness as rh                   # noqa: E402

MODEL = "Qwen/Qwen3.5-9B"
MAX_NEW = 512


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    print(f"model ready in {time.time()-t0:.0f}s\n", flush=True)

    @torch.no_grad()
    def agent_fn(prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    no_repair = rh.make_no_repair_solve(agent_fn)
    self_repair = rh.make_self_repair_solve(agent_fn, max_repairs=2)

    n0 = n1 = 0
    print(f"{'task':16} {'no_repair':>10} {'self_repair':>12}", flush=True)
    for t in hb.TASKS:
        c0 = no_repair(t)
        p0 = rn.run_solution(c0, t)            # HIDDEN grade, hack-proof runner
        c1 = self_repair(t)
        p1 = rn.run_solution(c1, t)
        n0 += p0; n1 += p1
        print(f"  {t['id']:14} {'PASS' if p0 else 'fail':>10} {'PASS' if p1 else 'fail':>12}", flush=True)

    N = len(hb.TASKS)
    print(f"\n--- HARD-TASK HEADROOM (n={N}) ---", flush=True)
    print(f"no_repair   hidden pass rate = {n0/N:.2f}  ({n0}/{N})", flush=True)
    print(f"self_repair hidden pass rate = {n1/N:.2f}  ({n1}/{N})", flush=True)
    spread = (n1 - n0) / N
    print(f"\nTRACE HEADROOM (self_repair - no_repair) = {spread:+.2f} -> "
          + ("REAL: traces lift the pass rate -> a full search is justified."
             if (n1 - n0) >= 1 else
             "none: traces don't move it here. Either the model already maxes the "
             "hard tasks (raise difficulty) or first attempts are unfixable from "
             "public-test traces (dead end on this model)."), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
