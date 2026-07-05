#!/usr/bin/env python3
"""RLVR Phase-1 go/no-go (research/rlvr/plan.md §3 Phase 1): measure the SFT'd
checkpoint's held-out math exact-match pass@1 + pass@k BEFORE any GRPO spend.
If pass@k ~= 0 in the eval band -> STOP: RL is null-by-construction (nothing to
sharpen); write the honest null. Only a nonzero pass@k unlocks P1 (prompt-set
prep) and the Dr.GRPO exploratory arm.

Pre-registered decision rule (this file, before results existed):
  GO   if the SFT checkpoint solves >= 3 items (any set) at pass@8 with n=8
       samples over the 150-item band (>=2% solved) — enough groups with
       nonzero advantage for GRPO gradients to exist.
  STOP otherwise (pass@k ~= 0 band-wide) -> record the null, keep GRPO parked.

Method notes (correctness over speed, §C21 spirit):
  * Scorer = research/eval_math_acc.py (`math-acc-v1`, 34 unit tests): boxed/####
    extraction, SymPy equivalence, Wilson pass@1, Chen-2021 pass@k.
  * Eval band = seeded subsample of math-eval-v1 *_clean.jsonl (vs-SFT decontam
    0-flagged 2026-07-01): 100 GSM8K-test + 50 MATH-500 level<=3.
  * Generation: NO KV cache exists in the canonical model -> exact full-prefix
    re-forward per token, n=8 same-prompt rows batched (identical lengths, no
    padding needed with is_causal SDPA). T=0.8 multinomial, max_new=256, seeded
    per item. Slow (~55 s/item) but bit-honest. vLLM unavailable on GB10 sm_121.
  * Checkpoints: sft_seed0 (the would-be GRPO policy init) + base (elicitation
    reference). Seed spread on SFT PPL was ~0.001, so seed0 represents the arm
    for a go/no-go; all 3 seeds only if GO.

Eval-only, no training. safe_cuda-guarded; one GPU job (§C4.5); preflight first.
--smoke: 2 items x 2 samples x 16 tokens, sft_seed0 only.
"""
from __future__ import annotations
import json, pathlib, random, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                  # BuildFromScratch/
MD = ROOT / "Qwen3-0.6B"
MODERN = next(MD.glob("builds/*_reproduce-modernized_*"))
for p in (str(ROOT), str(ROOT / "research"), str(MD), str(MODERN)):
    sys.path.insert(0, p)
import safe_cuda  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import model_imu1 as M  # noqa: E402  (flags-off == faithful model.py)
sys.path.insert(0, str(ROOT))
from research.eval_math_acc import run_math_acc, EXTRACTOR_VERSION  # noqa: E402

DATA = ROOT / "research" / "datasets" / "math-eval-v1"
SFT_DIR = MD / "experiments" / "2026-06-27_qwen3-0.6b_sft-3seed"
BASE_CKPT = (MD / "builds" / "2026-06-08_reproduce-faithful_qwen3-0.6b" /
             "checkpoint_qwen3_baseline2tpp.pt")
N_GSM8K, N_MATH, SUBSAMPLE_SEED = 100, 50, 20260701
N_SAMPLES, K_LIST, TEMP, MAX_NEW = 8, (1, 8), 0.8, 256
GO_MIN_SOLVED = 3                                        # pre-registered


def load_band(smoke=False):
    rng = random.Random(SUBSAMPLE_SEED)
    gsm = [json.loads(l) for l in (DATA / "gsm8k_test_clean.jsonl").open()]
    math = [json.loads(l) for l in (DATA / "math500_clean.jsonl").open()]
    math = [r for r in math if isinstance(r.get("level"), int) and r["level"] <= 3]
    gsm_pick = rng.sample(gsm, N_GSM8K)
    math_pick = rng.sample(math, N_MATH)
    if smoke:
        gsm_pick, math_pick = gsm_pick[:1], math_pick[:1]
    return {"gsm8k": gsm_pick, "math500_l13": math_pick}


def load_model(ckpt_path):
    cfg = M.Qwen3Config(use_value_residual=False, use_layernorm_scaling=False,
                        use_head_gating=False)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model = M.Qwen3ForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    model.load_state_dict(sd, strict=True)
    return model


def make_generate(model, tok, max_new, item_counter):
    eos = tok.eos_token_id

    @torch.no_grad()
    def generate_fn(prompt, n):
        # n same-prompt rows -> identical lengths, no padding; exact re-forward per step
        torch.manual_seed(SUBSAMPLE_SEED + item_counter[0]); item_counter[0] += 1
        ids = tok.encode(prompt, add_special_tokens=False)
        x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0).repeat(n, 1)
        done = torch.zeros(n, dtype=torch.bool, device="cuda")
        for _ in range(max_new):
            logits = model(input_ids=x)["logits"][:, -1, :].float()
            probs = F.softmax(logits / TEMP, dim=-1)
            nxt = torch.multinomial(probs, 1)
            if eos is not None:
                nxt[done] = eos
                done |= nxt.squeeze(1) == eos
            x = torch.cat([x, nxt], dim=1)
            if bool(done.all()):
                break
        outs = []
        for row in x[:, len(ids):].tolist():
            if eos is not None and eos in row:
                row = row[:row.index(eos)]
            outs.append(tok.decode(row))
        return outs
    return generate_fn


def main():
    # --ckpt PATH --label NAME [--out FILE]: score ONE arbitrary checkpoint on the SAME
    # paired band (identical items + per-item generation seeds) — used by the Phase-2
    # cohort to compare each arm against the Phase-1 sft_seed0/base floors.
    smoke = "--smoke" in sys.argv
    argv = sys.argv[1:]
    def _opt(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None
    safe_cuda.guard(0.85)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    band = load_band(smoke)
    n_samples = 2 if smoke else N_SAMPLES
    max_new = 16 if smoke else MAX_NEW
    if _opt("--ckpt"):
        ckpts = {(_opt("--label") or "custom"): pathlib.Path(_opt("--ckpt"))}
    else:
        ckpts = {"sft_seed0": SFT_DIR / "checkpoint_sft_seed0.pt"}
        if not smoke:
            ckpts["base"] = BASE_CKPT
    results = {"extractor": EXTRACTOR_VERSION, "band_seed": SUBSAMPLE_SEED,
               "temp": TEMP, "n_samples": n_samples, "max_new_tokens": max_new,
               "go_rule": f"GO iff SFT solved_items >= {GO_MIN_SOLVED} across the band (pre-registered)",
               "checkpoints": {}}
    for name, path in ckpts.items():
        print(f"== {name} ({path.name})", flush=True)
        model = load_model(path)
        counter = [0]
        gen = make_generate(model, tok, max_new, counter)
        per_set = {}
        for set_name, items in band.items():
            t0 = time.time()
            r = run_math_acc(gen, items, n_samples=n_samples, k_list=K_LIST)
            r["wall_s"] = round(time.time() - t0)
            per_set[set_name] = r
            print(f"   {set_name}: pass@1={r['pass1_wilson_ci']['acc']:.4f} "
                  f"CI[{r['pass1_wilson_ci']['ci_low']:.4f},{r['pass1_wilson_ci']['ci_high']:.4f}] "
                  f"pass@k={r['passk_chen2021']} solved={r['solved_items']}/{r['n_items']} "
                  f"({r['wall_s']}s)", flush=True)
        results["checkpoints"][name] = per_set
        del model; torch.cuda.empty_cache()
        out_name = (_opt("--out") or ("phase1_smoke.json" if smoke else
                    "phase1_passk.json" if not _opt("--ckpt") else
                    f"passk_{next(iter(ckpts))}.json"))
        (HERE / out_name).write_text(json.dumps(results, indent=2))
    if not smoke and "sft_seed0" in results["checkpoints"]:
        sft = results["checkpoints"]["sft_seed0"]
        solved = sum(s["solved_items"] for s in sft.values())
        results["decision"] = "GO" if solved >= GO_MIN_SOLVED else "STOP"
        results["solved_total_sft"] = solved
        (HERE / "phase1_passk.json").write_text(json.dumps(results, indent=2))
        print(f"\nDECISION: {results['decision']} (SFT solved {solved} items; "
              f"rule: GO iff >= {GO_MIN_SOLVED})", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
