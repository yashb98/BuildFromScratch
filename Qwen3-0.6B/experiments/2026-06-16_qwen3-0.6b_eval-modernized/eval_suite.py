"""
eval-harness suite runner — text-lm-v1.

TEMPLATE. Constructed per run by /eval-harness from
.claude/skills/eval-harness/references/eval_suite_template.py into
<MODEL_DIR>/experiments/<RUN_ID>/eval_suite.py  (depth: parents[3] = repo root).

Construction checklist (the skill enforces this):
  1. Fill every value in the RESOLVED-AT-CONSTRUCTION block — no double-brace
     placeholder may survive.
  2. Adjust build_model() to mirror EXACTLY how the build's own train/verify
     script constructs the model (read that script first).
  3. `python -m py_compile` the result; grep with the placeholder pattern from
     SKILL.md Phase 2 step 4 that no placeholder remains. (Outside the
     RESOLVED block this file deliberately contains no literal double braces,
     so a correctly filled copy greps clean; main()'s startup assert is the
     runtime backstop.)

PPL sections are ported from SmolLM2-134(base)/eval_after_vs_base.py per
finance-research-loop contracts §6: copied (that module cannot be imported —
it loads models at import time), the model's OWN tokenizer swapped in, and the
code-corpus fallback pointed at MODEL_FILE by absolute path instead of the
original's hardcoded open("model.py").
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# --- safe_cuda header (research-loop contracts §C1; finance contracts §1) ---
# Constructed copies live at <MODEL_DIR>/experiments/<RUN_ID>/ -> N = 3.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
assert (REPO_ROOT / "safe_cuda.py").exists(), f"wrong parents[N]; got {REPO_ROOT}"
sys.path.insert(0, str(REPO_ROOT))
import safe_cuda  # noqa: E402  (sets expandable_segments before torch sees CUDA)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# === RESOLVED AT CONSTRUCTION ===============================================
SUITE_VERSION = "text-lm-v2"           # must match references/suite.md ACTIVE
MODEL_DIR = r"/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B"           # absolute, e.g. .../Qwen3-0.6B
MODEL_FILE = r"/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/model_imu1.py"         # absolute path to the build's model file
MODEL_CLASS = "Qwen3ForCausalLM"        # e.g. Qwen3ForCausalLM | SmolLM2ForCausalLM
CONFIG_CLASS = "Qwen3Config"      # e.g. Qwen3Config | SmolLM2Config
TOKENIZER_REPO = "Qwen/Qwen3-0.6B-Base"  # REPO from <MODEL_DIR>/verify.py — own tokenizer ONLY
TARGET_CKPT = r"/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/checkpoint_imu1_2tpp_step18000.pt"       # checkpoint under evaluation (absolute)
BASELINE_CKPT = r""   # comparison ckpt (absolute); "" -> self-floor mode.
                                       # For OBJECTIVE=finetune this IS the BASE the FT started from.
OBJECTIVE = "pretrain-ablation"            # finetune | pretrain-ablation (§C13) — gates Section 4
OUT_DIR = r"/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/experiments/2026-06-16_qwen3-0.6b_eval-modernized/eval"               # <MODEL_DIR>/experiments/<RUN_ID>/eval
WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"      # pinned sha from suite.md
CODEPARROT_REV = "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd"  # pinned sha from suite.md (codeparrot/codeparrot-clean-valid)
# ============================================================================

# --- suite constants: text-lm-v1 (suite.md is normative; NEVER tune per run) -
SEQ, STRIDE, MAX_WINDOWS = 1024, 512, 200
FLOOR_WINDOWS = 66                       # per disjoint third, baseline model
CODE_CHARS, FALLBACK_REPEAT = 500_000, 30
GEN_SEED, GEN_TEMP, GEN_TOPK, GEN_MAXNEW = 42, 0.7, 40, 60
PROMPTS = [
    "def fibonacci(n):",
    "The Pythagorean theorem states",
    "The capital of France is",
    "In Python, a list comprehension is",
    "Once upon a time, a small robot",
    "The three branches of the United States government are",
    "To compute the average of a list of numbers, you",
    "Water boils at a temperature of",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# --- model loading ----------------------------------------------------------
def load_model_module():
    mf = pathlib.Path(MODEL_FILE)
    for p in (str(mf.parent), MODEL_DIR):          # build dir first, then root
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("eval_suite_model", mf)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register so dataclass type-resolution works
    spec.loader.exec_module(mod)
    return mod


def load_checkpoint_sd(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    # torch.compile-trained checkpoints carry an _orig_mod. prefix (e.g. the
    # Qwen3 modernized build) — strip per finance-research-loop SKILL.md.
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def build_model(mod):
    # ADJUST AT CONSTRUCTION if the build's own train/verify script constructs
    # its config differently (extra kwargs, config loaded from the ckpt, ...).
    return getattr(mod, MODEL_CLASS)(getattr(mod, CONFIG_CLASS)())


def load_model(ckpt_path: str):
    model = build_model(load_model_module())
    model.load_state_dict(load_checkpoint_sd(ckpt_path), strict=True)  # strict, always
    return model.to(device=DEVICE, dtype=DTYPE).eval()


# --- perplexity (ported from eval_after_vs_base.py, tokenizer-agnostic) ------
@torch.no_grad()
def ppl_ids(model, ids: torch.Tensor, max_windows: int = MAX_WINDOWS, tok=None):
    """Returns (ppl, n_tokens, sum_nll_nats, n_bytes). When `tok` is given, also
    accumulates the UTF-8 byte count of the SAME scored label span so BPB (the
    only cross-tokenizer-comparable metric, §C10) shares a consistent denominator
    with PPL under the overlapping windows (STRIDE<SEQ double-counts tokens AND
    their bytes, so the ratio sum_nll/bytes stays correct)."""
    nlls, n, n_bytes = [], 0, 0
    for i, begin in enumerate(range(0, len(ids) - SEQ, STRIDE)):
        if i >= max_windows:
            break
        chunk = ids[begin:begin + SEQ].unsqueeze(0).to(DEVICE)
        logits = model(chunk)["logits"][..., :-1, :].float()
        labels = chunk[..., 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), reduction="sum")
        nlls.append(loss.item())
        n += labels.numel()
        if tok is not None:
            n_bytes += len(tok.decode(labels[0].tolist()).encode("utf-8"))
    if n == 0:
        raise ValueError(f"corpus too short for SEQ={SEQ}: {len(ids)} tokens")
    sum_nll = sum(nlls)
    return math.exp(sum_nll / n), n, sum_nll, n_bytes


def noise_floor(model, ids: torch.Tensor) -> dict:
    """Finance contracts §6: 3 disjoint subsamples; spread = floor."""
    third = len(ids) // 3
    ppls = [ppl_ids(model, ids[i * third:(i + 1) * third], FLOOR_WINDOWS)[0]
            for i in range(3)]
    floor_abs = max(ppls) - min(ppls)
    return {"subsample_ppls": ppls, "floor_abs": floor_abs,
            "floor_pct": 100.0 * floor_abs / (sum(ppls) / 3)}


def significance(delta_abs: float, floor_abs: float) -> tuple[bool, str]:
    sig = abs(delta_abs) > floor_abs
    return sig, ("significant" if sig else "not significant")


# --- corpora (revisions pinned in suite.md) ----------------------------------
def load_wikitext_text() -> tuple[str, str]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="validation", revision=WIKITEXT_REV)
    return "wikitext2_raw_v1_val", "\n\n".join(
        ex["text"] for ex in ds if ex["text"].strip())


def load_code_text() -> tuple[str, str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train",
                          streaming=True, revision=CODEPARROT_REV)
        buf, total = [], 0
        for ex in ds:
            buf.append(ex["content"])
            total += len(ex["content"]) + 2
            if total > CODE_CHARS:
                break
        return "codeparrot_clean_valid", "\n\n".join(buf)
    except Exception as e:  # fallback corpus_id — NOT comparable to codeparrot
        print(f"[code_py] codeparrot unavailable ({e}); MODEL_FILE x{FALLBACK_REPEAT} fallback")
        return "code_fallback_modelfile", pathlib.Path(MODEL_FILE).read_text() * FALLBACK_REPEAT


# --- generation probes --------------------------------------------------------
@torch.no_grad()
def gen_one(model, tokenizer, prompt: str) -> str:
    torch.manual_seed(GEN_SEED)  # re-seeded before EVERY generation
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = model.generate(ids, max_new_tokens=GEN_MAXNEW,
                         temperature=GEN_TEMP, top_k=GEN_TOPK)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# --- main ---------------------------------------------------------------------
def main():
    for name in ("MODEL_DIR", "MODEL_FILE", "MODEL_CLASS", "CONFIG_CLASS",
                 "TOKENIZER_REPO", "TARGET_CKPT", "OBJECTIVE", "OUT_DIR",
                 "WIKITEXT_REV", "CODEPARROT_REV"):
        # "{" * 2 keeps this file itself free of literal double braces.
        assert "{" * 2 not in globals()[name], f"unresolved placeholder: {name}"
    safe_cuda.guard(0.85)

    out = pathlib.Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "suite_version": SUITE_VERSION,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": OBJECTIVE,
        "model_dir": MODEL_DIR, "tokenizer_repo": TOKENIZER_REPO,
        "target_ckpt": TARGET_CKPT,
        "baseline_ckpt": BASELINE_CKPT or None,
        "self_floor": BASELINE_CKPT == "",
        "ppl": {}, "noise_floor": {}, "generations": [],
        "forgetting": {}, "not_run": {},
    }

    def save():  # incremental — a crash never loses finished sections
        (out / "suite_results.json").write_text(json.dumps(results, indent=2))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    print(f"[suite {SUITE_VERSION}] loading target: {TARGET_CKPT}")
    target = load_model(TARGET_CKPT)
    baseline = load_model(BASELINE_CKPT) if BASELINE_CKPT else None
    floor_model = baseline if baseline is not None else target

    corpora = []
    for section, loader in (("wikitext2_val", load_wikitext_text),
                            ("code_py", load_code_text)):
        try:
            corpus_id, text = loader()
            corpora.append((section, corpus_id, text))
        except Exception as e:
            results["not_run"][section] = f"not run — {type(e).__name__}: {e}"
            save()

    for section, corpus_id, text in corpora:
        try:
            t0 = time.time()
            ids = tokenizer(text, return_tensors="pt").input_ids[0]
            t_ppl, n_tok, _, _ = ppl_ids(target, ids)
            fl = noise_floor(floor_model, ids)
            entry = {"corpus_id": corpus_id, "target": t_ppl, "n_tokens": n_tok}
            if baseline is not None:
                b_ppl, _, _, _ = ppl_ids(baseline, ids)
                delta = t_ppl - b_ppl
                sig, label = significance(delta, fl["floor_abs"])
                entry.update(baseline=b_ppl, delta_abs=delta,
                             delta_pct=100.0 * delta / b_ppl,
                             significant=sig, label=label)
                # Section 4 (suite.md): retention probe, finetune runs only.
                # Reuses t_ppl/b_ppl/fl — no extra forward pass. Here baseline
                # IS the BASE the fine-tune started from. Positive delta on a
                # general-domain held-out corpus = forgetting; sub-floor =
                # not significant (so "forgot" requires sig AND delta > 0).
                if OBJECTIVE == "finetune":
                    forgot = sig and delta > 0
                    results["forgetting"][section] = {
                        "corpus_id": corpus_id, "base_ppl": b_ppl,
                        "target_ppl": t_ppl, "retention_delta_abs": delta,
                        "retention_delta_pct": 100.0 * delta / b_ppl,
                        "floor_abs": fl["floor_abs"], "significant": sig,
                        "label": "forgot" if forgot else "retained"}
            results["ppl"][section] = entry
            results["noise_floor"][section] = fl
            save()
            print(f"[suite] {section} ({corpus_id}): target={t_ppl:.4f} "
                  f"floor_abs={fl['floor_abs']:.4f} ({time.time()-t0:.0f}s)")
        except Exception as e:
            results["not_run"][section] = f"not run — {type(e).__name__}: {e}"
            save()

    # Section 4 run-level summary (suite.md): "forgot" on ANY corpus else
    # "retained"; recorded N/A for non-finetune / self-floor (no distinct base).
    if OBJECTIVE == "finetune" and baseline is not None and results["forgetting"]:
        results["forgetting"]["summary"] = (
            "forgot" if any(c["label"] == "forgot"
                            for c in results["forgetting"].values()
                            if isinstance(c, dict)) else "retained")
    else:
        results["not_run"]["forgetting"] = (
            "not run — N/A (objective != finetune or no base)")
    save()

    try:
        lines = [f"# Generation probes — suite {SUITE_VERSION}", ""]
        for p in PROMPTS:
            row = {"prompt": p, "target": gen_one(target, tokenizer, p)}
            if baseline is not None:
                row["baseline"] = gen_one(baseline, tokenizer, p)
            results["generations"].append(row)
            save()
            lines += [f"## {p!r}", "", f"**target:** {row['target']}", ""]
            if baseline is not None:
                lines += [f"**baseline:** {row['baseline']}", ""]
        (out / "generations.md").write_text("\n".join(lines))
        print(f"[suite] generations: {len(PROMPTS)} prompts done")
    except Exception as e:
        results["not_run"]["generations"] = f"not run — {type(e).__name__}: {e}"
        save()

    save()
    print(f"SUITE COMPLETE -> {out / 'suite_results.json'}")


def smoke():
    """SKILL.md Phase 2.5 (§C5.0-style): cheap pre-flight on the EXACT
    constructed script. Loads the real model + the real tokenizer, computes ONE
    PPL metric on a tiny token slice, and exits 0 — proving load + metric + exit
    BEFORE the full pass burns GPU minutes. Does NOT tune suite constants
    (frozen per version): it only passes a tiny max_windows ARGUMENT to ppl_ids
    and reuses MODEL_FILE (always present) as the corpus, so it needs no network
    and never writes suite_results.json (no result may carry smoke numbers)."""
    safe_cuda.guard(0.85)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    model = load_model(TARGET_CKPT)                       # load path = real path
    text = pathlib.Path(MODEL_FILE).read_text() * FALLBACK_REPEAT
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    ppl, n, _, _ = ppl_ids(model, ids, max_windows=2)        # ONE real metric
    assert math.isfinite(ppl) and n > 0, f"non-finite smoke PPL: {ppl} (n={n})"
    out = model.generate(                                  # exercise generate()
        tokenizer(PROMPTS[0], return_tensors="pt").input_ids.to(DEVICE),
        max_new_tokens=4, temperature=GEN_TEMP, top_k=GEN_TOPK)
    assert out.shape[-1] > 0, "empty generation in smoke"
    print(f"SMOKE OK -> suite={SUITE_VERSION} objective={OBJECTIVE} "
          f"ppl={ppl:.4f} (n={n}, tiny max_windows; NOT a result)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="cheap load+metric+exit check (SKILL.md Phase 2.5)")
    if ap.parse_args().smoke:
        smoke()
    else:
        main()
