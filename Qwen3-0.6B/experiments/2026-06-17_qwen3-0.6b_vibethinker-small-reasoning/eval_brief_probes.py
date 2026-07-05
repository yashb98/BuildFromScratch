"""
eval-harness §C13 brief-specified probes — text-lm-v2 methodology.

The standing suite (eval_suite.py) already ran the v2 Sections 1-3 + the generic
Section-4 forgetting probe on wikitext-2 + code (see eval/suite_results.json).
That generic probe RETAINED but is NOT the corpus the brief names.

This script runs the TWO brief-specified §C13 probes that the prior pass did NOT
run, on the dataset-forge pre-tokenized held-out splits (Qwen3 tokenizer, the
model's OWN — §C10):

  1. §C13 FORGETTING PROBE = FineWeb-Edu held-out PPL (the corpus the base was
     pretrained on; brief line 31/71/83 — "the base's 28.65 number"). vs BASE.
  2. HELD-OUT REASONING PPL = OpenR1-Math held-out split (in-domain gain). vs BASE.

It REUSES the frozen text-lm-v2 window/floor/significance machinery verbatim
(SEQ/STRIDE/MAX_WINDOWS/FLOOR_WINDOWS, 3-disjoint-subsample noise floor on the
BASE) imported from the constructed eval_suite.py in this dir — suite constants
are NOT tuned per run (Hard rule 3). It does NOT re-stamp a new suite_version:
the numbers carry text-lm-v2's methodology; the FineWeb-Edu/OpenR1 corpus ids
are recorded so the §C10 comparability matrix is explicit.

It does NOT judge win/loss (the caller — /ablation-runner Phase 6 — judges).
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent          # the run dir
REPO_ROOT = HERE.parents[2]                              # .../BuildFromScratch
assert (REPO_ROOT / "safe_cuda.py").exists(), f"wrong REPO_ROOT: {REPO_ROOT}"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

# Import the constructed suite module to reuse its frozen constants + machinery
# (safe_cuda header, load_model, ppl_ids, noise_floor, significance) — no
# duplication, no constant drift.
spec = importlib.util.spec_from_file_location("eval_suite_run", HERE / "eval_suite.py")
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)

import torch  # noqa: E402

DATA = REPO_ROOT / "research/datasets/math-reasoning-openr1-math-220k/eval"
FORGET_BIN = DATA / "forgetting_tokens.bin"   # FineWeb-Edu held-out (§C13 probe)
REASON_BIN = DATA / "eval_tokens.bin"         # OpenR1-Math held-out (in-domain)
BASE_FINEWEB_PPL_CLAIM = 28.65                # base's README FineWeb-Edu PPL claim

OUT = HERE / "eval" / "brief_probes_results.json"


def load_ids(path: pathlib.Path) -> torch.Tensor:
    arr = np.fromfile(path, dtype=np.uint32).astype(np.int64)
    return torch.from_numpy(arr)


def main():
    S.safe_cuda.guard(0.85)
    print(f"[brief-probes] suite methodology={S.SUITE_VERSION}; "
          f"SEQ={S.SEQ} STRIDE={S.STRIDE} MAX_WINDOWS={S.MAX_WINDOWS} "
          f"FLOOR_WINDOWS={S.FLOOR_WINDOWS}")

    base = S.load_model(S.BASELINE_CKPT)   # the BASE the SFT started from (strict=True)
    target = S.load_model(S.TARGET_CKPT)   # the SFT checkpoint (strict=True)

    results = {
        "suite_version": S.SUITE_VERSION,
        "methodology_note": (
            "text-lm-v2 window/floor/significance machinery on brief-specified "
            "corpora (FineWeb-Edu forgetting probe + OpenR1-Math reasoning); "
            "these corpora are NOT the frozen Section-1 corpora, so numbers "
            "compare only to other runs scored on the SAME corpus_id + tokenizer."),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": S.OBJECTIVE,
        "tokenizer_repo": S.TOKENIZER_REPO,
        "base_ckpt": S.BASELINE_CKPT,
        "target_ckpt": S.TARGET_CKPT,
        "probes": {},
        "not_run": {},
    }

    probes = [
        ("forgetting_fineweb_edu", "fineweb_edu_sample10bt_heldout", FORGET_BIN,
         "§C13 catastrophic-forgetting probe — FineWeb-Edu (base pretrain corpus)"),
        ("reasoning_openr1", "openr1_math_220k_heldout", REASON_BIN,
         "in-domain held-out reasoning PPL — OpenR1-Math"),
    ]

    for key, corpus_id, binpath, desc in probes:
        try:
            ids = load_ids(binpath)
            t0 = time.time()
            b_ppl, b_n, _, _ = S.ppl_ids(base, ids)
            t_ppl, t_n, _, _ = S.ppl_ids(target, ids)
            # noise floor: 3 disjoint subsamples on the BASE (suite Section 2)
            fl = S.noise_floor(base, ids)
            delta = t_ppl - b_ppl
            sig, label = S.significance(delta, fl["floor_abs"])
            entry = {
                "corpus_id": corpus_id, "desc": desc, "n_tokens": int(t_n),
                "base_ppl": b_ppl, "target_ppl": t_ppl,
                "delta_abs": delta, "delta_pct": 100.0 * delta / b_ppl,
                "floor_abs": fl["floor_abs"], "floor_pct": fl["floor_pct"],
                "subsample_ppls": fl["subsample_ppls"],
                "significant": sig, "significance_label": label,
            }
            # forgetting-specific labels (positive delta beyond floor = forgot)
            if key.startswith("forgetting"):
                entry["forgetting_label"] = (
                    "forgot" if (sig and delta > 0) else "retained")
                entry["base_ppl_claim_readme"] = BASE_FINEWEB_PPL_CLAIM
                entry["base_ppl_measured_vs_claim_delta"] = b_ppl - BASE_FINEWEB_PPL_CLAIM
            results["probes"][key] = entry
            OUT.write_text(json.dumps(results, indent=2))
            print(f"[brief-probes] {key} ({corpus_id}): base={b_ppl:.4f} "
                  f"target={t_ppl:.4f} delta={delta:+.4f} floor={fl['floor_abs']:.4f} "
                  f"-> {label} ({time.time()-t0:.0f}s)")
        except Exception as e:
            results["not_run"][key] = f"not run — {type(e).__name__}: {e}"
            OUT.write_text(json.dumps(results, indent=2))
            print(f"[brief-probes] {key}: NOT RUN — {e}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"BRIEF PROBES COMPLETE -> {OUT}")


if __name__ == "__main__":
    main()
