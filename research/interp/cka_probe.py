"""Representational-convergence probe (linear CKA) for the NorMuon-vs-AdamW ladder.

Executes research/interp/prereg_2026-07-19_repconvergence.md VERBATIM: does NorMuon reach
AdamW's 420M residual-stream representation earlier? Pre-registered H1 / null + decision rule
are fixed there; this script only measures and applies §4.

SAFETY (§C1): imports safe_cuda before torch; drives the TRUNK ONLY (`model.model(...)`, never
the lm_head) so the 151,936-vocab logits are never materialized (the documented box-crash
vector). Residual activations are fp16, capped, CPU-cached; one 596M model on GPU at a time.

Usage:
  python3 research/interp/cka_probe.py --self-test     # CPU-only: CKA math + decision-rule sanity
  python3 research/interp/cka_probe.py                  # full pre-registered probe (needs GPU + preflight)
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path("/home/yashb98/Downloads/BuildFromScratch")
sys.path.insert(0, str(ROOT))                 # safe_cuda
import safe_cuda  # noqa: E402

RESULTS = ROOT / "Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results"
OUT = ROOT / "research/interp"
LAYERS = [6, 13, 20, 27]                       # pre-reg §2 (of 28)
N_SEQ, SEQ = 64, 512                           # pre-reg §2: 32,768 token positions
BOOT = 1000                                    # pre-reg §3 bootstrap resamples
DEVICE = "cuda"

# Pre-registered checkpoint arms (seed 0 primary; extra seeds for the noise ceiling).
CKPT = {
    "adamw_42M":    "checkpoint_adamw_seed0.pt",
    "normuon_42M":  "checkpoint_normuon_seed0.pt",
    "adamw_168M":   "checkpoint_persist_168M_adamw_s0.pt",
    "normuon_168M": "checkpoint_persist_168M_normuon_s0.pt",
    "adamw_420M":   "checkpoint_persist_420M_adamw_s0.pt",
    "normuon_420M": "checkpoint_persist_420M_normuon_s0.pt",
    "adamw_420M_s1":   "checkpoint_persist_420M_adamw_s1.pt",     # ceiling
    "normuon_420M_s1": "checkpoint_persist_420M_normuon_s1.pt",   # ceiling
}


def linear_cka(X, Y):
    """Feature-space linear CKA: ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F).
    X,Y are [N,d], row-aligned (same token positions). fp16/bf16 are upcast to fp32; fp32/fp64
    inputs keep their dtype (self-test passes fp64 for exactness, the probe passes fp32 on GPU)."""
    import torch
    if X.dtype in (torch.float16, torch.bfloat16):
        X = X.float()
    if Y.dtype in (torch.float16, torch.bfloat16):
        Y = Y.float()
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = X.t() @ Y
    denom = (X.t() @ X).norm() * (Y.t() @ Y).norm()
    return float((xty.norm() ** 2) / denom) if denom > 0 else 0.0


def bootstrap_delta_ci(acts, key_nb, key_ab, key_target, *, seed=0, B=BOOT, alpha=0.05):
    """Paired bootstrap CI on Δ = CKA(NorMuon@B, AdamW@420M) − CKA(AdamW@B, AdamW@420M),
    resampling token positions (rows) identically across the three aligned matrices."""
    import torch
    g = torch.Generator().manual_seed(seed)
    A_nb, A_ab, A_t = acts[key_nb], acts[key_ab], acts[key_target]
    n = A_t.shape[0]
    point = linear_cka(A_nb, A_t) - linear_cka(A_ab, A_t)
    deltas = []
    for _ in range(B):
        idx = torch.randint(0, n, (n,), generator=g).to(A_t.device)
        deltas.append(linear_cka(A_nb[idx], A_t[idx]) - linear_cka(A_ab[idx], A_t[idx]))
    deltas.sort()
    lo = deltas[int(alpha / 2 * B)]
    hi = deltas[int((1 - alpha / 2) * B)]
    return point, lo, hi


# ─────────────────────────── GPU path ───────────────────────────
def _load_model(name):
    import torch
    sys.path.insert(0, str(ROOT / "Qwen3-0.6B"))
    from model import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config()
    model = Qwen3ForCausalLM(cfg)
    if name != "RANDOM":
        sd = torch.load(RESULTS / CKPT[name], map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()


def _extract(name, seqs, batch=4):
    """Residuals at each LAYER for one checkpoint. Trunk-only forward (no lm_head)."""
    import torch
    model = _load_model(name)
    buf = {L: [] for L in LAYERS}

    def mk(L):
        def hook(_m, _i, out):
            # fp32, NOT fp16: NorMuon residual-stream magnitudes reach ~1e6 (>fp16 max 65504),
            # which would overflow to inf and silently collapse CKA to 0. Verified 2026-07-19.
            h = (out[0] if isinstance(out, (tuple, list)) else out).detach().float()
            buf[L].append(h.reshape(-1, h.shape[-1]).cpu())
        return hook

    handles = [model.model.layers[L].register_forward_hook(mk(L)) for L in LAYERS]
    try:
        with torch.no_grad():
            for b in range(0, seqs.shape[0], batch):
                model.model(input_ids=seqs[b:b + batch].to(DEVICE))   # TRUNK ONLY — no logits
    finally:
        for h in handles:
            h.remove()
    del model
    torch.cuda.empty_cache()
    return {L: torch.cat(buf[L], 0) for L in LAYERS}   # [N,d] fp32 CPU


def run_probe():
    safe_cuda.guard(0.85)
    import torch
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    # Load ONLY wikitext-2 val — identical corpus + tokenization to text-lm-v2's wikitext2_val
    # (score_cohort.load_corpora lines 54/65) — without its code corpus, which we do not use.
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    wt_text = "\n\n".join(t for t in wt["text"] if t.strip())
    ids = tok(wt_text, return_tensors="pt").input_ids[0]
    assert ids.numel() >= N_SEQ * SEQ, f"wikitext2_val has {ids.numel()} toks < {N_SEQ*SEQ}"
    seqs = ids[: N_SEQ * SEQ].reshape(N_SEQ, SEQ)      # [64,512], fixed deterministic slice

    # acts[name][L] = [N,d] residuals. Process one checkpoint at a time (memory-safe).
    names = list(CKPT) + ["RANDOM"]
    acts = {}
    for nm in names:
        acts[nm] = _extract(nm, seqs)
        # INTEGRITY GATE (anti-laundering): refuse to score non-finite activations. A silent
        # inf/nan (e.g. an fp16 overflow) collapses CKA to 0 and fakes a null — the exact class
        # of numerical confound this repo exists to catch.
        for L in LAYERS:
            mx = float(acts[nm][L].abs().max())
            assert torch.isfinite(acts[nm][L]).all(), \
                f"non-finite activations in {nm} L{L} (max|act|={mx:.1f}) — numerical bug, refusing to score"
        print(f"  extracted {nm}: layers {LAYERS}, N={acts[nm][LAYERS[0]].shape[0]}, "
              f"max|act|={max(float(acts[nm][L].abs().max()) for L in LAYERS):.0f}", flush=True)

    per_layer = {}
    for L in LAYERS:
        # Move this layer's matrices to GPU once (fp32→double in linear_cka); the 1000-iter
        # bootstrap on [32768,1024] is PFLOP-scale and impractical on CPU.
        a = {nm: acts[nm][L].to(DEVICE).float() for nm in names}
        ceiling_adamw = linear_cka(a["adamw_420M"], a["adamw_420M_s1"])
        ceiling_normuon = linear_cka(a["normuon_420M"], a["normuon_420M_s1"])
        floor = linear_cka(a["RANDOM"], a["adamw_420M"])
        # INTEGRITY GATE: two seeds of the SAME config must be highly similar; a near-0 ceiling
        # means a degenerate/numerical failure, not a real measurement — refuse to emit a verdict.
        assert ceiling_adamw > 0.3 and ceiling_normuon > 0.3, \
            f"degenerate same-config ceiling at L{L} (adamw={ceiling_adamw:.3f} normuon={ceiling_normuon:.3f}) — numerical bug"
        band = 1.0 - ceiling_adamw                      # pre-reg §4 across-seed band (AdamW@420M)
        budgets = {}
        for B in ("42M", "168M"):
            pt, lo, hi = bootstrap_delta_ci(a, f"normuon_{B}", f"adamw_{B}", "adamw_420M")
            cka_n = linear_cka(a[f"normuon_{B}"], a["adamw_420M"])
            cka_a = linear_cka(a[f"adamw_{B}"], a["adamw_420M"])
            confirmed = (lo > 0) and (pt > band)        # pre-reg §4 decision rule
            budgets[B] = {"cka_normuon_vs_adamw420": cka_n, "cka_adamw_vs_adamw420": cka_a,
                          "delta": pt, "ci95": [lo, hi], "exceeds_band": pt > band,
                          "ci_disjoint_pos": lo > 0, "H1_confirmed": confirmed}
        per_layer[L] = {"ceiling_adamw_s0s1": ceiling_adamw, "ceiling_normuon_s0s1": ceiling_normuon,
                        "floor_random_vs_adamw420": floor, "across_seed_band": band, "budgets": budgets}
        del a
        torch.cuda.empty_cache()

    # Headline (pre-reg §4): H1 needs ≥2 of 4 layers confirmed for the SAME budget.
    headline = "null"
    for B in ("42M", "168M"):
        nconf = sum(per_layer[L]["budgets"][B]["H1_confirmed"] for L in LAYERS)
        if nconf >= 2:
            headline = f"H1-confirmed@{B} ({nconf}/4 layers)"
    verdict = {
        "prereg": "research/interp/prereg_2026-07-19_repconvergence.md",
        "lifecycle_stage": "interpretability", "metric": "linear_cka", "layers": LAYERS,
        "eval_batch": {"corpus": "wikitext2_val", "n_seq": N_SEQ, "seq": SEQ, "tokens": N_SEQ * SEQ},
        "headline": headline, "per_layer": per_layer,
        "note": ("A null (headline='null') is the pre-registered PASSING deliverable at this scale; "
                 "CKA is a similarity summary, not a causal proof."),
    }
    (OUT / "cka_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"headline": headline,
                      "per_layer_ceiling": {L: round(per_layer[L]["ceiling_adamw_s0s1"], 4) for L in LAYERS},
                      "per_layer_floor": {L: round(per_layer[L]["floor_random_vs_adamw420"], 4) for L in LAYERS}},
                     indent=2))
    return verdict


# ─────────────────────────── CPU self-test ───────────────────────────
def _self_test():
    import torch
    torch.manual_seed(0)
    X = torch.randn(500, 64).double()                    # fp64 → exact-tolerance asserts below
    assert abs(linear_cka(X, X) - 1.0) < 1e-9, "CKA(X,X) must be 1"
    assert abs(linear_cka(X, X * 3.0 + 5.0) - 1.0) < 1e-6, "CKA is invariant to isotropic scale+shift"
    Y = torch.randn(500, 64).double()
    assert linear_cka(X, Y) < 0.15, f"CKA of independent gaussians should be low, got {linear_cka(X,Y)}"
    R = torch.linalg.qr(torch.randn(64, 64).double())[0]  # orthogonal rotation
    assert abs(linear_cka(X, X @ R) - 1.0) < 1e-6, "linear CKA invariant to orthogonal transform"
    # decision-rule plumbing: a clear positive delta with a tight CI confirms; a straddling CI nulls
    acts = {"n": X, "a": Y, "t": X.clone()}               # CKA(n,t)=1 >> CKA(a,t)≈0 → big positive Δ
    pt, lo, hi = bootstrap_delta_ci(acts, "n", "a", "t", B=200)
    assert lo > 0 and pt > 0.5, f"clear case must give disjoint-positive Δ, got {pt} [{lo},{hi}]"
    acts2 = {"n": Y, "a": Y.clone(), "t": X}              # both ≈0 vs t → Δ≈0, CI straddles
    pt2, lo2, hi2 = bootstrap_delta_ci(acts2, "n", "a", "t", B=200)
    assert lo2 <= 0 <= hi2, f"null case CI must straddle 0, got {pt2} [{lo2},{hi2}]"
    print("cka_probe self-test PASS — CKA identity/scale/rotation invariance hold; "
          f"decision rule: clear Δ={pt:.3f}[{lo:.3f},{hi:.3f}] confirms, null Δ={pt2:.3f}[{lo2:.3f},{hi2:.3f}] straddles")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        run_probe()
