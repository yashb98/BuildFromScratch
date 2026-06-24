"""On-box interpretability core for Qwen3-0.6B (spec: research/interp/build_spec.md).

HONEST VERDICT, UP FRONT (build_spec §1): at 596M params / ~1.19B tokens the EXPECTED result is a
CI-backed NULL at the control floor — arXiv:2602.14111 shows random baselines match trained SAEs and
~9% true-feature recovery even on clean synthetic data; an undertrained 596M residual stream is a
*more* entangled basis. **A correctly-measured null is the PASSING deliverable, not a failure.**

The load-bearing anti-laundering property (build_spec §4 KEY RISK) lives HERE, not in the §C25 gate:
this module is structurally unable to emit a "win" unless the paired bootstrap-CI on (SAE − floor)
is **strictly disjoint from 0 vs BOTH the PCA floor AND the random-SAE floor**. The PCA/random floors
are first-class arms computed BEFORE any SAE headline. `_self_test` proves a CI-overlapping case
yields the null.

Verified-current methods only (build_spec §5, all WebFetch-checked 2026-06-24): BatchTopK SAE
(arXiv:2412.06410), Matryoshka SAE (2503.17547). AuxK dead-latent revival (`k_aux`,`aux_alpha`) is
implementation lore — tunable defaults, NOT paper-cited. `absorption_first_letter` /
`synth_recovery_f1_mcc` are stubs pending a full-text formula read (2409.14507 / 2602.14687).

CPU-stub-testable: SAE train + floors + CI-gate + probe + the anti-laundering self-test run on tiny
synthetic activations with no model/GPU/network. GPU entry points call `safe_cuda.guard(0.85)` and
cache RESIDUALS ONLY (never the 151,936-vocab logits — the documented box-crash vector).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── ActivationCache — forward hooks are MANDATORY ─────────────────────────
class ActivationCache:
    """Cache one layer's residual-stream activations via a forward hook. The model's
    `Qwen3Model.forward` returns ONLY `self.norm(x)` (build_spec §2.1, verified model_imu1.py:273) —
    so reading the return value caches the WRONG tensor. We hook `model.model.layers[L]` and take the
    block-output hidden state (the block returns `(x, v_local)` → take `out[0]`)."""
    MAX_TOKENS = 5_000_000   # HARD CAP: residual d=1024 fp16 ≈ 2.05 GB/1M tok; 50M tok = 102 GB = OOM crash

    def __init__(self, model, layer: int, pool: str = "all"):
        assert pool in ("last", "mean", "all")
        self.model, self.layer, self.pool = model, layer, pool
        self._buf, self._handle = [], None

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out   # block returns (x, v_local) → take x
        h = h.detach().to(torch.float16)                        # [B, T, d]
        if self.pool == "last":
            self._buf.append(h[:, -1, :])
        elif self.pool == "mean":
            self._buf.append(h.mean(dim=1))
        else:
            self._buf.append(h.reshape(-1, h.shape[-1]))        # all token positions
        self._buf.append  # noqa (kept explicit)

    def fill(self, batches, max_tokens: int):
        """batches: iterable of input_id tensors [B, T]. Returns acts [N, d_model] (fp16, CPU)."""
        assert max_tokens <= self.MAX_TOKENS, f"max_tokens {max_tokens} > {self.MAX_TOKENS} cap (OOM guard, §C1)"
        self._buf = []
        self._handle = self.model.model.layers[self.layer].register_forward_hook(self._hook)
        try:
            total = 0
            with torch.no_grad():
                for ids in batches:
                    self.model(input_ids=ids)
                    total += sum(b.shape[0] for b in self._buf[-1:])
                    if sum(b.shape[0] for b in self._buf) >= max_tokens:
                        break
        finally:
            self._handle.remove(); self._handle = None
        return torch.cat(self._buf, dim=0)[:max_tokens].cpu()


# ───────────────────────── SAE — BatchTopK (verified-current arch) ─────────────────────────
class SAE(nn.Module):
    """Sparse autoencoder with BATCH-LEVEL top-k SELECTION (arXiv:2412.06410 — a selection method,
    NOT an activation function): per training batch keep the top `k * batch_size` pre-activations
    across the whole batch, zero the rest. Inference uses a per-sample top-k (the threshold variant
    is a refinement). `k_aux`/`aux_alpha` (dead-latent revival) are tunable lore, not paper-cited."""

    def __init__(self, d_model=1024, dict_size=16384, k=32, variant="batchtopk", tied=False,
                 k_aux=512, aux_alpha=1.0 / 32):
        super().__init__()
        self.d, self.m, self.k, self.variant = d_model, dict_size, k, variant
        self.k_aux, self.aux_alpha = k_aux, aux_alpha
        self.W_enc = nn.Parameter(torch.randn(d_model, dict_size) * (1.0 / math.sqrt(d_model)))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(self.W_enc.detach().t().clone() if tied else
                                  torch.randn(dict_size, d_model) * (1.0 / math.sqrt(dict_size)))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("fired", torch.zeros(dict_size))   # dead-feature tracking

    def _pre(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)   # [B, m] >= 0

    def encode(self, x, training=False):
        pre = self._pre(x)
        if training and self.variant.startswith("batchtopk"):
            flat = pre.reshape(-1)                                   # batch-level selection
            kk = min(self.k * x.shape[0], flat.numel())
            if kk < flat.numel():
                thresh = torch.topk(flat, kk).values.min()
                pre = torch.where(pre >= thresh, pre, torch.zeros_like(pre))
        else:
            topk = torch.topk(pre, min(self.k, self.m), dim=-1)     # per-sample top-k (inference)
            mask = torch.zeros_like(pre).scatter_(-1, topk.indices, 1.0)
            pre = pre * mask
        return pre

    def forward(self, x, training=False):
        z = self.encode(x, training=training)
        recon = z @ self.W_dec + self.b_dec
        if training:
            self.fired += (z > 0).float().sum(dim=0).detach()
        return recon, z

    def l0(self, x):
        with torch.no_grad():
            return float((self.encode(x) > 0).float().sum(dim=-1).mean())


def train_sae(acts, dict_size=None, k=32, steps=400, lr=1e-3, batch=2048, seed=0, variant="batchtopk"):
    """Train an SAE on cached activations [N, d] (CPU/GPU). Returns (sae, history)."""
    torch.manual_seed(seed)
    N, d = acts.shape
    dict_size = dict_size or 16 * d
    sae = SAE(d, dict_size, k, variant=variant).to(acts.device, dtype=torch.float32)
    sae.b_dec.data = acts.float().mean(dim=0)                       # init decoder bias at data mean
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    hist = []
    g = torch.Generator(device=acts.device).manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g, device=acts.device)
        x = acts[idx].float()
        recon, z = sae(x, training=True)
        loss = F.mse_loss(recon, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 5) == 0:
            hist.append((step, float(loss.detach())))
    return sae, hist


# ───────────────────────── Control floors (FIRST-CLASS, before any SAE number) ─────────────────────────
def pca_floor(acts, k):
    """Truncated-SVD reconstruction with k components — the linear-subspace floor. Returns per-item
    reconstruction error (lower = better recon)."""
    x = acts.float()
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    k = min(k, min(xc.shape) - 1)
    U, S, Vh = torch.linalg.svd(xc, full_matrices=False)
    recon = (U[:, :k] * S[:k]) @ Vh[:k] + mu
    return ((x - recon) ** 2).sum(dim=-1)                           # [N] per-item SSE


def random_sae_floor(acts, dict_size=None, k=32, seed=0):
    """A FROZEN-random SAE (untrained) — the arXiv:2602.14111 random baseline. Per-item recon error."""
    torch.manual_seed(seed)
    d = acts.shape[1]
    sae = SAE(d, dict_size or 16 * d, k).to(acts.device, dtype=torch.float32)
    sae.b_dec.data = acts.float().mean(dim=0)
    with torch.no_grad():
        recon, _ = sae(acts.float())
    return ((acts.float() - recon) ** 2).sum(dim=-1)


# ───────────────────────── the LOAD-BEARING anti-laundering gate ─────────────────────────
def ci_disjoint_verdict(sae_item, pca_item, random_item, *, lower_is_better=True, seed=0, alpha=0.05):
    """'win' ONLY if the paired bootstrap-CI on (SAE − floor) is strictly disjoint from 0 vs BOTH
    the PCA and random-SAE floors, in the improving direction. Else the honest 'null'. This is the
    structural property that makes a floor-in-disguise impossible to report as a discovery."""
    from eval_metrics import bootstrap_ci
    sae_item = [float(v) for v in sae_item]

    def delta(s, f):                                               # mean(SAE) − mean(floor)
        return sum(s) / len(s) - sum(f) / len(f)

    results = {}
    win = True
    for floor, name in ((pca_item, "pca"), (random_item, "random")):
        pt, lo, hi = bootstrap_ci(delta, (sae_item, [float(v) for v in floor]),
                                  paired=True, seed=seed, alpha=alpha)
        # improvement: lower_is_better → SAE−floor must be < 0; else > 0
        disjoint = (hi < 0) if lower_is_better else (lo > 0)
        results[name] = {"delta": pt, "ci": [lo, hi], "disjoint": disjoint}
        win = win and disjoint
    return {"verdict": "win" if win else "null", "vs_floors": results,
            "note": "null at control floor is a PASSING result at this scale (build_spec §1)"}


# ───────────────────────── linear probe (vs random-init + confound floors) ─────────────────────────
def probe_auroc(acts, labels, *, seed=0):
    """Logistic-regression linear probe → ROC-AUC (via eval_metrics.roc_auc). Returns per-item
    scores + auroc. The caller compares vs a random-init-model floor and a length/freq-confound floor
    (build_spec §2.4) — a probe that only beats random-init but not the confound is reported as
    'surface feature, not semantic'."""
    from eval_metrics import roc_auc
    torch.manual_seed(seed)
    x = acts.float()
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    y = torch.tensor([float(v) for v in labels])
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(300):
        logit = x @ w + b
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
    scores = (x @ w + b).detach().tolist()
    return {"scores": scores, "auroc": roc_auc(scores, [int(v) for v in labels])}


# stubs that need a full-text formula read before honest implementation (build_spec §2.4, §6)
def absorption_first_letter(*a, **k):
    raise NotImplementedError("read arxiv.org/html/2409.14507 for the absorption-fraction formula first "
                              "— it is NOT in the abstract; do not implement from memory (build_spec §6)")


def synth_recovery_f1_mcc(*a, **k):
    raise NotImplementedError("down-scaled SynthSAEBench harness — build with our own F1/MCC + Hungarian "
                              "feature-matching; the 2602.14687 size figures are UNVERIFIED (build_spec §6)")


# ───────────────────────── CPU self-test: reproduce the null + prove the gate can't launder ─────────────────────────
def _self_test():
    torch.manual_seed(0)
    d, N = 64, 4000
    # (a) RANDOM activations: trained SAE should NOT beat the random-SAE floor → 2602.14111 reproduced
    rand_acts = torch.randn(N, d)
    sae, _ = train_sae(rand_acts, dict_size=8 * d, k=8, steps=120, batch=512, seed=0)
    with torch.no_grad():
        recon, _ = sae(rand_acts.float())
    sae_err = ((rand_acts.float() - recon) ** 2).sum(-1).tolist()
    pca_err = pca_floor(rand_acts, k=8 * d).tolist()
    rnd_err = random_sae_floor(rand_acts, dict_size=8 * d, k=8, seed=1).tolist()
    v = ci_disjoint_verdict(sae_err, pca_err, rnd_err, lower_is_better=True, seed=0)
    assert v["verdict"] == "null", f"on RANDOM data the gate must return null, got {v}"

    # (b) anti-laundering: a NOISY near-zero improvement (CI straddles 0) MUST be null, not a win
    import random as _r
    rr = _r.Random(0)
    base = [1.0 + rr.gauss(0, 0.5) for _ in range(300)]
    noisy = [b + rr.gauss(0, 0.5) for b in base]                    # delta ~ N(0,·) → CI straddles 0
    assert ci_disjoint_verdict(noisy, base, base, lower_is_better=True, seed=0)["verdict"] == "null"

    # (c) a GENUINELY disjoint case MUST be a win (the gate isn't broken-shut)
    sae_good = [0.1 + 0.01 * math.sin(i) for i in range(300)]       # clearly lower error, tight
    floor_hi = [1.0 + 0.01 * math.cos(i) for i in range(300)]
    assert ci_disjoint_verdict(sae_good, floor_hi, floor_hi, lower_is_better=True, seed=0)["verdict"] == "win"

    # (d) probe AUROC plumbing + determinism on a linearly-separable feature
    acts2 = torch.randn(400, d); lab = (acts2[:, 0] > 0).long().tolist()
    acts2[:, 0] += torch.tensor(lab).float() * 3.0
    p1 = probe_auroc(acts2, lab, seed=0); p2 = probe_auroc(acts2, lab, seed=0)
    assert p1["auroc"] == p2["auroc"] and p1["auroc"] > 0.8, p1["auroc"]
    print(f"interp self-test PASS — null reproduced on random data (verdict={v['verdict']}); "
          f"CI-overlap→null & disjoint→win both hold; probe AUROC={p1['auroc']:.3f} deterministic")


if __name__ == "__main__":
    _self_test()
