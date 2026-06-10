"""NorMuon optimizer + cautious weight decay — Build 2 (Modernized).

Primary sources (verified 2026-06-08):
  - NorMuon: arXiv 2510.05491 (Oct 2025)
  - Cautious WD + use in IMU-1: arXiv 2602.02522 (Jan 2026, Eq7)

NorMuon = Muon (Newton-Schulz orthogonalization of the momentum) + per-NEURON
(row-wise) second-moment normalization of the orthogonalized update, with an
RMS-matched learning-rate scale. Applied to 2D hidden weight matrices only;
embeddings / norms / scalars use Adam (handled by the caller via param groups).

Verified update (2510.05491):
  M  = b1*M + (1-b1)*G
  O  = NS5(M)                                   # orthogonalize, Frobenius-normalized
  v  = b2*v + (1-b2)*mean_cols(O*O)             # per-row (per-neuron) 2nd moment
  Ohat = O / (sqrt(V) + eps)                    # V = v broadcast across columns
  eta_hat = 0.2*lr*sqrt(m*n)/||Ohat||_F         # RMS match to Adam (Jordan 2024)
  W  = W - lr*lambda*W - eta_hat*Ohat
Cautious WD (2602.02522 Eq7): apply the -lr*lambda*W term only where
sign(update) == sign(weight), i.e. masked by (Ohat * W > 0).

NS5 coefficients (a,b,c)=(3.4445,-4.7750,2.0315) are NOT printed in the NorMuon
paper; it cites Jordan et al. 2024 (Muon), whose standard quintic these are.
"""
import torch
from torch.optim.optimizer import Optimizer


def _newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Standard Muon quintic Newton-Schulz (Jordan 2024). Returns the orthogonal
    factor of G in G's original shape. Runs in fp32 for stability."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):          # keep the smaller dim first (cheaper A=XX^T)
        X = X.t(); transposed = True
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X


class NorMuon(Optimizer):
    def __init__(self, params, lr=0.011, weight_decay=0.1, beta1=0.95, beta2=0.95,
                 eps=1e-8, ns_steps=5, cautious=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                        eps=eps, ns_steps=ns_steps, cautious=cautious)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            b1, b2, eps = group["beta1"], group["beta2"], group["eps"]
            ns_steps, cautious = group["ns_steps"], group["cautious"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.dim() == 2, "NorMuon is for 2D matrices only; route others to Adam"
                st = self.state[p]
                if not st:
                    st["M"] = torch.zeros_like(p, dtype=torch.float32)
                    st["v"] = torch.zeros(p.size(0), device=p.device, dtype=torch.float32)
                M, v = st["M"], st["v"]

                M.mul_(b1).add_(g.to(torch.float32), alpha=1 - b1)
                O = _newton_schulz5(M, steps=ns_steps)                 # (m, n) fp32
                v.mul_(b2).add_((O * O).mean(dim=1), alpha=1 - b2)     # per-row 2nd moment
                Ohat = O / (v.sqrt().unsqueeze(1) + eps)               # row-normalize

                m_, n_ = p.shape
                eta_hat = 0.2 * lr * (m_ * n_) ** 0.5 / (Ohat.norm() + eps)

                if wd != 0.0:
                    if cautious:
                        mask = (Ohat * p.data > 0).to(p.dtype)        # sign(update)=sign(w)
                        p.data.add_(p.data * mask, alpha=-lr * wd)
                    else:
                        p.data.add_(p.data, alpha=-lr * wd)
                p.data.add_(Ohat.to(p.dtype), alpha=-eta_hat)
        return loss
