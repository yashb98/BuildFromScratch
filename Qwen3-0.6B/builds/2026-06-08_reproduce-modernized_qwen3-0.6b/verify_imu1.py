"""Build 2 verify gate — IMU-1 bundle + NorMuon. CPU/fp32, does not touch the GPU.

Checks:
 [1] bundle OFF (all toggles False) == faithful model, BIT-EXACT (unchanged components hold).
 [2] value-residual ON only == faithful at init too (a2=0 => V=V_local) — wiring correctness.
 [3] ln-scaling ON and head-gating ON each CHANGE the output (components are live).
 [4] full bundle ON: finite logits + computes a loss.
 [5] NorMuon minimizes a toy 2D objective over a few steps.
"""
import sys, pathlib, torch

HERE = pathlib.Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(MODEL_DIR))

import model as faithful
import model_imu1 as imu1
from normuon import NorMuon

torch.manual_seed(0)
m_f = faithful.Qwen3ForCausalLM(faithful.Qwen3Config()).eval()
sd = m_f.state_dict()
ids = torch.randint(0, 151936, (1, 16))
with torch.no_grad():
    lf = m_f(ids)["logits"]

def build(**flags):
    cfg = imu1.Qwen3Config(use_value_residual=flags.get("vr", False),
                           use_layernorm_scaling=flags.get("ln", False),
                           use_head_gating=flags.get("hg", False))
    return imu1.Qwen3ForCausalLM(cfg).eval()

ok = {}

# [1] all off -> bit exact (state_dict matches faithful, strict load)
m0 = build()
m0.load_state_dict(sd, strict=True)
with torch.no_grad():
    e = (m0(ids)["logits"] - lf).abs().max().item()
print(f"[1] bundle OFF        max_abs_err vs faithful = {e}")
ok["off==faithful"] = (e == 0.0)

# [2] value-residual ON only -> also bit exact at init (a2=0)
m_vr = build(vr=True)
m_vr.load_state_dict(sd, strict=False)  # vr_s/a1/a2 not in faithful sd; keep their init
with torch.no_grad():
    e_vr = (m_vr(ids)["logits"] - lf).abs().max().item()
print(f"[2] value-resid ON    max_abs_err vs faithful = {e_vr}  (expect 0 at init, a2=0)")
ok["vr_init==faithful"] = (e_vr == 0.0)

# [3] ln-scaling / head-gating each change output
m_ln = build(ln=True); m_ln.load_state_dict(sd, strict=False)
m_hg = build(hg=True); m_hg.load_state_dict(sd, strict=False)
with torch.no_grad():
    d_ln = (m_ln(ids)["logits"] - lf).abs().max().item()
    d_hg = (m_hg(ids)["logits"] - lf).abs().max().item()
print(f"[3] ln-scaling Δ={d_ln:.4f}   head-gating Δ={d_hg:.4f}  (both expect >0)")
ok["ln_live"] = d_ln > 0
ok["hg_live"] = d_hg > 0

# [4] full bundle: finite + loss
m_full = build(vr=True, ln=True, hg=True)
out = m_full(ids, labels=ids)
fin = bool(torch.isfinite(out["logits"]).all()) and bool(torch.isfinite(out["loss"]))
print(f"[4] full bundle       finite={fin}  loss={out['loss'].item():.3f}  "
      f"params={sum(p.numel() for p in m_full.parameters()):,}")
ok["full_finite"] = fin

# [5] NorMuon minimizes a toy 2D objective
torch.manual_seed(1)
W = torch.nn.Parameter(torch.randn(64, 32))
target = torch.randn(64, 32)
opt = NorMuon([W], lr=0.05, weight_decay=0.0)
l0 = None
for i in range(30):
    opt.zero_grad()
    loss = ((W - target) ** 2).mean()
    loss.backward()
    opt.step()
    if i == 0:
        l0 = loss.item()
print(f"[5] NorMuon toy loss  {l0:.4f} -> {loss.item():.4f}  (expect decrease)")
ok["normuon_descends"] = loss.item() < l0

print("\nALL PASS" if all(ok.values()) else f"\nFAIL: {[k for k,v in ok.items() if not v]}")
