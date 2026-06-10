"""Build 3 verify gate — partial RoPE.

Proves:
  (1) partial_rotary_factor=1.0 is BIT-IDENTICAL to the faithful reference model
      (every unchanged component — RMSNorm, QK-norm, GQA, SwiGLU, embeddings — holds).
  (2) factors 0.25 / 0.10 give the right rotary_dim, stay finite, and actually
      change the output (partial RoPE is doing something), while sharing identical
      weights with the baseline (only the RoPE buffer differs).

Runs on CPU in float32 so it does not touch the GPU sweep.
"""
import sys, pathlib, torch

HERE = pathlib.Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[1]          # Qwen3-0.6B/  (faithful model.py)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODEL_DIR))

import model as faithful              # Qwen3-0.6B/model.py
import model_partialrope as partial   # this folder

torch.manual_seed(0)
cfg_f = faithful.Qwen3Config()
m_f = faithful.Qwen3ForCausalLM(cfg_f).eval()
sd = m_f.state_dict()

ids = torch.randint(0, cfg_f.vocab_size, (1, 16))
with torch.no_grad():
    lf = m_f(ids)["logits"]

results = {}

# (1) factor=1.0 == faithful, bit-exact
m1 = partial.Qwen3ForCausalLM(partial.Qwen3Config(partial_rotary_factor=1.0)).eval()
m1.load_state_dict(sd, strict=True)
with torch.no_grad():
    l1 = m1(ids)["logits"]
err = (lf - l1).abs().max().item()
print(f"[1] factor=1.0  rotary_dim={m1.model.rotary_dim:3d}  max_abs_err vs faithful = {err}")
results["factor_1.0_bitexact"] = (err == 0.0)
assert err == 0.0, f"factor=1.0 must be bit-identical, got {err}"

# (2) partial factors: right rotary_dim, finite, and DIFFERENT from baseline
for fac, want_rd in [(0.25, 32), (0.10, 12)]:
    m = partial.Qwen3ForCausalLM(partial.Qwen3Config(partial_rotary_factor=fac)).eval()
    m.load_state_dict(sd, strict=True)   # identical weights → isolates RoPE effect
    with torch.no_grad():
        lp = m(ids)["logits"]
    rd = m.model.rotary_dim
    finite = bool(torch.isfinite(lp).all())
    delta = (lp - lf).abs().max().item()
    ok = (rd == want_rd) and finite and (delta > 0)
    print(f"[2] factor={fac:<4}  rotary_dim={rd:3d} (want {want_rd})  finite={finite}  "
          f"max|Δ| vs full-RoPE = {delta:.4f}  -> {'OK' if ok else 'FAIL'}")
    results[f"factor_{fac}"] = ok
    assert ok

print("\nALL PASS" if all(results.values()) else "\nFAIL")
