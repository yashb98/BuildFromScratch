"""verify.py — the correctness gate for HybridSSM-0.2B (novel design → NOT bit-exact; ~1e-2, §C14).

Gates: (1) the SSM parallel scan == its sequential reference; (2) chunked CE == naive CE (avoids the
151,936-logit materialization, §C1); (3) param count sane; (4) all ablation toggles forward finite;
(5) forward is deterministic. No training may start until this exits 0 (§C5.0 correctness-first).
"""
import sys
sys.path.insert(0, "/home/yashb98/Downloads/BuildFromScratch")
import jax_safe_env  # noqa: E402  (before jax, §C1)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import dataclasses  # noqa: E402
import ssm  # noqa: E402
import model as M  # noqa: E402

TOL_SCAN, TOL_CE = 1e-4, 1e-3
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def main():
    k = jax.random.PRNGKey(0)

    # 1) SSM scan vs sequential reference
    ks = jax.random.split(k, 5)
    a = jax.nn.sigmoid(jax.random.normal(ks[0], (2, 32, 8)))
    bx = jax.random.normal(ks[1], (2, 32, 8, 4)) * 0.1
    Cc = jax.random.normal(ks[2], (2, 32, 4))
    D = jax.random.normal(ks[3], (8,))
    x = jax.random.normal(ks[4], (2, 32, 8))
    d_scan = float(jnp.abs(ssm.selective_scan(a, bx, Cc, D, x) - ssm.selective_scan_ref(a, bx, Cc, D, x)).max())
    check("ssm_scan_vs_reference", d_scan < TOL_SCAN, f"max|Δ|={d_scan:.2e}")

    # 2) chunked CE vs naive CE
    h = jax.random.normal(ks[0], (128, 768)) * 0.5
    e = jax.random.normal(ks[1], (512, 768)) * 0.1
    t = jax.random.randint(ks[2], (128,), 0, 512)
    ce_c = float(M.chunked_cross_entropy(h, e, t, chunk=64))
    ce_n = float(-jax.nn.log_softmax((h @ e.T).astype(jnp.float32), axis=-1)[jnp.arange(128), t].mean())
    check("chunked_ce_vs_naive", abs(ce_c - ce_n) < TOL_CE, f"chunked={ce_c:.6f} naive={ce_n:.6f} |Δ|={abs(ce_c-ce_n):.2e}")

    # 3) param count + forward
    cfg = M.HybridConfig()
    net = M.HybridSSM(cfg)
    ids = jax.random.randint(k, (2, 64), 0, cfg.vocab_size)
    params = net.init(k, ids)["params"]
    ne, tot = M.count_params(params)
    hidden, emb = net.apply({"params": params}, ids)
    check("param_count_sane", 100e6 < ne < 500e6, f"non-embed={ne/1e6:.1f}M total={tot/1e6:.1f}M")
    check("forward_finite", bool(jnp.isfinite(hidden).all()), f"hidden={hidden.shape}")

    # 4) toggle sweep finite
    sc = M.HybridConfig(vocab_size=2048, d_model=256, n_layers=6, n_heads=8, n_kv_heads=2, head_dim=32, ffn=512)
    sids = jax.random.randint(k, (2, 48), 0, 2048)
    allfin = True
    for mixer in ("ssm", "swa128"):
        for ae in (2, 4):
            for nope in (False, True):
                c2 = dataclasses.replace(sc, mixer=mixer, attn_every=ae, nope_on_full=nope)
                n2 = M.HybridSSM(c2)
                hh, _ = n2.apply({"params": n2.init(k, sids)["params"]}, sids)
                allfin = allfin and bool(jnp.isfinite(hh).all())
    check("toggle_sweep_finite", allfin, "8 combos (mixer×attn_every×nope)")

    # 5) determinism
    h2, _ = net.apply({"params": params}, ids)
    check("forward_deterministic", float(jnp.abs(hidden - h2).max()) == 0.0)

    print(f"\nVERIFY {'PASS' if not FAILS else 'FAIL: ' + ','.join(FAILS)}")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
