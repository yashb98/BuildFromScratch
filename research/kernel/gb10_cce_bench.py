#!/usr/bin/env python3
"""GB10 (Blackwell) fused-CE benchmark — the same 'killing the logit tensor' evals as the Kaggle
notebook, but on real deployment hardware where CCE's 96 KB-shared-memory kernel can actually run.
safe_cuda-guarded; the anneal is paused while this runs (one GPU job at a time, §C4.5)."""
import sys, pathlib, json, gc
import numpy as np
ROOT = pathlib.Path("/home/yashb98/Downloads/BuildFromScratch")
sys.path.insert(0, str(ROOT))
import safe_cuda  # noqa: F401 — caps the process before torch touches CUDA
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd

safe_cuda.guard(0.85)
dev = torch.device("cuda")
cap = torch.cuda.get_device_capability(0)
DTYPE = torch.bfloat16 if cap[0] >= 8 else torch.float16   # GB10 is Blackwell -> bf16
V, D, SEED = 151936, 1024, 0
OUT = ROOT / "research/kernel"
print("GPU", torch.cuda.get_device_name(0), "| cc", cap, "| dtype", DTYPE,
      "| pool", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB", flush=True)

def make(N):
    g = torch.Generator(device=dev).manual_seed(SEED)
    h = (torch.randn(N, D, device=dev, dtype=DTYPE, generator=g) * .1).requires_grad_()
    W = (torch.randn(V, D, device=dev, dtype=DTYPE, generator=g) * .02).requires_grad_()
    y = torch.randint(0, V, (N,), device=dev, generator=g)
    return h, W, y

def ce_naive(h, W, y): return F.cross_entropy((h @ W.t()).float(), y)
ce_compiled = torch.compile(ce_naive)
methods = {"naive": ce_naive, "torch.compile": ce_compiled}
try:
    from cut_cross_entropy import linear_cross_entropy
    methods["CCE"] = lambda h, W, y: linear_cross_entropy(h, W, y); print("CCE imported", flush=True)
except Exception as e: print("CCE import failed:", repr(e)[:140], flush=True)
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _flce = LigerFusedLinearCrossEntropyLoss()
    methods["Liger"] = lambda h, W, y: _flce(W, h, y); print("Liger imported", flush=True)
except Exception as e: print("Liger import failed:", repr(e)[:140], flush=True)

# smoke each -> drop any that won't run on this GPU (e.g. CCE if Triton is too new for Blackwell)
for nm in list(methods):
    try:
        h, W, y = make(64); methods[nm](h, W, y).backward(); del h, W, y; torch.cuda.empty_cache()
        print(f"[{nm}] smoke OK", flush=True)
    except Exception as e:
        print(f"[{nm}] smoke FAILED -> dropped: {repr(e)[:150]}", flush=True); methods.pop(nm); torch.cuda.empty_cache()
print("ACTIVE METHODS:", list(methods), flush=True)

def isoom(e): return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()

# --- Eval 1: correctness vs fp32 oracle ---
def ref(h, W, y): return F.cross_entropy(h.float() @ W.float().t(), y)
def grads(fn, N):
    h, W, y = make(N); L = fn(h, W, y); L.backward()
    return float(L), h.grad.float().clone(), W.grad.float().clone()
Nc = 2048; Lr, ghr, gWr = grads(ref, Nc); rows = []
for nm, fn in methods.items():
    if nm in ("naive", "torch.compile"): continue
    try:
        Lf, ghf, gWf = grads(fn, Nc)
        rows.append([nm, round(Lr,6), round(Lf,6), abs(Lr-Lf),
                     (ghr-ghf).abs().max().item(), (gWr-gWf).abs().max().item()])
    except Exception as e: rows.append([nm, round(Lr,6), None, None, None, repr(e)[:50]])
    torch.cuda.empty_cache()
df_chk = pd.DataFrame(rows, columns=["method","loss_ref","loss","|d_loss|","d_hidden_max","d_weight_max"])
df_chk.to_csv(OUT/"gb10_correctness.csv", index=False); print("CORRECTNESS\n", df_chk.to_string(index=False), flush=True)

# --- Eval 2/3: memory + throughput sweep (GB10 has ~119 GB; safe_cuda catches OOM cleanly) ---
N_SWEEP = [4096, 16384, 32768, 65536]
def peak_mem(fn, N):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        h, W, y = make(N); fn(h, W, y).backward(); torch.cuda.synchronize()
        m = torch.cuda.max_memory_allocated()/1e9; del h, W, y
    except Exception as e:
        if not isoom(e): print("memerr", N, repr(e)[:80], flush=True)
        m = float("nan")
    gc.collect(); torch.cuda.empty_cache(); return m
def bench(fn, N, it=10, wu=3):
    gc.collect(); torch.cuda.empty_cache()
    try:
        h, W, y = make(N)
        for _ in range(wu): h.grad=None; W.grad=None; fn(h, W, y).backward()
        torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); ts=[]
        for _ in range(it):
            h.grad=None; W.grad=None; s.record(); fn(h, W, y).backward(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        t = float(np.median(ts)); del h, W, y
    except Exception: t = float("nan")
    gc.collect(); torch.cuda.empty_cache(); return t
mem  = {nm: [peak_mem(fn, N) for N in N_SWEEP] for nm, fn in methods.items()}
tput = {nm: [bench(fn, N)    for N in N_SWEEP] for nm, fn in methods.items()}
df_mem = pd.DataFrame(mem, index=N_SWEEP); df_mem.index.name="N_tokens"; df_mem.to_csv(OUT/"gb10_memory.csv")
df_t   = pd.DataFrame(tput, index=N_SWEEP); df_t.index.name="N_tokens";  df_t.to_csv(OUT/"gb10_throughput.csv")
print("MEMORY (GB)\n", df_mem.round(3).to_string(), flush=True)
print("THROUGHPUT (ms)\n", df_t.round(2).to_string(), flush=True)

# --- plots ---
for data, ylab, fn_png, title in [(mem,"peak GPU memory (GB)","gb10_mem.png","peak memory"),
                                   (tput,"fwd+bwd time (ms)","gb10_tput.png","throughput")]:
    plt.figure(figsize=(8,5))
    for nm in methods: plt.plot(N_SWEEP, data[nm], "o-", label=nm)
    plt.xscale("log"); plt.xlabel("tokens N (batch×seq, log)"); plt.ylabel(ylab)
    plt.title(f"GB10 Grace Blackwell — {title} vs tokens (V={V}, D={D}, bf16)")
    plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig(OUT/fn_png, dpi=120)

summary = {"gpu": torch.cuda.get_device_name(0), "cc": list(cap), "dtype": str(DTYPE),
           "pool_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9,1),
           "V": V, "D": D, "methods": list(methods), "N_sweep": N_SWEEP,
           "memory_gb": {k:[None if v!=v else round(v,3) for v in mem[k]] for k in methods},
           "throughput_ms": {k:[None if v!=v else round(v,2) for v in tput[k]] for k in methods},
           "correctness": rows}
json.dump(summary, open(OUT/"gb10_summary.json","w"), indent=2, default=str)
print("SAVED gb10_* ->", OUT, "\nDONE", flush=True)
