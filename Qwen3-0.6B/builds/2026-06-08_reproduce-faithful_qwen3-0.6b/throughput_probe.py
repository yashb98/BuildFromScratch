"""
Throughput probe — measures tokens/sec on Qwen3-0.6B at the planned training
shape (Config B: bs=16, seq_len=4096, bf16) on this hardware.

Runs two passes: torch.compile OFF then ON. Reports steady-state tokens/sec and
peak memory for both. Output drives the Phase-8 cost gate.

    python throughput_probe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # BuildFromScratch/
sys.path.insert(0, str(REPO_ROOT))
import safe_cuda  # noqa: E402,F401  — sets expandable_segments before torch touches CUDA

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Mean token cross-entropy without ever materializing the full (N, vocab)
    fp32 tensor. At vocab=151936 the naive `logits.float()` is ~40 GB at the
    training shape and OOM'd the unified pool on 2026-06-08; chunking caps the
    fp32 working set at `chunk` rows."""
    flat = logits.reshape(-1, logits.size(-1))
    tgt = targets.reshape(-1)
    total = flat.new_zeros(())
    n = flat.size(0)
    for i in range(0, n, chunk):
        total = total + torch.nn.functional.cross_entropy(
            flat[i : i + chunk].float(), tgt[i : i + chunk], reduction="sum"
        )
    return total / n

BUILD_DIR = Path(__file__).resolve().parent
OUT_JSON = BUILD_DIR / "results" / "throughput_probe.json"
OUT_JSON.parent.mkdir(exist_ok=True)

SEQ_LEN = 4096
MICRO_BS = 16
# Largest-to-smallest micro-batches to try. On this unified-memory box the
# target shape (bs=16) may not fit; the probe backs off and reports the largest
# that does, so the box never crashes hunting for a number.
MICRO_BS_CANDIDATES = [16, 8, 4, 2, 1]
WARMUP_STEPS = 5
MEASURE_STEPS = 50


def run_pass(label: str, compile_mode: bool, micro_bs: int) -> dict:
    print(f"\n=== {label} (compile={compile_mode}, micro_bs={micro_bs}) ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = Qwen3Config()
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
    model.train()

    train_model = model
    if compile_mode:
        print("  compiling (one-time cost) ...")
        t_c = time.time()
        train_model = torch.compile(model, mode="reduce-overhead")
        print(f"  compile setup: {time.time()-t_c:.1f}s (kernel autotune happens on first fwd)")

    optim = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    def step():
        x = torch.randint(0, cfg.vocab_size, (micro_bs, SEQ_LEN), device=device)
        logits = train_model(input_ids=x)["logits"]
        loss = chunked_cross_entropy(logits, x)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        return loss.item()

    # Warmup
    print(f"  warmup {WARMUP_STEPS} steps (incl. compile autotune if on) ...")
    t_w = time.time()
    for _ in range(WARMUP_STEPS):
        step()
    torch.cuda.synchronize()
    warmup_s = time.time() - t_w
    print(f"  warmup wall: {warmup_s:.1f}s ({warmup_s / WARMUP_STEPS:.2f}s/step)")

    # Measure
    print(f"  measuring {MEASURE_STEPS} steps ...")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(MEASURE_STEPS):
        step()
    torch.cuda.synchronize()
    measure_s = time.time() - t0

    tokens = MEASURE_STEPS * micro_bs * SEQ_LEN
    tps = tokens / measure_s
    sec_per_step = measure_s / MEASURE_STEPS
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    result = {
        "label": label,
        "compile": compile_mode,
        "micro_batch_size": micro_bs,
        "seq_len": SEQ_LEN,
        "measure_steps": MEASURE_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "tokens_per_second": round(tps, 1),
        "seconds_per_step": round(sec_per_step, 3),
        "tokens_per_step": micro_bs * SEQ_LEN,
        "peak_mem_gb": round(peak_mem_gb, 2),
        "warmup_wall_s": round(warmup_s, 1),
        "measure_wall_s": round(measure_s, 1),
    }
    print(f"  tokens/sec  : {tps:,.0f}")
    print(f"  sec/step    : {sec_per_step:.3f}")
    print(f"  peak mem    : {peak_mem_gb:.2f} GB")

    # Clean up before next pass.
    del model, train_model, optim
    torch.cuda.empty_cache()
    return result


def run_pass_with_backoff(label: str, compile_mode: bool) -> dict:
    """Try the target micro-batch, backing off down MICRO_BS_CANDIDATES on a
    clean CUDA OOM (safe_cuda.guard() guarantees OOM is catchable, not a crash)."""
    for micro_bs in MICRO_BS_CANDIDATES:
        try:
            return run_pass(label, compile_mode, micro_bs)
        except torch.OutOfMemoryError:
            print(f"  OOM at micro_bs={micro_bs} — backing off")
            torch.cuda.empty_cache()
    raise RuntimeError(f"{label}: OOM even at micro_bs={MICRO_BS_CANDIDATES[-1]}")


def main():
    if not torch.cuda.is_available():
        print("CUDA required for throughput probe.")
        return 1

    safe_cuda.guard(0.85)  # cap at 85% of the unified pool; OOM errors instead of crashing the box
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model:  Qwen3-0.6B (596M params), bf16")
    print(f"Probe:  target micro_batch={MICRO_BS} (backs off {MICRO_BS_CANDIDATES}), "
          f"seq_len={SEQ_LEN}, measure={MEASURE_STEPS} steps after {WARMUP_STEPS} warmup")

    a = run_pass_with_backoff("baseline", compile_mode=False)
    b = run_pass_with_backoff("compiled", compile_mode=True)

    summary = {
        "device": torch.cuda.get_device_name(0),
        "model_params": "596M",
        "passes": [a, b],
        "speedup_compile_vs_baseline": round(b["tokens_per_second"] / a["tokens_per_second"], 2),
    }
    print("\n=== Summary ===")
    print(f"  baseline tok/s : {a['tokens_per_second']:,.0f}")
    print(f"  compiled tok/s : {b['tokens_per_second']:,.0f}")
    print(f"  speedup        : {summary['speedup_compile_vs_baseline']}×")

    # Wall-clock projections for the 3-config sweep at the chosen budgets:
    # A: 0.5B tokens, B: 1B, C: 0.5B → total 2B tokens.
    for label, tps_key in [("baseline (no compile)", a["tokens_per_second"]),
                            ("compiled", b["tokens_per_second"])]:
        total_tokens = 2_000_000_000
        hours = total_tokens / tps_key / 3600
        print(f"  3-config sweep (2B tokens total) @ {label}: ~{hours:.1f} hr wall")

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
