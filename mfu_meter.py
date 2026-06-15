#!/usr/bin/env python3
"""mfu_meter.py — Model FLOPs Utilization (MFU) and Hardware FLOPs Utilization
(HFU) for a training run. The single source of the efficiency headline that the
§C5.3 launch probe records and that every Systems-spine artifact leads with.

Stdlib-only, pure CPU, no model load — safe while a trainer is active (§C1).

MFU = achieved_model_FLOPs_per_sec / device_peak_FLOPs_per_sec
    where achieved = flops_per_token(cfg) * tokens_per_sec   (PaLM App. B; see
    flop_accounting.py for the 6N + 12·L·H·Q·T convention).

HFU adds the work the hardware actually did but the model-FLOP count omits —
chiefly activation recomputation under gradient checkpointing, which recomputes
the forward pass (~+2N/token) so the *hardware* does more FLOPs than the model
"needs". HFU >= MFU; the gap is the recompute tax. Report BOTH and say which.

PEAK_FLOPS are BF16 *dense* tensor-core peaks (NO sparsity, NO FP8/FP4), pinned
with a vendor source and a verification date per §C2 — the denominator must be
honest or the ratio is theatre. An UNKNOWN device is REFUSED, never guessed.

The GB10 (this box) has NO authoritative vendor BF16-dense figure — NVIDIA
publishes a sparse FP4 number (~1 PFLOP). Its entry is flagged estimated=True and
compute_mfu surfaces that caveat so a GB10 MFU is never quoted as if exact.
"""
from __future__ import annotations

import argparse
import json
import sys

from flop_accounting import flops_per_token

# device -> BF16 DENSE tensor-core peak in TFLOP/s, with provenance (§C2).
# Verified 2026-06-15 against the cited vendor spec sheets.
PEAK_FLOPS = {
    "a100-sxm": {"bf16_dense_tflops": 312.0, "estimated": False,
                 "url": "https://www.nvidia.com/en-us/data-center/a100/",
                 "note": "A100 80GB SXM, BF16 dense (624 w/ sparsity)"},
    "h100-sxm": {"bf16_dense_tflops": 989.5, "estimated": False,
                 "url": "https://resources.nvidia.com/en-us-tensor-core",
                 "note": "H100 SXM, BF16 dense (1979 w/ sparsity)"},
    "h200-sxm": {"bf16_dense_tflops": 989.5, "estimated": False,
                 "url": "https://www.nvidia.com/en-us/data-center/h200/",
                 "note": "H200 = H100 compute, more/faster memory"},
    "b200": {"bf16_dense_tflops": 2250.0, "estimated": False,
             "url": "https://www.nvidia.com/en-us/data-center/dgx-b200/",
             "note": "Blackwell B200, BF16 dense (~4.5 PFLOPS w/ sparsity)"},
    "tpu-v5e": {"bf16_dense_tflops": 197.0, "estimated": False,
                "url": "https://cloud.google.com/tpu/docs/v5e",
                "note": "TPU v5e, bf16 per chip"},
    "gb10": {"bf16_dense_tflops": 125.0, "estimated": True,
             "url": "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
             "note": "ESTIMATED: NVIDIA publishes only ~1 PFLOP sparse FP4 for "
                     "GB10; BF16-dense derived (~FP4/8) — treat MFU as approximate"},
}

# Activation-recompute multiplier on the model-FLOP count for HFU. Full
# checkpointing recomputes the forward (~6N -> ~8N hardware FLOPs/token).
HFU_RECOMPUTE_FACTOR = 8.0 / 6.0


def device_peak_tflops(device: str) -> dict:
    key = device.strip().lower()
    if key not in PEAK_FLOPS:
        raise ValueError(
            f"unknown device {device!r}; add it to PEAK_FLOPS with a vendor "
            f"source (§C2) rather than guessing. known: {sorted(PEAK_FLOPS)}")
    return PEAK_FLOPS[key]


def compute_mfu(cfg: dict, tokens_per_sec: float, device: str,
                act_ckpt: bool = False) -> dict:
    """cfg keys: n_params, n_layers, n_heads, head_dim, seq_len.
    Returns model/hardware FLOPs utilisation + provenance. Raises on unknown
    device (never invents a denominator)."""
    if tokens_per_sec <= 0:
        raise ValueError(f"tokens_per_sec must be positive, got {tokens_per_sec!r}")
    fpt = flops_per_token(cfg["n_params"], cfg["n_layers"], cfg["n_heads"],
                          cfg["head_dim"], cfg["seq_len"])
    peak = device_peak_tflops(device)
    peak_flops = peak["bf16_dense_tflops"] * 1e12
    achieved = fpt * tokens_per_sec
    mfu = achieved / peak_flops
    hardware_achieved = achieved * (HFU_RECOMPUTE_FACTOR if act_ckpt else 1.0)
    hfu = hardware_achieved / peak_flops
    return {
        "mfu": mfu,
        "hfu": hfu,
        "achieved_tflops": achieved / 1e12,
        "model_flops_per_token": fpt,
        "device": device.strip().lower(),
        "device_peak_tflops": peak["bf16_dense_tflops"],
        "peak_is_estimated": peak["estimated"],
        "peak_source": peak["url"],
        "act_ckpt": act_ckpt,
        "formula": "MFU = (6N + 12*L*H*Q*T) * tok/s / peak_bf16_dense",
        "caveat": (peak["note"] if peak["estimated"] else None),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-params", type=float, required=True)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--tokens-per-sec", type=float, required=True)
    ap.add_argument("--device", required=True, help=f"one of {sorted(PEAK_FLOPS)}")
    ap.add_argument("--act-ckpt", action="store_true",
                    help="activation/gradient checkpointing was on (affects HFU)")
    a = ap.parse_args(argv)
    cfg = {"n_params": a.n_params, "n_layers": a.n_layers, "n_heads": a.n_heads,
           "head_dim": a.head_dim, "seq_len": a.seq_len}
    try:
        print(json.dumps(compute_mfu(cfg, a.tokens_per_sec, a.device, a.act_ckpt),
                         indent=2))
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
