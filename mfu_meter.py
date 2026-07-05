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

MEASURED ROOFLINE (§C-systems, Frontier-Eng "real measured MFU"): a vendor
datasheet peak is the theoretical ceiling; the number an engineer actually wants
in the denominator is the device's *achievable* BF16-dense peak measured on the
box (a large square GEMM microbenchmark, the empirical roofline). Pass a
``measured_peak_tflops`` to compute_mfu (or device="measured" with it) to use that
in place of the datasheet figure: peak_is_estimated becomes False and the
estimated-device caveat is DROPPED, because the denominator is no longer derived
— it was observed. Discipline is preserved: a measured peak must STILL be a
dense, no-sparsity, no-FP4 BF16 GEMM number (record_measured_peak refuses to
mark a sparse/FP4 measurement as a dense peak). Use record_measured_peak(...) to
capture the provenance (how/where it was measured) for the ledger.
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


# Methods we accept as a real, dense, no-sparsity BF16 roofline measurement.
# A measured peak MUST come from one of these (record_measured_peak enforces it);
# a sparse / FP4 / structured-2:4 number is NOT a dense peak and is refused.
_MEASURED_METHODS = {"gemm_microbench", "torch_profiler_gemm", "cublas_lt_bench",
                     "nsight_roofline", "vendor_dense_measured"}


def record_measured_peak(measured_peak_tflops: float, *, device: str,
                         method: str, dtype: str = "bf16",
                         sparsity: str = "dense",
                         gemm_shape=None, command: str = None,
                         measured_on: str = None, note: str = None) -> dict:
    """Capture the PROVENANCE of an empirically-measured BF16-DENSE peak so the
    ledger can show how the denominator was obtained (Frontier-Eng "real
    measured MFU"). Returns a dict suitable to pass to compute_mfu as
    ``measured_peak``.

    Discipline (§C-systems): a measured peak is honest only if it is a dense,
    no-sparsity BF16 GEMM number. This REFUSES to record a sparse/FP4 figure as a
    dense peak, and refuses a non-positive value or an unrecognised method — the
    whole point of "measured" is that it is auditable, not asserted.
    """
    if measured_peak_tflops <= 0:
        raise ValueError(
            f"measured_peak_tflops must be positive, got {measured_peak_tflops!r}")
    if sparsity.strip().lower() != "dense":
        raise ValueError(
            f"refusing to record sparsity={sparsity!r}: a measured PEAK must be "
            f"BF16 DENSE (no 2:4/structured sparsity). Re-measure dense.")
    if dtype.strip().lower() not in ("bf16", "bfloat16"):
        raise ValueError(
            f"refusing dtype={dtype!r}: the MFU denominator is BF16-dense; an "
            f"FP4/FP8/TF32 number inflates the peak and deflates MFU dishonestly.")
    if method not in _MEASURED_METHODS:
        raise ValueError(
            f"unknown measurement method {method!r}; use one of "
            f"{sorted(_MEASURED_METHODS)} so the roofline is reproducible.")
    return {
        "bf16_dense_tflops": float(measured_peak_tflops),
        "estimated": False,          # measured, not derived
        "measured": True,
        "device": device.strip().lower(),
        "method": method,
        "dtype": "bf16",
        "sparsity": "dense",
        "gemm_shape": tuple(gemm_shape) if gemm_shape is not None else None,
        "command": command,
        "measured_on": measured_on,
        "url": None,                 # provenance is the measurement, not a datasheet
        "note": note or (f"MEASURED bf16-dense peak via {method}"
                         + (f" on {measured_on}" if measured_on else "")),
    }


def compute_mfu(cfg: dict, tokens_per_sec: float, device: str,
                act_ckpt: bool = False,
                measured_peak_tflops: float = None,
                measured_peak: dict = None) -> dict:
    """cfg keys: n_params, n_layers, n_heads, head_dim, seq_len.
    Returns model/hardware FLOPs utilisation + provenance. Raises on unknown
    device (never invents a denominator).

    Measured roofline: pass ``measured_peak_tflops`` (a scalar) and/or
    ``measured_peak`` (a record_measured_peak dict) to use a REAL measured
    BF16-dense peak in the denominator. When given, peak_is_estimated=False and
    the estimated-device caveat is dropped — the denominator was observed, not
    derived. device="measured" is also accepted as an explicit override (it then
    REQUIRES a measured peak). A datasheet device + a measured peak uses the
    measured value but keeps the datasheet device name for context."""
    if tokens_per_sec <= 0:
        raise ValueError(f"tokens_per_sec must be positive, got {tokens_per_sec!r}")
    fpt = flops_per_token(cfg["n_params"], cfg["n_layers"], cfg["n_heads"],
                          cfg["head_dim"], cfg["seq_len"])
    dev_key = device.strip().lower()

    # Resolve a measured peak from either the scalar or the provenance dict.
    if measured_peak is not None:
        if measured_peak.get("sparsity", "dense") != "dense" or \
                measured_peak.get("estimated", False):
            raise ValueError(
                "measured_peak must be a dense, non-estimated record "
                "(use record_measured_peak()).")
        mp_val = measured_peak["bf16_dense_tflops"]
        if measured_peak_tflops is not None and \
                abs(measured_peak_tflops - mp_val) > 1e-9:
            raise ValueError(
                f"measured_peak_tflops ({measured_peak_tflops}) disagrees with "
                f"measured_peak record ({mp_val}); pass one source of truth.")
        measured_peak_tflops = mp_val

    if measured_peak_tflops is not None and measured_peak_tflops <= 0:
        raise ValueError(
            f"measured_peak_tflops must be positive, got {measured_peak_tflops!r}")

    if measured_peak_tflops is not None:
        # MEASURED path: observed dense bf16 peak, no caveat, never estimated.
        peak_val = float(measured_peak_tflops)
        peak_is_estimated = False
        peak_source = (measured_peak.get("note") if measured_peak
                       else "measured bf16-dense roofline")
        caveat = None
    elif dev_key == "measured":
        raise ValueError(
            "device='measured' requires a measured_peak_tflops (or measured_peak "
            "record); never invent a denominator for an unmeasured device (§C2).")
    else:
        # DATASHEET path: pinned vendor peak; GB10 stays estimated with caveat.
        peak = device_peak_tflops(device)
        peak_val = peak["bf16_dense_tflops"]
        peak_is_estimated = peak["estimated"]
        peak_source = peak["url"]
        caveat = peak["note"] if peak["estimated"] else None

    peak_flops = peak_val * 1e12
    achieved = fpt * tokens_per_sec
    mfu = achieved / peak_flops
    hardware_achieved = achieved * (HFU_RECOMPUTE_FACTOR if act_ckpt else 1.0)
    hfu = hardware_achieved / peak_flops
    return {
        "mfu": mfu,
        "hfu": hfu,
        "achieved_tflops": achieved / 1e12,
        "model_flops_per_token": fpt,
        "device": dev_key,
        "device_peak_tflops": peak_val,
        "peak_is_estimated": peak_is_estimated,
        "peak_is_measured": measured_peak_tflops is not None,
        "peak_source": peak_source,
        "act_ckpt": act_ckpt,
        "formula": "MFU = (6N + 12*L*H*Q*T) * tok/s / peak_bf16_dense",
        "caveat": caveat,
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
    ap.add_argument("--device", required=True,
                    help=f"one of {sorted(PEAK_FLOPS)}, or 'measured' "
                         f"(requires --measured-peak-tflops)")
    ap.add_argument("--act-ckpt", action="store_true",
                    help="activation/gradient checkpointing was on (affects HFU)")
    ap.add_argument("--measured-peak-tflops", type=float, default=None,
                    help="REAL measured BF16-dense peak (TFLOP/s); drops the "
                         "estimated caveat and uses the observed roofline")
    ap.add_argument("--measured-method", default=None,
                    help=f"how it was measured, one of {sorted(_MEASURED_METHODS)} "
                         f"(records provenance when --measured-peak-tflops is set)")
    a = ap.parse_args(argv)
    cfg = {"n_params": a.n_params, "n_layers": a.n_layers, "n_heads": a.n_heads,
           "head_dim": a.head_dim, "seq_len": a.seq_len}
    try:
        mp = None
        if a.measured_peak_tflops is not None and a.measured_method is not None:
            mp = record_measured_peak(a.measured_peak_tflops, device=a.device,
                                      method=a.measured_method)
        print(json.dumps(compute_mfu(cfg, a.tokens_per_sec, a.device, a.act_ckpt,
                                     measured_peak_tflops=a.measured_peak_tflops,
                                     measured_peak=mp),
                         indent=2))
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
