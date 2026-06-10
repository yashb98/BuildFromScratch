"""
safe_cuda — import this BEFORE allocating any CUDA memory in EVERY PyTorch
script on this box.

    import safe_cuda  # noqa: F401  (first project import)
    import torch
    safe_cuda.guard()  # call once, after `import torch`, before using the GPU

WHY THIS EXISTS
---------------
This machine is an NVIDIA GB10 (Grace Blackwell) with UNIFIED memory: the CPU
and GPU share a single ~119 GB LPDDR5X pool — there is NO separate VRAM.

When a CUDA process tries to allocate more than the pool can give, one of two
things happens:
  * CUDA returns a catchable `torch.OutOfMemoryError` (the good case), or
  * the kernel's GLOBAL OOM-killer fires first and kills other processes —
    the desktop, journald, the editor — and the machine hard-reboots.

Which one you get depends on system-wide memory pressure at the moment of the
allocation. On 2026-06-08 a throughput probe (micro_batch=16, seq_len=4096,
vocab=151936 -> a single ~40 GB fp32 logits tensor) hit the second case and
crashed the box.

WHAT THIS DOES
--------------
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to cut fragmentation
  (must be set before the CUDA caching allocator initializes).
- `guard(fraction)` caps this process at `fraction` of the unified pool via
  `torch.cuda.set_per_process_memory_fraction`, so an over-allocation raises a
  clean `torch.OutOfMemoryError` instead of taking down the desktop. Default
  0.85 leaves ~18 GB for the OS/editor/browser.

Use a LOWER fraction (e.g. 0.6) while you still have VS Code + a browser open;
raise toward 0.9 only for a dedicated run with everything else closed.
"""
import os

# Must be set before the caching allocator is first touched.
_prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _prev:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        (_prev + ",") if _prev else ""
    ) + "expandable_segments:True"


def guard(fraction: float = 0.85, device: int = 0) -> None:
    """Cap this process at `fraction` of the unified pool. Call after import torch."""
    import torch

    if not torch.cuda.is_available():
        return
    if not (0.0 < fraction <= 0.95):
        raise ValueError(
            f"fraction={fraction} unsafe on a unified-memory box; use 0.5–0.9."
        )
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(
        f"[safe_cuda] capped CUDA at {fraction:.0%} of {total:.0f} GB unified "
        f"pool (~{fraction * total:.0f} GB); over-allocation now errors cleanly."
    )
