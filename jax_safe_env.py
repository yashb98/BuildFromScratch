"""
jax_safe_env — import this BEFORE `import jax` in EVERY JAX script on this box.

    import jax_safe_env  # noqa: F401  (must be the first project import)
    import jax

WHY THIS EXISTS
---------------
This machine is an NVIDIA GB10 (Grace Blackwell) with UNIFIED memory: the CPU
and GPU share a single ~119 GB LPDDR5X pool — there is NO separate VRAM.

JAX/XLA defaults to preallocating ~75% of "GPU memory" at startup. On a unified
box that is 75% of *system RAM* (~90 GB) reserved in one shot, through the
NVIDIA system-memory allocator. Stacked on the OS + editor + browser it blows
past 119 GB + swap, the kernel OOM-killer fires, kills critical services and the
desktop session, and the machine hard-crashes and reboots.

This happened on 2026-06-08. Root cause confirmed from the kernel OOM dump:
`global_oom` (CONSTRAINT_NONE) while the nvidia driver was in
`osAllocPagesInternal -> memdescAlloc -> sysmemConstruct_IMPL`.

WHAT THIS DOES
--------------
- PREALLOCATE=false : allocate on demand instead of grabbing 75% up front.
- MEM_FRACTION=0.5  : even on demand, never exceed 50% of unified RAM (~60 GB),
                      always leaving headroom for the OS/editor/browser.

These only take effect if set BEFORE the JAX backend initializes, which is why
this module must be imported before `import jax`. If the vars are already set
(e.g. by ~/.bashrc), they are respected and not overridden.
"""
import os

# Never override a value the user set deliberately for a dedicated run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

# Hard safety: refuse the one setting that reliably crashes this box.
if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() == "true":
    raise RuntimeError(
        "XLA_PYTHON_CLIENT_PREALLOCATE=true will preallocate ~75% of UNIFIED "
        "memory on this GB10 box and crash the machine. Set it to 'false'."
    )
