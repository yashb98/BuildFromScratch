# text-lm-v3 downstream battery — comparison across all 25 checkpoints

_Scored 2026-06-24 via `research/run_v3_downstream.py`; LAMBADA last-token acc + per-task **BPB-on-gold** (the discriminator at near-chance scale, §C25.6). MC accuracy (ARC/HellaSwag/WinoGrande) is near-chance and omitted as no-signal._

## Builds (1.19B tok) — downstream confirms the PPL ordering

| Build | LAMBADA acc ↑ | mean BPB-gold ↓ |
|---|---|---|
| IMU-1 (modernized) | 0.212 | 1.142 |
| faithful baseline | 0.170 | 1.188 |
| partial-RoPE 0.25 | 0.166 | 1.202 |
| partial-RoPE 0.10 (abandoned) | 0.088 | 1.451 |

## Phase 1 de-confound (131M proxy, mean/3 seeds) — arch is the driver

| Axis | LAMBADA ↑ | BPB-gold ↓ |
|---|---|---|
| +arch | 0.142 | 1.338 |
| +WSD | 0.050 | 1.508 |
| +z-loss | 0.032 | 1.534 |
| baseline | 0.035 | 1.534 |

## Phase 2 arch sub-drill (131M proxy, mean/3 seeds) — all 3 beat baseline

| Flag | LAMBADA ↑ | BPB-gold ↓ |
|---|---|---|
| vr (value-residual) | 0.073 | 1.459 |
| ln (layernorm-scaling) | 0.067 | 1.479 |
| hg (head-gating) | 0.058 | 1.502 |
| baseline (Phase-1) | 0.035 | 1.534 |

**Findings:** the downstream battery independently confirms every prior attribution — IMU-1 > faithful > pRoPE on LAMBADA+BPB-gold (matching PPL); arch the Phase-1 driver; vr<ln<hg all below baseline (matching the canonical BPB verdict). MC accuracy is near-chance at this scale (§C25.6); LAMBADA + BPB-on-gold are the discriminators.
