# HybridSSM-0.2B — build status (updated 2026-07-20 04:15 UTC)

**Novel from-scratch hybrid attention-SSM LM in JAX/Flax is BUILT, VERIFIED, SMOKE-PASSED — and the
first pretrain arm (`ssm_base_s0`) is IN FLIGHT on real data.**

## Build phase — done ✓ (2026-07-19, all correctness-gated)

- **Architecture** (`ARCHITECTURE.md`): d=768, 24 layers (1:1 full-attn:efficient interleave), GQA 12/4,
  SwiGLU, RMSNorm, RoPE↔NoPE toggle, Qwen3 152k tokenizer, chunked CE. Design doc estimated ~146M
  non-embed; the built model reports **189.1M non-embed / 305.8M total** (`[build]` line of every run
  log) — tied embedding = 151,936 × 768 = 116.7M counted once.
- **Implementation** (`ssm.py`, `model.py`): SelectiveSSM (Mamba-2-style diagonal scan via
  `associative_scan`) + SlidingWindowAttention + GQA attention + the full hybrid decoder. Written from blank.
- **Verify gate** (`verify.py` → **PASS**): SSM parallel-scan == sequential reference (max|Δ|=2.4e-7);
  chunked CE == naive CE (|Δ|=4.8e-5, never materializes the 152k logits); param count sane; all 8
  ablation toggles forward-finite; forward deterministic. ⚠️ **See "Open gate gap" below — this PASS
  predates the `nn.remat` memory fix and has not been re-run since.**
- **Smoke** (`train.py --smoke` → **PASS**, all variants): SSM / SWA-128+NoPE / 1:3-attention each overfit
  a fixed batch 8.8 → ~0.003 loss (forward+backward+AdamW+chunked-CE all learn), grad norms healthy
  (33 → 0.03), checkpoint save→reload exact (max|Δ|=0.0 — recovery-chain ready). Smoke used synthetic data.
- **Fit probes on real data** (`probe.log` / `probe2.log` / `probe3.log`, 15 / 12 / 30 steps):
  step-0 loss 12.4317 / 12.4312 / 12.4312 ≈ ln(151936)=11.93 + init noise, and 30 steps moves 12.43 → 8.42.

## Pretrain arm `ssm_base_s0` — COMPLETE (2026-07-20 15:04 UTC)

Exited cleanly on its own: `[done] 21156 steps · final loss=3.8149`, `arm_ssm_base_s0.done`
written, final checkpoint saved, and the sentinel disarmed itself (`watched pid 3164922
exited on its own; disarming (no kill)`). Wall clock **990 min / 16.5 h** for the final
process — this excludes the killed first attempt, whose start time is not on disk, so
total GPU time is a lower bound.

| result | value |
|---|---|
| final train loss | **3.8149** |
| best val loss | **3.7839** @ step 19,200 (final eval 3.9245 @ 20,800 — noisy tail) |
| eval-harness `text-lm-v2` | PPL wikitext2_val **133.4628**, code_py **5142.6426** (`self_floor=true`; corpora pinned `wikitext-2-raw-v1:validation@b08601e`, `codeparrot-clean-valid@4db92d2`) |
| verdict | **directional** — n=1 seed, no comparand, no iso-FLOP match (§C17/§C18/§C25) |

**Verify gate CLOSED.** `verify.py` was re-run 2026-07-20 17:46 against the post-`nn.remat`
`model.py` (last modified 07-19 22:27) — `verify.log`, 6/6 PASS, exit 0: scan-vs-reference
max|Δ|=2.38e-07, chunked-vs-naive CE |Δ|=4.77e-05, param count, forward finite/deterministic,
all 8 toggle combos finite.

### ⚠️ Budget overshoot — a resume bug, and it matters for the ladder

`train_hybrid.py:129` is `for s in range(start_step, start_step + steps)`. A **resumed** run
therefore repeats the FULL step budget from the resume point instead of finishing the
original one. This arm resumed at step 400, so it ran **21,156 steps = 173,309,952 tokens
against a declared budget of 170,034,304 (+1.93%)**, wrapping ~1.9% into a second epoch.

Harmless at n=1, but it silently breaks **§C18 iso-FLOP** across arms: any arm that crashes
and resumes gets *more* compute than one that doesn't, scaling with the resume point — a
resume at step 5,000 would be **+24%**, far past the 5% tolerance, and nothing in the logs
would flag it. **Fix to `range(start_step, steps)` before running the ladder.**

## Pretrain arm `ssm_base_s0` — configuration as launched

Ledger run `2026-07-19_hybrid-ssm-0.2b_pretrain-ssm-base-s0` (type=ablation, status=running,
lifecycle_stage=architecture, framework=jax, technique `hybrid-attention-rethink`).

| field | value | source |
|---|---|---|
| data | FineWeb-Edu sample-10BT, Qwen3-0.6B-Base tokenizer, **170,034,304 train + 300,000 val** tokens, seed 0 | `tokcache_170034304_300000_seed0_Qwen3-0.6B-Base.pt`, built by `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:151` |
| config as launched | seq **2048**, batch **4**, 20,756 steps × 8,192 tok/step, AdamW lr 3e-3, warmup 200 | live cmdline of PID 3164922 |
| trainer / watchdog | PID 3164922 (`train_hybrid.py`) · sentinel PID 3167084 | `pgrep`, `sentinel.log` |
| progress @ 04:15Z | **step 7,480 / 20,756 (36.0%)** — 61.3M of 170.0M tokens | `run_ssm_base_s0.log` |
| loss | step-0 12.4317 → train ~4.88; val 6.6844@400 → 6.2845@1200 → **4.9020@7200** (best) | `run_ssm_base_s0.log` |
| grad norm | 0.28–0.31, stable | `run_ssm_base_s0.log` |
| throughput | ~1,247 steps/h ≈ **2,837 tok/s** (measured over 7,080 steps / 5.68 h since resume) | derived from log + process start |
| ETA | ≈ **2026-07-20 14:54 UTC** (13,276 steps remaining) | same |
| memory | pool 37–41%, rss 11.6 GiB, GPU 66–69 °C / SoC 72–74 °C | `sentinel.log` heartbeats |

**Deviations from `ARCHITECTURE.md`, recorded honestly:** the design doc specifies seq_len 4096 and
Muon(2D)+AdamW(1D); this arm runs **seq 2048 with plain AdamW**. The JAX Muon port (`muon_jax.py`) is not
written yet — AdamW-vs-Muon is itself a planned arm, and every arm in the ladder must use the same
optimizer for the comparison to hold, so the ladder's baseline optimizer is now AdamW unless re-based.

### Incident + recovery (the run survived a real kill)

First launch was killed by the sentinel at **step 580, 2026-07-19T16:58:48Z** — pool usage 81.3% ≥ the
0.80 kill line (MemAvailable 22.4 GiB / 119.7 GiB; SSM scan + chunked CE under autodiff held ~61 GB);
GPU 58 °C, no thermal component (`sentinel_kill_step580_2026-07-19.json`). Fix: **`nn.remat` on the
decoder block** (`model.py:129`, `BlockR = nn.remat(Block)`) + batch 8 → 4 → allocation 61.5 GB → 16.6 GB,
pool 81% → ~40%. Resumed from the step-400 checkpoint at 22:34:29 and has run clean since.
This was a *manual* recovery behind a config change, i.e. the §C5/S1-4a "not safe to auto-resume at the
same config" path — `loop_state.auto_resumes` correctly stayed at 0.

## Gate gap — CLOSED 2026-07-20

For the record, since it was flagged as blocking while the arm ran: `verify.py`'s original PASS
(2026-07-19 12:52) predated the 22:27 `nn.remat` change, so for the whole run the gate was stale
against the model actually training. It was re-run **2026-07-20 17:46**, after the arm finished
(GPU work — §C4.5 forbids co-running it beside a live trainer), and captured to `verify.log`:
**6/6 PASS, exit 0**. `nn.remat` is confirmed value-preserving here, as expected.

## Next

1. **Close the verify gap** (above) + write `verify.log`, so the artifact set is self-evidencing.
2. **Score the finished arm** via `/eval-harness` (BPB on wikitext-2 + code, text-lm-v2, `suite_version`
   stamped) → write `verdict.json`. A single arm is a *baseline datum*, not a win: no cross-arm claim
   exists until the ladder has ≥3 seeds and iso-FLOP-matched comparands (§C17/§C18), so the terminal
   verdict for this arm caps at `directional` at best (§C25).
3. **The emergence-speed ladder** (the study): mixer type × attention fraction × NoPE, each scored at
   increasing token budgets → does the efficient-mixer choice affect emergence SPEED but converge? Plus
   the NoPE-on-full-attn validation. Multi-day; drive arms as they complete.
4. **Recovery chain**: the user still pastes the `@reboot bash research/boot_resume.sh` cron line
   (§C4.2 — the agent never auto-installs cron).
5. **Downstream lifecycle** (long-context retrieval probe → data / mid-training / SFT), each to a §C25
   terminal verdict — the whole-lifecycle finish line for the new model.
