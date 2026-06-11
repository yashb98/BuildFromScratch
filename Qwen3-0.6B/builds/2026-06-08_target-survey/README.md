# Target survey — Qwen ~0.5B-class model selection

**What this folder is:** the **model-selection phase** (not a build). It captures
the decision of *which* model to reproduce before any architecture/training work
started. The whole deliverable is one plan doc: [`model_target_plan.md`](./model_target_plan.md).

- **Date:** 2026-06-08
- **Recency cutoff (≤6 mo):** 2025-12-08
- **Decision:** **Qwen3-0.6B** (`Qwen/Qwen3-0.6B-Base`) for all three planned builds.
- **Method:** survey 4 candidate Qwen models in the ~0.5B band; figures pulled
  at runtime from each repo's live `config.json` via `AutoConfig.from_pretrained`
  (not hand-quoted).

For the shared architecture, parameter accounting, and current results, see the
parent README at [`../../README.md`](../../README.md). This README is scoped to the
*selection decision* only.

---

## Files

| File | What it is |
|---|---|
| [`model_target_plan.md`](./model_target_plan.md) | The survey + decision: 4-candidate comparison table, per-model write-ups (recommended / runner-up / skipped / out-of-scope), three-build slot allocation, recommendation, and open questions for the user. |
| `README.md` | This file. |

There is no code to run in this folder — it is a planning artifact. The runnable
builds it allocates live in the sibling folders (see [slot allocation](#three-build-slot-allocation) below).

---

## Why Qwen3-0.6B

Selection criteria from the plan: **latest Qwen**, **~0.5B (sub-billion)**,
**text-only decoder**, **open weights / Apache 2.0**, and a **high-quality
community PyTorch reference** to triangulate verify-gate failures against.

| Candidate | Family | Params (text-only) | Repro difficulty | Verdict |
|---|---|---|---|---|
| **Qwen3-0.6B** | qwen3 | **596M** | **LOW** | **Chosen** — newest text-only Qwen; clean decoder; Raschka from-scratch reference exists |
| Qwen2.5-0.5B | qwen2 | ~494M | LOW | Runner-up — well-trodden, but older (Sep 2024); fails "latest Qwen" |
| Qwen2-0.5B | qwen2 | ~494M | LOW | Skipped — same arch as 2.5, superseded; no reason to pick |
| ~~Qwen3.5-0.8B~~ | qwen3_5 (multimodal) | ~0.9B incl. ViT | OUT OF SCOPE | Excluded — vision tower + hybrid linear/full attn + MTP + MRoPE; not a text-only decoder |

Key reasons Qwen3-0.6B won (per `model_target_plan.md`):

- **Newest** text-only Qwen in the band — Qwen2/2.5 are older; the latest "0.8B"
  (Qwen3.5) is multimodal and not a clean transformer.
- **Clean text decoder:** 28 layers × 1024 hidden, GQA 16 query / 8 KV heads
  (2:1), head_dim 128, SwiGLU FFN (intermediate 3072), RMSNorm pre-norm
  (eps 1e-6), RoPE θ=1e6, max ctx 40,960, tied embeddings, vocab 151,936.
- **Stable open base checkpoint** (`Qwen/Qwen3-0.6B-Base`) → clean base-PPL eval
  with no thinking-mode prompt tags.
- **Community reference** ([Raschka — *Understanding and Implementing Qwen3 From
  Scratch*](https://magazine.sebastianraschka.com/p/qwen3-from-scratch), `rasbt/qwen3-from-scratch`)
  to triangulate against on verify-gate failures.
- **Apache 2.0** license.
- **Backing paper:** Qwen3 Technical Report — [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) (May 2025).

**Why each alternative was rejected:** Qwen2.5-0.5B is the runner-up but is older
(fails the "latest Qwen" criterion). Qwen2-0.5B shares the Qwen2 arch with 2.5 and
is superseded — no reason to pick it. Qwen3.5-0.8B is multimodal (separate ViT,
hybrid linear/full attention, MRoPE, MTP head) — reproducing it from scratch
three times is out of scope; it was flagged so the "pick the newest Qwen"
instinct didn't drift the project into an SSM/multimodal reproduction.

---

## Three-build slot allocation

The survey allocated three sequential builds, all on Qwen3-0.6B, each in its own
sibling folder:

| # | Folder | Mode | What's varied |
|---|---|---|---|
| 1 | [`../2026-06-08_reproduce-faithful_qwen3-0.6b/`](../2026-06-08_reproduce-faithful_qwen3-0.6b/) | Faithful | Match HF config exactly; published-paper recipe; verify gate <1e-3 max-err vs HF weights |
| 2 | [`../2026-06-08_reproduce-modernized_qwen3-0.6b/`](../2026-06-08_reproduce-modernized_qwen3-0.6b/) | Modernized | Same arch; current best-practice recipe (fineweb-edu, modern WSD); verify gate on unchanged components |
| 3 | [`../2026-06-08_reproduce-exploratory_qwen3-0.6b/`](../2026-06-08_reproduce-exploratory_qwen3-0.6b/) | Exploratory | Arch as start; swap 1–2 components (candidates: NoPE vs RoPE, GQA→MQA, RMSNorm→DyT, …); smoke-test only |

Each build gets its own architecture / training / hp-tuning plan, train script,
eval, and notebooks.

---

## Open questions raised (for the user, at survey time)

The plan left three decisions open (see `model_target_plan.md` §"Open questions"):

1. **Base vs Instruct** — recommended reproducing the **base** checkpoint
   (`Qwen/Qwen3-0.6B-Base`), not instruct, for a clean base-PPL signal.
2. **Verify tolerance** — proposed default **<1e-3** max-abs-err in fp32 for the
   faithful build.
3. **Exploratory swap (Build 3)** — component-swap options (RoPE→NoPE, GQA→MQA,
   RMSNorm→DyT/SLA, attention-bias toggle) to be proposed in a later phase.

---

## Status & gotchas

- This is a **PRELIMINARY planning artifact** — a decision record, not a result.
  Nothing here is benchmarked. The actual reproduction outcomes live in the sibling
  build folders and the [parent README](../../README.md).
- **Numbers vs the live config:** the plan states all spec figures were pulled at
  runtime via `AutoConfig.from_pretrained` (live `config.json`), not hand-quoted.
- **Reported-quality numbers were deliberately not pinned:** the plan notes
  specific MMLU/GSM8K/HumanEval figures "vary by source and post-training stage,"
  so the reproduction compares via **PPL deltas** against the open base weights
  instead.
- **Survey-time gotchas flagged in the plan** (carried into the builds): RoPE
  θ=1e6 (not the GPT-style 1e4 default), QK-norm applied per-head inside Qwen3
  attention (new vs Qwen2), tied embeddings with no separate lm_head bias, and a
  common vocab_size pad mismatch.
