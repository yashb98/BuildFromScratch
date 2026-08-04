# Model-card fact sheet — provenance for every published number

**Purpose.** Before any BuildFromScratch result is published as a HuggingFace model card,
every number on that card must be attached to the eval that produced it: dataset id, config,
split, sequence length, stride, tokenizer, and the file on disk that recorded it. This document
is that attachment. It answers a reviewer's questions directly and, where the repo cannot answer,
says so instead of guessing.

**Status: this is a provenance audit, not a model card.** It is deliberately unflattering.
§9 lists 20 things that would embarrass the author if the current READMEs were published as-is —
including two claims in the repo's own READMEs that are demonstrably false. Fix those before
writing the card, not after a reviewer finds them.

**Produced** 2026-08-04 by an 8-dimension parallel file audit (one agent per dimension) followed
by an adversarial verification pass over every extracted fact (a second agent per dimension whose
instructions were to *refute*, defaulting to WRONG/NEEDS_QUALIFIER under uncertainty), then a
synthesis pass. 17 agents, 715 tool calls. Facts the adversarial pass overturned are marked
**[CORRECTED]**; facts needing a caveat are marked **[QUALIFIER]**.

**Scope limit.** Every claim here is traced to a file:line that was read, but this is an audit of
*what the repo records*, not a re-execution of the experiments. Where a recorded number could not
be re-derived from disk, §8 says so.

**Underlying data.** This file is the synthesis. The complete raw dataset — all 160 extracted facts
with their verbatim source quotes, all 166 adversarial verdicts, and all 57 gaps, nothing summarised
away — is in [`MODEL_CARD_FACTS_RAW.md`](MODEL_CARD_FACTS_RAW.md). The refute pass overturned 9 facts
outright and attached a qualifier to 38 more, so **47 of 166 checks caught something that would have
been misleading if published as first extracted** — which is the reason that file is worth keeping.

---

## Independent spot-check of the three load-bearing findings

The three findings below were re-verified by hand, outside the agent pipeline, because each one
contradicts something the repo currently asserts. All three reproduce.

**1. The Qwen3 perplexities are NOT on a common val slice — the README's comparability claim is false.**

`Qwen3-0.6B/README.md:35-37` states: *"All perplexities use **identical eval code on the identical
300k-token FineWeb-Edu val slice** … so every row is directly comparable."* Verified false:

| Run | Number | Val cache actually used |
|---|---|---|
| `eval_original_vs_repro.py` (released model + LR sweep) | **13.40**, **46.31** | `tokcache_133072000_300000.pt` — hardcoded at `eval_original_vs_repro.py:22` |
| `qwen3_baseline2tpp` (faithful, 1.19B tok) | **28.65** | `tokcache_1191478400_300000.pt` — streamed at `qwen3_baseline2tpp_train.log:3,6` |
| `qwen3_imu1_2tpp` (modernized, 1.19B tok) | **23.52** | `tokcache_1191478400_300000.pt` — `qwen3_imu1_2tpp_train.log` |

Both caches exist on disk with different sizes (1,066,978,101 B vs 9,534,229,373 B). So
**28.65/13.40 and 23.52/13.40 are cross-slice ratios** and must not be published as like-for-like.
Only 46.31/13.40 is same-slice.

**2. `13.40` is our own measurement, not a borrowed figure.**

`Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_vs_repro.txt`, verbatim:

```
[2026-06-09 16:51:36] Original vs reproduction — val=300,000 tokens, 50 windows x 4096
ORIGINAL  Qwen3-0.6B-Base (36T tok)   val PPL =   13.400  (204,800 tok, 21s)
```

The `21s` wall-clock and the `[safe_cuda] capped CUDA at 85% of 129 GB unified pool` banner in
`results/original_eval_run2.log:2` are execution evidence: the released checkpoint was downloaded
and scored on this box by `eval_original_vs_repro.py`, using this repo's own `eval_ppl`. The only
borrowed element in that string is the `36T tok` label, transcribed from the Qwen3 tech report.
**State this split explicitly on the card** — a reviewer cares a great deal which it is.

**3. The `−0.474 bpb` NorMuon win is advertised as a `significant win` but has been nulled at scale.**

`Qwen3-0.6B/README.md:52` still reads *"**NorMuon > AdamW** | wikitext −0.474 bpb [0.444, 0.505] ·
code −0.502 [0.456, 0.547] | **significant win**"*. The scaling-persistence ladder that closed
2026-07-28 says otherwise —
`Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/verdict.json`:

```
trend_verdict          CONVERGES
ledger_verdict         null
significance_verdict   null
rationale              gap shrinks toward 0 with scale and falls within the noise floor at the
                       largest budget — an early-training speedup that converges away (no
                       advantage at scale)
```

−0.474 is real *at a 42M-token budget*. It is not a standing result. Four published surfaces still
sell it as a win (§3).

---

## Live environment stamp — measured 2026-08-04 on the box

Recorded here because the repo stamps versions in exactly one place and bit-exactness is
version-sensitive. This is the current box, which is **not** necessarily the box that produced the
2026-05/06 artifacts (§4.3 covers that gap).

| | Value | Source |
|---|---|---|
| Python | `3.12.11` (conda-forge, GCC 13.3.0) | `sys.version` |
| torch | `2.11.0+cu130` | `torch.__version__` |
| CUDA (torch) | `13.0` | `torch.version.cuda` |
| CUDA (nvcc) | `13.0, V13.0.88` | `nvcc --version` |
| cuDNN | `91900` | `torch.backends.cudnn.version()` |
| transformers | `5.8.0` | `importlib.metadata` |
| datasets | `4.8.5` | `importlib.metadata` |
| safetensors / accelerate | `0.7.0` / `1.13.0` | `importlib.metadata` |
| numpy | `2.5.1` — **pyproject pins `2.4.4`; live env drifts** | `importlib.metadata` |
| tokenizers | `0.22.2` | `importlib.metadata` |
| jax / flax | `0.11.0` / `0.12.7` | `importlib.metadata` |
| GPU | `NVIDIA GB10`, driver `580.142` | `nvidia-smi` |
| OS / arch | Ubuntu 24.04.4 LTS, `aarch64` | `/etc/os-release`, `uname -m` |
| Repo commit | `3da9063`, branch `harden-research-loop`, **zero git tags** | `git rev-parse HEAD` |

torch/transformers/datasets/safetensors/accelerate match `SmolLM2-134(base)/pyproject.toml`
field-for-field, so **SmolLM2 parity is re-checkable today**. numpy has drifted. cuDNN `91900` and
driver `580.142` are recorded nowhere in the repo. Qwen3 has no environment file at all.

---

## Quick answers to the questions that prompted this audit

| Question | Answer | Detail |
|---|---|---|
| Which dataset/config/split gave **15.371**? | `Salesforce/wikitext` / `wikitext-2-raw-v1` / `validation`, seq 1024, stride 512 (overlapping), SmolLM2 tokenizer | §1 |
| Which gave **6.89 → 3.79**? | `roneneldan/TinyStories` / no config / `validation`, seq 1024, stride 1024 (non-overlapping) | §1, §5.3 |
| Which gave **28.65 / 46.31 / 23.52**? | `HuggingFaceFW/fineweb-edu` / `sample-10BT`, private val tail, seq 4096, stride 4096 — but on **two different tails** | §1 |
| Which wikitext for **−0.474 bpb**? | `wikitext-2-raw-v1` (rev `b08601e0…`), *not* wikitext-103 | §3 |
| Is **13.40** ours or borrowed? | **Ours** — measured on this box, 2026-06-09 | §2 |
| Commit hash for the results? | **None stamped.** Best anchors `e791875` / `84a96c0` both *postdate* the artifacts | §4.2 |
| CPU-only or GPU parity? | **CPU fp32 only** for the `max error 0.0` claim. GPU deltas exist and one **trips the repo's own 1e-3 gate** | §4.4 |
| Determinism flags? | **None, repo-wide** — verified negative for all six flags | §4.5 |
| Checkpoint formats? | 107 files, 270.68 GiB, **all `.pt`/`.pkl` pickles — zero safetensors**, no `config.json`, no tokenizer files | §6 |
| `modeling_*.py` or raw weights? | **Export SmolLM2 to stock `LlamaForCausalLM` safetensors** (an exporter exists). `trust_remote_code` only for IMU-1 / partial-RoPE / HybridSSM | §6.3 |
| Real loader API? | No `from_pretrained`. `Qwen3ForCausalLM(Qwen3Config())` + `load_official_weights_into_ours()` from `verify.py` | §7 |

**Two traps any published snippet must avoid:** passing `attention_mask` **disables causal
masking** in both models (`is_causal=(attention_mask is None)`), and `attention_dropout` is dead
config with no read site.

---

# Full fact sheet

## 1. Model-index provenance

**Read this first:** the eight numbers below come from **five mutually incomparable eval recipes** on **four different corpora** with **three different windowing schemes**. No two rows are like-for-like unless explicitly stated.

| Metric value | Model / run | HF dataset id | Config | Split | Seq len | Stride | Tokenizer | n eval tokens | Source file:line |
|---|---|---|---|---|---|---|---|---|---|
| **15.371** (ours 15.370989092449635 / HF 15.370989964425396, Δ +8.72e-07) | **NOT a repo-trained model.** Official `HuggingFaceTB/SmolLM2-135M` safetensors loaded into this repo's `SmolLM2ForCausalLM` via `load_official_weights_into_ours` **[QUALIFIER]** | `Salesforce/wikitext` (no `revision=` pin) | `wikitext-2-raw-v1` (IS the `-raw-` variant) | `validation` | 1024 | 512 — **overlapping, no `-100` masking** | `HuggingFaceTB/SmolLM2-135M` (own tokenizer; no BOS) | **62,403 scored targets = 31,743 distinct positions (1.99× double-count); first ~11.8% of the 268,140-token split** | `SmolLM2-134(base)/results/perplexity.json:2-3`; recipe `SmolLM2-134(base)/_build_notebook.py:233-257`; blank-row filter `:234` |
| **6.8945** (full precision 6.894546783281595) | Baseline "BEFORE" eval — again the **official** SmolLM2-135M weights in our class, bf16 on cuda, scored before any optimizer step | `roneneldan/TinyStories` (resolved rev `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`) **[CORRECTED — rev IS recoverable]** | none passed | `validation` | 1024 | **1024 — non-overlapping** **[CORRECTED: not 512]** | `HuggingFaceTB/SmolLM2-135M` | **199,485** = 195 windows × 1023, over the **first 1,040 non-empty stories** (200,068 packed tokens) **[CORRECTED — count recovered by re-running the packer]** | `SmolLM2-134(base)/results/tinystories_before.txt:2`; `results/tinystories_train.log:10`; eval fn `train_tinystories.py:62-78`, called `:196` |
| **3.7900** (full precision 3.7899503859716885) | `SmolLM2-134(base)/checkpoint_tinystories.pt` — continued pretrain from official weights, step 24,414 / 99,999,744 tokens | `roneneldan/TinyStories` | none | `validation` | 1024 | 1024 | `HuggingFaceTB/SmolLM2-135M` | 199,485 (identical `val_tokens` tensor as the 6.8945 row → strictly paired) | `SmolLM2-134(base)/results/tinystories_after.txt:2`; `results/tinystories_train.log:508`; eval call `train_tinystories.py:362` |
| **28.65** | `checkpoint_qwen3_baseline2tpp.pt`, faithful build, **final step 18,150** (1,189,478,400 tok) | `HuggingFaceFW/fineweb-edu` (no `revision=`) | `sample-10BT` | `train` (streaming) → private 300k-token val tail in `tokcache_1191478400_300000.pt` | 4096 | 4096 — non-overlapping | `Qwen/Qwen3-0.6B-Base` | 204,800 (50 × 4096) | `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.log:396` |
| **46.31** | `checkpoint_qwen3_lr24.pt` — a **2,000-step / 131,072,000-token LR-selection run**, not a headline model (siblings lr17 46.89, lr30 49.28) | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | `train` (streaming) → val tail in **`tokcache_133072000_300000.pt`** | 4096 | 4096 | `Qwen/Qwen3-0.6B-Base` | 204,800 | `.../results/qwen3_lr24_train.log:226`; also `qwen3_lr24_after.txt:2` |
| **23.52** | `checkpoint_imu1_2tpp_step18000.pt` — modernized/IMU-1 arm, **in-loop eval at step 18,000, NOT the 18,150 endpoint** (no AFTER eval exists) **[QUALIFIER]** | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | val tail in `tokcache_1191478400_300000.pt` | 4096 | 4096 | `Qwen/Qwen3-0.6B-Base` | 204,800 | `Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log:381` |
| **13.40** (13.400) | **The released `Qwen/Qwen3-0.6B-Base` HF checkpoint, scored by us** (see §2) | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | val tail in **`tokcache_133072000_300000.pt`** | 4096 | 4096 | `Qwen/Qwen3-0.6B-Base` | 204,800 | `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_vs_repro.txt:2` (stdout `results/original_eval_run2.log:6`, 2026-06-09 16:51:36) |
| **−0.474 bpb** (stored as `improvement_bpb` = +0.47432550192416323; AdamW 2.1098171365956357 − NorMuon 1.6354916346714725; CI95 **[0.4434844613250229, 0.5051665425233036]**) | NorMuon vs AdamW, **596,049,920-param** Qwen3, **42M tokens/cell (640 steps × 65,536)**, **n=3 seeds/arm** | `Salesforce/wikitext` **rev `b08601e04326c79dfdd32d625aee71d232d685c3`** | `wikitext-2-raw-v1` | `validation` | 1024 | 512 — overlapping | `Qwen/Qwen3-0.6B-Base` | 204,600 scored = **102,911 distinct** (MAX_WINDOWS=200 cap; 869,710 bytes denominator) | `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/verdict.json:19-30`; corpus `score_cohort.py:54-56`; windowing `score_cohort.py:24` |
| **−0.502 bpb (companion code corpus)** (+0.5015586517902451; AdamW 3.3846985755955523 vs NorMuon 2.8831399238053073; CI95 [0.4559911731303807, 0.5471261304501094]) | same 6 cells | `codeparrot/codeparrot-clean-valid` **rev `4db92d2ec0c1b4c41eeb439cfae16854511d9dcd`** | — | `train` (streaming, whole docs until >500,000 chars) | 1024 | 512 | `Qwen/Qwen3-0.6B-Base` | 204,600 tokens / 843,643 bytes | `Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/verdict.json:80-84`; corpus `score_cohort.py:57-63` |

**Cross-row warnings that must ship with any model index:**

- **[CORRECTED — the repo's own README is FALSE here]** `Qwen3-0.6B/README.md:35-37` claims all four Qwen3 PPLs use "identical eval code on the identical 300k-token FineWeb-Edu val slice". They do not. **13.40 and 46.31** are on `tokcache_133072000_300000.pt` (val sha1 `8ad9e246b0bf63bd`, first ids `[10879, 5547, 481, …]`); **28.65 and 23.52** are on `tokcache_1191478400_300000.pt` (val sha1 `ad3513719d0f81e4`, first ids `[38131, 6022, 369, …]`). Therefore **28.65/13.40 = "2.14×" and 23.52/13.40 = "1.76×" are CROSS-SLICE ratios and must not be published as like-for-like gaps.** Only 46.31/13.40 (3.456×) is same-slice. The same false claim is repeated at `Qwen3-0.6B/results_overview/plots/README.md:50`.
- **23.52 vs 28.65 is not step-matched.** The like-for-like pair is **28.66 @ step 18,000** (`qwen3_baseline2tpp_train.log:389`) vs 23.52 → −17.94%. The headline survives, but as printed the arms differ by 0.83% of budget.
- **Slice sensitivity is ~14%.** The same `checkpoint_qwen3_baseline2tpp.pt` that scores 28.65 on its own slice scores **24.5514** on the dataset-forge held-out FineWeb-Edu split under `text-lm-v2` windowing — recorded with an explicit delta field `"base_ppl_measured_vs_claim_delta": -4.09860353498452` in `Qwen3-0.6B/experiments/2026-06-17_qwen3-0.6b_vibethinker-small-reasoning/eval/brief_probes_results.json`.
- **None of 28.65 / 23.52 / 46.31 / 13.40 is decontaminated.** All four used caches built by the pre-audit splitter, which the repo's own current code calls leak-suspect: *"the old code cut the stream by token count — val was the sequential continuation of train, leak-suspect"* (`train_qwen3.py:130-136`). Proof independent of filenames: neither cache contains the `decontam` key that the post-fix splitter writes (`train_qwen3.py:190`). The fix landed in commit `86e79f3` (2026-06-16 21:57 UTC), **after** all four Phase-B runs finished.
- **The repo's own governance bans these as headlines.** `research/eval/base_eval_verdict.md:59`: *"It is **n=1 FineWeb val-PPL — the founding-mistake metric, banned as a sole/headline signal by §C25.7.3**"*.

**The §C10-comparable, suite-stamped alternative** (safe for a card) — `Qwen3-0.6B/experiments/2026-06-16_qwen3-faithful_eval-first/eval/suite_results.json`, `suite_version: "text-lm-v2"`, target `checkpoint_qwen3_baseline2tpp.pt`, dated 2026-06-16 22:18:01, SEQ/STRIDE/MAX_WINDOWS = 1024/512/200, revision-pinned corpora:

| Corpus | PPL | BPB | n_tokens | n_bytes |
|---|---|---|---|---|
| `wikitext2_raw_v1_val` | 37.010055463333096 | 1.2256204566076285 | 204,600 | 869,710 |
| `codeparrot_clean_valid` | 438.67295146042875 | 2.128595386220801 | 204,600 | 843,643 |

Downstream (`text-lm-v3`, 2026-06-24, n=500/task, Wilson CIs) — `research/eval/downstream_v3/*.json`: faithful LAMBADA **0.170** [0.1396, 0.2054], mean BPB-gold 1.18827; IMU-1 **0.212** [0.1784, 0.2500], 1.14232; pRoPE-0.25 **0.166** [0.1360, 0.2011], 1.20234. **[CORRECTED]** Do not write "MC tasks are at chance": `arc_easy` acc_norm 0.454 and `hellaswag` acc_norm 0.348 both carry `"signal": true`; only `winogrande` (0.500 vs chance 0.500) is `"signal": false`. The repo's own careful phrasing is `research/eval/base_eval_verdict.md:63` — *"flagged `signal: true` but only marginally above chance with wide CIs; not headline-bearing."*

---

## 2. Is 13.40 ours or borrowed?

**13.40 is OURS. It is a measurement this repo performed on this box.** It is not copied from the Qwen3 tech report, the HF model card, or any blog.

Evidence:

- `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/eval_original_vs_repro.py:47-51` downloads the released checkpoint and scores it with our own loop:
  ```
  hf = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(device)
  ppl_orig, n = eval_ppl(hf, val, device)
  lines.append(f"ORIGINAL  Qwen3-0.6B-Base (36T tok)   val PPL = {ppl_orig:8.3f}  ({n:,} tok, {time.time()-t0:.0f}s)")
  ```
  `REPO = "Qwen/Qwen3-0.6B-Base"` imported from `train_qwen3.py:59` via `eval_original_vs_repro.py:19`. Scoring uses the repo's own `eval_ppl` (`:26-36`), not any external harness.
- Live-run corroboration with a real safe_cuda banner: `results/original_eval_run2.log:2` *"[safe_cuda] capped CUDA at 85% of 129 GB unified pool (~109 GB)"*, `:3` *"Loading weights: 100%|██████████| 310/310"*, `:6` the 13.400 line, timestamped 2026-06-09 16:51:36.
- An earlier same-day attempt (`results/original_eval_wrapper.log`, 11:19) crashed on `UnpicklingError` before printing any PPL — so there is no conflicting earlier value.

**What IS borrowed, in the same string:** the label `(36T tok)` is a hardcoded f-string literal at `eval_original_vs_repro.py:51`. The 36T figure is transcribed from the Qwen3 tech report (`training_plan.md:17-19`: *"Per the Qwen3 tech report (verbatim summary): - **Corpus:** 36T tokens across 119 languages"*, cited as arXiv 2505.09388 at `Qwen3-0.6B/README.md:29`). **13.40 = ours (measured); 36T = theirs (copied).** Keep that split explicit.

**Wording hazard:** `Qwen3-0.6B/PLOTS_INDEX.md:37` and `Qwen3-0.6B/results_overview/plots/README.md:49` both use the word "published" next to 13.40. In context that means *"the published (released) model"*, not *"a published number"* — the generator `make_overview_plots.py:18-19,51` correctly traces `ORIGINAL_PPL = 13.40` to our `original_vs_repro.txt`. **Do not describe 13.40 on a model card as a reported/published figure.**

**No suite-comparable counterpart exists.** I enumerated all 9 `suite_results*.json` in the repo: every `target_ckpt` is a local `.pt`/`.pkl`; the `text-lm-v2` suite was **never** run against the released Qwen3-0.6B-Base. So there is no BPB/wikitext number for the released model to pair with our 37.01.

---

## 3. −0.474 bpb: which wikitext, and does the claim still stand?

**Which wikitext:** `wikitext-2-raw-v1` — *not* wikitext-103. Verbatim at `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/score_cohort.py:54-55`:
```
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation",
                  revision=WIKITEXT_REV)
```
with `WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"` at `score_cohort.py:25`. Same triple pinned as the suite standard at `.claude/skills/eval-harness/references/suite.md:133`. (For contrast, `SmolLM2-134(base)/train.py:78` uses `wikitext-103-raw-v1` **train** — a different thing entirely, and a training corpus, not this eval.)

**Sign convention:** the JSON stores **+0.47432550192416323** as `improvement_bpb = adamw_mean − normuon_mean` (`score_ladder.py:85-86`: *"so gap > 0 == NorMuon better"*; `research/eval_stats.py:138`). The literal string `−0.474` appears on disk **only** at `Qwen3-0.6B/README.md:52` and `:243`. Both forms mean "NorMuon 0.474 bpb lower (better)".

**Does it still stand? NO — it has been nulled at scale.**

`Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/verdict.json` (re-scored 2026-07-28, n=3 seeds/arm at **every** rung, fixed N=596M, budget swept):

| Budget | wikitext-2 gap (bpb) | CI95 | significant | code_py gap | CI95 |
|---|---|---|---|---|---|
| 42M | 0.47432550192416323 (`:27`) | [0.4435, 0.5052] | yes | 0.5015586517902451 (`:80`) | [0.4560, 0.5471] |
| 168M | 0.12590584068581911 (`:44`) | [0.0893, 0.1625] | yes | 0.1757807425441804 (`:97`) | [0.1369, 0.2146] |
| 420M | 0.07169397744785555 (`:61`) | [0.05525251860105098, 0.08813543629466011] (`:62-66`) | **yes** | 0.17708989020863175 (`:114`) | [0.1307, 0.2234] |

- `trend_verdict: "CONVERGES"` (`:198`), `ledger_verdict: "null"` (`:199`), rationale (`:194`): *"gap shrinks toward 0 with scale and falls within the noise floor at the largest budget — an early-training speedup that converges away (no advantage at scale)."*
- `research/ledger/ledger.json:1554,1565-1566` — run `2026-07-05_qwen3-0.6b_scaling-persistence`, status `done`, **verdict `null`**.

**Three nuances a reviewer will demand and that must not be lost:**

1. **It is a BUDGET-scaling null at FIXED model size N=596M.** Nothing on disk says anything about larger N.
2. **The 420M wikitext gap is still nominally SIGNIFICANT as measured** (+0.0717, CI excludes 0). The "falls within the noise floor" phrase refers to the **OLS-fitted** gap at the top rung — `gap_hi_fit`/`edge_at_top` = 0.029726712435672376 (`:140-142`) vs `gap_noise` 0.03675972213287565 (`:146`), `edge_resolved: false` on wikitext, **`true` on code**.
3. **[QUALIFIER — the "CONVERGES" label on the code corpus is weak]** code_py goes 0.50156 → 0.17578 → **0.17709**, i.e. it *increases* between the last two rungs with near-total CI overlap. That is a **plateau, not convergence**; only the 3-point OLS slope (−0.34197431351062624, r² 0.8412726939487719, `:156-158`) is negative, and it is dominated by the 42M rung. Rationale on disk (`:171`) is itself hedged: *"still above noise at the largest measured budget but trending out — the edge is eroding, extend the ladder before claiming it."*
4. The `null` is partly gate-driven: the §C25 `scaling` HARD battery is INCOMPLETE (missing `log_rmse_r2`, `holdout_extrapolation_pctdev`, `bootstrap_forecast_ci`, `:207-214`), so `win` was unreachable regardless — **but** `significance_verdict` was independently `null` from the CONVERGES trend mapping, with `c17_cap_applied: false` (`:203-206`).

**The published surfaces are stale and over-claim. [CORRECTED — exposure is 4 sites, not 2]:**

- `Qwen3-0.6B/README.md:52` — *"| **NorMuon > AdamW** | wikitext −0.474 bpb [0.444, 0.505] · code −0.502 [0.456, 0.547] | **significant win** |"*
- `Qwen3-0.6B/README.md:175` — *"AdamW by **+0.474 bpb on wikitext-2 (95% CI [0.444, 0.505])** and +0.502 on code — significant."*
- `Qwen3-0.6B/README.md:243` — same claim, tagged **significant win**
- `Qwen3-0.6B/PLOTS_INDEX.md:73` — *"+0.474 bpb, significant"*

`grep -in 'scaling-persistence|converge|persist|0\.126|0\.072' Qwen3-0.6B/README.md` → **zero hits**. The file's mtime is 2026-07-06, before the ladder completed. The 42M ledger entry still reads `"verdict": "win"` (`research/ledger/ledger.json:482`) with a `caveats` field at `:512` asserting *"no scaling curve"* — a statement that became false when the ladder completed. The `normuon-optimizer` technique's `run_ids` (`ledger.json:153-154`) omit the ladder run (whose `technique_slug` is `null`, `:1557`), so **a ledger query by technique will not surface the null**.

**Rounding defect:** the three README sites print the CI lower bound as `0.444`; the on-disk value is `0.4434844613250229`, which rounds to **0.443** (the root `README.md:112` gets this right).

**To the source run's credit:** `RESULT.md:7` does scope it correctly — *"This is an early-training optimization-speed signal at one architecture and one budget; we do NOT claim it holds at scale"* — and `RESULT.md:45` predicts the fade. The failure is that `Qwen3-0.6B/README.md` dropped those qualifiers.

**Root `README.md:105-121` is ALSO stale** (mtime 2026-07-23 vs verdict.json 2026-07-28): it says *"420M ×2 seeds"*, *"+0.073 [−0.038, +0.184] at 420M … **not significant** at the top"*, *"code_py: +0.502 → +0.176 → +0.192"*, code slope *"−0.328 (r² 0.81)"*, and *"**Verdict: directional, not a headline** — the 420M rung is n=2 (< 3 seeds, §C17)"*. Current truth: n=3/arm, top rung **significant**, code 0.17709, slope −0.34197 (r² 0.84127), `headline_capped_by_c17_power: false`, ledger verdict `null`. The 3rd 420M seed came from run `2026-07-23_qwen3-0.6b_normuon-at-scale` (`research/ledger/runs/2026-07-23_qwen3-0.6b_normuon-at-scale.md:356-360`).

**Eval pipeline identity (the numbers ARE mutually comparable):** `score_ladder.py:40` imports the 42M scorer directly (`import score_cohort as sc … reuse score() + load_corpora()`), `:454-455` reuse its corpora/device/dtype, `:456` copies the 42M rung verbatim (`"source": "reused:cohort_bpb.json"`). Same `suite_version: text-lm-v2` on both.

---

## 4. Reproduce

### 4.1 Literal commands

**Qwen3-0.6B — architecture parity (CPU, no GPU, no artifact written):**
```bash
cd /home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B
python3 verify.py          # asserts max|Δlogits| < 1e-3 and argmax equality; stdout only
```
(`Qwen3-0.6B/verify.py:11`; `Qwen3-0.6B/README.md:498`: *"python verify.py        # parity gate — runs on CPU, no GPU needed"*. The `cd` is load-bearing — `verify.py:16` does a flat `from model import …` with no sys.path handling.)

**Qwen3-0.6B — parity with a machine-readable artifact:**
```bash
cd /home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b
python3 verify_run.py      # writes results/verify.json; exit 0 pass / 1 fail
```
(`verify_run.py:9,31,93,99,102,106`; this is the first command in the paper appendix, `research/papers/qwen3-imu1-matched-compute/sections/reproducibility.tex:15`.)

**Qwen3-0.6B — recompute 13.40 / 46.31:**
```bash
cd /home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b
python3 eval_original_vs_repro.py    # writes results/original_vs_repro.txt
```
**Not runnable from a fresh clone.** `eval_original_vs_repro.py:41` hardcodes `device = torch.device("cuda")` with no CPU fallback, and it needs two gitignored artifacts: `results/tokcache_133072000_300000.pt` (1,066,978,101 B, 2026-06-08) and `checkpoint_qwen3_lr{17,24,30}.pt` (3,576,719,229 B each). `git check-ignore -v` → `.gitignore:20:*.pt`. It also reads the **pre-decontamination** cache (§1).

**SmolLM2 — parity:**
```bash
cd "/home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)"
pytest tests/ -v       # broader gate
# or the script form:
python3 verify.py      # produces the committed results/parity.log
```
(`README.md:159-162`; `SmolLM2-134(base)/verify.py:11`; `tests/test_parity.py:8`.) **[QUALIFIER]** The pytest form is broader (param count `test_parity.py:53`, tied-embedding pointer `:61`, 512-token long context `:89-101`, all 30 per-layer hidden states `:104-132`) **but it is not unconditionally stronger: `tests/test_parity.py:34-41` calls `pytest.skip(…)` on ImportError or model-load failure ("can't load {REPO} (no internet or HF cache miss?)"), so with no network a green run proves nothing.** It also never prints the Δ value.

**SmolLM2 — recompute 15.371 (there is no one-command path):**
```bash
cd "/home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)"
python3 _build_notebook.py       # writes results.ipynb (28 cells, no outputs)
jupyter nbconvert --to notebook --execute results.ipynb \
    --output results.ipynb --ExecutePreprocessor.timeout=2400
```
(`SmolLM2-134(base)/results/README.md:66-74`.) Three hazards: (1) `_build_notebook.py:561-562` **unconditionally overwrites** `results.ipynb`, wiping the executed outputs that are currently the only record; (2) all 28 cells re-execute, including a 150-step training cell that **overwrites `../checkpoint.pt`** (`_build_notebook.py:480-483`); (3) the notebook **never imports `safe_cuda`** (grep → zero hits) despite `CLAUDE.md` §C1 mandating it. **[CORRECTED]** A standalone wikitext-2 PPL script *does* exist — `SmolLM2-134(base)/eval_after_vs_base.py:50,74` — but it uses `max_windows=200` and bf16 (`:29`), so it produces a **different** number, it needs `checkpoint_tinystories.pt`, its declared outputs `results/tinystories_vs_base.{md,json}` are **absent from disk**, and its fallback at `:91` does `open("model.py")` on a file that does not exist.

**SmolLM2 — reproduce the continued-pretrain run (best reconstruction; exact CLI never recorded):**
```bash
cd "/home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)"
python3 train_tinystories.py --token_budget 100_000_000
```
(nearest documented form: `README.md:170-174`). It was **not** a `--resume` run (the log contains the baseline eval, which `train_tinystories.py:194` skips on resume; the CSV carries a header, written only when resume is None). **`grep -n "safe_cuda\|sentinel" train_tinystories.py` → zero hits: re-running as-is violates `CLAUDE.md` §C1 and §C6.**

**Qwen3 Phase-B training (the 1.19B runs) — the real launch record:** `Qwen3-0.6B/builds/phase_b_driver.sh`, verified verbatim:
```
:14  S=18150; W=900; COMMON="--eval_every 2000 --ckpt_every 2000 --log_every 50"
:19  cd "$FAITHFUL" && python train_qwen3.py --steps $S --peak_lr 2.4e-3 --end_lr 3.2e-4 \
:20    --warmup_steps $W $COMMON --run_name baseline2tpp
:24  cd "$MOD" && python train_imu1.py --steps $S --warmup_steps $W $COMMON --run_name imu1_2tpp
:28-29, :33-34   partial-RoPE 0.25 / 0.10
```

### 4.2 Commit hash

**There is no provenance stamp.** No results file for either model carries a commit/`git_sha` field. `verify.json`'s complete key set is `repo/prompt/dtype/tolerance/max_abs_error/relative_error/hf_next_token_id/our_next_token_id/hf_next_token_text/our_next_token_text/argmax_match/passed/input_shape/total_seconds`. `perplexity.json`'s is `ours_ppl/hf_ppl/tokens/dataset/seq_len/stride`. `parity.log` is raw stdout.

The only anchors are the commits that *added* the artifacts — and both **postdate** the artifact mtimes, so they bound from above rather than identify the tree that ran:

| Artifact | mtime | Adding commit |
|---|---|---|
| `Qwen3-0.6B/builds/.../results/verify.json` | 2026-06-08 14:36 | `e791875` "Add Qwen3-0.6B from-scratch reproduction + three-build experiment" (2026-06-10) |
| `SmolLM2-134(base)/results/parity.log` | 2026-05-13 22:20 | `84a96c0` "Initial commit: SmolLM2-135M from-scratch reproduction + harness" (2026-05-20) |

`git tag -l` → **empty**; HEAD is `3da9063` (2026-07-24). The ledger *does* auto-capture HEAD (`research/ledger/ledger.py:639`: `r["lineage"]["git_commit"] = git_head_commit()`) and all 29 runs carry one — but the earliest is `2026-06-16_qwen3-faithful_eval-first` (`86e79f3`), and **neither reproduction has a ledger run entry at all**. `lineage.env` is null for 28 of 29 runs.

### 4.3 Software versions

**Pinned in exactly two files, both under `SmolLM2-134(base)`, and they contradict each other:**

| | `pyproject.toml` | `requirements.txt` |
|---|---|---|
| python | `requires-python = ">=3.10"` (`:17`) | — |
| torch | `torch==2.11.0` (`:21`) | `torch>=2.4` (`:1`) |
| transformers | `transformers==5.8.0` (`:22`) | `transformers>=4.40` (`:2`) |
| datasets | `datasets==4.8.5` (`:23`) | unpinned |
| safetensors / accelerate / numpy | `0.7.0` / `1.13.0` / `2.4.4` (`:24-26`) | unpinned |
| pytest | `9.0.3` (`:31`) | — |

Root `README.md:157` offers them as equivalent (`pip install -e .   # or: pip install -r requirements.txt`) directly under `:156` *"Install pinned dependencies that produced the 0.0 logit-diff result."* — following the `requirements.txt` branch can install torch 2.4 / transformers 4.x, which is **not** the pinned-environment claim. `pyproject.toml:4-7` admits no lockfile exists and gives the `uv pip compile` command that was never run. **`Qwen3-0.6B/` has NO `requirements.txt`, NO `pyproject.toml`, NO lockfile** — its only install instruction is the unpinned one-liner `pip install torch transformers datasets safetensors accelerate` (`Qwen3-0.6B/README.md:497`).

**Stamped from an actual execution in exactly ONE place** — the executed `SmolLM2-134(base)/results.ipynb`: cell 1 output `Torch: 2.11.0+cu130` / `Device: cuda | NVIDIA GB10`; `metadata.language_info.version = "3.12.11"` (generator `_build_notebook.py:50-51`). A repo-wide grep for `torch.__version__|torch.version.cuda|platform.python_version|sys.version` over `*.py` returns only that line plus one unrelated HybridSSM file. **No Qwen3 script and neither `verify.py` stamps any version.**

**Live box (re-measured for this fact sheet):** python `3.12.11`, torch `2.11.0+cu130`, `torch.version.cuda` `13.0`, `torch.backends.cudnn.version()` `91900`, transformers `5.8.0`, datasets `4.8.5`; `nvidia-smi` → driver `580.142`, `NVIDIA GB10`. **These match `pyproject.toml` and the notebook stamp field-for-field** — SmolLM2 bit-exactness is currently re-checkable. Two gaps remain: cuDNN `91900` and driver `580.142` appear **nowhere** in the repo (grep over all `*.md/*.json/*.toml/*.txt/*.log` for `cuDNN|cudnn_version|580.14|CUDA 13|CUDA 12` returns one unrelated prose hit at `jax_vs_pytorch_tradeoffs.md:44`), and `torch==2.11.0` carries no `+cu130` local tag, so a CPU or CUDA-12 build satisfies the pin.

**For Qwen3 the versions behind `verify.json` (2026-06-08) and `original_vs_repro.txt` (2026-06-09) are UNDETERMINABLE.** The only signal is the `torch_dtype` deprecation banner at `results/original_eval_run2.log:1`, which bounds transformers from below but names no version.

**The paper appendix records hardware, dtype, seed, batch/step config and the commands — but no software versions.** `research/papers/qwen3-imu1-matched-compute/sections/reproducibility.tex:4-10`: *"a single NVIDIA GB10 (Grace Blackwell, unified ≈119 GB CPU+GPU memory) in bfloat16 with seed 0 … effective batch is 4 × 4 accumulation = 65,536 tokens, and both arms train for 18,150 steps."* No version block anywhere in the file.

### 4.4 CPU-vs-GPU parity scope — **read carefully, the headline is CPU-only**

| Claim | Device | dtype | Value | Source |
|---|---|---|---|---|
| Qwen3 `max_abs_error` | **CPU** | fp32 | **0.0** (relative 0.0; argmax `" Paris"`, id 12095, `argmax_match: true`, `passed: true`) | `Qwen3-0.6B/builds/.../results/verify.json` |
| SmolLM2 `max\|Δlogits\|` | **CPU** | fp32 | **0.000e+00** (relative 0.000e+00; argmax `" the"`, id 260) | `SmolLM2-134(base)/results/parity.log:6-9` |
| SmolLM2 final-logits parity | **GPU** | — | **4.72e-05** | `SmolLM2-134(base)/results/comparison_with_hf.md:10` |
| SmolLM2 per-layer (30 layers) | **GPU** | — | **1.95e-03 at layer 14** — **EXCEEDS the repo's own 1e-3 gate** | `comparison_with_hf.md:11` |
| SmolLM2 long-context RoPE | GPU | — | 4.01e-05 (labelled "401-token"; the code truncates at `max_length=512`, `compare_with_hf.py:180-181`) | `comparison_with_hf.md:14` |

**[CORRECTED — the original fact-finding said "no GPU parity number is stamped anywhere". That is wrong: the GPU numbers exist, in prose.]** They are `prose-only` — the machine-written `results/comparison_with_hf.json` that `compare_with_hf.py:259-261` would produce is **absent from disk**, and `comparison_with_hf.md`'s mtime (2026-05-13 22:07:53) *predates* the notebook run (22:19–22:20), so it was not produced by that run. `results/README.md:3-4` claims *"Every file here is produced live … No values are typed in by hand"* — that blanket claim is **not supported** for `comparison_with_hf.md`.

**So: the "max error 0.0 / bit-exact" claim is CPU-fp32-only. On GPU the reproduction is close but NOT bit-exact, and one per-layer delta trips the 1e-3 gate.** The repo itself explains this at `comparison_with_hf.md:22-42` (SDPA backend dispatch: HF passes an explicit mask, we pass `is_causal=True`) and devotes a section at `:49` to *"What the earlier ✗ at '1.953e-3' meant — and didn't mean."* `comparison_with_hf.md:51` also notes *"The threshold in `compare_with_hf.py` was `1e-3`, picked for bf16 tolerance"* — i.e. a loose gate for an fp32 claim.

**Qwen3 has NO GPU parity check at all**, and no long-context or per-layer parity check anywhere.

**Parity tolerance and input:** `assert max_abs < 1e-3` and `assert hf_next == our_next` — identical in five implementations: `Qwen3-0.6B/verify.py:74,81`; `SmolLM2-134(base)/verify.py:76`; `verify_run.py:33` (`TOLERANCE = 1e-3`); `tests/test_parity.py:73,98,132`; `compare_with_hf.py:235,240,249`. Input is a **single 5-token prompt, batch 1**: `"The capital of France is"` → `input_shape: [1, 5]` (`verify.json`). SmolLM2 token ids `[504, 3575, 282, 4649, 314]`. **This is a thin gate for a model card.** Note the two models' argmaxes differ (Qwen3 `" Paris"`, SmolLM2 `" the"`) — do not present one as shared.

**Stale docstring:** `SmolLM2-134(base)/verify.py:6` says the logits match *"to bf16 numerical tolerance"* while the code and run are fp32; `Qwen3-0.6B/verify.py:6` correctly says fp32.

### 4.5 Determinism flags

**NONE. This is a definitive negative finding, verified twice with and without `--include` filters:**

```
grep -rn "allow_tf32|use_deterministic_algorithms|cudnn.deterministic|cudnn.benchmark|CUBLAS_WORKSPACE_CONFIG|set_float32_matmul_precision"  →  no output
```
Not one of the six flags appears anywhere in the repo. The only `tf32` strings are an error message in `mfu_meter.py:115` and a `peak_fp32_tf32_tflops_assumed` constant in `research/systems/roofline_hybridssm.py`. Neither `verify.py` sets any seed or flag either — their complete setup is two imports plus a module constant (`Qwen3-0.6B/verify.py:13-19`).

This matters precisely for the GPU deltas above: `comparison_with_hf.md` attributes them to unpinned backend dispatch, which is what a determinism flag would have controlled. TF32 on Blackwell is left at the PyTorch default and never recorded.

**Seeds that ARE set (for the non-parity numbers):** `_build_notebook.py:45` `torch.manual_seed(0)` (the 15.371 notebook), `:201`/`:447` (`42`/`0`); `compare_with_hf.py:39`; Qwen3 trainer `train_qwen3.py:225-228` (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`); SmolLM2 `train.py:95-96`, `train_tinystories.py:99-100`. Paper appendix records seed 0.

---

## 5. Training details

### 5.1 Qwen3-0.6B Phase B — the four ~1.19B-token runs

| | Faithful baseline | IMU-1 / NorMuon (modernized) | partial-RoPE 0.25 | partial-RoPE 0.10 |
|---|---|---|---|---|
| Script / run_name | `train_qwen3.py` / `baseline2tpp` | `train_imu1.py` / `imu1_2tpp` | `train_partialrope.py` / `prope25_2tpp` | `prope10_2tpp` |
| Steps × tok/step | 18,150 × 65,536 | 18,150 × 65,536 | 18,150 × 65,536 | **died at step 5,450/18,150** |
| Total tokens | 1,189,478,400 | 1,189,478,400 | 1,189,478,400 | ~357M |
| seq_len / micro_batch / grad_accum | 4096 / 4 / 4 (= 16 seqs = 65,536 tok) | identical | identical | identical |
| Precision | **full bf16** — weights cast at construction; **no `torch.autocast`, no `GradScaler`, no fp32 master weights** (grep → 0 hits in all three trainers) | same | same | same |
| Cross-entropy | chunked, fp32 accumulator, chunk 8192 (`train_qwen3.py:87,91`) | **`train_imu1.py:38-45` does NOT `.float()` its CE chunks — CE accumulated in bf16. A real, unremarked between-arm numerical difference.** | fp32 chunked | fp32 chunked |
| Optimizer | AdamW, betas (0.9, 0.95), eps 1e-8 (`train_qwen3.py:302`) | **hybrid**: 224 2D non-embed → `NorMuon(lr=0.011, wd=0.1, beta1=0.95, beta2=0.95)`; 198 embed/1D → `AdamW(lr=0.006, betas=(0.9,0.95), eps=1e-8, wd=0.0)` (`train_imu1.py:82-85`; split measured at `qwen3_imu1_2tpp_train.log:2`) | AdamW | AdamW |
| LR schedule | cosine, peak **2.4e-3** → end 3.2e-4 (floor 0.13333), **warmup 900** | **WSD**: linear warmup **900** → stable → linear decay-to-zero over final 20% | cosine, 2.4e-3 → 3.2e-4 | same |
| Weight decay | 0.01, `dim>=2` only (`dim<2` → 0.0) | 0.1 on 2D; 0.0 on 1D | 0.01 | 0.01 |
| Grad clip | 1.0 (`clip_grad_norm_`, `:391`) | 1.0 | 1.0 | 1.0 |
| Extra loss | — | chunked z-loss, weight 1e-4 | — | — |
| Seed | 0 (argparse default; driver never passes `--seed`) | 0 | 0 | 0 |
| torch.compile | on | on | on | on |
| Wall-clock | **2,663.1 min = 44.4 h** (`log:395`) | ~63.9 h (DERIVED — no completion line) | ~46.1 h (DERIVED) | ~14.0 h |
| Throughput | **7,444 tok/s final** (cumulative avg; run span 7,414–7,483) **[CORRECTED — README.md:167's "7,480" is the step-100 reading]** | 5,172 tok/s | 7,168 tok/s | ~7,100 tok/s |
| Peak memory | 52.4 GB | 66.1 GB | 54.3 GB | 54.3 GB |

**[CORRECTED — arithmetic error propagated from `Qwen3-0.6B/README.md:80`]** NorMuon is **30.5% lower throughput**, which is **+43.9% wall-clock** (63.9 h / 44.4 h = 1.439), **not "~30–31% more wall-clock"**. Do not copy README.md:80's phrasing.

**[CORRECTED]** The IMU-1 CLI **is** on disk (`phase_b_driver.sh:24`), and it passes **`--warmup_steps 900`, not the script default 50**. Verified against the run's own LR ramp: 0.011 × 50/900 = 6.111e-4 = `log:5`'s `lr 6.11e-04`; 0.011 × 400/900 = 4.889e-3 = `log:12`'s `lr 4.89e-03`. Under warmup=50 the LR would be at peak by step 50. `normuon_lr 0.011 / adam_lr 0.006 / weight_decay 0.1 / decay_frac 0.2 / z_weight 1e-4` were not overridden, so those defaults do hold.

**Iso-FLOP status: iso-TOKEN only.** No `train_flops` artifact exists for any Phase-B run (grep over `Qwen3-0.6B/builds/` → zero files), so the §C18 ≤5% gate was **never evaluated on disk** for the three-build comparison. `Qwen3-0.6B/README.md:81` asserts params are "iso-FLOP at 1.00043" — that is a parameter-count claim, not a FLOP artifact.

**IMU-1 confound (5 variables at once, violating the repo's own one-variable rule):** optimizer + schedule shape (WSD vs cosine) + z-loss + architecture (`vr=True ln=True hg=True` — value residuals, LayerNorm scaling, head gating, `qwen3_imu1_2tpp_train.log:1`) + weight_decay 0.1 vs 0.01.

**MFU: NOT_FOUND for these four runs.** `mfu_meter.py` exists; **[CORRECTED]** MFU *has* been computed elsewhere in the repo (`research/ledger/ledger.json:503` `"mfu": 0.2909` for the normuon-vs-adamw run; `Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/c5_evidence.json:11-14` `mfu 0.3209, achieved_tflops 40.11`) — but never for the 1.19B runs. The GB10 device peak is `estimated: True` (`mfu_meter.py:63-66`, 125.0 bf16-dense TFLOPs), so per `CLAUDE.md:19` a GB10 MFU must **never** be quoted as exact.

**Chinchilla:** ~2 tokens/param (1.19B / 596,049,920). `builds/2026-06-08_reproduce-faithful_qwen3-0.6b/README.md:20`: *"The paper used ~36T tokens; we use 131M (Phase A) to 1.19B (Phase B) … Chinchilla-optimal for 596M is ~12B tokens (20 tok/param); even Phase B is ~10× under-trained."* 36T/1.19B ≈ 30,252× less data (corroborated at `research/brutal_scorecard.md:57`).

### 5.2 Qwen3 training corpus + tokenization

- **HF dataset id:** `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, split `train`, `streaming=True` — `train_qwen3.py:151`. **No `revision=` pinned.** The line is unchanged in the pre-`86e79f3` version that actually ran (confirmed via `git diff e791875 86e79f3`).
- **Tokenizer:** `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")` (`train_qwen3.py:59,281`). **[CORRECTED]** `151,936` is the **model config's `vocab_size`** (`Qwen3-0.6B/model.py:37`), not the tokenizer's vocabulary size — `len(tokenizer)` is **151,669** (`research/eval/private_heldout_v1/private_prose_v1.txt:456`). Write "model vocab_size 151,936".
- **Packing:** each doc encoded `add_special_tokens=False`, one EOS appended, ids concatenated into one flat stream; contiguous **non-overlapping** 4096-token windows (`PackedTextDataset`, `train_qwen3.py:110-123`); `DataLoader(shuffle=True, drop_last=True)` shuffles window order. **No cross-document attention masking.** 290,888 windows available, 290,400 consumed = **99.83% of exactly one epoch** (single-pass).
- The other three arms **loaded the same cache** the faithful run built (`tokcache_1191478400_300000.pt`, 9,534,229,373 B) — `qwen3_imu1_2tpp_train.log:3`, `qwen3_prope25_2tpp_train.log:2`, `qwen3_prope10_2tpp_train.log:2`.
- **No dataset card exists** for this corpus. `research/datasets/` holds only `data-selection-dclm-edu`, `grpo-math-prompts-v1`, `hybridssm-fineweb-edu`, `math-eval-v1`, `math-reasoning-openr1-math-220k` — none fed the 1.19B runs.

### 5.3 SmolLM2 continued pretrain (the 6.8945 → 3.7900 run)

| Field | Value | Source |
|---|---|---|
| Init | **official HF SmolLM2-135M safetensors** loaded into our class — **not from scratch** | `train_tinystories.py:39,145-149` |
| Corpus | `roneneldan/TinyStories` split `train` (2,119,719 stories) / `validation` (21,990) | `train_tinystories.py:154-155`; `results/tinystories_train.log:4` |
| Resolved dataset revision | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (cached 2026-05-13 14:16–14:20, ~7 h before the run) **[CORRECTED — not unrecoverable]** | `HF_HOME=/home/yashb98/projects/qwen-distill/hf_cache`, `hub/datasets--roneneldan--TinyStories/refs/main` |
| Model revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (cached 2026-05-13 11:44–11:47) **[CORRECTED]** | same HF_HOME |
| Packed train tokens | 102,000,116 (97.9 s) | `log:5` |
| Packed val tokens | 200,068 (first **1,040** non-empty stories, 0 empties skipped — re-derived by re-running the packer today) | `log:6` |
| Steps / budget | 24,414 steps → 99,999,744 of 100,000,000 tokens | `log:8,17`; `ck['step']`, `ck['tok_seen']` |
| tok/step | 4,096 (= 1024 × micro_batch × grad_accum) | `log:8`; formula `train_tinystories.py:189` |
| micro_batch / grad_accum | **PROSE-ONLY** — only the product 4,096 is logged. `micro_batch 4` appears at `results/POST_DATA.md:52` and as the current script's argparse default | — |
| seq_len | 1024 | argparse default |
| LR schedule | **WSD, MEASURED from the per-step trace**: linear warmup 200 → peak **3e-4**; stable through step 19,531; linear decay from 19,532 (2.9993856235920535e-4 = 3e-4·(1−1/4883)) to **0.0** at 24,414 | `results/tinystories_train.csv:2,201,19532,19533,24415`; shape `train.py:45-56` |
| Optimizer / betas / eps / wd / clip / seed | **NOT_FOUND in any run artifact.** AdamW, (0.9, 0.95), 1e-8, 0.01 on `dim>=2`, clip 1.0, seed 0 exist ONLY in prose + the *current* script's argparse defaults | see the drift block below |
| Precision | bf16 (`log:1`), logits `.float()` before CE, `reduction="sum"` | `train_tinystories.py:70` |
| Wall-clock | **116.1 min** loop (process 21:23:28 → 23:21:27 ≈ 118.0 min) | `log:506` |
| Throughput | 14,356 tok/s (**cumulative average**, rose 12,766 → 14,356 over the run) | `log:505` |
| GPU | `Device: cuda` only — **"NVIDIA GB10" is PROSE for this run** (`README.md:62`, `POST_DATA.md:17`); `nvidia-smi` today confirms the box's GPU but that is present-day corroboration | `log:1` |
| Peak memory | not recorded (run-era CSV has no `peak_mem_mb` column) | — |

**Script drift — the on-disk `train_tinystories.py` is NOT the version that produced this run.** Six independent proofs: (1) the script writes a 6-column CSV header (`:256`) but `results/tinystories_train.csv:1` is `step,loss,lr,tok_seen`; (2) `:139` logs `device=… dtype=… seed=…`, the log reads `Device: cuda   dtype: torch.bfloat16`; (3) `:140` logs `args={vars(args)}` — `grep -c "args=" log` → 0; (4) defaults `--eval_every 2000 --ckpt_every 2000` would emit 12 and 7 marker lines — `grep -c` → 0 and 0; (5) `save_ckpt` writes `training_recipe/optim/sched/rng_*`, the actual checkpoint has none of them (`ck.keys() == ['model','config','step','tok_seen','baseline_ppl','trained_ppl']`); (6) `:197` formats `PPL={base_ppl:.3f}  ({base_n:,} target tokens)` — a different template from `log:10`. mtimes: script 2026-05-19 23:16, results 2026-05-14 00:21. `git log` on the file → single commit `84a96c0`, whose blob is **identical to the working tree**, i.e. git holds only the later version. **The run's exact source is unrecoverable.**

**[CORRECTED — important]** `SmolLM2-134(base)/results/training_recipe_resolved.json` DOES list AdamW / betas [0.9,0.95] / eps 1e-8 / weight_decay 0.01 / clip_grad 1.0 — but its own line 2 declares `"source": "https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm2/config_smollm2_135M.yaml"`, `"fetched": "2026-05-13"`, and its values are the **UPSTREAM FROM-SCRATCH nanotron config** (lr 0.003, warmup 2000, seq_len 2048, global_batch 512, tokens/step 1,048,576, 2,000,000 steps, ~2.097T tokens, implied_data_parallel 64), **not this run**. `results/tinystories_summary.md:80-82` likewise sources wd/clip to "nanotron config" and the optimizer to "paper §4.1". **These hyperparameters are COPIED FROM AN EXTERNAL CONFIG and must not go on a card as measured.**

**Two different TinyStories runs are described in the repo.** `results/tinystories_summary.md` documents an **earlier** run: after-PPL **3.7893** (`:9`), wall clock **137.3 min** (`:14`), mean 12,150 tok/s (`:90`), "max temp 72 °C … 5 users on the box" (`:92`). `POST_DATA.md:165` labels it `(prior run)`. Its recipe table (`:67-82`) and throughput table (`:84-92`) **must not be quoted as the 116.1-min run's**. **[CORRECTED]** All three BEFORE generations are byte-identical to `tinystories_before.txt` and the prompt-1 AFTER sample is byte-identical to `tinystories_after.txt`; only prompts 2 and 3 diverge.

**Derived-statistic check:** best single-batch loss **0.9087928533554077 @ step 22,353** — CONFIRMED (`csv:22354`). First 1000-step bucket mean **1.5860** — CONFIRMED. **`POST_DATA.md:57`'s "1.316 (last)" is off by one bucket**: (23000,24000] = 1.3162; the true last bucket (24000,24414], 414 rows, = **1.3138**.

### 5.4 SmolLM2 from-scratch (`checkpoint.pt`) — a 150-step demo, not a reproduction

`train.py:140` `print("Initializing model from scratch (random init)...")`; corpus `Salesforce/wikitext` / `wikitext-103-raw-v1` / `train` (`train.py:78`); seq_len 2048, micro 2 × accum 8 = 32,768 tok/step; AdamW (0.9,0.95) eps 1e-8, peak lr 3.0e-3, wd 0.01 on `dim>=2`, WSD warmup 20 / 20% decay, clip 1.0, bf16. Only recorded run: **150 steps** (~4.9M tokens, DERIVED from defaults — nothing on disk records the demo's batch shape), final loss **6.288341** from start **11.254480** (`results/loss_curve.csv`, 151 lines; `results/summary.json`). `train.py:13-18` calls it a single-GPU starter, not a reproduction.

**Naming trap:** the notebook's 150-step demo cell header and surrounding prose say "wikitext-103 slice" while the code at `_build_notebook.py:436` loads **wikitext-2-raw-v1 train**. Do not write "trained on wikitext-103" for the notebook demo.

---

## 6. Weights: what exists on disk

**Totals (independently re-measured with `os.walk` + `getsize`, symlinks excluded, >1 MB):** **107 weights-bearing files, 290,643,004,043 B = 270.68 GiB = 290.64 GB.** Split by extension: `.pt` n=89 (215.08 GiB), `.pkl` n=17 (52.51 GiB), `.discarded_*` n=1 (3.10 GiB). **Zero `.safetensors`, `.msgpack`, `.npz`, `.pth`, `.ckpt` anywhere in the repo.** Everything the naive `find` pattern matches totals 315,669,745,533 B = 293.99 GiB, because it also catches 26 `tokcache_*.pt` **token caches** (23.92 GB — the largest single `.pt` in the repo, 9,534,229,373 B = 8.88 GiB, is a token cache, not a model) and 10 `research/datasets/**/*.bin` **uint32** token shards (1.11 GB). Volume: 2.5 T free of 3.7 T.

### 6.1 Inventory — the publishable candidates

| File | Size | Format | In-file keys | config.json? | tokenizer files? | Publishable? |
|---|---|---|---|---|---|---|
| `SmolLM2-134(base)/checkpoint_tinystories.pt` | 269,144,681 B (256.68 MiB) | torch.save zip, `compression method=store` (a **pickle**) | `model, config, step, tok_seen, baseline_ppl, trained_ppl` | **NO** — config is the in-file `config` dict | **NO** | **YES** — a fine-tune of `HuggingFaceTB/SmolLM2-135M` (Apache-2.0 upstream); card MUST say so |
| `SmolLM2-134(base)/checkpoint.pt` | 538,173,921 B | torch.save zip, fp32 | `model, config, losses, lrs, step` (step=150) | NO | NO | **NO** — 150-step random-init demo |
| `Qwen3-0.6B/builds/.../checkpoint_qwen3_baseline2tpp.pt` | ~1.19 GB class | pickle | `model, config, step, tok_seen, arm, seed, fineweb_val_ppl, baseline_ppl, recipe` | NO | NO | Maybe — but see §9 (leak-suspect val, n=1) |
| `.../checkpoint_imu1_2tpp_step18000.pt` | 1,193,196,711 B | pickle | `model, config, step` (no `tok_seen`) | NO | NO | **Loads into stock HF Qwen3? NO** — 752,091,220 elems, config carries `use_value_residual/use_layernorm_scaling/use_head_gating`, **and its 423 state-dict keys are prefixed `_orig_mod.`** (torch.compile) |
| `.../checkpoint_prope10_2tpp_.pt` / `_prope25_` | 1,192,229,775 B (step 4000) | pickle | `model, config, step` | NO | NO | **[CORRECTED]** `partial_rotary_factor: 0.1` is **NOT supported by transformers 5.8.0 Qwen3** — grep over `site-packages/transformers/models/qwen3/` returns nothing; `Qwen3Config.__init__` has no such parameter; empirically `Qwen3RotaryEmbedding(Qwen3Config(head_dim=128, partial_rotary_factor=0.1))` yields `inv_freq` of length 64, identical to the default = **full RoPE**. Loading it into stock Qwen3 silently runs the wrong architecture. |
| 6× `checkpoint_{adamw,normuon}_seed{0,1,2}.pt` (42M cohort) | 1,192,229,527 / 1,192,230,159 B | pickle, `step=640, tok_seen=41,943,040` | + `arm`, `seed`, `fineweb_val_ppl` | NO | NO | Research artifacts, not a model release |
| 12× `checkpoint_persist_{168M,420M}_{adamw,normuon}_s{0,1,2}.pt` | 1,192,232,687 / 1,192,233,319 B | pickle | same | NO | NO | Ditto. **`168M`/`420M` are TOKEN BUDGETS — all are the same 596,049,920-param model.** |
| 18× HybridSSM `*.pkl` | 55.60 GiB total; **params-only = 18,792,709,255 B = 17.50 GiB** | **Python pickle wrapping flax msgpack**: `{params: bytes, opt_state: bytes (exactly 2× params), step: int, rng: uint32[2] (post-fix only)}` | — | NO | NO | Only `checkpoint_ssm_base_s0.pkl` has a §C10 suite score; 4 are quarantined/void (below) |
| **Nothing** | — | — | — | — | — | **There is NO "parity-verified reproduction" checkpoint.** Both `verify.py` scripts download the official weights at runtime and save nothing. What is publishable is the *code + the parity artifact*, not a weights file. |

**Format warning for any consumer: every checkpoint in this repo is a `torch.save` / `pickle.dump` file, not safetensors.** The 3.4 GB full-training-state Qwen3 variants (keys add `optim, sched, rng_torch, rng_cuda, rng_numpy, rng_python`) **fail `torch.load(weights_only=True)`** with *"Unsupported global: GLOBAL numpy._core.multiarray._reconstruct was not an allowed global by default"*; they load only under `torch.serialization.safe_globals([...])`.

**Structure of the Qwen3 state dict:** 311 keys, all `torch.bfloat16`, 751,632,384 tensor elements including the tied `lm_head.weight` duplicate = **596,049,920 unique params**. SmolLM2: 273 keys, bf16, 162,826,560 elements = **134,515,008 unique** (49,152 × 576 = 28,311,552 duplicated).

**HybridSSM params (measured by deserializing the flax tree):** `ssm_base` 305,818,368 (266 leaves) · `attn1to3` 324,867,840 (290) · `fullattn` 267,719,424 (218) · `swa128` 277,156,608 (194). All fp32, tied embed 151,936 × 768 = 116,686,848, no separate `lm_head`. Config `d_model 768, n_layers 24, n_heads 12, n_kv_heads 4, vocab 151,936` (`HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/model.py:21-26`). **The folder is named "-0.2B" but no arm is 0.2B total; 0.2B ≈ non-embedding (ssm_base 189,131,520). No file on disk states the convention.**

### 6.2 Must NOT be published

- **Hard-quarantined:** `HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/checkpoint_swa128_nope_85M_s0.pkl.discarded_rng_confound_20260723` (3,325,901,386 B).
- **Comparability-void (iso-FLOP error, +18.88% extra compute):** `checkpoint_swa128_42M_s0.pkl`, `checkpoint_swa128_nope_42M_s0.pkl`, `checkpoint_swa128_85M_s0.pkl`. `c5_evidence_CORRECTION_2026-07-28.md:326-328`: *"The words \"iso-FLOP\" must not be attached to `swa128` or `swa128_nope` in any artifact until those cells are re-run at 4,929 steps / 40,378,368 tokens."* Replacements: `*_isofix_s0.pkl` (both step 4,929).
- **Undocumented second split (my own measurement, not a repo classification):** 10 of the 18 HybridSSM pickles lack the `rng` key (pre-PRNG-fix), 8 carry it. The RNG-confound rationale that quarantined the 85M file applies structurally to every pre-fix file that was resumed. Only one was actually quarantined.
- **7 smoke-test artifacts, 18.88 GiB of pure overhead:** `smoke_{baseline,baseline_resumed,wsd,zloss,arch}.pt` in `2026-06-18_qwen3-0.6b_imu1-deconfound-p1` (16.66 GiB) + `checkpoint_imu1_smoke_step{500,1000}.pt`.
- **12 symlinks** (§C13 control reuse, zero extra disk) with **absolute** paths under `/home/yashb98/Downloads/BuildFromScratch` — they break on any copy/move. Do not `tar` them without dereferencing.

### 6.3 Recommendation: `trust_remote_code` modeling file vs raw checkpoint

**Ship the SmolLM2 TinyStories checkpoint as a standard HF `LlamaForCausalLM` in safetensors. Do NOT ship raw `.pt` pickles, and do NOT reach for `trust_remote_code` for SmolLM2.**

Rationale, all from disk:
1. **SmolLM2 needs no custom code.** Its architecture is stock Llama — the repo's own exporter round-trips into `LlamaForCausalLM` (`SmolLM2-134(base)/scripts/export_to_hf.py:56-59`) and the config comes straight from `AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")`. A `trust_remote_code` release would force every downloader to opt into executing our Python for zero architectural benefit, and our `.pt` files are **pickles** — the exact format `safetensors` exists to avoid.
2. **Qwen3 faithful is likewise stock** — it is a bit-exact re-implementation of an architecture `transformers` already ships. Convert, don't vendor.
3. **`trust_remote_code` is only justified for architectures HF cannot express**: the **modernized/IMU-1** arm (`use_value_residual`, `use_layernorm_scaling`, `use_head_gating`), the **partial-RoPE** arms (the kwarg is silently ignored by transformers 5.8.0 — publishing them without custom modeling code would ship a *wrong* model), and **HybridSSM** (a novel JAX/Flax architecture with no HF class at all). For those, either write a `modeling_*.py` or do not publish weights.
4. **No exporter exists for Qwen3 or HybridSSM.** `export_to_hf.py` is the only conversion script in the repo (a repo-wide grep for `save_pretrained|save_file` over `*.py` finds no other), it covers SmolLM2 only, requires network, and **has never been run to completion** — `hf_export/` does not exist on disk.

**Loader snippet — the export path that actually exists (SmolLM2):**
```bash
cd "/home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)"
python3 scripts/export_to_hf.py --ckpt checkpoint_tinystories.pt --out hf_export/smollm2-135m-tinystories
```
Which internally does (`scripts/export_to_hf.py:56-67`):
```python
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
cfg = AutoConfig.from_pretrained(args.repo)          # HuggingFaceTB/SmolLM2-135M
hf  = LlamaForCausalLM(cfg)
missing, unexpected = hf.load_state_dict(ours_sd, strict=False)
# :60-64 filters the tied lm_head.weight and raises SystemExit on any other mismatch
hf.save_pretrained(out, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(args.repo)
tok.save_pretrained(out)
```
Then the consumer side is ordinary:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("<org>/smollm2-135m-tinystories")   # no trust_remote_code
tok   = AutoTokenizer.from_pretrained("<org>/smollm2-135m-tinystories")
```

**If a raw checkpoint must be consumed directly** (the repo's own pattern, `SmolLM2-134(base)/eval_after_vs_base.py:42-45`):
```python
import torch
from model_full import SmolLM2ForCausalLM, SmolLM2Config
m = SmolLM2ForCausalLM(SmolLM2Config())
ck = torch.load("checkpoint_tinystories.pt", map_location="cpu", weights_only=False)  # PICKLE
m.load_state_dict(ck["model"])
m = m.to(device="cuda", dtype=torch.bfloat16).eval()
```
For Qwen3 checkpoints add the compile-prefix strip that the eval suite uses (`Qwen3-0.6B/experiments/2026-06-16_qwen3-0.6b_eval-faithful/eval_suite.py:97-114`):
```python
ck = torch.load(path, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}   # torch.compile-trained
model.load_state_dict(sd, strict=True)
```

**Backup status: none.** Zero checkpoints are tracked in git (`git ls-files | grep -cE '\.(pt|pth|bin|safetensors|ckpt|pkl|msgpack|npz)$'` → **0**; `.gitignore:19-24`, `HybridSSM-0.2B/.gitignore:2`). No HF Hub copy, no `hf_export/`. The 269 MB TinyStories weights are the **sole copy** of the −45% result. (The surrounding evidence is safe — `tinystories_train.{log,csv}`, `tinystories_{before,after}.txt`, `POST_DATA.md`, `training_recipe_resolved.json` are all git-tracked.)

**No checksums.** **[CORRECTED]** 27 of 29 ledger runs have `lineage.artifact_sha256 = null`; the two exceptions are `2026-06-16_qwen3-faithful_eval-first` (`"checkpoint_qwen3_baseline2tpp.pt@step18150"` — a filename, not a hash) and `2026-07-29_hybrid-ssm-0.2b_fineweb-edu-carding` (`c83b7d608a0ca320ae7b7e41dbee05282f074a004a87a9a90f2f4fd0f5032491`, a dataset-carding run). **No model checkpoint on disk has a verifiable checksum.**

---

## 7. Loader API (real code)

**There is no HF-style API.** Neither model class has `from_pretrained` / `save_pretrained` (grep for `from_pretrained` in `Qwen3-0.6B/model.py` matches only a comment at `:33`; zero hits in `model_full.py`). Neither folder has an `__init__.py`, so **every snippet must `cd` into the model folder or `sys.path.insert` it**. (Note the SmolLM2 folder name contains parentheses and must be quoted in any shell.)

| | Qwen3 | SmolLM2 |
|---|---|---|
| Model class | `Qwen3ForCausalLM(nn.Module)`, `__init__(self, cfg: Qwen3Config)` (`model.py:237-246`) | `SmolLM2ForCausalLM(nn.Module)`, `__init__(self, cfg: SmolLM2Config)` (`model_full.py:238-248`) |
| Config | `@dataclass Qwen3Config` — **14 fields**, all defaulted; `head_dim` is a real settable field (`model.py:35-51`) | `@dataclass SmolLM2Config` — **13 fields** **[CORRECTED: not 12]**, all defaulted; **`head_dim` is a read-only `@property`** (576//9 = 64) — `SmolLM2Config(head_dim=64)` raises TypeError (`model_full.py:28-49`) |
| Weight loader | `load_official_weights_into_ours(ours, hf_state_dict)` in **`verify.py:22`** (not model.py). No key remapping — module names mirror HF exactly; `load_state_dict(strict=False)` then assert only `lm_head.weight` missing and nothing unexpected (`verify.py:39-44`) | same function at `SmolLM2-134(base)/verify.py:22`, body `:39-46` |
| forward | `forward(input_ids, labels=None, attention_mask=None) -> {"logits": (B,T,151936), "loss": scalar-or-None}` — a **plain dict**, indexed `model(x)["logits"]` (`model.py:259-274`) | identical contract, vocab 49,152 (`model_full.py:263-279`) |
| generate | `@torch.no_grad() generate(input_ids, max_new_tokens=64, temperature=0.8, top_k=50) -> Tensor` (prompt+continuation). `temperature<=0` → greedy. **No KV cache — recomputes the prefix each step** (`model.py:276-295`) | same defaults (`model_full.py:281-…`) |
| Tokenizer repo | `Qwen/Qwen3-0.6B-Base` (`verify.py:19`) | `HuggingFaceTB/SmolLM2-135M` (`verify.py:19`) |
| `safe_cuda` dependency | **NONE** in the model file or `verify.py` (grep exit 1). It is a caller-side §C1 obligation honoured by the training/eval scripts (`train_qwen3.py:42,269`; `eval_suite.py:43,202`). `safe_cuda.guard(fraction=0.85, device=0)` no-ops without CUDA (`safe_cuda.py:47,51-52`) | **zero `safe_cuda` hits in the entire `SmolLM2-134(base)/` tree** |

**Two silent traps a card must warn about:**
1. **`attention_mask` disables causal masking.** Both models: `F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=(attention_mask is None))` (`Qwen3-0.6B/model.py:162-167`; `model_full.py:156-161`). Passing an HF-style 2-D padding mask **turns off causality**. Published snippets must not pass `attention_mask`.
2. **`attention_dropout` is dead config.** Declared at `Qwen3-0.6B/model.py:50` and `model_full.py:42`; grep finds **no read site** in either file — dropout is hardcoded to 0.0. Setting it has no effect.

### 7.1 Faithful usage snippet — SmolLM2

Copied from `SmolLM2-134(base)/generate.py` (the repo's own 30-line end-to-end script) with per-line provenance:

```python
# cd "SmolLM2-134(base)"  — flat-module imports, no __init__.py
import torch                                                        # generate.py:7
from transformers import AutoTokenizer, AutoModelForCausalLM        # generate.py:8
from model_full import SmolLM2ForCausalLM, SmolLM2Config            # generate.py:10
from verify import load_official_weights_into_ours, REPO            # generate.py:11  (REPO = "HuggingFaceTB/SmolLM2-135M", verify.py:19)

tokenizer = AutoTokenizer.from_pretrained(REPO)                     # generate.py:15
hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)  # eval_after_vs_base.py:35 (generate.py:16 uses the deprecated torch_dtype=)
model = SmolLM2ForCausalLM(SmolLM2Config())                         # generate.py:18
load_official_weights_into_ours(model, hf.state_dict())             # generate.py:19
del hf                                                              # generate.py:20
model.eval()                                                        # generate.py:21

input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids   # call generate.py:23; prompt verify.py:62
with torch.no_grad():                                               # ADDED — the repo runs this under @torch.no_grad() (verify.py:49)
    logits = model(input_ids)["logits"]                             # verify.py:66
    next_id = logits[0, -1].argmax().item()                         # verify.py:80
print(tokenizer.decode([next_id]))                                  # simplified from verify.py:82  → " the"

out = model.generate(input_ids, max_new_tokens=64, temperature=0.8, top_k=50)      # generate.py:24
print(tokenizer.decode(out[0], skip_special_tokens=True))                          # generate.py:25
```

### 7.2 Faithful usage snippet — Qwen3

**There is no equivalent standalone script for Qwen3** — this is composed from `verify.py` + an experiment `eval_suite.py`, and the composition is flagged:

```python
# cd "Qwen3-0.6B"
import torch                                                         # verify.py:13
from transformers import AutoModelForCausalLM, AutoTokenizer         # verify.py:14
from model import Qwen3ForCausalLM, Qwen3Config                      # verify.py:16
from verify import load_official_weights_into_ours, REPO             # COMPOSED (import form from SmolLM2 generate.py:11); both names real at verify.py:22 and :19; verify.py:86 is __main__-guarded so the import is safe

tokenizer = AutoTokenizer.from_pretrained(REPO)                      # verify.py:50  (REPO = "Qwen/Qwen3-0.6B-Base")
hf_model  = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)  # verify.py:51
ours = Qwen3ForCausalLM(Qwen3Config())                               # verify.py:55
load_official_weights_into_ours(ours, hf_model.state_dict())         # verify.py:56
ours.eval()                                                          # verify.py:57

text = "The capital of France is"                                    # verify.py:60
input_ids = tokenizer(text, return_tensors="pt").input_ids           # verify.py:61
with torch.no_grad():                                                # ADDED — verify.py:47-48 decorates main() with @torch.no_grad()
    our_out  = ours(input_ids)["logits"]                             # verify.py:64
    our_next = our_out[0, -1].argmax().item()                        # verify.py:78  → 12095 = " Paris"
print(tokenizer.decode([our_next]))                                  # simplified from verify.py:80

out = ours.generate(input_ids, max_new_tokens=60, temperature=0.7, top_k=40)  # call form eval_suite.py:190-191; literals eval_suite.py:68
print(tokenizer.decode(out[0], skip_special_tokens=True))                     # eval_suite.py:192
```
Cost note: this loads the full fp32 HF model plus a second full copy (~2 × 596M × 4 B ≈ 4.8 GB), CPU-only, so no `safe_cuda.guard` is needed — matching `verify.py`, which has none. With no KV cache, 60 new tokens = 60 full fp32 CPU forwards of a 596M model — minutes-slow.

---

## 8. CANNOT ANSWER FROM DISK / open gaps

1. **No git tag, and no commit hash stamped in any results file.** `git tag -l` empty; `verify.json` / `perplexity.json` / `parity.log` carry no commit, versions, device, or timestamp. Best anchors (`e791875`, `84a96c0`) both postdate the artifacts they added.
2. **No dataset revision pinned for `HuggingFaceFW/fineweb-edu` sample-10BT.** `train_qwen3.py:151` passes no `revision=`. The exact snapshot behind the 1.19B tokens and behind 13.40/28.65/23.52/46.31 is unrecoverable. (A pinned sha `87f09149ef…` exists at `research/eval/private_heldout_v1/private_prose_v1.txt:455` but belongs to a *later* dataset-forge prep.)
3. **No dataset revision pinned for the SmolLM2 wikitext-2 PPL.** `_build_notebook.py:233` has no `revision=`. (For TinyStories and the SmolLM2 model the resolved shas *were* recovered from the active `HF_HOME`; for wikitext they were not.)
4. **The exact `torch`/`transformers` versions behind Qwen3's `verify.json` (2026-06-08) and `original_vs_repro.txt` (2026-06-09) are undeterminable.** Qwen3 has no requirements file, no script stamps a version, no ledger entry covers those runs. Same for SmolLM2's `parity.log` (2026-05-13) — only its same-day sibling notebook carries a stamp.
5. **cuDNN 91900 and driver 580.142 are recorded nowhere in the repo.** Version drift in those two is undetectable from disk.
6. **The optimizer hyperparameters of the SmolLM2 100M-token TinyStories run** (weight_decay, betas, eps, grad_clip, seed) — no `args=` line in the log, no `training_recipe` key in the checkpoint, no `grad_norm` column in the CSV, and git holds only a *later* script. The cited values are that later version's argparse defaults, plus an external nanotron config.
7. **The micro_batch / grad_accum factorization for that run** — only the product (4,096 tok/step) is logged.
8. **The exact CLI invocation of the SmolLM2 TinyStories run** was never recorded anywhere.
9. **The GPU model for that run** — `log:1` says only `Device: cuda`. "NVIDIA GB10" is prose.
10. **Peak GPU memory for that run** — no `peak_mem_mb` column in the run-era CSV.
11. **The exact source code that produced the SmolLM2 runs is gone** (overwritten 2026-05-19; git holds only the later version).
12. **The six GPU-side SmolLM2 parity numbers have no machine-written backing** — `results/comparison_with_hf.json` is absent; `compare_with_hf.py:259-261` would create it on re-run. The "401-token RoPE" label contradicts `max_length=512` in the code.
13. **`results/tinystories_vs_base.{md,json}` do not exist** — so **no OOD / catastrophic-forgetting / downstream number exists for `checkpoint_tinystories.pt`**. `results/lm_eval/` does not exist either: `scripts/run_lm_eval.sh` has never run (its own `mkdir -p` never fired). **No downstream benchmark (HellaSwag/ARC/MMLU/…) was ever measured for SmolLM2 in this repo.**
14. **No suite-comparable counterpart to 13.40** — the `text-lm-v2` suite was never run against the released Qwen3-0.6B-Base.
15. **No same-slice re-measurement of the released model on `tokcache_1191478400_300000.pt`** — so the true same-slice gap between 28.65/23.52 and the released model is **unknown**.
16. **Whether the released Qwen3-0.6B-Base saw these FineWeb-Edu val documents during its 36T pretraining is undeterminable.** No overlap test was or could be run here.
17. **No `train_flops` artifact for any Phase-B run** — the §C18 iso-FLOP gate was never evaluated for the three-build comparison.
18. **No MFU for any of the 1.19B runs**, and the GB10 device peak is `estimated: True` regardless.
19. **The Phase-B training runs have no ledger entries at all** (only their downstream evals do) — no ledger-recorded wall_clock, gpu_hours, git_commit, `c5_evidence.json`, or `verdict.json`. No smoke-test artifact either.
20. **No 840M rung on the scaling ladder.** `SEEDS` in `score_ladder.py:49` still declares 840,000,000 but no `checkpoint_persist_840M_*.pt` exists. The trend fit rests on exactly 3 budgets.
21. **No per-horizon LR re-tuning anywhere on disk.** Both AdamW 2.4e-3 and NorMuon 0.011 were tuned at 42M and held fixed at 168M/420M (`verdict.json:257`: *"Inherited confound: AdamW/NorMuon LRs tuned at 42M, not re-tuned per horizon"*). Part of the observed fade could be an LR artifact; nothing on disk separates the two.
22. **Three §C25 HARD scaling-battery items were never computed** — `log_rmse_r2`, `holdout_extrapolation_pctdev`, `bootstrap_forecast_ci` (`verdict.json:210-214`). A §C26 figure for the ladder is also missing.
23. **No ledger detail doc for the ladder run itself** — `research/ledger/runs/` has no `2026-07-05_qwen3-0.6b_scaling-persistence.md`.
24. **Nothing measures whether the NorMuon convergence holds at model sizes other than N=596M.**
25. **HybridSSM ladder scores cannot be traced to specific `.pkl` files** — `arch_ladder_scores.json` contains zero checkpoint filenames (regex sweep → empty). The only link is the cell id via `run_arch_ladder.sh:395`.
26. **Whether "HybridSSM-0.2B" means 0.2B total or 0.2B non-embedding params** — no file states the convention.
27. **License/provenance for redistributing HybridSSM weights** — novel architecture but Qwen3 tokenizer + FineWeb-Edu training data; no LICENSE or data-license record beside the checkpoints.
28. **Whether the 8 resumed pre-PRNG-fix HybridSSM checkpoints carry the same confound** that quarantined the 85M one — no assessment document exists; the rng-key partition is my measurement, not a repo classification.
29. **No SHA256 for any model checkpoint** (see §6).
30. **`research/ledger/ledger.json` is currently uncommitted-modified** (`git status`), so ledger values quoted here are working-tree state, not committed state. **The scaling-ladder `verdict.json` and `ladder_bpb.json` — the null's entire evidence — are NOT git-tracked**, while the 42M "win" evidence IS.
31. **`model.py` does not exist in `SmolLM2-134(base)/`** (only `model_full.py`), yet `results/POST_DATA.md:20` cites `wc -l model.py` for a "198 lines" figure, `results/README.md:25` says `param_count.log` is the "output of `python3 model.py`", `results/README.md:4` credits `../model.py`, and `eval_after_vs_base.py:91` would `FileNotFoundError` on it. **The "198 lines" claim and the `python3 model.py` reproduce instruction are unbacked at any line number.** (The param count itself IS backed: `results/param_count.log:1-2` and `tests/test_parity.py:53`.)
32. **`Qwen3-0.6B/README.md:39-40` mis-sources the param count** to `verify.json`, which has no params field. 596,049,920 is genuinely measured, but at `Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/run_sft_seed0.log:7`. Similarly `SmolLM2-134(base)/README.md:60` attributes the token ids `[504, 3575, 282, 4649, 314]` to `results/summary.json`, which has no tokenization key — they come from `results.ipynb` cell 6.
33. **No HF model card, `MODEL_CARD.md`, or exported HF repo exists anywhere under the repo root** — there is no existing Reproduce section on disk to diff these commands against.
34. **`Qwen3-0.6B/model.py:44` hardcodes `max_position_embeddings = 40_960` with the comment "config.json: max_position_embeddings", but the cached `Qwen/Qwen3-0.6B-Base` config.json on this box (snapshot `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, dir mtime 2026-06-08 — the very date `model.py:33` cites) says `32768`.** 13 of 14 defaults match; this one does not. Harmless for weight loading (RoPE buffers are `persistent=False`, so parity still passes) but **the claim "every default matches the Base config.json" must not be repeated, and 40,960 must not be published as the Base context length.** I did not fetch live HF HEAD, so this mismatch is asserted only against the on-disk snapshot.

---

## 9. Reviewer red flags — what would embarrass the author if published as-is

Ranked by how fast a reviewer finds them.

1. **`Qwen3-0.6B/README.md:35-37` states a falsehood that one `grep` disproves:** "identical eval code on the identical 300k-token FineWeb-Edu val slice … so every row is directly comparable." Two different caches, different sha1, different leading tokens. The derived "2.14×" and "1.76×" gaps are cross-slice. Duplicated at `results_overview/plots/README.md:50`.
2. **A `null` verdict is being advertised as a `significant win` in four places.** `Qwen3-0.6B/README.md:52, :175, :243` and `PLOTS_INDEX.md:73` all sell −0.474 bpb as a win with zero mention of the ladder, CONVERGES, or the null. The source run's own `RESULT.md:7` scoped it honestly; the README dropped the qualifier. The 42M ledger entry still says `"verdict": "win"` with a now-false caveat `"no scaling curve"` (`ledger.json:512`), and the ladder run is missing from the technique's `run_ids` so **a ledger query by technique will not surface the null**.
3. **The root `README.md:105-121` account of the ladder is stale in five specifics** (n=2, "+0.073 [−0.038, +0.184]", "not significant at the top", "code_py … +0.192", "slope −0.328 (r² 0.81)") and concludes "Verdict: directional, not a headline — the 420M rung is n=2". It is now n=3 and the top rung IS significant. Two of the repo's own top-level docs disagree with each other and with `verdict.json`.
4. **"Bit-exact / max error 0.0" is CPU-only, and the repo's own GPU numbers break its own gate.** `comparison_with_hf.md:11` records **1.95e-03** per-layer at layer 14 on GPU against a 1e-3 assert. Any card saying "bit-exact" without "CPU fp32, 5-token prompt" is misleading — and the six GPU numbers have **no** machine-written backing file, while `results/README.md:3-4` claims "Every file here is produced live … No values are typed in by hand."
5. **The headline PPL recipes are non-standard in ways that make the numbers incomparable to published values.** SmolLM2's 15.371 uses an overlapping 1024/512 window with **no `-100` masking**, so 62,403 "target tokens" are really 31,743 distinct positions double-counted, over **the first ~11.8% of the split**, with blank rows filtered out of the join. The BPB suite likewise caps at MAX_WINDOWS=200 = the first ~103k tokens, ~99% double-counted. Both prose sites (`results/README.md:38-39`, `POST_DATA.md:34-36`) state "62,403 target tokens" with no qualification.
6. **15.371 is not a number about a model this repo trained.** It characterizes the *official* SmolLM2-135M checkpoint under a nonstandard recipe. Attaching it to either local checkpoint (`checkpoint.pt` = 150-step demo; `checkpoint_tinystories.pt` = TinyStories) would be flatly wrong.
7. **Every Qwen3 headline PPL sits on a leak-suspect val split the repo itself indicts.** `train_qwen3.py:131-136` calls the old splitter's val "the sequential continuation of train, leak-suspect"; the fix landed *after* all four runs. `decontam_report.json` (2026-07-07) covers a different, later cache.
8. **All four Qwen3 headline numbers are n=1, single-seed, no CI, in-distribution val PPL — the metric the repo's own contracts ban as a headline** (`research/eval/base_eval_verdict.md:59`, §C25.7.3; `research/eval/per_stage_eval_batteries.md:9`). The −45.0% TinyStories result is likewise single-seed, single-corpus, in-domain, with **no** iso-FLOP control and **no** post-training OOD measurement (the repo's own note: *"We didn't measure wikitext-2 PPL post-training but it almost certainly got worse"*, `tinystories_summary.md:124-127`).
9. **A published TinyStories PPL that does not match the checkpoint.** `results/tinystories_summary.md:9` says 3.7893; `ck['trained_ppl']` is 3.78995 (rounds to 3.7900). Same doc reports 137.3 min and 12,150 tok/s for what is a **different, earlier run** — labelled `(prior run)` only in `POST_DATA.md:165`.
10. **Two arithmetic errors that propagate from the repo's own prose.** `Qwen3-0.6B/README.md:80` says NorMuon cost "~30% more wall-clock"; it is **+43.9%**. `README.md:167` quotes 7,480 tok/s (the step-100 reading) for a run whose final cumulative rate is 7,444. And three README sites round the CI lower bound to 0.444 where the value is 0.4434.
11. **Broken citations in the results docs.** `model.py` is cited three times and does not exist; `Qwen3-0.6B/README.md:39` sources the param count to a JSON that has no params field; `SmolLM2-134(base)/README.md:60` sources token ids to a JSON that has no tokenization key; `research/ledger/runs/2026-06-17_…md:113-114` mis-attributes 28.65 to `eval_original_vs_repro.py` (it came from the in-loop `evaluate()`); `SmolLM2-134(base)/verify.py:6` claims bf16 tolerance for an fp32 gate.
12. **A checkpoint that silently loads wrong.** `checkpoint_prope10_2tpp_.pt` carries `partial_rotary_factor: 0.1`, a key transformers 5.8.0's Qwen3 **never reads** — stock loading runs full RoPE. The modernized checkpoint carries 423 `_orig_mod.`-prefixed keys and three non-HF config flags.
13. **An unmeasured claim written as if it were a result.** `results/comparison_with_hf.md:86-88`: *"any benchmark score will match by construction."* No downstream benchmark was ever run (`results/lm_eval/` does not exist). This must not be transcribed in any form that reads as a measurement.
14. **The documented reproduce path destroys its own evidence.** `python3 _build_notebook.py` overwrites `results.ipynb` (wiping the only record of the 15.371 run) and the nbconvert pass overwrites `../checkpoint.pt`. Neither the notebook nor `train_tinystories.py` imports `safe_cuda` or runs `sentinel.py preflight`, violating the repo's own §C1/§C6 on a box where over-allocation reboots the machine.
15. **Two contradictory dependency files offered as equivalent** (`pyproject.toml` hard pins vs `requirements.txt` `torch>=2.4`), no lockfile, and **no environment spec at all for Qwen3** — under a README line that says "Install pinned dependencies that produced the 0.0 logit-diff result."
16. **Zero determinism flags repo-wide**, on a repo whose central claim is numerical equivalence, and whose own GPU deltas are attributed to unpinned backend dispatch.
17. **The null's evidence is untracked while the win's evidence is committed.** `verdict.json` and `ladder_bpb.json` for the scaling ladder are working-tree-only; `MEMORY.md` records a prior incident where a branch switch destroyed gitignored evidence. The 269 MB TinyStories weights are similarly a single uncommitted copy.
18. **A stale ledger `running` entry** (`2026-07-28_hybrid-ssm-0.2b_arch-ladder-repair`, `eta_hours 30.66`) with no live process — `sentinel.py preflight` reports `trainers=none`.
19. **`c5_evidence_CORRECTION_2026-07-28.md:326-328` forbids the words "iso-FLOP" on three HybridSSM cells** until they are re-run; one HybridSSM checkpoint is hard-quarantined for an RNG confound and 8 more resumed pre-fix files were never assessed for the same defect.
20. **Reproducibility claims that a reviewer can falsify with `env`.** The tokenizer/dataset snapshots are not in `~/.cache/huggingface`; the active `HF_HOME` is `/home/yashb98/projects/qwen-distill/hf_cache`, **outside the repo**. So a card claiming "reproducible from this repo alone" is wrong — and any earlier claim that the caches don't exist is also wrong.