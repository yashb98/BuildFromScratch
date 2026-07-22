# NorMuon vs AdamW on Qwen3-0.6B 2D Weights — Single-Variable Iso-FLOP Ablation (42M tokens)

**Run ID:** `2026-06-16_qwen3_normuon-vs-adamw` · **Suite:** text-lm-v2 · **Verdict:** WIN (scoped) · **Status:** verified against on-disk evidence (all 6 logs, both result JSONs, scorer, stats gate, NorMuon impl re-read; CI reproduced bit-for-bit).

## 1. Headline (correctly scoped)

> At a **fixed 42M-token, iso-FLOP budget** (640 steps × 65,536 tok) on an **identical faithful Qwen3-0.6B** (440M non-embedding params), swapping **AdamW → NorMuon on only the 2D hidden weights** — everything else (architecture, data, fixed split seed 0, schedule shape, embedding/1D treatment, weight decay) held byte-identical — improves text-LM **bits-per-byte by +0.474 on wikitext-2 (95% CI [+0.443, +0.505]) and +0.502 on code ([+0.456, +0.547])**, 3 seeds/arm, fully disjoint arms, significant. A **matched-config AdamW LR sweep (1.7/2.4/3.5/4.8e-3)** shows AdamW's BPB is **flat within seed noise across 1.7–3.5e-3** and **no AdamW LR comes within ~10× of closing the gap** (full LR spread ~0.047 bpb vs the 0.474 gap), so this is **NorMuon vs AdamW-anywhere-in-a-reasonable-LR-range**, not an undertuned baseline. **This is an early-training optimization-speed signal at one architecture and one budget; we do NOT claim it holds at scale.**

The qualifiers are load-bearing — see Limitations. The unqualified claim "NorMuon gives a 22% BPB improvement" would be an over-claim and is not what this run shows.

## 2. Results

**wikitext-2-raw-v1 (val, pinned rev `b08601e…`)** — lower is better

| Arm | Seed BPB | Mean | ±SEM | fineweb-val PPL |
|---|---|---|---|---|
| AdamW @ 2.4e-3 | 2.1050, 2.1024, 2.1221 | **2.1098** | 0.0062 | 147, 145, 156 |
| NorMuon @ 0.011 | 1.6499, 1.6248, 1.6317 | **1.6355** | 0.0075 | 61, 60, 60 |
| **Improvement (AdamW − NorMuon)** | | **+0.4743** | | |
| **95% CI (Welch-t, df 3.86→3, t=3.182)** | | **[+0.4435, +0.5052]** | significant ✓ | |

**codeparrot-clean-valid (pinned rev `4db92d2…`, 500k chars)**

| Arm | Mean BPB | ±SEM | Improvement | 95% CI |
|---|---|---|---|---|
| AdamW | 3.3847 | 0.0099 | — | — |
| NorMuon | 2.8831 | 0.0104 | **+0.5016** | **[+0.4560, +0.5471]** ✓ |

Arms are **fully separated**: worst NorMuon (1.6499) ≪ best AdamW (2.1024). Within-arm SD ≈ 0.011–0.013; the gap is ≈40 within-arm SD. I re-ran `seed_delta_significant` on the raw seeds and it reproduces the recorded CI, df, and verdict **exactly**.

**Throughput / MFU.** Both arms trained cleanly at ~52.4 GB peak (under the 0.85 unified-memory guard), AdamW ~7,340 tok/s, NorMuon ~6,664 tok/s (NorMuon's Newton–Schulz orthogonalization is the ~9% overhead). Reported **MFU ≈ 29%** uses a **GB10 bf16-dense peak of 125 TFLOP/s that is an *estimated* spec number, not a measured device roofline** — treat the MFU figure as approximate.

**Training health (verified all 6 logs).** Every cell trained monotone-down with no NaN/spike/plateau-from-instability. AdamW seed0 loss 8.18→5.15, grad-norm decaying smoothly 1.0→0.22; NorMuon seed0 loss 7.94→4.25, grad-norm 1.9→0.17. The win is **not** an artifact of a broken or divergent baseline — AdamW was a healthy run that simply converged slower at this budget.

## 3. Why this matters

The project's IMU-1 matched-compute result (NorMuon **bundled with ~5 other changes** — partial-RoPE, WSD-to-zero, etc. — at 1.19B tokens, −17.9% PPL) had to mark optimizer attribution as an explicit **limitation**: it could not say how much of the gain came from the optimizer versus the architecture/schedule changes. This run **isolates the optimizer as a single variable** (`train_ablation.py` changes only how the 196 2D matrices are stepped; the 114 embedding/1D params are AdamW@2.4e-3 wd=0 in *both* arms; 2D wd=0.1 in both; data split seed 0 fixed). It directly answers that open question at this budget: at 42M tokens, the NorMuon update rule **alone** moves BPB substantially. That clean attribution — not the magnitude — is the result.

## 4. Limitations (every surviving red-team caveat, stated plainly)

1. **AdamW LR — RESOLVED by a matched-config sweep (no longer an open caveat).** The original concern was that AdamW's 2.4e-3 was tuned at a 28×-longer budget and might be undertuned for this 42M-token horizon. We ran a **confirmatory AdamW LR sweep at the EXACT ablation config** (42M tok, wd=0.1, 2D-AdamW split, seed 0, same cached data): wikitext BPB = 1.7e-3 → **2.1246**, 2.4e-3 → **2.1098** (the 3-seed cohort mean), 3.5e-3 → **2.1223**, 4.8e-3 → **2.1569**. **Honest reading (not a clean U-shape):** AdamW is **flat within seed noise across 1.7–3.5e-3** — those three span only **0.015 bpb**, inside the cohort's ±0.011 seed band, so 2.4e-3 vs 3.5e-3 is a statistical tie and we do **not** claim a single exact optimum — and degrades only at 4.8e-3. The conclusion is **robust to which LR is best**: the **full AdamW LR spread is ~0.047 bpb, ~10× smaller than NorMuon's +0.474 advantage**, so **no AdamW LR in the swept range comes close to closing the gap**. The headline is therefore **NorMuon vs AdamW-anywhere-in-a-reasonable-LR-range**, not an undertuned-baseline artifact. **Caveat:** the off-baseline LRs are **single-seed** (only the 2.4e-3 point has 3 seeds), so the within-band ordering is unresolved — but the load-bearing conclusion (no LR closes the gap) does not depend on it. wd is held at 0.1 for single-variable isolation rather than AdamW's own 0.01 — a deliberate design choice (see #2). Evidence: `results/lr_sweep_bpb.json` (3 off-baseline points) + `results/verdict.json` (2.4e-3, 3 seeds), `results/adamw_lr{17,35,48}_seed0.log`.

2. **Cross-arm weight-decay provenance.** Both arms use 2D wd=0.1 (NorMuon's/IMU-1's tuned value). AdamW's faithful recipe was tuned at wd=0.01. So AdamW runs at a 10× wd it was never tuned for, on a budget it was never tuned for — part of the gap may be baseline handicap rather than genuine optimizer advantage. (wd is *held equal* across arms, which is correct for single-variable isolation, but it is not AdamW's own tuned wd.)

3. **42M tokens is deep in the early-training regime where Muon-family optimizers are most flattered.** Both arms are ~28× under-trained versus the faithful 1.19B baseline (wikitext BPB **1.2256**) — both 42M models (2.11 / 1.64) are far worse than that baseline. Orthogonalized/RMS-matched updates help most exactly here; the same NorMuon inside the full IMU-1 bundle at 1.19B gave only −17.9% PPL, so the 22–75% relative gap here is **expected to compress substantially with budget and may largely vanish**. There is **no scaling curve** (≥3 budgets) — nothing licenses extrapolation.

4. **Seeds vary init + DataLoader shuffle on a FIXED data split (seed 0).** Corpus-resampling variance is structurally excluded, so the reported ±0.006–0.008 SEM **under-estimates true end-to-end seed variance**; the CI is narrower than a fully-randomized design would give. The verdict is robust by a wide margin (breaking significance needs ~15× SEM inflation at the actual df), but the claim must be stated as **"significant under init+shuffle variance on a fixed split,"** not as a fully-randomized 3-seed result. n=3 is also exactly the gate minimum (df floored to 3, t=3.182 — honestly wide, not "comfortably large").

5. **Scope.** One architecture (Qwen3-0.6B), two corpora, one optimizer pair, one budget, one seed-triple per arm. The BPB *scoring* is clean (sum_nll/ln2/bytes; byte denominator and SEQ=1024/STRIDE=512/pinned corpora bit-identical across all 6; all loads `strict=True`; checkpoints identical size; an independent in-training fineweb-val PPL on a different corpus reproduces the same ~2.4× gap) — the caveats above are about the **comparison**, not the metric.

## 5. Verdict

**Yes — this is a genuine, publishable miniature result, scoped tightly.** It is a clean single-variable, iso-FLOP, multi-seed ablation with a correctly-computed conservative Student-t CI, fully disjoint arms, healthy baseline training, and an independent corpus corroborating the gap. It legitimately isolates what the bundled IMU-1 paper could not attribute. But it is **a convergence-speed result at 42M tokens with an un-re-tuned, foreign-wd AdamW baseline and a fixed-split seed design — not a steady-state quality claim and not evidence the gap survives at scale.** Reported with all five caveats, an Anthropic RS would sign it as "NorMuon wins the early-training optimizer race on Qwen3-0.6B's 2D weights at 42M tokens, single variable, n=3 — needs a scaling curve and a per-horizon AdamW LR sweep before any general claim." Without those caveats, it would be an over-claim and should not ship.

## 6. Reproducibility

**Commands** (sequential on one GB10; each cell ~95–106 min):
```
# 6 training cells — only --optimizer and --seed vary
for opt in adamw normuon; do for s in 0 1 2; do
  python train_ablation.py --optimizer $opt --seed $s --steps 640
done; done
# defaults: --peak_lr 2.4e-3  --normuon_lr 0.011  --weight_decay 0.1  --grad_clip 1.0  --mem_fraction 0.85
# then score all 6 checkpoints through text-lm-v2:
python score_cohort.py     # -> results/cohort_bpb.json
# verdict (Welch-t via research/eval_stats.seed_delta_significant) -> results/verdict.json
```
- **Budget:** 640 steps × (SEQ 4096 × micro-batch 4 × grad-accum 4 = 65,536 tok) = **41,943,040 tok/cell**, 6 cells.
- **Single variable:** 196 2D non-embed matrices → {AdamW@2.4e-3 | NorMuon@0.011}; 114 embedding/1D params → AdamW@2.4e-3 wd=0 (both arms); 2D wd=0.1 (both arms); shared cosine schedule, warmup 50, end-ratio 0.1, global grad-clip 1.0; bf16; `torch.compile`.
- **Seeds:** per-cell `--seed` sets init + shuffle only; **data split `SPLIT_SEED=0` fixed across all 6 cells** (identical corpus, token-cached).
- **Decontam:** FineWeb-Edu stream, decontam dropped **0/451** val docs (`results/decontam_report.json`); eval corpora (wikitext-2, codeparrot) are independent public sources, not FineWeb-Edu — no train→eval leak path.
- **Pinned eval corpora:** `Salesforce/wikitext` wikitext-2-raw-v1 rev `b08601e04326c79dfdd32d625aee71d232d685c3`; `codeparrot/codeparrot-clean-valid` rev `4db92d2ec0c1b4c41eeb439cfae16854511d9dcd` (500k chars). SEQ 1024 / STRIDE 512 / 200 windows; tokenized once, reused for all 6 → n_bytes 869,710 (wikitext) / 843,643 (code) identical across cells.
- **Checkpoints:** `results/checkpoint_{adamw,normuon}_seed{0,1,2}.pt` (AdamW 1,192,229,527 B; NorMuon 1,192,230,159 B — identical `Qwen3Config()`, loaded `strict=True`).

**Evidence files** (all under `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/`): `results/{adamw,normuon}_seed{0,1,2}.log`, `results/cohort_bpb.json`, `results/verdict.json`, `train_ablation.py`, `score_cohort.py`, `normuon.py`; stats gate `research/eval_stats.py`; faithful baseline `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/` (wikitext BPB 1.2256 @ 1.19B tok).