<!-- Generated 2026-06-24 by workflow wf_ae4dc7ba-516 (interp-skill-build-research): 3 personas (RS@Anthropic / frontier-eng mini-scale / MLOps) recency-first + adversarially verified. Every method WebFetch-rechecked. -->

# Build Spec: On-Box Interpretability Core for Qwen3-0.6B

**Chair synthesis of three verified research positions. Every method below survived a WebFetch adversarial re-check on 2026-06-24. Fabrications/mis-attributions are dropped and flagged in §6.**

Deliverable: `research/interp.py` + 3 new primitives in `research/eval_metrics.py` + `/interpretability` skill, CPU-unit-tested. Build target: the missing **interpretability lifecycle stage** for the 596,049,920-param Qwen3-0.6B trained on ~1.19B tokens.

---

## 1. The honest verdict (state it up front so the build cannot oversell)

**Expected outcome: a CI-backed NULL at the control floor. This is the PASSING deliverable, not a failure.**

At 596M params / 1.19B tokens (~20-40× under Chinchilla, orders of magnitude under the Gemma-scale models where the field's own results were measured), the fetch-verified evidence is unanimous and points *worse* for us than for the papers:

- **arXiv:2602.14111** (verified verbatim 2026-06-24): on real activations, **random baselines match trained SAEs** — interp 0.87(rand)/0.90(trained), sparse-probing 0.69/0.72, causal-editing 0.73/0.72 — and on **clean synthetic** data SAEs recover only **9% of true features despite 71% explained variance**. An undertrained 596M residual stream is a *more* entangled/smeared basis than Gemma's, so our SAEs land **at or below** this floor.
- **SAEBench (search-confirmed)**: "the sparsity-fidelity frontier does not reliably indicate downstream performance" — so a good Loss-Recovered-vs-L0 curve is **not** evidence of recovered features. This is the exact trap.
- **AxBench (2501.17148)**: difference-in-means and prompting beat SAEs even at 2B scale → our DiffMean baseline will beat our SAE; prompting will beat our steering.

**Per-metric prediction (what escapes the floor, if anything):**

| Arm | Predicted result at 596M/1.19B |
|---|---|
| SAE Loss-Recovered-vs-L0 Pareto | **Within CI of PCA + random-SAE floor.** NULL. |
| Feature absorption (2409.14507) | Within CI of floor. NULL. |
| SynthSAEBench-style F1/MCC (synthetic ground-truth) | **The one place a real signal *can* appear** — model isn't in the synthetic loop. Report it, but it is a statement about the *harness*, never about the 596M model. |
| Linear probe AUROC vs **random-init-model** floor | **Likely BEATS the floor on surface features** (token identity, sequence length) — those are linearly present. This is the single arm most likely to clear its floor. |
| Linear probe AUROC vs **length/token-freq confound** floor | Likely **fails to separate** on semantic concepts (probe exploits the confound). |
| DiffMean vs SAE concept detection | **DiffMean wins** (mirrors AxBench). |
| Steering (SAE/DiffMean dir) vs prompting | **Prompting wins.** |
| Sandbagging / eval-awareness | Null by construction (capability absent). **Methodology stub only.** |

**The reportable headline is gated:** the only thing the skill may call a "win" is an arm whose bootstrap-CI on `(SAE − floor)` is **strictly disjoint from 0** against **both** PCA and random. Otherwise the verdict is the honest null: *"at 596M/1.19B, trained SAEs do not recover structure beyond the random/PCA control floor on metrics X, Y, Z; linear probes recover [surface features] above their random-init floor but not above their length/frequency-confound floor."* The §C25 `interpretability` gate (verified at `eval_completeness.py:78`) treats a correctly-measured null as **complete/pass**.

---

## 2. `research/interp.py` — the module (concrete API)

Header on **every** entry point: `import safe_cuda; safe_cuda.guard(0.85)` (signature verified `guard(fraction=0.85, device=0)`). Cache **residual stream d=1024 only — NEVER full logits** (vocab=151,936 fp32 logits is the documented box-crash vector).

### 2.1 ActivationCache — hooks are MANDATORY
```
class ActivationCache:
    def __init__(self, model, layer:int, pool:str='last'):  # pool in {'last','mean','all'}
    def fill(self, token_iter, max_tokens:int, shard_path:str) -> dict  # returns manifest
```
**Verified hard constraint** (`model_imu1.py:263-273`): `Qwen3Model.forward` returns ONLY `self.norm(x)` — there is **no `output_hidden_states` path**. You **must** `register_forward_hook` on `model.model.layers[L]` (verified `nn.ModuleList`, line 255) to capture the block-output residual. Reading the forward return value silently caches the *final-norm* activation (the wrong tensor) and produces a meaningless result. The hook captures `output[0]` (the block returns `(x, v_local)`; take the hidden state, not the value vector).

Writes an fp16 shard + a **manifest** `{ckpt_sha, layer, pool, n_tokens, seed, data_sha}` (reproducibility + anti-laundering: synthetic vs real provenance is pinned).

**Memory (hard-capped in code, assert it):** residual d=1024 fp16 = **2.05 GB / 1M tokens**. Budget **2-5M tokens, single layer → 4-10 GB**, fits the 119 GB pool under guard. The spec **hard-caps `max_tokens ≤ 5_000_000` per layer with an assert** — this is the one place a naive impl re-triggers the unified-memory OOM the whole project guards against (50M tokens = 102 GB = crash). If multiple layers are wanted, stream-and-discard or chunk to disk one layer at a time.

### 2.2 SAE — the architecture (verified-current ONLY)
```
class SAE(nn.Module):
    # variant='batchtopk' (default) | 'matryoshka_batchtopk'
    def __init__(self, d_model=1024, dict_size=16384, k=32, variant='batchtopk',
                 groups=(2048,6144,16384), tied=False, k_aux=512, aux_alpha=1/32)
```
- **Primary: BatchTopK** (arXiv:2412.06410, 2024-12-09, **verified**) — encoder `d→m`, **batch-level top-k selection** (NOT an activation function — verified; docstring must say "batch-level selection method"), decoder `m→d`. Untied weights (standard); `tied=False` default, expose `tied` for ablation.
- **Secondary arm: Matryoshka-BatchTopK** (Matryoshka arXiv:2503.17547, 2025-03-21, **verified**; the combined variant is the **SAEBench Pareto leader for absorption/concept-detection/disentanglement in L0 40-200**, search-confirmed 2026-06-24). Nested dictionaries `groups`; smaller groups must reconstruct without the larger. This is the arch that *directly targets absorption*, so it is the right secondary even though absorption is expected to null out at our scale.
- **AuxK dead-latent revival** (`k_aux`, `aux_alpha`): **UNVERIFIED HYPERPARAMETERS — implementation lore, not in the 2412.06410 abstract** (confirmed absent on fetch). Carry it as an *optional* auxiliary loss reviving dead latents, with `k_aux`/`aux_alpha` exposed as **tunable defaults**, and the docstring **must not cite the paper for these numbers**. Track `dead_features` and report it regardless.
- Dict size sweep `×16/×32/×64` → **33.6M / 67.1M / 134.3M params** (0.13/0.27/0.54 GB fp32 weights, ≤1.6 GB with Adam). Trains in **1-3 hrs single-job** on the GB10.

### 2.3 Control floors — FIRST-CLASS, computed BEFORE any SAE number
```
class PCAFloor:        # truncated SVD of the SAME activation matrix, k=dict_size
class RandomSAEFloor:  # frozen-random decoder directions per arXiv:2602.14111
def random_rotation_floor(acts): ...  # random orthogonal projection
```
All three are CPU/tiny-GPU. The gate (§4) refuses any SAE headline that isn't CI-disjoint from **both PCA and random**.

### 2.4 Metrics (verified-defensible only)
```
def loss_recovered_vs_l0(sae, cache, model, layer) -> [(L0, ce_degradation)]   # splice recon back, measure CE
def absorption_first_letter(sae, ...)   # arXiv:2409.14507
def synth_recovery_f1_mcc(sae, synth)   # small SynthSAEBench-style harness
def probe_auroc_vs_floors(acts, labels, lengths, freqs) -> {auroc, vs_random_init, vs_length_confound}
def diffmean_concept_detection(acts, pos_idx, neg_idx)   # AxBench baseline SAEs must beat
def steer_vs_prompting(direction, model, concept)        # expected to lose to prompting
```
**Every SAE metric is returned as `delta_vs_pca` and `delta_vs_random` with a `bootstrap_ci`.** Pass = CI-disjoint from both; else null.

- **`absorption_first_letter`**: ⚠ **BLOCKING PREREQUISITE** — the absorption-fraction formula and the first-letter task are **NOT in the 2409.14507 abstract** (confirmed on fetch). The build **MUST read the full arXiv HTML (`arxiv.org/html/2409.14507`) for the exact equation before coding this**. Do not implement from memory.
- **`synth_recovery_f1_mcc`**: a **down-scaled, on-box** synthetic harness (generate known features → activations → score recovery with **our own F1/MCC via greedy/Hungarian feature-matching**). The "16384 features / 768-dim / 200M samples / 4096 latents" figures are **NOT abstract-confirmed (UNVERIFIED)** — do **not** cite them; this is the *only* arm with true-feature ground truth, so report it but keep it strictly separate in the ledger from real-model claims.
- **DROPPED, do not implement: TPP and SCR** (arXiv:2605.18229, verified 2026-05-18: "should not be used to evaluate SAEs"). Also dropped as a headline: auto-interp LLM-judge (rented + noisy/low-discriminability per 2605.18229).

**CPU-stub-testable** (`test_interp.py`, no GPU/model/network): tiny synthetic acts; assert random-SAE floor ≈ trained-SAE on random data (**the sanity check itself passes** = the 2602.14111 result reproduces); assert absorption/F1/probe plumbing runs and is seed-deterministic; assert a CI-**overlapping** synthetic case provably yields the **null** verdict (this is the anti-laundering unit test — see §4 KEY RISK).

---

## 3. Missing primitives → `research/eval_metrics.py` (verified ABSENT in-file AND tree-wide)

Pure-stdlib at runtime; `sklearn`/`statsmodels` imported **only inside tests** as the cross-check oracle (mirrors the existing `accuracy_wilson_ci`/`conformal_halfwidth` discipline).

### 3.1 `roc_auc(scores, labels)` — rank-based (Mann-Whitney U), ties-averaged
Equivalent to the trapezoid ROC, deterministic, no sklearn:
```
AUC = (sum_of_tie_averaged_ranks_of_positives − n_pos*(n_pos+1)/2) / (n_pos * n_neg)
```
**Unit test:** perfect ranking → 1.0; inverted → 0.0; `[1,1,0,0]` all-ties balanced → 0.5; cross-check vs `sklearn.metrics.roc_auc_score` on a fixed random fixture (≤1e-9).

### 3.2 `bootstrap_ci(stat_fn, data, n=2000, alpha=0.05, seed, method='percentile', paired=False)`
Seeded `random.Random(seed)` resampler → **byte-identical reruns**. Percentile interval = `[quantile(boots, α/2), quantile(boots, 1−α/2)]`. `paired=True` resamples index *pairs* for `(A−B)` deltas — **required for the CI-disjoint-vs-floor gate**. Optional BCa (bias z0 + jackknife acceleration) behind `method='bca'`.
**Unit test:** constant array → zero-width CI; fixed seed → exact reproducible endpoints; symmetric data → CI brackets the mean; coverage sanity vs a known normal (~95%).

### 3.3 `mcnemar(b, c)` — paired binary test on the 2×2 discordant counts
SAE-probe-correct vs DiffMean-probe-correct **on the same items** → "SAE beats floor" is a *paired* test, not two independent intervals.
- Small-count path (`b+c < 25`): **exact two-sided binomial**, `p = 2·Σ_{i≤min(b,c)} C(b+c,i)·0.5^(b+c)` (clamp ≤1).
- Else: continuity-corrected chi-square `χ² = (|b−c|−1)² / (b+c)` → `p` via chi2(df=1).

**Unit test:** `b==c → p=1.0`; textbook fixture matches `statsmodels.stats.contingency_tables.mcnemar`; assert the small-count branch uses **exact**, not chi².

> **Test-naming collision (verified):** `research/tests/test_bootstrap.py` already tests the **shell** `bootstrap.sh` cron — do **NOT** reuse that name. New tests: extend `test_eval_metrics.py` for the 3 primitives; new `test_interp.py` for the core.

---

## 4. The `/interpretability` skill

**Phases (control-floors-FIRST is a hard order):**
1. **Preflight** — `sentinel.py preflight` (§C6 health gate), `safe_cuda.guard(0.85)`, one-GPU-job rule (§C5/§C4.5).
2. **Cache** — `ActivationCache` on **one layer**, ≤5M tokens fp16, manifest written.
3. **Floors FIRST** — PCA + random-rotation + random-SAE + (for probes) random-init-model + length/freq-confound, all logged **before** any SAE number exists.
4. **Train SAE** — BatchTopK (+ Matryoshka-BatchTopK arm), dict ×16/×32/×64, k swept for the L0 axis.
5. **Metrics-vs-floors** — every SAE metric as `bootstrap_ci(paired)` delta vs PCA and random.
6. **Linear probe** — AUROC vs random-init floor AND length/freq-confound floor; `mcnemar` for SAE-vs-DiffMean on the same items.
7. **Steering** — SAE/DiffMean direction vs prompting baseline (expected to lose).
8. **CI-disjoint gate** — headline = "win" **only if** `bootstrap_ci(SAE − floor)` is strictly disjoint from 0 vs **both** floors; else honest **null** (capped `directional` by `eval_completeness.gate_verdict`).
9. **Figure** — via `eval_plots.figure_for_run`. ⚠ **`figure_for_run` (verified `eval_plots.py:81`) only handles `verdict.json` + `*train*.csv` — it has NO Pareto/AUROC branch.** The build **MUST add a dispatch branch** that renders the Loss-Recovered-vs-L0 Pareto + AUROC-vs-floor grouped bars when an `interp_metrics.json` is present (reuse the existing `grouped_bars`/`lines`/`bar_with_ci` helpers). This is a build-task, not a free reuse.
10. **Ledger** — ONE run via **`ledger.py` CLI `cmd_add_run`** (never hand-edit `ledger.json`), `stage='interpretability'`, `suite='interp-suite-v1'`, items `{control_floor_first, ci_disjoint_vs_baseline, suite_stamped}` (exact required set verified at `eval_completeness.py:78-79`). Keep synthetic-F1 and real-model claims in **separate** ledger fields so a synthetic success cannot be laundered into a real-model feature claim.

**On-box (this build, zero rented spend):** activation cache, BatchTopK/Matryoshka SAE train, PCA/random/confound floors, Loss-Recovered-vs-L0, absorption, down-scaled synthetic-F1, linear probes + AUROC, DiffMean, simple steering, all 3 stats primitives.

**Rented — §C20 propose-only, do NOT run here:** auto-interp LLM-judge (needs hosted judge; also low-value per 2605.18229); SAEBench/MIB/Gemma-Scope reference SAEs on Gemma-2-2B/9B (cross-paper comparison only, not our model); full canonical-scale SynthSAEBench (200M-sample protocol); AxBench at native Gemma scale with judge-scored steering.

**§C5 safety:** `safe_cuda.guard(0.85)` at every entry; one GPU job at a time (no concurrent training); cache residuals only, never logits; `max_tokens` hard-capped + asserted.

### KEY RISK — result-laundering the null into a "win" (structural mitigation, verified necessary)
The dominant failure at this scale is **not** a crash — it's reporting an SAE/probe number that looks like recovered structure but is (a) the random/PCA floor in disguise (the exact 2602.14111 trap), (b) a probe exploiting a length/token-frequency confound, or (c) a retracted metric (TPP/SCR). **The `eval_completeness` gate only checks that the registry ITEMS are *present* — it does NOT itself compute the floor or verify CI-disjointness** (verified: `gate_verdict` checks item presence, `eval_plots`/`eval_completeness` never touch activations). So the "physically unable to emit a win when CIs overlap" property **must live in `interp.py`**: the random-rotation/PCA floor is a first-class arm computed by `bootstrap_ci` **before** any SAE number is recorded, and there is a **unit test asserting a CI-overlapping case yields the null verdict**. Item-presence is necessary but not sufficient — the disjointness check is the load-bearing code.

---

## 5. Pinned facts

| Fact | Value | Source / verification |
|---|---|---|
| Layer to hook | **layer 14** (mid-depth of 28); optionally also 7, 21 | one layer at a time for the memory cap |
| Hook target | `model.model.layers[L]`, capture block-output residual (hook on `output[0]`) | `model_imu1.py:255` `nn.ModuleList`; forward returns only `self.norm(x)` (line 273) — **hooking mandatory** |
| d_model / layers / vocab | **1024 / 28 / 151,936** | `model_imu1.py:39,41` + verified |
| Checkpoint | `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/checkpoint_qwen3_baseline2tpp.pt` | exists on disk (verified) |
| Activation budget | **2-5M tokens, single layer, fp16 = 4-10 GB**; HARD CAP `max_tokens ≤ 5M` asserted | 2.05 GB/1M tok; 50M tok = 102 GB = OOM crash |
| Crash guard | `import safe_cuda; safe_cuda.guard(0.85)` first; residuals only, never logits | `safe_cuda.guard(fraction=0.85, device=0)` verified |
| SAE dict / params / train | ×16/×32/×64 = **33.6/67.1/134.3M params**, ≤1.6 GB w/ Adam, **1-3 hrs single-job** | cache-fill ~3-10 min |
| Stats primitives | `roc_auc`, `bootstrap_ci`, `mcnemar` — **VERIFIED ABSENT in `eval_metrics.py` and tree-wide** | grep confirmed |
| Test names | extend `test_eval_metrics.py`; new `test_interp.py` — **NOT `test_bootstrap.py`** (collision with shell-bootstrap test, verified) | |

**Methods carried forward (all WebFetch-verified 2026-06-24, ✓ = current-best):**

| Method | arXiv | Date | Status |
|---|---|---|---|
| Sanity Checks for SAEs (random==trained; 9% recovery/71% EV) | 2602.14111 | 2026-02-15 | ✓ current-best, **load-bearing null evidence** |
| Are SAE Benchmarks Reliable? (drop TPP+SCR; sae-probes most reliable) | 2605.18229 | 2026-05-18 | ✓ current-best |
| BatchTopK SAE (**primary arch**; training/selection method) | 2412.06410 | 2024-12-09 | ✓ current-best (low-L0) |
| Matryoshka SAE (**Matryoshka-BatchTopK = SAEBench leader for absorption/concept-detect, L0 40-200**) | 2503.17547 | 2025-03-21 | ✓ current-best (interp metrics) |
| SAEBench (arch leaderboard; "frontier ≠ downstream") | 2503.09532 | 2025-03 | ✓ usable |
| A is for Absorption (NeurIPS 2025 Oral) | 2409.14507 | 2024-09-22 | ✓ current-best — **must read full HTML for formula** |
| SynthSAEBench (synthetic ground-truth) | 2602.14687 | 2026-02-16 | ✓ usable (down-scaled on-box) |
| AxBench (DiffMean/prompting > SAEs) | 2501.17148 | 2025-01-28 | ✓ current-best |
| Probing/Steering Eval-Awareness | 2507.01786 | 2025-07 | usable (methodology STUB only — null by construction) |

---

## 6. Adversarial check: what was dropped / corrected / flagged UNVERIFIED

**DROPPED outright:**
- **TPP and SCR metrics** — arXiv:2605.18229 (verified): "should not be used to evaluate SAEs."
- **Auto-interp LLM-judge as a headline** — rented + noisy/low-discriminability (2605.18229).
- **Any "we found N interpretable features" absolute claim** — refused at this scale.

**CORRECTED mis-attributions (verified wrong, do not propagate into the skill):**
- **AxBench introduces "supervised dictionary learning (SDL)"** → **FALSE**. WebFetch-confirmed: AxBench introduces **ReFT-r1 (Rank-1 Representation Finetuning)**, a weakly-supervised representational method. The load-bearing conclusions (DiffMean best for detection, prompting beats all for steering, SAEs not competitive) are CORRECT; only the named contribution was wrong.
- **Eval-awareness paper (2507.01786) "Claude family"** → the paper names **Llama-3.3-70B-Instruct**, not Claude. Minor; doesn't change the null-stub verdict.

**Flagged UNVERIFIED — the build MUST NOT cite these as paper facts:**
- **BatchTopK "AuxK, k_aux=512, alpha=1/32"** — NOT in the 2412.06410 abstract (confirmed absent). Implementation lore. Expose as tunable defaults; docstring must not cite the paper for them.
- **AxBench "DiffMean 0.942 vs SAE 0.695 AUROC"** — exact numbers are PDF-body, not abstract-confirmed. Carry the *direction* (DiffMean wins) as verified; the exact figures are not abstract-confirmed.
- **SynthSAEBench "16384 features / 768-dim / 200M samples / 4096 latents / 15-20 min H100" and "F1 vs PCA vs random"** — NOT abstract-confirmed. Build the down-scaled harness with our own F1/MCC; do not cite these numbers.
- **Absorption-fraction formula + first-letter task (2409.14507)** — NOT in the abstract. **BLOCKING: read `arxiv.org/html/2409.14507` before coding `absorption_first_letter`.**

**Scale honesty: confirmed FLOOR-level.** No reviewer softened the null. The single strongest residual risk is the **unpinned activation-cache token budget** — now hard-capped at ≤5M tokens/layer with an in-code assert, which is the one place a naive impl re-triggers the unified-memory OOM the project exists to prevent. Every other claim (model is hookable via ModuleList; the 3 primitives genuinely absent; `figure_for_run` needs a new branch; `eval_completeness` checks item-presence only so the CI-disjoint logic must live in `interp.py`) is verified against source on this box.

Relevant files: `/home/yashb98/Downloads/BuildFromScratch/research/interp.py` (new), `/home/yashb98/Downloads/BuildFromScratch/research/eval_metrics.py` (+3 primitives), `/home/yashb98/Downloads/BuildFromScratch/research/eval_plots.py` (+interp dispatch branch), `/home/yashb98/Downloads/BuildFromScratch/research/eval_completeness.py:78` (interpretability registry — unchanged), `/home/yashb98/Downloads/BuildFromScratch/research/tests/test_eval_metrics.py` (+primitive tests) and `/home/yashb98/Downloads/BuildFromScratch/research/tests/test_interp.py` (new), model at `/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/model_imu1.py`, checkpoint at `/home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/checkpoint_qwen3_baseline2tpp.pt`.
