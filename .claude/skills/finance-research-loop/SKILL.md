---
name: finance-research-loop
description: End-to-end research → fine-tune → evaluate loop for the from-scratch models under BuildFromScratch/. This is FINE-TUNING ONLY — continued pretraining of the existing base checkpoint on finance-domain data, never from-scratch retraining. Surveys the FULL space of current fine-tuning methods (PEFT, full-parameter, layer-wise unfreezing, distillation, continued-pretraining variants, anti-forgetting recipes, etc.) using web search dated to the current run (never a frozen date), validates reported results with a second-pass search, writes a comparative `finetune_techniques_plan.md` so the user can decide which method to go with, then runs a hyperparameter-tuning phase for the chosen method before training. Applies ONE method per run and writes a side-by-side comparison notebook vs the base checkpoint. Mirrors the existing train_tinystories.py / eval_after_vs_base.py pattern in the model folder. Pauses for approval at every phase. Invoke with `/finance-research-loop` or when the user asks to "research and improve" a from-scratch model.
---

# finance-research-loop

A phase-gated workflow. Each phase has a goal, the exact tool calls to make, and an approval gate that MUST pass before the next phase starts. Never skip a gate. Never combine two phases into a single user prompt.

The domain is fixed to **finance**. The training mode is fixed to **fine-tuning** (continued pretraining of the existing base checkpoint). The model is whichever subfolder of `BuildFromScratch/` the user picks in Phase 0.

**Dates are never hardcoded.** Every reference to "today" or recency cutoffs must be derived at runtime from `Bash date` (see Phase 2 step 1). The skill file deliberately contains no fixed dates — if you're tempted to write one, run `date` instead.

If the user interrupts to redirect, drop the current phase and re-enter at the phase they're describing.

## Pattern to mirror (read these first, every run)

Before Phase 2, `Read` these existing files in `MODEL_DIR` (whichever exist — names may differ slightly across model folders):

- `train_tinystories.py` — the canonical fine-tune script in this repo. Load base weights, tokenize a domain corpus, WSD-schedule AdamW with LR 3e-4, bf16, token-budget loop, save `checkpoint_<domain>.pt`, log to `results/<domain>_train.{csv,log}`, save `results/<domain>_{before,after}.txt`.
- `eval_after_vs_base.py` — the canonical eval. In-domain PPL vs. OOD PPL (wikitext-2 + code) vs. OOD generations, saved to `results/<domain>_vs_base.{md,json}`. This catches catastrophic forgetting, which is the #1 risk of continued pretraining.
- `train.py` — provides `PackedTextDataset` and `make_wsd_scheduler`. Import from there, don't reimplement.
- `verify.py` — provides `REPO` and `load_official_weights_into_ours` for loading the official HF weights as the baseline.
- `model.py` / `model_full.py` — the from-scratch model definition.

Your `train_finance_<method>.py` and eval scripts in Phase 7/8 should mirror these almost line-for-line, swapping the dataset, the method-specific bits, and the output paths. Same recipe, same logging shape, same eval structure (in-domain + OOD). Do not introduce a new pattern unless the method forces it.

---

## Phase 0 — Pick the model folder

**Goal:** know which from-scratch model this run targets.

1. `Bash`: `ls -1 /home/yashb98/Downloads/BuildFromScratch/ | grep -v '^\.' | grep -v '^experiments$'` to list candidate model folders.
2. `Bash`: for each candidate, run `ls <folder>` and note presence of `model.py`, `train.py`, `*.ipynb`, `*.pt` checkpoints.
3. `AskUserQuestion` (header: "Model folder", single-select): list the candidates as options, with a one-line description per option showing the files found (e.g., "model.py, train.py, checkpoint.pt (538MB), results.ipynb"). Always include "Other" via default.

**Gate:** user picks one folder. Save its absolute path as `MODEL_DIR` for the rest of the run.

---

## Phase 1 — Pick the survey scope

**Goal:** decide whether this run surveys the full universe of fine-tuning methods (default) or narrows to one axis. Either way, the survey in Phase 2 must be comprehensive *within* the chosen scope — never collapse to a single technique like LoRA.

1. `Read` `<MODEL_DIR>/README.md` if present, and `<MODEL_DIR>/model.py` (full) to understand the architecture in front of you. Skim `train.py` and `train_tinystories.py` for the current training setup.
2. `AskUserQuestion` (header: "Survey scope", single-select):
   - **Comprehensive (Recommended)** — survey ALL relevant fine-tuning methods across every axis below. Best when you don't yet know which class of method fits.
   - **Narrow to one axis** — if the user picks this, ask a second question to choose the axis.
3. If "Narrow to one axis", ask `AskUserQuestion` (header: "Axis", single-select) with these axes. All axes must be fine-tuning-compatible; any technique requiring from-scratch retraining is out of scope.
   - PEFT (LoRA, QLoRA, DoRA, IA³, BitFit, adapters, prefix-tuning, prompt-tuning, LoKr, LoHa, VeRA, etc.)
   - Full-parameter fine-tuning recipes (optimizer choice, LR schedule, warmup/decay shape, weight averaging/EMA, sequence packing, gradient clipping)
   - Partial / surgical fine-tuning (layer-wise unfreezing, head-only, last-N-layers, gradient masking, surgical fine-tuning by layer importance)
   - Anti-forgetting / continued-pretraining variants (replay mixing, KL/EWC regularization, sharpness-aware minimization, model averaging vs base)
   - Distillation / teacher-guided fine-tuning (logit distillation, hidden-state distillation, on-policy vs off-policy data)
   - Data / problem (finance corpus mixture, dedup, packing, masking, instruction reformatting of filings, tokenizer-domain alignment)
   - Evaluation (in-domain benchmarks, OOD probes, calibration, generation eval, contamination checks)
   - Observability (per-layer loss, attention/gradient diagnostics, activation stats, dead-neuron tracking during fine-tune)
   - Testing (unit tests for new modules, numerical equivalence vs reference impl, training-loop smoke tests)
   - Design / inference (decoding tricks, KV-cache, quantization of fine-tuned weights, speculative decoding, merge-vs-keep-adapter)

**Gate:** user picks scope. Save as `SCOPE` ∈ {"comprehensive", "<axis-name>"}.

---

## Phase 2 — Research fine-tuning methods + write the plan document

**Goal:** produce a comprehensive, evidence-backed comparison of fine-tuning methods (within `SCOPE`) for a model of this size on finance, written as a structured decision document the user can read end-to-end and pick from. The output is the `finetune_techniques_plan.md` document — that is the deliverable of this phase.

Aim for **6–10 candidate methods** if `SCOPE` is comprehensive, **4–6** if narrowed to one axis. Never collapse to a single technique just because it's well-known (e.g., LoRA): always present alternatives so the choice is informed.

### Step 1 — Anchor the date

1. `Bash`: `TODAY=$(date +%Y-%m-%d) && CUTOFF=$(date -d '6 months ago' +%Y-%m-%d) && echo "today=$TODAY cutoff=$CUTOFF"` — capture both values from the actual clock. NEVER hardcode either date in queries or documents; substitute the values you just printed.
2. State the values in your first user-facing message of this phase ("Today: $TODAY · Recency cutoff: $CUTOFF").

### Step 2 — Cast a wide net (web search)

3. `WebSearch` queries to run in parallel. Replace `<size>` with the model's param count (read from `model.py` / `README.md`) and `<YYYY>` with the year(s) from `$TODAY` and `$TODAY - 1`. Issue queries for both years. All queries are scoped to fine-tuning / continued-pretraining, never from-scratch:
   - General landscape: `fine-tuning methods small language model <YYYY> survey`
   - Sized to model: `fine-tuning ~<size>M parameter LLM <YYYY> recipe`
   - Domain-relevant: `domain adaptation continued pretraining finance <YYYY> results`
   - PEFT broad: `parameter efficient fine-tuning <YYYY> comparison LoRA QLoRA DoRA adapters BitFit`
   - Full-FT vs PEFT: `full fine-tuning vs LoRA <YYYY> small model benchmark`
   - Anti-forgetting: `catastrophic forgetting continued pretraining replay rehearsal <YYYY>`
   - Distillation: `knowledge distillation language model fine-tune <YYYY>`
   - Model-family-specific if applicable: e.g. `SmolLM2 fine-tune <YYYY> improvements`
   - If `SCOPE` is narrowed, add 2–3 more queries focused on that axis specifically.
4. For each promising method that surfaces, run a **second** `WebSearch` of the form `"<method name>" fine-tune results perplexity benchmark <YYYY>` to confirm independent replication and reported numbers. If only the original paper abstract turns up, flag as `claim, unreplicated` in the plan doc.
5. Filter out any method that requires from-scratch retraining or breaks checkpoint compatibility with the existing base. Record filtered ones with a one-line reason.
6. If fewer than 5 strong methods come back, widen the recency to 12 months (`date -d '12 months ago' +%Y-%m-%d`) and re-search. If still thin, surface that to the user and ask whether to broaden `SCOPE` before proceeding.

### Step 3 — Scaffold the experiment folder

7. Build `EXPERIMENT_ID = "${TODAY}_${SCOPE}_<short-method-slug-tbd>"` (use `tbd` until the user picks at the end of this phase).
8. `Bash`: `mkdir -p <MODEL_DIR>/experiments/<EXPERIMENT_ID>`
9. `Write` `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/sources.json` — raw search results (title, url, snippet, published date).

### Step 4 — Write `finetune_techniques_plan.md` (the decision document)

10. `Write` `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/finetune_techniques_plan.md` with this exact structure:

    **Header**
    - Date: `$TODAY` · Recency cutoff: `$CUTOFF` · Model: `<MODEL_DIR>` · Param count · Scope: `$SCOPE`
    - Baseline reference: how the base checkpoint is loaded (HF repo or local `.pt`)

    **Comparison table** (the decision aid — must be a real markdown table)
    | Method | Trainable params | Memory cost | Compute cost | Reported lift (in-domain) | Forgetting risk | Implementation complexity | Replicated? | Best for |
    |---|---|---|---|---|---|---|---|---|

    One row per candidate method. Numbers cited from the second-pass search; if a cell is unknown, write `unknown` — never invent.

    **Per-method detail sections** (one `### <method name>` block each, ordered by recommendation)
    - Source: title + URL + publish date
    - One-paragraph description (what does it actually do)
    - Reported results (verbatim quotes with citations; note the scale they were reported at)
    - Why it might / might not fit a ~<size>M model fine-tuned on finance
    - Compatibility notes (does it touch state-dict shape; does it merge cleanly; quantization implications)
    - Risks and known failure modes
    - Estimated hyperparameter search surface (handed to Phase 3)

    **Filtered-out methods** (with one-line reasons)

    **Recommendation** — your top 1–3 picks with a single-sentence rationale each. Be honest about uncertainty.

    **Open questions** that the user should weigh in on.

11. `Write` `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/research.md` as a thinner log of how the research was conducted (queries issued, sources scanned, why some were dropped). This is the audit trail; the plan doc is the decision artifact.

### Step 5 — Decide

12. Present the plan doc path to the user and a one-screen summary (the comparison table + top-3 recommendation).
13. `AskUserQuestion` (header: "Method", single-select): up to 4 of the highest-ranked methods with a one-line summary per option. Always allow "research deeper" / "narrow scope and redo" via "Other".

**Gate:** user picks ONE method. Rename `EXPERIMENT_ID` to use the real slug (e.g., `${TODAY}_${SCOPE}_lora-r16`), and `Bash` `mv` the experiment folder accordingly.

---

## Phase 3 — Hyperparameter tuning for the chosen method

**Goal:** research the recommended hyperparameter ranges for the chosen method at this scale on this kind of corpus, propose candidate configurations, and let the user pick one (or opt for a mini-sweep). Without this phase, "we tried LoRA" means nothing — `lora_r`, `lora_alpha`, target modules, LR, etc. determine whether it works.

### Step 1 — Research the HP surface

1. `Bash`: re-anchor `TODAY` and `CUTOFF` if any time has passed.
2. `WebSearch` queries scoped to the chosen `METHOD` (use the slug from Phase 2). All queries year-stamped with `$TODAY`'s year and the previous year:
   - `"<METHOD>" hyperparameters small language model fine-tune <YYYY>`
   - `"<METHOD>" learning rate batch size recommendation LLM`
   - `"<METHOD>" ablation <YYYY>` — find papers that swept HPs and reported sensitivity
   - Method-specific: e.g. for LoRA, `"LoRA" rank alpha target modules ablation`; for full FT, `continued pretraining learning rate small model warmup`; for distillation, `distillation temperature alpha weighting`.
3. For each HP, note: (a) the reported sensible range, (b) which papers reported sensitivity to it (so you know which knobs matter), (c) any interaction effects.

### Step 2 — Write `hp_tuning_plan.md`

4. `Write` `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/hp_tuning_plan.md` with:

    **Header** — date, method, baseline recipe (the train_tinystories.py defaults: AdamW(0.9, 0.95), bf16, peak LR 3e-4, WSD warmup 200 + decay 20%, seq_len 1024, 100M-token budget).

    **HP surface table** — one row per knob:
    | HP | Default (from recipe) | Reported sensible range | Sensitivity (low/med/high) | Source |

    **Candidate configurations** — at least 2, typically 3:
    - **Config A — Conservative**: minimal deviation from the train_tinystories.py recipe, just the method-specific knobs set to recommended defaults
    - **Config B — Research-backed**: HP values pulled directly from the strongest paper/repo found in Phase 2 for this method
    - **Config C — Aggressive** (optional): higher LR / larger LoRA rank / larger batch — based on what papers say works at scale, scaled down for this size
    
    Each config is presented as a full dict the user can read at a glance, with a one-sentence rationale and expected tradeoff.

    **Suggested mini-sweep** (optional) — if the user picks the "sweep" option in step 5, the 2–4 configs to run and how long each will take. Always cite a published HP sweep when proposing this.

    **Recommendation** — your top single config with one-sentence rationale.

### Step 3 — Decide

5. Present `hp_tuning_plan.md` path to the user with a one-screen summary (table + recommended config).
6. `AskUserQuestion` (header: "HP config", single-select):
   - "Use recommended config" — single run with your top pick
   - "Pick a specific config (A / B / C)" — single run with that config
   - "Run mini-sweep of 2–4 configs" — multiple short runs, then pick the best for full evaluation; requires extra budget approval in Phase 6
   - "Custom — I'll specify values" — collect overrides via a follow-up message

**Gate:** user picks HP path. Save as `HP_CFG` (a dict) and `HP_MODE` ∈ {"single", "sweep"}.

---

## Phase 4 — Pick finance data

**Goal:** select a HuggingFace finance dataset compatible with the chosen method.

1. `WebSearch` HuggingFace for finance datasets matching the method's data shape. Examples to consider: `FinGPT/fingpt-fineval`, `gbharti/finance-alpaca`, `financial_phrasebank`, `JanosAudran/financial-reports-sec`, `Open-Orca/FinQA`. Prefer datasets with enough community usage signal (downloads, dated discussions).
2. For each candidate, note: size, license, format, whether it fits the method (distillation needs paired examples; perplexity work just needs corpus text; SFT needs instruction-response pairs; etc.).
3. `AskUserQuestion` (header: "Dataset", single-select): 2–4 strongest candidates with size + license + fit. Allow "user-provided local path" as one option.

**Gate:** user picks dataset. Save as `DATA_REF` (HF id or local path).

---

## Phase 5 — Implement the code changes

**Goal:** apply the chosen method + chosen HP config to the fine-tuning code in `MODEL_DIR`, preserving checkpoint compatibility with the base.

1. `Read` the files you will touch in full (`model.py`, `model_full.py`, `train.py`, `train_tinystories.py`, `verify.py`, others as needed). Do not assume — read.
2. Build the new fine-tune script `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/train_finance_<method>.py` by copying `train_tinystories.py` as a template and modifying ONLY what the method requires. The HP values come from `HP_CFG` (Phase 3) — bake them in as named constants at the top of the file so the diff is readable.
   - Swap the dataset (TinyStories → chosen finance HF dataset)
   - Swap output paths to `experiments/<EXPERIMENT_ID>/` (checkpoint, CSV, log, before/after txt)
   - Apply the method-specific deltas (e.g., wrap modules with LoRA, swap optimizer, change LR schedule, add replay mixing, etc.)
   - Bake `HP_CFG` into the top-of-file constants — readable as a single block
   - Keep the same WSD-style recipe and bf16 structure unless the method or `HP_CFG` specifically changes it
3. If the method requires module changes (LoRA, adapters, IA³, etc.):
   - Add the new module as a NEW file `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/<method>_modules.py`. Do NOT modify `model.py` or `model_full.py` in place — those are the canonical base and must remain checkpoint-compatible.
   - The fine-tune script wraps the loaded base model with the new modules at runtime.
4. If `HP_MODE == "sweep"`, the script must accept the HP values as CLI args (so a wrapper can loop). Otherwise hardcode `HP_CFG`.
5. Verify checkpoint compatibility: the script must be able to `load_state_dict` from the existing base into the unwrapped backbone without missing/unexpected key errors. State this guarantee explicitly in the script's docstring.
6. Present a tight diff summary to the user (paths + what changed + why, tied back to evidence in `finetune_techniques_plan.md` and `hp_tuning_plan.md`). Do NOT run training yet.
7. `AskUserQuestion` (header: "Approve code", single-select): "Approve & continue", "Request changes", "Abort".

**Gate:** user approves. If "Request changes", iterate on this phase; do not advance.

---

## Phase 6 — Estimate cost and approve training

**Goal:** prevent surprise long training runs. If `HP_MODE == "sweep"`, multiply the estimate by the number of configs.

1. Compute an estimate: dataset token count × epochs × seq-len / (tokens-per-second for this model on user hardware — if unknown, ask once and remember in a comment in `hp_tuning_plan.md`). Output as steps and wall-clock minutes per config, and the total if sweeping.
2. `AskUserQuestion` (header: "Train budget", single-select): "Approve estimated run (~X min, Y steps[, N configs])", "Reduce to ~30 min total", "Custom (ask me)", "Abort".

**Gate:** user approves the budget. Save final settings as `TRAIN_CFG` (a dict you'll print into the notebook).

---

## Phase 7 — Fine-tune

**Goal:** continue-pretrain from the base checkpoint into a new finance-adapted checkpoint without touching the baseline. If `HP_MODE == "sweep"`, run the script once per config and pick the best by held-out finance PPL for full evaluation in Phase 8.

1. Confirm baseline checkpoint:
   - `Bash` `ls -lS <MODEL_DIR>/*.pt` to list candidates.
   - Default: load the official HF weights via `verify.load_official_weights_into_ours` (matches the pattern in `train_tinystories.py`) — cleanest baseline because it's reproducible from a string ID.
   - Alternative: load `checkpoint.pt` (the existing local base) if the user wants to continue from a previously continued-pretrained state.
   - `AskUserQuestion` (header: "Baseline ckpt", single-select) only if there are multiple plausible candidates. Show options with sizes/dates.
2. Run the fine-tune script written in Phase 5 via `Bash` with `run_in_background: true`. Working directory must be `<MODEL_DIR>`. Capture stdout to `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/train.log` (or `train_<config-name>.log` if sweeping). Use `Monitor` to stream progress — never `sleep` to poll.
3. The script must save its checkpoint to `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/checkpoint_finance_<method>.pt` (mirror the `checkpoint_tinystories.pt` naming pattern; suffix with config name if sweeping). It must ALSO save:
   - `experiments/<EXPERIMENT_ID>/results/finance_train.csv` — per-step (step, loss, lr, tok_seen)
   - `experiments/<EXPERIMENT_ID>/results/finance_before.txt` — base-model generations + eval-loss on a held-out finance slice
   - `experiments/<EXPERIMENT_ID>/results/finance_after.txt` — trained-model generations + eval-loss + BEFORE→AFTER delta
4. If sweeping: after all configs finish, record each config's held-out finance PPL in `experiments/<EXPERIMENT_ID>/results/sweep_summary.json`, pick the best by PPL, and tell the user which config won before advancing to Phase 8.
5. NEVER overwrite `<MODEL_DIR>/checkpoint.pt`, `<MODEL_DIR>/checkpoint_tinystories.pt`, or any baseline file. If a path collision is possible, refuse to run and ask the user.
6. If training crashes: capture the error, do NOT retry blindly. Return to Phase 5 with the diagnosis surfaced to the user.

---

## Phase 8 — Evaluate (in-domain + OOD + six methods)

**Goal:** compare the new fine-tuned checkpoint (the sweep-winner, if applicable) vs. baseline on in-domain finance AND on OOD axes (to catch catastrophic forgetting, the #1 risk of continued pretraining). Mirror the `eval_after_vs_base.py` pattern.

Write `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/eval_finance_<method>_vs_base.py` by copying `eval_after_vs_base.py` and swapping in the new checkpoint path and the finance corpus. Run it via `Bash`, save raw outputs under `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/eval/`.

For each method, run on **both** baseline and the new checkpoint with the same seed/decoding params.

1. **Perplexity — in-domain (finance held-out).** Hold out 5–10% of `DATA_REF` (seeded split), report PPL for both. This is the primary signal for "did fine-tuning help in-domain."
2. **Perplexity — OOD wikitext-2.** Mirror eval_after_vs_base.py exactly. Catches catastrophic forgetting on general text.
3. **Perplexity — OOD code (codeparrot Python).** Mirror eval_after_vs_base.py exactly. Second forgetting probe.
4. **Domain-term next-token probes (custom).** Small probe set of finance terms (tickers like AAPL/MSFT, ratios like P/E/EBITDA/ROIC, accounting verbs like "amortize"/"accrue"). Report top-1 / top-5 next-token accuracy for both. Save as `eval/probes.json`.
5. **Qualitative side-by-side generations.** Fixed prompt set (≥10 prompts: finance prompts AND non-finance OOD prompts, mirroring eval_after_vs_base.py's `ood_prompts` for the OOD half). Same decoding params, same seed. Save as `eval/generations.md` with a side-by-side table.
6. **FinanceBench / FinQA accuracy.** Directional signal only. Note in the notebook that ~134M-class models are not expected to score well; the question is whether the technique *moves* the score (positive delta = signal even at low absolute score). Save as `eval/financebench.json`.

Don't fabricate eval results. If a method can't run (missing dep, dataset gated, OOM), record `"not run — <reason>"` in the JSON/notebook and tell the user. Never silently omit a method.

The final verdict in the notebook MUST report both in-domain delta AND OOD deltas. A technique that improves finance PPL by 10% while crashing wikitext-2 PPL by 50% is not a win — it's a forgetting failure, and the writeup must say so.

---

## Phase 9 — Write the experiment notebook

**Goal:** a single new notebook capturing the whole experiment, diffable and self-contained.

1. `Write` `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/<EXPERIMENT_ID>.ipynb` with this section order:
   - **Overview:** scope, chosen method, HP config (the actual values), one-paragraph rationale (pulled from `finetune_techniques_plan.md`), date (from `$TODAY`), baseline source (HF repo or `checkpoint.pt`), new checkpoint path.
   - **Plan summary:** the comparison table from `finetune_techniques_plan.md` (link the full file). Why this method was picked over the alternatives.
   - **Hyperparameters:** the candidate configs from `hp_tuning_plan.md`, which config won (and how it was picked — single run or sweep), the per-config sweep results table if applicable.
   - **Data:** dataset id, size, train/test split, sample row, license.
   - **Code changes:** file-by-file diff blocks of what was added in `experiments/<EXPERIMENT_ID>/`. Confirm `model.py` / `model_full.py` were NOT modified.
   - **Training:** `TRAIN_CFG`, final loss, training-loss curve (load from `experiments/<EXPERIMENT_ID>/results/finance_train.csv`), total tokens seen, wall-clock.
   - **Evaluation:** one subsection per eval method (6 total: in-domain PPL, wikitext-2 PPL, code PPL, probes, generations, FinanceBench). Each subsection has a comparison table (baseline vs new + delta%) and a short verdict line. Group the three perplexity sections so the in-domain-vs-OOD tradeoff is visually obvious.
   - **Forgetting check:** dedicated section calling out OOD deltas. Green if wikitext-2 / code PPL deltas are within ±5% of baseline; red otherwise with explicit warning.
   - **Overall verdict:** keep / discard / iterate, with reasoning. Must reference both in-domain and forgetting numbers.
   - **Reproduce:** the exact shell commands to re-run this experiment from scratch (cd into `MODEL_DIR`, run the fine-tune script with the winning HP config, then the eval script).

Use the same notebook construction approach as the existing `results.ipynb` / `proof.ipynb` in the model folder — match their style for plots and tables so the artifact looks at home in the repo. Do NOT edit `results.ipynb` or `proof.ipynb` themselves.

---

## Phase 10 — Hand off

**Goal:** end the run cleanly.

1. Print the absolute paths of: `finetune_techniques_plan.md`, `hp_tuning_plan.md`, the new notebook, and the experiment folder.
2. State the verdict in one sentence.
3. `AskUserQuestion` (header: "Next", single-select):
   - "Retune HPs for this method (back to Phase 3)"
   - "Try a different method from the plan doc (back to Phase 2 step 5 with same plan)"
   - "Re-survey methods (back to Phase 2 with same scope)"
   - "Change scope (back to Phase 1)"
   - "Done"

Do not auto-loop. The user decides.

---

## Constants

- **Training mode:** fine-tuning / continued pretraining only. Never from-scratch retraining.
- **Dates:** always derived from `Bash date` at runtime. Never hardcoded anywhere — not in queries, not in plan documents, not in this skill file.
- **Recency window for research:** 6 months back from `$TODAY`, expand to 12 months if <5 strong methods. Never fixed to a specific calendar date.
- **Domain:** finance (fixed).
- **Survey breadth:** 6–10 methods if `SCOPE == "comprehensive"`, 4–6 if narrowed. Never less than 4 — single-method "research" defeats the plan-doc purpose.
- **Eval methods:** all six in Phase 8 every run (in-domain PPL, wikitext-2 PPL, code PPL, probes, generations, FinanceBench).
- **Baseline:** official HF weights (via `verify.load_official_weights_into_ours`) by default; user can override to a local checkpoint.
- **Recipe defaults (copied from train_tinystories.py, method/HP_CFG may override):** AdamW(0.9, 0.95) + weight_decay 0.01, bf16, peak LR 3e-4, WSD schedule (200-step warmup, 20% decay), 100M-token budget, seq_len 1024.
- **Compute budget:** soft cap, ask before training kickoff. Sweep mode multiplies the estimate by config count.
- **Artifacts root:** `<MODEL_DIR>/experiments/<EXPERIMENT_ID>/`.

## Artifacts produced per run

```
<MODEL_DIR>/experiments/<EXPERIMENT_ID>/
├── finetune_techniques_plan.md                    # Phase 2 deliverable: structured comparison of methods, the decision aid
├── hp_tuning_plan.md                              # Phase 3 deliverable: HP surface + candidate configs for chosen method
├── research.md                                    # audit trail of how the research was conducted
├── sources.json                                   # raw web search results
├── train_finance_<method>.py                      # fine-tune script (copy of train_tinystories.py, modified)
├── <method>_modules.py                            # only if method needs new modules (LoRA etc.)
├── eval_finance_<method>_vs_base.py               # eval script (copy of eval_after_vs_base.py, modified)
├── checkpoint_finance_<method>.pt                 # fine-tuned checkpoint (baseline untouched); sweep mode: one per config
├── train.log                                      # training stdout/stderr (per-config if sweeping)
├── results/
│   ├── finance_train.csv                          # per-step (step, loss, lr, tok_seen)
│   ├── finance_before.txt                         # baseline generations + eval-loss
│   ├── finance_after.txt                          # post-fine-tune generations + delta
│   └── sweep_summary.json                         # sweep mode only: per-config PPL, winner
├── eval/
│   ├── perplexity.json                            # finance / wikitext-2 / code, base vs new
│   ├── probes.json                                # domain-term next-token accuracy
│   ├── generations.md                             # side-by-side qualitative
│   └── financebench.json                          # directional QA accuracy
└── <EXPERIMENT_ID>.ipynb                          # the comparison notebook
```

## Hard rules

- **Fine-tuning only.** Never from-scratch retraining. If a researched method requires retraining the backbone from random init, filter it out in Phase 2.
- **No hardcoded dates anywhere.** Always derive `$TODAY` and `$CUTOFF` from `Bash date` at the start of every research phase. If you catch yourself typing a YYYY-MM-DD literal in a query or a document, stop and run `date`.
- **Never collapse the survey to one method.** Even when LoRA is the obvious answer, the plan doc must present alternatives. The decision is the user's, not yours.
- **Plan documents are required artifacts, not optional summaries.** `finetune_techniques_plan.md` and `hp_tuning_plan.md` must exist on disk before their respective gates.
- **Never skip Phase 3 (HP tuning).** A method without justified HPs is not a tried method.
- **Never overwrite the baseline checkpoint** (`checkpoint.pt`, `checkpoint_tinystories.pt`, or any pre-existing `.pt` at `MODEL_DIR` root).
- **Never modify `model.py` or `model_full.py` in place.** Module additions go in a new file under `experiments/<EXPERIMENT_ID>/`.
- **Never edit `results.ipynb` or `proof.ipynb`** — new notebook per experiment.
- **Never skip Phase 8 methods silently;** record `"not run — reason"` if blocked.
- **Never start training without an explicit Phase 6 approval.**
- **Never claim a method works without showing both in-domain AND OOD numbers from Phase 8.** Forgetting is a fail condition, not a footnote.
- **If web search turns up fewer than 4 credible methods for the scope + size, surface that honestly and ask the user how to proceed — do not invent methods.**
- **Mirror the existing scripts** (`train_tinystories.py`, `eval_after_vs_base.py`) for layout, recipe, and output shape. Do not invent a new convention without the user's okay.
