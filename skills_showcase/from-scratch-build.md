---
name: from-scratch-build
description: End-to-end research → implement → verify → train → evaluate loop for building a language model from scratch under BuildFromScratch/. Supports both REPRODUCING a published model (faithful / modernized / exploratory) and NOVEL architectural designs. Surveys the current published-model landscape and per-component architectural choices via web search dated to the current run (never a frozen date), writes structured plan documents at each decision point (model target, architecture, training, hyperparameters), then implements `model.py` from a blank file, verifies numerical equivalence vs the HF reference (hard gate at <1e-3 max-error for reproduction mode), trains from scratch with a cost gate, and produces both a pedagogical proof notebook and a results notebook mirroring the existing `proof.ipynb` / `results.ipynb` pattern. Pauses for approval at every phase. Invoke with `/from-scratch-build` or when the user asks to "build a model from scratch" or "reproduce <model>".
---

# from-scratch-build

A phase-gated workflow. Each phase has a goal, the exact tool calls to make, and an approval gate that MUST pass before the next phase starts. Never skip a gate. Never combine two phases into a single user prompt.

This skill is **from-scratch training only** — it writes `model.py` from a blank file, initializes weights randomly, and trains. For fine-tuning an existing checkpoint, use `/finance-research-loop` instead.

Two supported build modes:
- **Reproduction** (faithful / modernized / exploratory) — pick a published model, implement its architecture, verify numerical equivalence vs official HF weights, train.
- **Novel design** — assemble an original architecture from researched components, smoke-test that it trains, train.

**Dates are never hardcoded.** Every recency cutoff must be derived at runtime from `Bash date` (see Phase 2 step 1). The skill file deliberately contains no fixed dates — if you're tempted to write one, run `date` instead.

If the user interrupts to redirect, drop the current phase and re-enter at the phase they're describing.

## Pattern to mirror (read these first, every run)

Before Phase 3, `Read` these existing files in the canonical reference folder (`SmolLM2-134(base)` if no other reference exists). They define the conventions every new build should mirror:

- `model.py` / `model_full.py` — the from-scratch model definition. Class layout, config dataclass, forward signature, generate method.
- `verify.py` — loads official HF weights into the from-scratch impl and checks numerical equivalence. Provides `REPO`, `load_official_weights_into_ours`. The numerical-equivalence pattern is the verify gate in Phase 5.
- `compare_with_hf.py` — broader comparison vs the HF model (generations, embeddings, etc.). Pattern for Phase 10.
- `train.py` — from-scratch training: WSD-schedule AdamW with peak LR 3e-3 (10× higher than fine-tune), bf16, token-budget loop, `PackedTextDataset`, `make_wsd_scheduler`. The recipe baseline for Phase 6.
- `_build_notebook.py` — Python script that programmatically builds the proof/results notebooks. Pattern for Phase 11.
- `proof.ipynb` — pedagogical walkthrough of the architecture (module-by-module). Structure to mirror in Phase 11.
- `results.ipynb` — training curves, eval tables, comparison plots. Structure to mirror in Phase 11.
- `requirements.txt` — pinned deps.

Your new `model.py`, `verify.py`, `train.py`, and notebooks should mirror these in layout, naming, and recipe. Do not invent a new convention unless the target model forces it.

---

## Phase 0 — Pick or create the build folder

**Goal:** know where this build's artifacts live.

1. `Bash`: `ls -1 /home/yashb98/Downloads/BuildFromScratch/ | grep -v '^\.'` to list existing folders.
2. `AskUserQuestion` (header: "Folder", single-select):
   - "Continue an existing build folder" — list the folders as sub-options if any exist
   - "Create a new folder (name deferred until target is picked in Phase 2)" — `Recommended` for new builds; folder gets named after the model target
   - "Create a new folder with a specific name now" — user provides the name in a follow-up
3. Save the chosen path (or the deferred placeholder) as `MODEL_DIR`. If deferred, set `MODEL_DIR = "<deferred>"` and create the folder at the end of Phase 2.

**Gate:** user picks folder strategy.

---

## Phase 1 — Pick build mode

**Goal:** decide whether this run reproduces a published model or designs a novel one, and (for reproduction) the fidelity stance.

1. `AskUserQuestion` (header: "Build mode", single-select):
   - **Reproduce — faithful** (Recommended) — match the published config + recipe + data as exactly as possible. Goal: replicate reported numbers. Numerical-equivalence verify gate applies in Phase 5.
   - **Reproduce — modernized** — arch faithful; use current best-practice recipe and data. Verify gate applies to unchanged components; smoke-test the rest.
   - **Reproduce — exploratory** — arch as starting point, swap in alternative components based on recent research (e.g., RoPE → ALiBi). Verify gate replaced with smoke-test.
   - **Novel design** — original architecture assembled from researched components. No HF reference; smoke-test only.

**Gate:** user picks mode. Save as `BUILD_MODE`.

---

## Phase 2 — Pick the target model (or design brief) + write `model_target_plan.md`

**Goal:** select what's being built. For reproduction modes, this is a published model. For novel design, this is a design brief (size budget + intended use). The deliverable is a structured plan document the user can read end-to-end.

### Step 1 — Anchor the date

1. `Bash`: `TODAY=$(date +%Y-%m-%d) && CUTOFF=$(date -d '6 months ago' +%Y-%m-%d) && echo "today=$TODAY cutoff=$CUTOFF"`. State both values in your first user-facing message of this phase.

### Step 2 — Survey candidate targets (reproduction mode)

2. Ask the user for their target param budget (e.g., 100M, 300M, 1B) and any family preference. Use `AskUserQuestion` (header: "Size budget", single-select) with sensible bands.
3. `WebSearch` queries (parallel) for published models in that size band, year-stamped with `$TODAY`'s year and the previous year:
   - `published small language model ~<size>M parameters <YYYY> open weights`
   - `<family preference> model card huggingface <YYYY>` (if family was specified)
   - `LLM reproduction from scratch <YYYY> tutorial` — for community-known gotchas
   - `<size>M language model paper architecture details <YYYY>`
4. Build a candidate list: SmolLM2-135M/360M, Pythia-{70M,160M,410M,1B}, GPT-2 (125M/355M), TinyLlama-1.1B, Qwen2.5-0.5B, Gemma-2-2B, Phi-mini, etc. — whatever fits the band and has open weights + a public model card.
5. For each candidate, note: arch family, param count (exact), training data (if disclosed), reported benchmark numbers, license, HF repo id.

### Step 2alt — Survey candidate designs (novel mode)

2alt. `WebSearch` for recent (≤6 months) architectural ideas relevant to the user's size budget: efficient attention (FlashAttention-3, ring attention, GQA/MQA variants), state-space models (Mamba-2, Hyena), MoE small-scale, novel position encodings (RoPE variants, NoPE, ALiBi successors), etc.
3alt. Build a shortlist of 4–6 design directions, each with citation + one-line "what's new."

### Step 3 — Write `model_target_plan.md`

6. Build `EXPERIMENT_ID = "${TODAY}_<mode>_<target-slug-tbd>"` (use `tbd` until the user picks).
7. If `MODEL_DIR == "<deferred>"`: leave creation until step 9. Otherwise: `Bash mkdir -p <MODEL_DIR>/builds/<EXPERIMENT_ID>`.
8. `Write` the plan file at `<MODEL_DIR>/builds/<EXPERIMENT_ID>/model_target_plan.md` (or in a scratch path if `MODEL_DIR` is deferred) with this exact structure:

    **Header** — date (`$TODAY`), recency cutoff (`$CUTOFF`), build mode, size budget.

    **Comparison table** (must be a real markdown table):
    | Target | Family | Params (exact) | Training data | Reported PPL/bench | License | HF repo | Reproducibility difficulty |
    |---|---|---|---|---|---|---|---|

    **Per-target detail sections** (one `### <target>` block each):
    - HF model card URL + paper URL + publish date
    - Architecture summary (layers, dim, heads, head dim, vocab size, context, activation, norm, positional encoding)
    - Training corpus (if disclosed) + token count
    - Reported numbers (verbatim quotes with citations)
    - Known gotchas from the community (forum threads, blog posts, common reproduction bugs)
    - Why it might / might not be a good fit for this build

    **Recommendation** — top 1–2 picks with one-sentence rationale each.

    **Open questions** for the user to weigh in on.

### Step 4 — Decide

9. Present the plan doc path + a one-screen summary (table + recommendation).
10. `AskUserQuestion` (header: "Target", single-select): up to 4 of the top candidates with one-line summary. Always allow "research deeper" via "Other".
11. **Now create / rename the folder.** Update `EXPERIMENT_ID` with the real target slug (e.g., `${TODAY}_reproduce-faithful_pythia-160m`). If `MODEL_DIR == "<deferred>"`, set `MODEL_DIR = /home/yashb98/Downloads/BuildFromScratch/<target-name>/` (e.g., `Pythia-160M/`) and `Bash mkdir -p <MODEL_DIR>/builds/<EXPERIMENT_ID>`. If `MODEL_DIR` already exists, just `mkdir -p` the build subfolder.
12. Move/rewrite the plan doc to its final location: `<MODEL_DIR>/builds/<EXPERIMENT_ID>/model_target_plan.md`.

**Gate:** user picks target. Save as `TARGET` (HF repo id + paper URL for reproduction; design brief dict for novel).

---

## Phase 3 — Research architecture + write `architecture_plan.md`

**Goal:** produce a complete per-component spec the user can review before any code is written.

1. `Bash`: re-anchor `TODAY` and `CUTOFF`.
2. `WebSearch` for the architecture details of `TARGET` (reproduction) or each candidate component (novel/exploratory):
   - `<TARGET> architecture details config json` — pull the config from HF
   - `<TARGET> implementation pytorch from scratch` — find published reproductions for comparison
   - For modernized/exploratory: also search `<component> language model <YYYY>` for each component the user wants to consider swapping (attention, norm, positional encoding, activation, init scheme).
3. Read the official HF config: `Bash` `python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('<repo>').to_json_string())"` — capture the ground-truth config dict.
4. `Write` `<MODEL_DIR>/builds/<EXPERIMENT_ID>/architecture_plan.md`:

    **Header** — target, build mode, date, source URLs (HF config, paper, reference repos).

    **Component-by-component table:**
    | Component | Faithful spec (from HF) | Decision for this build | Rationale | Source |
    |---|---|---|---|---|

    Components: attention type (MHA/GQA/MQA), head dim, number of heads/kv-heads, FFN type (gated vs vanilla), FFN expansion, activation (GELU/SiLU/SwiGLU), normalization (LayerNorm/RMSNorm), norm placement (pre/post), positional encoding (RoPE/ALiBi/learned/sinusoidal), RoPE theta, vocab size, tied embeddings (yes/no), init scheme, dropout, max context length.

    For faithful mode: "Decision" column = exactly the HF spec. For modernized/exploratory: "Decision" may differ; rationale must cite a source published ≤6 months ago.

    **Numerical-equivalence target** (reproduction modes only): max-abs-error tolerance for the Phase 5 verify gate (default `1e-3` for fp32, `1e-2` for bf16).

    **Open questions** for the user.

5. Present the plan doc to the user.
6. `AskUserQuestion` (header: "Arch plan", single-select): "Approve & implement", "Request changes", "Abort".

**Gate:** user approves. Save the final config dict as `ARCH_CFG`.

---

## Phase 4 — Implement `model.py` + tests

**Goal:** write a from-scratch implementation matching `ARCH_CFG`, plus a tests file. Single approval gate covers both.

1. `Read` the reference `model.py`, `model_full.py`, and `verify.py` (whichever exist in any sibling folder) in full. Match their class layout, config dataclass style, forward signature.
2. `Write` `<MODEL_DIR>/model.py` from scratch (or `<MODEL_DIR>/builds/<EXPERIMENT_ID>/model.py` if continuing an existing build folder without overwriting). Implement:
   - Config dataclass (mirror `SmolLM2Config` style)
   - Embedding, transformer blocks (attention + FFN + norm), final norm, lm_head
   - Forward returning `{"logits": ...}`
   - `generate` method (greedy + temperature/top-k)
3. `Write` `<MODEL_DIR>/builds/<EXPERIMENT_ID>/test_model.py` with:
   - `test_forward_shapes`: input `(B, T)` → logits `(B, T, V)` for several B, T
   - `test_backward`: loss.backward() runs without error, all params receive grads
   - `test_param_count`: matches the target's published param count within 1%
   - `test_generate_no_nan`: generate 16 tokens, assert no NaN/Inf
   - `test_load_state_dict_roundtrip`: save → load → forward matches
4. `Bash` `cd <MODEL_DIR> && python -m pytest builds/<EXPERIMENT_ID>/test_model.py -v` — all tests must pass.
5. Present diff summary + pytest output to the user.
6. `AskUserQuestion` (header: "Approve impl", single-select): "Approve & continue", "Request changes", "Abort".

**Gate:** user approves AND all tests passed. Otherwise iterate; do not advance.

---

## Phase 5 — Verify numerical equivalence vs HF (reproduction modes) OR smoke-test (novel/exploratory)

**Goal:** prove the implementation matches the reference before spending compute on training.

### Reproduction-faithful and reproduction-modernized (for unchanged components)

1. `Write` `<MODEL_DIR>/verify.py` mirroring the existing `verify.py` pattern: define `REPO` (the HF repo id), `load_official_weights_into_ours(model, hf_state_dict)` that maps HF parameter names to your model's parameter names.
2. `Write` `<MODEL_DIR>/builds/<EXPERIMENT_ID>/verify_run.py`:
   - Load official HF model via `AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)`
   - Load weights into the from-scratch model via `load_official_weights_into_ours`
   - Run forward on a fixed input (e.g., `tokenizer("The capital of France is", return_tensors='pt').input_ids`)
   - Compute `max_abs_error = (ours_logits - hf_logits).abs().max()`
   - Print the error; assert it's below the `ARCH_CFG` tolerance (default `1e-3` for fp32)
3. `Bash` `cd <MODEL_DIR> && python builds/<EXPERIMENT_ID>/verify_run.py`. Capture output.
4. **Hard gate:** if `max_abs_error >= tolerance`, return to Phase 4 with the diagnosis. Do NOT advance.
5. Save the verify result to `<MODEL_DIR>/builds/<EXPERIMENT_ID>/results/verify.json` (max-error, tolerance, pass/fail, fixed-input text).

### Reproduction-exploratory and novel design

1alt. `Write` `<MODEL_DIR>/builds/<EXPERIMENT_ID>/smoke_train.py`: instantiate the model, run 50 training steps on a tiny synthetic batch (random integers, fixed seed), record loss per step.
2alt. `Bash` `cd <MODEL_DIR> && python builds/<EXPERIMENT_ID>/smoke_train.py`. Assert:
   - loss at step 50 < loss at step 0 (it's actually learning)
   - no NaN/Inf in loss or any param
   - grad norms in `[1e-6, 1e3]` (not collapsed, not exploding)
3alt. **Hard gate:** if any assertion fails, return to Phase 4. Save smoke result to `results/smoke.json`.

---

## Phase 6 — Research training recipe + data + write `training_plan.md`

**Goal:** decide what corpus and what recipe to train on. Single plan doc covering both.

1. `Bash`: re-anchor `TODAY` and `CUTOFF`.
2. `WebSearch` queries (parallel), year-stamped:
   - Recipe: `<TARGET family> training recipe LR schedule batch size <YYYY>`
   - Recipe (faithful): `<TARGET> reproduce training hyperparameters paper`
   - Data: `<TARGET> training corpus dataset disclosed` (faithful) OR `pretraining corpus small language model <YYYY> recommendation` (modernized/novel)
   - Data alternatives: `fineweb-edu slimpajama TinyStories finance pretraining corpus comparison`
3. `Write` `<MODEL_DIR>/builds/<EXPERIMENT_ID>/training_plan.md`:

    **Header** — target, build mode, date, recency cutoff.

    **Data candidates table:**
    | Dataset | HF id | Size (tokens) | License | Domain | Fit for target | Notes |

    Always include: the original training corpus (if disclosed and accessible), a general default (e.g., `HuggingFaceFW/fineweb-edu`), TinyStories (for cheap proof runs), and finance (link to `/finance-research-loop`'s data picker if user wants domain-fixed). Add user-provided local path as an option.

    **Recipe candidates table:**
    | Recipe | LR (peak) | Schedule | Optimizer | Batch | Seq len | Token budget | Source |

    Always include: the original published recipe (faithful), the existing `train.py` recipe (AdamW(0.9,0.95), bf16, peak LR 3e-3, WSD warmup 200 + decay 20%, seq_len 1024 — adapt to target), and 1 modern alternative if research surfaces one.

    **Recommendation** — top data + top recipe for this build mode.

    **Open questions.**

4. `AskUserQuestion` (header: "Data", single-select): up to 4 dataset candidates.
5. `AskUserQuestion` (header: "Recipe", single-select): up to 4 recipe candidates.

**Gate:** user picks both. Save as `DATA_REF` and `RECIPE`.

---

## Phase 7 — Hyperparameter tuning + write `hp_tuning_plan.md`

**Goal:** propose HP configurations specific to the chosen recipe and target. From-scratch training is expensive — defaults to a single run; sweeps are explicit opt-in.

1. `Bash`: re-anchor `TODAY`.
2. `WebSearch` for HP recommendations specific to `TARGET` and `RECIPE`:
   - `"<TARGET>" learning rate batch size pretraining <YYYY>`
   - `<RECIPE family> hyperparameter ablation <YYYY>`
   - Scale-law guidance: `chinchilla scaling laws optimal tokens <param-count>M` to sanity-check the token budget
3. `Write` `<MODEL_DIR>/builds/<EXPERIMENT_ID>/hp_tuning_plan.md`:

    **HP surface table:**
    | HP | Default (from RECIPE) | Sensible range | Sensitivity | Source |

    **Candidate configs:**
    - **Config A — Faithful**: HPs from the published paper (reproduction modes) or the existing train.py defaults (novel)
    - **Config B — Compute-adjusted**: same shape as A but scaled to the user's compute budget (estimated wall-clock)
    - **Config C — Modern alternative** (optional): HPs from the strongest recent paper found in step 2

    Each config presented as a full dict.

    **Recommendation** — top single config.

4. `AskUserQuestion` (header: "HP config", single-select):
   - "Use recommended config"
   - "Pick a specific config (A / B / C)"
   - "Run mini-sweep of 2–3 configs" — explicit opt-in; warn that each config = full training run
   - "Custom — I'll specify values"

**Gate:** user picks. Save as `HP_CFG` (dict) and `HP_MODE` ∈ {"single", "sweep"}.

---

## Phase 8 — Estimate cost and approve training

**Goal:** prevent surprise long training runs. From-scratch is much more expensive than fine-tune; the cost gate matters more here.

1. Estimate: token budget × seq-len / (tokens-per-second for this model on user hardware). If TPS is unknown, ask once and remember in `hp_tuning_plan.md`. Output as steps and wall-clock minutes/hours. If `HP_MODE == "sweep"`, multiply by config count.
2. Cross-check the token budget against Chinchilla-optimal (~20 tokens per param): if your budget is more than 4× below Chinchilla-optimal, flag it ("model will be under-trained"); if more than 4× above, flag it ("diminishing returns").
3. `AskUserQuestion` (header: "Train budget", single-select):
   - "Approve estimated run (~X hours, Y steps[, N configs])"
   - "Reduce to smoke run (~1000 steps total)"
   - "Custom (ask me)"
   - "Abort"

**Gate:** user approves. Save as `TRAIN_CFG`.

---

## Phase 9 — Train from scratch

**Goal:** train the model from random init, producing a checkpoint without touching any existing checkpoints.

1. `Write` `<MODEL_DIR>/builds/<EXPERIMENT_ID>/train_<target-slug>.py` by copying the existing `train.py` as a template and modifying:
   - Use the new `model.py` (import from `<MODEL_DIR>/model.py` or the build-local `model.py`)
   - Tokenize `DATA_REF`
   - Apply `RECIPE` + `HP_CFG` as top-of-file constants
   - Output paths under `<MODEL_DIR>/builds/<EXPERIMENT_ID>/`
   - Save checkpoint as `checkpoint_<target-slug>.pt` — NEVER `checkpoint.pt` at the folder root if one exists
2. `Bash` `cd <MODEL_DIR> && python builds/<EXPERIMENT_ID>/train_<target-slug>.py` with `run_in_background: true`. Capture stdout to `train.log`. Use `Monitor` to stream progress — never `sleep` to poll.
3. The script must save:
   - `builds/<EXPERIMENT_ID>/checkpoint_<target-slug>.pt` — final state_dict + config + step + tok_seen
   - `builds/<EXPERIMENT_ID>/results/train.csv` — per-step (step, loss, lr, tok_seen)
   - `builds/<EXPERIMENT_ID>/results/sample_generations.txt` — periodic generations during training
4. If `HP_MODE == "sweep"`: train each config, record per-config final PPL in `results/sweep_summary.json`, pick the best by held-out PPL, tell the user which config won.
5. NEVER overwrite any existing `*.pt` at `MODEL_DIR` root.
6. If training crashes (NaN loss, OOM, etc.): capture the error, do NOT retry blindly. Return to Phase 7 (HPs) or Phase 4 (model) depending on the failure mode, with diagnosis surfaced.

---

## Phase 10 — Evaluate (vs HF reference + standalone)

**Goal:** compare the from-scratch trained model against the HF reference (reproduction modes) and against the random-init baseline (all modes). Mirror `compare_with_hf.py`.

`Write` `<MODEL_DIR>/builds/<EXPERIMENT_ID>/eval_<target-slug>.py` and run it. Save raw outputs under `builds/<EXPERIMENT_ID>/eval/`.

1. **Perplexity — validation split of `DATA_REF`.** Three models: random-init, your trained, HF reference (reproduction modes only). Report PPL for each.
2. **Perplexity — OOD wikitext-2.** Same three models. Catches in-domain overfit.
3. **Perplexity — OOD code (codeparrot).** Same three models. Second OOD probe.
4. **Side-by-side generations.** Fixed prompt set (≥10 prompts: in-domain + OOD). Same decoding params + seed for all three models. Save as `eval/generations.md`.
5. **Reproduction delta** (reproduction modes only): a table of `your_metric - hf_metric` per benchmark. Negative = your model is worse, positive = better. The goal of faithful reproduction is delta ≈ 0; modernized may differ; exploratory should differ deliberately.
6. **Param count + FLOPs check**: report total params, embedded vs non-embedded, FLOPs per token. Should match the target spec within 1%.

Don't fabricate eval results. If a method can't run (missing dep, dataset gated, OOM), record `"not run — <reason>"` and tell the user. Never silently omit.

---

## Phase 11 — Write proof notebook + results notebook

**Goal:** produce two notebooks mirroring `proof.ipynb` (pedagogical) and `results.ipynb` (training/eval), per the existing pattern.

1. `Read` the existing `proof.ipynb` and `results.ipynb` (whichever are present in any sibling folder) to understand the section structure and plot style. Match it.
2. `Read` `_build_notebook.py` if present — that's the existing convention for programmatic notebook construction. If using it, write a build-script for this build and let it generate the notebooks.

### proof notebook — `builds/<EXPERIMENT_ID>/proof_<target-slug>.ipynb`

Pedagogical walkthrough of the implementation. Section order:
- **Goal** — what model is being reproduced and why
- **Architecture overview** — config dict, parameter count, FLOPs, a diagram (text/ascii is fine)
- **Module-by-module walkthrough** — for each major module (embedding, attention, FFN, norm, lm_head): show the code, explain what it does, demonstrate a forward pass on a toy input
- **Numerical equivalence vs HF** — pull from `results/verify.json`, show the max-error and what input was used (reproduction modes only)
- **Tests** — pytest output from Phase 4

### results notebook — `builds/<EXPERIMENT_ID>/results_<target-slug>.ipynb`

Training and evaluation results. Section order:
- **Overview** — target, build mode, HP config, date (`$TODAY`), checkpoint path, total wall-clock
- **Training** — loss curve (from `results/train.csv`), LR schedule, tok/s, total tokens seen, periodic generations
- **Evaluation** — one subsection per Phase 10 method, comparison tables (random-init / your trained / HF reference), delta vs HF
- **Forgetting / OOD section** — wikitext-2 and code PPL deltas, called out separately
- **Reproduction verdict** (reproduction modes only) — did you match? within what tolerance? open questions about remaining gap
- **Reproduce** — exact shell commands to re-run this build from scratch

Use the same plot and table style as the existing notebooks. Do NOT edit any pre-existing `proof.ipynb` or `results.ipynb` at the folder root.

---

## Phase 12 — Hand off

**Goal:** end the run cleanly.

1. Print the absolute paths of: `model_target_plan.md`, `architecture_plan.md`, `training_plan.md`, `hp_tuning_plan.md`, both notebooks, and the build folder.
2. State the verdict in one sentence (reproduction delta or training viability for novel).
3. `AskUserQuestion` (header: "Next", single-select):
   - "Retune HPs and re-train (back to Phase 7)"
   - "Try a different architecture decision (back to Phase 3 with same target)"
   - "Try a different target model (back to Phase 2)"
   - "Fine-tune this trained model on a domain (handoff to /finance-research-loop)"
   - "Done"

Do not auto-loop. The user decides.

---

## Constants

- **Build mode:** from-scratch training only (random init). For fine-tuning an existing checkpoint, hand off to `/finance-research-loop`.
- **Dates:** always derived from `Bash date` at runtime. Never hardcoded anywhere.
- **Recency window for research:** 6 months back from `$TODAY`, expand to 12 months if <4 strong sources. Never fixed to a specific calendar date.
- **Survey breadth:** at least 4 candidate targets in Phase 2; component-by-component table in Phase 3; ≥2 dataset and ≥2 recipe candidates in Phase 6; ≥2 HP configs in Phase 7.
- **Verify gate (reproduction modes):** hard gate at max-abs-error < `1e-3` for fp32, `1e-2` for bf16. Falls back to smoke-test for exploratory/novel.
- **Recipe defaults (copied from existing train.py, RECIPE may override):** AdamW(0.9, 0.95) + weight_decay 0.01, bf16, peak LR 3e-3 (note: 10× higher than fine-tune), WSD schedule (200-step warmup, 20% decay), token-budget loop.
- **Token budget sanity:** Chinchilla-optimal ~20 tokens/param. Flag if budget is >4× off in either direction.
- **Compute budget:** soft cap, ask before training kickoff. Sweep multiplies the estimate by config count.
- **Artifacts root:** `<MODEL_DIR>/builds/<EXPERIMENT_ID>/`.

## Artifacts produced per run

```
<MODEL_DIR>/                                       # the model folder (new or existing)
├── model.py                                       # the from-scratch implementation (or builds/<id>/model.py if MODEL_DIR pre-existed)
├── verify.py                                      # HF-weights-into-ours loader (reproduction modes only)
└── builds/<EXPERIMENT_ID>/
    ├── model_target_plan.md                       # Phase 2 deliverable: target shortlist + decision
    ├── architecture_plan.md                       # Phase 3 deliverable: per-component spec
    ├── training_plan.md                           # Phase 6 deliverable: data + recipe options
    ├── hp_tuning_plan.md                          # Phase 7 deliverable: HP surface + candidates
    ├── test_model.py                              # Phase 4 tests
    ├── verify_run.py                              # Phase 5 verify (reproduction) OR smoke_train.py (exploratory/novel)
    ├── train_<target-slug>.py                     # Phase 9 train script
    ├── eval_<target-slug>.py                      # Phase 10 eval script
    ├── checkpoint_<target-slug>.pt                # trained checkpoint; sweep mode: one per config
    ├── train.log                                  # training stdout/stderr
    ├── results/
    │   ├── verify.json                            # max-error, tolerance, pass/fail (reproduction)
    │   ├── smoke.json                             # smoke-train result (exploratory/novel)
    │   ├── train.csv                              # per-step (step, loss, lr, tok_seen)
    │   ├── sample_generations.txt                 # periodic generations during training
    │   └── sweep_summary.json                     # sweep mode only
    ├── eval/
    │   ├── perplexity.json                        # in-domain / wikitext-2 / code, three-way comparison
    │   ├── generations.md                         # side-by-side qualitative
    │   └── repro_delta.json                       # reproduction delta vs HF (reproduction only)
    ├── proof_<target-slug>.ipynb                  # pedagogical walkthrough
    └── results_<target-slug>.ipynb                # training + eval results
```

## Hard rules

- **From-scratch only.** If the user wants to fine-tune an existing checkpoint, hand off to `/finance-research-loop`.
- **No hardcoded dates anywhere.** Always derive `$TODAY` and `$CUTOFF` from `Bash date` at the start of every research phase. If you catch yourself typing a YYYY-MM-DD literal in a query or document, stop and run `date`.
- **Never collapse the survey to one candidate.** Even when the obvious choice is obvious, the plan doc must present alternatives. The decision is the user's, not yours.
- **Plan documents are required artifacts, not optional summaries.** `model_target_plan.md`, `architecture_plan.md`, `training_plan.md`, `hp_tuning_plan.md` must all exist on disk before their respective gates.
- **Hard verify gate (reproduction modes).** Numerical equivalence below tolerance is non-negotiable; if it fails, return to Phase 4. Do not paper over implementation bugs by relaxing the tolerance — fix the code.
- **Never overwrite existing checkpoints.** New checkpoint name uses the target slug; if a collision is possible, refuse to run and ask.
- **Never edit pre-existing `model.py`, `proof.ipynb`, `results.ipynb`, or `verify.py`** at any folder root. New artifacts go under `builds/<EXPERIMENT_ID>/` or, if creating a brand-new folder, at the new folder's root.
- **Tests must pass before Phase 5.** Phase 4 doesn't gate-pass on tests failing.
- **Never skip Phase 5.** Verify (or smoke-test) before spending compute on training.
- **Never start training without an explicit Phase 8 approval.**
- **Never claim reproduction works without showing the numerical-equivalence number from Phase 5 AND the trained-vs-HF delta from Phase 10.**
- **If web search turns up fewer than 4 credible candidates in Phase 2 (or Phase 6), surface that honestly and ask the user how to proceed — do not invent candidates.**
- **Mirror the existing scripts** (`model.py`, `verify.py`, `train.py`, `_build_notebook.py`, `proof.ipynb`, `results.ipynb`) for layout, recipe, and output shape. Do not invent a new convention without the user's okay.
- **Hand-off to `/finance-research-loop`** for any fine-tuning follow-up. This skill ends at the trained from-scratch checkpoint.
