# Run 2026-06-16_qwen3-faithful_eval-first

**Type:** eval · **Suite:** text-lm-v2 · **Objective:** pretrain-ablation · **Status:** done
**Model:** Qwen3-0.6B faithful reproduction — `checkpoint_qwen3_baseline2tpp.pt` (step 18,150, the 2-TPP baseline)
**Significance:** this is the **first checkpoint ever scored through the standing `/eval-harness` suite** (previously the suite had produced zero result files). It turns the eval-harness from *tested-but-unexecuted* into *executed*.

## Result (single checkpoint, self-floor mode)

| Corpus | PPL | BPB (bits/byte) | Noise floor (PPL) | tokens scored |
|---|---|---|---|---|
| `wikitext2_raw_v1_val` (headline) | 37.01 | **1.2256** | ±1.30 | 204,600 |
| `codeparrot_clean_valid` | 438.67 | 2.1286 | ±280.6 | 204,600 |

- **Headline cross-tokenizer metric: BPB = 1.2256** on wikitext-2. This is the comparable number future runs (and other tokenizers, e.g. SmolLM2) are measured against.
- The high code PPL (438) is honest and expected: this faithful baseline was trained on FineWeb-Edu with little code, so it is poor at code — the suite reports it without flattering.
- wikitext PPL 37.01 is higher than the build's own FineWeb-Edu val PPL of 28.65 because **it is a different corpus** (wikitext is out-of-distribution for a FineWeb-trained model); the numbers are not directly comparable, which is exactly why a *standing* suite with pinned corpora matters.

## A real bug this run surfaced (and fixed)
The constructed eval script crashed on first smoke with `AttributeError: 'NoneType' object has no attribute '__dict__'` — `load_model_module()` exec'd `model.py` (which uses `@dataclass`) via `importlib` **without registering the module in `sys.modules` first**, so `@dataclass`'s `cls.__module__` lookup returned `None`. Fixed in `eval_suite_template.py` (register before exec). This is exactly the class of defect that only a real run surfaces — it would have crashed the first eval-harness invocation in production.

## Honest caveats — what this is NOT
- **n = 1, single checkpoint, single seed.** This is NOT a 9+/10 result and makes NO capability or comparison claim. It is the first *executed* eval, full stop.
- No baseline comparison (self-floor mode), so no `win|loss|inconclusive` verdict — that requires a multi-seed iso-FLOP cohort (the next step on the path to 9+).
- The noise floor here is the single-checkpoint subsample floor (corpus jitter), explicitly NOT a seed CI.

## Provenance
- Constructed script: `Qwen3-0.6B/experiments/2026-06-16_qwen3-faithful_eval-first/eval_suite.py`
- Raw output: `.../eval/suite_results.json` (+ `generations.md`)
- Corpora pinned: wikitext-2 `b08601e0…`, codeparrot `4db92d2e…`
- Deploy gate (`research/ci/eval_gate.py`) verified it can read this artifact (exit 0).
