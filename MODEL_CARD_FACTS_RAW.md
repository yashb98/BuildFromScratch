# Model-card fact sheet — complete raw dataset

Companion to [`MODEL_CARD_FACTS.md`](MODEL_CARD_FACTS.md). That file is the synthesis; this one is
**every fact the audit extracted, unabridged** — value, evidence path, verbatim source quote,
self-assessed confidence and caveat — each paired with the ruling the adversarial verification pass
returned for it. Nothing is summarised away and nothing is dropped: verifier entries that do not
correspond 1:1 to an extracted fact are reproduced in each dimension's *Additional verifier findings*.

Generated 2026-08-04 from the audit's structured output: 8 dimensions, each run as extract → refute.
The verifier's standing instruction was to *refute* every fact by opening the cited file, defaulting
to `WRONG` / `NEEDS_QUALIFIER` under uncertainty — so `CONFIRMED` means a second independent pass
opened the file and the claim survived.

| Verdict | Meaning |
|---|---|
| ✅ CONFIRMED | Cited file opened; claim holds as stated |
| ⚠️ NEEDS QUALIFIER | Value correct but materially incomplete without the attached caveat |
| ❌ WRONG | Claim does not survive; corrected value given |
| ⚠️ UNVERIFIABLE | Could not be settled from disk |

---
## Verdict summary

| # | Dimension | Facts | Verdicts | ✅ | ⚠️ Qual | ❌ Wrong | Gaps |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | [SmolLM2 eval provenance (15.371)](#1-smollm2-eval-provenance-15-371) | 11 | 11 | 9 | 2 | 0 | 6 |
| 2 | [SmolLM2 continued pretrain (6.8945 → 3.7900)](#2-smollm2-continued-pretrain-6-8945-3-7900) | 23 | 24 | 18 | 5 | 1 | 10 |
| 3 | [Qwen3 eval provenance (28.65 / 46.31 / 23.52 / 13.40)](#3-qwen3-eval-provenance-28-65-46-31-23-52-13-40) | 20 | 22 | 15 | 5 | 2 | 7 |
| 4 | [−0.474 bpb — NorMuon vs AdamW](#4-0-474-bpb-normuon-vs-adamw) | 12 | 12 | 8 | 4 | 0 | 7 |
| 5 | [Reproduce — commands, versions, parity, determinism](#5-reproduce-commands-versions-parity-determinism) | 28 | 28 | 24 | 3 | 1 | 7 |
| 6 | [Training details (the 1.19B-token runs)](#6-training-details-the-1-19b-token-runs) | 22 | 22 | 12 | 8 | 2 | 9 |
| 7 | [Loader API (real code)](#7-loader-api-real-code) | 21 | 23 | 17 | 5 | 1 | 5 |
| 8 | [Checkpoint inventory on disk](#8-checkpoint-inventory-on-disk) | 23 | 24 | 16 | 6 | 2 | 6 |
| | **Total** | **160** | **166** | **119** | **38** | **9** | **57** |

**160 facts extracted; 166 verdicts returned** (verifiers occasionally split or added a check, so the
counts differ). **9** facts were overturned outright and **38** needed a qualifier —
i.e. **47 of 166** checks found something that would have been misleading if published as first
extracted. The **57 gaps** are things the repo genuinely cannot answer, reproduced verbatim per dimension.

---

## 1. SmolLM2 eval provenance (15.371)

<sub>Audit dimension: SmolLM2-135M reproduction eval provenance (wikitext-2 val PPL 15.371 + HF parity)</sub>

### 1.1 Which HF dataset id, config name, and split produced 15.371? (verify whether it is the -raw- variant)

**Value**

```
dataset id = `Salesforce/wikitext` (the Salesforce-namespaced mirror, not the bare `wikitext`), config = `wikitext-2-raw-v1` (it IS the -raw- variant, confirmed), split = `validation`. Loaded with no `revision=` pin.
```

**Evidence** — `SmolLM2-134(base)/_build_notebook.py:233`

**Source quote**

```
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')   [_build_notebook.py:233]
"ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')\n",   [executed notebook cell, results.ipynb:739]
  "dataset": "wikitext-2-raw-v1 validation",   [results/perplexity.json:5]
```

**Confidence** — measured from code

**Caveat** — Text preprocessing is NOT the vanilla HF perplexity recipe. The script joins with '\n\n' but FILTERS OUT blank/whitespace-only rows: `text = '\n\n'.join(ex['text'] for ex in ds if ex['text'].strip())` (_build_notebook.py:234 / results.ipynb:740). The standard `"\n\n".join(test["text"])` keeps them. Different token stream => this absolute PPL is NOT directly comparable to published wikitext-2 numbers, only to the co-run HF model. No dataset revision sha is pinned, so the exact snapshot is not recoverable from disk.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
VERIFIED at SmolLM2-134(base)/_build_notebook.py:233 -> `ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')`. Byte-identical source line survives into the executed notebook at SmolLM2-134(base)/results.ipynb:739. Stamped into the machine-written artifact at SmolLM2-134(base)/results/perplexity.json:5 -> "dataset": "wikitext-2-raw-v1 validation". It IS the -raw- variant and it IS the Salesforce-namespaced mirror; no revision= argument on the line. The blank-row filter is real: _build_notebook.py:234 / results.ipynb:740 -> `text = '\n\n'.join(ex['text'] for ex in ds if ex['text'].strip())`. ONE NIT: the caveat's assertion that the standard recipe is `"\n\n".join(test["text"])` is an external-convention claim, not something on disk in this repo — correct as far as I know but flag it as not repo-verifiable. Bonus corroboration the fact missed: eval_after_vs_base.py:74 independently names the same dataset triple, and train.py:78 names wikitext-103-raw-v1 (train split) — so the repo does distinguish the two.
```


### 1.2 What sequence length and stride, and is it sliding-window or non-overlapping chunks?

**Value**

```
SEQ = 1024, STRIDE = 512. SLIDING WINDOW with 50% overlap, but WITHOUT the standard -100 masking of the overlapped context: every window contributes all 1023 shifted targets to the summed NLL, so overlap-region tokens are scored TWICE (once with short context, once with long). Neither non-overlapping chunks nor the canonical HF strided PPL.
```

**Evidence** — `SmolLM2-134(base)/_build_notebook.py:239`

**Source quote**

```
# Slide a 1024-token window with stride 512 over the first 32K tokens.
SEQ = 1024
STRIDE = 512
N_TOKENS = min(len(input_ids), 32_000)
def ppl(net):
    net = net.to(device).eval()
    nlls, n = [], 0
    with torch.no_grad():
        for begin in range(0, N_TOKENS - SEQ, STRIDE):
            ids = input_ids[begin:begin+SEQ].unsqueeze(0).to(device)
            out = net(ids)
            logits = out.logits if hasattr(out, 'logits') else out['logits']
            shift_logits = logits[..., :-1, :].float()
            shift_labels = ids[..., 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                   shift_labels.reshape(-1), reduction='sum')
            nlls.append(loss.item())
            n += shift_labels.numel()
    return math.exp(sum(nlls) / n), n   [_build_notebook.py:239-257]
  "seq_len": 1024,
  "stride": 512   [results/perplexity.json:6-7]
```

**Confidence** — measured from code

**Caveat** — Two scope limits a reviewer will ask about: (1) `N_TOKENS = min(len(input_ids), 32_000)` caps the eval to the FIRST 32,000 tokens of the concatenated validation text; the executed notebook printed `Validation tokens: 268,140` (results.ipynb:721), so only the first 31,744 token positions (~11.8% of the validation set) were ever fed to the model. (2) Because overlapped labels are not masked, the 62,403 scored targets cover only 31,743 DISTINCT positions (indices 1..31,743) — a ~1.97x double-count. I re-derived the window arithmetic: range(0, 32000-1024, 512) = 61 windows, first begin=0, last begin=30720, 1023 targets each, 61*1023 = 62,403, max index touched 31,744.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
VERIFIED. _build_notebook.py:240-241 `SEQ = 1024` / `STRIDE = 512`; :242 `N_TOKENS = min(len(input_ids), 32_000)`; :247 `for begin in range(0, N_TOKENS - SEQ, STRIDE)`; :252 `shift_labels = ids[..., 1:]` (1023 targets/window, no -100 masking anywhere in the cell); :256 `n += shift_labels.numel()`. Echoed in results/perplexity.json:6-7 (seq_len 1024, stride 512). I re-ran the arithmetic independently: range(0, 32000-1024, 512) = 61 windows, first begin 0, last begin 30720, 61*1023 = 62,403 targets, 31,743 DISTINCT label indices (1..31,743), 31,744 positions ever fed, 31744/268140 = 11.84% coverage, duplication factor 1.9659. All match the fact. TWO PRECISION NITS in the fact's own derivation prose: (a) it writes 'max index touched 31,744' — the max index is 31,743; 31,744 is the position COUNT. (b) 'overlap-region tokens are scored TWICE' is true for 30,660 of the 31,743 distinct positions; the remaining 1,083 are scored once (I computed the exact multiplicity histogram: Counter({2: 30660, 1: 1083})). Also worth knowing: `SEQ` is REBOUND to 512 later at _build_notebook.py:442 for the training-demo cell, but perplexity.json is written at :270-273 inside the PPL cell, before the rebind — so the 1024 in the JSON is correct.
```


### 1.3 Which tokenizer (exact HF repo id) was used?

**Value**

```
`HuggingFaceTB/SmolLM2-135M`, via `AutoTokenizer.from_pretrained(REPO)` with REPO imported from verify.py. No revision pin. BPE, vocab 49,152. `add_special_tokens` left at the transformers default; the executed output shows 5 tokens for a 5-word prompt, i.e. no BOS prepended. Same repo id as the model => own-tokenizer PPL.
```

**Evidence** — `SmolLM2-134(base)/verify.py:19`

**Source quote**

```
REPO = "HuggingFaceTB/SmolLM2-135M"   [verify.py:19]
from verify import load_official_weights_into_ours, REPO   [_build_notebook.py:43 / results.ipynb cell 1]
tokenizer = AutoTokenizer.from_pretrained(REPO)   [_build_notebook.py:112]
encodings = tokenizer(text, return_tensors='pt')   [_build_notebook.py:235]
      "Tokens : [504, 3575, 282, 4649, 314]\n",   [results.ipynb:269 — 'The capital of France is' -> 5 tokens, no BOS]
  "Tokenizer": "BPE, vocab 49,152",   [results/summary.json:6]
```

**Confidence** — measured from code

**Caveat** — I could not verify `add_bos_token` from a config on disk: the local HF hub cache (~/.cache/huggingface/hub) contains only `models--Qwen--Qwen3.5-9B`, no SmolLM2-135M snapshot. The no-BOS conclusion is inferred from the executed 5-token/5-piece output, not from a tokenizer_config.json.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Tokenizer id/vocab/no-BOS are all CORRECT and are now CONFIRMED FROM A CONFIG ON DISK. The caveat 'no SmolLM2-135M snapshot exists on disk' is REFUTED: a full snapshot lives at /home/yashb98/projects/qwen-distill/hf_cache/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/ (a second project's HF_HOME, not ~/.cache). That directory also yields the revision sha the 'gaps' section says is unrecoverable: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2.
```

**Verifier note**

```
VALUE CONFIRMED: verify.py:19 `REPO = "HuggingFaceTB/SmolLM2-135M"`; _build_notebook.py:43 `from verify import load_official_weights_into_ours, REPO`; :112 `tokenizer = AutoTokenizer.from_pretrained(REPO)` (no revision=); :235 `encodings = tokenizer(text, return_tensors='pt')` (add_special_tokens left at default True). results/summary.json:6 "Tokenizer": "BPE, vocab 49,152". CAVEAT REFUTED — from the snapshot above I read: tokenizer_config.json has NO `add_bos_token` key, tokenizer_class = 'GPT2Tokenizer' (byte-level BPE), bos_token = eos_token = '<|endoftext|>', model_max_length 8192; tokenizer.json has `post_processor: null` (so add_special_tokens=True adds NOTHING). That is a config-level proof of the no-BOS behaviour, not an inference. I further decoded the ids from that snapshot's vocab.json: 504='The', 3575='Ġcapital', 282='Ġof', 4649='ĠFrance', 314='Ġis', and 260='Ġthe' — exactly matching results.ipynb:269 and :280-281. vocab.json length = 49152, config.json vocab_size = 49152, tie_word_embeddings=True, 30 layers / 576 hidden / 9Q / 3KV / rope_theta 100000. TWO STANDING QUALIFIERS: (1) that snapshot's blobs are dated May 13 11:44 while results.ipynb ran May 13 ~22:20 (same day) — strongly suggestive but NOT proof it is the snapshot the run used, since the code pins no revision and the cache is outside this repo; (2) it is outside the repo root, so a card claiming 'reproducible from this repo alone' still cannot pin the tokenizer.
```


### 1.4 How many target tokens did the PPL average over? (verify the '62,403 target tokens' prose figure)

**Value**

```
62,403 scored target tokens — VERIFIED three ways: the machine-written JSON, the executed notebook stdout, and independent re-derivation of the loop bounds (61 windows x 1023 targets).
```

**Evidence** — `SmolLM2-134(base)/results/perplexity.json:4`

**Source quote**

```
"tokens": 62403,   [results/perplexity.json:4]
      "Ours : ppl = 15.371   (62,403 target tokens, 4.8s)\n",
      "HF   : ppl = 15.371   (62,403 target tokens, 9.0s)\n",   [results.ipynb:729-730, executed output]
- `perplexity.json` — sliding-window CE perplexity on wikitext-2-raw-v1
  validation, 62,403 target tokens, seq=1024 stride=512.   [results/README.md:38-39]
```

**Confidence** — results JSON

**Caveat** — 62,403 is the count of SCORED targets, not distinct tokens. Overlapping windows without label masking mean only 31,743 distinct positions are covered; roughly half the 62,403 are duplicate scorings of the same token under a different context length. results/README.md:38-39 and results/POST_DATA.md:34-36 state '62,403 target tokens' without that qualification.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
VERIFIED four ways. (1) Machine-written: results/perplexity.json:4 `"tokens": 62403`. (2) Executed stdout: results.ipynb:729-730 'Ours : ppl = 15.371   (62,403 target tokens, 4.8s)' / 'HF   : ppl = 15.371   (62,403 target tokens, 9.0s)'. (3) Prose: results/README.md:38-39 and results/POST_DATA.md:34-36. (4) My own re-derivation: 61 windows x 1023 = 62,403. The fact's qualifier is correct and material — 62,403 is SCORED targets over only 31,743 distinct positions. Both prose sites (results/README.md:38-39, results/POST_DATA.md:34-36) state the number with no such qualification, so if the figure goes on a model card it MUST be labelled 'scored targets (1.97x double-counted by overlapping windows), 31,743 distinct positions, first 11.8% of the validation split'.
```


### 1.5 Exact values: ours vs HF to full precision, and the delta

**Value**

```
ours_ppl = 15.370989092449635; hf_ppl = 15.370989964425396; delta (hf - ours) = +8.719757609298995e-07 (abs 8.72e-07). BOTH are MEASURED on this box in the same notebook kernel — the 'HF' figure is a live forward pass of HF's LlamaForCausalLM, NOT copied from a model card or paper. fp32 (HF loaded dtype=torch.float32, our model default fp32, logits .float() before CE), run on GPU (Device: cuda | NVIDIA GB10).
```

**Evidence** — `SmolLM2-134(base)/results/perplexity.json:2`

**Source quote**

```
"ours_ppl": 15.370989092449635,
  "hf_ppl": 15.370989964425396,   [results/perplexity.json:2-3]
ours_ppl, n_tok = ppl(model)
...
hf_ppl, _ = ppl(hf_model)   [_build_notebook.py:260,263 — both computed live]
hf_model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)   [_build_notebook.py:110]
      "Device: cuda | NVIDIA GB10\n"   [results.ipynb:45]
```

**Confidence** — results JSON

**Caveat** — Delta recomputed by me from the JSON floats: 15.370989964425396 - 15.370989092449635 = 8.719757609298995e-07. README.md:57-58 rounds to 15.370989 / 15.370990 with 'Δ ≈ 9 × 10⁻⁷' — consistent. Important regime split: the PPL comparison ran on GPU (both models), whereas the max|Δlogits| = 0 'bit-exact' claim ran on CPU; do not merge them into one claim. Also n=1, one corpus slice, no seeds, no CI — an implementation-equivalence check, not a quality benchmark.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Values and delta are exactly right and both ARE measured (not copied). Missing material qualifier for a MODEL CARD: 'ours' is not a model this repo trained — it is the OFFICIAL SmolLM2-135M safetensors loaded into the reimplementation class (_build_notebook.py:114 `load_official_weights_into_ours(model, hf_model.state_dict())`). 15.371 therefore characterizes the official checkpoint under a nonstandard eval recipe, and must NOT be attached to either local checkpoint (checkpoint.pt = the 150-step demo; checkpoint_tinystories.pt = TinyStories continued-pretrain).
```

**Verifier note**

```
VALUES VERIFIED: results/perplexity.json:2-3 give exactly 15.370989092449635 and 15.370989964425396. I recomputed the delta in Python: 15.370989964425396 - 15.370989092449635 = 8.719757609298995e-07 — identical to the fact, correct sign (HF higher). Consistent with the executed 'Δppl = 0.000001' at results.ipynb:731 and with README.md:57-58 ('15.370989' / '15.370990 (Δ ≈ 9 × 10⁻⁷)'). MEASURED-not-copied CONFIRMED: _build_notebook.py:260 `ours_ppl, n_tok = ppl(model)` and :263 `hf_ppl, _ = ppl(hf_model)` are both live forward passes; the HF model is loaded at :110 `AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)`. fp32/GPU CONFIRMED: :251 `shift_logits = logits[..., :-1, :].float()`, :244 `net = net.to(device).eval()`, device set at :46, and the run printed 'Device: cuda | NVIDIA GB10' (results.ipynb:45) with 'Torch: 2.11.0+cu130' (results.ipynb:44). The fact's own caveats (GPU-PPL vs CPU-bit-exact regime split; n=1, one slice, no seeds/CI) are correct and must survive to the card.
```


### 1.6 Which script computes 15.371, and what is the exact command to re-run it?

**Value**

```
There is NO standalone perplexity script. It is cell 14 of `results.ipynb`; that cell's source is authored by the notebook generator `_build_notebook.py:230-273`. Documented regeneration = the two-step generate-then-nbconvert pass in results/README.md:66-73.
```

**Evidence** — `SmolLM2-134(base)/results/README.md:66`

**Source quote**

````
```bash
# from /home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)/
python3 _build_notebook.py       # writes results.ipynb (28 cells, no outputs)
jupyter nbconvert --to notebook \
    --execute results.ipynb \
    --output results.ipynb \
    --ExecutePreprocessor.timeout=2400
# total runtime ~3 minutes (perplexity & training are the slow cells)
```   [results/README.md:66-74]
with open(RESULTS / 'perplexity.json', 'w') as f:
    json.dump({'ours_ppl': ours_ppl, 'hf_ppl': hf_ppl, 'tokens': n_tok,
               'dataset': 'wikitext-2-raw-v1 validation',
               'seq_len': SEQ, 'stride': STRIDE}, f, indent=2)   [_build_notebook.py:270-273]
````

**Confidence** — measured from code

**Caveat** — Three re-run hazards: (1) `python3 _build_notebook.py` OVERWRITES results.ipynb and wipes the executed outputs that are currently the only record of the run; (2) the nbconvert pass re-executes all 28 cells including a 150-step from-scratch training demo (results.ipynb cell 23), so it is not a cheap PPL-only re-run; (3) the notebook never imports safe_cuda / calls safe_cuda.guard() before torch (_build_notebook.py:32-52), which CLAUDE.md §C1 mandates for every PyTorch script on this GB10 box. Provenance timestamps are consistent: results/perplexity.json mtime 2026-05-13 22:19:09, results.ipynb 22:20:20; `git diff HEAD` on both is empty (identical to the single commit 84a96c0).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
VERIFIED. I enumerated the notebook with nbformat: 28 cells total; index 13 = markdown '## 6. Perplexity on wikitext-2-raw-v1 validation', index 14 = code '# %% Perplexity on wikitext-2 validation'. So 'cell 14' is correct 0-indexed (it is section 6 by the notebook's own numbering — say 'cell index 14' on any card to avoid ambiguity). results/README.md:66-74 carries the exact bash block quoted. No standalone PPL script exists — I grepped every .py/.sh in the folder for wikitext/perplex/math.exp; the only other eval-PPL path is eval_after_vs_base.py (fact 10) and train_tinystories.py:64 (TinyStories, non-overlapping). ALL THREE RE-RUN HAZARDS CONFIRMED: (1) _build_notebook.py:561-562 `out = Path('results.ipynb')` / `nbf.write(nb, str(out))` — unconditional overwrite, wiping the executed outputs; (2) 28 cells re-execute, including cell 23 whose body sets `STEPS = 150` (its own header comment says '200 optimizer steps on wikitext-103 slice' and its code at :435-436 actually loads wikitext-2 train — the cell comment is stale twice over, but the fact's '150-step' is the correct figure); (3) grep for 'safe_cuda' in _build_notebook.py returns ZERO hits — the setup cell at :32-52 imports torch at :35 with no guard, violating CLAUDE.md §C1. PROVENANCE CONFIRMED: stat gives results/perplexity.json mtime 2026-05-13 22:19:09 and results.ipynb 22:20:20. MINOR WORDING FIX: 'the single commit 84a96c0' should read 'the only commit that has ever touched these files' — `git rev-list --count HEAD` = 77; `git log -- SmolLM2-134(base)/results/perplexity.json` and `... results.ipynb` each return only 84a96c0, and `git status --porcelain 'SmolLM2-134(base)/'` is empty.
```


### 1.7 What are the parity / bit-exactness numbers (max|Δlogits|, argmax agreement) and which script produces them?

**Value**

```
max|Δlogits| = 0.000e+00 (relative 0.000e+00); argmax agreement YES — HF token 260 -> ' the', ours 260 -> ' the'. Prompt "The capital of France is", fp32, CPU. Produced by `python3 verify.py` (logged to results/parity.log) and independently reproduced by results.ipynb cell 6. Gates: `assert max_abs < 1e-3` (verify.py:76) and `assert hf_next == our_next` (verify.py:83); the same gates are wrapped as pytest in tests/test_parity.py (short-prompt logits, argmax, 512-token long context, all 30 per-layer hidden states, param count 134,515,008, tied-embedding pointer).
```

**Evidence** — `SmolLM2-134(base)/results/parity.log:6`

**Source quote**

```
max |Δlogits| = 0.000e+00
relative      = 0.000e+00
HF next token : ' the'
Ours next     : ' the'

✓ Architecture parity verified.   [results/parity.log:6-11]
    assert max_abs < 1e-3, f"Outputs diverge: {max_abs}. Architecture mismatch."   [verify.py:76]
    assert hf_next == our_next, "Next-token disagreement"   [verify.py:83]
      "max |Δlogits|     = 0.000e+00\n",
      "HF   argmax last : 260 → ' the'\n",
      "Ours argmax last : 260 → ' the'\n",   [results.ipynb:278,280,281]
  "max |Δlogits| vs HF": "0.000e+00",   [results/summary.json:7]
```

**Confidence** — measured from code

**Caveat** — The 0.0 is a CPU result on a 5-token prompt: verify.py never calls .to(cuda), and results.ipynb cell 6 runs before any .to(device). The GPU-side numbers quoted in README.md:73-78 / results/comparison_with_hf.md:10-15 are prose-only — see next fact.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
VERIFIED line by line. results/parity.log:6-11 reads exactly 'max |Δlogits| = 0.000e+00' / 'relative      = 0.000e+00' / "HF next token : ' the'" / "Ours next     : ' the'" / '✓ Architecture parity verified.'. verify.py:76 `assert max_abs < 1e-3, ...` and verify.py:83 `assert hf_next == our_next, "Next-token disagreement"` — both present verbatim. Token id 260 comes from results.ipynb:280-281 ("HF   argmax last : 260 → ' the'" / "Ours argmax last : 260 → ' the'"), NOT from parity.log, and the fact attributes it correctly; I independently confirmed 260='Ġthe' from the SmolLM2 vocab.json. results/summary.json:7 "max |Δlogits| vs HF": "0.000e+00". CPU CLAIM CONFIRMED BY CONSTRUCTION: verify.py has no .to('cuda') anywhere, and in _build_notebook.py the first `.to(device)` occurrences are at :244/:248 (inside the PPL helper) and :449/:466 (training demo) — all AFTER the parity cell at :108-134. tests/test_parity.py fully checks out: :53 `assert n == 134_515_008`, :61 tied data_ptr, :73 short-prompt <1e-3, :83 argmax, :93 `max_length=512` long context, :127 `assert len(hf_states) == len(our_states) == 30` with :132 per-layer <1e-3. Extra corroboration: parity.log:1 emits the `torch_dtype` deprecation warning, which matches verify.py:53's `torch_dtype=torch.float32` (the notebook uses the newer `dtype=` at :110) — independent evidence parity.log really is verify.py's output.
```


### 1.8 Are the six GPU-side cross-check parity numbers (4.72e-05, 1.95e-03 @ L14, 5/5 greedy, 5/5 top-10, 4.01e-05 long-context, 0.072 vs 0.080 sampling) backed by a results file?

**Value**

```
NO. The .md write-up exists but the machine-written JSON it should derive from is absent from disk.
```

**Evidence** — `SmolLM2-134(base)/compare_with_hf.py:259`

**Source quote**

```
with open(RESULTS / "comparison_with_hf.json", "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\nSaved {RESULTS}/comparison_with_hf.json")   [compare_with_hf.py:259-261]

$ ls SmolLM2-134(base)/results/  ->  attention, comparison_with_hf.md, generations.txt, loss_curve.csv, param_count.log, parity.log, perplexity.json, plots, POST_DATA.md, README.md, summary.json, tinystories_after.txt, tinystories_before.txt, tinystories_summary.md, tinystories_train.csv, tinystories_train.log, topk_predictions.json, training_recipe_resolved.json   (no comparison_with_hf.json)
```

**Confidence** — PROSE ONLY

**Caveat** — Concrete unverifiable discrepancy: comparison_with_hf.md:14 and README.md:77 report the long-context check as '401-token RoPE', but compare_with_hf.py:180-181 truncates at `max_length=512`. 401 is presumably the actual tokenized length of the 2000-char probe, but with the JSON missing nothing on disk proves it. Do not put any of these six numbers on a model card as measured without re-running `python3 compare_with_hf.py`.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
VERIFIED. `ls SmolLM2-134(base)/results/comparison_with_hf.json` -> 'No such file or directory' (exit 2); the full results/ listing contains comparison_with_hf.md but no .json. compare_with_hf.py:259-261 is exactly as quoted and is the only writer. All six numbers do exist as prose in results/comparison_with_hf.md:10-15 and are duplicated in README.md:73-78. THE 401-vs-512 DISCREPANCY IS REAL: comparison_with_hf.md:14 and README.md:77 say '401-token RoPE', while compare_with_hf.py:177 banners '5. Long-context (RoPE sanity at 512 tokens)' and :179-181 build `long_text = ("In the field of language modeling, " * 60)[:2000]` then tokenize with `truncation=True, max_length=512`; :251 prints the label '5. Long-context (512 tok)'. tests/test_parity.py:92-93 uses the identical construction. So 401 is plausibly the realized token count of the 2000-char probe, but with the JSON gone nothing on disk proves it. ADDITIONAL FINDING the fact should carry: results/README.md:3-4 asserts 'Every file here is produced live ... No values are typed in by hand' — that blanket claim is NOT supported for comparison_with_hf.md. Its mtime (2026-05-13 22:07:53) also predates the notebook run (22:19-22:20), so it was not produced by that run.
```


### 1.9 Were downstream benchmark numbers (HellaSwag/ARC/MMLU etc.) measured in this repo?

**Value**

```
NOT_FOUND — never run. A harness script exists (scripts/run_lm_eval.sh) but its output directory does not exist.
```

**Evidence** — `SmolLM2-134(base)/scripts/run_lm_eval.sh:27`

**Source quote**

```
OUT_DIR="results/lm_eval"   [scripts/run_lm_eval.sh:27]
$ ls SmolLM2-134(base)/results/lm_eval  ->  "ls: cannot access 'results/lm_eval': No such file or directory"

- **Published downstream benchmarks** (HellaSwag, ARC, MMLU, etc. — model card
  reports these). Computing them from scratch would take a few hours per task.   [results/comparison_with_hf.md:84-85]
```

**Confidence** — NOT FOUND

**Caveat** — results/comparison_with_hf.md:86-88 then asserts 'any benchmark score will match by construction' — an unmeasured claim; it must not be transcribed to a model card as a result.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
VERIFIED. scripts/run_lm_eval.sh:27 `OUT_DIR="results/lm_eval"`; :23 `TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,commonsense_qa,openbookqa,mmlu"`; :22 BASE_REPO=HuggingFaceTB/SmolLM2-135M; :29 `mkdir -p "$OUT_DIR"` would create it on any run. `ls SmolLM2-134(base)/results/lm_eval` -> 'No such file or directory' (exit 2), and the results/ listing has no lm_eval entry — the script has never completed even its mkdir. results/comparison_with_hf.md:84-85 confirms the omission is deliberate ('Published downstream benchmarks ... Computing them from scratch would take a few hours per task'), and :86-88 does make the unmeasured assertion 'any benchmark score will match by construction'. That sentence is a prediction, not a result; it must not be transcribed onto a card in any form that reads as a measurement.
```


### 1.10 Is there a second, different wikitext-2 PPL path in this folder that could be confused with 15.371?

**Value**

```
YES — eval_after_vs_base.py computes a DIFFERENT wikitext-2 PPL: same dataset/config/split and same seq=1024/stride=512, but capped at max_windows=200 (not 32,000 tokens) and run in bf16 on GPU (not fp32). It serves the TinyStories before/after comparison, not the 15.371 headline.
```

**Evidence** — `SmolLM2-134(base)/eval_after_vs_base.py:50`

**Source quote**

```
def ppl(model, text, seq=1024, stride=512, max_windows=200):   [eval_after_vs_base.py:50]
wk = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")   [eval_after_vs_base.py:74]
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32   [eval_after_vs_base.py:29]
# --- 2. wikitext-2 validation (general text, was 15.371 for both before) -   [eval_after_vs_base.py:72]
```

**Confidence** — measured from code

**Caveat** — eval_after_vs_base.py:8 says it writes results/tinystories_vs_base.{md,json}; neither file is on disk. Its code-corpus fallback at line 91 does `open("model.py").read()`, and model.py does not exist in this folder — that branch would crash.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
VERIFIED. eval_after_vs_base.py:50 `def ppl(model, text, seq=1024, stride=512, max_windows=200):` with :53-54 `for i, begin in enumerate(...)` / `if i >= max_windows: break`; :74 `wk = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")`; :29 `dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32` applied at :38 and :45; :72 comment 'was 15.371 for both before'. So it is a WIDER slice (200 windows ~= 204,600 targets vs 61 windows / 62,403) at LOWER precision — genuinely non-comparable to 15.371 in both directions. Its outputs are missing: eval_after_vs_base.py:8 declares results/tinystories_vs_base.md and .json; `ls` on both -> 'No such file or directory'. The :91 fallback `code_text = open("model.py").read() * 30` would crash (model.py absent). TWO MORE wikitext paths the fact did not enumerate, both TRAINING not eval, but confusable on a card: train.py:78 loads wikitext-103-raw-v1 TRAIN, and _build_notebook.py:436 loads wikitext-2-raw-v1 TRAIN for the 150-step demo while the surrounding prose at :402 and the cell header at :428 both say 'wikitext-103 slice' — a code/prose mismatch. Do not write 'trained on wikitext-103' for the notebook demo.
```


### 1.11 Do the prose docs cite any file that does not exist?

**Value**

```
YES — `model.py`. results/POST_DATA.md:20 cites '198 lines | model.py line count | `wc -l model.py`' and results/README.md:25 cites 'param_count.log — output of `python3 model.py`', but the folder contains only model_full.py (a stale __pycache__/model.cpython-312.pyc suggests model.py once existed).
```

**Evidence** — `SmolLM2-134(base)/results/POST_DATA.md:20`

**Source quote**

```
| **198 lines** | `model.py` line count for the from-scratch architecture | `wc -l model.py` |   [results/POST_DATA.md:20]
- `param_count.log` — output of `python3 model.py` (param count + random-init forward).   [results/README.md:25]
$ ls SmolLM2-134(base)/model.py  ->  "ls: cannot access 'model.py': No such file or directory"
```

**Confidence** — measured from code

**Caveat** — The parameter count itself IS backed: results/param_count.log:1-2 reads 'params: 134,515,008    (target 134,515,008)' / 'tied:   True', and tests/test_parity.py:53 asserts n == 134_515_008. Only the '198 lines of model.py' claim and the 'python3 model.py' reproduce instruction are broken.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
VERIFIED. results/POST_DATA.md:20 reads '| **198 lines** | `model.py` line count for the from-scratch architecture | `wc -l model.py` |'. results/README.md:25 reads '- `param_count.log` — output of `python3 model.py` (param count + random-init forward).' `ls SmolLM2-134(base)/model.py` -> 'No such file or directory'; the directory listing shows model_full.py (15,443 bytes) only, and __pycache__/model.cpython-312.pyc is present alongside model_full.cpython-312.pyc, so model.py did once exist. A THIRD broken reference the fact missed: results/README.md:4 also credits '`../verify.py` / `../model.py`' as producers of the results files. PARAM COUNT IS SOUND: results/param_count.log:1-2 reads 'params: 134,515,008    (target 134,515,008)' / 'tied:   True', tests/test_parity.py:53 asserts `n == 134_515_008`, and the official config.json in the snapshot I found confirms vocab 49152 / hidden 576 / 30 layers / tie_word_embeddings=True. Only the '198 lines' figure and the 'python3 model.py' reproduce instruction are unbacked — a card must not repeat either.
```


### 1.G Gaps — not determinable from disk

- No dataset revision/sha pin. `load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')` (_build_notebook.py:233) carries no `revision=` argument, so the exact dataset snapshot behind 15.371 is not recoverable from disk. Same for tokenizer/model: `AutoTokenizer.from_pretrained(REPO)` and `AutoModelForCausalLM.from_pretrained(REPO, ...)` pin no revision.
- Tokenizer special-token behaviour cannot be confirmed from a config file. The local HF hub cache (~/.cache/huggingface/hub) holds only `models--Qwen--Qwen3.5-9B`; there is no SmolLM2-135M snapshot to read tokenizer_config.json / add_bos_token from. The 'no BOS prepended' conclusion rests only on the executed notebook output (5 tokens for a 5-word prompt, results.ipynb:269).
- results/comparison_with_hf.json is missing, so the six GPU-side cross-check numbers in results/comparison_with_hf.md and README.md §0.2 (4.72e-05 final logits, 1.95e-03 @ layer 14, '401-token' long context, 0.072 vs 0.080 sampling) have no machine-written backing file. compare_with_hf.py:259-260 would create it on re-run.
- eval_after_vs_base.py:8 declares outputs results/tinystories_vs_base.{md,json}; neither exists. The TinyStories before/after PPLs (6.8945 -> 3.7900, 199,485 target tokens) exist only as plain text in results/tinystories_before.txt and results/tinystories_after.txt:2-3, not as structured JSON.
- Environment versions only partially recorded. results.ipynb cell 1 output gives 'Torch: 2.11.0+cu130' and 'Device: cuda | NVIDIA GB10'; no transformers or datasets version is recorded for the run that produced 15.371, and requirements.txt (62 bytes) was not pinned to that run.
- No statistical rigor around 15.371: single run, single corpus slice (first ~11.8% of wikitext-2 validation), no seeds, no confidence interval, no second corpus, no BPB. It was NOT produced by the research/ eval-harness, so it carries no suite_version stamp — by the repo's own §C10/§C17 rules it is an implementation-equivalence check, not a comparable quality number.

---

## 2. SmolLM2 continued pretrain (6.8945 → 3.7900)

<sub>Audit dimension: SmolLM2-135M continued-pretraining run on TinyStories (val PPL 6.8945 -> 3.7900, -45.0%)</sub>

### 2.1 Which HF dataset id / config / split produced 6.8945 and 3.7900?

**Value**

```
hf id `roneneldan/TinyStories`, split="validation", NO config/subset name passed (positional config arg omitted), NO revision pin. Training data was the same id, split="train".
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:155`

**Source quote**

```
val_ds = load_dataset("roneneldan/TinyStories", split="validation")
(line 154:)    train_ds = load_dataset("roneneldan/TinyStories", split="train")
```

**Confidence** — measured from code

**Caveat** — Corroborated by the run log: `SmolLM2-134(base)/results/tinystories_train.log:4` -> `train: 2,119,719 stories, val: 21,990`. NOT pinned to a dataset revision/sha, so a re-run is not byte-reproducible. The eval does NOT use the whole validation split: it consumes stories in order until the packed token buffer reaches 200,000 (train_tinystories.py:172-181), landing at 200,068 tokens -> a PREFIX of the 21,990 stories. How many stories that was is not recorded anywhere.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
VALUE IS CORRECT. Two caveat sentences must be replaced: (a) the resolved revision IS recoverable — `roneneldan/TinyStories` @ f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 (cached 2026-05-13 14:16-14:20, i.e. ~7h BEFORE the run started); (b) the eval consumed the FIRST 1,040 non-empty validation stories (I re-ran the packer today and it reproduced 200,068 tokens byte-identically).
```

**Verifier note**

```
VALUE CONFIRMED verbatim. train_tinystories.py:154 = `    train_ds = load_dataset("roneneldan/TinyStories", split="train")`; :155 = `    val_ds = load_dataset("roneneldan/TinyStories", split="validation")`. No positional config arg, no revision= kwarg — correct. results/tinystories_train.log:4 = `[21:23:34]   train: 2,119,719 stories, val: 21,990` — confirmed verbatim. CAVEAT PARTLY REFUTED: the agent concluded "a re-run is not byte-reproducible" and "how many stories that was is not recorded anywhere". Both are overstated. HF_HOME is redirected to /home/yashb98/projects/qwen-distill/hf_cache (env var, not ~/.cache), and it contains hub/datasets--roneneldan--TinyStories with refs/main = f54c09fd23315a6f9c86f9dc80f725de7d8f9c64, blobs dated 2026-05-13 14:16-14:20 — before the run began. I re-executed the exact packing loop (train_tinystories.py:172-181) on CPU today after `python3 sentinel.py preflight` (exit 0) and got: `packed tokens: 200068  stories consumed: 1040  empty skipped: 0` — an exact byte-level reproduction of log:6. So the eval subset IS reconstructible: the first 1,040 stories of the validation split. It remains true that the CODE passes no revision pin.
```


### 2.2 How was the eval text assembled from that split?

**Value**

```
Stories are stripped, empty ones skipped, each encoded with add_special_tokens=False and followed by tokenizer.eos_token_id, concatenated until >= 200,000 tokens. Actual packed length: 200,068 tokens.
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:172`

**Source quote**

```
val_buf = []
    for ex in val_ds:
        text = ex["text"].strip()
        if not text:
            continue
        val_buf.extend(tokenizer.encode(text, add_special_tokens=False))
        val_buf.append(eos)
        if len(val_buf) >= 200_000:
            break
```

**Confidence** — measured from code

**Caveat** — Packed length confirmed in the log: `results/tinystories_train.log:6` -> `[21:25:09]   packed 200,068 val tokens`. No cross-document attention masking — EOS separator only, attention flows across story boundaries (documented as a known simplification at SmolLM2-134(base)/README.md:792-795).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_tinystories.py:172-181 matches the quote verbatim (I read the file; :172 = `    val_buf = []`, :179 = `        if len(val_buf) >= 200_000:`). results/tinystories_train.log:6 = `[21:25:09]   packed 200,068 val tokens` — verbatim. Independently reproduced today: 200,068 tokens, 1,040 stories, 0 empties skipped, eos_token_id = 0 (`<|endoftext|>`). Cross-doc caveat CONFIRMED: SmolLM2-134(base)/README.md:792-795 reads `- *Cross-document attention masking.* Real pretraining packs multiple / documents into one sequence and masks attention so a position in doc A can't / see tokens from doc B. We pack with a simple EOS separator and let attention / flow freely — fine for a tiny demo, sloppy for real runs.` One scoping nuance a reviewer may raise: that passage sits under README.md:785 `### What this script intentionally does *not* do` inside §10 (README.md:751), which is written about train.py — but train_tinystories.py:38 imports PackedTextDataset from train.py, so the simplification does apply to this run.
```


### 2.3 Sequence length and stride used for that eval

**Value**

```
seq_len = 1024, stride = 1024 (NON-OVERLAPPING windows; stride == seq_len). Window count capped at max_windows=200. Targets are the 1023 shifted tokens per window.
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:68`

**Source quote**

```
def evaluate(model, val_tokens, device, seq_len: int, max_windows: int = 200):
    """Sliding-window CE perplexity over val_tokens (no overlap)."""
    ...
    for begin in range(0, min(len(val_tokens) - seq_len, max_windows * seq_len), seq_len):
        ids = val_tokens[begin:begin + seq_len].unsqueeze(0).to(device)
        logits = model(ids)["logits"][..., :-1, :].float()
        labels = ids[..., 1:]
```

**Confidence** — measured from code

**Caveat** — IMPORTANT: stride is 1024, NOT 512. The stride-512 setting belongs to a DIFFERENT eval — `SmolLM2-134(base)/eval_after_vs_base.py:50` (`def ppl(model, text, seq=1024, stride=512, max_windows=200)`) and the wikitext-2 parity eval (`results/perplexity.json` -> "seq_len": 1024, "stride": 512). eval_after_vs_base.py never produced output (see separate fact). Any writeup that attaches "stride 512" to the 6.8945/3.7900 pair would be wrong.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
STRONGEST-VERIFIED FACT IN THE SET, and I can add a hard proof the agent did not give. train_tinystories.py:63 and :68 match the quote verbatim (`for begin in range(0, min(len(val_tokens) - seq_len, max_windows * seq_len), seq_len):`). The stride=1024 claim is not merely code-read, it is PROVEN by the recorded token count: with stride 1024 → range(0, min(200068-1024, 204800)=199044, 1024) = 195 windows × 1023 = 199,485 = exactly the logged value. With stride 512 the same buffer would give 200 windows (max_windows binds) × 1023 = 204,600 ≠ 199,485. So stride 512 is arithmetically excluded. Contrast eval CONFIRMED: eval_after_vs_base.py:50 = `def ppl(model, text, seq=1024, stride=512, max_windows=200):` (grep -n, exact line). results/perplexity.json confirmed: {"dataset": "wikitext-2-raw-v1 validation", "seq_len": 1024, "stride": 512}. ONE IMPRECISION: the value says "Window count capped at max_windows=200" — the cap did NOT bind (199,044 < 204,800); the buffer length bound at 195 windows. Phrase it as "max_windows=200 cap present but not reached; 195 windows actually scored."
```


### 2.4 Tokenizer used for that eval

**Value**

```
`AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")` — the official SmolLM2 BPE, vocab 49,152. Same tokenizer for train packing and eval packing.
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:144`

**Source quote**

```
tokenizer = AutoTokenizer.from_pretrained(REPO)
(SmolLM2-134(base)/verify.py:19:)  REPO = "HuggingFaceTB/SmolLM2-135M"
```

**Confidence** — measured from code

**Caveat** — vocab_size 49152 independently confirmed from the checkpoint's stored config (read live: ck['config']['vocab_size'] == 49152). No local HF cache entry for HuggingFaceTB/SmolLM2-135M exists today (~/.cache/huggingface/hub contains only CACHEDIR.TAG and models--Qwen--Qwen3.5-9B), so a re-run re-downloads unpinned.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
VALUE IS CORRECT. The caveat sentence "No local HF cache entry for HuggingFaceTB/SmolLM2-135M exists today (~/.cache/huggingface/hub contains only CACHEDIR.TAG and models--Qwen--Qwen3.5-9B)" is FALSE and must be deleted. Correct statement: HF_HOME=/home/yashb98/projects/qwen-distill/hf_cache; that cache holds models--HuggingFaceTB--SmolLM2-135M with refs/main = 93efa2f097d58c2a74874c7e644dbc9b0cee75a2, blobs dated 2026-05-13 11:44-11:47 (before the run).
```

**Verifier note**

```
VALUE CONFIRMED: train_tinystories.py:144 = `    tokenizer = AutoTokenizer.from_pretrained(REPO)`; verify.py:19 = `REPO = "HuggingFaceTB/SmolLM2-135M"`. Live-loaded the tokenizer today: vocab_size 49152, eos_token_id 0. ck['config']['vocab_size'] == 49152 confirmed by torch.load. CAVEAT REFUTED: the agent checked ~/.cache/huggingface/hub, but `env | grep HF_` shows HF_HOME=/home/yashb98/projects/qwen-distill/hf_cache — the default cache is not the active one. The real cache contains the model snapshot (config.json, model.safetensors, tokenizer.json, merges.txt, vocab.json) at revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2. This is a materially wrong statement about reproducibility and would embarrass the card if a reviewer ran `env`.
```


### 2.5 Eval numerical precision

**Value**

```
Model held in torch.bfloat16 on cuda; logits upcast to fp32 (.float()) before cross_entropy with reduction="sum"; PPL = exp(sum(nll)/n_target_tokens).
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:70`

**Source quote**

```
logits = model(ids)["logits"][..., :-1, :].float()
        labels = ids[..., 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), reduction="sum")
        nlls.append(loss.item())
        n += labels.numel()
    ...
    return math.exp(sum(nlls) / n), n
```

**Confidence** — measured from code

**Caveat** — bf16 confirmed at results/tinystories_train.log:1 -> `[21:23:28] Device: cuda   dtype: torch.bfloat16`. So 6.8945 is a bf16-forward number, not the fp32 number a parity-grade eval would give.

**Verdict** — _no 1:1 verifier entry; see Additional verifier findings below._


### 2.6 How many eval target tokens? (prose says 199,485 — verify)

**Value**

```
199,485 — VERIFIED, and reproduced by arithmetic: 200,068 packed val tokens -> range(0, min(200068-1024, 200*1024), 1024) yields 195 windows -> 195 x 1023 target tokens = 199,485.
```

**Evidence** — `SmolLM2-134(base)/results/tinystories_before.txt:2`

**Source quote**

```
Validation PPL: 6.8945  (199,485 target tokens)
(results/tinystories_after.txt:2:) Validation PPL: 3.7900  (199,485 target tokens)
(results/tinystories_train.log:10:) [21:25:14]   baseline TinyStories-val PPL = 6.895  on 199,485 target tokens
```

**Confidence** — results JSON

**Caveat** — I re-executed the window arithmetic against the on-disk evaluate() and the logged 200,068 val tokens; it lands exactly on 195 windows x 1023 = 199,485. Both before and after used the identical val_tokens tensor in the same process, so the two PPLs are strictly paired on the same tokens.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Three independent on-disk sources verified verbatim: results/tinystories_before.txt:2 = `Validation PPL: 6.8945  (199,485 target tokens)`; results/tinystories_after.txt:2 = `Validation PPL: 3.7900  (199,485 target tokens)`; results/tinystories_train.log:10 = `[21:25:14]   baseline TinyStories-val PPL = 6.895  on 199,485 target tokens`. I re-ran the window arithmetic in Python: range(0, min(200068-1024, 200*1024), 1024) → 195 begins → 195*1023 = 199485. Pairing claim CONFIRMED structurally: val_tokens is built once at train_tinystories.py:181 and passed unchanged to evaluate() at :196 and :362. Minor labelling nit only: confidence is tagged "results-json" but the cited artifacts are .txt/.log, not JSON.
```


### 2.7 Was 6.8945 measured by this repo, or copied from a paper/model card?

**Value**

```
MEASURED by this repo. It is the official HuggingFaceTB/SmolLM2-135M safetensors loaded into this repo's own SmolLM2ForCausalLM (via load_official_weights_into_ours), cast to bf16 on cuda, then scored by the same evaluate() before any optimizer step. Not copied from anywhere.
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:196`

**Source quote**

```
if args.resume is None:
        log("Baseline (BEFORE) eval...", log_path)
        base_ppl, base_n = evaluate(model, val_tokens, device, args.seq_len)
(lines 144-149:)    tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
    model = SmolLM2ForCausalLM(SmolLM2Config())
    load_official_weights_into_ours(model, hf.state_dict())
    del hf
    model = model.to(device=device, dtype=dtype)
```

**Confidence** — measured from code

**Caveat** — "Base checkpoint" here = the OFFICIAL HF weights, not a repo-trained base. Full-precision value read live out of the checkpoint: ck['baseline_ppl'] = 6.894546783281595.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_tinystories.py:144-149 verified verbatim (AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32) → SmolLM2ForCausalLM(SmolLM2Config()) → load_official_weights_into_ours → .to(device, dtype)). Lines 194-197 verified: the baseline evaluate() call at :196 sits BEFORE the AdamW construction at :213, so "before any optimizer step" is structurally guaranteed. Live-read from checkpoint_tinystories.pt via torch.load(map_location='cpu'): ck['baseline_ppl'] = 6.894546783281595 — matches the 4-dp prose exactly. The caveat's clarification that "base checkpoint" = official HF weights, not a repo-trained base, is correct and material — keep it on the card. This is unambiguously a repo measurement, not a copied number.
```


### 2.8 Was 3.7900 measured on the continued-pretrained checkpoint?

**Value**

```
YES. Measured in-process at end of training by the same evaluate() on the same val_tokens. Full-precision value stored in the checkpoint: trained_ppl = 3.7899503859716885 (baseline_ppl = 6.894546783281595). Exact delta = -45.0297%, which rounds to the cited -45.0%.
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:362`

**Source quote**

```
trained_ppl, trained_n = evaluate(model, val_tokens, device, args.seq_len)
(results/tinystories_train.log:508:) [23:21:25]   AFTER PPL = 3.790   (BEFORE was 6.895; improvement +3.105 = +45.0%)
```

**Confidence** — results JSON

**Caveat** — I read the two floats live out of checkpoint_tinystories.pt (torch.load, map_location='cpu'). This is n=1: one seed, one corpus, no across-seed CI, no iso-FLOP control arm, no downstream evals. Under CLAUDE.md's rigor bar (§C10/§C17/§C18/§C25) this is a directional in-domain result, not a `win`.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_tinystories.py:362 = `    trained_ppl, trained_n = evaluate(model, val_tokens, device, args.seq_len)` — verbatim. results/tinystories_train.log:508 = `[23:21:25]   AFTER PPL = 3.790   (BEFORE was 6.895; improvement +3.105 = +45.0%)` — verbatim at the cited line number. Live torch.load gives ck['trained_ppl'] = 3.7899503859716885, ck['baseline_ppl'] = 6.894546783281595; I computed 100*(b-t)/b = 45.029738645593945 → -45.0% correct to the stated precision and sign. The n=1 / no-CI / no-iso-FLOP-control / no-downstream caveat is correct and, per CLAUDE.md §C10/§C17/§C18/§C25, MUST ship with the number — it caps this at `directional`, not `win`.
```


### 2.9 Training config — tokens, steps, seq len, tokens/step (MEASURED)

**Value**

```
token_budget 100,000,000; final tok_seen 99,999,744; total_steps 24,414; seq_len 1024; tokens per optimizer step 4,096; 99,609 packed train windows from 102,000,116 packed train tokens.
```

**Evidence** — `SmolLM2-134(base)/results/tinystories_train.log:5`

**Source quote**

```
[21:25:09]   packed 102,000,116 train tokens in 97.9s
[21:25:09]   packed 200,068 val tokens
[21:25:09]   99,609 train windows of 1024
[21:25:09]   tok/step = 4,096   total_steps = 24,414
```

**Confidence** — measured from code

**Caveat** — Cross-checked: 100_000_000 // 4096 = 24,414 exactly; results/tinystories_train.csv has 24,414 data rows; last row is `24414,1.3074886798858643,0.0,99999744`; ck['step']=24414, ck['tok_seen']=99999744.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
results/tinystories_train.log:5-8 verified verbatim at the cited line numbers (`packed 102,000,116 train tokens in 97.9s` / `packed 200,068 val tokens` / `99,609 train windows of 1024` / `tok/step = 4,096   total_steps = 24,414`). CSV: 24,415 lines = 1 header + 24,414 data rows (wc -l); last row read directly = `24414,1.3074886798858643,0.0,99999744`. torch.load gives ck['step'] = 24414, ck['tok_seen'] = 99999744. 100_000_000 // 4096 = 24414 exactly. Every element independently reproduces.
```


### 2.10 Micro batch / grad accum / global batch

**Value**

```
Global batch = 4,096 tokens = 4 sequences of 1024 per optimizer step (MEASURED as the product). The claimed split micro_batch=4 x grad_accum=1 is PROSE-ONLY — nothing in the run artifacts records the two factors separately.
```

**Evidence** — `SmolLM2-134(base)/results/POST_DATA.md:52`

**Source quote**

```
**0.01** (on 2D params only), grad_clip 1.0, bf16, seq_len 1024, micro_batch 4.
(SmolLM2-134(base)/train_tinystories.py:189:)    tok_per_step = args.seq_len * args.micro_batch * args.grad_accum
```

**Confidence** — PROSE ONLY

**Caveat** — Only the product (4,096) is logged. micro_batch=4 / grad_accum=1 are the argparse defaults of the CURRENT on-disk script (train_tinystories.py:106-107), which is NOT the version that produced this run (see the script-drift fact).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
results/POST_DATA.md:52 = `  **0.01** (on 2D params only), grad_clip 1.0, bf16, seq_len 1024, micro_batch 4.` — verbatim at the cited line. train_tinystories.py:189 = `    tok_per_step = args.seq_len * args.micro_batch * args.grad_accum` — verbatim. Only the product is logged (log:8), so the factorization is genuinely unrecoverable from artifacts. Confirming detail the agent missed: results/tinystories_summary.md:74-75 tabulates `| Micro batch | 4 | this run |` and `| Grad accumulation | 1 | this run |` — but summary.md documents the PRIOR run (see the two-runs fact), so it is not evidence for the 116.1-min run either. The prose-only verdict holds and is if anything better supported.
```


### 2.11 LR schedule (MEASURED from the per-step LR trace)

**Value**

```
WSD: linear warmup 200 steps to peak 3e-4; stable 3e-4 through step 19,531; linear decay over the final 20% (decay_start = int(24414*0.8) = 19,531, first decayed LR at step 19,532 = 2.9993856e-4 = 3e-4*(1 - 1/4883)) to exactly 0.0 at step 24,414.
```

**Evidence** — `SmolLM2-134(base)/results/tinystories_train.csv:19533`

**Source quote**

```
1,1.9550740718841553,1.4999999999999998e-06,4096          (csv:2  -> 3e-4 * 1/200)
200,1.7322626113891602,0.0003,819200                       (csv:201)
19531,1.4177279472351074,0.0003,79998976                   (csv:19532)
19532,1.4647554159164429,0.00029993856235920535,80003072   (csv:19533)
24414,1.3074886798858643,0.0,99999744                      (csv:24415)
```

**Confidence** — measured from code

**Caveat** — Schedule shape matches make_wsd_scheduler in SmolLM2-134(base)/train.py:45-56 exactly (`decay_start = int(total_steps * (1.0 - decay_frac))`), which pins peak_lr=3e-4, warmup_steps=200, decay_frac=0.20 from the data, not from prose. LR recorded is sched.get_last_lr() after sched.step(), i.e. the LR for the following step.

**Verdict** — _no 1:1 verifier entry; see Additional verifier findings below._


### 2.12 Optimizer, betas, eps, weight decay, grad clip, seed

**Value**

```
NOT_FOUND in any run artifact. Claimed values (AdamW, betas (0.9, 0.95), eps 1e-8, weight_decay 0.01 on 2D params only, grad_clip 1.0, seed 0) exist ONLY in prose + in the argparse defaults of the current on-disk script.
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:213`

**Source quote**

```
optim = AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )
(prose, SmolLM2-134(base)/README.md:101-102:) recipe AdamW(0.9, 0.95), peak LR **3e-4** ... wd 0.01, grad-clip 1.0, bf16
```

**Confidence** — PROSE ONLY

**Caveat** — The run log contains NO `args=` line and NO seed line (grep for 'args=' across results/tinystories_train.log returns nothing; log:1 is only `Device: cuda   dtype: torch.bfloat16`). The saved checkpoint has NO 'training_recipe' key (live-read keys: ['model','config','step','tok_seen','baseline_ppl','trained_ppl']). The CSV has no grad_norm column, so gradient clipping left no trace at all. These hyperparameters were never recorded for this run.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
CONFIRMED and STRENGTHENED. Absence verified by direct grep on results/tinystories_train.log: `grep -c "args="` → 0; `grep -in seed` → no matches; `grep -c "\[eval @ step"` → 0; `grep -c "\[ckpt @ step"` → 0. Live torch.load: list(ck.keys()) == ['model','config','step','tok_seen','baseline_ppl','trained_ppl'] — no 'training_recipe'. CSV header is `step,loss,lr,tok_seen` — no grad_norm column, so clipping left no trace. train_tinystories.py:213-217 and README.md:101-102 both quoted verbatim and correct. ADDITIONAL EVIDENCE THE AGENT MISSED, which strengthens the finding: results/training_recipe_resolved.json exists and does list AdamW / betas [0.9,0.95] / eps 1e-8 / weight_decay 0.01 / clip_grad 1.0 — but its own line 2 declares `"source": "https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm2/config_smollm2_135M.yaml"` and its values are the UPSTREAM FROM-SCRATCH config (lr 0.003, warmup 2000, seq_len 2048, 2M steps), not this run. results/tinystories_summary.md:80-82 likewise sources wd/clip to "nanotron config_smollm2_135M.yaml" and the optimizer to "paper §4.1". So these hyperparameters are COPIED FROM AN EXTERNAL CONFIG, not measured here — an even stronger reason to keep them off a model card as measured values.
```


### 2.13 Precision, wall-clock, throughput, GPU

**Value**

```
Precision bf16. Training-loop wall clock 116.1 min; whole process 21:23:28 -> 23:21:27 = ~118.0 min (includes 97.9 s tokenization + before/after evals + generations). Mean throughput 14,356 tok/s (= 99,999,744 / (116.1*60) = 14,355). GPU: the log records only `Device: cuda`.
```

**Evidence** — `SmolLM2-134(base)/results/tinystories_train.log:506`

**Source quote**

```
[23:21:17] step  24400/24414  loss 1.4281  lr 8.60e-07  tok/s  14,356  tok  99.9M/100M  ETA   0.1 min
[23:21:21] Training complete in 116.1 min.
(log:1:) [21:23:28] Device: cuda   dtype: torch.bfloat16
```

**Confidence** — measured from code

**Caveat** — "NVIDIA GB10" is PROSE-ONLY for this run (README.md:62, results/POST_DATA.md:17) — the log never records a device name. `nvidia-smi` on this box today returns `NVIDIA GB10`, which makes it near-certain but is present-day corroboration, not a run record. The logged tok/s is a cumulative average (tok_seen/elapsed), so POST_DATA.md:55-56's "~14,300 tok/s through the second hour" describes the running mean, not an instantaneous second-hour rate. Peak memory was not recorded (no peak_mem column in the CSV).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
All numbers verified. log:1 = `[21:23:28] Device: cuda   dtype: torch.bfloat16`; log:505 = `[23:21:17] step  24400/24414  loss 1.4281  lr 8.60e-07  tok/s  14,356  tok  99.9M/100M  ETA   0.1 min`; log:506 = `[23:21:21] Training complete in 116.1 min.` (NOTE: evidence_path says :506 but the first quoted line is :505 — the 116.1-min headline is at :506, so the citation still lands on the load-bearing line). 21:23:28→23:21:27 = 117.98 min ✓. 99,999,744/(116.1*60) = 14,355.4 ✓. GB10-is-prose CONFIRMED: README.md:62 = `| TinyStories run wall-clock (NVIDIA GB10, bf16) | **116.1 min**, 100M tokens, 24,414 steps | ...`; POST_DATA.md:17 same claim; nvidia-smi today returns `NVIDIA GB10`. Cumulative-average claim CONFIRMED twice over: train_tinystories.py:331 computes tps from t0 set once before the loop, AND the log shows a monotone rise (12,766 @ step 50 → 14,302 @ 11,750 → 14,356 @ 24,400) which only a running mean does. So POST_DATA.md:55-56's "~14,300 tok/s through the second hour" is indeed the running mean, not an instantaneous rate.
```


### 2.14 Which training script produced the run?

**Value**

```
SmolLM2-134(base)/train_tinystories.py — but the on-disk copy is NOT the version that produced these results.
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:256`

**Source quote**

```
csv_w.writerow(["step", "loss", "lr", "grad_norm", "peak_mem_mb", "tok_seen"])
(but results/tinystories_train.csv:1 is:) step,loss,lr,tok_seen
```

**Confidence** — measured from code

**Caveat** — Five independent proofs of script drift: (1) script writes a 6-column CSV header, the CSV has 4 columns; (2) script:139 logs `device=... dtype=... seed=...`, log:1 reads `Device: cuda   dtype: torch.bfloat16`; (3) script:140 logs `args={vars(args)}` — no such line in the log; (4) script defaults --eval_every 2000 and --ckpt_every 4000 would emit 12 `[eval @ step` and 6 `[ckpt @ step` lines — the log has 0 of each (grep -c returns 0,0); (5) script's save_ckpt writes training_recipe/optim/sched/rng keys, the actual checkpoint has none of them. File mtimes agree: train_tinystories.py 2026-05-19 23:16 vs results 2026-05-14 00:21. Git has a single commit (84a96c0) containing only the LATER version, so the run's exact source is not recoverable.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
All five drift proofs independently verified. (1) train_tinystories.py:256 writes a 6-column header; results/tinystories_train.csv:1 = `step,loss,lr,tok_seen` (4 cols). (2) script:139 emits `device={device}  dtype={dtype}  seed={args.seed}`; log:1 is `Device: cuda   dtype: torch.bfloat16` — different capitalisation, no seed field. (3) script:140 emits `args={vars(args)}`; grep -c "args=" on the log → 0. (4) grep -c for `[eval @ step` and `[ckpt @ step` → 0 and 0, though defaults 2000/4000 over 24,414 steps would force 12 and 7 respectively. (5) script save_ckpt (262-294) writes training_recipe/optim/sched/rng_*; the actual ck has none. I ADD A SIXTH PROOF: log:10 reads `PPL = 6.895  on 199,485 target tokens` while script:197 formats `PPL={base_ppl:.3f}  ({base_n:,} target tokens)` — different template. Git verified: `git log -- SmolLM2-134(base)/train_tinystories.py` → single commit 84a96c0; I extracted that blob and diffed it against the working tree — IDENTICAL, i.e. git holds only the later version. mtimes verified: train_tinystories.py 2026-05-19 23:16:42; results 2026-05-14 00:21. Run source is not recoverable.
```


### 2.15 Which eval script produced 6.8945 / 3.7900?

**Value**

```
The SAME training script's internal `evaluate()` (train_tinystories.py:62-78), called at line 196 (before) and line 362 (after). It was NOT eval_after_vs_base.py.
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:62`

**Source quote**

```
@torch.no_grad()
def evaluate(model, val_tokens, device, seq_len: int, max_windows: int = 200):
    """Sliding-window CE perplexity over val_tokens (no overlap)."""
```

**Confidence** — measured from code

**Caveat** — eval_after_vs_base.py exists and is the in-domain + OOD comparison script, but its declared outputs `results/tinystories_vs_base.json` and `results/tinystories_vs_base.md` DO NOT EXIST on disk (`ls results/tinystories_vs_base.json` -> No such file). It also would crash on its own fallback path: line 91 does `open("model.py")`, and model.py is absent from both disk and git. So no OOD/catastrophic-forgetting number for this checkpoint exists anywhere.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_tinystories.py:62-64 verified verbatim (`@torch.no_grad()` / `def evaluate(...)` / docstring `"""Sliding-window CE perplexity over val_tokens (no overlap)."""`). Absence of the alternative verified: eval_after_vs_base.py:138 = `with open(RESULTS / "tinystories_vs_base.json", "w") as f:` and :148 prints the save path, but `ls results/tinystories_vs_base.json` and `.md` both return No such file. eval_after_vs_base.py:91 = `    code_text = open("model.py").read() * 30` and model.py is absent from disk and from git ls-files — so the fallback path would indeed raise FileNotFoundError. Conclusion that no OOD/forgetting number exists for this checkpoint holds.
```


### 2.16 Exact re-run command

**Value**

```
NOT_FOUND as a recorded invocation. Nearest documented forms are in the root README quickstart. Best reconstruction: `cd "SmolLM2-134(base)" && python3 train_tinystories.py --token_budget 100_000_000` (everything else at script defaults).
```

**Evidence** — `README.md:170`

**Source quote**

```
# Continued pretraining on TinyStories from official weights.
python train_tinystories.py --token_budget 10_000_000     # ~10M tokens for a quick run

# Resume a run that died:
python train_tinystories.py --resume checkpoint_tinystories.pt --token_budget 100_000_000
```

**Confidence** — PROSE ONLY

**Caveat** — Three caveats. (a) The 100M run was NOT a --resume run: the log contains the baseline eval, which train_tinystories.py:194 skips whenever --resume is set, and the CSV carries a header (written only when resume is None). (b) Re-running the CURRENT script with those defaults would produce extra artifacts the original run does not have (mid-training eval CSV, periodic checkpoints, grad_norm/peak_mem columns) and would save optimizer/scheduler/RNG state. (c) train_tinystories.py does NOT `import safe_cuda` and there is no `sentinel.py preflight` in the path (imports at lines 17-39 are argparse/csv/math/pathlib/random/time/warnings/numpy/torch/datasets/transformers only) — re-running it as-is violates CLAUDE.md §C1 and §C6.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Root README.md:170-174 verified verbatim (`# Continued pretraining on TinyStories from official weights.` at :170, `python train_tinystories.py --token_budget 10_000_000` at :171, resume form at :174). Caveat (a) CONFIRMED: the log contains the baseline eval block, and script:194 (`if args.resume is None:`) skips it on resume; the CSV carries a header, written only at script:255-256 when resume is None. Caveat (b) CONFIRMED by the drift evidence. Caveat (c) CONFIRMED by direct grep: `grep -n "safe_cuda\|sentinel" train_tinystories.py` returns ZERO hits — the script imports only argparse/csv/math/pathlib/random/time/warnings/numpy/torch/datasets/transformers (lines 19-39). Re-running it as-is would violate CLAUDE.md §C1 (safe_cuda.guard before torch) and §C6 (sentinel preflight). That is a real, material warning and should stay attached to any published reproduce recipe.
```


### 2.17 Where is the resulting checkpoint, filename, size, format?

**Value**

```
/home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)/checkpoint_tinystories.pt — 269,144,681 bytes (269.1 MB / 256.7 MiB), mtime 2026-05-14 00:21:27 +0100. Format: torch.save zip archive (uncompressed, method=store), a dict with keys ['model','config','step','tok_seen','baseline_ppl','trained_ppl'].
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:294`

**Source quote**

```
torch.save(ck, args.ckpt_path)
(argparse default, line 124:) ap.add_argument("--ckpt_path", default="checkpoint_tinystories.pt")
(shell:) checkpoint_tinystories.pt: Zip archive data, at least v0.0 to extract, compression method=store
(stat:) checkpoint_tinystories.pt 269144681 bytes  mtime=2026-05-14 00:21:27.908514546 +0100
```

**Confidence** — measured from code

**Caveat** — Keys read live via torch.load(map_location='cpu'). The 'model' state_dict holds 273 tensors, ALL torch.bfloat16, totalling 162,826,560 elements = 134,515,008 unique params + 28,311,552 duplicated (lm_head.weight stored separately despite tie_word_embeddings=true). 269,144,681 bytes is consistent with 162,826,560 x 2 bytes + pickle overhead. It carries NO optimizer/scheduler/RNG state, so it is not resumable in the sense the current script's --resume path advertises.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
stat -c '%s' → 269144681 exactly; stat mtime → 2026-05-14 00:21:27.908514546 +0100; `file` → `Zip archive data, at least v0.0 to extract, compression method=store`. torch.load(map_location='cpu') gives exactly those six keys. I independently verified the tensor accounting: 273 tensors, dtype Counter({'torch.bfloat16': 273}), total 162,826,560 elements, and 'lm_head.weight' IS present in the state_dict despite tie_word_embeddings=true — 162,826,560 = 134,515,008 + 28,311,552 (= 49152×576) exactly, so the duplication claim is arithmetically confirmed. 162,826,560 × 2 = 325,653,120 bytes > the 269,144,681 file size, so the fact's line "269,144,681 bytes is consistent with 162,826,560 × 2 bytes + pickle overhead" is ARITHMETICALLY WRONG as a consistency argument — the file is SMALLER than 2 bytes/elem would imply (likely the tied lm_head is stored as a storage alias rather than a second copy in the zip). Drop that sentence; it does not affect the size, format, or key claims, which all verify. The "not resumable" note is correct: script:229-244 expects optim/sched/rng_* keys the file lacks.
```


### 2.18 Is the checkpoint version-controlled or backed up?

**Value**

```
No. It is gitignored (`*.pt`) and untracked — it exists only on this box's local disk. No HF Hub copy and no export dir found (results/lm_eval/ and hf_export/ do not exist).
```

**Evidence** — `.gitignore:20`

**Source quote**

```
# Checkpoints (270MB+ each; not for version control — use HF Hub or git-lfs)
*.pt
(git check-ignore -v output:) .gitignore:20:*.pt	SmolLM2-134(base)/checkpoint_tinystories.pt
```

**Confidence** — measured from code

**Caveat** — Given MEMORY.md's "Branch switch wipes gitignored evidence" guard, this 269 MB artifact is the sole copy of the -45% result's weights and is exactly the class of file that has been destroyed before on this repo.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
`git check-ignore -v` returns exactly `.gitignore:20:*.pt	SmolLM2-134(base)/checkpoint_tinystories.pt`. .gitignore:19 is the comment `# Checkpoints (270MB+ each; not for version control — use HF Hub or git-lfs)` and :20 is `*.pt` (evidence_path :20 is the load-bearing line even though the quote starts at :19). `ls results/lm_eval` and `ls hf_export` → No such file or directory. USEFUL SCOPING THE AGENT DID NOT ADD: the surrounding EVIDENCE is safe — `git ls-files SmolLM2-134\(base\)/results` shows tinystories_train.log, tinystories_train.csv, tinystories_before.txt, tinystories_after.txt, tinystories_summary.md, POST_DATA.md and training_recipe_resolved.json are ALL tracked in git. Only the 269 MB weights are the single-copy artifact. That sharpens the MEMORY.md branch-switch risk to the weights alone.
```


### 2.19 Does the config stored in the checkpoint match SmolLM2-135M?

**Value**

```
Yes: vocab_size 49152, hidden_size 576, intermediate_size 1536, num_hidden_layers 30, num_attention_heads 9, num_key_value_heads 3, max_position_embeddings 8192, rope_theta 100000.0, rms_norm_eps 1e-05, tie_word_embeddings true, attention_bias false.
```

**Evidence** — `SmolLM2-134(base)/checkpoint_tinystories.pt:0`

**Source quote**

```
ck['config'] = {"vocab_size": 49152, "hidden_size": 576, "intermediate_size": 1536, "num_hidden_layers": 30, "num_attention_heads": 9, "num_key_value_heads": 3, "max_position_embeddings": 8192, "rope_theta": 100000.0, "rms_norm_eps": 1e-05, "initializer_range": 0.041666666666666664, "tie_word_embeddings": true, "attention_bias": false, "attention_dropout": 0.0}
```

**Confidence** — results JSON

**Caveat** — Binary artifact, so there is no line number — read live with torch.load. Matches the architecture table at SmolLM2-134(base)/results/POST_DATA.md:41-43.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
CONFIRMED and STRENGTHENED beyond what the agent did. It compared only against POST_DATA.md:41-43 (verified verbatim: `30 decoder layers · hidden 576 · intermediate 1536 · 9 Q heads / 3 KV heads (GQA / 3:1) · head_dim 64 · vocab 49,152 · RoPE θ=100,000 (split-halves) · RMSNorm · / SwiGLU · tied embeddings · no biases anywhere.`). I additionally diffed the checkpoint config against the OFFICIAL upstream config.json in the HF cache (/home/yashb98/projects/qwen-distill/hf_cache/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json): all eleven claimed fields match exactly, including initializer_range 0.041666666666666664, attention_dropout 0.0, rms_norm_eps 1e-05, max_position_embeddings 8192, tie_word_embeddings true, attention_bias false. Only cosmetic difference: upstream rope_theta is the int 100000, the checkpoint stores 100000.0. Binary artifact so ":0" as a line number is a placeholder, correctly flagged.
```


### 2.20 Do the derived training-loss statistics in the prose check out?

**Value**

```
Mostly. Best single-batch loss 0.9087928533554077 at step 22,353 -> "0.9088 @ step 22,353" CONFIRMED. First 1000-step bucket mean 1.5860 -> "1.586" CONFIRMED. But "1.316 (last)" is the 23000-24000 bucket (1.3162); the actual LAST bucket 24000-24414 means 1.3138.
```

**Evidence** — `SmolLM2-134(base)/results/tinystories_train.csv:22354`

**Source quote**

```
22353,0.9087928533554077,0.00012662297767765718,91557888
(claim at results/POST_DATA.md:19:) | **0.9088** | Best single-step training loss (step 22,353, deep in WSD decay) | `results/tinystories_train.csv` |
(claim at results/POST_DATA.md:57:) - Bucket-mean training loss (1000-step buckets): 1.586 (first) → **1.316** (last).
```

**Confidence** — measured from code

**Caveat** — I recomputed min-loss and the bucket means directly from the 24,414-row CSV. The "1.316 (last)" label is off by one bucket — cosmetic, but it is a headline-adjacent number.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
I recomputed all of it from the 24,414-row CSV in Python. min loss = 0.9087928533554077 at step 22353 — and csv:22354 reads verbatim `22353,0.9087928533554077,0.00012662297767765718,91557888`. Bucket means: (0,1000] = 1.5859884355068208 → 1.586 ✓; (23000,24000] = 1.3161785732507705 → 1.316; (24000,24414] = 1.3138229582044814 (414 rows) → 1.314 at 3dp. POST_DATA.md:19 and :57 both quoted verbatim and correct as quotes. So the "1.316 (last)" label on POST_DATA.md:57 is genuinely off by one bucket. Worth noting the repo is internally inconsistent here: results/tinystories_summary.md:12 uses 1.313 for the last bucket, which is the (nearly) right value.
```


### 2.21 Are there two different TinyStories runs described in the repo?

**Value**

```
YES. results/tinystories_summary.md describes an EARLIER run: 137.3 min wall clock, after-PPL 3.7893, different generated samples. The 6.8945 -> 3.7900 / 116.1 min pair comes from the LATER run (the one in tinystories_train.log/.csv and the current checkpoint).
```

**Evidence** — `SmolLM2-134(base)/results/tinystories_summary.md:9`

**Source quote**

```
| TinyStories-val perplexity | **6.8945** | **3.7893** | **−45.0%** |
(line 14:) Wall clock: **137.3 minutes** on NVIDIA GB10, bf16. (Original estimate: 135 min — off by 2 min, ~1.5%.)
```

**Confidence** — measured from code

**Caveat** — results/POST_DATA.md:165 does label it `results/tinystories_summary.md  long-form write-up (prior run)`, and mtimes agree (summary.md 2026-05-13 22:07, i.e. before the final run's baseline eval at 22:25). But summary.md's recipe table (lines 67-82) and throughput table (lines 84-92, incl. "mean 12,150 tok/s", "max temp 72 C", "5 users on the box") describe the PRIOR run and must not be quoted as the recipe/throughput of the 116.1-min run.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Conclusion (two runs; summary.md = the earlier one) is CORRECT and well-evidenced. But "different generated samples" is only 2/3 true and must be tightened to: "all three BEFORE samples are byte-identical to results/tinystories_before.txt, and the AFTER sample for prompt 1 is byte-identical to results/tinystories_after.txt; only prompts 2 and 3 diverge in the AFTER column."
```

**Verifier note**

```
Core claim CONFIRMED on three independent axes. (1) results/tinystories_summary.md:9 = `| TinyStories-val perplexity | **6.8945** | **3.7893** | **−45.0%** |` vs after.txt's 3.7900 / ck['trained_ppl'] 3.78995. (2) summary.md:14 = `Wall clock: **137.3 minutes** on NVIDIA GB10, bf16.` vs log:506's 116.1 min. (3) POST_DATA.md:165 = `results/tinystories_summary.md            long-form write-up (prior run)`. Cited ranges verified: the recipe table spans summary.md:67-82 and the throughput block 84-92, including `| Mean across full run | **12,150** |` at :90 and the `max temp 72 °C ... 5 users on the box` note at :92 — the warning not to quote these as the 116.1-min run's recipe/throughput is CORRECT and important. TWO CORRECTIONS. (a) I diffed the generations: summary.md:23-24 (prompt 1 AFTER, ending `...He started to climb`) is byte-identical to results/tinystories_after.txt, as are all three BEFORE samples; prompts 2 and 3 AFTER genuinely diverge (`He was very sad because he had no friends` vs `He was very strong and brave`; `a small fairy named Lila` vs `a small, beautiful bird who loved to whistle`). Identical BEFOREs are expected — set_seed(0) then a deterministic baseline eval leaves the same RNG state. (b) The mtime argument needs a stated correction factor: log timestamps run exactly 1 h behind file mtimes (before.txt mtime 2026-05-13 22:25:16 +0100 vs log `[21:25:16] Baseline generations`), so summary.md's 22:07:53 mtime = log-clock 21:07:53, ~16 min BEFORE the run's 21:23:28 start. The agent's "before the baseline eval at 22:25" silently mixes clocks but lands on the right conclusion. Also note summary.md's bucket means (1.586 / 1.339 / 1.324 / 1.312 / 1.313) match the CURRENT CSV to 3-4 digits (I computed 1.5860 / 1.3383 / 1.3240 / 1.3121 / 1.3138) — consistent with both runs using seed 0 and the same shuffle order, which is why POST_DATA could reuse "1.586".
```


### 2.22 Was any out-of-domain / catastrophic-forgetting measurement made on the trained checkpoint?

**Value**

```
NO. No post-training wikitext-2 PPL, no code PPL, no downstream benchmark results exist on disk for checkpoint_tinystories.pt.
```

**Evidence** — `SmolLM2-134(base)/results/tinystories_summary.md:125`

**Source quote**

```
2. **Lower peak LR** (1e-4 instead of 3e-4): less catastrophic-forgetting risk
   if you also care about preserving general-text quality. We didn't measure
   wikitext-2 PPL post-training but it almost certainly got worse — that's the
   tradeoff continued pretraining always makes.
```

**Confidence** — measured from code

**Caveat** — Confirmed by absence: `results/tinystories_vs_base.json` (eval_after_vs_base.py's output) does not exist; `results/lm_eval/` (scripts/run_lm_eval.sh's output dir) does not exist; `results/tinystories_eval.csv` (mid-training PPL trace) does not exist. The wikitext-2 15.371 in results/perplexity.json is the BASE-vs-HF parity number, not a post-training number. So the -45.0% is an unbalanced, in-domain-only claim.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Quote verified but the line number is off by one: results/tinystories_summary.md:124 begins `2. **Lower peak LR** (1e-4 instead of 3e-4): less catastrophic-forgetting risk`, so the quoted block spans :124-127, not starting at :125 (:125 is `   if you also care about preserving general-text quality. We didn't measure`). Absence CONFIRMED by direct ls: results/tinystories_vs_base.json, results/tinystories_vs_base.md, results/lm_eval/, results/tinystories_eval.csv and hf_export/ ALL return "No such file or directory". scripts/run_lm_eval.sh:27 sets `OUT_DIR="results/lm_eval"` — never produced. results/perplexity.json verified: {"ours_ppl": 15.370989092449635, "hf_ppl": 15.370989964425396, "tokens": 62403, "dataset": "wikitext-2-raw-v1 validation", ...} — that is the BASE-vs-HF parity number, correctly not a post-training number. The "unbalanced, in-domain-only" framing is right and must ship with the -45.0%.
```


### 2.23 Is `model.py`, cited as a source in the dossier, actually on disk?

**Value**

```
NO — model.py does not exist in SmolLM2-134(base)/ (only model_full.py) and is not tracked in git. A stale __pycache__/model.cpython-312.pyc is the only trace.
```

**Evidence** — `SmolLM2-134(base)/results/POST_DATA.md:20`

**Source quote**

```
| **198 lines** | `model.py` line count for the from-scratch architecture | `wc -l model.py` |
(shell: `ls model*.py` -> ) model_full.py
(shell: `git ls-files | grep -i smollm` -> lists model_full.py, no model.py)
```

**Confidence** — NOT FOUND

**Caveat** — Also affects results/README.md:29 ("`param_count.log` — output of `python3 model.py`") and eval_after_vs_base.py:91 (`code_text = open("model.py").read() * 30`), which would raise FileNotFoundError. Peripheral to the PPL claim, but it is a broken citation in the same dossier.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
VALUE IS CORRECT. One citation in the caveat is wrong: the param_count.log/model.py reference is at results/README.md:25, NOT :29. results/README.md:29 is part of an unrelated bullet about training_recipe_resolved.json. results/README.md:4 also references `../model.py`.
```

**Verifier note**

```
VALUE CONFIRMED: `ls model*.py` → model_full.py only; `git ls-files | grep SmolLM2 | grep .py` lists model_full.py, train.py, train_tinystories.py, verify.py, generate.py, compare_with_hf.py, eval_after_vs_base.py, benchmark_training.py, _build_notebook.py, scripts/export_to_hf.py, tests/test_parity.py — no model.py. __pycache__/model.cpython-312.pyc exists (16,082 bytes, mtime 2026-05-19 10:46). POST_DATA.md:20 verified verbatim (`| **198 lines** | \`model.py\` line count ... | \`wc -l model.py\` |`), eval_after_vs_base.py:91 verified verbatim. CITATION ERROR: grep -n on results/README.md gives `25:- \`param_count.log\` — output of \`python3 model.py\` (param count + random-init forward).` and `4:\`../verify.py\` / \`../model.py\`)`. Line 29 is `  (warmup=2000, weight_decay=0.01, clip_grad=1.0, seq_len=2048, global_batch=512,` — a different bullet entirely. Note also that results/param_count.log IS tracked in git while its stated generator is gone, so the 198-line claim is unverifiable at any line number.
```


### 2.V Additional verifier findings (no 1:1 extracted fact)

**2.V1 — ✅ CONFIRMED** · LR schedule

**Checked against**

```
WSD: 200-step linear warmup to 3e-4; stable through step 19,531; linear decay from 19,532 (2.9993856e-4) to 0.0 at 24,414.
```

**Verifier note**

```
All five CSV rows verified verbatim at the exact cited line numbers: csv:2 = `1,1.9550740718841553,1.4999999999999998e-06,4096`; csv:201 = `200,1.7322626113891602,0.0003,819200`; csv:19532 = `19531,1.4177279472351074,0.0003,79998976`; csv:19533 = `19532,1.4647554159164429,0.00029993856235920535,80003072`; csv:24415 = `24414,1.3074886798858643,0.0,99999744`. train.py:45-56 verified: `def make_wsd_scheduler(optimizer, warmup_steps, total_steps, decay_frac=0.2)` with `decay_start = int(total_steps * (1.0 - decay_frac))`. int(24414*0.8) = 19531; 3e-4*(1 - 1/(24414-19531)) = 3e-4*(1-1/4883) = 2.9993856235920535e-4 — matches the logged float to all 17 digits. 3e-4*(1/200) = 1.5e-6 matches csv:2. The "derived from data not prose" framing is fair.
```


**2.V2 — ❌ WRONG** · GAP CHECK: "No dataset revision/sha pin ... and no local HF cache entry for either the dataset or HuggingFaceTB/SmolLM2-135M — a re-run is not byte-reproducible."

**Checked against**

```
No local HF cache entry for either the dataset or the model; re-run not byte-reproducible.
```

**Corrected value**

```
Both caches EXIST at the active HF_HOME (/home/yashb98/projects/qwen-distill/hf_cache): hub/models--HuggingFaceTB--SmolLM2-135M @ 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 (cached 2026-05-13 11:44-11:47) and hub/datasets--roneneldan--TinyStories @ f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 (cached 2026-05-13 14:16-14:20). Both predate the run's 21:23 start. The surviving true statement is narrower: the CODE passes no revision= pin, so a re-run on a machine without this cache could resolve a different revision.
```

**Verifier note**

```
This is the one hard refutation in the set. The agent inspected ~/.cache/huggingface/hub, but `env | grep HF_` shows HF_HOME=/home/yashb98/projects/qwen-distill/hf_cache, so the default path was never the active cache. `find /home/yashb98 -iname "*SmolLM2*"` and `-iname "*TinyStories*"` locate both snapshots there, with refs/main pinned to the shas above. I then re-ran the exact val packing loop against those cached artifacts today and reproduced results/tinystories_train.log:6 byte-for-byte (200,068 tokens). Byte-level data reproducibility is therefore DEMONSTRATED, not absent.
```


**2.V3 — ⚠️ NEEDS QUALIFIER** · GAP CHECK: "How many of the 21,990 TinyStories validation stories were consumed to reach the 200,068-token eval buffer is not recorded, so the exact eval subset cannot be reconstructed without re-running the packer."

**Checked against**

```
Not recorded; eval subset cannot be reconstructed without re-running the packer.
```

**Corrected value**

```
1,040 stories. The eval set is the first 1,040 non-empty stories of roneneldan/TinyStories split="validation" @ f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 (0 empty stories skipped).
```

**Verifier note**

```
Literally true that no artifact records it, but I closed the gap: after `python3 sentinel.py preflight` (exit 0, mem_available=83%), I re-executed train_tinystories.py:172-181 verbatim on CPU with the cached tokenizer and dataset. Output: `packed tokens: 200068  stories consumed: 1040  empty skipped: 0`. The 200,068 figure matches log:6 exactly, which both closes the gap and independently validates that the run-era packing code was identical to the on-disk version despite the script drift. Restate the gap as "not recorded in any artifact, but deterministically recoverable — measured today as 1,040 stories."
```


### 2.G Gaps — not determinable from disk

- The actual optimizer hyperparameters of the 100M run (weight_decay, betas, eps, grad_clip, seed) are unrecoverable: the run log has no `args=` line, the checkpoint has no `training_recipe` key, the CSV has no grad_norm column, and git holds only a LATER version of train_tinystories.py (single commit 84a96c0). The cited values are that later version's argparse defaults, restated as prose.
- The micro_batch / grad_accum split cannot be separated — only their product with seq_len (4,096 tokens/step) is logged. micro_batch=4, grad_accum=1 is prose.
- The exact CLI invocation of the run was never recorded anywhere on disk.
- The GPU model is not recorded in any run artifact (log line 1 says only `Device: cuda`). `NVIDIA GB10` for this run is prose; nvidia-smi confirms the box's GPU today but that is present-day corroboration.
- No dataset revision/sha pin for roneneldan/TinyStories, and no local HF cache entry for either the dataset or HuggingFaceTB/SmolLM2-135M — a re-run is not byte-reproducible.
- How many of the 21,990 TinyStories validation stories were consumed to reach the 200,068-token eval buffer is not recorded, so the exact eval subset cannot be reconstructed without re-running the packer.
- Peak GPU memory for the run was not recorded (the run-era CSV has no peak_mem_mb column).
- No OOD / catastrophic-forgetting / downstream-benchmark number exists for checkpoint_tinystories.pt — eval_after_vs_base.py and scripts/run_lm_eval.sh both left no outputs on disk.
- No across-seed CI, no iso-FLOP control arm, no second corpus — the -45.0% is a single-seed, single-corpus, in-domain paired measurement, which under CLAUDE.md §C10/§C17/§C18/§C25 supports at most a `directional` claim.
- The exact source code that produced the run is gone (not in git, overwritten on disk 2026-05-19), so nothing beyond what the log/CSV/checkpoint recorded can be recovered.

---

## 3. Qwen3 eval provenance (28.65 / 46.31 / 23.52 / 13.40)

<sub>Audit dimension: Qwen3-0.6B eval provenance for 28.65 / 46.31 / 23.52 / 13.40</sub>

### 3.1 CRITICAL: Is 13.40 this repo's OWN measurement of the released Qwen3-0.6B-Base, or a number copied from the Qwen3 tech report / HF card?

**Value**

```
OUR OWN MEASUREMENT. The repo downloaded the released HF checkpoint and evaluated it with its own eval code. Script: Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/eval_original_vs_repro.py, run 2026-06-09 16:51:36, result 13.400. It is NOT copied from any paper or model card.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/eval_original_vs_repro.py:47-51`

**Source quote**

```
from transformers import AutoModelForCausalLM
    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(device)
    ppl_orig, n = eval_ppl(hf, val, device)
    lines.append(f"ORIGINAL  Qwen3-0.6B-Base (36T tok)   val PPL = {ppl_orig:8.3f}  ({n:,} tok, {time.time()-t0:.0f}s)")
```

**Confidence** — measured from code

**Caveat** — The word 'published' used next to 13.40 in Qwen3-0.6B/PLOTS_INDEX.md:37 and Qwen3-0.6B/results_overview/plots/README.md:49 is MISLEADING wording — it means 'the published (released) model', not 'a published number'. The generator script make_overview_plots.py:18-19,51 correctly traces it to our own original_vs_repro.txt. Do not describe 13.40 on a model card as a reported/published figure.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Evidence quote exists verbatim at Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/eval_original_vs_repro.py:47-51. The script downloads and scores the real HF checkpoint: line 49 `hf = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16)`, with REPO="Qwen/Qwen3-0.6B-Base" imported from train_qwen3 (eval_original_vs_repro.py:19; train_qwen3.py:59). Scoring uses the repo's own eval_ppl (eval_original_vs_repro.py:26-36), not an external number. The caveat is also confirmed: Qwen3-0.6B/PLOTS_INDEX.md:37 reads "vs published 13.40 dashed line" and Qwen3-0.6B/results_overview/plots/README.md:49-51 reads "The published Qwen3-0.6B-Base 13.40 is an EXTERNAL reference" — both ambiguous wordings; Qwen3-0.6B/results_overview/make_overview_plots.py:18-19 and :51 correctly trace ORIGINAL_PPL = 13.40 to original_vs_repro.txt. Agree the card must not call 13.40 a reported/published figure.
```


### 3.2 13.40 — which artifact/log records it, and when?

**Value**

```
results/original_vs_repro.txt line 2 = 13.400; stdout captured in results/original_eval_run2.log line 6, timestamped 2026-06-09 16:51:36. No .json exists — the only result files are .txt and .log.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_vs_repro.txt:1-2`

**Source quote**

```
[2026-06-09 16:51:36] Original vs reproduction — val=300,000 tokens, 50 windows x 4096
ORIGINAL  Qwen3-0.6B-Base (36T tok)   val PPL =   13.400  (204,800 tok, 21s)
```

**Confidence** — measured from code

**Caveat** — Both files are git-tracked (verified with git ls-files --error-unmatch).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
original_vs_repro.txt:1-2 matches the quote character-for-character. original_eval_run2.log:6 identical. Both git-tracked (`git ls-files --error-unmatch` exit 0). "No .json" independently verified: repo-wide `grep -rl '13\.400\|13\.40' --include='*.json'` returns only Qwen3-0.6B/experiments/2026-06-24_qwen3-0.6b_data-dclm-vs-fineweb/c5_evidence.json (a methodology file for a different experiment), no result JSON. Minor timing nuance not in the fact: the 16:51:36 stamp is written at eval_original_vs_repro.py:43 at script start; 13.400 was measured ~21 s later (the `21s` field). An earlier attempt the same day (results/original_eval_wrapper.log, 11:19) crashed on an UnpicklingError at line 62 BEFORE printing any PPL — so there is no conflicting earlier value.
```


### 3.3 13.40 — what metric exactly, on which HF dataset id / config / split?

**Value**

```
FineWeb-Edu validation perplexity (NOT wikitext, NOT any standard benchmark). Corpus = HuggingFaceFW/fineweb-edu, config 'sample-10BT', split 'train' (streaming), from which a 300,000-token 'val' tail was carved. exp(mean token NLL) over 204,800 scored tokens.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:151`

**Source quote**

```
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
```

**Confidence** — measured from code

**Caveat** — NO dataset revision is pinned in the load_dataset call, and NO model revision is pinned in from_pretrained(REPO). Neither the fineweb-edu snapshot sha nor the Qwen3-0.6B-Base commit sha used on 2026-06-09 is recorded anywhere on disk. This number is therefore not exactly reproducible.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Dataset id/config/split CONFIRMED at the cited line. The '300,000-token val TAIL' clause is NOT supported by the cited path in its current state — it is only true of the historical splitter (git show e791875:Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:123-148).
```

**Verifier note**

```
train_qwen3.py:151 is verbatim: `ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)`. BUT the surrounding function at that path TODAY (train_qwen3.py:126-191) is the doc-disjoint seeded-hash splitter, which does NOT carve a tail; the tail behaviour is in commit e791875 line 141 `(buf if len(buf) < n_train else val).extend(ids + [eos])`. Citing the current file for a 'tail' is a citation mismatch (the fact list does disclose this in fact 17, but the card must not cite train_qwen3.py:151 as evidence for the tail). I independently corroborated that the 2026-06-09 caches came from the OLD splitter: torch.load on both caches shows keys == ['train','val'] with no 'decontam' key, whereas the new splitter saves {'train','val','decontam'} (train_qwen3.py:190). 'No revision pinned' also CONFIRMED — line 151 has no revision arg, eval_original_vs_repro.py:49 has no revision arg. 204,800 = 50x4096 confirmed by the loop bound at eval_original_vs_repro.py:30.
```


### 3.4 13.40 — sequence length, stride, tokenizer, tokens evaluated?

**Value**

```
seq_len 4096, stride 4096 (NON-overlapping windows, no sliding window), 50 windows = 204,800 scored tokens out of the 300,000-token val slice. Tokenizer = Qwen/Qwen3-0.6B-Base (the model's own tokenizer, vocab 151,936). Model loaded in bfloat16. Loss via chunked_cross_entropy (fp32 accumulation, chunk 8192).
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/eval_original_vs_repro.py:21-22,30-36`

**Source quote**

```
SEQ_LEN, MAX_WINDOWS = 4096, 50
CACHE = HERE / "results" / "tokcache_133072000_300000.pt"
...
    for begin in range(0, min(len(val) - SEQ_LEN, MAX_WINDOWS * SEQ_LEN), SEQ_LEN):
        ids = val[begin:begin + SEQ_LEN + 1].unsqueeze(0).to(device)
        out = model(input_ids=ids[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        loss = chunked_cross_entropy(logits, ids[:, 1:]) * (ids.size(1) - 1)
        nlls += loss.item(); n += ids.size(1) - 1
    return math.exp(nlls / max(1, n)), n
```

**Confidence** — measured from code

**Caveat** — Tokenizer identity confirmed at train_qwen3.py:59 (REPO = "Qwen/Qwen3-0.6B-Base") and :281 (AutoTokenizer.from_pretrained(REPO)); eval_original_vs_repro.py imports REPO from train_qwen3 (line 19).

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
All correct EXCEPT the vocab attribution: 151,936 is the MODEL config's vocab_size (Qwen3-0.6B/model.py:37 `vocab_size: int = 151_936  # config.json: vocab_size`), not the tokenizer's vocabulary size. len(tokenizer) is 151,669 (research/eval/private_heldout_v1/private_prose_v1.txt:456).
```

**Verifier note**

```
eval_original_vs_repro.py:21-22 and :30-36 match the quote verbatim. Stride == SEQ_LEN confirmed by `range(..., SEQ_LEN)` at line 30 → non-overlapping. bf16 at line 49. chunked_cross_entropy fp32 accumulation with chunk=8192 confirmed at train_qwen3.py:81-93 (`total = flat.new_zeros((), dtype=torch.float32)`, `flat[i:i+chunk].float()`). Tokenizer identity confirmed at train_qwen3.py:59 and :281. Do not write 'tokenizer vocab 151,936' on a card — say 'model vocab_size 151,936'.
```


### 3.5 13.40 — how many training tokens for that arm?

**Value**

```
36T tokens — but this is a COPIED figure, not measured here. It comes from the Qwen3 tech report summary transcribed into the build's training_plan.md. The '36T' annotation printed inside our own eval log is a hardcoded f-string label, not a measurement.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/training_plan.md:17-19`

**Source quote**

```
Per the Qwen3 tech report (verbatim summary):

- **Corpus:** 36T tokens across 119 languages
```

**Confidence** — PROSE ONLY

**Caveat** — So 13.40 is OURS (measured) but the '36T' it is paired with is THEIRS (copied from arxiv.org/abs/2505.09388, cited as [qwen3paper] at Qwen3-0.6B/README.md:29). Keep that split explicit on a model card.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
training_plan.md:17-19 matches the quote verbatim ("Per the Qwen3 tech report (verbatim summary):" / "- **Corpus:** 36T tokens across 119 languages"). The '36T tok' string in our own log is indeed a hardcoded f-string literal at eval_original_vs_repro.py:51, not a measurement. Citation [qwen3paper]: https://arxiv.org/abs/2505.09388 confirmed at Qwen3-0.6B/README.md:29. The measured/copied split is correctly drawn.
```


### 3.6 28.65 — which run/build produced it?

**Value**

```
Build 1, faithful reproduction: Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b, run_name 'baseline2tpp', checkpoint_qwen3_baseline2tpp.pt. It is the post-training AFTER eval at the final step 18,150.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.log:396`

**Source quote**

```
[18:05:19] AFTER val PPL=28.65 (BEFORE 185810.49; improvement +185781.84 = +100.0%)
```

**Confidence** — measured from code

**Caveat** — Also written to results/qwen3_baseline2tpp_after.txt:2 ("val PPL: 185810.49 -> 28.65"). Single run, single seed (seed 0), no CI. Fresh run, not resumed (log:12 'Training to 18,150 steps (starting at 0)').

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
qwen3_baseline2tpp_train.log:396 matches verbatim: `[18:05:19] AFTER val PPL=28.65 (BEFORE 185810.49; improvement +185781.84 = +100.0%)`. Corroborated at qwen3_baseline2tpp_after.txt:2 `val PPL: 185810.49 -> 28.65`. Fresh (not resumed) confirmed at log:12 `Training to 18,150 steps (starting at 0)` and log:2 `'resume': None`. Single seed 0 confirmed at log:1 and log:2.
```


### 3.7 28.65 — metric, dataset, seq len, stride, tokenizer, tokens evaluated?

**Value**

```
FineWeb-Edu val PPL, same eval function as 13.40 (train_qwen3.evaluate): seq_len 4096, stride 4096 non-overlapping, max_windows 50 = 204,800 scored tokens, tokenizer Qwen/Qwen3-0.6B-Base, bf16, chunked fp32 CE. Dataset HuggingFaceFW/fineweb-edu / sample-10BT / split=train (streaming).
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:195,199-207`

**Source quote**

```
def evaluate(model, val_tokens, device, seq_len: int, max_windows: int = 50):
...
    for begin in range(0, min(len(val_tokens) - seq_len, max_windows * seq_len), seq_len):
        ids = val_tokens[begin:begin + seq_len + 1].unsqueeze(0).to(device)
        logits = model(ids[:, :-1])["logits"]
        loss = chunked_cross_entropy(logits, ids[:, 1:]) * (ids.size(1) - 1)
        nlls += loss.item()
        n += ids.size(1) - 1
    ...
    return math.exp(nlls / max(1, n)), n
```

**Confidence** — measured from code

**Caveat** — CRITICAL: 28.65 was measured on a DIFFERENT 300k val slice than 13.40 — see the dedicated finding below.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_qwen3.py:195 and :199-207 match the quote verbatim. max_windows defaults to 50 and is never overridden — all three call sites pass only (model, val_tokens, device, args.seq_len): train_qwen3.py:341, :412, :426. stride == seq_len via `range(..., seq_len)` at :199. The cross-slice caveat is the correct one to carry forward.
```


### 3.8 28.65 — training tokens / steps / recipe?

**Value**

```
18,150 optimizer steps x 65,536 tok/step = 1,189,478,400 tokens (1.19B, ~2 tokens-per-parameter). AdamW betas (0.9,0.95) eps 1e-8, cosine peak_lr 2.4e-3 -> end_lr 3.2e-4, warmup 900, weight_decay 0.01, grad_clip 1.0, bf16, seq_len 4096, micro_batch 4 x grad_accum 4, seed 0, torch.compile on. ~2663 min wall (~44 h) at ~7,480 tok/s.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.log:2-3`

**Source quote**

```
[21:14:18] args={'steps': 18150, 'seq_len': 4096, 'micro_batch': 4, 'grad_accum': 4, 'peak_lr': 0.0024, 'end_lr': 0.00032, 'warmup_steps': 900, 'weight_decay': 0.01, 'grad_clip': 1.0, 'mem_fraction': 0.85, 'seed': 0, 'dtype': 'bfloat16', 'log_every': 50, 'eval_every': 2000, 'ckpt_every': 2000, 'no_compile': False, 'run_name': 'baseline2tpp', ...}
[21:14:18] tok/step=65,536  steps=18,150  token_budget=1,189,478,400
```

**Confidence** — measured from code

**Caveat** — Wall-clock/throughput figures from builds/2026-06-08_reproduce-faithful_qwen3-0.6b/README.md:167 and log:395 ('Training complete in 2663.1 min').

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Everything confirmed EXCEPT the throughput: the run-average is ~7,444 tok/s, not ~7,480. 1,189,478,400 tok / (2663.1 min x 60) = 7,444 tok/s, and the log's own final counter reads 7,444 (qwen3_baseline2tpp_train.log:393). 7,480 is the step-100 reading (log:14) that README.md:167 generalised.
```

**Verifier note**

```
Args dict at log:2 matches the quote verbatim; log:3 gives tok/step=65,536, steps=18,150, token_budget=1,189,478,400. betas=(0.9,0.95) eps=1e-8 confirmed at train_qwen3.py:302 (`lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8`) — these are NOT in the log's args dict, so they come from source defaults, which is fine but worth stating. 'Training complete in 2663.1 min.' confirmed at log:395. If the card quotes throughput, quote 7,444 tok/s (measured) not README.md:167's 7,480.
```


### 3.9 23.52 — which run/build produced it, and at which step?

**Value**

```
Build 2, modernized 'IMU-1' bundle: Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b, run_name 'imu1_2tpp'. It is the IN-LOOP eval at STEP 18,000 — NOT the final step 18,150. train_imu1.py has no post-training AFTER eval, so no end-of-run number exists.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log:381`

**Source quote**

```
[09:27:03]   [eval @ 18000] val PPL=23.52
```

**Confidence** — measured from code

**Caveat** — MATERIAL ASYMMETRY: 23.52 is at 18,000 steps = 1,179,648,000 tokens, while 28.65 is at 18,150 steps = 1,189,478,400 tokens (0.83% more). The like-for-like same-step comparison is baseline 28.66 @ step 18000 (qwen3_baseline2tpp_train.log:389) vs IMU-1 23.52, so the -17.9% headline survives — but 23.52 vs 28.65 is not step-matched as printed.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
qwen3_imu1_2tpp_train.log:381 matches verbatim: `[09:27:03]   [eval @ 18000] val PPL=23.52`. The log has exactly 386 lines and ends `[09:58:25] step 18150/18150 ...` / `[09:58:25] DONE` — no AFTER eval, confirming no end-of-run number exists. The step-asymmetry caveat is also confirmed: baseline 28.66 @ step 18000 at qwen3_baseline2tpp_train.log:389. Arithmetic checks: 23.52/28.66 = -17.94%, 23.52/28.65 = -17.91% — the -17.9% headline does survive the step-matching. This asymmetry is material and correctly flagged.
```


### 3.10 23.52 — metric, dataset, seq len, stride, tokenizer, val slice?

**Value**

```
Identical harness to 28.65: train_imu1.py imports evaluate + stream_tokens + PackedTextDataset directly from train_qwen3, and loaded the SAME token cache (tokcache_1191478400_300000.pt). seq 4096, stride 4096, 50 windows = 204,800 scored tokens, tokenizer Qwen/Qwen3-0.6B-Base, FineWeb-Edu sample-10BT.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log:3`

**Source quote**

```
[18:05:40]   loaded cached tokens from tokcache_1191478400_300000.pt (1,191,478,400 train + 300,000 val)
```

**Confidence** — measured from code

**Caveat** — Cross-check: train_imu1.py:155-158 ('from train_qwen3 import stream_tokens, evaluate, PackedTextDataset' ... 'stream_tokens(tokenizer, token_budget + 2_000_000, 300_000, log_path)') and :156 AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base"). 23.52 and 28.65 ARE on the same val slice and are mutually comparable.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
qwen3_imu1_2tpp_train.log:3 matches verbatim. train_imu1.py:155 (`from train_qwen3 import stream_tokens, evaluate, PackedTextDataset`), :156 (`AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")`), :158 (`stream_tokens(tokenizer, token_budget + 2_000_000, 300_000, log_path)`) all confirmed. train_imu1.py:202 calls `evaluate(model, val_tokens, device, seq)` with max_windows defaulted to 50. The faithful run built that same cache (baseline log:6 `streamed 1,191,478,748 train + 300,012 val tokens`, cache key n_train=18150*65536+2,000,000=1,191,478,400 per train_qwen3.py:283-285). Same-slice claim for this PAIR is sound.
```


### 3.11 23.52 — training tokens / recipe?

**Value**

```
1,179,648,000 tokens at the 23.52 eval point (18,000 x 65,536); full budget 18,150 steps = 1,189,478,400. Recipe: value-residual + layernorm-scaling + head-gating architecture (vr=ln=hg=True), NorMuon on 224 2-D matrices + AdamW on 198 1-D/embedding params, WSD schedule (normuon_lr 0.011 / adam_lr 0.006 defaults, 20% linear decay tail), chunked z-loss 1e-4, seq 4096, micro_batch 4 x grad_accum 4, seed 0.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log:1-2`

**Source quote**

```
[18:05:35] device=cuda bundle: vr=True ln=True hg=True  steps=18150
[18:05:35] param split: 224 NorMuon (2D), 198 AdamW (1D/embed)  tok/step=65,536
```

**Confidence** — measured from code

**Caveat** — The log does NOT echo the LR/z-weight values, so normuon_lr=0.011 / adam_lr=0.006 / z_weight=1e-4 / weight_decay=0.1 / warmup=50 / decay_frac=0.2 are the argparse DEFAULTS at train_imu1.py:96-107, not confirmed from the run record. If the launcher passed overrides they are not on disk in this log.

**Verdict — ❌ WRONG**

**Corrected value**

```
warmup_steps was 900, NOT 50. And the exact CLI IS on disk: Qwen3-0.6B/builds/phase_b_driver.sh:24 records `cd "$MOD" && python train_imu1.py --steps $S --warmup_steps $W $COMMON --run_name imu1_2tpp` with S=18150, W=900, COMMON="--eval_every 2000 --ckpt_every 2000 --log_every 50" (phase_b_driver.sh:14). normuon_lr 0.011 / adam_lr 0.006 / weight_decay 0.1 / decay_frac 0.2 / z_weight 1e-4 were NOT overridden, so those defaults do hold.
```

**Verifier note**

```
I confirmed warmup=900 empirically against the run's own LR ramp, which the defaults cannot produce: 0.011 x 50/900 = 6.111e-4 == log:5 `lr 6.11e-04`; 0.011 x 100/900 = 1.222e-3 == log:6 `lr 1.22e-03`; 0.011 x 400/900 = 4.889e-3 == log:12 `lr 4.89e-03`. Under warmup=50 the LR would already be at peak 0.011 by step 50. decay_frac=0.2 also confirmed empirically: decay tail starts at step 14,520; at step 17,750 the predicted lr = 0.011 x 400/3630 = 1.212e-3 == log:375 `lr 1.21e-03`. Token count 18,000 x 65,536 = 1,179,648,000 confirmed. Both the '50' value and the 'no CLI on disk' gap claim must be corrected before publication.
```


### 3.12 46.31 — which run produced it, and what does it actually measure?

**Value**

```
Phase-A LR sweep arm 'lr24' (peak_lr 2.4e-3) in the faithful build. AFTER val PPL = 46.31 on FineWeb-Edu val — the same in-loop metric as 28.65, NOT a wikitext or benchmark number. Sweep siblings: lr17 (1.7e-3) = 46.89, lr30 (3.0e-3) = 49.28; lr24 best.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_lr24_train.log:226`

**Source quote**

```
[06:15:14] AFTER val PPL=46.31 (BEFORE 183922.14; improvement +183875.83 = +100.0%)
```

**Confidence** — measured from code

**Caveat** — Also in results/qwen3_lr24_after.txt:2 ('val PPL: 183922.14 -> 46.31') and independently re-scored as 46.310 by eval_original_vs_repro.py (original_vs_repro.txt:4). Siblings from qwen3_lr17_after.txt:2 (46.89) and qwen3_lr30_after.txt:2 (49.28).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
qwen3_lr24_train.log:226 matches verbatim: `[06:15:14] AFTER val PPL=46.31 (BEFORE 183922.14; improvement +183875.83 = +100.0%)`. qwen3_lr24_after.txt:2 = `val PPL: 183922.14 -> 46.31`. Siblings: qwen3_lr17_after.txt:2 = 46.89, qwen3_lr30_after.txt:2 = 49.28. Re-score confirmed at original_vs_repro.txt:3-5 (46.892 / 46.310 / 49.276). One framing nit: the original_vs_repro re-score is the SAME eval code on the SAME cache, so it is a consistency check, not an independent measurement — do not present it on a card as corroboration by a second method.
```


### 3.13 46.31 — training tokens / steps / recipe?

**Value**

```
2,000 steps x 65,536 tok/step = 131,072,000 tokens (131M). AdamW, cosine peak_lr 2.4e-3 -> end_lr 3.2e-4, warmup 150, weight_decay 0.01, grad_clip 1.0, bf16, seq 4096, micro_batch 4 x grad_accum 4, seed 0, torch.compile on.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_lr24_train.log:2-3`

**Source quote**

```
[01:21:12] args={'steps': 2000, 'seq_len': 4096, 'micro_batch': 4, 'grad_accum': 4, 'peak_lr': 0.0024, 'end_lr': 0.00032, 'warmup_steps': 150, ... 'run_name': 'lr24', ...}
[01:21:12] tok/step=65,536  steps=2,000  token_budget=131,072,000
```

**Confidence** — measured from code

**Caveat** — Confirms the orchestrator's framing: 46.31 is a SHORT 2k-step / 131M-token LR-selection run, not a headline model.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
qwen3_lr24_train.log:2 args dict and :3 (`tok/step=65,536  steps=2,000  token_budget=131,072,000`) match the quote verbatim, including warmup_steps: 150 and no_compile: False. The 'short LR-selection run, not a headline model' framing is correct — log:225 shows the run took only 293.4 min vs the baseline's 2663.1.
```


### 3.14 46.31 — seq len, stride, tokenizer, tokens evaluated, val slice?

**Value**

```
seq 4096, stride 4096 non-overlapping, 50 windows = 204,800 scored tokens, tokenizer Qwen/Qwen3-0.6B-Base, val slice tokcache_133072000_300000.pt — the SAME slice used for 13.40.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_lr24_train.log:6`

**Source quote**

```
[01:21:13]   loaded cached tokens from tokcache_133072000_300000.pt (133,072,000 train + 300,000 val)
```

**Confidence** — measured from code

**Caveat** — GOOD NEWS: 46.31 vs 13.40 IS an apples-to-apples same-slice comparison (both from tokcache_133072000_300000.pt, both via 50x4096 non-overlapping windows). The '3.5x' gap at 131M tokens is internally consistent.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
qwen3_lr24_train.log:6 matches verbatim: `loaded cached tokens from tokcache_133072000_300000.pt (133,072,000 train + 300,000 val)`. That is the exact file hardcoded at eval_original_vs_repro.py:22 (`CACHE = HERE / "results" / "tokcache_133072000_300000.pt"`), so 46.31 and 13.40 really are same-slice, same-windowing. The 3.5x ratio is internally consistent (46.310/13.400 = 3.456).
```


### 3.15 BLOCKER: Is the README's claim that all four numbers share one val slice true?

**Value**

```
NO — IT IS FALSE. 13.40 and 46.31 were measured on tokcache_133072000_300000.pt; 28.65 and 23.52 on tokcache_1191478400_300000.pt. These are two DIFFERENT 300,000-token FineWeb-Edu slices. I verified this empirically: their val tensors have different sha1 (8ad9e246b0bf63bd vs ad3513719d0f81e4) and different leading tokens ([10879, 5547, 481, ...] vs [38131, 6022, 369, ...]).
```

**Evidence** — `Qwen3-0.6B/README.md:35-37`

**Source quote**

```
All perplexities use **identical eval code on the identical 300k-token FineWeb-Edu
val slice** ([`eval_original_vs_repro.py`](builds/2026-06-08_reproduce-faithful_qwen3-0.6b/eval_original_vs_repro.py)),
so every row is directly comparable.
```

**Confidence** — measured from code

**Caveat** — Root cause: under the ORIGINAL splitter (git show e791875:.../train_qwen3.py, stream_tokens line 33 'cache = RESULTS / f"tokcache_{n_train}_{n_val}.pt"' with val = the sequential stream continuation AFTER n_train tokens), a different n_train yields a different val tail. n_train was 133,072,000 for the sweep and 1,191,478,400 for the 2-TPP runs. CONSEQUENCE: the '2.14x gap vs the original' (28.65/13.40) and the '1.76x' (23.52/13.40) are CROSS-SLICE ratios and should NOT be stated on a model card as a like-for-like gap. The 3.5x (46.31/13.40) is same-slice and is fine.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
I reproduced this empirically rather than taking it on trust. torch.load(..., mmap=True) on both caches: tokcache_133072000_300000.pt val sha1 = 8ad9e246b0bf63bd, first10 = [10879, 5547, 481, 37969, 1935, 82, 481, 6467, 11, 18652]; tokcache_1191478400_300000.pt val sha1 = ad3513719d0f81e4, first10 = [38131, 6022, 369, 66863, 25471, 2757, 374, 3709, 5313, 6529]. Both len 300,000, dtype int64. Exact match to the claimed values. Qwen3-0.6B/README.md:35-37 quote is verbatim and is FALSE as written. Two nits: (a) the caveat cites 'stream_tokens line 33' for the old cache-key line — the actual line in `git show e791875:...train_qwen3.py` is 127, not 33 (substance correct, line number wrong); (b) the same false same-slice claim appears a SECOND time at Qwen3-0.6B/results_overview/plots/README.md:50 ('36T-token model evaluated on the same 300k-token val set'), so a card fix must touch both docs. The 2.14x and 1.76x ratios are cross-slice and must not be published as like-for-like gaps.
```


### 3.16 How slice-sensitive is this metric in practice?

**Value**

```
Very. The SAME faithful checkpoint (checkpoint_qwen3_baseline2tpp.pt) that scores 28.65 on its own 300k slice scores 24.55 on the dataset-forge held-out FineWeb-Edu split under text-lm-v2 windowing (SEQ=1024, STRIDE=512, 202 docs / 204,600 scored tokens) — a ~14% swing from slice + windowing alone.
```

**Evidence** — `research/ledger/runs/2026-06-17_qwen3-0.6b_vibethinker-small-reasoning.md:102`

**Source quote**

```
| **§C13 forgetting — FineWeb-Edu** (`fineweb_edu_sample10bt_heldout`, 202 docs / 204,600 scored tok) | 24.5514 | 24.7331 | **+0.18 (+0.74%)** | 8.0155 | **retained — not significant** |
```

**Confidence** — results JSON

**Caveat** — The same ledger doc (line 113-114) mis-attributes 28.65 to eval_original_vs_repro.py; the actual source is the in-loop train_qwen3.evaluate() AFTER eval in qwen3_baseline2tpp_train.log:396 / qwen3_baseline2tpp_after.txt:2. eval_original_vs_repro.py only ever scored the HF original + lr17/lr24/lr30.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
research/ledger/runs/2026-06-17_qwen3-0.6b_vibethinker-small-reasoning.md:102 matches verbatim. Backed by an actual results JSON: Qwen3-0.6B/experiments/2026-06-17_qwen3-0.6b_vibethinker-small-reasoning/eval/brief_probes_results.json gives base_ppl 24.55139646501548, base_ckpt = .../checkpoint_qwen3_baseline2tpp.pt, n_tokens 204600, and an explicit field `"base_ppl_claim_readme": 28.65` with `"base_ppl_measured_vs_claim_delta": -4.09860353498452` (= -14.3%). Windowing SEQ=1024 STRIDE=512 MAX_WINDOWS=200 confirmed at the same ledger doc line 95. The caveat is confirmed too: lines 113-114 do mis-attribute 28.65 to eval_original_vs_repro.py; the real source is train_qwen3.evaluate's AFTER eval (log:396 / after.txt:2).
```


### 3.17 Were the val slices behind these four numbers decontaminated / document-disjoint?

**Value**

```
NO. All four numbers used caches built by the ORIGINAL splitter, which the repo's own current code calls leak-suspect. The doc-disjoint seeded-hash split + 13-gram decontamination was added LATER (caches carrying it are named with a _seed0_<tokenizer> suffix, e.g. tokcache_133072000_300000_seed0_Qwen3-0.6B-Base.pt dated 2026-06-18). The caches used here have no seed/tokenizer suffix and no decontam record.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:130-141`

**Source quote**

```
"""Stream FineWeb-Edu sample-10BT, tokenize on the fly, and build a
    DOCUMENT-DISJOINT, DECONTAMINATED train/val split (audit fix DATA-1/3):

      * each whole document is routed to train or val by a seeded hash
        (`is_val_doc`), so train/val never share a document and no document spans
        the boundary (the old code cut the stream by token count — val was the
        sequential continuation of train, leak-suspect);
      * val documents whose 13-gram word overlap with a bounded sample of train
        documents exceeds `decontam_threshold` are DROPPED
```

**Confidence** — measured from code

**Caveat** — results/decontam_report.json exists but is dated 2026-07-07 and belongs to a LATER cache (tokcache_422020224_300000_seed0_...). It does NOT cover the 133M or 1191M caches. Separately, for 13.40 specifically: the released Qwen3-0.6B-Base's 36T corpus may itself contain FineWeb-Edu / its CommonCrawl sources, so this slice is not verifiably held out FOR THAT MODEL. That is an inference from the tech-report data description, not something provable on disk.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_qwen3.py:130-141 matches the quote verbatim, including the self-indictment 'the old code cut the stream by token count — val was the sequential continuation of train, leak-suspect'. I proved the two caches predate the fix independently of filenames: neither contains a 'decontam' key (keys == ['train','val']), whereas the post-fix splitter saves {'train','val','decontam'} (train_qwen3.py:190). Cache-name scheme with seed/tokenizer suffix confirmed at train_qwen3.py:145; tokcache_133072000_300000_seed0_Qwen3-0.6B-Base.pt mtime 2026-06-18 13:45. decontam_report.json mtime 2026-07-07 15:13, matching tokcache_422020224_300000_seed0_*.pt (2026-07-07 15:13). One precision note: 'belongs to a LATER cache' is an mtime inference — decontam_report.json contains no cache-name field (its keys are split_seed, val_fraction, ngram_n, overlap_threshold, n_train_docs, n_val_docs_raw, docs_dropped, n_val_docs_kept, method, train_sample_docs_for_index) — but the operative claim (it does not cover the 133M/1191M caches) is proven by the missing 'decontam' key. The Qwen3-36T-corpus-overlap point is correctly labelled an inference, not disk evidence.
```


### 3.18 Does this repo's own governance permit these four numbers as a model-card headline?

**Value**

```
NO. The repo explicitly classifies the 28.65 / 23.52 / 29.54 val-PPL family as the 'founding-mistake metric', banned as a sole or headline signal by contract §C25.7.3, and reports it only as cross-check context. All four numbers are n=1, single-seed, no CI, in-distribution val PPL.
```

**Evidence** — `research/eval/base_eval_verdict.md:59`

**Source quote**

```
- val-PPL headline (28.65 / 23.52 / 29.54) source `Qwen3-0.6B/PLOTS_INDEX.md:20,22,24`. It is **n=1 FineWeb val-PPL — the founding-mistake metric, banned as a sole/headline signal by §C25.7.3** — reported here as cross-check context only. Published Qwen3-0.6B reference is 13.40 (`PLOTS_INDEX.md:37`); the reproductions are ~1.76–2.21× above it, consistent with the ≤1.19B-token undertraining.
```

**Confidence** — PROSE ONLY

**Caveat** — Corroborated at research/eval/per_stage_eval_batteries.md:9 ('the three-build pretraining headline ... was shipped on n=1 FineWeb val PPL — no downstream, no seed CI, no contamination performance-check').

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
research/eval/base_eval_verdict.md:59 matches the quote verbatim, including '**n=1 FineWeb val-PPL — the founding-mistake metric, banned as a sole/headline signal by §C25.7.3**'. Corroboration confirmed at research/eval/per_stage_eval_batteries.md:9 verbatim ('shipped on **n=1 FineWeb val PPL** — no downstream ... no seed CI, no contamination performance-check'). The 'prose-only' confidence label is the right one — these are governance documents, not results files.
```


### 3.19 What ARE the §C10-comparable, suite-stamped numbers for the 28.65 checkpoint (safer for a model card)?

**Value**

```
From /eval-harness suite text-lm-v2, run 2026-06-16 on checkpoint_qwen3_baseline2tpp.pt: wikitext2_raw_v1_val PPL 37.0101 / BPB 1.22562 (204,600 tokens, 869,710 bytes); codeparrot_clean_valid PPL 438.673 / BPB 2.12860 (204,600 tokens, 843,643 bytes). Windowing SEQ=1024, STRIDE=512, MAX_WINDOWS=200. Corpora revision-pinned. Tokenizer Qwen/Qwen3-0.6B-Base.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3-faithful_eval-first/eval/suite_results.json:1-25`

**Source quote**

```
"suite_version": "text-lm-v2",
 "tokenizer_repo": "Qwen/Qwen3-0.6B-Base",
 "target_ckpt": ".../checkpoint_qwen3_baseline2tpp.pt",
 "ppl": {
  "wikitext2_val": {
   "corpus_id": "wikitext2_raw_v1_val",
   "target": 37.010055463333096,
   "n_tokens": 204600,
   "bpb": 1.2256204566076285,
```

**Confidence** — results JSON

**Caveat** — Window config at Qwen3-0.6B/experiments/2026-06-16_qwen3-faithful_eval-first/eval_suite.py:65 ('SEQ, STRIDE, MAX_WINDOWS = 1024, 512, 200'); wikitext loaded as Salesforce/wikitext config wikitext-2-raw-v1 split validation with a pinned revision (eval_suite.py:165-166), codeparrot/codeparrot-clean-valid split train pinned to 4db92d2ec0c1b4c41eeb439cfae16854511d9dcd (eval_suite.py:61,174-175). These windows OVERLAP (stride<seq), which the script itself flags as double-counting — comparable across runs of this suite, not to the 4096/non-overlapping in-loop numbers. NOTE: this suite was NOT run on the released Qwen3-0.6B-Base, so there is no suite-comparable counterpart to 13.40.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Every value matches Qwen3-0.6B/experiments/2026-06-16_qwen3-faithful_eval-first/eval/suite_results.json to full precision (target 37.010055463333096, bpb 1.2256204566076285, n_bytes 869710; target 438.67295146042875, bpb 2.128595386220801, n_bytes 843643; date '2026-06-16 22:18:01'; target_ckpt = checkpoint_qwen3_baseline2tpp.pt). Dataset-config check — the classic failure mode the brief warns about — PASSES: eval_suite.py:165-166 genuinely says `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation", revision=WIKITEXT_REV)`, the '-raw-' variant IS what the code names. Revisions pinned at eval_suite.py:60 (wikitext b08601e04326c79dfdd32d625aee71d232d685c3) and :61 (codeparrot 4db92d2ec0c1b4c41eeb439cfae16854511d9dcd — matching the fact); codeparrot loaded split='train' at :174-175. SEQ, STRIDE, MAX_WINDOWS = 1024, 512, 200 at :65. Double-count self-flag confirmed at :126 ('STRIDE<SEQ double-counts tokens AND'). 'Not run on the released Qwen3-0.6B-Base' CONFIRMED: I enumerated all 9 suite_results*.json in the repo — every target_ckpt is a local .pt/.pkl, none is an HF repo id. Extra finding, no conflict: a second same-day run (2026-06-16_qwen3-0.6b_eval-faithful/eval/suite_results.json, 22:26:53) reports the identical PPLs with bpb null.
```


### 3.20 Downstream (non-PPL) numbers for the same checkpoints, if the card needs a defensible metric

**Value**

```
text-lm-v3 battery, scored 2026-06-24, n=500 items/task with Wilson CIs: faithful baseline LAMBADA acc 0.170 [0.140, 0.205], mean BPB-gold 1.188; IMU-1 LAMBADA 0.212 [0.178, 0.250], mean BPB-gold 1.142; partial-RoPE-0.25 LAMBADA 0.166 [0.136, 0.201], BPB-gold 1.202. MC tasks are at/near chance and are reported as no-signal.
```

**Evidence** — `research/eval/downstream_v3/build_faithful.json:1-15`

**Source quote**

```
{
 "name": "build_faithful",
 "tokens": "1.19B",
 "tasks": {
  "lambada": {
   "metric": "lambada_acc",
   "acc": 0.17,
   "ci": [
    0.13962036610396117,
    0.20541169860066333
   ],
   "chance": 0.0,
   "bpb_gold": 1.2357421338326326,
   "n": 500
  },
```

**Confidence** — results JSON

**Caveat** — The per-build JSONs contain NO val-PPL field — the 28.65/23.52 column in research/eval/base_eval_verdict.md:51-56 was transcribed from PLOTS_INDEX.md, not produced by the downstream harness. winogrande sits exactly at chance (acc 0.500, signal:false) for the faithful build.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
The LAMBADA and BPB numbers are all correct. But 'MC tasks are at/near chance and reported as no-signal' is over-broad: in the very JSON cited, arc_easy acc_norm 0.454 (chance 0.25) and hellaswag acc_norm 0.348 (chance 0.25) both carry `"signal": true`. ONLY winogrande is `"signal": false` (acc_norm 0.500, chance 0.500).
```

**Verifier note**

```
build_faithful.json:1-15 matches the quote verbatim (acc 0.17, ci [0.13962036610396117, 0.20541169860066333], bpb_gold 1.2357421338326326, n 500). CIs for the other builds confirmed: build_imu1.json lambada ci [0.17843950428829852, 0.24995211581755555]; build_prope25.json [0.13595774323101012, 0.2011353161973068]. I recomputed the mean BPB-gold from all four per-task values in each JSON: faithful 1.18827, imu1 1.14232, prope25 1.20234 — the fact's 1.188/1.142/1.202 are correct. Two provenance gaps: 'scored 2026-06-24' and 'mean BPB-gold' are NOT in the cited build_faithful.json — they come from research/eval/downstream_v3/RESULTS.md:3,9-11 and research/eval/base_eval_verdict.md:49,58. The over-broad MC wording is copied from RESULTS.md:3; the repo's own more careful statement is at research/eval/base_eval_verdict.md:63: arc_easy and hellaswag are 'flagged `signal: true` but only marginally above chance with wide CIs; not headline-bearing'. Use that phrasing, not 'no-signal'.
```


### 3.V Additional verifier findings (no 1:1 extracted fact)

**3.V1 — ❌ WRONG** · GAPS-LIST AUDIT: is the claim 'The exact CLI arguments used to launch the IMU-1 2-TPP run are not on disk' true?

**Checked against**

```
gaps[5]: 'The exact CLI arguments ... are not on disk — qwen3_imu1_2tpp_train.log echoes only the bundle flags ... The LR / z-weight / weight-decay values I report are train_imu1.py:96-107 DEFAULTS, unverified against the actual invocation.'
```

**Corrected value**

```
The invocation IS on disk at Qwen3-0.6B/builds/phase_b_driver.sh:24 (with S=18150, W=900, COMMON="--eval_every 2000 --ckpt_every 2000 --log_every 50" defined at phase_b_driver.sh:14). It is only the log that omits them. The driver also proves --warmup_steps 900 was passed, refuting the 'warmup=50 default' claim.
```

**Verifier note**

```
phase_b_driver.sh is the Phase-B sequencer for all four 2-TPP runs; line 19-20 is the faithful baseline invocation (whose flags exactly reproduce the args dict at qwen3_baseline2tpp_train.log:2, independently validating the driver as the real launch record), line 24 is the IMU-1 one, lines 28-34 the two partial-RoPE arms. This is the strongest single correction in this audit: a reviewer checking the IMU-1 recipe would find the driver in one grep, and a card asserting 'warmup 50' would be demonstrably wrong against the run's own logged LR ramp.
```


**3.V2 — ⚠️ NEEDS QUALIFIER** · MISSING CAVEAT: is there a later run that undercuts the 23.52-vs-28.65 (-17.9%) story, which no fact mentions?

**Checked against**

```
(not stated anywhere in the fact list)
```

**Corrected value**

```
YES. Run 2026-07-05_qwen3-0.6b_scaling-persistence has ledger verdict `null`: NorMuon's advantage over AdamW CONVERGES with token budget. Any card headline resting on the IMU-1 (NorMuon-bearing) 23.52 must carry this.
```

**Verifier note**

```
Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/verdict.json: question = 'Does NorMuon's +0.474 BPB win over AdamW (2D weights, fixed N=596M) persist or converge with token budget?'; wikitext2_val gap_bpb 0.47433 @42M -> 0.12591 @168M -> 0.07169 @420M, trend verdict 'CONVERGES', rationale 'gap shrinks toward 0 with scale and falls within the noise floor at the largest budget — an early-training speedup that converges away (no advantage at scale)'. n_seeds [3,3] at every rung. research/ledger/ledger.json records run_id 2026-07-05_qwen3-0.6b_scaling-persistence with status 'done', verdict 'null'. Note the code_py corpus plateaus rather than converges (0.50156 -> 0.17578 -> 0.17709), so 'converges' is corpus-dependent — do not overstate in either direction. Meanwhile Qwen3-0.6B/README.md:52 still advertises 'NorMuon > AdamW | wikitext -0.474 bpb | significant win', which is the 42M rung only. A reviewer would consider the omission of this later null material.
```


### 3.G Gaps — not determinable from disk

- Environment versions (torch / transformers / datasets) at the time of the 2026-06-09 13.40 measurement are not recorded in any file I could find. The only hint is a deprecation banner in results/original_eval_run2.log:1 ('`torch_dtype` is deprecated! Use `dtype` instead!'), which gives no version.
- The HuggingFace revision (commit sha) of Qwen/Qwen3-0.6B-Base actually downloaded for the 13.40 run is NOT recorded — eval_original_vs_repro.py:49 calls from_pretrained(REPO) with no revision argument, and no lockfile/manifest exists in the build folder.
- The HuggingFaceFW/fineweb-edu dataset revision behind tokcache_133072000_300000.pt and tokcache_1191478400_300000.pt is NOT recorded — train_qwen3.py:151 passes no revision. (A pinned sha 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 appears in research/eval/private_heldout_v1/private_prose_v1.txt:455 but that is the LATER dataset-forge prep, not these caches.)
- No results .json exists for any of the four numbers. 13.40 and 46.31 live only in results/original_vs_repro.txt (+ .log); 28.65 in qwen3_baseline2tpp_after.txt / _train.log; 23.52 only in qwen3_imu1_2tpp_train.log. There is no suite_version stamp on any of them.
- Whether the released Qwen3-0.6B-Base saw these exact FineWeb-Edu val documents during its 36T-token pretraining is undeterminable from disk; the training_plan.md summary of the tech report describes a web-heavy corpus but no overlap test was or could be run here.
- The exact CLI arguments used to launch the IMU-1 2-TPP run are not on disk — qwen3_imu1_2tpp_train.log echoes only the bundle flags, param split and tok/step, not the full argparse namespace (unlike the faithful runs). The LR / z-weight / weight-decay values I report are train_imu1.py:96-107 DEFAULTS, unverified against the actual invocation.
- No re-measurement of the released Qwen3-0.6B-Base exists on the 1191478400 val slice, so the true same-slice gap between 28.65/23.52 and the released model is unknown; it can only be obtained by re-running eval_original_vs_repro.py against that cache.

---

## 4. −0.474 bpb — NorMuon vs AdamW

<sub>Audit dimension: the -0.474 bpb figure (NorMuon vs AdamW)</sub>

### 4.1 Which wikitext is the -0.474 bpb measured on: wikitext-2-raw-v1 or wikitext-103-raw-v1?

**Value**

```
wikitext-2-raw-v1 (HF dataset `Salesforce/wikitext`, config `wikitext-2-raw-v1`), pinned revision b08601e04326c79dfdd32d625aee71d232d685c3. NOT wikitext-103.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/score_cohort.py:54`

**Source quote**

```
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation",
                      revision=WIKITEXT_REV)   [WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3", score_cohort.py:25]
```

**Confidence** — measured from code

**Caveat** — Same corpus definition is the versioned suite standard: .claude/skills/eval-harness/references/suite.md:133 `| wikitext2_val | Salesforce/wikitext | wikitext-2-raw-v1 / validation | b08601e04326c79dfdd32d625aee71d232d685c3 |`. Corpus text is assembled as "\n\n".join of non-empty ex["text"] (score_cohort.py:56), i.e. one concatenated stream, not per-document.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Opened Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/score_cohort.py. Line 54 reads exactly: `wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation",` and line 55 `revision=WIKITEXT_REV)`. WIKITEXT_REV is defined at score_cohort.py:25 as `b08601e04326c79dfdd32d625aee71d232d685c3`. The '-raw-' variant IS what the script names — no substitution. Line 56 confirms the text-assembly claim: `wt_text = "\n\n".join(e["text"] for e in wt if e["text"].strip())`. Cross-check .claude/skills/eval-harness/references/suite.md:133 matches the quote verbatim, including the same pinned sha. Independent third corroboration at RESULT.md:72 ('`Salesforce/wikitext` wikitext-2-raw-v1 rev `b08601e04326...`'). No occurrence of wikitext-103 anywhere in the scorer.
```


### 4.2 Which split?

**Value**

```
validation (`split="validation"`). The scored token stream is 204,600 tokens / 869,710 bytes.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/score_cohort.py:54`

**Source quote**

```
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation",
```

**Confidence** — measured from code

**Caveat** — n_tokens/n_bytes per cell confirmed identical across all 6 cells in Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/cohort_bpb.json:8-9 ("n_tokens": 204600, "n_bytes": 869710).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
score_cohort.py:54 literally contains `split="validation"`. Verified n_tokens/n_bytes are identical across all six cells by reading the whole file, not just the cited lines: cohort_bpb.json lines 8-9 (adamw_seed0), 22-23, 36-37, 50-51, 64-65, 78-79 all read `"n_tokens": 204600, "n_bytes": 869710`. MATERIAL CLARIFICATION a reviewer will want: 204,600 is NOT the size of the wikitext-2 validation split. It is 200 windows x 1023 label tokens under the MAX_WINDOWS=200 cap (score_cohort.py:24, 37-38), and because STRIDE(512) < SEQ(1024) those 204,600 token-scorings come from only 102,911 DISTINCT label positions (windows start at b = 0,512,...,101888; union of label spans = [1, 102912)). So the eval scores the first ~103k tokens of the split, each counted ~twice — it does not score the whole split.
```


### 4.3 Is it bits-per-byte (bpb) or bits-per-token? How is bpb computed (which byte count / normalization)?

**Value**

```
True bits-per-byte. bpb = (sum of per-token NLL in nats over the eval windows / ln2) / (UTF-8 byte count of the decoded LABEL span of those same windows). Denominator for wikitext-2 = 869,710 bytes; for code = 843,643 bytes. Per-token PPL is reported separately, never as the headline.
```

**Evidence** — `research/eval_metrics.py:33`

**Source quote**

```
def bits_per_byte(total_nll_nats, total_bytes):
    """Bits-per-byte = (Σ NLL in nats / ln2) / (raw UTF-8 byte count).
...
    return (float(total_nll_nats) / _LN2) / total_bytes
```

**Confidence** — measured from code

**Caveat** — The byte count is accumulated per window as `nbytes += len(tok.decode(labels[0].tolist()).encode("utf-8"))` (score_cohort.py:46) and NLL as `F.cross_entropy(..., reduction="sum")` (score_cohort.py:43-44). Because STRIDE(512) < SEQ(1024) the windows overlap and roughly half the corpus is counted twice — but numerator and denominator are double-counted identically, which eval_metrics.py:39-41 explicitly requires ("under overlapping eval windows, the byte denominator must match the double-counted token span"). Logits are cast .float() before CE (score_cohort.py:41) though the model runs bf16 (score_cohort.py:30).

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Value body is correct. The CAVEAT contains a numeric error: 'roughly half the corpus is counted twice' is wrong. Correct statement: essentially ALL of the scored span is counted twice — 204,600 scored token-positions come from 102,911 distinct positions (ratio 1.988), i.e. ~98.8% of the span is counted exactly twice, only ~1,222 positions once. Additionally, MAX_WINDOWS=200 caps the scan, so the metric covers only the first 102,911 label positions of the tokenized corpus, not the whole validation split.
```

**Verifier note**

```
The formula IS confirmed: research/eval_metrics.py:33-45 defines `def bits_per_byte(total_nll_nats, total_bytes)` returning `(float(total_nll_nats) / _LN2) / total_bytes`, with `_LN2 = math.log(2.0)` at line 28. Docstring lines 39-41 do say the byte denominator 'must match the double-counted token span'. score_cohort.py:46 `nbytes += len(tok.decode(labels[0].tolist()).encode("utf-8"))` and :43-44 `F.cross_entropy(..., reduction="sum")` confirmed. `.float()` at :41 and `DTYPE = torch.bfloat16` at :30 confirmed. Byte denominators 869,710 (wikitext) / 843,643 (code) confirmed in cohort_bpb.json. eval_metrics.py:48-50 confirms PPL is per-token and 'NOT comparable across tokenizers — report alongside bits_per_byte, never as the sole cross-run headline'. Only the double-counting arithmetic in the caveat is wrong, and it is wrong in a way that understates how truncated/overlapped the eval window set is.
```


### 4.4 Sequence length, stride, tokenizer used for the eval

**Value**

```
Eval window SEQ = 1024, STRIDE = 512, MAX_WINDOWS = 200 (so 200 windows × 1023 label tokens = 204,600 tokens scored). Tokenizer = HF `Qwen/Qwen3-0.6B-Base` AutoTokenizer (the model's own tokenizer, vocab 151,936). Note this is the EVAL seq len; TRAINING used seq_len 4096.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/score_cohort.py:24`

**Source quote**

```
SEQ, STRIDE, MAX_WINDOWS = 1024, 512, 200            # text-lm-v2 constants
...
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")   [score_cohort.py:72]
...
SEQ_LEN, MICRO_BATCH, GRAD_ACCUM = 4096, 4, 4   [train_ablation.py:51]
```

**Confidence** — measured from code

**Caveat** — suite.md:115-116 pins the same constants for text-lm-v2 (`window SEQ | 1024`, `STRIDE | 512`) and says "never lift SEQ without a version bump". The tokenizer is loaded from the HF hub at score time (network dependency), not from a local pinned copy.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
score_cohort.py:24 reads exactly `SEQ, STRIDE, MAX_WINDOWS = 1024, 512, 200            # text-lm-v2 constants`. score_cohort.py:72 reads `tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")`. train_ablation.py:51 reads `SEQ_LEN, MICRO_BATCH, GRAD_ACCUM = 4096, 4, 4`. Vocab 151,936 verified independently at Qwen3-0.6B/model.py:37 `vocab_size: int = 151_936  # config.json: vocab_size`. suite.md:115-117 pins `window SEQ | 1024`, `STRIDE | 512`, `MAX_WINDOWS (main metric) | 200`; suite.md:123 contains 'never lift `SEQ` without a version bump'. 200 x 1023 = 204,600 matches cohort_bpb.json's n_tokens exactly. The network-dependency caveat (tokenizer pulled from the hub at score time, no local pin) is correct as read.
```


### 4.5 Which two arms, at what model size, what token budget, how many seeds?

**Value**

```
Arms: AdamW @ peak_lr 2.4e-3 vs NorMuon @ lr 0.011, applied ONLY to the 196 2D non-embedding weight matrices; the 114 embedding/1D params are AdamW@2.4e-3 wd=0 in BOTH arms; 2D weight_decay=0.1 in both. Model: full faithful Qwen3-0.6B, 596,049,920 total params / 440,467,456 non-embedding. Budget: 640 steps × 65,536 tok/step = 41,943,040 tokens per cell ("42M"), iso-FLOP. Seeds: 3 per arm (6 cells total), seed varies init + DataLoader shuffle only.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/train_ablation.py:82`

**Source quote**

```
ap.add_argument("--optimizer", choices=["adamw", "normuon"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=600)            # ~39M tokens
    ap.add_argument("--peak_lr", type=float, default=2.4e-3)     # faithful-tuned AdamW LR
    ap.add_argument("--normuon_lr", type=float, default=0.011)   # IMU-1-tuned NorMuon 2D LR
    ap.add_argument("--weight_decay", type=float, default=0.1)   # on 2D, BOTH arms (held equal)
[train_ablation.py:53]  TOK_PER_STEP = SEQ_LEN * MICRO_BATCH * GRAD_ACCUM   # 65,536
```

**Confidence** — measured from code

**Caveat** — Budget arithmetic + arm split also stated in RESULT.md:68-70 and recorded in research/ledger/ledger.json:466-469 ("tokens_per_cell": 41943040, "cells": 6, "source": "6-cell cohort (2 arms x 3 seeds), 640 steps x 65536 tok"). n_params_nonembed 440467456 is from ledger.json metrics.systems (line ~489 block); total 596,049,920 from Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/architecture_plan.md:60. IMPORTANT confound disclosed by the repo itself (RESULT.md:47): all 6 cells share a FIXED data split (SPLIT_SEED=0), so the ±0.006–0.008 SEM under-estimates true end-to-end seed variance.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
All stated values confirmed, but a MATERIAL caveat is missing and one line pointer is wrong. MISSING CAVEAT (RESULT.md:43, Limitation #2): 'Both arms use 2D wd=0.1 (NorMuon's/IMU-1's tuned value). AdamW's faithful recipe was tuned at wd=0.01. So AdamW runs at a 10x wd it was never tuned for, on a budget it was never tuned for — part of the gap may be baseline handicap rather than genuine optimizer advantage.' Stating 'weight_decay=0.1 in both' without this reads as if the control were neutral. LINE FIX: n_params_nonembed 440467456 is at research/ledger/ledger.json:508, not '~489'.
```

**Verifier note**

```
Verified: 196/114 split is MEASURED, printed by the trainer itself — results/adamw_seed0.log:11 `param split: 196 2D->adamw | 114 rest->AdamW` and results/normuon_seed0.log:11 `param split: 196 2D->normuon | 114 rest->AdamW`. 640 steps is MEASURED not defaulted — adamw_seed0.log:8 `start: optimizer=adamw seed=0 steps=640 tok_budget=41,943,040 peak_lr=0.0024 normuon_lr=0.011 wd=0.1` (the script DEFAULT at train_ablation.py:84 is 600; run_arms.sh passes $STEPS). train_ablation.py:82-87 quoted correctly. train_ablation.py:69 confirms the split rule `(twod if (p.dim() == 2 and "embed_tokens" not in name) else rest)`; :70-77 confirm rest->AdamW wd=0.0 in both arms. SPLIT_SEED=0 at :52. 596,049,920 is a real runtime-printed count for this Qwen3Config (e.g. Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/results/sft_seed0.log:7 `params=596,049,920`) and matches architecture_plan.md:60. Budget arithmetic corroborated at RESULT.md:68 and ledger.json:466-469. Iso-FLOP is asserted-by-construction, not measured: verifier_report.json:31 states 'a numeric metrics.train_flops field is not stored in any results JSON; iso-FLOP holds by construction + the recorded confound_check flag.'
```


### 4.6 What is the sign convention — is -0.474 NorMuon better?

**Value**

```
Yes. The on-disk JSON stores a POSITIVE +0.47432550192416323 as `improvement_bpb` = adamw_mean(2.1098) − normuon_mean(1.6355), i.e. NorMuon's bpb is 0.474 LOWER (better; lower bpb is better). The "−0.474" form used in Qwen3-0.6B/README.md is the same number expressed as NorMuon's signed delta relative to AdamW. Both mean NorMuon better by 0.474 bpb.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/score_ladder.py:85`

**Source quote**

```
# claim. Sign convention: gap_bpb = adamw_mean - normuon_mean (eval_stats.seed_delta_significant,
# direction="lower_is_better"), so gap > 0 == NorMuon better.
```

**Confidence** — measured from code

**Caveat** — Cross-checked against research/eval_stats.py:138 `improvement = (b_mean - t_mean) if direction == "lower_is_better" else ...` and against Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/verdict.json:19 `"improvement_bpb": 0.47432550192416323`. The literal string "-0.474" appears on disk only at Qwen3-0.6B/README.md:52 and :243.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
score_ladder.py:85-86 reads exactly: '# claim. Sign convention: gap_bpb = adamw_mean - normuon_mean (eval_stats.seed_delta_significant,' / '# direction="lower_is_better"), so gap > 0 == NorMuon better.' Cross-checked research/eval_stats.py:138 `improvement = (b_mean - t_mean) if direction == "lower_is_better" else (t_mean - b_mean)`, and :115 'the improvement delta (positive == better in the given direction)'. results/verdict.json:19 `"improvement_bpb": 0.47432550192416323`; means at :10 (2.1098171365956357) and :17 (1.6354916346714725) — 2.1098171 - 1.6354916 = 0.4743255, arithmetic checks. The sub-claim about the literal minus form holds: a repo-wide grep for `[-−–]0.474` returns hits ONLY at Qwen3-0.6B/README.md:52 and :243.
```


### 4.7 What exactly is the 42M-token headline number and its CI?

**Value**

```
wikitext-2 val BPB: AdamW mean 2.1098171365956357 (seeds 2.104968, 2.102416, 2.122067), NorMuon mean 1.6354916346714725 (seeds 1.649904, 1.624842, 1.631729); improvement +0.47432550192416323 bpb, 95% CI [0.4434844613250229, 0.5051665425233036], Welch-t df 3.861, significant=true, n=[3,3], suite_version text-lm-v2, verdict "win".
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/verdict.json:19`

**Source quote**

```
"improvement_bpb": 0.47432550192416323,
      "ci95": [
        0.4434844613250229,
        0.5051665425233036
      ],
      "significant": true,
      "df": 3.8610499345338067,
      "n": [3, 3]
...
  "headline_corpus": "wikitext2_val",
  "verdict": "win"
```

**Confidence** — results JSON

**Caveat** — This is a number this repo MEASURED itself (raw per-seed BPB in results/cohort_bpb.json, produced by score_cohort.py from the 6 on-disk checkpoints). Independently re-derived in results/verifier_report.json:16,36 ("reproduces the recorded wikitext-2 result bit-for-bit"). Not copied from any paper.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Every digit verified against Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/verdict.json: suite_version :2, adamw_bpb :5-9, adamw_mean :10, normuon_bpb :12-16, normuon_mean :17, improvement_bpb :19, ci95 :20-23, significant :24, df :26 (3.8610499345338067), n :27-30, headline_corpus :61, verdict :62. Per-seed values independently traced to results/cohort_bpb.json:7, 21, 35 (adamw) and :49, 63, 77 (normuon). Re-derivation confirmed at results/verifier_report.json:16, 21, 26 (ci_match true) and :36 ('reproduces the recorded wikitext-2 result bit-for-bit'). This is a repo-MEASURED number (produced by score_cohort.py from six on-disk checkpoints), not copied from a paper. THREE small notes for a card author: (1) the quoted JSON snippet silently elides `"warning": null` (verdict.json:25) between `significant` and `df`; (2) the df 3.861 is the Welch-Satterthwaite df, but the CI was computed with df FLOORED to 3 / t_crit 3.182 (verifier_report.json:19-20) — conservative, but 'df 3.861' alone misdescribes the interval; (3) the test is UNPAIRED Welch on a design that is paired-by-seed — the ladder's own verdict.json:257 flags this ('a paired-t on per-seed diffs is the stricter test').
```


### 4.8 CURRENT status of the claim — did a later run null it?

**Value**

```
YES, NULLED AT SCALE. The scaling-persistence ladder (fixed N=596M, token budget swept 42M/168M/420M, n=3 seeds per arm at EVERY rung) records ledger_verdict "null" and trend CONVERGES on BOTH corpora. wikitext-2 gap: 0.47432550 (42M) -> 0.12590584 (168M) -> 0.07169398 (420M). The ledger run 2026-07-05_qwen3-0.6b_scaling-persistence carries "verdict": "null", status "done".
```

**Evidence** — `Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/verdict.json:194`

**Source quote**

```
"rationale": "gap shrinks toward 0 with scale and falls within the noise floor at the largest budget — an early-training speedup that converges away (no advantage at scale)",
[line 199]  "ledger_verdict": "null",
[line 6]  "question": "Does NorMuon's +0.474 BPB win over AdamW (2D weights, fixed N=596M) persist or converge with token budget?"
[lines 27/44/61]  "gap_bpb": 0.47432550192416323, ... "gap_bpb": 0.12590584068581911, ... "gap_bpb": 0.07169397744785555,
[research/ledger/ledger.json:1554,1566]  "run_id": "2026-07-05_qwen3-0.6b_scaling-persistence", ... "verdict": "null",
```

**Confidence** — results JSON

**Caveat** — THREE nuances a reviewer must not lose. (1) The null is a BUDGET-scaling null at FIXED model size N=596M — it is not evidence about larger N. (2) The 420M wikitext gap is still nominally SIGNIFICANT as measured (+0.0717, CI [0.0553, 0.0881], excludes 0, verdict.json:61-71); the "falls within the noise floor" phrase refers to the OLS-FITTED gap at the top rung (edge_at_top / gap_hi_fit = 0.029726712435672376 < gap_noise 0.03675972213287565, verdict.json:141-146), not to the measured rung. edge_resolved=false on wikitext, TRUE on code. (3) The `null` word is partly gate-driven: verdict.json:207-214 records the §C25 `scaling` HARD battery as INCOMPLETE (missing log_rmse_r2, holdout_extrapolation_pctdev, bootstrap_forecast_ci) so `win` was unreachable regardless — but verdict.json:203 shows significance_verdict was independently "null" from the CONVERGES trend mapping, with c17_cap_applied=false.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Read the full 258-line Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/verdict.json. Confirmed: question :6; wikitext gaps :27 (0.47432550192416323), :44 (0.12590584068581911), :61 (0.07169397744785555); n_seeds [3,3] at all three rungs (:35-38, :52-55, :69-72); trend_verdict CONVERGES :198; ledger_verdict 'null' :199; rationale :194 (and the identical text at :149). ledger.json:1554 run_id, :1565 status 'done', :1566 verdict 'null'. All three of the fact's nuances verified: (1) budget-only sweep at fixed N=596M — BUDGETS at score_ladder.py:42 sweep tokens only; (2) 420M wikitext IS nominally significant as measured — verdict.json:61-66 gap 0.0717, ci95 [0.05525251860105098, 0.08813543629466011], `"significant": true`; the 'noise floor' phrase refers to the FITTED gap_hi_fit/edge_at_top 0.029726712435672376 (:140, :142) vs gap_noise 0.03675972213287565 (:146); edge_resolved false on wikitext (:143), true on code (:165); (3) §C25 incompleteness at :207-214 AND significance_verdict independently 'null' at :203 with c17_cap_applied false at :206. ADDITIONAL PROVENANCE CAVEAT I found: this verdict.json is NOT tracked by git (`git ls-files --error-unmatch` errors on it), nor is the ladder's per-seed ladder_bpb.json — the null's evidence is working-tree-only, while the 42M 'win' evidence (cohort_bpb.json, verdict.json) IS committed.
```


### 4.9 What are the companion numbers on the second (code) corpus?

**Value**

```
Corpus = codeparrot/codeparrot-clean-valid, split `train` (streaming), pinned rev 4db92d2ec0c1b4c41eeb439cfae16854511d9dcd, first 500,000 chars, 204,600 tokens / 843,643 bytes, same SEQ 1024 / STRIDE 512. 42M: AdamW 3.3846985755955523 vs NorMuon 2.8831399238053073, gap +0.5015586517902451, CI [0.4559911731303807, 0.5471261304501094], significant. 168M: gap +0.1757807425441804, CI [0.1369433296166722, 0.21461815547168864]. 420M: gap +0.17708989020863175, CI [0.13074040899389106, 0.22343937142337245]. Trend verdict CONVERGES (slope -0.34197431351062624 over log10 tokens, r2 0.8412726939487719) but edge_resolved=true (still above noise at the top rung).
```

**Evidence** — `Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/verdict.json:80`

**Source quote**

```
"gap_bpb": 0.5015586517902451,   [42M]
        "gap_bpb": 0.1757807425441804,   [168M, line 97]
        "gap_bpb": 0.17708989020863175,  [420M, line 114]
[line 171] "rationale": "gap shrinks toward 0 with scale; still above noise at the largest measured budget but trending out — the edge is eroding, extend the ladder before claiming it"
[score_cohort.py:57-58] cp = load_dataset("codeparrot/codeparrot-clean-valid", split="train",
                      streaming=True, revision=CODEPARROT_REV)
```

**Confidence** — results JSON

**Caveat** — The code corpus does NOT converge monotonically: 168M 0.17578 -> 420M 0.17709 is a slight INCREASE, i.e. a plateau over the last two rungs, and only the fitted slope is negative. MEMORY's "plateau, not convergence — scrutinize that label" is correct and traceable: the CONVERGES word for code_py comes from an OLS slope across 3 points dominated by the 42M rung, not from the last two rungs. Also note the code corpus is a raw 500k-char prefix of a streamed split, so it is a fixed but arbitrary slice.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
score_cohort.py:57-58 reads exactly `cp = load_dataset("codeparrot/codeparrot-clean-valid", split="train",` / `streaming=True, revision=CODEPARROT_REV)`; CODEPARROT_REV = '4db92d2ec0c1b4c41eeb439cfae16854511d9dcd' at :26; CODE_CHARS = 500_000 at :27. Ladder verdict.json code block verified line-by-line: 42M gap :80, ci95 :81-84; 168M gap :97, ci95 :98-101; 420M gap :114, ci95 :115-118; all `"significant": true` (:85, :102, :119). trend_by_corpus.code_py: verdict CONVERGES :154, slope :156, r2 :158, edge_resolved true :165, gap_noise 0.046349481214740695 :168, rationale :171. 843,643 bytes / 204,600 tokens confirmed in cohort_bpb.json:14-15. The MEMORY-flagged 'plateau not convergence' point is CORRECT and I reproduced it: 0.1757807 (168M) -> 0.1770899 (420M) is an INCREASE of +0.0013, and the two CIs overlap almost entirely — only the 3-point OLS slope, dominated by the 42M rung, is negative. MINOR IMPRECISION: 'first 500,000 chars' overstates exactness — score_cohort.py:59-63 appends WHOLE documents until cumulative `len(content)+2` EXCEEDS 500,000, so the slice is >=500,000 chars; suite.md:134 phrases it correctly as 'until > 500,000 chars'.
```


### 4.10 Does the model card / README currently present -0.474 as a standing win despite the null?

**Value**

```
YES — the de-facto model card does. Qwen3-0.6B/README.md:52 and :243 both present it as a "significant win" with NO mention of the scaling ladder, CONVERGES, or the null verdict. grep of Qwen3-0.6B/README.md for scaling-persistence / converge / CONVERGES / 0.126 / 0.072 / persist returns ZERO hits. The ledger entry for the 42M run also still reads verdict "win" (research/ledger/ledger.json:482), and technique `normuon-optimizer`'s run_ids list does not include the ladder run (which carries technique_slug: null, ledger.json:1557).
```

**Evidence** — `Qwen3-0.6B/README.md:52`

**Source quote**

```
| **NorMuon > AdamW** | wikitext −0.474 bpb [0.444, 0.505] · code −0.502 [0.456, 0.547] | **significant win** |
[Qwen3-0.6B/README.md:243] | **Optimizer ablation (clean, single-variable)** | NorMuon **beats** AdamW: wikitext **−0.474 bpb** (95% CI [0.444, 0.505]), code −0.502 bpb ([0.456, 0.547]) — **significant win** | ledger `2026-06-16_qwen3_normuon-vs-adamw` |
```

**Confidence** — measured from code

**Caveat** — To be fair to the source run: RESULT.md:7 DOES scope it correctly ("This is an early-training optimization-speed signal at one architecture and one budget; we do NOT claim it holds at scale") and RESULT.md:45 predicts the fade. The failure is that Qwen3-0.6B/README.md dropped those qualifiers. Any model card must carry the ladder null.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
The YES answer and every cited line are correct, but the exposure is UNDERCOUNTED. There are at least FOUR un-caveated standing-win presentations, not two: Qwen3-0.6B/README.md:52, :175 ('NorMuon beats AdamW by **+0.474 bpb on wikitext-2 (95% CI [0.444, 0.505])** and +0.502 on code — significant.'), :243, and Qwen3-0.6B/PLOTS_INDEX.md:73 ('+0.474 bpb, significant'). Also add: the 42M run's own ledger caveats field, research/ledger/ledger.json:512, still asserts 'no scaling curve' — a statement that became FALSE when the ladder completed.
```

**Verifier note**

```
Verified by running the exact greps. `grep -in 'scaling-persistence|converge|persist|0\.126|0\.072' Qwen3-0.6B/README.md` returns ZERO hits for every term — confirmed. README.md:52 and :243 quoted verbatim and match. ledger.json:482 `"verdict": "win"` confirmed; I read the entire 42M entry (ledger.json:461-535) and found NO pointer to the ladder or the null. normuon-optimizer run_ids at ledger.json:153-154 are exactly ['2026-06-16_qwen3_normuon-vs-adamw', '2026-07-23_qwen3-0.6b_normuon-at-scale'] — the ladder run is absent; ledger.json:1557 `"technique_slug": null` confirmed. Staleness corroborated by mtime: Qwen3-0.6B/README.md was last modified 2026-07-06, i.e. BEFORE the ladder's first completion (2026-07-12) and long before the 2026-07-28 re-score. SEPARATE ROUNDING DEFECT worth flagging to a card author: README.md:52/:175/:243 all print the CI lower bound as 0.444, but the on-disk value is 0.4434844613250229, which rounds to 0.443 at 3dp (root README.md:112 gets it right).
```


### 4.11 Is the root README's account of the ladder current?

**Value**

```
NO — root README.md:105-121 is STALE relative to verdict.json (which was re-scored 2026-07-28 after the 3rd 420M seed landed). README says "420M ×2 seeds", "+0.073 [−0.038, +0.184] at 420M ... not significant at the top", "code_py: +0.502 → +0.176 → +0.192", code slope "−0.328 (r² 0.81)", and "Verdict: directional, not a headline — the 420M rung is n=2". Current verdict.json: n=3/arm at 420M, wikitext gap +0.0717 CI [0.0553, 0.0881] SIGNIFICANT, code 420M +0.1771, code slope −0.34197 r² 0.84127, headline_capped_by_c17_power false, ledger_verdict "null".
```

**Evidence** — `README.md:112`

**Source quote**

```
(AdamW − NorMuon, BPB): **+0.474** [+0.443, +0.505] at 42M → **+0.126**
  [+0.089, +0.163] at 168M → **+0.073** [−0.038, +0.184] at 420M — significant at
  the two smaller budgets, **not significant** at the top. code_py: +0.502 → +0.176
  → +0.192. OLS over log10(tokens) gives slope **−0.416** (r² 0.92) on wikitext and
  **−0.328** (r² 0.81) on code → **CONVERGES** on both.
[README.md:117] - **Verdict: directional, not a headline** — the 420M rung is n=2 (< 3 seeds, §C17).
```

**Confidence** — measured from code

**Caveat** — The 3rd 420M seed pair was trained by run 2026-07-23_qwen3-0.6b_normuon-at-scale (checkpoints persist_420M_{adamw,normuon}_s2.pt dated 2026-07-25/26) and re-scored 2026-07-28. research/ledger/runs/2026-07-23_qwen3-0.6b_normuon-at-scale.md:355-360 documents the n=2 -> n=3 upgrade and the current numbers. The direction of the staleness matters: the top-rung gap became MORE statistically resolved (significant), while the overall verdict stayed `null` via the CONVERGES trend + §C25 gate.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Read /home/yashb98/Downloads/BuildFromScratch/README.md:105-125. Every stale value quoted is present verbatim: ':109 + **420M ×2 seeds**, ten cells'; ':112-116' the full '+0.474 [+0.443, +0.505] at 42M → +0.126 [+0.089, +0.163] at 168M → +0.073 [−0.038, +0.184] at 420M — significant at the two smaller budgets, **not significant** at the top. code_py: +0.502 → +0.176 → +0.192. OLS ... slope −0.416 (r² 0.92) on wikitext and −0.328 (r² 0.81) on code'; ':117 - **Verdict: directional, not a headline** — the 420M rung is n=2 (< 3 seeds, §C17).' Current values all confirmed in verdict.json (:13-16 top_budget_seeds [3,3]; :17 headline_capped_by_c17_power false; :19 cap_reason 'top rung has >=3 seeds'; :61-66; :114; :156; :158; :199). The upgrade provenance is confirmed at research/ledger/runs/2026-07-23_qwen3-0.6b_normuon-at-scale.md:356-360, which states the rung moved from n_seeds [2,2] with warning 'CI is wide/unreliable' to [3,3] warning null, and quotes the current wikitext gap 0.0717 CI95 [0.0553, 0.0881] and code 0.1771 [0.1307, 0.2234]. Independently corroborated by mtimes: root README.md 2026-07-23 18:05, verdict.json 2026-07-28 19:17. The fact's own framing of the staleness direction (top rung became MORE resolved, verdict stayed null) is accurate.
```


### 4.12 Was the ladder scored with the SAME eval pipeline as the 42M headline (i.e. are the numbers comparable)?

**Value**

```
Yes. score_ladder.py imports the IMU-1 scorer directly and reuses its score()/load_corpora() — identical SEQ/STRIDE/corpora/bpb — and the 42M rung is not re-run but copied verbatim from cohort_bpb.json. Same suite_version text-lm-v2 stamped on both.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/score_ladder.py:40`

**Source quote**

```
import score_cohort as sc  # noqa: E402  (reuse score() + load_corpora(): text-lm-v2 SEQ/STRIDE/bpb)
[line 454]     corpora = sc.load_corpora(tok)
[line 238]     return {c: sc.score(model, ids, tok) for c, ids in corpora.items()}
[line 505]             r = seed_delta_significant(a, n, direction="lower_is_better")
```

**Confidence** — measured from code

**Caveat** — Per-seed ladder BPBs are durably recorded in Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/ladder_bpb.json (18 cells, run_id stamped as the LADDER's, scored 2026-07-28) — note the file lives in the 42M experiment's results/ dir by a documented location decision (score_ladder.py:53-60), which is easy to misread as belonging to the 42M run.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Pipeline-identity claim is fully CONFIRMED. The caveat's word 'durably recorded' needs qualifying: Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/ladder_bpb.json is NOT tracked by git (`git ls-files --error-unmatch` errors on it), whereas the 42M rung's cohort_bpb.json and verdict.json ARE tracked. Given this repo's own recorded incident of a branch switch destroying untracked evidence, the null's per-seed evidence exists on the working tree only. Minor: the location-decision comment spans score_ladder.py:53-62, not 53-60.
```

**Verifier note**

```
Verified every cited line. score_ladder.py:40 reads exactly `import score_cohort as sc  # noqa: E402  (reuse score() + load_corpora(): text-lm-v2 SEQ/STRIDE/bpb)`; :454 `corpora = sc.load_corpora(tok)`; :238 `return {c: sc.score(model, ids, tok) for c, ids in corpora.items()}`; :505 `r = seed_delta_significant(a, n, direction="lower_is_better")`. :455 confirms the same model class/dtype (`sc.DEVICE, sc.DTYPE`). :456 `reuse = json.loads((RES / "cohort_bpb.json").read_text())["cells"]   # 42M rung` confirms the 42M rung is copied, not re-run. I loaded ladder_bpb.json directly: 18 cells, `run_id` = '2026-07-05_qwen3-0.6b_scaling-persistence', `scored` = '2026-07-28', and every 42M cell carries `"source": "reused:cohort_bpb.json"` while the 420M s2 cells carry `source: scored`. suite_version 'text-lm-v2' stamped in both ladder_bpb.json and cohort_bpb.json:2. verdict.json:21 `per_seed_bpb_file` points back at the file, so the cross-reference the caveat describes is real. Also note score_ladder.py itself is uncommitted-modified (git status ` M`), though its mtime 19:09 precedes verdict.json's 19:17, so verdict.json was produced by the current scorer.
```


### 4.G Gaps — not determinable from disk

- No 840M rung exists. c5_evidence_scale_ext.json declared a 840M (n=1) point but it was descoped before launch; SEEDS in score_ladder.py:49 still declares 840_000_000, and no checkpoint_persist_840M_*.pt is on disk. So the trend fit rests on exactly 3 budgets (42M/168M/420M), the minimum for a non-descriptive fit.
- No per-horizon LR re-tuning exists anywhere on disk. Both AdamW 2.4e-3 and NorMuon 0.011 were tuned at the 42M horizon and held fixed at 168M/420M (verdict.json:257 'Inherited confound: AdamW/NorMuon LRs tuned at 42M, not re-tuned per horizon'). Part of the observed fade could therefore be an LR artifact; nothing on disk separates the two.
- The three §C25 HARD scaling-battery items (log_rmse_r2, holdout_extrapolation_pctdev, bootstrap_forecast_ci) were never computed — no file on disk contains them (verdict.json:210-214, 236-241). A §C26 figure for the ladder is also missing ('c25_report_missing': ['figure']).
- No ledger detail doc exists for the ladder itself: research/ledger/runs/ contains 2026-06-16_qwen3_normuon-vs-adamw.md and 2026-07-23_qwen3-0.6b_normuon-at-scale.md, but NO 2026-07-05_qwen3-0.6b_scaling-persistence.md. The ladder's narrative lives only in verdict.json and inside the 2026-07-23 child doc.
- The 42M run's ledger entry has never been amended to reference the ladder: research/ledger/ledger.json:482 still reads "verdict": "win" with no pointer to the null, and the technique entry 'normuon-optimizer' run_ids = [2026-06-16_qwen3_normuon-vs-adamw, 2026-07-23_qwen3-0.6b_normuon-at-scale] omits the ladder run (whose technique_slug is null, ledger.json:1557). A ledger query by technique will not surface the null directly.
- Nothing on disk measures whether the convergence holds at model sizes other than N=596M — the ladder sweeps token budget only, at one fixed N. Any 'NorMuon does not help' generalization beyond 596M / 420M tokens is unsupported by this repo.
- research/ledger/ledger.json is currently uncommitted-modified (git status), so the ledger values quoted here are the working-tree state, not a committed state.

---

## 5. Reproduce — commands, versions, parity, determinism

<sub>Audit dimension: A real "Reproduce" section for a HuggingFace model card — exact commands, parity-check semantics (device/dtype/determinism/tolerance/inputs), recorded software versions, and commit/provenance stamping, for Qwen3-0.6B and SmolLM2-134(base).</sub>

### 5.1 Qwen3-0.6B: exact command to run the architecture-parity / bit-exactness check (script form)

**Value**

```
cd /home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B && python3 verify.py   — prints max |Δlogits| and asserts < 1e-3 + argmax agreement. Writes NO file.
```

**Evidence** — `Qwen3-0.6B/verify.py:11`

**Source quote**

```
python verify.py
```

**Confidence** — measured from code

**Caveat** — Corroborated by Qwen3-0.6B/README.md:498 `python verify.py        # parity gate — runs on CPU, no GPU needed` and root README.md:182 `cd Qwen3-0.6B && python verify.py`. This form is stdout-only — it produces no artifact on disk. There is no committed stdout log of Qwen3-0.6B/verify.py anywhere in the repo (searched for *.log under Qwen3-0.6B/); the only on-disk Qwen3 parity artifact comes from verify_run.py (next fact).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Qwen3-0.6B/verify.py:11 is exactly `    python verify.py`. I read the whole file (89 lines): it contains no write/open/json.dump call, so 'writes NO file' holds. Asserts confirmed at verify.py:74 (`assert max_abs < 1e-3`) and :81 (`assert hf_next == our_next`). The `cd` is load-bearing: verify.py:16 does `from model import Qwen3ForCausalLM, Qwen3Config` with no sys.path manipulation, so it only runs from Qwen3-0.6B/. Corroborations verified by line: Qwen3-0.6B/README.md:498 `python verify.py        # parity gate — runs on CPU, no GPU needed`; README.md:182 `cd Qwen3-0.6B && python verify.py`. The 'no committed stdout log' claim is confirmed independently: `grep -rl "Architecture parity verified" Qwen3-0.6B/` returns only Qwen3-0.6B/verify.py, and `grep -rl "max |Δlogits|" Qwen3-0.6B/` returns only verify.py, README.md, architecture_plan.md, verify_run.py — no log artifact.
```


### 5.2 Qwen3-0.6B: exact command that produces the machine-readable parity artifact (verify.json)

**Value**

```
cd /home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b && python3 verify_run.py   — writes results/verify.json, exit 0 on pass / 1 on fail
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/verify_run.py:9,31,93`

**Source quote**

```
line 9:     python verify_run.py
line 31: OUT_JSON = RESULTS_DIR / "verify.json"
line 93:     OUT_JSON.write_text(json.dumps(result, indent=2))
```

**Confidence** — measured from code

**Caveat** — This is the command the paper's reproducibility appendix lists first (research/papers/qwen3-imu1-matched-compute/sections/reproducibility.tex:15 `python verify_run.py`). It imports REPO + load_official_weights_into_ours from the parent Qwen3-0.6B/verify.py (verify_run.py:26), so it exercises the same code path as verify.py but additionally serializes the result.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
All three cited lines match verbatim: verify_run.py:9 `    python verify_run.py`, :31 `OUT_JSON = RESULTS_DIR / "verify.json"`, :93 `OUT_JSON.write_text(json.dumps(result, indent=2))`. Exit semantics verified: verify_run.py:99 `return 1` on fail, :102 `return 0` on pass, :106 `raise SystemExit(main())`. Import path confirmed at verify_run.py:26 `from verify import REPO, load_official_weights_into_ours`. Paper cite confirmed: research/papers/qwen3-imu1-matched-compute/sections/reproducibility.tex:15 `python verify_run.py` and it is the FIRST command in the \begin{verbatim} block (block starts :13).
```


### 5.3 Qwen3-0.6B: exact command to recompute the perplexity numbers

**Value**

```
cd /home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b && python3 eval_original_vs_repro.py   — writes results/original_vs_repro.txt
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/eval_original_vs_repro.py:22-23,75`

**Source quote**

```
line 22: CACHE = HERE / "results" / "tokcache_133072000_300000.pt"
line 23: OUT = HERE / "results" / "original_vs_repro.txt"
line 75:     OUT.write_text(report + "\n")
```

**Confidence** — measured from code

**Caveat** — REQUIRES GPU (line 41 `device = torch.device("cuda")` — hardcoded, no CPU fallback) and requires two large gitignored artifacts that DO exist on this box but are not in git: results/tokcache_133072000_300000.pt (1,066,978,101 B, dated Jun 8) and checkpoint_qwen3_lr17/lr24/lr30.pt (3,576,719,229 B each). A fresh clone CANNOT run this command. Qwen3-0.6B/README.md:489-490 states checkpoints and token caches are gitignored and must be regenerated with the training scripts.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Cited lines match verbatim: eval_original_vs_repro.py:22 CACHE, :23 OUT, :75 `OUT.write_text(report + "\n")`. GPU-hardcoded confirmed at :41 `device = torch.device("cuda")` (no is_available() fallback anywhere in the file). File sizes independently re-measured with `ls -la`: results/tokcache_133072000_300000.pt = 1066978101 bytes, mtime 2026-06-08 21:13; checkpoint_qwen3_lr17.pt / _lr24.pt / _lr30.pt = 3576719229 bytes each. Gitignore status independently confirmed: `git check-ignore -v` reports .gitignore:20 `*.pt` for both. README cite verified: Qwen3-0.6B/README.md:489-490 'Checkpoints (`*.pt`, ~3.5 GB each) and token caches are **gitignored** — regenerate / them with the training scripts.'
```


### 5.4 SmolLM2-134(base): exact commands to run the architecture-parity / bit-exactness check

**Value**

```
Script form: cd "/home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)" && python3 verify.py   |   Pytest form: cd "/home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)" && pytest tests/ -v
```

**Evidence** — `README.md:159-162`

**Source quote**

```
# Architecture parity gate (the non-negotiable test before training).
pytest tests/ -v
# or, the script form:
python verify.py
```

**Confidence** — measured from code

**Caveat** — Both forms verified present: SmolLM2-134(base)/verify.py:11 docstring `    python verify.py`; SmolLM2-134(base)/tests/test_parity.py:8 `    pytest tests/ -v`. The pytest form is STRICTLY STRONGER than verify.py — it adds param-count (134,515,008), tied-embedding storage-pointer, 512-token long-context, and per-layer (all 30 blocks) parity assertions (test_parity.py:49-132). The committed parity.log is the output of the SCRIPT form only.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Both commands are correct and both exist. But the pytest form is not unconditionally stronger: SmolLM2-134(base)/tests/test_parity.py:34-41 wraps the model load in try/except and calls `pytest.skip(...)` on ImportError or on any load failure ('can't load {REPO} (no internet or HF cache miss?)'), so with no network/HF cache the suite reports SKIPPED, not FAILED — a green pytest run does not by itself prove parity. It also never prints the max|Δlogits| value, only asserts on it.
```

**Verifier note**

```
Commands verified: README.md:159-162 matches the quote line-for-line (159 '# Architecture parity gate (the non-negotiable test before training).', 160 'pytest tests/ -v', 161 '# or, the script form:', 162 'python verify.py'). SmolLM2-134(base)/verify.py:11 and tests/test_parity.py:8 confirmed. Extra-coverage claims all confirmed by reading test_parity.py: :53 `assert n == 134_515_008`, :61 data_ptr tie check, :89-101 512-token long-context (`max_length=512` at :93), :104-132 per-layer over exactly 30 blocks (:127 `assert len(hf_states) == len(our_states) == 30`). One further mismatch worth knowing: the pytest fixture loads with the MODERN kwarg (test_parity.py:39 `dtype=torch.float32`) while verify.py:53 uses the deprecated `torch_dtype=`, so the two forms are not the identical call. The claim that parity.log is the SCRIPT form's output is confirmed — parity.log:6-9 reproduces verify.py's exact print strings (:70, :71, :81, :82).
```


### 5.5 SmolLM2-134(base): exact command to recompute the perplexity numbers

**Value**

```
There is NO standalone perplexity script. The only path is to regenerate and execute the notebook: cd "/home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)" && python3 _build_notebook.py && jupyter nbconvert --to notebook --execute results.ipynb --output results.ipynb --ExecutePreprocessor.timeout=2400
```

**Evidence** — `SmolLM2-134(base)/results/README.md:66-73`

**Source quote**

```
# from /home/yashb98/Downloads/BuildFromScratch/SmolLM2-134(base)/
python3 _build_notebook.py       # writes results.ipynb (28 cells, no outputs)
jupyter nbconvert --to notebook \
    --execute results.ipynb \
    --output results.ipynb \
    --ExecutePreprocessor.timeout=2400
# total runtime ~3 minutes (perplexity & training are the slow cells)
```

**Confidence** — measured from code

**Caveat** — MAJOR reproducibility weakness for a model card: the PPL code is a string literal inside a notebook GENERATOR (_build_notebook.py:230-273 `code("""# %% Perplexity on wikitext-2 validation ...""")`), not an importable/runnable .py. Running this command also re-executes a 150-step training cell that OVERWRITES ../checkpoint.pt (_build_notebook.py cell at :431-447 + results.ipynb cell 23 output 'Saved checkpoint.pt'). There is no way to recompute only the PPL without extracting the cell by hand.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
The command is correct, but 'There is NO standalone perplexity script' is false. SmolLM2-134(base)/eval_after_vs_base.py IS a standalone .py that computes wikitext-2-raw-v1 validation PPL — :74 `wk = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")`, with helper :50 `def ppl(model, text, seq=1024, stride=512, max_windows=200)` — for both the base (official weights in our class) and the TinyStories-trained checkpoint. The accurate statement is: no standalone script recomputes the HEADLINE 15.371 number. eval_after_vs_base.py would produce a DIFFERENT number (max_windows=200 vs the notebook's 61 windows; :29 `dtype = torch.bfloat16 if torch.cuda.is_available()` vs the notebook's fp32), it requires checkpoint_tinystories.pt (:43), and it has never been run to disk — its declared outputs results/tinystories_vs_base.md/.json are ABSENT from SmolLM2-134(base)/results/.
```

**Verifier note**

````
The nbconvert command matches results/README.md:67-73 verbatim (cited range 66-73 also picks up the ```bash fence at :66). The PPL code being a string literal in the generator is confirmed: _build_notebook.py:230 `code("""# %% Perplexity on wikitext-2 validation` through :273 `'seq_len': SEQ, 'stride': STRIDE}, f, indent=2)""")`. results.ipynb has exactly 28 cells (json len). Checkpoint-overwrite claim is TRUE but the line cite is wrong: the save is at _build_notebook.py:480-483 (`torch.save({'model': demo_model.state_dict(), ...` / `'checkpoint.pt')` / `print('Saved checkpoint.pt')`), NOT :431-447 (which is the cell's import block through `torch.manual_seed(0)`). Corroborated by results.ipynb cell 23 output 'Saved checkpoint.pt' and SmolLM2-134(base)/checkpoint.pt (538173921 B, mtime 2026-05-13 22:20, same minute as results.ipynb).
````


### 5.6 Was Qwen3 parity verified in fp32 on CPU only, or also on GPU?

**Value**

```
CPU ONLY, fp32. Neither Qwen3-0.6B/verify.py nor verify_run.py contains any .to(device)/.cuda()/device= call — tensors stay on the PyTorch CPU default.
```

**Evidence** — `Qwen3-0.6B/verify.py:51,55,61-64`

**Source quote**

```
line 51:     hf_model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
line 55:     ours = Qwen3ForCausalLM(Qwen3Config())
line 61:     input_ids = tokenizer(text, return_tensors="pt").input_ids
line 63:     hf_out = hf_model(input_ids).logits          # (1, T, V)
line 64:     our_out = ours(input_ids)["logits"]
```

**Confidence** — measured from code

**Caveat** — Independently asserted in prose at Qwen3-0.6B/README.md:498 `python verify.py        # parity gate — runs on CPU, no GPU needed`. Note verify.py also does NOT `import safe_cuda` — it is the one PyTorch entry point exempt from the CLAUDE.md §C1 rule, which is consistent with it never touching CUDA. verify_run.py:42 likewise uses `dtype=torch.float32` with no device move. No GPU parity check exists for Qwen3 anywhere in the repo.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Independently re-grepped: `grep -n "manual_seed|deterministic|benchmark|tf32|matmul_precision|cuda|device"` across Qwen3-0.6B/verify.py, SmolLM2-134(base)/verify.py and verify_run.py returns ONLY two comment lines (Qwen3 verify.py:59 and SmolLM2 verify.py:61, both '# Same prompt, same dtype, same device.'). No .cuda(), no .to(), no device= anywhere. Cited lines match: verify.py:51 `dtype=torch.float32`, :55, :61, :63, :64. verify_run.py:42 `dtype=torch.float32` confirmed. safe_cuda absence confirmed (neither verify.py nor verify_run.py imports it). README.md:498 corroboration confirmed. No GPU parity artifact exists for Qwen3 — unlike SmolLM2, which does have one (see next fact).
```


### 5.7 Was SmolLM2 parity verified in fp32 on CPU only, or also on GPU?

**Value**

```
verify.py: CPU ONLY, fp32 (no device move). A SEPARATE script, compare_with_hf.py, runs an expanded parity battery on GPU-if-available — but its JSON output was never written to disk.
```

**Evidence** — `SmolLM2-134(base)/verify.py:53,57,63-66`

**Source quote**

```
line 53:     hf_model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32)
line 57:     ours = SmolLM2ForCausalLM(SmolLM2Config())
line 63:     input_ids = tokenizer(text, return_tensors="pt").input_ids
line 65:     hf_out = hf_model(input_ids).logits          # (1, T, V)
line 66:     our_out = ours(input_ids)["logits"]
```

**Confidence** — measured from code

**Caveat** — GPU variant: SmolLM2-134(base)/compare_with_hf.py:40 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and :45/:48 `.to(device)`. It is designed to write results/comparison_with_hf.json (compare_with_hf.py:259-260), but that file does NOT exist on disk — `ls SmolLM2-134(base)/results/` shows only comparison_with_hf.md. So no GPU parity NUMBER is stamped anywhere. Prose confirmation that the committed number is CPU: SmolLM2-134(base)/results/POST_DATA.md:13 `| **0.000e+00** | `max \|Δlogits\|` between ours and HuggingFace (CPU fp32) | `results/parity.log` |`.

**Verdict — ❌ WRONG**

**Corrected value**

```
SmolLM2 parity was run on BOTH CPU and GPU, and the GPU numbers ARE recorded on disk (in prose, not JSON). SmolLM2-134(base)/results/comparison_with_hf.md:10 records 'Final-logits parity ... | max|Δ| = **4.72e-05** [GPU] | max|Δ| = **0.00e+00** [CPU]' and :11 'Per-layer hidden-state parity (30 layers) | max|Δ| = **1.95e-03** at layer 14 [GPU] | max|Δ| = **0.00e+00** at every layer [CPU]', plus :14 'Long-context (401-token RoPE) | max|Δ| = **4.01e-05**'. The same table is duplicated at SmolLM2-134(base)/README.md:71-74. Critically for a model card: the GPU per-layer delta 1.95e-03 EXCEEDS the repo's own 1e-3 gate — comparison_with_hf.md:49 is a whole section titled 'What the earlier ✗ at "1.953e-3" meant — and didn't mean'. The bit-exact 0.0 claim is CPU-ONLY; on GPU the reproduction is close but not bit-exact, attributed at comparison_with_hf.md:22-42 to SDPA backend dispatch (HF explicit mask vs our `is_causal=True`).
```

**Verifier note**

```
Only the machine-readable JSON is genuinely absent — `ls SmolLM2-134(base)/results/` confirms comparison_with_hf.json is not present while comparison_with_hf.md (4863 B, 2026-05-13 22:07) is. The verifier listed that .md but did not open it. Everything else in the fact checks out: verify.py:53/57/63/65/66 as quoted; compare_with_hf.py:39 manual_seed(0), :40 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`, :45/:48 `.to(device)`, :259-260 the unwritten JSON dump; POST_DATA.md:13 '| **0.000e+00** | `max |Δlogits|` between ours and HuggingFace (CPU fp32) | `results/parity.log` |'. Mark the GPU numbers confidence 'prose-only' — no JSON or stdout log backs them; grep for '4.72e-05|1.95e-03' across the repo hits only comparison_with_hf.md and README.md.
```


### 5.8 Which dtype kwarg does each verify.py use (version-sensitivity signal)?

**Value**

```
Qwen3 uses the MODERN `dtype=`; SmolLM2 uses the DEPRECATED `torch_dtype=`, which emits a deprecation warning captured in the committed parity.log.
```

**Evidence** — `SmolLM2-134(base)/results/parity.log:1`

**Source quote**

```
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
```

**Confidence** — results JSON

**Caveat** — SmolLM2-134(base)/verify.py:53 `torch_dtype=torch.float32` vs Qwen3-0.6B/verify.py:51 `dtype=torch.float32`. The warning proves parity.log was produced under a transformers version that had already deprecated torch_dtype (consistent with the pinned transformers==5.8.0), but the log does NOT record the version number. The same deprecation line appears in Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_eval_run2.log:1, because eval_original_vs_repro.py:49 also uses torch_dtype=.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
SmolLM2-134(base)/results/parity.log:1 is exactly '[transformers] `torch_dtype` is deprecated! Use `dtype` instead!'. Qwen3-0.6B/verify.py:51 `dtype=torch.float32`; SmolLM2-134(base)/verify.py:53 `torch_dtype=torch.float32`. Qwen3 .../results/original_eval_run2.log:1 carries the same warning, consistent with eval_original_vs_repro.py:49 `torch_dtype=torch.bfloat16`. Two nuances if this is stated on a card: (a) the split is per-FILE, not per-repo — SmolLM2's own tests/test_parity.py:39 uses the modern `dtype=`, so 'SmolLM2 uses torch_dtype' is true only of verify.py; (b) the warning bounds transformers from below but names no version — parity.log carries no version stamp at all, so 'consistent with transformers==5.8.0' is inference, not evidence.
```


### 5.9 What determinism flags are set (torch.manual_seed, use_deterministic_algorithms, cudnn.deterministic/benchmark, TF32, CUBLAS_WORKSPACE_CONFIG) in the two verify.py files?

**Value**

```
NONE. Neither verify.py sets any seed or any determinism/TF32 flag. Both files' complete import+setup blocks are two imports and a module constant.
```

**Evidence** — `Qwen3-0.6B/verify.py:13-19`

**Source quote**

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Qwen3ForCausalLM, Qwen3Config


REPO = "Qwen/Qwen3-0.6B-Base"
```

**Confidence** — measured from code

**Caveat** — SmolLM2-134(base)/verify.py:13-19 is structurally identical (`import torch` / `from transformers import ...` / `from model_full import ...` / `REPO = "HuggingFaceTB/SmolLM2-135M"`). Absence of a seed is defensible here — both scripts are pure forward passes with no sampling — but it means the scripts carry no determinism contract of their own.

**Verdict** — _no 1:1 verifier entry; see Additional verifier findings below._


### 5.10 Are TF32 / deterministic-algorithm / cuDNN flags set ANYWHERE in the repo?

**Value**

```
NO — zero occurrences repo-wide of allow_tf32, use_deterministic_algorithms, cudnn.deterministic, cudnn.benchmark, CUBLAS_WORKSPACE_CONFIG, or set_float32_matmul_precision.
```

**Evidence** — `(repo-wide grep, /home/yashb98/Downloads/BuildFromScratch)`

**Source quote**

```
$ grep -rn "allow_tf32\|use_deterministic_algorithms\|cudnn.deterministic\|cudnn.benchmark\|CUBLAS_WORKSPACE_CONFIG\|set_float32_matmul_precision" --include="*.py" --include="*.sh" --include="*.md" . | grep -v "\.git/"
(no output)
```

**Confidence** — measured from code

**Caveat** — This is a definitive NEGATIVE finding, not an absence of searching. It matters for the GPU-side numbers (the SmolLM2 PPL and all Qwen3 PPL/training), where TF32 on Blackwell is a real numerical variable that is left at the PyTorch default and never recorded. It does NOT affect the CPU fp32 parity checks.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
I re-ran the grep with the stated includes (exit 1, no output) AND without any --include filter across every file type in the repo. The only hits for the broader term 'tf32/TF32' are prose/constants unrelated to a PyTorch flag: mfu_meter.py:115 (an error-message string) and research/systems/roofline_hybridssm.py:45-49/:471-472/:754 (a `peak_fp32_tf32_tflops_assumed` device-peak constant). Not one of the six actual flags appears anywhere. Definitive negative finding.
```


### 5.11 Which seeds ARE set, and where (for the numbers that are not from verify.py)?

**Value**

```
torch.manual_seed(0) in the notebook that produced the SmolLM2 PPL; torch.manual_seed(0) in compare_with_hf.py; torch.manual_seed(seed)+torch.cuda.manual_seed_all(seed) in the Qwen3 trainer. No seed in either verify.py.
```

**Evidence** — `SmolLM2-134(base)/_build_notebook.py:45-46`

**Source quote**

```
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Confidence** — measured from code

**Caveat** — Also: SmolLM2-134(base)/compare_with_hf.py:39 `torch.manual_seed(0)`; SmolLM2-134(base)/_build_notebook.py:201 `torch.manual_seed(42)` (generation cell) and :211/:447; Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:227-228 `torch.manual_seed(seed)` / `torch.cuda.manual_seed_all(seed)`. The paper appendix records seed 0 for training (research/papers/qwen3-imu1-matched-compute/sections/reproducibility.tex:6 'in bfloat16 with seed $0$').

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Every cited line verified: _build_notebook.py:45 `torch.manual_seed(0)` / :46 device line (quote matches exactly); :201 `torch.manual_seed(42)`; :447 `torch.manual_seed(0)`; compare_with_hf.py:39 and :211; train_qwen3.py:227 `torch.manual_seed(seed)` / :228 `torch.cuda.manual_seed_all(seed)`. Two completeness notes for a card: the Qwen3 trainer's set_seed also covers Python and NumPy (train_qwen3.py:225 `random.seed(seed)`, :226 `np.random.seed(seed)`), and the enumeration is not exhaustive — a repo-wide grep also finds SmolLM2-134(base)/train.py:95-96, train_tinystories.py:99-100, benchmark_training.py:23, eval_after_vs_base.py:112/:115. The paper cite is a cross-line merge: reproducibility.tex:5 ends '...memory) in' and :6 begins 'bfloat16 with seed $0$.' — the phrase is real, the single-line attribution to :6 is approximate.
```


### 5.12 What exactly is the parity tolerance?

**Value**

```
max |Δlogits| < 1e-3 (absolute), plus a hard next-token argmax-equality assertion. Identical threshold in all four parity implementations.
```

**Evidence** — `Qwen3-0.6B/verify.py:74,81`

**Source quote**

```
line 74:     assert max_abs < 1e-3, f"Outputs diverge: {max_abs}. Architecture mismatch."
line 81:     assert hf_next == our_next, "Next-token disagreement"
```

**Confidence** — measured from code

**Caveat** — Same 1e-3 at: SmolLM2-134(base)/verify.py:76; Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/verify_run.py:33 `TOLERANCE = 1e-3`; SmolLM2-134(base)/tests/test_parity.py:73, :98, :132. NOTE a docstring inconsistency: SmolLM2-134(base)/verify.py:6 says the logits match 'to bf16 numerical tolerance' while the code comment at :73-75 and the run are fp32 — the module docstring is stale/wrong. Qwen3-0.6B/verify.py:6 correctly says 'fp32 numerical tolerance'.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Verified at Qwen3-0.6B/verify.py:74 and :81 (quote exact), SmolLM2-134(base)/verify.py:76, verify_run.py:33 `TOLERANCE = 1e-3`, tests/test_parity.py:73/:98/:132. The docstring inconsistency is real and correctly identified: SmolLM2-134(base)/verify.py:6 says 'the logits match to bf16 numerical tolerance' while :73-75 comments describe fp32 ~1e-5 noise and the run is fp32; Qwen3-0.6B/verify.py:6 correctly says 'fp32 numerical tolerance'. Count nuance: there are FIVE implementations, not four — compare_with_hf.py also gates on 1e-3 at :235, :240, :249. That is corroborating, but the enumeration is incomplete. Material context found while checking: SmolLM2-134(base)/results/comparison_with_hf.md:51 records 'The threshold in `compare_with_hf.py` was `1e-3`, picked for bf16 tolerance' — i.e. the repo itself notes the 1e-3 gate is loose for an fp32 claim, and the GPU per-layer run tripped it at 1.95e-03.
```


### 5.13 Is "max error 0.0" a real reported value, and from which run/file?

**Value**

```
YES — it is real and appears in FOUR on-disk artifacts, two of them primary machine outputs. SmolLM2: results/parity.log records max |Δlogits| = 0.000e+00. Qwen3: results/verify.json records the raw float "max_abs_error": 0.0.
```

**Evidence** — `SmolLM2-134(base)/results/parity.log:6-9`

**Source quote**

```
max |Δlogits| = 0.000e+00
relative      = 0.000e+00
HF next token : ' the'
Ours next     : ' the'
```

**Confidence** — results JSON

**Caveat** — Second primary artifact: Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/verify.json — `"max_abs_error": 0.0, "relative_error": 0.0, "hf_next_token_text": " Paris", "argmax_match": true, "passed": true`. Because verify.json stores the raw float, Qwen3's exact-zero is unambiguous; parity.log is a `.3e` format string (verify.py:70) so 0.000e+00 also implies exact zero (any nonzero would print an exponent, e.g. 1.234e-07). Derived/secondary copies: SmolLM2-134(base)/results/summary.json:7 `"max |Δlogits| vs HF": "0.000e+00"` and the executed notebook results.ipynb cell 6 output `max |Δlogits|     = 0.000e+00`. The repo itself flags exact-zero as needing justification and explains it (SmolLM2-134(base)/README.md:708-718: 'The Δ = 0.0 result deserves a sanity check... Because we use the same PyTorch primitives ... in the same order, with the same dtypes, on the same input.').

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
The value is real and correctly sourced, but it MUST be stated as a CPU-fp32-only result. The same repo records non-bit-exact GPU parity for SmolLM2 (comparison_with_hf.md:10-11: 4.72e-05 final-logits, 1.95e-03 per-layer at L14). A model card that says 'max error 0.0' without the device qualifier is misleading. Also the artifact count is 5+, not 4: comparison_with_hf.md:10 carries a fifth on-disk copy ('max|Δ| = **0.00e+00**' in the CPU column), plus README.md:56 and Qwen3-0.6B/README.md:39.
```

**Verifier note**

```
Primary artifacts verified verbatim. SmolLM2-134(base)/results/parity.log:6-9 matches the quote exactly, and verify.py:70 is `print(f'max |Δlogits| = {max_abs:.3e}')` so the .3e format does imply exact zero. Qwen3 results/verify.json read in full: "max_abs_error": 0.0, "relative_error": 0.0, "hf_next_token_text": " Paris", "argmax_match": true, "passed": true — a raw float, unambiguous. Derived copies confirmed: summary.json:7 '"max |Δlogits| vs HF": "0.000e+00"'; results.ipynb cell 6 stream output 'max |Δlogits|     = 0.000e+00'. The self-scrutiny passage is at SmolLM2-134(base)/README.md:708-718 as cited and reads as quoted.
```


### 5.14 What input(s) is parity checked on (prompt, token count, batch shape)?

**Value**

```
A single 5-token prompt, batch size 1: "The capital of France is" → shape (1, 5). Same prompt for both models.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/verify.json`

**Source quote**

```
"prompt": "The capital of France is",
  "dtype": "float32",
  "tolerance": 0.001,
  "max_abs_error": 0.0,
  "input_shape": [
    1,
    5
  ]
```

**Confidence** — results JSON

**Caveat** — Prompt set in code at Qwen3-0.6B/verify.py:60 `text = "The capital of France is"`, verify_run.py:34 `PROMPT = "The capital of France is"`, SmolLM2-134(base)/verify.py:62 (identical string). SmolLM2's 5 tokens are recorded explicitly as [504, 3575, 282, 4649, 314] (results.ipynb cell 6 output 'Tokens : [504, 3575, 282, 4649, 314]'; SmolLM2-134(base)/README.md:60). This is a THIN gate for a model card — n=1 prompt, 5 tokens, no batching, no long context. SmolLM2 alone has broader coverage via tests/test_parity.py (512-token long-context at :89-101, per-layer over all 30 blocks at :104-132); Qwen3 has NO long-context or per-layer parity check anywhere.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
verify.json read in full — "prompt": "The capital of France is", "dtype": "float32", "tolerance": 0.001, "max_abs_error": 0.0, "input_shape": [1, 5] — quote is exact. Prompt strings confirmed at Qwen3-0.6B/verify.py:60, verify_run.py:34, SmolLM2-134(base)/verify.py:62, test_parity.py:68/:79/:107. SmolLM2 token ids confirmed from results.ipynb cell 6 output 'Tokens : [504, 3575, 282, 4649, 314]'. The 'thin gate' judgement is sound. Two side findings: (1) SmolLM2-134(base)/README.md:60 attributes those token ids to `results/summary.json`, but I read summary.json in full and it has NO tokenization key (keys: Architecture, Param count (unique), lm_head tied, RoPE θ, Tokenizer, max |Δlogits| vs HF, Argmax for "France is", PPL ours, PPL HF, Demo-run final loss, Demo-run steps) — same misattribution class as fact 26; cite results.ipynb cell 6 instead. (2) The two models' argmaxes differ — Qwen3 predicts ' Paris' (verify.json), SmolLM2 predicts ' the' (parity.log:8) — so a card must not present one argmax as shared.
```


### 5.15 What is the SmolLM2 perplexity number and its exact eval recipe (dataset id, config, split, seq_len, stride, tokenizer, device, dtype)?

**Value**

```
ours_ppl 15.370989092449635 / hf_ppl 15.370989964425396 on 62,403 target tokens. Recipe: dataset 'Salesforce/wikitext', config 'wikitext-2-raw-v1', split 'validation'; non-blank rows joined by '\n\n'; SmolLM2's OWN tokenizer (HuggingFaceTB/SmolLM2-135M); SEQ=1024, STRIDE=512, capped at the first 32,000 tokens; fp32 weights with logits upcast .float(); device = cuda (NVIDIA GB10).
```

**Evidence** — `SmolLM2-134(base)/_build_notebook.py:233-242`

**Source quote**

```
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')
text = '\\n\\n'.join(ex['text'] for ex in ds if ex['text'].strip())
encodings = tokenizer(text, return_tensors='pt')
input_ids = encodings.input_ids[0]
print(f'Validation tokens: {len(input_ids):,}')

# Slide a 1024-token window with stride 512 over the first 32K tokens.
SEQ = 1024
STRIDE = 512
N_TOKENS = min(len(input_ids), 32_000)
```

**Confidence** — results JSON

**Caveat** — Values from SmolLM2-134(base)/results/perplexity.json. THREE caveats a model card must state: (1) NON-STANDARD sliding window — loss is taken over ALL 1023 shifted positions of every window (_build_notebook.py:251-254) while advancing by STRIDE=512, so every overlapped token is COUNTED TWICE. This is NOT the standard HF masked/target_len sliding-window PPL, so the 15.371 is not comparable to published wikitext-2 PPLs. I verified the arithmetic: range(0, 32000-1024, 512) = 61 windows x 1023 = 62,403, exactly matching perplexity.json's `"tokens": 62403`. (2) Only the first 32,000 of 268,140 validation tokens are used (results.ipynb cell 14 output 'Validation tokens: 268,140') — 12% of the split. (3) The PPL ran on GPU while parity ran on CPU (_build_notebook.py:244 `net = net.to(device).eval()` with device from :46); results.ipynb cell 16 output confirms 'Model on: cuda:0'. No dataset revision is pinned.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
This is the strongest-verified fact in the set. perplexity.json read in full: ours_ppl 15.370989092449635, hf_ppl 15.370989964425396, tokens 62403, dataset 'wikitext-2-raw-v1 validation', seq_len 1024, stride 512. The config name IS the -raw- variant in the code, not just the prose: _build_notebook.py:233 `ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='validation')`. Quoted block :233-242 matches line-for-line. I independently re-derived the arithmetic: range(0, 32000-1024, 512) = range(0, 30976, 512) = 61 starts (0…30720); 61 × 1023 = 62,403 = perplexity.json's `tokens`. The double-counting caveat is correct — window 1024 with stride 512 counts every overlapped target twice, so this is NOT standard HF masked sliding-window PPL and is not comparable to published wikitext-2 numbers. 268,140 total validation tokens confirmed at results.ipynb cell 14 output; 32000/268140 = 11.9%. Tokenizer confirmed: _build_notebook.py:112 `tokenizer = AutoTokenizer.from_pretrained(REPO)` with REPO imported at :43 from verify.py (= HuggingFaceTB/SmolLM2-135M). fp32 confirmed at :110 `hf_model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)` and :251 `.float()` upcast. GPU confirmed via :244 `net = net.to(device).eval()` with device from :46; note the 'Model on: cuda:0' string the fact cites is from cell 16 (the attention cell), not the PPL cell — the PPL-on-GPU conclusion rests on the code path, which is sound. Windows actually span the first 31,744 tokens (last window 30720:31744), slightly less than the stated 32,000 cap.
```


### 5.16 What is the Qwen3 perplexity number and its exact eval recipe?

**Value**

```
Published Qwen3-0.6B-Base val PPL = 13.400; repro checkpoints lr17 46.892 / lr24 46.310 / lr30 49.276. Recipe: FineWeb-Edu (HuggingFaceFW/fineweb-edu, 'sample-10BT') 300,000-token val slice; 50 NON-overlapping windows of SEQ_LEN 4096 = 204,800 target tokens; Qwen3-0.6B-Base tokenizer; bfloat16; device cuda.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_vs_repro.txt:1-2`

**Source quote**

```
[2026-06-09 16:51:36] Original vs reproduction — val=300,000 tokens, 50 windows x 4096
ORIGINAL  Qwen3-0.6B-Base (36T tok)   val PPL =   13.400  (204,800 tok, 21s)
```

**Confidence** — results JSON

**Caveat** — CRITICAL: this is NOT wikitext — the two models' PPL numbers are on DIFFERENT corpora and are not comparable to each other. Recipe from eval_original_vs_repro.py:21 `SEQ_LEN, MAX_WINDOWS = 4096, 50`, :30 `for begin in range(0, min(len(val) - SEQ_LEN, MAX_WINDOWS * SEQ_LEN), SEQ_LEN)` (stride == SEQ_LEN, so NO overlap and no double-counting, unlike SmolLM2), :41 `device = torch.device("cuda")`, :49 `AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16)`. Corpus identity from train_qwen3.py:151 `ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)`; tokenizer from train_qwen3.py:281 `tokenizer = AutoTokenizer.from_pretrained(REPO)` with REPO='Qwen/Qwen3-0.6B-Base' (:59). bf16 (not fp32) means this number carries real numerical noise that is never quantified.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
original_vs_repro.txt read in full — all four PPL values and the header line match verbatim, including '(204,800 tok, 21s)'. Recipe lines all verified: eval_original_vs_repro.py:21 `SEQ_LEN, MAX_WINDOWS = 4096, 50`, :30 the range with stride == SEQ_LEN, :41 cuda, :49 `torch_dtype=torch.bfloat16`. I re-derived the token count: min(300000-4096, 50*4096) = 204800, range(0,204800,4096) = 50 windows, each contributing ids.size(1)-1 = 4096 → 204,800. Targets do not overlap, so the no-double-counting claim holds (inputs share exactly 1 boundary token per window). Corpus at train_qwen3.py:151 `load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)`; tokenizer at :281 with REPO='Qwen/Qwen3-0.6B-Base' at :59. The 'different corpora, not comparable to SmolLM2' warning is correct and material. The bf16-noise-unquantified caveat is fair: no seed-repeat or CI exists for this number.
```


### 5.17 Is the Qwen3 13.400 a number this repo MEASURED, or one COPIED from a paper / model card?

**Value**

```
MEASURED by this repo. It is the output of eval_original_vs_repro.py loading the published Qwen/Qwen3-0.6B-Base weights and scoring them with the repo's own eval loop on the repo's own val slice.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/eval_original_vs_repro.py:46-51`

**Source quote**

```
# --- the original published model ---
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(device)
    ppl_orig, n = eval_ppl(hf, val, device)
    lines.append(f"ORIGINAL  Qwen3-0.6B-Base (36T tok)   val PPL = {ppl_orig:8.3f}  ({n:,} tok, {time.time()-t0:.0f}s)")
```

**Confidence** — measured from code

**Caveat** — The run is corroborated by a live stdout log with a real safe_cuda banner: results/original_eval_run2.log:2 '[safe_cuda] capped CUDA at 85% of 129 GB unified pool (~109 GB)' and :3 'Loading weights: 100%|██████████| 310/310'. The '36T tokens' attribution in the label IS copied from Qwen3 published material, not measured. Same for SmolLM2: perplexity.json's hf_ppl 15.370990 is measured by the same loop on the same slice (_build_notebook.py:263 `hf_ppl, _ = ppl(hf_model)`), not copied.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
eval_original_vs_repro.py:46-51 matches the quote line-for-line, including `hf = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(device)` at :49 and the ppl_orig eval at :50. Live-run corroboration confirmed: results/original_eval_run2.log:2 '[safe_cuda] capped CUDA at 85% of 129 GB unified pool (~109 GB); over-allocation now errors cleanly.' and :3 'Loading weights: 100%|██████████| 310/310'; :6 of that log carries the same 13.400 line, and :13 confirms it wrote original_vs_repro.txt. The separation the fact draws is correct and important: 13.400 is MEASURED, the '(36T tok)' label inside the same string is COPIED from Qwen3 published material and is not verifiable from any file in this repo. SmolLM2's hf_ppl is likewise measured by the same loop (_build_notebook.py:263 `hf_ppl, _ = ppl(hf_model)`).
```


### 5.18 Does the Qwen3 PPL command use the decontaminated val split?

**Value**

```
NO. eval_original_vs_repro.py hardcodes the PRE-audit token cache tokcache_133072000_300000.pt, whose filename lacks the seed/tokenizer tag that the decontamination fix introduced.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:130-136`

**Source quote**

```
"""Stream FineWeb-Edu sample-10BT, tokenize on the fly, and build a
    DOCUMENT-DISJOINT, DECONTAMINATED train/val split (audit fix DATA-1/3):

      * each whole document is routed to train or val by a seeded hash
        (`is_val_doc`), so train/val never share a document and no document spans
        the boundary (the old code cut the stream by token count — val was the
        sequential continuation of train, leak-suspect);
```

**Confidence** — measured from code

**Caveat** — The post-fix cache naming is train_qwen3.py:145 `cache = RESULTS / f"tokcache_{n_train}_{n_val}_seed{seed}_{tok_tag}.pt"`. eval_original_vs_repro.py:22 points at `tokcache_133072000_300000.pt` — no seed/tok tag — i.e. the OLD leak-suspect sequential split. Both files exist on disk: the old one dated Jun 8 (used for the 13.400 run on Jun 9) and tokcache_133072000_300000_seed0_Qwen3-0.6B-Base.pt dated Jun 18 (post-fix). results/decontam_report.json (docs_dropped: 0) is dated Jul 7 and corresponds to a later, larger cache, NOT to the 13.400 run. So the headline PPL row predates the decontamination fix and re-running the command today reproduces it on the same pre-fix split.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Strongly supported. train_qwen3.py:130-136 matches the quoted docstring verbatim, including 'the old code cut the stream by token count — val was the sequential continuation of train, leak-suspect'. Post-fix naming confirmed at train_qwen3.py:145 `cache = RESULTS / f"tokcache_{n_train}_{n_val}_seed{seed}_{tok_tag}.pt"`; eval_original_vs_repro.py:22 points at the untagged name. Both caches exist: tokcache_133072000_300000.pt mtime 2026-06-08 21:13 and tokcache_133072000_300000_seed0_Qwen3-0.6B-Base.pt mtime 2026-06-18 13:45 — and the 13.400 run is timestamped 2026-06-09 16:51:36 in original_vs_repro.txt:1, i.e. 9 days BEFORE the post-fix cache existed. decontam_report.json mtime 2026-07-07 15:13, content docs_dropped 0 / n_val_docs_kept 451 — confirmed. One softening: 'corresponds to a later, larger cache' is not verifiable — decontam_report.json contains no cache filename or token count (keys: split_seed, val_fraction, ngram_n, overlap_threshold, n_train_docs, n_val_docs_raw, docs_dropped, method, train_sample_docs_for_index), and train_qwen3.py:184 writes it to a fixed path `RESULTS / "decontam_report.json"` that any re-stream overwrites. What IS provable is the date ordering, which is sufficient for the claim.
```


### 5.19 Every place a python/torch/CUDA version is PINNED in the repo

**Value**

```
Exactly TWO files, both under SmolLM2-134(base), and they CONTRADICT each other. pyproject.toml has hard pins (torch==2.11.0, transformers==5.8.0, datasets==4.8.5, safetensors==0.7.0, accelerate==1.13.0, numpy==2.4.4, requires-python>=3.10, pytest==9.0.3). requirements.txt has loose floors (torch>=2.4, transformers>=4.40).
```

**Evidence** — `SmolLM2-134(base)/pyproject.toml:1-2,17,19-27`

**Source quote**

```
line 1: # Concrete pins, matching the working environment that produced
line 2: # max|Δlogits|=0.0 against HuggingFaceTB/SmolLM2-135M.
line 17: requires-python = ">=3.10"
line 19: dependencies = [
line 20:     # Pinned to the versions that verified parity in this repo.
line 21:     "torch==2.11.0",
line 22:     "transformers==5.8.0",
line 23:     "datasets==4.8.5",
line 24:     "safetensors==0.7.0",
line 25:     "accelerate==1.13.0",
line 26:     "numpy==2.4.4",
line 27: ]
```

**Confidence** — measured from code

**Caveat** — The contradiction: SmolLM2-134(base)/requirements.txt:1-2 reads `torch>=2.4` / `transformers>=4.40`, yet root README.md:157 offers them as equivalent (`pip install -e .                # or: pip install -r requirements.txt`) directly under the comment at :156 'Install pinned dependencies that produced the 0.0 logit-diff result.' Following the requirements.txt branch can install torch 2.4 / transformers 4.x, under which SmolLM2's verify.py would still run (torch_dtype= is valid there) but bit-exactness is NOT the pinned-environment claim. No CUDA/cuDNN/driver version is pinned in either file. pyproject.toml:6-8 itself admits no lockfile exists: 'To freeze a true lockfile (recommended for reproducibility): uv pip compile pyproject.toml -o requirements.lock'.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
pyproject.toml read in full — :1-2 comment, :17 `requires-python = ">=3.10"`, :21-26 torch==2.11.0 / transformers==5.8.0 / datasets==4.8.5 / safetensors==0.7.0 / accelerate==1.13.0 / numpy==2.4.4, :31 pytest==9.0.3 — all as claimed. requirements.txt read in full: `torch>=2.4` / `transformers>=4.40` / safetensors / accelerate / datasets (the last three fully unpinned, which the fact does not mention but which strengthens it). The contradiction via root README.md:157 `pip install -e .                # or: pip install -r requirements.txt` under :156's 'Install pinned dependencies that produced the 0.0 logit-diff result.' is confirmed. No CUDA/cuDNN/driver pin in either. pyproject.toml:4-7 does carry the no-lockfile admission ('To freeze a true lockfile (recommended for reproducibility): uv pip compile pyproject.toml -o requirements.lock') — cited as :6-8, the text actually spans :4-7.
```


### 5.20 Every place a python/torch/CUDA version is STAMPED from an actual execution

**Value**

```
Exactly ONE: the executed notebook SmolLM2-134(base)/results.ipynb. Cell 1 output stamps 'Torch: 2.11.0+cu130' and 'Device: cuda | NVIDIA GB10'; notebook metadata.language_info.version stamps python '3.12.11'.
```

**Evidence** — `SmolLM2-134(base)/results.ipynb (cell 1 output; metadata.language_info.version)`

**Source quote**

```
Torch: 2.11.0+cu130
Device: cuda | NVIDIA GB10

(metadata.language_info.version = "3.12.11")
```

**Confidence** — results JSON

**Caveat** — Produced by SmolLM2-134(base)/_build_notebook.py:50-51 `print('Torch:', torch.__version__)` / `print('Device:', device, '|', torch.cuda.get_device_name(0) ...)`. A repo-wide grep for `torch.__version__|torch.version.cuda|sys.version|platform.python_version` in *.py returns only this line and one unrelated HybridSSM file (HybridSSM-0.2B/experiments/2026-07-29_hybrid-ssm-0.2b_throughput/step_runner.py:245 `"host": platform.node(), "python": platform.python_version(),`). So NO Qwen3 script and NEITHER verify.py stamps any version.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Read results.ipynb programmatically: metadata.language_info.version == '3.12.11', and cell 1's stream output is exactly 'Torch: 2.11.0+cu130\nDevice: cuda | NVIDIA GB10'. Generator lines confirmed at _build_notebook.py:50-51. I re-ran the repo-wide grep for `torch.__version__|torch.version.cuda|platform.python_version|sys.version` over all *.py: exactly two hits, _build_notebook.py:50 and HybridSSM-0.2B/experiments/2026-07-29_hybrid-ssm-0.2b_throughput/step_runner.py:245 — as claimed. I additionally checked whether step_runner's stamp reached disk: grep for '"python":' under that experiment dir returns only the .py itself, no output artifact, so 'exactly ONE' holds repo-wide. I also grepped all *.log/*.txt/*.json for 'Torch:|torch==|2.11.0': zero hits, confirming no result artifact carries a version.
```


### 5.21 Do the repo's recorded versions AGREE with the live box (python 3.12.11, torch 2.11.0+cu130, CUDA 13.0, cuDNN 91900, driver 580.142, NVIDIA GB10)?

**Value**

```
For SmolLM2: YES, exactly, on every field the repo records. For Qwen3: UNDETERMINABLE — the Qwen3 subproject records no versions at all.
```

**Evidence** — `SmolLM2-134(base)/pyproject.toml:21-23 vs live interpreter`

**Source quote**

```
repo pins:   "torch==2.11.0", "transformers==5.8.0", "datasets==4.8.5"
live box:    python 3.12.11 / torch 2.11.0+cu130 / torch.version.cuda 13.0 / transformers 5.8.0 / datasets 4.8.5
notebook:    Torch: 2.11.0+cu130 | Device: cuda | NVIDIA GB10 | language_info.version 3.12.11
```

**Confidence** — measured from code

**Caveat** — I verified the live stack myself by running `python3 -c "import sys,torch,transformers,datasets; ..."` (no CUDA init): python 3.12.11, torch 2.11.0+cu130, torch.version.cuda 13.0, transformers 5.8.0, datasets 4.8.5. These match pyproject and the notebook stamp field-for-field, so the SmolLM2 results were NOT produced under an older stack — bit-exactness there is currently re-checkable. Two gaps remain: (a) cuDNN 91900 and driver 580.142 are recorded NOWHERE in the repo, so version-drift in those two cannot be detected; (b) pyproject pins the base version 'torch==2.11.0' without the +cu130 local tag, so a CUDA-12 or CPU build of the same version satisfies the pin. For Qwen3 there is NO requirements.txt and NO pyproject.toml — its only install instruction is fully unpinned.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
I re-ran the live check myself rather than trusting the report: python 3.12.11, torch 2.11.0+cu130, torch.version.cuda 13.0, torch.backends.cudnn.version() 91900, transformers 5.8.0, datasets 4.8.5; `nvidia-smi --query-gpu=driver_version,name` returns '580.142, NVIDIA GB10'. These match pyproject.toml:21-23 and the results.ipynb stamp field-for-field. Both stated gaps are real: (a) 91900 and 580.142 appear nowhere on disk — my grep of all *.md/*.json/*.toml/*.txt/*.log for 'cuDNN|cudnn_version|580.14|CUDA 13|CUDA 12' returned exactly one unrelated prose hit, jax_vs_pytorch_tradeoffs.md:44; (b) `torch==2.11.0` has no +cu130 local tag, so a CPU or CUDA-12 build satisfies the pin. Qwen3-0.6B has no requirements.txt and no pyproject.toml (confirmed by find, below).
```


### 5.22 What does Qwen3-0.6B document as its dependency install?

**Value**

```
An unpinned one-liner in the README. Qwen3-0.6B has no requirements.txt, no pyproject.toml, and no lockfile.
```

**Evidence** — `Qwen3-0.6B/README.md:496-499`

**Source quote**

````
```bash
pip install torch transformers datasets safetensors accelerate
python verify.py        # parity gate — runs on CPU, no GPU needed
```
````

**Confidence** — measured from code

**Caveat** — Confirmed by a repo-wide find for requirements*.txt / environment*.yml / pyproject.toml / *.lock / setup.py / Pipfile: the only hits are SmolLM2-134(base)/pyproject.toml, SmolLM2-134(base)/requirements.txt, and skills_showcase/server/requirements.txt (a FastAPI web server, unrelated to either model: fastapi/uvicorn/pydantic/anthropic). For a model card, Qwen3's bit-exact claim therefore has no reproducible environment attached.

**Verdict — ✅ CONFIRMED**

**Verifier note**

````
Qwen3-0.6B/README.md:496-499 matches the quote exactly (496 ```bash, 497 `pip install torch transformers datasets safetensors accelerate`, 498 `python verify.py        # parity gate — runs on CPU, no GPU needed`, 499 ```). I re-ran the repo-wide find for requirements*.txt / pyproject.toml / environment*.yml / setup.py / Pipfile: exactly three hits — SmolLM2-134(base)/pyproject.toml, SmolLM2-134(base)/requirements.txt, skills_showcase/server/requirements.txt. Nothing under Qwen3-0.6B/. The conclusion — Qwen3's bit-exact claim ships with no reproducible environment — is sound.
````


### 5.23 Is there a git tag corresponding to these results?

**Value**

```
NO. The repository has zero tags.
```

**Evidence** — `/home/yashb98/Downloads/BuildFromScratch (git)`

**Source quote**

```
$ git tag -l
(no output)
```

**Confidence** — measured from code

**Caveat** — Current HEAD is 3da9063 on branch harden-research-loop, which is ~2 months of commits after both parity artifacts were produced (parity.log May 13, verify.json Jun 8). A model card cannot point at a tag; the best available anchors are the commits that ADDED the artifacts (next fact).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
`git tag -l | wc -l` = 0. HEAD = 3da9063 on branch harden-research-loop, confirmed by `git rev-parse --short HEAD` / `git branch --show-current`. One precision fix: HEAD's commit date is 2026-07-24 (`git log -1 --date=short`), so the gap is ~1.5 months after verify.json (mtime 2026-06-08 14:36) and ~2.3 months after parity.log (mtime 2026-05-13 22:20) — 'about 2 months' is a fair rounding but not exact for either.
```


### 5.24 Is there a recorded commit hash that these specific results correspond to?

**Value**

```
NOT_FOUND as a provenance stamp. No results file for either model carries a commit/git_sha field. The only commit anchors are the git history entries that first added the artifacts: verify.json → e791875; parity.log → 84a96c0.
```

**Evidence** — `/home/yashb98/Downloads/BuildFromScratch (git log)`

**Source quote**

```
$ git log --oneline -3 -- "Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/verify.json"
e791875 Add Qwen3-0.6B from-scratch reproduction + three-build experiment
$ git log --oneline -3 -- "SmolLM2-134(base)/results/parity.log"
84a96c0 Initial commit: SmolLM2-135M from-scratch reproduction + harness
```

**Confidence** — measured from code

**Caveat** — These are inferences from file history, NOT provenance the scripts recorded. Direct inspection of the artifacts confirms the absence: verify.json's full key set is repo/prompt/dtype/tolerance/max_abs_error/relative_error/hf_next_token_id/our_next_token_id/hf_next_token_text/our_next_token_text/argmax_match/passed/input_shape/total_seconds — no commit, no versions, no device, no timestamp. perplexity.json's full key set is ours_ppl/hf_ppl/tokens/dataset/seq_len/stride — likewise none. parity.log is raw stdout.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Re-ran both git-log queries: verify.json → e791875 'Add Qwen3-0.6B from-scratch reproduction + three-build experiment' (2026-06-10); parity.log → 84a96c0 'Initial commit: SmolLM2-135M from-scratch reproduction + harness' (2026-05-20). Note both adding commits POSTDATE the artifact mtimes (Jun 10 vs Jun 8; May 20 vs May 13), so these commits bound the results from above, not identify the tree they ran against — which reinforces the fact's own 'inference, not provenance' framing. The absence claim is verified by direct inspection: verify.json's full key set is exactly repo/prompt/dtype/tolerance/max_abs_error/relative_error/hf_next_token_id/our_next_token_id/hf_next_token_text/our_next_token_text/argmax_match/passed/input_shape/total_seconds; perplexity.json's is exactly ours_ppl/hf_ppl/tokens/dataset/seq_len/stride. Neither has commit, versions, device, or timestamp.
```


### 5.25 Does the repo's provenance/ledger machinery cover these two reproductions?

**Value**

```
NO. research/ledger/ledger.py DOES auto-capture the repo HEAD into every run entry, and 29 runs carry a git_commit — but the earliest is 2026-06-16 and NEITHER reproduction (SmolLM2 parity/PPL, nor the Qwen3 faithful verify) has a ledger run entry.
```

**Evidence** — `research/ledger/ledger.py:639`

**Source quote**

```
r["lineage"]["git_commit"] = git_head_commit()  # auto-capture repo HEAD
```

**Confidence** — measured from code

**Caveat** — Verified by enumerating ledger.json: run_ids run 2026-06-16_qwen3-faithful_eval-first (git_commit 86e79f3) through 2026-07-29_hybrid-ssm-0.2b_private-heldout-v1 (3da9063); no entry names the SmolLM2 repro or a Qwen3 verify/parity run. Additionally lineage.env is null for 28 of 29 runs (the sole exception is run[21], free text 'jax/flax on GB10; jax_safe_env guard active'), so even the covered runs stamp no software versions. research/provenance.py (§C22 span writer) exists but research/provenance/ contains a single file, 2026-07-14.jsonl, which post-dates both reproductions. No c5_evidence.json exists for either reproduction build — the 12 c5_evidence.json files all belong to later ablation/experiment runs.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
research/ledger/ledger.py:639 is exactly `r["lineage"]["git_commit"] = git_head_commit()  # auto-capture repo HEAD`. I enumerated ledger.json programmatically: 29 runs, all 29 carry git_commit; first = 2026-06-16_qwen3-faithful_eval-first (86e79f3), last = 2026-07-29_hybrid-ssm-0.2b_private-heldout-v1 (3da9063). No run_id names SmolLM2 or a Qwen3 verify/parity run. lineage.env is null for 28 of 29; the sole exception is index 21, 2026-07-19_hybrid-ssm-0.2b_pretrain-ssm-base-s0, env = 'jax/flax on GB10; jax_safe_env guard active' — exactly as claimed. research/provenance/ contains only 2026-07-14.jsonl. Minor count nuance: `find -name "c5_evidence*.json"` returns 12 paths, of which 11 are named exactly c5_evidence.json and one is c5_evidence_scale_ext.json; none is under a reproduction build — the substantive claim stands.
```


### 5.26 Does the paper's reproducibility appendix record software versions?

**Value**

```
NO. It records hardware (single NVIDIA GB10, ~119 GB unified), dtype (bfloat16), seed (0), batch/step config, and the exact commands — but no python, torch, CUDA, or transformers version.
```

**Evidence** — `research/papers/qwen3-imu1-matched-compute/sections/reproducibility.tex:4-10`

**Source quote**

```
are released at \url{https://github.com/yashb98/BuildFromScratch}. All training used
a single NVIDIA GB10 (Grace Blackwell, unified $\approx\!119$\,GB CPU+GPU memory) in
bfloat16 with seed $0$. Because the unified pool can be exhausted by a single large
allocation, every entry point caps the process at $85\%$ of the pool before initializing
the accelerator and computes the $151{,}936$-way cross-entropy in chunks; \texttt{torch.compile}
is enabled. The effective batch is $4 \times 4$ accumulation $= 65{,}536$ tokens, and both
arms train for $18{,}150$ steps.
```

**Confidence** — measured from code

**Caveat** — The appendix is otherwise unusually honest — reproducibility.tex:35-37 explicitly flags an unsaved artifact: 'its bundle-off-equals-baseline verification was run and recorded in the build documentation (printed to standard output rather than saved as a file---we note this explicitly so the provenance is not overstated).' Its command list (lines 15-27) matches the scripts on disk. Absence of a version block is the one clear gap for a model-card Reproduce section.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
I read reproducibility.tex in full (40 lines). Lines 4-10 match the quote verbatim. No version string appears anywhere in the file. The honesty passage is real, at :35-37 as cited: 'its bundle-off-equals-baseline verification was run and recorded in the build documentation (printed to standard output rather than saved as a file---we note this explicitly so the provenance is not overstated).' The command block at :13-28 lists verify_run.py, train_qwen3.py, verify_imu1.py, train_imu1.py, eval_original_vs_repro.py, figures/make_figures.py — all of which exist on disk. Worth pairing with fact 18 on a card: the appendix presents `python eval_original_vs_repro.py` as authoritative while that script reads the pre-decontamination cache.
```


### 5.27 Is the Qwen3 README's sourcing of the parity/param claim accurate?

**Value**

```
PARTLY WRONG. The README attributes params 596,049,920 to verify.json, but verify.json contains no params field. The max_abs_error = 0.0 and argmax " Paris" attributions ARE correct.
```

**Evidence** — `Qwen3-0.6B/README.md:39-40`

**Source quote**

```
**Bit-exact reproduction** — `verify.json`: `max_abs_error = 0.0`, argmax `" Paris"`,
params **596,049,920**. Our `model.py` *is* Qwen3-0.6B.
```

**Confidence** — measured from code

**Caveat** — verify.json's complete key list (quoted in an earlier fact) has no params/n_params/param_count key. 596,049,920 IS a genuine measured value, but it lives elsewhere: Qwen3-0.6B/model.py:306 prints it as an EXPECTED constant (`print(f"Expected:      ~596,049,920  (596M-branded, '0.6B')")`), and it is emitted as a real measurement in training logs, e.g. Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/run_sft_seed0.log:7 'base loaded strict=True (base PPL on FineWeb-Edu was 28.650247632362024); params=596,049,920'. A model card should cite the log, not verify.json. SmolLM2's equivalent claim is better sourced: 134,515,008 is asserted by a real test (tests/test_parity.py:53) and recorded in results/summary.json:3.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Qwen3-0.6B/README.md:39-40 matches the quote exactly, and I read verify.json in full: no params/n_params/param_count key. The alternative sources check out: Qwen3-0.6B/model.py:306 `print(f"Expected:      ~596,049,920  (596M-branded, '0.6B')")` — note this is a printed EXPECTED constant in a __main__ block, and the adjacent :305 `print(f"Unique params: {num_params(m):,}")` is the actual measurement, so model.py:306 alone is not a measurement either. The genuine measured stamp is Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/run_sft_seed0.log:7 'base loaded strict=True (base PPL on FineWeb-Edu was 28.650247632362024); params=596,049,920' — verified verbatim. SmolLM2's better sourcing also verified: tests/test_parity.py:53 `assert n == 134_515_008`, results/summary.json:3, and results/param_count.log:1 'params: 134,515,008    (target 134,515,008)'.
```


### 5.28 Is the documented standardized-benchmark path (lm-evaluation-harness) actually exercised?

**Value**

```
NO. SmolLM2-134(base)/scripts/run_lm_eval.sh exists and is documented in the root README quickstart, but its output directory results/lm_eval/ does not exist — it has never been run.
```

**Evidence** — `SmolLM2-134(base)/scripts/run_lm_eval.sh:22-27`

**Source quote**

```
BASE_REPO="HuggingFaceTB/SmolLM2-135M"
TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,commonsense_qa,openbookqa,mmlu"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
OUT_DIR="results/lm_eval"
```

**Confidence** — measured from code

**Caveat** — `ls "SmolLM2-134(base)/results/lm_eval"` → 'No such file or directory'. The script is advertised in root README.md:176-179 ('# Standardized benchmarks (lm-evaluation-harness wrapper): pip install lm-eval / bash scripts/run_lm_eval.sh'). A model card must not imply any downstream benchmark number for SmolLM2 from this repo — none exists. Note also the script would evaluate in bfloat16 (line 41 `dtype=bfloat16`), not the fp32 used for parity.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
scripts/run_lm_eval.sh:22-27 matches the quote line-for-line (22 BASE_REPO, 23 TASKS, 24 DEVICE, 25 BATCH_SIZE, 26 NUM_FEWSHOT, 27 OUT_DIR="results/lm_eval"). `ls "SmolLM2-134(base)/results/lm_eval"` → 'No such file or directory', and my full listing of results/ shows no lm_eval entry. Root README.md:176-179 advertises it as claimed. The bf16-vs-fp32 mismatch is confirmed at run_lm_eval.sh:41 `--model_args "pretrained=$BASE_REPO,dtype=bfloat16"`. Supporting evidence the verifier did not cite: SmolLM2-134(base)/results/comparison_with_hf.md:84-88 explicitly states downstream benchmarks were NOT computed ('Computing them from scratch would take a few hours per task') and asserts 'any benchmark score will match by construction' — an unverified claim a card must not repeat. Same non-execution pattern applies to eval_after_vs_base.py: its declared outputs results/tinystories_vs_base.md/.json are absent from disk.
```


### 5.V Additional verifier findings (no 1:1 extracted fact)

**5.V1 — ✅ CONFIRMED** · What determinism flags are set in the two verify.py files?

**Checked against**

```
NONE. Neither verify.py sets any seed or any determinism/TF32 flag.
```

**Verifier note**

```
Verified by direct grep across both files for manual_seed / deterministic / benchmark / tf32 / matmul_precision: zero hits. Qwen3-0.6B/verify.py:13-19 matches the quote line-for-line (13 `import torch`, 14 `from transformers import ...`, 16 `from model import Qwen3ForCausalLM, Qwen3Config`, 19 `REPO = "Qwen/Qwen3-0.6B-Base"`); SmolLM2-134(base)/verify.py:13-19 is structurally identical with `from model_full import SmolLM2ForCausalLM, SmolLM2Config` at :16 and `REPO = "HuggingFaceTB/SmolLM2-135M"` at :19. Both scripts are pure @torch.no_grad() forward passes (verify.py:47 / :49), so seed absence is inconsequential for the CPU number — but see the refuted GPU fact: comparison_with_hf.md attributes the non-zero GPU deltas precisely to unpinned backend dispatch, which is what a determinism flag would have controlled.
```


### 5.G Gaps — not determinable from disk

- cuDNN version (live: 91900) and NVIDIA driver version (live: 580.142) are recorded NOWHERE in the repo. I grepped all *.md/*.json/*.toml/*.txt/*.log for 'cuDNN|cudnn_version|580.14|CUDA 13|CUDA 12' and the only hit was an unrelated prose mention in jax_vs_pytorch_tradeoffs.md:44. Version drift in these two cannot be detected from disk.
- The exact torch/transformers versions under which the Qwen3 verify.json (2026-06-08) and original_vs_repro.txt (2026-06-09) were produced are undeterminable. Qwen3-0.6B has no requirements.txt/pyproject.toml, no script stamps torch.__version__, verify.json has no env fields, and no ledger entry covers those runs. The only weak signal is original_eval_run2.log:1's torch_dtype deprecation warning, which bounds transformers from below but names no version.
- The exact transformers version behind SmolLM2's parity.log (2026-05-13) is likewise unstamped. parity.log:1's deprecation warning is consistent with the pinned transformers==5.8.0 but does not prove it. The sibling results.ipynb (same day) stamps torch 2.11.0+cu130 / python 3.12.11, which is strong circumstantial evidence for the same session, but parity.log itself carries no stamp.
- No GPU-side parity number exists for either model. compare_with_hf.py is written to run parity on cuda and to save results/comparison_with_hf.json (compare_with_hf.py:259-260), but that JSON is absent from disk (only comparison_with_hf.md is present). Whether the bit-exact 0.0 survives on GB10 GPU, under default TF32 settings, is untested and unrecorded.
- The SmolLM2 15.371 PPL and the Qwen3 13.400 PPL are on DIFFERENT corpora with DIFFERENT window semantics, dtypes, and devices (wikitext-2-raw-v1 / overlapping stride-512 / fp32 vs FineWeb-Edu / non-overlapping / bf16). Nothing on disk makes them comparable, and no single command recomputes both.
- There is no HuggingFace model-card file, no MODEL_CARD.md, and no exported HF repo for either reproduction anywhere under the repo root — so there is no existing Reproduce section on disk to check these commands against. (SmolLM2-134(base)/scripts/export_to_hf.py exists as an exporter, but results/lm_eval/ and any exported dir are absent.)
- No lockfile exists anywhere (find for *.lock returned only research/loop_state.json.lock, .claude/scheduled_tasks.lock, research/ledger/ledger.json.lock — all mutex files, not dependency locks). pyproject.toml:6-8 acknowledges this and gives the uv/pip-compile command to create one, but it was never run.

---

## 6. Training details (the 1.19B-token runs)

<sub>Audit dimension: Training details (from-scratch Qwen3-0.6B runs behind the ~1.19B-token figure, + SmolLM2)</sub>

### 6.1 Which concrete runs are behind the ~1.19B-token figure?

**Value**

```
Four 'Phase B' runs at 18,150 steps x 65,536 tok/step = 1,189,478,400 tokens each, launched sequentially by phase_b_driver.sh: (1) faithful baseline (train_qwen3.py, run_name=baseline2tpp), (2) IMU-1/NorMuon modernized (train_imu1.py, imu1_2tpp), (3) partial-RoPE 0.25 (train_partialrope.py, prope25_2tpp), (4) partial-RoPE 0.10 (prope10_2tpp). Run 4 died incomplete at step ~5450/18150.
```

**Evidence** — `Qwen3-0.6B/builds/phase_b_driver.sh:2-34`

**Source quote**

```
# Phase B — 4 matched-compute runs @ 2 TPP (18,150 steps = 1.19B tokens each).
# Sequential (one GPU job at a time on the GB10). Best LR from Phase A = 2.4e-3.
S=18150; W=900; COMMON="--eval_every 2000 --ckpt_every 2000 --log_every 50"
cd "$FAITHFUL" && python train_qwen3.py --steps $S --peak_lr 2.4e-3 --end_lr 3.2e-4 \
  --warmup_steps $W $COMMON --run_name baseline2tpp
```

**Confidence** — measured from code

**Caveat** — prope10 is incomplete — the comparison README itself says 'died incomplete at step 5450/18150 (~30%)' (Qwen3-0.6B/builds/comparison/README.md:10).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Verified verbatim. Qwen3-0.6B/builds/phase_b_driver.sh:2 ('Phase B — 4 matched-compute runs @ 2 TPP (18,150 steps = 1.19B tokens each).'), :3 (best LR 2.4e-3), :14 ('S=18150; W=900; COMMON=...'), :19-20 (faithful/baseline2tpp), :24 (train_imu1.py --run_name imu1_2tpp), :28-29 (--partial_rotary_factor 0.25 prope25_2tpp), :33-34 (0.10 prope10_2tpp). 18150*65536 = 1,189,478,400 exactly. prope10 incompleteness independently confirmed: qwen3_prope10_2tpp_train.log has 116 lines and its LAST line is '[22:04:32] step 5450/18150 ... tok/s 7,096'. Qwen3-0.6B/builds/comparison/README.md:10 quote is verbatim.
```


### 6.2 GPU model and count

**Value**

```
1 x NVIDIA GB10 (Grace Blackwell), unified ~119 GB CPU+GPU pool, no separate VRAM. Device string is measured (torch.cuda.get_device_name(0)). Count = 1 is NOT recorded as a measured value anywhere on disk — no file contains torch.cuda.device_count() output.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/throughput_probe.json:2`

**Source quote**

```
"device": "NVIDIA GB10",
```

**Confidence** — measured from code

**Caveat** — Single-GPU-ness is asserted in prose only: CLAUDE.md:3 ('single-box ML research repo on an NVIDIA GB10 ... unified ~119 GB CPU+GPU memory pool') and CLAUDE.md:9 ('ONE GPU job at a time on the GB10'). safe_cuda.guard(device=0) and get_device_name(0) only prove device 0 exists. Grep for 'device_count' across all .py/.json/.md returned zero hits.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
throughput_probe.json:2 is verbatim ('"device": "NVIDIA GB10",'). I re-ran the device_count grep repo-wide excluding .git: ZERO hits, so the 'count is not measured' claim survives. CLAUDE.md:3 and CLAUDE.md:9 quotes verified verbatim. The honest MEASURED/PROSE split here is correct and should be preserved on the card.
```


### 6.3 Wall-clock hours — faithful baseline (the 1.19B headline run)

**Value**

```
2,663.1 minutes = 44.4 h of training-loop time (excludes the ~26 min corpus stream+tokenize and the baseline eval). Sustained ~7,444-7,481 tok/s.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.log:395`

**Source quote**

```
[18:05:05] Training complete in 2663.1 min.
```

**Confidence** — measured from code

**Caveat** — The corpus stream/tokenize step took an extra 1573.8 s (~26 min), logged separately at line 6 of the same file.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
2,663.1 min = 44.4 h (confirmed). Throughput sub-claim should read: cumulative tok/s ranged 7,414–7,483 over the run, ending at 7,444 — not '7,444–7,481'.
```

**Verifier note**

```
Hours CONFIRMED: qwen3_baseline2tpp_train.log:395 is verbatim '[18:05:05] Training complete in 2663.1 min.' 2663.1 min = 44.385 h. Scope confirmed by arithmetic: training started at :12 '[21:41:59] Training to 18,150 steps' and 21:41:59 + 2663.1 min lands exactly on 18:05:05, so the timer excludes both the stream (:6, 1573.8 s = 26.2 min) and the baseline eval (:11). REFUTED sub-claim: the max cumulative tok/s in the log is 7,483 (line 41, step 1450), not 7,481; the min is 7,414. The stated '7,444-7,481' band is wrong at both ends.
```


### 6.4 Wall-clock hours — IMU-1/NorMuon and partial-RoPE arms

**Value**

```
IMU-1 (imu1_2tpp): ~63.9 h DERIVED (no 'Training complete' line; final logged rate 5,172 tok/s, 1,189,478,400/5172 = 229,984 s). partial-RoPE 0.25: ~46.1 h DERIVED (final rate 7,168 tok/s). partial-RoPE 0.10: ~14 h before it stopped at step 5450 (7,096-7,104 tok/s).
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log:114-115`

**Source quote**

```
[09:58:25] step 18150/18150  ce 2.4688  z 0.00659  lr 0.00e+00  |grad| 0.06  mem 66.1GB  tok/s 5,172
[09:58:25] DONE
```

**Confidence** — measured from code

**Caveat** — DERIVED, not logged: train_imu1.py and train_partialrope.py write no wall-clock summary line. The IMU-1 derivation is corroborated by checkpoint mtimes (step2000 Jun 12 02:10 -> step18000 Jun 14 10:27 = 56.28 h for 16,000 steps = 5,175 tok/s). NorMuon is ~31% slower per token than AdamW at the same shape.

**Verdict — ❌ WRONG**

**Corrected value**

```
Hours are right (IMU-1 ~63.9 h, pRoPE-0.25 ~46.1 h, pRoPE-0.10 ~14 h) but TWO errors must be fixed: (a) the evidence citation is qwen3_imu1_2tpp_train.log:385-386, NOT :114-115; (b) NorMuon is 30.5% LOWER THROUGHPUT, which is +43.9% wall-clock / +43.9% time per token — not '~31% slower per token'.
```

**Verifier note**

```
CITATION REFUTED: the file is 386 lines. Lines 114-115 are '[12:05:16] step 5100/18150 ... tok/s 5,160' and '[12:15:44] step 5150/18150 ... tok/s 5,160'. The quoted 'step 18150/18150 ... tok/s 5,172' + 'DONE' pair is at lines 385-386 (verified by grep -n). ARITHMETIC REFUTED: 5,172/7,444 = 0.6948, i.e. 30.5% lower throughput; the reciprocal is 1.439, so per-token time and wall-clock are 43.9% HIGHER. The fact's own derived hours prove it: 63.9 h / 44.4 h = 1.439. Derivations themselves check out: 1,189,478,400/5172 = 229,984 s = 63.88 h; checkpoint mtimes step2000 2026-06-12 02:10 -> step18000 2026-06-14 10:27 = 56.28 h for 16,000 steps = 5,175 tok/s. pRoPE-0.25 log clock 09:58:33 -> 08:05:04 next-next day = 46.11 h, final rate 7,168 (verified in tail). pRoPE-0.10 08:05:09 -> 22:04:32 same day = 13.99 h, rates 7,096-7,104 (verified in tail).
```


### 6.5 GPU-hours / cost recorded in the ledger for these runs

**Value**

```
NOT_FOUND — the four Phase-B training runs have NO ledger run entries at all. Only their downstream EVAL runs are in the ledger, and those carry cost.wall_clock_min=null, cost.gpu_hours=null.
```

**Evidence** — `research/ledger/ledger.json:451`

**Source quote**

```
"artifact_sha256": "checkpoint_qwen3_baseline2tpp.pt@step18150"
```

**Confidence** — NOT FOUND

**Caveat** — I enumerated all 29 ledger run_ids; the earliest is 2026-06-16_qwen3-faithful_eval-first (type=eval). The training runs themselves (Jun 9-16) predate ledger adoption. The only trace of baseline2tpp in ledger.json is the lineage artifact string above.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Independently re-enumerated all 29 ledger runs via ledger.json: earliest is 2026-06-16_qwen3-faithful_eval-first (type=eval), confirming the Phase-B trainers predate ledger adoption. grep for baseline2tpp|imu1_2tpp|prope25_2tpp|prope10_2tpp across ledger.json returns exactly ONE hit — research/ledger/ledger.json:451 '"artifact_sha256": "checkpoint_qwen3_baseline2tpp.pt@step18150"' — verbatim as quoted. All five 2026-06-16 eval entries carry cost.wall_clock_min=null and cost.gpu_hours=null. (Side note, not a refutation: 3 unrelated later runs DO carry costs — 2026-06-30_qwen3-0.6b_midtrain-anneal 2220 min / 36.7 gpu_h; 2026-07-19_hybrid-ssm-0.2b_pretrain-ssm-base-s0 990 / 16.5. So the null is specific to these runs, not a repo-wide absence.)
```


### 6.6 Sequence length, micro-batch, grad-accum, global batch (sequences and tokens)

**Value**

```
seq_len 4096; micro_batch 4; grad_accum 4; global batch = 16 sequences = 65,536 tokens/step. Identical across all four Phase-B arms.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:234-236`

**Source quote**

```
ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--micro_batch", type=int, default=4, help="DO NOT raise; >=8 OOMs at seq 4096 (probe-verified)")
    ap.add_argument("--grad_accum", type=int, default=4, help="effective batch = micro_batch * grad_accum seqs")
```

**Confidence** — measured from code

**Caveat** — Confirmed in the run log itself: 'tok/step=65,536  steps=18,150  token_budget=1,189,478,400' (qwen3_baseline2tpp_train.log:3). train_imu1.py:93-95 and train_partialrope.py:37-39 carry the same 4096/4/4 defaults.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_qwen3.py:234-236 quote is verbatim at exactly those line numbers. Cross-arm identity independently verified: train_imu1.py:93 (--seq_len 4096), :94 (--micro_batch 4), :95 (--grad_accum 4); train_partialrope.py:37/:38/:39 same. Log line 3 verbatim '[21:14:18] tok/step=65,536  steps=18,150  token_budget=1,189,478,400'. Derivation in code at train_qwen3.py:275 'tok_per_step = args.seq_len * args.micro_batch * args.grad_accum'.
```


### 6.7 Precision: bf16/fp16/fp32, autocast, master weights

**Value**

```
FULL bf16 — model weights cast to torch.bfloat16 at construction; there is NO torch.autocast, NO GradScaler, and NO fp32 master-weight copy in any of the three trainers. AdamW moment buffers are therefore allocated in the parameter dtype (bf16). Only cross-entropy is upcast: each 8192-row chunk is .float()'d into an fp32 accumulator.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:293`

**Source quote**

```
model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
```

**Confidence** — measured from code

**Caveat** — Verified by grep: 'autocast|GradScaler|master' returns zero hits in train_qwen3.py, train_imu1.py, train_partialrope.py. The fp32 CE path is train_qwen3.py:87 ('total = flat.new_zeros((), dtype=torch.float32)') and :91 ('flat[i:i + chunk].float()'). NOTE train_imu1.py:38-45 does NOT .float() its CE chunks (only its z-loss does), so the IMU-1 arm's CE is accumulated in bf16 — a real, unremarked difference between arms.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
All file-level claims CONFIRMED. Qualify one clause: 'AdamW moment buffers are allocated in bf16' is an INFERENCE from PyTorch's zeros_like(p) semantics — no file on disk records optimizer-state dtype. State it as inferred, not measured.
```

**Verifier note**

```
train_qwen3.py:293 verbatim. I re-ran the grep for 'autocast|GradScaler|master' across all three trainers: exit 1 (zero hits) on each — confirmed. fp32 CE path verified at train_qwen3.py:87 ('total = flat.new_zeros((), dtype=torch.float32)') and :91 ('flat[i:i + chunk].float()'). The IMU-1 asymmetry is REAL and correctly reported: train_imu1.py:41 initialises 'total, n = 0.0, flat_tgt.numel()' and :43-44 calls cross_entropy on flat_logits WITHOUT .float(), while chunked_z_loss:54 does use '.float()'. This is a genuine unremarked between-arm numerical difference and belongs on the card.
```


### 6.8 Optimizer / LR / betas / eps / weight decay / warmup / schedule / grad clip — faithful baseline

**Value**

```
AdamW, peak_lr 2.4e-3 (Phase-A winner; script default is 1.7e-3), end_lr 3.2e-4, betas (0.9, 0.95), eps 1e-8, weight_decay 0.01 applied ONLY to params with dim>=2 (dim<2 gets 0.0), warmup 900 steps linear, then cosine decay from peak to end_lr (floor = end_lr/peak_lr = 0.1333), grad_clip 1.0 (global L2 via clip_grad_norm_).
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:296-304`

**Source quote**

```
decay, no_decay = [], []
    for _, p in model.named_parameters():
        (no_decay if p.dim() < 2 else decay).append(p)
    optim = AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )
    sched = make_cosine_scheduler(optim, args.warmup_steps, args.steps, args.peak_lr, args.end_lr)
```

**Confidence** — measured from code

**Caveat** — The run's own resolved args are echoed at qwen3_baseline2tpp_train.log:2: "'peak_lr': 0.0024, 'end_lr': 0.00032, 'warmup_steps': 900, 'weight_decay': 0.01, 'grad_clip': 1.0". Cosine shape at train_qwen3.py:96-107: floor + 0.5*(1-floor)*(1+cos(pi*prog)). The file's own docstring (line 6) still claims 'peak 1.7e-3' — stale vs the actual 1.19B run, which used 2.4e-3.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_qwen3.py:296-304 quote is verbatim at exactly those lines. Script default peak_lr=1.7e-3 confirmed at :237; driver overrides to 2.4e-3 at phase_b_driver.sh:19. Resolved args echoed verbatim at log line 2. Cosine shape verified at :96-107 ('floor = end_lr / peak_lr'; 'floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * prog))'); 3.2e-4/2.4e-3 = 0.13333. grad_clip via torch.nn.utils.clip_grad_norm_ confirmed at :391. The stale-docstring catch is real: train_qwen3.py:6 still reads 'cosine schedule, peak 1.7e-3 -> end 3.2e-4'.
```


### 6.9 Optimizer for the IMU-1 / NorMuon 1.19B arm (the arm that won)

**Value**

```
Hybrid split: 2D non-embedding matrices (224 params) -> NorMuon(lr=0.011, weight_decay=0.1, beta1=0.95, beta2=0.95); embeddings/norms/1D (198 params) -> AdamW(lr=0.006, betas=(0.9,0.95), eps=1e-8, weight_decay=0.0). Schedule = WSD (linear warmup 900, stable, then linear decay-to-ZERO over the final 20%), not cosine. grad_clip 1.0. Extra chunked z-loss weighted 1e-4 added to CE.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/train_imu1.py:82-87`

**Source quote**

```
def build_optimizers(model, normuon_lr, adam_lr, wd):
    n_params, a_params = split_params(model)
    opt_n = NorMuon(n_params, lr=normuon_lr, weight_decay=wd, beta1=0.95, beta2=0.95)
    opt_a = torch.optim.AdamW(a_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
```

**Confidence** — measured from code

**Caveat** — CONFOUND for any 'IMU-1 wins' claim: this arm changes optimizer AND schedule shape (WSD vs cosine) AND adds z-loss AND changes the model (value residuals + LayerNorm scaling + head gating, train_imu1.py:1-8) AND weight_decay 0.1 vs 0.01 — five variables at once, violating the repo's own one-variable rule. Param split (224/198) measured at qwen3_imu1_2tpp_train.log:2.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_imu1.py:82-85 quote verbatim at those lines. Defaults confirmed: :96 normuon_lr 0.011, :97 adam_lr 0.006, :98 weight_decay 0.1 ('# 2D only (NorMuon)'), :101 z_weight 1e-4, :102 grad_clip 1.0. WSD implementation at :59-66 (linear warmup, stable, 'max(0.0, (total - step) / max(1.0, total - decay_start))' with decay_frac 0.2). Param split 224/198 verbatim at qwen3_imu1_2tpp_train.log:2. The 5-variable-confound caveat is correct and load-bearing. ADD A SECOND CAVEAT the fact omits: the NorMuon advantage was later NULLED — ledger run 2026-07-05_qwen3-0.6b_scaling-persistence carries verdict 'null' and metrics.conclusion 'The NorMuon advantage CONVERGES toward 0 with budget — an early-training speedup'. Calling this 'the arm that won' without that is misleading on a public card.
```


### 6.10 Total steps, total tokens, seeds

**Value**

```
18,150 steps; 1,189,478,400 tokens (18150 x 65,536); seed = 0 for ALL arms. n=1 per arm — no seed replicates exist for any 1.19B run.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.log:1-3`

**Source quote**

```
[21:14:18] device=cuda dtype=torch.bfloat16 seed=0
[21:14:18] args={'steps': 18150, 'seq_len': 4096, 'micro_batch': 4, 'grad_accum': 4, 'peak_lr': 0.0024, 'end_lr': 0.00032, 'warmup_steps': 900, 'weight_decay': 0.01, 'grad_clip': 1.0, 'mem_fraction': 0.85, 'seed': 0, 'dtype': 'bfloat16', ...}
[21:14:18] tok/step=65,536  steps=18,150  token_budget=1,189,478,400
```

**Confidence** — measured from code

**Caveat** — Seed 0 is the argparse default in all three trainers (train_qwen3.py:243, train_imu1.py:104, train_partialrope.py:47) and phase_b_driver.sh never passes --seed. Every 1.19B headline number is therefore n=1, single seed, no CI.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Log lines 1-3 quoted verbatim and verified. Seed defaults confirmed at train_qwen3.py:243, train_imu1.py:104, train_partialrope.py:47 (all '--seed', default=0); phase_b_driver.sh contains no '--seed' anywhere (read the full 37-line file). The 'n=1, single seed, no CI' framing is the honest one and must survive to the card.
```


### 6.11 The ACTUAL training corpus behind the 1.19B tokens — HF dataset id, config, split

**Value**

```
HuggingFaceFW/fineweb-edu, config 'sample-10BT', split 'train', streaming=True. Streamed once by the faithful run and cached to results/tokcache_1191478400_300000.pt; the other three arms LOADED THAT SAME CACHE (identical corpus, identical stream order).
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:151`

**Source quote**

```
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
```

**Confidence** — measured from code

**Caveat** — No revision/sha is pinned — load_dataset is called without a revision arg, so the exact snapshot is unrecoverable. Cache reuse proven at qwen3_imu1_2tpp_train.log:3, qwen3_prope25_2tpp_train.log:2, qwen3_prope10_2tpp_train.log:2: 'loaded cached tokens from tokcache_1191478400_300000.pt (1,191,478,400 train + 300,000 val)'. There is NO prepare_*.py and NO dataset card under research/datasets/ for this corpus — that directory holds only dclm-edu, openr1-math-220k, math-eval-v1, grpo-math-prompts-v1 and hybridssm-fineweb-edu, none of which fed the 1.19B Qwen3 runs.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_qwen3.py:151 verbatim at that exact line. CRITICALLY, I also confirmed the load_dataset line is UNCHANGED in the pre-86e79f3 version that actually ran (it appears as unmodified context in `git diff e791875 86e79f3`), so the dataset id/config/split claim holds for the runs, not just for today's file. No revision= arg — confirmed by reading the call. Cache reuse verified verbatim at qwen3_imu1_2tpp_train.log:3, qwen3_prope25_2tpp_train.log:2, qwen3_prope10_2tpp_train.log:2. Cache file exists on disk: results/tokcache_1191478400_300000.pt, 9,534,229,373 bytes, mtime 2026-06-09 22:41. research/datasets/ listing verified: exactly data-selection-dclm-edu, grpo-math-prompts-v1, hybridssm-fineweb-edu, math-eval-v1, math-reasoning-openr1-math-220k — none is this corpus.
```


### 6.12 How the 1.19B corpus was tokenized and packed

**Value**

```
Tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base') (vocab 151,936). Each doc encoded with add_special_tokens=False, one EOS id appended, all ids concatenated into one flat stream. Packing = contiguous non-overlapping 4096-token windows (window i = tokens[i*4096 : i*4096+4097], input=[:-1], label=[1:]). No cross-document attention masking — documents bleed across window boundaries. DataLoader shuffles WINDOW order (shuffle=True, drop_last=True).
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:110-123`

**Source quote**

```
class PackedTextDataset(torch.utils.data.Dataset):
    """Pre-tokenized contiguous windows of seq_len (mirrors SmolLM2 train.py)."""
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.tokens = token_ids
        self.seq_len = seq_len
        self.n_windows = (len(token_ids) - 1) // seq_len
```

**Confidence** — measured from code

**Caveat** — 290,888 windows were available and 18150*16 = 290,400 were consumed = 99.83% of exactly one epoch (single-pass, effectively no repetition). Log line 7: '290,888 train windows of 4096'. REPO id at train_qwen3.py:59, tokenizer load at :281, doc+EOS at :166.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
train_qwen3.py:110-123 quote verbatim at exactly those lines. REPO='Qwen/Qwen3-0.6B-Base' at :59, tokenizer load at :281, doc+EOS at :166, DataLoader(shuffle=True, drop_last=True) at :287. Epoch arithmetic checks: log line 7 '290,888 train windows of 4096'; 18150 steps x 16 seqs = 290,400 = 99.83% of one epoch, so effectively single-pass. The old (actually-executed) code did the same encode: `git show e791875` line 141 '(buf if len(buf) < n_train else val).extend(ids + [eos])' after `tokenizer.encode(text, add_special_tokens=False)`.
```


### 6.13 Was the train/val split for the 1.19B runs document-disjoint / decontaminated?

**Value**

```
NO. The 1.19B runs used the OLD stream_tokens, which cut the stream by token count so val was the sequential continuation of train. The document-disjoint + 13-gram-decontam version now on disk landed in commit 86e79f3 (2026-06-16), AFTER all four Phase-B runs finished.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:131-136`

**Source quote**

```
* each whole document is routed to train or val by a seeded hash
        (`is_val_doc`), so train/val never share a document and no document spans
        the boundary (the old code cut the stream by token count — val was the
        sequential continuation of train, leak-suspect);
```

**Confidence** — measured from code

**Caveat** — Proof of ordering: (a) `git diff e791875 86e79f3 -- .../train_qwen3.py` shows the old body was `(buf if len(buf) < n_train else val).extend(ids + [eos])` with cache key `tokcache_{n_train}_{n_val}.pt`; (b) the cache the runs loaded is named `tokcache_1191478400_300000.pt` — the OLD key, no seed/tokenizer tag; (c) baseline2tpp log line 6 has no decontam sentence, unlike the current code's log call. So the 28.65 / 23.52 / 29.54 val-PPL headlines rest on a leak-suspect split by the repo's own later admission. The file you read today is NOT the file that produced those numbers.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
This is the strongest fact in the set and I could not break it. train_qwen3.py:131-136 quote verbatim. `git diff e791875 86e79f3` shows the removed lines '-    cache = RESULTS / f"tokcache_{n_train}_{n_val}.pt"' and '-        (buf if len(buf) < n_train else val).extend(ids + [eos])', replaced by the seeded is_val_doc routing + decontaminate_val + the seed/tokenizer-tagged cache key at :145. The cache the runs loaded is literally named tokcache_1191478400_300000.pt (old key, no seed tag) — on disk, verified. Commit timestamp 86e79f3 = 2026-06-16 21:57:33 +0000, after prope10 stopped. Corroborating detail the fact could have added: results/decontam_report.json DOES exist but its mtime is 2026-07-07 15:13, matching tokcache_422020224_300000_seed0_Qwen3-0.6B-Base.pt (same mtime) — i.e. it documents a LATER cache, not the 1.19B one, exactly as the fact asserts.
```


### 6.14 Throughput (tokens/sec) for the 1.19B runs

**Value**

```
faithful baseline 7,444-7,481 tok/s sustained (final 7,444); IMU-1/NorMuon 5,172 tok/s; partial-RoPE 0.25 7,168 tok/s; partial-RoPE 0.10 ~7,100 tok/s. Peak memory: 52.4 GB (faithful), 66.1 GB (IMU-1), 54.3 GB (partial-RoPE).
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.log:394`

**Source quote**

```
[18:05:01] step 18150/18150  loss 3.2811  lr 3.20e-04  |grad| 0.13  mem 52.4GB  tok/s  7,444  tok 1189.5M  ETA 0.0min
```

**Confidence** — measured from code

**Caveat** — These are CUMULATIVE averages (tok_seen / elapsed since t0), not instantaneous rates — train_qwen3.py:404-405 'tps = tok_seen / max(1e-9, dt)'. An independent standalone probe measured 7,167.4 tok/s compiled vs 3,787.5 uncompiled at the same shape (throughput_probe.json:26 and :12), i.e. torch.compile gives 1.89x.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
faithful cumulative tok/s spans 7,414–7,483 (max at log line 41, step 1450), ending 7,444 — the stated 7,444–7,481 band is wrong at both ends. All other figures confirmed.
```

**Verifier note**

```
Log line 394 quote verbatim (mem 52.4GB, tok/s 7,444). IMU-1 5,172 + mem 66.1GB confirmed at log:385; pRoPE-0.25 7,168 + 54.3GB at its tail; pRoPE-0.10 7,096-7,104 + 54.3GB at its tail. Cumulative-not-instantaneous caveat CONFIRMED verbatim at train_qwen3.py:404-405 ('dt = time.time() - t0' / 'tps = tok_seen / max(1e-9, dt)'). Probe figures confirmed: throughput_probe.json:12 = 3787.5 (uncompiled), :26 = 7167.4 (compiled), :34 '"speedup_compile_vs_baseline": 1.89'.
```


### 6.15 MFU for the 1.19B runs

**Value**

```
NOT_FOUND. No MFU number was ever computed or stored for these runs. mfu_meter.py exists but has no JSON output anywhere in the repo (find '*mfu*' returns only the module, its pycache, and its unit test). No README claims an MFU for the three builds.
```

**Evidence** — `mfu_meter.py:63-66`

**Source quote**

```
"gb10": {"bf16_dense_tflops": 125.0, "estimated": True,
                     "GB10; BF16-dense derived (~FP4/8) — treat MFU as approximate"},
```

**Confidence** — NOT FOUND

**Caveat** — Even if computed post-hoc, the GB10 peak entry is estimated=True (NVIDIA publishes only a sparse-FP4 ~1 PFLOP figure), so per CLAUDE.md:19 a GB10 MFU must never be quoted as exact.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Scoped claim survives: no MFU exists for the four 1.19B Phase-B runs. But 'no JSON output anywhere in the repo' is FALSE, and the companion gap line 'MFU was never computed for any Qwen3 run' is FLATLY WRONG.
```

**Verifier note**

```
mfu_meter.py:63-66 quote is verbatim. But the find '*mfu*' filename search was the wrong instrument — MFU output is embedded inside other JSONs. `grep -rln '\"mfu\"' --include=*.json` returns TWO real hits: (1) research/ledger/ledger.json:503 '\"mfu\": 0.2909' inside run 2026-06-16_qwen3_normuon-vs-adamw, alongside mfu_normuon 0.2907, achieved_tflops 36.36, device_peak_tflops 125.0, peak_is_estimated true; (2) Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/c5_evidence.json:11-14, a full block with mfu 0.3209, hfu 0.3209, achieved_tflops 40.11, tokens_per_sec 7344, formula 'MFU = (6N + 12*L*H*Q*T) * tok/s / peak_bf16_dense'. Neither is a 1.19B run, so the narrow claim holds — but the card must not say the repo has never computed MFU. The estimated=True caveat is correct and mandatory (CLAUDE.md:19 verified).
```


### 6.16 How was the reported val PPL (28.65 etc.) measured?

**Value**

```
In-trainer evaluate(): non-overlapping windows of 4096 tokens (stride = seq_len = 4096), capped at max_windows=50 => 204,800 target tokens, chunked fp32 CE, exp(mean NLL), on the FineWeb-Edu val slice with the model's own Qwen3 tokenizer. NOT the eval-harness.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:194-207`

**Source quote**

```
def evaluate(model, val_tokens, device, seq_len: int, max_windows: int = 50):
    for begin in range(0, min(len(val_tokens) - seq_len, max_windows * seq_len), seq_len):
        ids = val_tokens[begin:begin + seq_len + 1].unsqueeze(0).to(device)
        logits = model(ids[:, :-1])["logits"]
```

**Confidence** — measured from code

**Caveat** — Log line 342-equivalent confirms the count: 'baseline val PPL=185810.49 (204,800 tokens)'. These in-trainer PPLs are not cross-run-comparable under the repo's own §C10; the comparable numbers are the separate text-lm-v2 eval-harness ledger entries: faithful wikitext2 ppl 37.01 / bpb 1.2256, modernized 27.8, prope25 38.08, prope10 69.63 — the last from a step-4000 checkpoint of the incomplete run (ledger note: 'run stopped early at ~step 5400 (undertrained vs 18150-step peers)').

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Substance confirmed, but three fixes: (a) 'def evaluate' is at train_qwen3.py:195, not 194; (b) the 204,800-token line is qwen3_baseline2tpp_train.log:11, not 'line 342-equivalent'; (c) the modernized 27.8 / 23.52 were scored on checkpoint_imu1_2tpp_step18000.pt — step 18,000 (1.180B tok), NOT the step-18150 endpoint, so it is 0.8% short of iso-token vs the faithful number.
```

**Verifier note**

```
evaluate() body verified: :199 'for begin in range(0, min(len(val_tokens) - seq_len, max_windows * seq_len), seq_len)' — stride == seq_len, 50*4096 = 204,800. fp32 CE via chunked_cross_entropy:87/:91. Log line 11 verbatim: '[21:41:59]   baseline val PPL=185810.49 (204,800 tokens) — expect ~vocab_size (151,936) at init' (grep confirms it is the ONLY '(204,800 tokens)' line in the file). All four ledger eval numbers verified exactly: 37.0101/1.2256, 27.8, 38.08, 69.63, all suite_version 'text-lm-v2'. prope10 note verbatim: 'step-4000 checkpoint; run stopped early at ~step 5400 (undertrained vs 18150-step peers)'. Checkpoint (c) verified at Qwen3-0.6B/experiments/2026-06-16_qwen3-0.6b_eval-modernized/eval_suite.py:55; no step-18150 IMU-1 checkpoint exists on disk (last '[ckpt @ 18000]' at imu1 log:382). Dataset-config check (the flagged failure mode) PASSES: eval_suite.py:162-163 says load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation", revision=WIKITEXT_REV) — the '-raw-' variant is genuinely what the code names.
```


### 6.17 Was the 1.19B-token budget iso-FLOP across arms?

**Value**

```
Iso-TOKEN, not iso-FLOP or iso-wall-clock. All four arms ran the same 18,150 steps / 1,189,478,400 tokens, but IMU-1 took ~31% more wall-clock (5,172 vs 7,444 tok/s) and adds a z-loss plus three architecture changes.
```

**Evidence** — `Qwen3-0.6B/README.md:80`

**Source quote**

```
matched **tokens** (1.19B); IMU-1 also ran NorMuon (~30% more wall-clock, uncounted by the 6ND
```

**Confidence** — PROSE ONLY

**Caveat** — The prose claim matches the tok/s I read in the logs (7,444 vs 5,172 = 30.5% slower), so it is corroborated by measurement — but no train_flops artifact exists for these runs, so the §C18 '<=5% train_flops match' gate was never actually evaluated on disk.

**Verdict — ❌ WRONG**

**Corrected value**

```
IMU-1 took ~44% more wall-clock (~63.9 h vs 44.4 h), not ~31%. 30.5% is the tok/s DROP; the wall-clock inflation is its reciprocal, 1/0.695 = 1.439.
```

**Verifier note**

```
Qwen3-0.6B/README.md:80 quote is verbatim, but the fact then endorses the prose as 'corroborated by measurement' — it is not. 7,444 vs 5,172 tok/s is a 30.5% throughput reduction, which is +43.9% wall-clock, and the fact set's OWN derived hours (63.9 h vs 44.4 h = 1.439x) prove it. The repo's README:80 prose ('~30% more wall-clock') is itself the original error and must not be copied onto a card. Two further points: (a) the iso-TOKEN-not-iso-FLOP core claim is CONFIRMED — grep for 'train_flops' across Qwen3-0.6B/builds/ returns zero files, so the §C18 <=5% gate was never evaluated for these four runs; (b) the fact omits README.md:81, which does assert 'params are iso-FLOP at 1.00043' — a parameter-count-based claim, not a train_flops artifact, and it should be characterised as such rather than ignored.
```


### 6.18 Chinchilla ratio / degree of under-training

**Value**

```
~2 tokens per parameter (1.19B tokens / 596M params). The repo states Chinchilla-optimal for 596M is ~12B tokens, so Phase B is ~10x under-trained, and ~30,000x less data than the real Qwen3 run (36T).
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/README.md:20`

**Source quote**

```
- **The unavoidable deviation = the token budget.** The paper used ~36T tokens; we use 131M (Phase A) to 1.19B (Phase B). ... Chinchilla-optimal for 596M is ~12B tokens (20 tok/param); even Phase B is ~10× under-trained.
```

**Confidence** — PROSE ONLY

**Caveat** — '36T tokens' is COPIED from the Qwen3 paper; the 'original PPL 13.40' comparator is a measured eval of the released HF model, not a run this repo trained. The 596M param count is measured (throughput_probe.json:3 'model_params': '596M').

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
faithful README.md:20 quote verbatim, including '~36T tokens', 'we use 131M (Phase A) to 1.19B (Phase B)' and 'Chinchilla-optimal for 596M is ~12B tokens (20 tok/param); even Phase B is ~10x under-trained'. The '~30,000x' is DERIVED arithmetic (36T/1.19B = 30,252) but is independently corroborated in prose at research/brutal_scorecard.md:57 and research/brutal_scorecard_core.md:35 ('~30,000x under-trained'). The COPIED-vs-MEASURED split is right: 36T is from the Qwen3 report; 13.40 is genuinely MEASURED on this box — Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_vs_repro.txt:2 'ORIGINAL  Qwen3-0.6B-Base (36T tok)   val PPL =   13.400  (204,800 tok, 21s)'. 596M confirmed at throughput_probe.json:3. (The '275,000x' at Qwen3-0.6B/README.md:74 is the Phase-A comparison, 36T/131M — do not conflate the two on the card.)
```


### 6.19 The other Qwen3 from-scratch family: the scaling-persistence ladder

**Value**

```
Separate runs at the SAME 596M params but SMALLER token budgets: 2,564 steps = 168M tokens (3 seeds/arm) and 6,409 steps = 420M tokens (2-3 seeds/arm), arms = adamw vs normuon, same 4096/4/4 = 65,536 tok/step, bf16, peak_lr 2.4e-3 (AdamW) / normuon_lr 0.011, wd 0.1 on 2D for BOTH arms, warmup 50, cosine to 10% of peak, grad_clip 1.0, fixed data split seed 0 across all cells.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/train_ablation.py:51-52`

**Source quote**

```
SEQ_LEN, MICRO_BATCH, GRAD_ACCUM = 4096, 4, 4
WARMUP, END_RATIO, SPLIT_SEED = 50, 0.1, 0          # fixed data split across cells
```

**Confidence** — measured from code

**Caveat** — Cell list at Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/run_ladder.sh:49-53 ('2564 adamw 0' ... '6409 normuon 1'). The '168M'/'420M' in the run tags are TOKEN budgets (bM=steps*65536/1e6), NOT param counts — easy to misread. This ladder DOES carry 3 seeds/arm and per-cell sentinel logs; the 1.19B three-build runs do not.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Hyperparameters all confirmed, but the seed count must be stated precisely (420M is n=3 per arm FINAL, via a separate extension script) and the ladder's TERMINAL VERDICT — 'null', NorMuon's advantage CONVERGES to 0 with budget — is a material missing caveat.
```

**Verifier note**

```
train_ablation.py:51-52 quote verbatim. Defaults verified at :85 (--peak_lr 2.4e-3), :86 (--normuon_lr 0.011), :87 (--weight_decay 0.1 '# on 2D, BOTH arms (held equal)'), :88 (--grad_clip 1.0); END_RATIO 0.1 and WARMUP 50 at :52; cosine at :56-60. 2564*65536 = 168,034,304; 6409*65536 = 420,020,224 — the 'token budget not param count' warning is correct (run_ladder.sh:57 'budgetM=$(( steps * 65536 / 1000000 ))'). SEED PRECISION: run_ladder.sh:49 states '168M=2564 steps (3 seeds/arm), 420M=6409 steps (2 seeds/arm)' and CELLS at :50-53 list only s0/s1 at 6409; the third 420M seed came from run_ladder_scale_ext.sh:593-594 (CELLS=("6409 adamw 2" "6409 normuon 2")) under ledger run 2026-07-23_qwen3-0.6b_normuon-at-scale. Final ledger metrics for 2026-07-05_qwen3-0.6b_scaling-persistence: top_budget_seeds [3,3], trend_verdict_wikitext 'CONVERGES', significance_verdict 'null', c25_complete false, c25_missing_hard [log_rmse_r2, holdout_extrapolation_pctdev, bootstrap_forecast_ci]. Any card mentioning NorMuon must carry this.
```


### 6.20 SmolLM2 — is there a from-scratch or continued-pretrain config/run?

**Value**

```
Both exist, but neither is a real pretrain. (a) FROM-SCRATCH: train.py on Salesforce/wikitext config 'wikitext-103-raw-v1' split 'train', seq_len 2048, micro 2 x accum 8 = 16 seqs = 32,768 tok/step, AdamW betas (0.9,0.95) eps 1e-8, peak lr 3.0e-3, wd 0.01 (dim>=2 only), WSD warmup 20 / 20% linear decay-to-zero, grad_clip 1.0, bf16 — the only recorded run is a 150-step DEMO (~4.9M tokens). (b) CONTINUED-PRETRAIN: train_tinystories.py from official HF SmolLM2-135M weights.
```

**Evidence** — `SmolLM2-134(base)/train.py:101-116`

**Source quote**

```
ap.add_argument("--steps", type=int, default=200, help="total optimizer steps")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2, help="micro-batch (per accum step)")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3.0e-3, help="peak LR; paper §6")
```

**Confidence** — measured from code

**Caveat** — Corpus at train.py:78 ('load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")'); betas/eps at train.py:152. The from-scratch run's scale is recorded ONLY as 'Demo-run steps': '150  (warmup 20, decay 20%)' and 'Demo-run final loss': '6.288 (start 11.254, baseline ln(V)=10.803)' in SmolLM2-134(base)/results/summary.json. train.py:13-19 calls this a single-GPU starter, not a reproduction.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
All named hyperparameters and the dataset config are CONFIRMED. Qualify the token figure: '~4.9M tokens' is DERIVED (150 x 32,768) from argparse defaults — nothing on disk records the demo run's batch shape.
```

**Verifier note**

```
train.py:101-105 quote verbatim at those lines (:101 steps default 200, :102 seq_len 2048, :103 batch_size 2, :104 grad_accum 8, :105 lr 3.0e-3). Dataset config check (the flagged failure mode) PASSES: train.py:78 is literally load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train") — the '-raw-' variant and the 103 (not 2) are exactly as claimed. betas/eps at :152, wd split at :145-151, WSD at :154, warmup 20 at :106, wd 0.01 at :108, grad_clip 1.0 at :110, bf16 at :115. Demo scale corroborated beyond summary.json:11-12: results/loss_curve.csv has 151 lines (header + steps 0-149), step 0 loss 11.254480, step 149 loss 6.288341 — matching summary.json exactly. But that CSV records only step/loss/lr, so the token count remains an inference from defaults. 'single-GPU starter, not a reproduction' verified at train.py:13-18.
```


### 6.21 SmolLM2 continued-pretrain run details

**Value**

```
roneneldan/TinyStories (split 'train' for training, 'validation' for eval), 102,000,116 tokens packed, seq_len 1024, micro_batch 4, grad_accum 1 => 4,096 tok/step, 24,414 steps, 100,000,000-token budget, AdamW betas (0.9,0.95) eps 1e-8 peak_lr 3e-4, wd 0.01 (dim>=2 only), grad_clip 1.0, WSD warmup 200 / 20% decay, bf16, seed 0, 116.1 min wall-clock, ~14,356 tok/s. PPL 6.895 -> 3.790.
```

**Evidence** — `SmolLM2-134(base)/results/tinystories_train.log:1-17`

**Source quote**

```
[21:23:28] Device: cuda   dtype: torch.bfloat16
[21:23:28] Loading tokenizer + official SmolLM2-135M weights...
[21:25:09]   packed 102,000,116 train tokens in 97.9s
[21:25:09]   tok/step = 4,096   total_steps = 24,414
[21:25:16] Training 24,414 steps to 100,000,000 tokens...
```

**Confidence** — measured from code

**Caveat** — Starts from OFFICIAL HF weights (train_tinystories.py:2-3), so NOT a from-scratch result. Wall-clock at line 506 ('Training complete in 116.1 min.'), final PPL at line 508. Dataset ids at train_tinystories.py:154-155; hyperparams at :105-122 and :213-217. The 100M budget is ~1/4 epoch of TinyStories per the docstring.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
tinystories_train.log:1-17 quote verbatim (verified line-by-line: :1 device/bf16, :2 official weights, :5 'packed 102,000,116 train tokens in 97.9s', :8 'tok/step = 4,096   total_steps = 24,414', :17 'Training 24,414 steps to 100,000,000 tokens...'). Wall-clock at :506 'Training complete in 116.1 min.'; PPL at :508 'AFTER PPL = 3.790   (BEFORE was 6.895; improvement +3.105 = +45.0%)'; 14,356 tok/s at :505. Hyperparams verified at train_tinystories.py:105-122 and optimizer at :213-217; dataset ids at :154-155. Two nits worth carrying: (1) 'seed 0' is the argparse default (:120) — the log never echoes resolved args, so it is inferred-from-default, though 24,414 = floor(100,000,000/4,096) is consistent with defaults throughout; (2) 14,356 tok/s is a cumulative average, not instantaneous. The 'starts from official HF weights, NOT from-scratch' caveat is correct — train_tinystories.py:4 'Starts from the official HF safetensors loaded into our SmolLM2ForCausalLM' (line 4, the fact cited :2-3).
```


### 6.22 SmolLM2 — is the nanotron reference config MEASURED here or COPIED from the publication?

**Value**

```
COPIED, and explicitly labelled as such: global_batch_size 512, tokens_per_step 1,048,576, 2,000,000 steps, ~2.1T tokens, implied data-parallel 64 (i.e. 64 GPUs), warmup 2000, decay 400,000 steps, clip_grad 1.0, bf16. None of this was run on this box.
```

**Evidence** — `SmolLM2-134(base)/results/training_recipe_resolved.json:2-3`

**Source quote**

```
"source": "https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm2/config_smollm2_135M.yaml",
  "fetched": "2026-05-13",
```

**Confidence** — measured from code

**Caveat** — This file is the clean case where COPIED vs MEASURED is unambiguous — it names its source URL and fetch date. It also records a real bug fix in its notes: 'Previous train.py default weight_decay=0.1 was 10× too high — corrected to 0.01'.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Every value CONFIRMED verbatim, but two labelling fixes: (a) the fact's confidence tag says 'measured-from-code' while the value itself says COPIED — for a model card it must be tagged COPIED/external; (b) 'implied data-parallel 64 (i.e. 64 GPUs)' — the JSON key is literally 'implied_data_parallel', a DERIVED quantity (512 / (8 x 1)); the file never says 'GPUs'.
```

**Verifier note**

```
training_recipe_resolved.json read in full (40 lines). Verified: :2 source URL (huggingface/smollm .../config_smollm2_135M.yaml), :3 '"fetched": "2026-05-13"', :15 warmup_steps 2000, :17 decay_steps 400000, :22 sequence_length 2048, :23 micro_batch_size 8, :24 batch_accumulation_per_replica 1, :25 implied_data_parallel 64, :26 global_batch_size 512, :27 tokens_per_step 1048576, :30 total_steps 2000000, :31 total_tokens_approx 2097152000000 (= 2.097T, i.e. '~2.1T'), :32 clip_grad 1.0, :33 precision bf16. The bug-fix note is verbatim at :38: 'Previous train.py default weight_decay=0.1 was 10x too high — corrected to 0.01'. This file is indeed the cleanest COPIED-vs-MEASURED case in the repo.
```


### 6.G Gaps — not determinable from disk

- GPU COUNT is not measured anywhere on disk. No file records torch.cuda.device_count(); only get_device_name(0) and safe_cuda.guard(device=0). 'Single GPU' rests on CLAUDE.md prose (lines 3, 9) and contract §C4.5.
- MFU was never computed for any Qwen3 run. mfu_meter.py has no output artifact anywhere in the repo, and no README quotes an MFU for the three builds. The GB10 peak in mfu_meter.py is estimated=True anyway.
- No train_flops / FLOP-accounting artifact exists for the 1.19B runs, so the §C18 iso-FLOP (<=5%) gate was never evaluated on disk for the three-build comparison — it is iso-TOKEN only, and the arms differ by ~31% in wall-clock.
- The Phase-B training runs have NO ledger entries (only their downstream evals do), so there is no ledger-recorded wall_clock_min, gpu_hours, git_commit, c5_evidence.json, or verdict.json for them. They predate ledger adoption.
- No HF dataset revision/sha is pinned for HuggingFaceFW/fineweb-edu sample-10BT — load_dataset is called with no revision arg, so the exact corpus snapshot behind the 1.19B tokens is unrecoverable.
- Wall-clock for the IMU-1 and partial-RoPE arms is NOT logged (those trainers write no completion line); I could only derive it from logged cumulative tok/s and corroborate against checkpoint mtimes.
- train_qwen3.py on disk today is NOT the version that produced the 1.19B numbers — the document-disjoint/decontam split landed in commit 86e79f3 after the runs. The pre-86e79f3 behaviour is recoverable from git, but no decontam_report.json exists for the tokcache_1191478400_300000.pt cache those runs actually consumed.
- No smoke-test artifact or c5_evidence.json exists for the Phase-B launches (the §C5 contract postdates them).
- I did not open the .pt checkpoints, which embed a 'training_recipe' dict written by save_ckpt (train_qwen3.py:358-367). That in-checkpoint record is therefore unverified — everything above comes from the scripts, the driver, and the logs. Loading a 1.2 GB checkpoint would need torch and I avoided touching the GPU path.

---

## 7. Loader API (real code)

<sub>Audit dimension: Real loader API of Qwen3-0.6B/model.py and SmolLM2-134(base)/model_full.py (classes, config dataclasses, checkpoint loading, forward contract, tokenizer, faithful end-to-end snippet, safe_cuda dependency)</sub>

### 7.1 Qwen3: exact top-level model class name and __init__ signature

**Value**

```
class Qwen3ForCausalLM(nn.Module) with __init__(self, cfg: Qwen3Config). Single positional arg named `cfg`, no kwargs, no from_pretrained classmethod. Inner module is Qwen3Model(cfg) exposed as .model; .lm_head is nn.Linear(hidden_size, vocab_size, bias=False), tied to .model.embed_tokens.weight when cfg.tie_word_embeddings.
```

**Evidence** — `Qwen3-0.6B/model.py:237-246`

**Source quote**

```
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            # lm_head.weight aliases embed_tokens.weight (same storage).
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)
```

**Confidence** — measured from code

**Caveat** — There is NO from_pretrained classmethod on this class — grep for 'from_pretrained' in Qwen3-0.6B/model.py matches only a comment on line 33. Weight loading is always external (see the load_official_weights_into_ours fact).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Verbatim match. Qwen3-0.6B/model.py:237 `class Qwen3ForCausalLM(nn.Module):`, :238 `def __init__(self, cfg: Qwen3Config):`, :241 `self.model = Qwen3Model(cfg)`, :242 lm_head nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False), :243-245 tie branch aliasing embed_tokens.weight, :246 `self.apply(self._init_weights)`. Quoted block = lines 237-246 exactly. `grep -n from_pretrained Qwen3-0.6B/model.py` returns exactly one hit, line 33 (a comment) — caveat CONFIRMED. Qwen3Model is defined at model.py:214 and .model is that instance.
```


### 7.2 Qwen3: config dataclass name and exact fields with defaults

**Value**

```
@dataclass Qwen3Config — 14 fields, all with defaults, so Qwen3Config() is valid with zero args.
```

**Evidence** — `Qwen3-0.6B/model.py:35-51`

**Source quote**

```
@dataclass
class Qwen3Config:
    vocab_size: int = 151_936                     # config.json: vocab_size
    hidden_size: int = 1024                       # config.json: hidden_size
    intermediate_size: int = 3072                 # config.json: intermediate_size
    num_hidden_layers: int = 28                   # config.json: num_hidden_layers
    num_attention_heads: int = 16                 # config.json: num_attention_heads
    num_key_value_heads: int = 8                  # config.json: num_key_value_heads (GQA 16/8 = 2:1)
    head_dim: int = 128                           # config.json: head_dim — INDEPENDENT field, not hidden/n_heads
    max_position_embeddings: int = 40_960         # config.json: max_position_embeddings
    rope_theta: float = 1_000_000.0               # config.json: rope_theta
    rms_norm_eps: float = 1e-6                    # config.json: rms_norm_eps
    initializer_range: float = 0.02               # config.json: initializer_range
    tie_word_embeddings: bool = True              # config.json: tie_word_embeddings
    attention_bias: bool = False                  # config.json: attention_bias
    attention_dropout: float = 0.0                # config.json: attention_dropout
    # hidden_act = "silu" → SwiGLU(silu(gate) * up). Hardcoded in MLP below.
```

**Confidence** — measured from code

**Caveat** — head_dim is a real dataclass FIELD here (128), unlike SmolLM2 where it is a derived @property. The inline comments claim each default matches Qwen/Qwen3-0.6B-Base config.json 'pulled via AutoConfig.from_pretrained on 2026-06-08' (model.py:33) — that provenance claim is a code comment, NOT something I re-verified against a config.json on disk.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Verifier note**

```
FIELD COUNT AND DEFAULTS CONFIRMED: AST parse of Qwen3-0.6B/model.py yields exactly 14 AnnAssign fields ['vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','max_position_embeddings','rope_theta','rms_norm_eps','initializer_range','tie_word_embeddings','attention_bias','attention_dropout'], all defaulted; quoted block = model.py:35-51 exactly; head_dim IS a real field at model.py:43.

MATERIAL QUALIFIER THE ORIGINAL AGENT DID NOT CATCH (it declared the provenance 'not re-verified' — I verified it and it FAILS on one field): the file header comment model.py:32-33 says 'every default matches config.json at the Qwen3-0.6B-Base repo HEAD (pulled via AutoConfig.from_pretrained on 2026-06-08)'. The actual cached config.json for that repo is on this box at HF_HOME=/home/yashb98/projects/qwen-distill/hf_cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/config.json (snapshot dir mtime 2026-06-08 14:36 — the very date the comment cites). It states "max_position_embeddings": 32768, whereas Qwen3-0.6B/model.py:44 hardcodes `max_position_embeddings: int = 40_960  # config.json: max_position_embeddings`. 13 of 14 defaults match that config.json (vocab 151936, hidden 1024, inter 3072, layers 28, heads 16, kv 8, head_dim 128, rope_theta 1e6, eps 1e-6, init 0.02, tie true, attn_bias false, attn_dropout 0.0); max_position_embeddings does NOT. 40960 is the Qwen3-0.6B instruct/thinking value, not the -Base value in this snapshot. I did NOT fetch the live HF HEAD, so I can only assert the mismatch against the on-disk snapshot.
Impact: harmless for weight loading (RoPE tables are built at model.py:222 and registered with persistent=False, so they never enter the state_dict — verify.py still passes), but a published model card MUST NOT repeat 'every default matches the Base config.json', and must not print 40,960 as the Base context length.
```


### 7.3 Qwen3: how are HF weights mapped in — is there a named converter/loader function?

**Value**

```
Yes: load_official_weights_into_ours(ours: Qwen3ForCausalLM, hf_state_dict: dict) defined in Qwen3-0.6B/verify.py:22. It does NO key remapping — module names were written to mirror HF exactly, so it is a plain load_state_dict(strict=False) that then asserts the only missing key is the tied lm_head.weight and that nothing is unexpected.
```

**Evidence** — `Qwen3-0.6B/verify.py:22,39-44`

**Source quote**

```
def load_official_weights_into_ours(ours: Qwen3ForCausalLM, hf_state_dict: dict):
...
    missing, unexpected = ours.load_state_dict(hf_state_dict, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"Unexpected missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")
```

**Confidence** — measured from code

**Caveat** — The function lives in verify.py, not model.py. Importing it (`from verify import load_official_weights_into_ours, REPO`) is safe because verify.py's main() is guarded by `if __name__ == "__main__":` at Qwen3-0.6B/verify.py:86. The SmolLM2 side proves this import pattern is the repo's own idiom (SmolLM2-134(base)/generate.py:11).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Qwen3-0.6B/verify.py:22 signature exact. Body lines 39-44 match the quote verbatim (39 load_state_dict strict=False, 40 filter lm_head.weight, 41-42 raise on missing, 43-44 raise on unexpected). Import-safety caveat CONFIRMED: verify.py:86 is `if __name__ == "__main__":` and :87 `main()`, so importing verify.py executes only lines 13-19 (torch, transformers, `from model import ...`, REPO). SmolLM2-134(base)/generate.py:11 is indeed `from verify import load_official_weights_into_ours, REPO`.
```


### 7.4 Qwen3: real call site that constructs the model and loads OFFICIAL HF weights

**Value**

```
Qwen3-0.6B/verify.py main() — the canonical parity gate.
```

**Evidence** — `Qwen3-0.6B/verify.py:49-64`

**Source quote**

```
print(f"Loading official {REPO} ...")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf_model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
    hf_model.eval()

    print("Building our model and copying weights ...")
    ours = Qwen3ForCausalLM(Qwen3Config())
    load_official_weights_into_ours(ours, hf_model.state_dict())
    ours.eval()

    # Same prompt, same dtype, same device.
    text = "The capital of France is"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    hf_out = hf_model(input_ids).logits          # (1, T, V)
    our_out = ours(input_ids)["logits"]
```

**Confidence** — measured from code

**Caveat** — Note `dtype=torch.float32` here, whereas the SmolLM2 equivalent uses the older `torch_dtype=torch.float32` (SmolLM2-134(base)/verify.py:53). The two repro folders are inconsistent on this transformers kwarg — pick one deliberately for a published snippet.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Quoted block is byte-for-byte lines 49-64 of Qwen3-0.6B/verify.py (49 print, 50 AutoTokenizer, 51 `AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)`, 55 `ours = Qwen3ForCausalLM(Qwen3Config())`, 56 load_official_weights_into_ours, 57 ours.eval(), 60 text, 61 input_ids, 63 hf_out, 64 our_out). The transformers-kwarg divergence caveat is CONFIRMED: Qwen3-0.6B/verify.py:51 uses `dtype=`, SmolLM2-134(base)/verify.py:53 uses `torch_dtype=`.
```


### 7.5 Qwen3: real call site that loads a TRAINED .pt checkpoint (not HF weights)

**Value**

```
The eval-harness-constructed suite script. It handles both bare state_dicts and {'model': sd} wrappers, and strips the torch.compile `_orig_mod.` prefix, then load_state_dict(..., strict=True).
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3-0.6b_eval-faithful/eval_suite.py:97-114`

**Source quote**

```
def load_checkpoint_sd(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    # torch.compile-trained checkpoints carry an _orig_mod. prefix (e.g. the
    # Qwen3 modernized build) — strip per finance-research-loop SKILL.md.
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def build_model(mod):
    # ADJUST AT CONSTRUCTION if the build's own train/verify script constructs
    # its config differently (extra kwargs, config loaded from the ckpt, ...).
    return getattr(mod, MODEL_CLASS)(getattr(mod, CONFIG_CLASS)())


def load_model(ckpt_path: str):
    model = build_model(load_model_module())
    model.load_state_dict(load_checkpoint_sd(ckpt_path), strict=True)  # strict, always
    return model.to(device=DEVICE, dtype=DTYPE).eval()
```

**Confidence** — measured from code

**Caveat** — build_model uses getattr indirection driven by module constants MODEL_CLASS="Qwen3ForCausalLM" / CONFIG_CLASS="Qwen3Config" (eval_suite.py:52-53). For a published snippet, inline the direct call `Qwen3ForCausalLM(Qwen3Config())` — that is exactly what the getattr resolves to. DEVICE/DTYPE are cuda+bfloat16 when CUDA is available (eval_suite.py:80-81).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Qwen3-0.6B/experiments/2026-06-16_qwen3-0.6b_eval-faithful/eval_suite.py:97 `def load_checkpoint_sd`, :98 torch.load(weights_only=False), :99 `sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck`, :102 removeprefix("_orig_mod."), :105 build_model, :108 getattr(mod, MODEL_CLASS)(getattr(mod, CONFIG_CLASS)()), :113 load_state_dict(..., strict=True), :114 .to(DEVICE, DTYPE).eval() — quote matches 97-114 verbatim. Caveat CONFIRMED: MODEL_CLASS="Qwen3ForCausalLM" at eval_suite.py:52, CONFIG_CLASS="Qwen3Config" at :53; DEVICE at :80 (`cuda` if available), DTYPE at :81 (bfloat16 if cuda else float32).
```


### 7.6 Qwen3: training-script construction + resume path (the from-scratch build)

**Value**

```
cfg = Qwen3Config(); model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype); resume via torch.load(...)['model'] into model.load_state_dict. Checkpoints are saved as a dict with keys model/config/step/tok_seen/baseline_ppl/trained_ppl/training_recipe/optim/sched/rng_*.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:292-293,311-312,350-356`

**Source quote**

```
cfg = Qwen3Config()
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
...
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
...
    def save_ckpt(step: int, tok_seen: int, trained_ppl=None):
        torch.save({
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "step": step,
            "tok_seen": tok_seen,
            "baseline_ppl": base_ppl,
```

**Confidence** — measured from code

**Caveat** — The saved 'config' is cfg.__dict__ (a plain dict), NOT a pickled Qwen3Config. No loader in the repo reconstructs the config from the checkpoint — every load site calls Qwen3Config() with defaults instead.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:292 `cfg = Qwen3Config()`, :293 `model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)`; :311 `ck = torch.load(args.resume, map_location="cpu", weights_only=False)`, :312 `model.load_state_dict(ck["model"])`; :350 `def save_ckpt(...)`, :351 torch.save({, :352 model, :353 `"config": cfg.__dict__,`, :354 step, :355 tok_seen, :356 baseline_ppl — quote matches exactly. Remaining keys verified: trained_ppl :357, training_recipe :358, optim/sched :368, rng_torch :369, rng_cuda :370, rng_numpy/rng_python :371. Caveat CONFIRMED: `cfg.__dict__` is a plain dict and no loader in the repo reads it back.
```


### 7.7 Qwen3: forward() signature and return type/shapes

**Value**

```
forward(self, input_ids, labels=None, attention_mask=None) -> dict with keys 'logits' and 'loss'. logits shape (B, T, vocab_size); loss is a scalar tensor when labels is passed, else None. Not a tuple, not a HF ModelOutput — a plain Python dict, so call sites index it as model(x)["logits"].
```

**Evidence** — `Qwen3-0.6B/model.py:259-274`

**Source quote**

```
def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None):
        hidden = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}
```

**Confidence** — measured from code

**Caveat** — The labels path materializes a full (B*T, 151936) fp32 CE — this is exactly the pattern CLAUDE.md §C1 warns about for large vocab. The model file itself does NOT chunk cross-entropy. Shapes confirmed by the module's own __main__ demo at model.py:307-309 (x = torch.randint(0, cfg.vocab_size, (2, 16)); out = m(x, labels=x)).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Qwen3-0.6B/model.py:259-274 matches the quote verbatim (259-261 signature, 262 hidden, 263 logits, 265 loss=None, 267-272 shift + F.cross_entropy over view(-1, 151936), 274 `return {"logits": logits, "loss": loss}`). Shape demo caveat CONFIRMED: model.py:307 `x = torch.randint(0, cfg.vocab_size, (2, 16))`, :308 `out = m(x, labels=x)`, :309 prints logits shape + loss. §C1 chunked-CE caveat is a correct reading — there is no chunking anywhere in model.py.
```


### 7.8 Qwen3: generate() signature and semantics

**Value**

```
@torch.no_grad() generate(self, input_ids, max_new_tokens=64, temperature=0.8, top_k=50) -> torch.Tensor of token ids (prompt + continuation concatenated). temperature<=0 means greedy argmax. No KV cache — the prefix is recomputed each step.
```

**Evidence** — `Qwen3-0.6B/model.py:276-295`

**Source quote**

```
@torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int | None = 50) -> torch.Tensor:
        """Greedy/top-k sampling. No KV cache — recomputes the prefix each step.
        Defaults mirror SmolLM2's generate() so the harness stays consistent."""
        self.eval()
```

**Confidence** — measured from code

**Caveat** — generate() calls self.eval() internally (model.py:281), so an explicit .eval() before it is redundant though harmless. Real call site: eval_suite.py:190-192 uses max_new_tokens=60, temperature=0.7, top_k=40 (suite constants at eval_suite.py:68), NOT the class defaults.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Qwen3-0.6B/model.py:276 `@torch.no_grad()`, :277-278 signature with those exact defaults, :279-280 docstring 'No KV cache — recomputes the prefix each step', :281 `self.eval()`, :285 `if temperature <= 0:` greedy argmax, :295 `return input_ids` after torch.cat at :294 — so the return is prompt+continuation. Caveats CONFIRMED: self.eval() at :281; real call site eval_suite.py:190-191 `model.generate(ids, max_new_tokens=GEN_MAXNEW, temperature=GEN_TEMP, top_k=GEN_TOPK)` with `GEN_SEED, GEN_TEMP, GEN_TOPK, GEN_MAXNEW = 42, 0.7, 40, 60` at eval_suite.py:68.
```


### 7.9 Qwen3: how is the tokenizer obtained, and with which repo id?

**Value**

```
AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base"). The repo id is the module constant REPO in verify.py and is re-declared as REPO in the train script and as TOKENIZER_REPO in the eval suite.
```

**Evidence** — `Qwen3-0.6B/verify.py:19,50`

**Source quote**

```
REPO = "Qwen/Qwen3-0.6B-Base"
...
    tokenizer = AutoTokenizer.from_pretrained(REPO)
```

**Confidence** — measured from code

**Caveat** — Same id independently at Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:59 (REPO = "Qwen/Qwen3-0.6B-Base") and Qwen3-0.6B/experiments/2026-06-16_qwen3-0.6b_eval-faithful/eval_suite.py:54 (TOKENIZER_REPO = "Qwen/Qwen3-0.6B-Base"  # REPO from <MODEL_DIR>/verify.py — own tokenizer ONLY). Note it is the -Base repo, not Qwen/Qwen3-0.6B.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Qwen3-0.6B/verify.py:19 `REPO = "Qwen/Qwen3-0.6B-Base"`, :50 `tokenizer = AutoTokenizer.from_pretrained(REPO)`. Independent restatements confirmed: train_qwen3.py:59 `REPO = "Qwen/Qwen3-0.6B-Base"` and eval_suite.py:54 `TOKENIZER_REPO = "Qwen/Qwen3-0.6B-Base"  # REPO from <MODEL_DIR>/verify.py — own tokenizer ONLY`. The -Base (not -Instruct) distinction is correct.
```


### 7.10 SmolLM2: exact top-level model class name and __init__ signature

**Value**

```
class SmolLM2ForCausalLM(nn.Module) with __init__(self, cfg: SmolLM2Config). Structurally identical to the Qwen3 class: one positional `cfg`, .model = SmolLM2Model(cfg), tied .lm_head.
```

**Evidence** — `SmolLM2-134(base)/model_full.py:238-248`

**Source quote**

```
class SmolLM2ForCausalLM(nn.Module):
    def __init__(self, cfg: SmolLM2Config):
        super().__init__()
        self.cfg = cfg
        self.model = SmolLM2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            # Weight tying: lm_head.weight IS embed_tokens.weight (same storage).
            # config.json: tie_word_embeddings = true.
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)
```

**Confidence** — measured from code

**Caveat** — No from_pretrained classmethod (grep for 'from_pretrained' in model_full.py returns nothing).

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
SmolLM2-134(base)/model_full.py:238-248 matches the quote verbatim (238 class, 239 __init__, 242 self.model = SmolLM2Model(cfg), 243 lm_head, 244-246 tie comment+alias, 248 self.apply). `grep -n from_pretrained "SmolLM2-134(base)/model_full.py"` exits 1 (zero hits) — caveat CONFIRMED.
```


### 7.11 SmolLM2: config dataclass name and exact fields with defaults

**Value**

```
@dataclass SmolLM2Config — 12 fields, all defaulted, plus a derived head_dim @property (576//9 = 64). SmolLM2Config() is valid with zero args.
```

**Evidence** — `SmolLM2-134(base)/model_full.py:28-49`

**Source quote**

```
@dataclass
class SmolLM2Config:
    vocab_size: int = 49152                       # config.json: vocab_size
    hidden_size: int = 576                        # config.json: hidden_size
    intermediate_size: int = 1536                 # config.json: intermediate_size
    num_hidden_layers: int = 30                   # config.json: num_hidden_layers
    num_attention_heads: int = 9                  # config.json: num_attention_heads
    num_key_value_heads: int = 3                  # config.json: num_key_value_heads (GQA: 9 Q / 3 KV)
    max_position_embeddings: int = 8192           # config.json: max_position_embeddings
    rope_theta: float = 100_000.0                 # config.json: rope_theta (note: v2 uses 100k, v1 was 10k)
    rms_norm_eps: float = 1e-5                    # config.json: rms_norm_eps
    initializer_range: float = 1.0 / math.sqrt(576)  # config.json: initializer_range = 0.041666... = 1/sqrt(576)
    tie_word_embeddings: bool = True              # config.json: tie_word_embeddings
    attention_bias: bool = False                  # config.json: attention_bias
    attention_dropout: float = 0.0                # config.json: attention_dropout
    # hidden_act = "silu" → SwiGLU(silu(gate) * up), per HF LlamaMLP. Hardcoded below.

    @property
    def head_dim(self) -> int:
        # Llama convention: head_dim = hidden_size // num_attention_heads.
        # 576 / 9 = 64.
        return self.hidden_size // self.num_attention_heads
```

**Confidence** — measured from code

**Caveat** — KEY ASYMMETRY vs Qwen3: head_dim here is a read-only @property derived from hidden_size//num_attention_heads, NOT a settable dataclass field. `SmolLM2Config(head_dim=64)` would raise TypeError. Also initializer_range hardcodes sqrt(576) rather than sqrt(hidden_size), so overriding hidden_size silently leaves the old init std.

**Verdict — ❌ WRONG**

**Corrected value**

```
13 fields (not 12): vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, max_position_embeddings, rope_theta, rms_norm_eps, initializer_range, tie_word_embeddings, attention_bias, attention_dropout — plus the derived head_dim @property.
```

**Verifier note**

```
FIELD COUNT IS WRONG. AST parse of SmolLM2-134(base)/model_full.py returns 13 AnnAssign fields in SmolLM2Config, not 12. Cross-check: Qwen3Config = 14 = the same 13 + head_dim, which the fact list itself asserts, so 12 is internally inconsistent with its own sibling fact. Everything else in this fact is CONFIRMED: quoted block = model_full.py:28-49 verbatim; attention_dropout at :42 (grep-confirmed); @property head_dim at :45-49 returning hidden_size // num_attention_heads = 576//9 = 64; initializer_range at :39 literally `1.0 / math.sqrt(576)` (hardcoded 576, not hidden_size); all 13 fields defaulted so SmolLM2Config() is valid with zero args, and head_dim is not an accepted kwarg. BONUS VERIFICATION the original agent did not do: all 13 defaults DO match the cached HuggingFaceTB/SmolLM2-135M config.json under HF_HOME (/home/yashb98/projects/qwen-distill/hf_cache/hub/models--HuggingFaceTB--SmolLM2-135M/.../config.json: vocab 49152, hidden 576, inter 1536, layers 30, heads 9, kv 3, max_pos 8192, rope_theta 100000, eps 1e-05, initializer_range 0.041666666666666664, tie true, attention_bias false, attention_dropout 0.0) — unlike Qwen3, this config has no mismatch.
```


### 7.12 SmolLM2: how are HF weights mapped in — named converter function?

**Value**

```
Yes: load_official_weights_into_ours(ours: SmolLM2ForCausalLM, hf_state_dict: dict) in SmolLM2-134(base)/verify.py:22. Same shape as the Qwen3 one — no key remapping, strict=False load then assert only the tied lm_head.weight is missing.
```

**Evidence** — `SmolLM2-134(base)/verify.py:22,39-46`

**Source quote**

```
def load_official_weights_into_ours(ours: SmolLM2ForCausalLM, hf_state_dict: dict):
...
    # Filter strict=False to ignore the absent lm_head.weight (it's tied).
    missing, unexpected = ours.load_state_dict(hf_state_dict, strict=False)
    # We expect lm_head.weight to be "missing" (tied), and nothing unexpected.
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"Unexpected missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")
```

**Confidence** — measured from code

**Caveat** — Re-used by import in at least four call sites: generate.py:11, compare_with_hf.py:28, eval_after_vs_base.py:25, tests/test_parity.py:33. That makes `from verify import load_official_weights_into_ours, REPO` the repo's blessed public loader idiom.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
verify.py:22 signature exact; lines 39-46 match the quote verbatim (39 comment, 40 load_state_dict strict=False, 41 comment, 42 filter, 43-44 missing raise, 45-46 unexpected raise). Four re-use sites CONFIRMED at the cited lines: generate.py:11, compare_with_hf.py:28, eval_after_vs_base.py:25, tests/test_parity.py:33. Minor: three of the four write the names in the opposite order (`from verify import REPO, load_official_weights_into_ours`); only generate.py:11 uses the order the fact quotes. Semantically irrelevant, but the 'blessed idiom' wording overstates uniformity.
```


### 7.13 SmolLM2: real call sites constructing the model and loading weights (HF and trained .pt)

**Value**

```
HF path: verify.py:57-58 and generate.py:18-19. Trained-checkpoint path: eval_after_vs_base.py:42-44 (torch.load(...) then load_state_dict(ckpt['model'])).
```

**Evidence** — `SmolLM2-134(base)/eval_after_vs_base.py:34-45`

**Source quote**

```
print("Loading BASE (official) model...")
hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
base = SmolLM2ForCausalLM(SmolLM2Config())
load_official_weights_into_ours(base, hf.state_dict())
base = base.to(device=device, dtype=dtype).eval()
del hf

print("Loading TRAINED (TinyStories continued-pretrained) model...")
trained = SmolLM2ForCausalLM(SmolLM2Config())
ckpt = torch.load("checkpoint_tinystories.pt", map_location="cpu", weights_only=False)
trained.load_state_dict(ckpt["model"])
trained = trained.to(device=device, dtype=dtype).eval()
```

**Confidence** — measured from code

**Caveat** — eval_after_vs_base.py:43 uses a RELATIVE path 'checkpoint_tinystories.pt' — it only works with cwd = SmolLM2-134(base)/. That file does exist on disk (269144681 bytes), as does checkpoint.pt (538173921 bytes). Unlike the Qwen3 eval suite, this site does NOT strip an `_orig_mod.` prefix, so it would break on a torch.compile-saved checkpoint.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
eval_after_vs_base.py:34-45 matches the quote verbatim: :35 `hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)`, :36 base build, :37 loader, :38 .to().eval(), :39 del hf, :42 `trained = SmolLM2ForCausalLM(SmolLM2Config())`, :43 `ckpt = torch.load("checkpoint_tinystories.pt", map_location="cpu", weights_only=False)`, :44 `trained.load_state_dict(ckpt["model"])`, :45 .to().eval(). HF path confirmed at verify.py:57-58 and generate.py:18-19. Caveats CONFIRMED by `ls -la`: checkpoint_tinystories.pt = 269144681 bytes, checkpoint.pt = 538173921 bytes, both present; the relative path at :43 does require cwd = SmolLM2-134(base)/; and there is no _orig_mod. stripping at :44 (default strict=True).
```


### 7.14 SmolLM2: forward() signature and return type/shapes

**Value**

```
Identical contract to Qwen3: forward(self, input_ids, labels=None, attention_mask=None) -> {"logits": (B,T,49152), "loss": scalar-or-None}. Plain dict.
```

**Evidence** — `SmolLM2-134(base)/model_full.py:263-279`

**Source quote**

```
def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None):
        hidden = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Standard causal LM shift: predict token t+1 from positions ≤ t.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}
```

**Confidence** — measured from code

**Caveat** — attention_mask is forwarded straight into F.scaled_dot_product_attention as attn_mask, and is_causal flips to False whenever a mask is supplied (model_full.py:156-161; identically Qwen3-0.6B/model.py:162-167). So passing an HF-style 2-D padding mask would SILENTLY DISABLE causal masking — a published snippet should not pass attention_mask.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
model_full.py:263-279 matches the quote verbatim (263-265 signature, 266 hidden, 267 logits, 269 loss=None, 271-277 shift + F.cross_entropy, 279 return dict). The attention_mask caveat is the most valuable item in this dimension and is CONFIRMED at both files: model_full.py:156-161 and Qwen3-0.6B/model.py:162-167 are the identical `F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=(attention_mask is None))` — passing any 2-D HF padding mask does silently disable causal masking. SmolLM2's own inline comment at model_full.py:154-155 acknowledges this.
```


### 7.15 SmolLM2: how is the tokenizer obtained, and with which repo id?

**Value**

```
AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M"), via the module constant REPO in verify.py.
```

**Evidence** — `SmolLM2-134(base)/verify.py:19,52`

**Source quote**

```
REPO = "HuggingFaceTB/SmolLM2-135M"
...
    tokenizer = AutoTokenizer.from_pretrained(REPO)
```

**Confidence** — measured from code

**Caveat** — Also hardcoded literally (not via REPO) at SmolLM2-134(base)/train.py:134: `tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")`. Note the folder is named SmolLM2-134(base) but the HF repo id says 135M.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
SmolLM2-134(base)/verify.py:19 `REPO = "HuggingFaceTB/SmolLM2-135M"`, :52 `tokenizer = AutoTokenizer.from_pretrained(REPO)`. Caveat CONFIRMED: train.py:134 hardcodes the literal `tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")` rather than importing REPO. Folder-vs-repo-name mismatch (134 vs 135M) is real.
```


### 7.16 Does model.py / model_full.py itself depend on safe_cuda / guard() being imported first?

**Value**

```
NO. Neither model file imports safe_cuda — `grep -rn safe_cuda Qwen3-0.6B/model.py Qwen3-0.6B/verify.py` exits 1 (no match), and `grep -rn safe_cuda` over the ENTIRE SmolLM2-134(base)/ tree exits 1 (zero matches anywhere, including train.py). The model files import only torch/torch.nn/torch.nn.functional (+ dataclasses, and math for SmolLM2). safe_cuda is a CALLER-SIDE obligation imposed by CLAUDE.md §C1, honored by the research-loop-constructed scripts, not by the model modules.
```

**Evidence** — `Qwen3-0.6B/model.py:22-28`

**Source quote**

```
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
```

**Confidence** — measured from code

**Caveat** — IMPORTANT for a published snippet: Qwen3-0.6B/verify.py — the repo's own canonical usage script — does NOT import safe_cuda either (grep exit 1). The GPU-touching research-loop scripts DO: eval_suite.py:38-45 imports safe_cuda before torch then calls safe_cuda.guard(0.85) at line 202; train_qwen3.py:42 imports safe_cuda before torch and calls safe_cuda.guard(args.mem_fraction) at line 269. safe_cuda.guard's real signature is `def guard(fraction: float = 0.85, device: int = 0) -> None` (safe_cuda.py:47) and it no-ops when CUDA is unavailable (safe_cuda.py:51-52). So: a CPU-only snippet needs no guard; any snippet that moves the model to CUDA should include the two-line safe_cuda header to be repo-compliant.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Reproduced every grep: `grep -n safe_cuda Qwen3-0.6B/model.py Qwen3-0.6B/verify.py` exits 1 (no matches); `grep -rn safe_cuda "SmolLM2-134(base)/"` exits 1 (zero matches over the whole tree incl. train.py, tests/, scripts/). Qwen3-0.6B/model.py:22-28 imports exactly what the quote shows. Caveat lines all CONFIRMED: eval_suite.py:38 safe_cuda header comment, :43 `import safe_cuda`, :45 `import torch` (so 38-45 is the correct span), `safe_cuda.guard(0.85)` at :202 (also a second call at :315, not mentioned); train_qwen3.py:42 `import safe_cuda` before torch at :47, `safe_cuda.guard(args.mem_fraction)` at :269 with the flag defined at :242 (default 0.85). safe_cuda.py:47 `def guard(fraction: float = 0.85, device: int = 0) -> None:` exact; :51-52 `if not torch.cuda.is_available(): return` exact. Additional un-noted guard behavior: safe_cuda.py:53-56 raises ValueError unless 0.0 < fraction <= 0.95.
```


### 7.17 SmolLM2: is there an existing end-to-end usage script I can copy verbatim?

**Value**

```
YES — SmolLM2-134(base)/generate.py is a complete import -> build config -> build model -> load weights -> tokenize -> generate -> decode script in 31 lines. This is the highest-fidelity source for a published usage example on the SmolLM2 side; nothing needs to be invented.
```

**Evidence** — `SmolLM2-134(base)/generate.py:6-25`

**Source quote**

```
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_full import SmolLM2ForCausalLM, SmolLM2Config
from verify import load_official_weights_into_ours, REPO


def main(prompt: str, max_new_tokens: int = 64):
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32)

    model = SmolLM2ForCausalLM(SmolLM2Config())
    load_official_weights_into_ours(model, hf.state_dict())
    del hf
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
```

**Confidence** — measured from code

**Caveat** — generate.py:16 uses `torch_dtype=` which is deprecated in current transformers; three other SmolLM2 call sites in the same folder use `dtype=` (compare_with_hf.py:45, eval_after_vs_base.py:35, tests/test_parity.py:39), as does Qwen3-0.6B/verify.py:51. If publishing, prefer `dtype=` and note the divergence rather than silently 'fixing' generate.py.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Verifier note**

```
SUBSTANCE CONFIRMED: generate.py:6-25 matches the quote verbatim and is a complete import->config->model->load->tokenize->generate->decode path. LINE COUNT WRONG: the file is 30 lines, not 31 (`wc -l generate.py` = 30; bytes=980; endswith newline=True; len(splitlines())=30). Deprecation caveat CONFIRMED: generate.py:16 uses `torch_dtype=`, while compare_with_hf.py:45, eval_after_vs_base.py:35, tests/test_parity.py:39 and Qwen3-0.6B/verify.py:51 all use `dtype=`.
```


### 7.18 FAITHFUL end-to-end usage snippet — SmolLM2 (every line traced to a repo line)

**Value**

```
import torch  # generate.py:7\nfrom transformers import AutoTokenizer, AutoModelForCausalLM  # generate.py:8\nfrom model_full import SmolLM2ForCausalLM, SmolLM2Config  # generate.py:10\nfrom verify import load_official_weights_into_ours, REPO  # generate.py:11\n\ntokenizer = AutoTokenizer.from_pretrained(REPO)  # generate.py:15  (REPO = "HuggingFaceTB/SmolLM2-135M", verify.py:19)\nhf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)  # eval_after_vs_base.py:35 (generate.py:16 is the same call with the deprecated torch_dtype=)\nmodel = SmolLM2ForCausalLM(SmolLM2Config())  # generate.py:18\nload_official_weights_into_ours(model, hf.state_dict())  # generate.py:19\ndel hf  # generate.py:20\nmodel.eval()  # generate.py:21\n\ninput_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids  # call = generate.py:23; prompt string = verify.py:62\nlogits = model(input_ids)["logits"]  # verify.py:66 (`our_out = ours(input_ids)["logits"]`)\nnext_id = logits[0, -1].argmax().item()  # verify.py:80 (`our_next = our_out[0, -1].argmax().item()`)\nprint(tokenizer.decode([next_id]))  # decode-a-single-id form from verify.py:82; skip_special_tokens form from generate.py:25\n\nout = model.generate(input_ids, max_new_tokens=64, temperature=0.8, top_k=50)  # generate.py:24 with its own defaults (generate.py:14, model_full.py:282-283)\nprint(tokenizer.decode(out[0], skip_special_tokens=True))  # generate.py:25
```

**Evidence** — `SmolLM2-134(base)/generate.py:6-25`

**Source quote**

```
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
```

**Confidence** — measured from code

**Caveat** — LINES I WROTE MYSELF (not verbatim in any single repo file): (a) the blank-line/ordering assembly — generate.py wraps these in `def main(prompt, max_new_tokens=64)`, I unwrapped to module scope; (b) inlining the literal "The capital of France is" in place of the `prompt` parameter — the literal is real (verify.py:62) but appears there as a separate `text = ...` binding; (c) `print(tokenizer.decode([next_id]))` — verify.py:82 is `print(f"Ours next     : {tokenizer.decode([our_next])!r}")`, so the decode call is real but the print wrapper is simplified; (d) swapping torch_dtype= for dtype= as noted. Everything else is verbatim. REQUIRES cwd = SmolLM2-134(base)/ (or that dir on sys.path) because `from model_full import ...` and `from verify import ...` are flat-module imports — there is no __init__.py in that folder.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Verifier note**

```
EVERY CITATION IN THE SNIPPET VERIFIED CORRECT, line by line: generate.py:7 import torch; :8 transformers import; :10 model_full import; :11 verify import; :14 default max_new_tokens=64; :15 tokenizer; :18 model build; :19 loader; :20 del hf; :21 model.eval(); :23 input_ids; :24 generate(temperature=0.8, top_k=50); :25 decode(skip_special_tokens=True). verify.py:19 REPO; :62 `text = "The capital of France is"`; :66 `our_out = ours(input_ids)["logits"]`; :80 `our_next = our_out[0, -1].argmax().item()`; :82 the decode print. eval_after_vs_base.py:35 the `dtype=` form. model_full.py:282-283 the generate defaults. The self-declared 'lines I wrote myself' list is honest and complete.
MISSING MATERIAL CAVEAT (asymmetric with the Qwen3 sibling fact, which does flag it): the repo runs this whole sequence under `@torch.no_grad()` (SmolLM2-134(base)/verify.py:49 decorates main(); generate.py has no forward call outside model.generate, which is itself @torch.no_grad() at model_full.py:281). The composed snippet calls `model(input_ids)["logits"]` at module scope with grad tracking ON. A published snippet should keep `with torch.no_grad():` around the forward.
Second qualifier: the snippet needs the official weights. They are NOT in ~/.cache/huggingface (that dir holds only models--Qwen--Qwen3.5-9B); they are at HF_HOME=/home/yashb98/projects/qwen-distill/hf_cache/hub/models--HuggingFaceTB--SmolLM2-135M. First run on a clean machine downloads from the Hub.
```


### 7.19 FAITHFUL end-to-end usage snippet — Qwen3 (every line traced to a repo line)

**Value**

```
import torch  # verify.py:13\nfrom transformers import AutoModelForCausalLM, AutoTokenizer  # verify.py:14\nfrom model import Qwen3ForCausalLM, Qwen3Config  # verify.py:16\nfrom verify import load_official_weights_into_ours, REPO  # WRITTEN BY ME (import form copied from SmolLM2-134(base)/generate.py:11); both names are real at Qwen3-0.6B/verify.py:22 and :19\n\ntokenizer = AutoTokenizer.from_pretrained(REPO)  # verify.py:50  (REPO = "Qwen/Qwen3-0.6B-Base", verify.py:19)\nhf_model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)  # verify.py:51\nours = Qwen3ForCausalLM(Qwen3Config())  # verify.py:55\nload_official_weights_into_ours(ours, hf_model.state_dict())  # verify.py:56\nours.eval()  # verify.py:57\n\ntext = "The capital of France is"  # verify.py:60\ninput_ids = tokenizer(text, return_tensors="pt").input_ids  # verify.py:61\nour_out = ours(input_ids)["logits"]  # verify.py:64\nour_next = our_out[0, -1].argmax().item()  # verify.py:78\nprint(tokenizer.decode([our_next]))  # simplified from verify.py:80\n\nout = ours.generate(input_ids, max_new_tokens=60, temperature=0.7, top_k=40)  # call form + exact args from eval_suite.py:190-191\nprint(tokenizer.decode(out[0], skip_special_tokens=True))  # eval_suite.py:192 (`return tokenizer.decode(out[0], skip_special_tokens=True)`)
```

**Evidence** — `Qwen3-0.6B/verify.py:49-64`

**Source quote**

```
tokenizer = AutoTokenizer.from_pretrained(REPO)
    hf_model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
    hf_model.eval()

    print("Building our model and copying weights ...")
    ours = Qwen3ForCausalLM(Qwen3Config())
    load_official_weights_into_ours(ours, hf_model.state_dict())
    ours.eval()

    # Same prompt, same dtype, same device.
    text = "The capital of France is"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    hf_out = hf_model(input_ids).logits          # (1, T, V)
    our_out = ours(input_ids)["logits"]
```

**Confidence** — measured from code

**Caveat** — LINES I WROTE MYSELF: (a) `from verify import load_official_weights_into_ours, REPO` — this exact line does NOT exist in Qwen3-0.6B/ (it exists only as the SmolLM2 analogue at generate.py:11). Both imported names are real module-scope objects in Qwen3-0.6B/verify.py (lines 22 and 19) and verify.py's main() is __main__-guarded (line 86), so the import is sound, but it is my composition, not copied text. (b) unwrapping verify.py's `@torch.no_grad() def main()` (lines 47-48) to module scope — the repo runs this whole block under torch.no_grad(); a published snippet should keep the decorator or wrap in `with torch.no_grad():`. (c) the simplified print. (d) the two generate lines come from a DIFFERENT file (the eval suite) than the rest. REQUIRES cwd = Qwen3-0.6B/ or sys.path insertion — the repo's own idiom for that is train_qwen3.py:55-57: `MODEL_DIR = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(MODEL_DIR)); from model import Qwen3Config, Qwen3ForCausalLM`. Also: this snippet loads the FULL fp32 HF model plus a second full copy of the weights — on the GB10 that is ~2x596M x 4B = ~4.8 GB, CPU-only here (nothing is moved to CUDA), so no safe_cuda.guard is required, matching verify.py which has none.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Verifier note**

```
ALL VERIFY.PY CITATIONS EXACT: :13 import torch; :14 `from transformers import AutoModelForCausalLM, AutoTokenizer`; :16 `from model import Qwen3ForCausalLM, Qwen3Config`; :19 REPO; :22 loader def; :47-48 `@torch.no_grad()` + `def main():`; :50 tokenizer; :51 `dtype=torch.float32`; :55 model build; :56 loader call; :57 ours.eval(); :60 text; :61 input_ids; :64 logits; :78 argmax; :80 the decode print; :86 the __main__ guard. eval_suite.py:192 is `return tokenizer.decode(out[0], skip_special_tokens=True)`. The sys.path idiom quoted from train_qwen3.py:55-57 is verbatim correct (55 MODEL_DIR parents[2], 56 sys.path.insert, 57 `from model import Qwen3Config, Qwen3ForCausalLM`). The memory math checks out: 596,049,920 x 4 B = 2.38 GB per copy, ~4.8 GB for two, CPU-only, no safe_cuda needed.
QUALIFIER 1 (citation precision): `max_new_tokens=60, temperature=0.7, top_k=40` is annotated 'exact args from eval_suite.py:190-191', but lines 190-191 pass the NAMES GEN_MAXNEW/GEN_TEMP/GEN_TOPK; the literals 60/0.7/40 live at eval_suite.py:68 (`GEN_SEED, GEN_TEMP, GEN_TOPK, GEN_MAXNEW = 42, 0.7, 40, 60`). Re-anchor to :68.
QUALIFIER 2 (not stated): with no KV cache (model.py:279-280) 60 new tokens = 60 full fp32 CPU forward passes of a 596M model — this snippet is minutes-slow on CPU, unlike the 135M SmolLM2 one.
QUALIFIER 3: as with SmolLM2, the weights come from HF_HOME=/home/yashb98/projects/qwen-distill/hf_cache (snapshot da87bfb608c14b7cf20ba1ce41287e8de496c0cd), not ~/.cache/huggingface. The self-flagged 'written by me' items (the `from verify import ...` line, the no_grad unwrap, the simplified print, the cross-file generate lines) are all accurately declared.
```


### 7.20 Is there a public helper for parameter counting?

**Value**

```
Yes, in BOTH files, identically: def num_params(model: nn.Module, only_trainable: bool = False) -> int. Both module __main__ blocks print it against a hardcoded expected value.
```

**Evidence** — `Qwen3-0.6B/model.py:298-299`

**Source quote**

```
def num_params(model: nn.Module, only_trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_trainable))
```

**Confidence** — measured from code

**Caveat** — The 'expected' counts printed in the __main__ demos are prose-in-code, NOT computed at read time: Qwen3-0.6B/model.py:306 prints 'Expected:      ~596,049,920  (596M-branded, "0.6B")' and SmolLM2-134(base)/model_full.py:314 prints 'Expected:      ~134,515,008  (135M-branded)'. I did not execute either module, so I am reporting those as literal source strings, not as verified parameter counts. Also note the `only_trainable` logic reads `p.requires_grad or not only_trainable`, which counts ALL params when only_trainable=False and only requires_grad ones when True — correct, but the condition is inverted-looking.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Qwen3-0.6B/model.py:298-299 and SmolLM2-134(base)/model_full.py:304-305 are byte-identical two-line definitions matching the quote. Hardcoded expected strings CONFIRMED as literal source text: model.py:306 `print(f"Expected:      ~596,049,920  (596M-branded, '0.6B')")` (the fact's caveat renders the inner quotes as double — the source uses single quotes; cosmetic only) and model_full.py:314 `print(f"Expected:      ~134,515,008  (135M-branded)")`. The caveat's honesty about not having executed either module is correct and should be preserved on any card: those two counts are prose-in-code, not values I re-computed.
```


### 7.21 Config fields that are declared but never read (traps for a published snippet)

**Value**

```
attention_dropout is declared in BOTH configs (Qwen3-0.6B/model.py:50, SmolLM2-134(base)/model_full.py:42) but is never referenced anywhere else in either file — attention hardcodes dropout_p=0.0. Setting it has NO effect.
```

**Evidence** — `Qwen3-0.6B/model.py:162-167`

**Source quote**

```
out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None),
        )
```

**Confidence** — measured from code

**Caveat** — Verified by `grep -n "attention_dropout" Qwen3-0.6B/model.py "SmolLM2-134(base)/model_full.py"` — the only hits are the two dataclass declaration lines; there is no read site. Same conclusion for both models.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
`grep -n attention_dropout Qwen3-0.6B/model.py "SmolLM2-134(base)/model_full.py"` returns exactly two hits total — Qwen3-0.6B/model.py:50 and SmolLM2-134(base)/model_full.py:42, both dataclass declarations, zero read sites. Quoted SDPA block = Qwen3-0.6B/model.py:162-167 verbatim, with `dropout_p=0.0` hardcoded at :165; the SmolLM2 twin is model_full.py:156-161. Setting attention_dropout has no effect in either model.
```


### 7.V Additional verifier findings (no 1:1 extracted fact)

**7.V1 — ⚠️ NEEDS QUALIFIER** · [GAP CHECK] Qwen3-0.6B has no standalone end-to-end usage script; top-level listing

**Checked against**

```
ls Qwen3-0.6B/ shows only model.py, verify.py, make_phase_plots.py at top level
```

**Verifier note**

```
The load-bearing part is CONFIRMED: those are the only three top-level .py files, and there is no generate.py equivalent. But the listing is incomplete as stated — `ls -1 Qwen3-0.6B/` also shows builds/, experiments/, results_overview/, __pycache__/, README.md and PLOTS_INDEX.md. The `.generate(` grep is exactly reproducible: 9 files, all under experiments/*/eval_suite.py (6 of them), builds/2026-06-08_reproduce-faithful_qwen3-0.6b/test_model.py, builds/.../train_qwen3.py, and experiments/2026-06-27_qwen3-0.6b_sft-3seed/eval_suite.py.
```


**7.V2 — ✅ CONFIRMED** · [GAP CHECK] No __init__.py in either model folder (flat-module imports required)

**Checked against**

```
ls Qwen3-0.6B/__init__.py 'SmolLM2-134(base)/__init__.py' -> No such file
```

**Verifier note**

```
Reproduced exactly: `ls: cannot access 'Qwen3-0.6B/__init__.py': No such file or directory` and the same for 'SmolLM2-134(base)/__init__.py'. Neither folder is an importable package, so every published snippet must set cwd to the model folder or sys.path.insert it (repo idiom at Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:55-57). Additional friction worth stating on a card: the SmolLM2 folder name contains parentheses — 'SmolLM2-134(base)' — which must be quoted in any shell cd.
```


### 7.G Gaps — not determinable from disk

- Qwen3-0.6B has NO standalone end-to-end usage script equivalent to SmolLM2-134(base)/generate.py. `ls /home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/` shows only model.py, verify.py, make_phase_plots.py at top level; `grep -rln '\.generate(' --include=*.py .` under Qwen3-0.6B/ returns only experiment eval_suite.py files, builds/.../test_model.py and builds/.../train_qwen3.py. So the Qwen3 usage snippet must be composed from verify.py + an experiment eval_suite.py, which I flagged inline.
- Neither model class exposes a from_pretrained / save_pretrained classmethod, and neither folder has an __init__.py (`ls Qwen3-0.6B/__init__.py 'SmolLM2-134(base)/__init__.py'` -> No such file). There is therefore no importable package API: every published snippet must either set cwd to the model folder or do a sys.path.insert, and I could not find any repo-provided convenience wrapper that hides this.
- I did not EXECUTE either model file, verify.py, or any eval script during this task — no GPU/model run was performed. Every claim here is static source reading. Therefore the parameter counts, the max|Δlogits| tolerance actually achieved, and whether the HF repos are present in the local HF cache are all unverified by me.
- No checkpoint-to-config reconstruction path exists on disk. train scripts save `"config": cfg.__dict__` (train_qwen3.py:353, SmolLM2 train.py:184) but I found no loader anywhere that reads it back — all load sites call Qwen3Config()/SmolLM2Config() with defaults. If a checkpoint were ever trained with non-default config, no repo code would detect the mismatch beyond load_state_dict shape errors.
- I could not determine from disk whether the `torch_dtype=` (SmolLM2 verify.py:53, generate.py:16) vs `dtype=` (Qwen3 verify.py:51, SmolLM2 compare_with_hf.py:45 / eval_after_vs_base.py:35 / tests/test_parity.py:39) split is deliberate or just file-age drift — there is no comment or requirements pin explaining it. SmolLM2-134(base)/requirements.txt exists (62 bytes) but I did not read it as part of this dimension.

---

## 8. Checkpoint inventory on disk

<sub>Audit dimension: checkpoint inventory on disk (what can actually be uploaded as weights)</sub>

### 8.1 How many weights-bearing checkpoint files exist under the repo (>1MB), and what is their total size?

**Value**

```
107 real (non-symlink) weights-bearing files, 290,643,004,043 bytes = 270.68 GiB = 290.64 GB. Extensions found: only .pt (PyTorch) and .pkl (JAX/Flax) plus one .discarded_* suffix file. ZERO .safetensors, .msgpack, .npz, .pth, .ckpt files exist anywhere in the repo.
```

**Evidence** — `HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/ , Qwen3-0.6B/ , SmolLM2-134(base)/ (measured via os.walk + os.path.getsize, symlinks excluded)`

**Source quote**

```
weights    n=107 bytes=290,643,004,043  GiB=270.68  GB=290.64
```

**Confidence** — measured from code

**Caveat** — The find command in the task brief double-counts: 12 of the paths it returns are SYMLINKS to other runs' checkpoints (§C13 control-reuse), and 26 more matches named tokcache_*.pt are TOKENIZED DATA CACHES, not weights, plus 10 research/datasets/*.bin are raw uint16 token shards. Naive summing gives ~334 GiB; the true unique weights figure is 270.68 GiB.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Headline is exact and reproduces. Caveat correction: the 10 research/datasets/*.bin shards are uint32, NOT uint16.
```

**Verifier note**

```
RE-MEASURED independently with os.walk + os.path.getsize (symlinks excluded, >1MB): weights n=107 bytes=290,643,004,043 -> 270.68 GiB / 290.64 GB. EXACT match. Extension split re-measured: .pt n=89 (215.08 GiB), .pkl n=17 (52.51 GiB), .discarded_* n=1 (3.10 GiB) = 107. `find` for *.safetensors/*.msgpack/*.npz/*.pth/*.ckpt (excluding .git) returned nothing. Symlink count 12 confirmed. Naive-sum '~334 GiB' also checks out (315,669,745,533 B + ~12x3.58 GB dereferenced symlinks = ~358.6 GB = ~334 GiB). ERROR IN CAVEAT: 'raw uint16 token shards' is wrong. research/datasets/data-selection-dclm-edu/prepare_dclm_edu.py:94 reads `arr = np.array(train_buf, dtype=np.uint32)` and :125 `np.array(eval_toks, dtype=np.uint32).tofile(...)`; research/datasets/data-selection-dclm-edu/meta.json:5 `"dtype": "uint32"`; research/datasets/math-reasoning-openr1-math-220k/meta.json:2 `"dtype": "uint32"`. uint16 is physically impossible here — vocab is 151,936 > 65,535.
```


### 8.2 What is the total disk footprint of everything the find pattern matches (weights + token caches + dataset shards)?

**Value**

```
315,669,745,533 bytes = 293.99 GiB = 315.67 GB total, split: weights 290.64 GB (n=107), tokcache_*.pt token caches 23.92 GB (n=26), research/datasets/*.bin shards 1.11 GB (n=10). Volume has 2.5T free of 3.7T.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/ (tokcache files), research/datasets/`

**Source quote**

```
tokcache   n= 26 bytes=23,921,715,722  GiB=22.28  GB=23.92
dataset    n= 10 bytes=1,105,025,768  GiB=1.03  GB=1.11
```

**Confidence** — measured from code

**Caveat** — tokcache_*.pt are torch.save'd token-id tensors produced by train_qwen3.py, NOT model weights. The largest single .pt in the repo (9,534,229,373 B = 9.09 GiB, tokcache_1191478400_300000.pt) is a token cache, not a model.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Totals exact. But the largest .pt is 9,534,229,373 B = 8.88 GiB = 9.53 GB, NOT 9.09 GiB.
```

**Verifier note**

```
Re-measured: total 315,669,745,533 B = 293.99 GiB = 315.67 GB EXACT; tokcache n=26 = 23,921,715,722 B (22.28 GiB / 23.92 GB) EXACT; dataset n=10 = 1,105,025,768 B (1.03 GiB / 1.11 GB) EXACT. `df -h` on /dev/nvme0n1p2: 3.7T size, 2.5T avail, 30% used — CONFIRMED. Largest .pt confirmed as Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/tokcache_1191478400_300000.pt at 9,534,229,373 B, but 9,534,229,373 / 2^30 = 8.879 GiB and / 1e9 = 9.534 GB — the quoted '9.09 GiB' matches neither unit. Producer confirmed: Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/train_qwen3.py:145 `cache = RESULTS / f"tokcache_{n_train}_{n_val}_seed{seed}_{tok_tag}.pt"`. Minor: the shorthand 'research/datasets/*.bin' matches zero files literally — they live at research/datasets/<name>/shard_*.bin and .../eval/*.bin.
```


### 8.3 Per-run-directory breakdown of the weights footprint

**Value**

```
18 files 55.60 GiB HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build; 17 files 56.64 GiB Qwen3-0.6B/experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1; 21 files 23.32 GiB Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results; 9 files 29.99 GiB 2026-06-21_qwen3-0.6b_arch-subdrill-p2; 6 files 19.99 GiB 2026-06-27_qwen3-0.6b_sft-3seed; 6 files 19.99 GiB 2026-06-30_qwen3-0.6b_midtrain-anneal; 5 files 16.66 GiB Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b; 11 files 12.22 GiB builds/2026-06-08_reproduce-modernized.../results; 3 files 9.99 GiB 2026-06-24_data-dclm-vs-fineweb; 3 files 9.99 GiB 2026-06-26_data-mix-composition; 3 files 9.99 GiB 2026-07-02_grpo-phase2; 1 file 3.33 GiB 2026-06-17_vibethinker.../results; 2 files 2.22 GiB builds/2026-06-08_reproduce-exploratory.../results; 2 files 0.75 GiB SmolLM2-134(base)
```

**Evidence** — `Qwen3-0.6B/experiments/ , HybridSSM-0.2B/experiments/ , SmolLM2-134(base)/`

**Source quote**

```
18 files     55.60 GiB  HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build
 21 files     23.32 GiB  Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results
 17 files     56.64 GiB  Qwen3-0.6B/experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1
```

**Confidence** — measured from code

**Caveat** — The HybridSSM arch-ladder run (2026-07-21_..._arch-ladder) has ZERO checkpoints in its own directory — run_arch_ladder.sh cd's into the 2026-07-19 build dir and writes them there. Likewise the scaling-persistence ladder writes into the normuon-vs-adamw results dir. Directory name does not equal owning run.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
All 14 rows reproduce EXACTLY from an independent os.walk aggregation (file counts and GiB to 2dp). Caveat also verified: HybridSSM-0.2B/experiments/2026-07-21_hybrid-ssm-0.2b_arch-ladder/run_arch_ladder.sh:19 `BUILD="$ROOT/HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build"` and :392 `( cd "$BUILD" && $PY train_hybrid.py ...` with :395 `--ckpt "checkpoint_${id}.pkl"` — the ladder writes into the 2026-07-19 dir. Likewise Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/run_ladder_scale_ext.sh:83 `RESULTS=$IMU1/results` points at the 2026-06-16 dir.
```


### 8.4 Is there a config.json anywhere alongside the checkpoints?

**Value**

```
NOT_FOUND. Zero config.json files exist in the entire repo (excluding .git). Architecture config is embedded INSIDE each checkpoint dict under the key 'config' (a plain dict of the dataclass fields), not as a sidecar file.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/checkpoint_persist_420M_normuon_s0.pt (in-file key 'config')`

**Source quote**

```
config: {
 "vocab_size": 151936,
 "hidden_size": 1024,
 "intermediate_size": 3072,
 "num_hidden_layers": 28,
 "num_attention_heads": 16,
 "num_key_value_heads": 8,
 "head_dim": 128,
 "max_position_embeddings": 40960,
 "rope_theta": 1000000.0,
 "rms_norm_eps": 1e-06,
 "initializer_range": 0.02,
 "tie_word_embeddings": true,
 "attention_bias": false,
 "attention_dropout": 0.0
}
```

**Confidence** — measured from code

**Caveat** — Searched with: find . -name 'config.json' -not -path './.git/*' — returned nothing. Any HF upload must synthesize config.json from the in-checkpoint dict.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
`find . -path ./.git -prune -o -name 'config.json' -print` returned nothing. I loaded Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/checkpoint_persist_420M_normuon_s0.pt (torch.load map_location='cpu', weights_only=True) and its ck['config'] is byte-for-byte the quoted dict: vocab_size 151936, hidden_size 1024, intermediate_size 3072, num_hidden_layers 28, num_attention_heads 16, num_key_value_heads 8, head_dim 128, max_position_embeddings 40960, rope_theta 1000000.0, rms_norm_eps 1e-06, initializer_range 0.02, tie_word_embeddings true, attention_bias false, attention_dropout 0.0. EXACT match.
```


### 8.5 Are tokenizer files vendored in the repo, or downloaded from HF at runtime?

**Value**

```
NOT vendored — downloaded from HF Hub at runtime. Zero tokenizer.json / tokenizer_config.json / vocab.json / merges.txt / *.model / special_tokens_map.json / generation_config.json exist anywhere in the repo, and no HF cache dir lives inside it. Qwen3 + HybridSSM use AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base'); SmolLM2 uses 'HuggingFaceTB/SmolLM2-135M'.
```

**Evidence** — `Qwen3-0.6B/verify.py:19 ; SmolLM2-134(base)/verify.py:19 ; HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/eval_suite_jax.py:87`

**Source quote**

```
Qwen3-0.6B/verify.py:19: REPO = "Qwen/Qwen3-0.6B-Base"
SmolLM2-134(base)/verify.py:19: REPO = "HuggingFaceTB/SmolLM2-135M"
eval_suite_jax.py:87: TOKENIZER_REPO = "Qwen/Qwen3-0.6B-Base"  # model's OWN tokenizer (data cache built with it)
```

**Confidence** — measured from code

**Caveat** — HybridSSM-0.2B is a NOVEL from-scratch architecture that nonetheless uses the Qwen3 tokenizer (vocab_size 151,936 in HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/model.py:22). Any weights upload inherits Qwen's tokenizer licensing, and reproduction requires network access to HF.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Tokenizer-file absence and all three line citations are correct. But 'no HF cache dir lives inside it' is false: research/datasets/data-selection-dclm-edu/.raw/.cache/huggingface/ exists (28 KB).
```

**Verifier note**

```
All three cited lines verified verbatim at the exact line numbers: Qwen3-0.6B/verify.py:19, SmolLM2-134(base)/verify.py:19, HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/eval_suite_jax.py:87 (comment text matches too). find for tokenizer.json/tokenizer_config.json/vocab.json/merges.txt/*.model/special_tokens_map.json/generation_config.json returned ZERO. HybridSSM vocab_size 151_936 confirmed at HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/model.py:22. HOWEVER an HF cache directory DOES exist inside the repo: research/datasets/data-selection-dclm-edu/.raw/.cache/huggingface/{download/data/000_00000.parquet.metadata, CACHEDIR.TAG, .gitignore}. It holds dataset-download metadata only — no tokenizer or model files — so the substantive conclusion (tokenizer must be fetched from the Hub) survives, but the blanket 'no HF cache dir' wording must be dropped.
```


### 8.6 What is the on-disk format and top-level structure of the Qwen3 checkpoints?

**Value**

```
Raw torch.save'd Python dicts (NOT safetensors, NOT bare state_dicts). Two variants: (a) WEIGHTS-ONLY ~1,192,232,687 B / 1137 MiB — keys ['model','config','step','tok_seen','arm','seed','fineweb_val_ppl','baseline_ppl','recipe']; (b) FULL TRAINING STATE ~3,576,xxx,xxx B / 3411 MiB — adds ['optim','sched','rng_torch','rng_cuda','rng_numpy','rng_python']. model is a 311-key state_dict, all torch.bfloat16, 751,632,384 tensor elements including the tied lm_head.weight duplicate = 596,049,920 unique params.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/checkpoint_persist_420M_normuon_s0.pt ; Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/checkpoint_sft_seed0.pt`

**Source quote**

```
checkpoint_persist_420M_normuon_s0.pt | keys= ['model','config','step','tok_seen','arm','seed','fineweb_val_ppl','baseline_ppl','recipe']
   model: dict len=311 ... model.embed_tokens.weight: Tensor (151936, 1024) torch.bfloat16
   total(incl tied dup): 751632384 minus embed: 596049920
checkpoint_sft_seed0.pt keys: ['model','config','step','sample_cursor','tok_seen','base_reasoning_ppl','recipe','optim','sched','rng_torch','rng_cuda','rng_numpy','rng_python']
```

**Confidence** — measured from code

**Caveat** — The 3.4 GB variants FAIL torch.load(weights_only=True) — they pickle numpy RNG state (UnpicklingError: 'Unsupported global: GLOBAL numpy._core.multiarray._reconstruct'). I loaded them under torch.serialization.safe_globals([numpy._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]) with weights_only=True still enforced. The 1.1 GB weights-only variants load cleanly with weights_only=True. Any external consumer must be warned they are pickles, not safetensors.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Verified by loading four files CPU-only. checkpoint_persist_420M_normuon_s0.pt: 1,192,233,319 B (=1137.0 MiB), keys exactly the (a) list, 311 keys, all torch.bfloat16, 751,632,384 elems, minus one embed = 596,049,920. checkpoint_sft_seed0.pt (Qwen3-0.6B/experiments/2026-06-27_qwen3-0.6b_sft-3seed/): 3,576,718,301 B (=3411.0 MiB), keys ['model','config','step','sample_cursor','tok_seen','base_reasoning_ppl','recipe','optim','sched','rng_torch','rng_cuda','rng_numpy','rng_python'] — EXACT match to the quote. Raw weights_only=True raised UnpicklingError whose text I captured verbatim: 'WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray._reconstruct was not an allowed global by default.' Loaded successfully under torch.serialization.safe_globals([...]) with weights_only=True still on.
```


### 8.7 What is the on-disk format and structure of the HybridSSM (JAX/Flax) checkpoints?

**Value**

```
Python pickle wrapping flax msgpack byte-strings, written by train_hybrid.py:132-139. Top-level dict keys: {'params': bytes (flax msgpack), 'opt_state': bytes (flax msgpack, exactly 2x the params size), 'step': int, and (post-fix only) 'rng': numpy uint32[2]}. Params deserialize (flax.serialization.msgpack_restore) to a nested dict keyed block_0..block_23 + embed + norm_f, all float32, weight-tied (no separate lm_head).
```

**Evidence** — `HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/train_hybrid.py:132-139`

**Source quote**

```
def save(step_i):
        blob = {"params": serialization.to_bytes(params), "opt_state": serialization.to_bytes(opt_state),
                "step": step_i, "rng": np.asarray(rng)}
        tmp = a.ckpt + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(blob, f)
```

**Confidence** — measured from code

**Caveat** — Two-thirds of every HybridSSM file is Adam optimizer state, not weights. E.g. checkpoint_ssm_base_42M_s0.pkl is 3,669,849,670 B total but params is only 1,223,283,175 B. Stripping opt_state before upload cuts the family from 55.6 GiB to roughly 18-19 GiB.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Structure fully confirmed. Params-only sum is 18,792,709,255 B = 17.50 GiB (18.79 GB), not '18-19 GiB' — a GiB/GB slip against the 55.60 GiB baseline.
```

**Verifier note**

```
train_hybrid.py save() verified at lines 132-139 (132 `def save(step_i):`, 135-136 blob, 137 tmp, 138 open, 139 pickle.dump) — NOTE the quote silently elides the two comment lines 133-134, so it is not a contiguous verbatim excerpt. Deserialized 4 files with flax.serialization.msgpack_restore: top-level tree has n=26 (block_0..block_23 + embed + norm_f), all float32, single tied '/embed' (151936,768) with no separate lm_head. opt_state_bytes is exactly 2x params_bytes in all four (e.g. ssm_base_42M_s0: 1,223,283,175 -> 2,446,566,418). checkpoint_ssm_base_42M_s0.pkl total 3,669,849,670 B and params 1,223,283,175 B — both EXACT. Summing params_bytes over all 18 pkl-family files gives 18,792,709,255 B = 17.50 GiB.
```


### 8.8 Exact parameter counts of the HybridSSM arms (measured by deserializing the flax tree)

**Value**

```
ssm_base = 305,818,368 params (266 leaves); attn1to3 = 324,867,840 (290 leaves); fullattn = 267,719,424 (218 leaves); swa128 = 277,156,608 (194 leaves). All float32, tied embedding (151,936 x 768 = 116,686,848 of ssm_base's total). Architecture: d_model 768, n_layers 24, n_heads 12, n_kv_heads 4, vocab 151,936.
```

**Evidence** — `HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/model.py:21-26`

**Source quote**

```
class HybridConfig:
    vocab_size: int = 151_936
    d_model: int = 768
    n_layers: int = 24
    n_heads: int = 12
    n_kv_heads: int = 4
```

**Confidence** — measured from code

**Caveat** — The folder is named 'HybridSSM-0.2B' but every arm on disk is 267M-325M total params. 0.2B appears to refer to non-embedding params (ssm_base: 305,818,368 - 116,686,848 = 189,131,520). I did NOT find a file stating which convention '-0.2B' uses — do not publish '0.2B' as a total-parameter claim without resolving it.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Independently deserialized checkpoint_{ssm_base,attn1to3,fullattn,swa128}_42M_s0.pkl and counted leaves/elements: 305,818,368/266; 324,867,840/290; 267,719,424/218; 277,156,608/194 — ALL EXACT. All float32; '/embed' is (151936, 768) = 116,686,848 in every arm with no lm_head leaf. HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/model.py:21-26 quoted verbatim and correct. Non-embedding arithmetic 305,818,368 - 116,686,848 = 189,131,520 verified. The '-0.2B' naming caveat is properly flagged as unresolved — I also found no file on disk stating the convention.
```


### 8.9 CRITICAL: do the '42M' / '85M' / '168M' / '420M' filename tokens mean parameters?

**Value**

```
NO — they are TOKEN BUDGETS, not parameter counts. HybridSSM cells.json records rung_base_tokens 42000000 / 85000000 / 150000000. Qwen3 checkpoint_persist_168M_adamw_s0.pt carries tok_seen=168,034,304 with 596,049,920 params; checkpoint_persist_420M_normuon_s0.pt carries tok_seen=420,020,224 with the same 596,049,920 params.
```

**Evidence** — `HybridSSM-0.2B/experiments/2026-07-21_hybrid-ssm-0.2b_arch-ladder/cells.json`

**Source quote**

```
{'id': 'ssm_base_42M_s0', 'arm': 'ssm_base', 'seed': 0, 'rung_base_tokens': 42000000, 'tokens': 42000000, 'steps': 5126, ...}
```

**Confidence** — measured from code

**Caveat** — This is the single easiest thing to get wrong when writing a model card. Every 'persist_*' Qwen3 checkpoint is the SAME 596M-param model at a different token budget.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
HybridSSM-0.2B/experiments/2026-07-21_hybrid-ssm-0.2b_arch-ladder/cells.json cells[0] is byte-identical to the quote: {'id': 'ssm_base_42M_s0', 'arm': 'ssm_base', 'seed': 0, 'rung_base_tokens': 42000000, 'tokens': 42000000, 'steps': 5126, ...}. The full set of (rung_base_tokens, tokens) pairs across its 15 cells is {(42000000,42000000),(42000000,48000000),(85000000,85000000),(85000000,96000000),(150000000,150000000),(150000000,170000000)} — the three rungs 42M/85M/150M confirmed. Loaded checkpoint_persist_168M_adamw_s0.pt: step=2564, tok_seen=168034304, 751,632,384 elems -> 596,049,920 unique. checkpoint_persist_420M_normuon_s0.pt: step=6409, tok_seen=420020224, identical param count. This is the single most important fact in the set and it is fully backed.
```


### 8.10 Which checkpoint corresponds to the parity-verified (bit-exact) Qwen3-0.6B / SmolLM2 reproduction?

**Value**

```
NOT_FOUND — no such checkpoint exists on disk, by design. Both verify.py scripts download the official HF weights at runtime and load them into the repo's model.py; nothing is saved. The parity EVIDENCE that exists on disk is Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/verify.json (max_abs_error 0.0, argmax_match true, passed true) and SmolLM2-134(base)/results/parity.log (max |Δlogits| = 0.000e+00).
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/verify.json`

**Source quote**

```
{
  "repo": "Qwen/Qwen3-0.6B-Base",
  "max_abs_error": 0.0,
  "relative_error": 0.0,
  "hf_next_token_id": 12095,
  "our_next_token_id": 12095,
  "argmax_match": true,
  "passed": true
}
```

**Confidence** — results JSON

**Caveat** — PUBLISHING IMPLICATION: there is nothing to upload for 'the parity-verified repro' — the weights are Qwen's / HuggingFaceTB's, already on the Hub. What is publishable is the code + the parity artifact, not a weights file. Do not describe any .pt in this repo as 'the bit-exact reproduction weights'.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
verify.json read in full — quoted fields are exact (repo 'Qwen/Qwen3-0.6B-Base', max_abs_error 0.0, relative_error 0.0, hf_next_token_id 12095, our_next_token_id 12095, argmax_match true, passed true; also prompt 'The capital of France is', dtype float32, tolerance 0.001). SmolLM2-134(base)/results/parity.log contains 'max |Δlogits| = 0.000e+00' and 'relative = 0.000e+00' and '✓ Architecture parity verified.' Both verify.py scripts fetch weights at runtime (Qwen3-0.6B/verify.py:51 AutoModelForCausalLM.from_pretrained(REPO); SmolLM2-134(base)/verify.py:53 same) and save nothing. The publishing implication is correctly stated.
```


### 8.11 Which checkpoint is the continued-pretrained SmolLM2?

**Value**

```
SmolLM2-134(base)/checkpoint_tinystories.pt — 269,144,681 B (256.68 MB), mtime 2026-05-14 00:21. bf16, 273-key state_dict, 134,515,008 unique params (162,826,560 incl. tied lm_head duplicate). In-file: step=24414, tok_seen=99,999,744, baseline_ppl=6.894546783281595, trained_ppl=3.7899503859716885. Produced by train_tinystories.py, which initializes FROM the official HF weights (line 39 imports load_official_weights_into_ours from verify.py; line 145 loads the HF model).
```

**Evidence** — `SmolLM2-134(base)/train_tinystories.py:39,145`

**Source quote**

```
39:from verify import REPO, load_official_weights_into_ours
145:    hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
154:    train_ds = load_dataset("roneneldan/TinyStories", split="train")
```

**Confidence** — measured from code

**Caveat** — This is a DERIVATIVE of HuggingFaceTB/SmolLM2-135M (Apache-2.0 upstream) — publishing it is a fine-tune release, and the model card must say so. The in-checkpoint numbers (6.8945 -> 3.7893) match SmolLM2-134(base)/results/tinystories_summary.md exactly, so that prose is backed.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
Every stated value is exact, but the caveat's 'match exactly' is false: in-checkpoint trained_ppl = 3.78995 (rounds to 3.7900); results/tinystories_summary.md:9 reports 3.7893. Δ = 0.00065.
```

**Verifier note**

```
Loaded the file: size 269,144,681 B, mtime 2026-05-14 00:21:27, keys ['model','config','step','tok_seen','baseline_ppl','trained_ppl'], 273 keys all torch.bfloat16, 162,826,560 elems - one embed (49152x576) = 134,515,008. step=24414, tok_seen=99999744, baseline_ppl=6.894546783281595, trained_ppl=3.7899503859716885 — ALL EXACT. Provenance lines verified verbatim: train_tinystories.py:39 `from verify import REPO, load_official_weights_into_ours`, :145 `hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)`, :154 `train_ds = load_dataset("roneneldan/TinyStories", split="train")`. BUT tinystories_summary.md:9 reads `| TinyStories-val perplexity | **6.8945** | **3.7893** | **−45.0%** |` — baseline matches to 4dp, trained does NOT (3.7893 vs 3.78995). grep for '3.789' across the subproject found only that one line, so there is no second file carrying 3.78995. A model card must not claim the prose is byte-backed.
```


### 8.12 What is SmolLM2-134(base)/checkpoint.pt (the other SmolLM2 file)?

**Value**

```
A FROM-SCRATCH random-init toy run — NOT a reproduction and NOT publishable as SmolLM2 weights. 538,173,921 B (513.24 MB), mtime 2026-05-13 22:20, float32, step=150, 150 loss values. train.py:140 prints 'Initializing model from scratch (random init)' and trains on wikitext-103-raw-v1 (train.py:78).
```

**Evidence** — `SmolLM2-134(base)/train.py:78,140`

**Source quote**

```
78:    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
140:    print("Initializing model from scratch (random init)...")
```

**Confidence** — measured from code

**Caveat** — 150 steps of random init is a demo artifact. It has no 'training_recipe' key even though the current train.py:182 save_ckpt writes one — i.e. the file predates the current script (ckpt mtime 2026-05-13 22:20 vs train.py mtime 2026-05-19 23:17). Same script-drift applies to checkpoint_tinystories.pt (2026-05-14 vs train_tinystories.py 2026-05-19). Provenance is not byte-reproducible from HEAD.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Loaded: 538,173,921 B, mtime 2026-05-13 22:20:18, keys ['model','config','losses','lrs','step'], step=150, len(losses)=150, all torch.float32, 273 keys, 134,515,008 unique params. 'training_recipe' absent — CONFIRMED. Dataset config check (the classic -raw- trap): SmolLM2-134(base)/train.py:78 literally reads `ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")` — the claimed config name is the one the script actually names. train.py:140 `print("Initializing model from scratch (random init)...")` verbatim. train.py:181 `def save_ckpt(step: int):` / :182 `torch.save({` / :186 `"training_recipe": {` — the recipe-writing claim is right (the block starts at 182, the key at 186). Script-drift dates confirmed by ls: train.py 2026-05-19 23:17, train_tinystories.py 2026-05-19 23:16, checkpoints 2026-05-13/14.
```


### 8.13 Which checkpoints back the headline NorMuon-vs-AdamW 'win' (2026-06-16, verdict=win)?

**Value**

```
6 files in Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/: checkpoint_{adamw,normuon}_seed{0,1,2}.pt, each 1,192,229,527-1,192,230,159 B (1137.00 MB), mtime 2026-06-17 01:20-09:46. Verified in-file: step=640, tok_seen=41,943,040, arm='normuon'/'adamw', seed=0..2. checkpoint_normuon_seed0 fineweb_val_ppl=61.3435; checkpoint_adamw_seed0 fineweb_val_ppl=147.4181.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/checkpoint_normuon_seed0.pt (in-file keys)`

**Source quote**

```
checkpoint_normuon_seed0.pt | step 640 tok_seen 41943040 arm normuon seed 0 ppl 61.34354760675183
checkpoint_adamw_seed0.pt   | step 640 tok_seen 41943040 arm adamw   seed 0 ppl 147.41809644622612
```

**Confidence** — measured from code

**Caveat** — The ledger's headline for this run is BPB not the in-file fineweb PPL: wikitext_bpb_normuon_mean 1.6355 vs adamw 2.1098 (research/ledger/ledger.json, run 2026-06-16_qwen3_normuon-vs-adamw). Also present in the same directory: 3 LR-sweep checkpoints (checkpoint_adamw_lr{17,35,48}_seed0.pt) that are sweep artifacts, not arms.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
ls -l on the results dir: adamw_seed{0,1,2} all 1,192,229,527 B at 2026-06-17 01:20/02:54/04:29; normuon_seed{0,1,2} all 1,192,230,159 B at 06:14/08:01/09:46 — size range and mtime span EXACT. Loaded both seed0 files: step=640, tok_seen=41943040, arm='normuon'/'adamw', seed=0, fineweb_val_ppl 61.34354760675183 and 147.41809644622612 — EXACT. Ledger run 2026-06-16_qwen3_normuon-vs-adamw (status done, verdict win) metrics: wikitext_bpb_adamw_mean 2.1098, wikitext_bpb_normuon_mean 1.6355, wikitext_improvement_bpb 0.4743, wikitext_ci95 [0.4435,0.5052], code_improvement_bpb 0.5016 — the caveat's BPB pair is EXACT. checkpoint_adamw_lr{17,35,48}_seed0.pt confirmed present (1,192,231,107 B each, 2026-06-17 13:51/15:28/17:03).
```


### 8.14 Which checkpoints back the scaling-persistence ladder (verdict=null, NorMuon win converges)?

**Value**

```
12 files in Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/: checkpoint_persist_{168M,420M}_{adamw,normuon}_s{0,1,2}.pt, each 1,192,232,687 / 1,192,233,319 B (1137.00 MB), mtimes 2026-07-06 to 2026-07-26. Together with the 6 x 42M cohort files they are the 18 cells the scorer reads. ladder_bpb.json names all 18 by filename.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence/run_ladder_scale_ext.sh:80-83`

**Source quote**

```
IMU1=$ROOT/Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw
LDIR=$ROOT/Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence
TRAIN=$IMU1/train_ablation.py
RESULTS=$IMU1/results
```

**Confidence** — measured from code

**Caveat** — Directory/run mismatch: the run_id is 2026-07-05_qwen3-0.6b_scaling-persistence and 2026-07-23_qwen3-0.6b_normuon-at-scale, but the CHECKPOINTS live under the 2026-06-16 experiment. The 2026-07-05 dir contains only .done markers, logs and c5_evidence. The ledger entry for normuon-at-scale explicitly records absolute checkpoint paths pointing back to .../2026-06-16_qwen3_normuon-vs-adamw/results/.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
ls confirms exactly 12 persist_* files: adamw variants 1,192,232,687 B, normuon variants 1,192,233,319 B; mtime span 2026-07-06 00:36 to 2026-07-26 13:18. A regex sweep of Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw/results/ladder_bpb.json returned exactly 18 unique checkpoint_*.pt filenames — the 12 persist plus checkpoint_{adamw,normuon}_seed{0,1,2}.pt. run_ladder_scale_ext.sh lines 80-83 are verbatim at those exact line numbers (ROOT is line 79). Ledger run 2026-07-05_qwen3-0.6b_scaling-persistence is status=done verdict=null with trend_verdict_wikitext=CONVERGES and trend_code_py=CONVERGES. Minor imprecision: the 2026-07-05 directory holds more than 'only .done markers, logs and c5_evidence' — it also contains verdict.json, run_ladder_scale_ext.sh, boot_resume.sh, RESUME_STATE.md, thermal_log.py. The load-bearing claim (zero checkpoints there) is correct.
```


### 8.15 Which checkpoints are flagged DISCARDED / CONFOUNDED and must not be published?

**Value**

```
(1) HARD-QUARANTINED, 1 file: HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/checkpoint_swa128_nope_85M_s0.pkl.discarded_rng_confound_20260723 (3,325,901,386 B, mtime 2026-07-23 15:09) — renamed out of the .pkl namespace for an RNG-restore confound. (2) COMPARABILITY-VOID (iso-FLOP error, +18.88% extra compute), 3 completed files: checkpoint_swa128_42M_s0.pkl, checkpoint_swa128_nope_42M_s0.pkl, checkpoint_swa128_85M_s0.pkl. Their replacements are checkpoint_swa128_42M_isofix_s0.pkl and checkpoint_swa128_nope_42M_isofix_s0.pkl (step 4929, mtimes 2026-07-29/30).
```

**Evidence** — `HybridSSM-0.2B/experiments/2026-07-21_hybrid-ssm-0.2b_arch-ladder/c5_evidence_CORRECTION_2026-07-28.md:303-307,326-330`

**Source quote**

```
**Affected — comparability void:**
- `swa128_42M_s0`, `swa128_nope_42M_s0` (both complete, `.done` on disk)
- `swa128_85M_s0` (complete, 11,718 steps — same budget error at the 85M rung ...)
- `swa128_nope_85M_s0` (killed at ~step 7,960 ... checkpoint already discarded for a separate RNG confound)
...
**The words "iso-FLOP" must not be attached to `swa128` or `swa128_nope` in any artifact until those cells are re-run at 4,929 steps / 40,378,368 tokens**
```

**Confidence** — measured from code

**Caveat** — There is a SECOND, undocumented split I found by inspecting file tails: 10 HybridSSM pickles lack the 'rng' key (pre-PRNG-fix) and 8 carry it. Pre-fix (no rng): checkpoint.pkl(smoke), ssm_base_s0, ssm_base_42M_s0, ssm_base_85M_s0, swa128_42M_s0, swa128_85M_s0, swa128_nope_42M_s0, attn1to3_42M_s0, fullattn_42M_s0, and the quarantined 85M. Post-fix (has rng): ssm_base_42M_s1/s2, attn1to3_42M_s1/s2, fullattn_42M_s1/s2, swa128_42M_isofix_s0, swa128_nope_42M_isofix_s0. The quarantine rationale (RNG confound) applies structurally to every pre-fix file that was RESUMED; only the 85M one was actually quarantined. Flag this to the user before publishing any pre-fix seed-0 arm.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
This is the strongest fact in the set. The quarantined file is 3,325,901,386 B at mtime 2026-07-23 15:09 — EXACT. c5_evidence_CORRECTION_2026-07-28.md quote verified verbatim: line 302 '**Affected — comparability void:**', 303-307 the swa128 list, 317-318 '+18.88 % extra compute relative to the base arm', 326-328 'The words "iso-FLOP" must not be attached to `swa128` or `swa128_nope` in any artifact until those cells are re-run at 4,929 steps / 40,378,368 tokens'. (Cited range 303-307 actually starts at 302; 326-330 ends at 328 — trivial off-by-one.) I independently unpickled all 18 files and the rng partition matches the claim EXACTLY, file for file: 10 without 'rng' (checkpoint.pkl step30, ssm_base_s0 step21156, ssm_base_42M_s0 step5126, ssm_base_85M_s0 step10375, swa128_42M_s0 step5859, swa128_85M_s0 step11718, swa128_nope_42M_s0 step5859, attn1to3_42M_s0 step5126, fullattn_42M_s0 step5126, and the .discarded file step7800) and 8 with 'rng' (ssm_base_42M_s1/s2, attn1to3_42M_s1/s2, fullattn_42M_s1/s2, swa128_42M_isofix_s0, swa128_nope_42M_isofix_s0). Both isofix files are step=4929, matching the correction doc's required budget.
```


### 8.16 Which checkpoints are smoke-test artifacts (never a result)?

**Value**

```
7 files, 18.88 GiB total. In Qwen3-0.6B/experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1/: smoke_baseline.pt, smoke_baseline_resumed.pt, smoke_wsd.pt, smoke_zloss.pt, smoke_arch.pt (5 files, 16.66 GiB, all mtime 2026-06-18 13:31-13:40). In Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/: checkpoint_imu1_smoke_step500.pt and checkpoint_imu1_smoke_step1000.pt (1,193,196,283 / 1,193,196,711 B, mtime 2026-06-09). Plus HybridSSM checkpoint.pkl (66,064,696 B, step=30, smoke).
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1/smoke_baseline.pt`

**Source quote**

```
-rw-rw-r-- 1 yashb98 yashb98 3576681053 Jun 18 13:31 smoke_baseline.pt
```

**Confidence** — measured from code

**Caveat** — These are §C5.0 smoke-test outputs (1 step on a tiny batch). 18.88 GiB of pure deletable overhead. Never publishable.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
ls -l verified: smoke_baseline.pt 3,576,681,053 B 2026-06-18 13:31 (the quoted line, exact), smoke_wsd.pt 3,576,674,813 13:33, smoke_zloss.pt 3,576,677,309 13:36, smoke_arch.pt 3,579,557,149 13:39, smoke_baseline_resumed.pt 3,576,694,557 13:40 — sum 17,886,284,881 B = 16.657 GiB, mtime window 13:31-13:40, EXACT. checkpoint_imu1_smoke_step500.pt 1,193,196,283 B and _step1000.pt 1,193,196,711 B, both mtime 2026-06-09 — EXACT. 7-file sum = 20,272,677,875 B = 18.88 GiB, EXACT (the HybridSSM pkl is correctly excluded from the 7/18.88 and listed as 'plus'). HybridSSM checkpoint.pkl: 66,064,696 B, unpickled step=30, no rng — EXACT.
```


### 8.17 Are any checkpoints tracked in git?

**Value**

```
ZERO. `git ls-files | grep -cE '\.(pt|pth|bin|safetensors|ckpt|pkl|msgpack|npz)$'` returns 0. Both .gitignore files exclude them.
```

**Evidence** — `.gitignore:19-24 ; HybridSSM-0.2B/.gitignore:2-3`

**Source quote**

```
.gitignore:19: # Checkpoints (270MB+ each; not for version control — use HF Hub or git-lfs)
.gitignore:20: *.pt
.gitignore:21: *.pth
.gitignore:22: *.safetensors
.gitignore:23: *.bin
.gitignore:24: *.ckpt
HybridSSM-0.2B/.gitignore:2: *.pkl
```

**Confidence** — measured from code

**Caveat** — MEMORY.md records a prior incident where gitignored-but-force-added files were DELETED from the working tree on branch checkout (guard_branch_switch_wipes_gitignored_evidence). Current branch is harden-research-loop; do not force-add checkpoints to publish them.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Ran the exact command: output 0. .gitignore lines 19-24 are verbatim at those numbers: 19 '# Checkpoints (270MB+ each; not for version control — use HF Hub or git-lfs)', 20 '*.pt', 21 '*.pth', 22 '*.safetensors', 23 '*.bin', 24 '*.ckpt'. HybridSSM-0.2B/.gitignore:2 '*.pkl', :3 '*.pkl.tmp' — both verbatim. The branch-checkout caveat is a live risk worth keeping.
```


### 8.18 Is there tooling on disk to convert a checkpoint to HF/safetensors format?

**Value**

```
Exactly ONE script, and it covers only SmolLM2: SmolLM2-134(base)/scripts/export_to_hf.py. It loads the .pt, round-trips through SmolLM2ForCausalLM, copies into a HF LlamaForCausalLM built from AutoConfig.from_pretrained('HuggingFaceTB/SmolLM2-135M'), then save_pretrained(safe_serialization=True) + tokenizer.save_pretrained. No equivalent exists for Qwen3-0.6B or HybridSSM-0.2B. No hf_export/ directory exists on disk.
```

**Evidence** — `SmolLM2-134(base)/scripts/export_to_hf.py:56-67`

**Source quote**

```
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
    cfg = AutoConfig.from_pretrained(args.repo)
    hf = LlamaForCausalLM(cfg)
    missing, unexpected = hf.load_state_dict(ours_sd, strict=False)
...
    hf.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.repo)
    tok.save_pretrained(out)
```

**Confidence** — measured from code

**Caveat** — The script requires network (AutoConfig/AutoTokenizer from the Hub) and has NEVER been run to completion on this box as far as disk shows — hf_export/ does not exist. Uploading Qwen3 or HybridSSM weights requires writing a new exporter; HybridSSM especially, since it is a novel flax architecture with no HF modelling class at all.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
export_to_hf.py lines 56-59 and 65-67 quoted verbatim and correct (56 `from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM`, 57 cfg=AutoConfig.from_pretrained(args.repo), 58 hf=LlamaForCausalLM(cfg), 59 load_state_dict(..., strict=False), 65 hf.save_pretrained(out, safe_serialization=True), 66-67 tokenizer save). The elision '...' between 59 and 65 skips lines 60-64, which include a real guard: `missing = [k for k in missing if k != "lm_head.weight"]` then `raise SystemExit` on any other mismatch. A repo-wide grep for save_pretrained/save_file over all *.py found these as the ONLY save calls — every other hit is prose or from_pretrained. `find -type d -name hf_export` returned nothing (note: hf_export/ is gitignored at .gitignore:27, so its absence proves only that it is not on disk now).
```


### 8.19 Do the modernized/exploratory Qwen3 build checkpoints load into stock HF Qwen3?

**Value**

```
NO for modernized. checkpoint_imu1_2tpp_step18000.pt has 752,091,220 tensor elements (vs 751,632,384 for faithful) and its embedded config adds non-HF keys including 'use_value_residual': true and 'use_layernorm_sc...'. checkpoint_prope10_2tpp_.pt (exploratory) is 751,632,384 elements with config key 'partial_rotary_factor': 0.1 — that key IS supported by HF Qwen3. Both are weights-only dicts with keys ['model','config','step'] and carry NO tok_seen.
```

**Evidence** — `Qwen3-0.6B/builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/checkpoint_imu1_2tpp_step18000.pt (in-file 'config')`

**Source quote**

```
keys ['model', 'config', 'step']
  step 18000 tok_seen None params 752091220
  config {..., "tie_word_embeddings": true, "attention_bias": false, "attention_dropout": 0.0, "use_value_residual": true, "use_layernorm_sc...
```

**Confidence** — measured from code

**Caveat** — The modernized family is 11 intermediate step-checkpoints (step500,1000,2000..18000) of ONE run, 12.22 GiB — at most one (step18000) is publishable; the rest are training-curve snapshots. Exploratory is 2 files (prope10, prope25) at step 4000.

**Verdict — ❌ WRONG**

**Corrected value**

```
'partial_rotary_factor IS supported by HF Qwen3' is FALSE on this box. transformers 5.8.0 Qwen3 never reads it, so the exploratory checkpoint would load into stock Qwen3 and silently run FULL RoPE — architecturally wrong, not portable. Also the modernized checkpoint is 423 keys prefixed '_orig_mod.' (torch.compile), not 311 — a second, unmentioned blocker.
```

**Verifier note**

```
Loaded both files. checkpoint_imu1_2tpp_step18000.pt: 1,193,196,711 B, keys ['model','config','step'], step 18000, 752,091,220 elems, config adds use_value_residual:true, use_layernorm_scaling:true, use_head_gating:true, no tok_seen — the 'NO for modernized' verdict is right and understated: its state_dict has 423 keys named `_orig_mod.model.embed_tokens.weight` etc. checkpoint_prope10_2tpp_.pt: 1,192,229,775 B, step 4000, 751,632,384 elems, config partial_rotary_factor 0.1 — all correct. BUT I disproved the HF-support claim three ways: (1) grep -rn 'partial_rotary_factor' over site-packages/transformers/models/qwen3/ returns NOTHING (it appears only in laguna/moonshine/etc.); (2) inspect.signature(Qwen3Config.__init__) has no such parameter and modeling_qwen3.py source never mentions it; (3) empirically, Qwen3RotaryEmbedding(Qwen3Config(head_dim=128, partial_rotary_factor=0.1)) yields inv_freq of length 64 — identical to the default — i.e. full 128-dim RoPE. The kwarg is merely absorbed into config.rope_parameters and ignored. Installed transformers 5.8.0.
```


### 8.20 Which files are symlinks (§C13 control reuse) rather than real checkpoints?

**Value**

```
12 symlinks, zero extra disk. 3 in 2026-06-21_arch-subdrill-p2 (checkpoint_baseline_seed{0,1,2}.pt), 3 in 2026-06-24_data-dclm-vs-fineweb (checkpoint_control_seed{0,1,2}.pt), 6 in 2026-06-26_data-mix-composition (checkpoint_dclm_seed{0,1,2}.pt, checkpoint_fineweb_seed{0,1,2}.pt). All point at either 2026-06-18_imu1-deconfound-p1/checkpoint_baseline_seed*.pt or 2026-06-24.../checkpoint_treatment_seed*.pt.
```

**Evidence** — `Qwen3-0.6B/experiments/2026-06-26_qwen3-0.6b_data-mix-composition/checkpoint_fineweb_seed0.pt (symlink)`

**Source quote**

```
lrwxrwxrwx 1 yashb98 yashb98 133 Jun 26 02:15 checkpoint_fineweb_seed0.pt -> /home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1/checkpoint_baseline_seed0.pt
```

**Confidence** — measured from code

**Caveat** — The symlinks use ABSOLUTE paths under /home/yashb98/Downloads/BuildFromScratch — they break on any copy/move/archive. Do not tar these into a release bundle without dereferencing.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
`find -type l` over *.pt/*.pkl returned exactly 12, with readlink targets matching the claimed mapping precisely: arch-subdrill-p2/checkpoint_baseline_seed{0,1,2}.pt and data-dclm-vs-fineweb/checkpoint_control_seed{0,1,2}.pt and data-mix-composition/checkpoint_fineweb_seed{0,1,2}.pt all -> imu1-deconfound-p1/checkpoint_baseline_seed{0,1,2}.pt; data-mix-composition/checkpoint_dclm_seed{0,1,2}.pt -> data-dclm-vs-fineweb/checkpoint_treatment_seed{0,1,2}.pt. The quoted ls line is exact: 'lrwxrwxrwx 1 yashb98 yashb98 133 Jun 26 02:15 checkpoint_fineweb_seed0.pt -> /home/yashb98/Downloads/BuildFromScratch/Qwen3-0.6B/experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1/checkpoint_baseline_seed0.pt'. All targets are absolute — the tar/archive warning is correct.
```


### 8.21 What is the only HybridSSM checkpoint that has actually been scored by the §C10 eval harness?

**Value**

```
checkpoint_ssm_base_s0.pkl. HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/eval/suite_results.json (dated 2026-07-21T03:21:22Z, suite_version text-lm-v2) names target_ckpt='checkpoint_ssm_base_s0.pkl', baseline_ckpt=null, self_floor=true; wikitext2_val PPL 133.4628, code_py PPL 5142.6426, both on 204,600 tokens.
```

**Evidence** — `HybridSSM-0.2B/experiments/2026-07-19_hybrid-ssm-0.2b_build/eval/suite_results.json`

**Source quote**

```
"suite_version": "text-lm-v2",
 "target_ckpt": "checkpoint_ssm_base_s0.pkl",
 "baseline_ckpt": null,
 "self_floor": true,
 ... "wikitext2_val": {"target": 133.4628, ...}, "code_py": {"target": 5142.6426, ...}
```

**Confidence** — results JSON

**Caveat** — That file is step=21156 (the 170M-token build run), NOT any of the 42M/85M ladder cells. The ladder cells were scored by a different script (arch_ladder_scores.json) which records NO checkpoint filenames at all — I searched its JSON for /checkpoint_.*\.pkl/ and got an empty list, so ladder scores cannot be traced to specific files from that artifact alone.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
suite_results.json read directly: suite_version 'text-lm-v2', date '2026-07-21T03:21:22Z', target_ckpt 'checkpoint_ssm_base_s0.pkl', baseline_ckpt null, self_floor true; ppl.wikitext2_val.target 133.4628 with n_tokens 204600, ppl.code_py.target 5142.6426 with n_tokens 204600 — ALL EXACT. Dataset-config check passes the -raw- trap: corpus_id is 'Salesforce/wikitext:wikitext-2-raw-v1:validation@b08601e04326c79dfdd32d625aee71d232d685c3' and code is 'codeparrot/codeparrot-clean-valid:train@4db92d2ec0c1b4c41eeb439cfae16854511d9dcd (streaming first-N >500k chars)' — pinned revisions, no baseline. I unpickled checkpoint_ssm_base_s0.pkl: step=21156, EXACT. grep -oE 'checkpoint_[A-Za-z0-9_]+\.pkl' over arch_ladder_scores.json returns count 0 — the traceability gap is real, and the only link is run_arch_ladder.sh:395 `--ckpt "checkpoint_${id}.pkl"` as the gap section says.
```


### 8.22 Which Qwen3 experiment directories contain NO checkpoints at all?

**Value**

```
2026-06-16_qwen3-0.6b_eval-faithful, eval-modernized, eval-prope10, eval-prope25, 2026-06-16_qwen3-faithful_eval-first, 2026-06-27_qwen3-0.6b_midtraining, 2026-07-01_qwen3-0.6b_rlvr-phase1-passk. Also HybridSSM-0.2B/experiments/2026-07-21_..._arch-ladder, 2026-07-28_..._arch-ladder-repair, 2026-07-29_..._throughput, and HybridSSM-0.2B/results/. There is no Qwen3-0.6B/experiments/2026-07-23* directory despite a ledger run of that id.
```

**Evidence** — `Qwen3-0.6B/experiments/ (directory listing) ; research/ledger/ledger.json (run 2026-07-23_qwen3-0.6b_normuon-at-scale)`

**Source quote**

```
"artifacts_location": "Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence (shared with the parent ladder; its ch...
```

**Confidence** — measured from code

**Caveat** — Run-id to directory is many-to-one and sometimes non-existent. Any 'weights for run X' claim must be resolved through the launch script, not the run id.

**Verdict — ⚠️ NEEDS QUALIFIER**

**Corrected value**

```
The Qwen3 list is incomplete — it must also include 2026-07-05_qwen3-0.6b_scaling-persistence. Correct count is 8 of 17 Qwen3 experiment dirs, not 7.
```

**Verifier note**

```
I enumerated all 17 Qwen3-0.6B/experiments/*/ dirs and tested each for *.pt/*.pkl (excluding tokcache). Zero-checkpoint dirs: the 7 named PLUS 2026-07-05_qwen3-0.6b_scaling-persistence. The omission is odd because the same fact set correctly states elsewhere (checkpoints-back-the-ladder fact) that the scaling-persistence dir holds no checkpoints — so the two facts contradict each other and a reader could conclude the ladder dir does hold weights. HybridSSM side confirmed: only 2026-07-19_..._build holds pickles; 2026-07-21_arch-ladder, 2026-07-28_arch-ladder-repair, 2026-07-29_throughput and HybridSSM-0.2B/results have none. `ls -d Qwen3-0.6B/experiments/2026-07-23*` -> 'No such file or directory', while the ledger carries run 2026-07-23_qwen3-0.6b_normuon-at-scale — confirmed. The quoted artifacts_location string is verbatim, but it lives at runs[24].metrics.artifacts_location, not at the run's top level (research/ledger/ledger.json:2077).
```


### 8.23 Is a GPU job currently running that would make loading checkpoints unsafe?

**Value**

```
No trainer live. pgrep -af train returned only this session's own bash wrapper (a known false-positive documented in MEMORY.md guard_pgrep_self_match). free -g showed 100 GB available of 119 GB at inspection time. The ledger does list one run status=running (2026-07-28_hybrid-ssm-0.2b_arch-ladder-repair, eta_hours 30.66) with no live process.
```

**Evidence** — `research/ledger/ledger.json (runs[].status)`

**Source quote**

```
runs (29): crashed=2, done=26, running=1
  in-flight: 2026-07-28_hybrid-ssm-0.2b_arch-ladder-repair  status=running  eta_hours=30.66
```

**Confidence** — measured from code

**Caveat** — The ledger 'running' entry is STALE relative to the process table — no matching trainer exists. I loaded at most one checkpoint at a time, CPU-only (map_location='cpu'), and never touched a file >8 GB, per the task's safety constraint.

**Verdict — ✅ CONFIRMED**

**Verifier note**

```
Independently re-verified before any checkpoint load: `python3 sentinel.py preflight` exited 0 with 'PREFLIGHT OK mem_available=83% disk_free=2705GB load1=0.78 cores=20 trainers=none' — sentinel's own trainer detector says none, which is stronger evidence than pgrep. `pgrep -af train` again matched only my own bash wrapper (the documented self-match). free -g showed 99 GB available of 119 at my check (vs the claimed 100 — time-varying, not a discrepancy). Ledger: 29 runs, Counter({'done': 26, 'crashed': 2, 'running': 1}), the single running entry is 2026-07-28_hybrid-ssm-0.2b_arch-ladder-repair with eta_hours 30.66 — EXACT. The stale-ledger observation is correct and worth surfacing. I loaded at most one checkpoint per process, map_location='cpu', under safe_cuda.guard(0.85), and used msgpack_restore on params only (never opt_state).
```


### 8.V Additional verifier findings (no 1:1 extracted fact)

**8.V1 — ❌ WRONG** · [GAPS SECTION] No SHA256 or any checksum exists for any checkpoint; every ledger run entry has lineage.artifact_sha256 = null

**Checked against**

```
Every ledger run entry I inspected has lineage.artifact_sha256 = null
```

**Corrected value**

```
27 of 29 runs have lineage.artifact_sha256 = null; 2 do not. 2026-07-29_hybrid-ssm-0.2b_fineweb-edu-carding carries a real 64-hex digest 'c83b7d608a0ca320ae7b7e41dbee05282f074a004a87a9a90f2f4fd0f5032491', and 2026-06-16_qwen3-faithful_eval-first carries the non-hash string 'checkpoint_qwen3_baseline2tpp.pt@step18150'.
```

**Verifier note**

```
Verified by walking every runs[] entry in research/ledger/ledger.json and counting lineage.artifact_sha256: Counter({None: 27, 'checkpoint_qwen3_baseline2tpp.pt@step18150': 1, 'c83b7d608a0ca320ae7b7e41dbee05282f074a004a87a9a90f2f4fd0f5032491': 1}). The broader conclusion — that no on-disk model checkpoint has a verifiable checksum — probably still stands (the hex digest belongs to a dataset-carding run and the other value is a filename, not a hash), but the absolute quantifier 'every entry' is false as written and must not be repeated. I did not verify what artifact the hex digest covers.
```


### 8.G Gaps — not determinable from disk

- Which HybridSSM ladder score row came from which .pkl file. arch_ladder_scores.json contains no checkpoint filenames (regex search for checkpoint_*.pkl returned an empty list), so ladder BPB/CE numbers cannot be traced to specific on-disk weights from that artifact alone. Only the cell id links them, via run_arch_ladder.sh:395 `--ckpt "checkpoint_${id}.pkl"`.
- Whether 'HybridSSM-0.2B' means 0.2B total or 0.2B non-embedding params. Measured totals are 267M-325M; non-embedding for ssm_base is 189,131,520. I found no file on disk that states the convention.
- License/provenance status for redistributing HybridSSM weights. The model is novel but uses the Qwen3-0.6B-Base tokenizer (vocab 151,936) and was trained on FineWeb-Edu; no LICENSE file or data-license record was found alongside the checkpoints.
- Byte-level reproducibility of the two SmolLM2 checkpoints. Both were written 2026-05-13/14 but train.py and train_tinystories.py were modified 2026-05-19; checkpoint.pt lacks the 'training_recipe' key the current train.py:182 writes, so the exact script that produced them is not the version at HEAD.
- Whether the 8 pre-PRNG-fix HybridSSM checkpoints that were resumed mid-run carry the same RNG confound that got the 85M one quarantined. Only checkpoint_swa128_nope_85M_s0.pkl was quarantined; I found no document assessing the others, and 'has rng key' is my own measurement from file tails, not a repo-stated classification.
- No SHA256 or any checksum exists for any checkpoint. Every ledger run entry I inspected has lineage.artifact_sha256 = null, so on-disk file integrity cannot be verified against any record.

---
