# BuildFromScratch

From-scratch language-model reproductions and the tooling around them. Three
subprojects live here:

| Path | What it is |
|---|---|
| [`SmolLM2-134(base)/`](SmolLM2-134(base)/) | Single-file PyTorch reproduction of [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), verified **bit-exact** against the official HuggingFace weights (`max |Δlogits| = 0.0`). Includes from-scratch training, continued pretraining on TinyStories, multi-axis parity diagnostics, in-domain vs OOD eval, and an `lm-evaluation-harness` wrapper. |
| [`.claude/skills/`](.claude/skills/) | Claude Code skills that drive this project: `from-scratch-build` (research → implement → verify → train → evaluate loop for new architectures) and `finance-research-loop` (continued-pretraining research/tune/eval loop on finance data). |
| [`skills_showcase/`](skills_showcase/) | A FastAPI mini-agent harness that exposes the skills above as an HTTP-driven Claude agent loop with tool-use approval, prompt caching, and a static showcase page. |

## Quickstart — the SmolLM2 reproduction

```bash
cd "SmolLM2-134(base)"

# Install pinned dependencies that produced the 0.0 logit-diff result.
pip install -e .                # or: pip install -r requirements.txt

# Architecture parity gate (the non-negotiable test before training).
pytest tests/ -v
# or, the script form:
python verify.py

# Sample from the official weights via our class.
python generate.py "Once upon a time"

# A toy training run (random init, wikitext-103 demo) — proves the loop works.
python train.py --steps 100

# Continued pretraining on TinyStories from official weights.
python train_tinystories.py --token_budget 10_000_000     # ~10M tokens for a quick run

# Resume a run that died:
python train_tinystories.py --resume checkpoint_tinystories.pt --token_budget 100_000_000

# Standardized benchmarks (lm-evaluation-harness wrapper):
pip install lm-eval
bash scripts/run_lm_eval.sh                          # base only
bash scripts/run_lm_eval.sh checkpoint_tinystories.pt # base + trained
```

See [`SmolLM2-134(base)/README.md`](SmolLM2-134(base)/README.md) for the full
architectural walkthrough, design-decision narrative, and reproduction recipe.

## Repository layout

```
BuildFromScratch/
├── README.md                      # this file
├── .gitignore
├── SmolLM2-134(base)/             # the SmolLM2-135M reproduction (see its README)
│   ├── model_full.py              # the architecture, one file
│   ├── verify.py                  # parity gate vs HF reference
│   ├── compare_with_hf.py         # 6-axis diagnostic suite
│   ├── train.py                   # from-scratch training (random init)
│   ├── train_tinystories.py       # continued pretraining on TinyStories
│   ├── eval_after_vs_base.py      # in-domain + OOD ppl comparison
│   ├── benchmark_training.py      # tok/s + peak memory measurement
│   ├── generate.py                # CLI sampler
│   ├── tests/test_parity.py       # pytest gate
│   ├── scripts/                   # lm-eval wrapper + checkpoint export
│   ├── results/                   # generated artifacts (JSON/CSV/PNG/MD)
│   ├── proof.ipynb                # pedagogical notebook
│   ├── results.ipynb              # end-to-end results notebook
│   ├── pyproject.toml             # pinned deps
│   └── README.md                  # the long-form architecture/recipe doc
├── .claude/skills/                # Claude Code skill definitions
│   ├── from-scratch-build/SKILL.md
│   └── finance-research-loop/SKILL.md
└── skills_showcase/               # FastAPI agent-harness showcase
    └── server/                    # /api/tools/* direct invocation + /api/agent/* loop
```
