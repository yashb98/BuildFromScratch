"""Math-accuracy scorer — the `math-acc-v1` capability axis (the decision metric the
rlvr stage needs, per research/rlvr/plan.md §C27). The MISSING piece: eval-harness has
PPL + multiple-choice but NO exact-match / pass@k, and the SFT cohort proved PPL cannot
separate reasoning arms — so held-out math EXACT-MATCH pass@1 + pass@k is the only valid
win metric for SFT/distillation/GRPO reasoning runs.

DESIGN — model-agnostic & CPU-testable, mirroring research/eval_downstream.py. The scorer
takes a `generate_fn` callable (`(prompt:str, n:int) -> list[str]` of n sampled
completions), NOT a model, so all of extraction / equivalence / pass@k is unit-tested here
with a deterministic stub and the SAME code runs on the real checkpoint under eval-harness.
This module NEVER trains, NEVER loads a model, NEVER allocates CUDA — the CALLER
(eval-harness template) loads the checkpoint behind `safe_cuda.guard(...)` and passes
`generate_fn`.

The three §C25 `rlvr` battery items this module supplies:
  - `pass1_wilson_ci`   — accuracy_wilson_ci over all (item×sample) trials.
  - `passk_chen2021`    — the unbiased HumanEval/Chen-2021 pass@k estimator, mean over items.
  - `extractor_pinned`  — EXTRACTOR_VERSION is stamped into every result; the extractor +
                          verifier are frozen here and version-bumped on any change, so a
                          number is comparable only within one extractor version.
Plus `verifier_honesty_ipt` support: `verifier_false_positive_rate()` runs the verifier on
KNOWN-wrong (question, answer) pairs — it MUST stay near 0, else the reward is gameable.

HONEST SCALE CAVEAT (carried into every result): at 596M params / ~1.19B training tokens
the SFT'd model very likely sits near pass@k≈0 on any non-trivial math band. A pass@k of 0
is a REAL, decisive result (RL is null-by-construction — nothing to sharpen), not a bug.
Report the number; do not over-read a near-zero accuracy as a failure of the harness.
"""
from __future__ import annotations
import re
from typing import Callable, Sequence

from research.eval_metrics import accuracy_wilson_ci

# Pinned extractor+verifier identity. BUMP on ANY change to _extract/_normalize/is_equiv —
# numbers are only comparable within one version (the `extractor_pinned` §C25 item).
EXTRACTOR_VERSION = "math-acc-v1"

GenerateFn = Callable[[str, int], Sequence[str]]

# --------------------------------------------------------------- pass@k (Chen 2021)

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, the HumanEval formula):
    the probability that at least one of k samples drawn WITHOUT replacement from n
    total samples (of which c are correct) is correct = 1 - C(n-c,k)/C(n,k), computed
    in the numerically-stable product form. k=1 ⇒ c/n (plain pass@1)."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    if n <= 0 or k > n:
        raise ValueError(f"need 1 <= k <= n; got n={n} k={k}")
    if c <= 0:
        return 0.0
    if n - c < k:          # not enough wrong samples to fill k slots ⇒ guaranteed a hit
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod

# --------------------------------------------------------------- answer extraction

def _last_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...} (balanced braces), or None."""
    idx = text.rfind(r"\boxed")
    if idx < 0:
        return None
    i = idx + len(r"\boxed")
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth, start = 0, i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None

_ANS_IS = re.compile(r"(?:final answer|answer)\s*(?:is|:|=)\s*\$?([^\n.$]+)", re.IGNORECASE)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def extract_answer(text: str) -> str | None:
    """Pinned+versioned answer extractor. Priority: \\boxed{} → GSM8K '#### X' →
    'the answer is X' → last number in the text. Returns the raw span (normalized at
    compare time), or None if nothing parseable. Deterministic, no model, no network."""
    if not text:
        return None
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    if "####" in text:                          # GSM8K gold/solution convention
        tail = text.rsplit("####", 1)[1]
        m = _NUM.search(tail)
        if m:
            return m.group(0).strip()
        return tail.strip().splitlines()[0].strip() if tail.strip() else None
    m = _ANS_IS.search(text)
    if m:
        return m.group(1).strip()
    nums = _NUM.findall(text)                    # last-number fallback
    return nums[-1].strip() if nums else None

# --------------------------------------------------------------- normalization + verifier

_TEXT_CMD = re.compile(r"\\(?:text|mathrm|mbox|mathbf|mathit)\s*\{([^}]*)\}")
_FRAC = re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")

def _normalize(s: str) -> str:
    """Canonicalize a math answer string for a fast string-equality path (the classic
    MATH/Minerva cleanup): strip $, \\left/\\right, \\text{}, units, spaces, thousands
    commas, trailing punctuation; \\frac{a}{b} → (a)/(b); ^ stays for the sympy path."""
    if s is None:
        return ""
    s = s.strip()
    s = _TEXT_CMD.sub(r"\1", s)
    s = s.replace(r"\boxed", "").replace(r"\left", "").replace(r"\right", "")
    s = _FRAC.sub(r"(\1)/(\2)", s)          # \frac{a}{b} → (a)/(b) BEFORE braces are stripped
    for a, b in ((r"\!", ""), (r"\,", ""), (r"\ ", " "), ("$", ""), (r"\%", ""), ("%", ""),
                 (r"^\circ", ""), (r"\circ", ""), (r"\cdot", "*"), (r"\times", "*"),
                 ("{", ""), ("}", "")):
        s = s.replace(a, b)
    s = s.replace("\\", "").replace(" ", "")
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):   # 1,234 → 1234 (only true separators)
        s = s.replace(",", "")
    return s.lower()

def _as_float(s: str):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def _sympy_equal(a: str, b: str) -> bool:
    """Symbolic equivalence via sympy (1/2 == 0.5, (1)/(2) == 0.5, 2^3 == 8). Fully
    guarded: any parse/timeout/exception → False (a verifier must never crash training).
    Length-capped to avoid pathological sympify blow-ups."""
    if len(a) > 100 or len(b) > 100:
        return False
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr
        expr = f"({a.replace('^', '**')})-({b.replace('^', '**')})"
        return simplify(parse_expr(expr, evaluate=True)) == 0
    except Exception:
        return False

def is_equiv(pred: str | None, gold: str | None) -> bool:
    """The VERIFIER (verifiable-reward, no neural RM). True iff the extracted prediction
    is mathematically equivalent to the gold answer. Order: normalized string equality →
    numeric equality (tol 1e-6) → sympy symbolic equality. Conservative: unparseable ⇒ False."""
    if pred is None or gold is None:
        return False
    np_, ng = _normalize(pred), _normalize(gold)
    if np_ == "" or ng == "":
        return False
    if np_ == ng:
        return True
    fp, fg = _as_float(np_), _as_float(ng)
    if fp is not None and fg is not None:
        return abs(fp - fg) <= 1e-6 * max(1.0, abs(fg))
    return _sympy_equal(np_, ng)

def verifier_false_positive_rate(wrong_pairs: Sequence[tuple[str, str]]) -> float:
    """`verifier_honesty_ipt` support: fraction of KNOWN-WRONG (pred, gold) pairs the
    verifier wrongly accepts. MUST be ~0 — a nonzero rate means the reward is gameable."""
    if not wrong_pairs:
        return 0.0
    fp = sum(1 for pred, gold in wrong_pairs if is_equiv(pred, gold))
    return fp / len(wrong_pairs)

# --------------------------------------------------------------- scoring harness

def score_item(generate_fn: GenerateFn, prompt: str, gold: str, n_samples: int) -> dict:
    """Generate n_samples completions for one prompt, extract+verify each. Returns
    {n, c, correct: [0/1...]} — c = number of verifier-correct samples."""
    comps = list(generate_fn(prompt, n_samples))
    correct = [1 if is_equiv(extract_answer(c), gold) else 0 for c in comps]
    return {"n": len(correct), "c": sum(correct), "correct": correct}

def run_math_acc(generate_fn: GenerateFn, items: Sequence[dict],
                 n_samples: int = 16, k_list: Sequence[int] = (1, 8, 16)) -> dict:
    """Score a held-out math set. `items` = [{"prompt", "gold"}...]. For each item sample
    n_samples completions; aggregate pass@1 (Wilson CI over all item×sample trials) and
    pass@k (Chen-2021, mean over items). Returns a flat metrics dict stamped with
    EXTRACTOR_VERSION — the §C25 rlvr items pass1_wilson_ci / passk_chen2021 / extractor_pinned."""
    if not items:
        raise ValueError("no items to score")
    k_list = [k for k in k_list if k <= n_samples]
    per_item, flat = [], []
    for it in items:
        r = score_item(generate_fn, it["prompt"], it["gold"], n_samples)
        per_item.append(r)
        flat.extend(r["correct"])
    acc, lo, hi = accuracy_wilson_ci(flat)
    passk = {k: sum(pass_at_k(r["n"], r["c"], k) for r in per_item) / len(per_item)
             for k in k_list}
    return {
        "extractor_version": EXTRACTOR_VERSION,     # → extractor_pinned
        "n_items": len(items),
        "n_samples": n_samples,
        "pass1_wilson_ci": {"acc": acc, "ci_low": lo, "ci_high": hi},
        "passk_chen2021": passk,
        "solved_items": sum(1 for r in per_item if r["c"] > 0),
    }

def _self_test() -> None:
    """Deterministic CPU smoke — no model, no network. Run: python3 -m research.eval_math_acc"""
    assert extract_answer(r"so the answer is \boxed{72}.") == "72"
    assert extract_answer("...\n#### 18") == "18"
    assert is_equiv(r"\boxed{1/2}", "0.5") and is_equiv("72", "72") and not is_equiv("72", "73")
    assert abs(pass_at_k(4, 1, 1) - 0.25) < 1e-9 and pass_at_k(4, 0, 2) == 0.0 and pass_at_k(4, 4, 2) == 1.0
    # a stub policy that always emits the gold answer ⇒ pass@1 == 1
    items = [{"prompt": "2+2?", "gold": "4"}, {"prompt": "3+3?", "gold": "6"}]
    gen = lambda p, n: [f"the answer is \\boxed{{{ '4' if '2+2' in p else '6' }}}"] * n
    out = run_math_acc(gen, items, n_samples=4, k_list=(1, 4))
    assert out["pass1_wilson_ci"]["acc"] == 1.0 and out["passk_chen2021"][4] == 1.0
    print("SELF-TEST OK", out["extractor_version"], out["passk_chen2021"])

if __name__ == "__main__":
    _self_test()
