"""CPU unit tests for research/eval_math_acc.py — the math-acc-v1 decision-metric scorer.
Covers the pinned extractor, the SymPy-normalized verifier, the Chen-2021 pass@k estimator,
the pass@1 Wilson aggregation, and the verifier-honesty (permissive-vs-strict) differential
fuzz. No model, no network, no CUDA."""
import math

import pytest

from research.eval_math_acc import (
    EXTRACTOR_VERSION, extract_answer, is_equiv, pass_at_k,
    verifier_false_positive_rate, run_math_acc,
)

# --------------------------------------------------------------- extractor

@pytest.mark.parametrize("text,expected", [
    (r"the reasoning ... so the answer is \boxed{72}.", "72"),
    (r"\boxed{-3/4}", "-3/4"),
    (r"first \boxed{1} then \boxed{2}", "2"),                 # LAST boxed wins
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),                 # balanced braces kept
    ("Chain of thought...\n#### 18", "18"),                   # GSM8K gold convention
    ("The final answer is 42", "42"),
    ("blah blah 5 then 6 then 7", "7"),                       # last-number fallback
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected

def test_extract_answer_none():
    assert extract_answer("") is None
    assert extract_answer("no digits or box here") is None

# --------------------------------------------------------------- verifier (is_equiv)

@pytest.mark.parametrize("pred,gold", [
    ("72", "72"),
    (r"\boxed{72}", "72"),                    # extraction not needed; is_equiv normalizes box-free
    ("1/2", "0.5"),
    (r"\frac{1}{2}", "0.5"),
    ("1,234", "1234"),                        # thousands separator
    ("2^3", "8"),                             # sympy symbolic
    ("-3/4", "-0.75"),
    ("  6 ", "6"),
    (r"\boxed{50\%}", "50"),                  # unit strip
])
def test_is_equiv_true(pred, gold):
    assert is_equiv(pred, gold) is True

@pytest.mark.parametrize("pred,gold", [
    ("72", "73"),
    ("1/2", "0.51"),
    ("8", "9"),
    (None, "5"),
    ("5", None),
    ("", "5"),
])
def test_is_equiv_false(pred, gold):
    assert is_equiv(pred, gold) is False

def test_is_equiv_never_crashes_on_garbage():
    # a verifier must never throw during training — pathological latex ⇒ False, not an exception
    for junk in [r"\frac{{{", "((((", r"\boxed{" * 50, "x" * 500, "1/0"]:
        assert is_equiv(junk, "5") is False

# --------------------------------------------------------------- pass@k (Chen 2021)

def test_pass_at_k_edges():
    assert pass_at_k(4, 0, 2) == 0.0        # no correct samples
    assert pass_at_k(4, 4, 2) == 1.0        # all correct
    assert pass_at_k(4, 4, 1) == 1.0
    assert math.isclose(pass_at_k(4, 1, 1), 0.25)   # k=1 ⇒ c/n
    assert math.isclose(pass_at_k(10, 3, 1), 0.3)

def test_pass_at_k_matches_closed_form():
    # 1 - C(n-c,k)/C(n,k)
    for n, c, k in [(10, 2, 5), (16, 3, 8), (8, 1, 4), (20, 7, 16)]:
        ref = 1.0 - math.comb(n - c, k) / math.comb(n, k)
        assert math.isclose(pass_at_k(n, c, k), ref, rel_tol=1e-12)

def test_pass_at_k_monotonic_in_k():
    vals = [pass_at_k(16, 3, k) for k in (1, 2, 4, 8, 16)]
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))   # pass@k non-decreasing in k

def test_pass_at_k_bad_args():
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 0)
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 5)      # k > n

# --------------------------------------------------------------- verifier honesty (differential)

def test_verifier_false_positive_rate_strict_is_zero():
    # KNOWN-WRONG pairs: the strict verifier must accept NONE of them
    wrong = [("72", "73"), ("1/2", "1/3"), ("10", "100"), ("5", "-5"), ("2^3", "9")]
    assert verifier_false_positive_rate(wrong) == 0.0

def test_permissive_vs_strict_differential_fuzz():
    # The pinned strict verifier vs a naive PERMISSIVE one (substring / last-number match)
    # on adversarial wrong pairs — the permissive verifier reward-hacks, the strict must not.
    def permissive(pred, gold):                       # the tempting-but-wrong extractor
        return gold in pred or extract_answer(pred) == extract_answer(gold)
    adversarial = [
        ("the answer is 3 (not 30)", "30"),           # substring 30⊄ but 3 present → naive trips on '3'
        ("1234", "34"),                               # gold '34' is a substring of pred
        ("100", "10"),                                # gold '10' substring of '100'
        ("x = 5 or 15", "15"),
    ]
    strict_fp = sum(1 for p, g in adversarial if is_equiv(p, g))
    perm_fp = sum(1 for p, g in adversarial if permissive(p, g))
    assert strict_fp == 0                              # pinned verifier: no false positives
    assert perm_fp > strict_fp                         # the permissive one DOES get gamed → why we pin strict

# --------------------------------------------------------------- end-to-end harness

def _stub(correct_frac):
    """A deterministic stub policy: emits the gold answer for the first round(correct_frac*n)
    samples and a wrong one for the rest. No model."""
    def gen(prompt, n):
        gold = "4" if "2+2" in prompt else "6"
        n_ok = round(correct_frac * n)
        return [rf"\boxed{{{gold}}}"] * n_ok + [r"\boxed{999}"] * (n - n_ok)
    return gen

def test_run_math_acc_all_correct():
    items = [{"prompt": "2+2?", "gold": "4"}, {"prompt": "3+3?", "gold": "6"}]
    out = run_math_acc(_stub(1.0), items, n_samples=8, k_list=(1, 8))
    assert out["pass1_wilson_ci"]["acc"] == 1.0
    assert out["passk_chen2021"][8] == 1.0
    assert out["solved_items"] == 2
    assert out["extractor_version"] == EXTRACTOR_VERSION      # extractor_pinned stamped

def test_run_math_acc_all_wrong_is_clean_zero():
    # the honest-null case the plan predicts for our 0.6B base: pass@k == 0 is a REAL result
    items = [{"prompt": "2+2?", "gold": "4"}]
    out = run_math_acc(_stub(0.0), items, n_samples=16, k_list=(1, 8, 16))
    assert out["pass1_wilson_ci"]["acc"] == 0.0
    assert out["passk_chen2021"][16] == 0.0
    assert out["solved_items"] == 0

def test_run_math_acc_partial_passk_gt_pass1():
    # 2/8 correct per item ⇒ pass@1 = 0.25, pass@8 = 1.0 (Chen): pass@k must exceed pass@1
    items = [{"prompt": "2+2?", "gold": "4"}]
    out = run_math_acc(_stub(0.25), items, n_samples=8, k_list=(1, 8))
    assert math.isclose(out["pass1_wilson_ci"]["acc"], 0.25)
    assert out["passk_chen2021"][8] > out["passk_chen2021"][1]

def test_run_math_acc_empty_raises():
    with pytest.raises(ValueError):
        run_math_acc(_stub(1.0), [], n_samples=4)
