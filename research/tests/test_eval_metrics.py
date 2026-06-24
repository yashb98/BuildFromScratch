"""Tests for research/eval_metrics.py — the within-run eval quality battery
(capability accuracy + CI, calibration, contamination, robustness, conformal).
Pure math, no model. Where the relevant library is installed, the test also
cross-validates against it (statsmodels Wilson / sklearn Brier) so the numbers
are externally correct, not just self-consistent.
"""
import math

import pytest

import eval_metrics as em


# ----------------------------------------------------------- accuracy + Wilson CI

def test_accuracy_point():
    acc, lo, hi = em.accuracy_wilson_ci([1, 1, 1, 0, 0])
    assert acc == 0.6 and lo < 0.6 < hi


def test_wilson_known_value():
    # 8/10 successes, 95% Wilson interval ~= [0.490, 0.943] (textbook).
    acc, lo, hi = em.accuracy_wilson_ci([1] * 8 + [0] * 2)
    assert acc == 0.8
    assert math.isclose(lo, 0.4901, abs_tol=1e-3)
    assert math.isclose(hi, 0.9433, abs_tol=1e-3)


def test_wilson_extremes_stay_in_unit_interval():
    # all-correct: upper bound must not exceed 1, lower bound stays < 1.
    acc, lo, hi = em.accuracy_wilson_ci([1] * 20)
    assert acc == 1.0 and 0.0 <= lo < 1.0 and hi == 1.0
    acc, lo, hi = em.accuracy_wilson_ci([0] * 20)
    assert acc == 0.0 and lo == 0.0 and 0.0 < hi <= 1.0


def test_wilson_cross_validate_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    _, lo, hi = em.accuracy_wilson_ci([1] * 8 + [0] * 2)
    smlo, smhi = sm.proportion_confint(8, 10, alpha=0.05, method="wilson")
    assert math.isclose(lo, smlo, abs_tol=1e-6) and math.isclose(hi, smhi, abs_tol=1e-6)


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        em.accuracy_wilson_ci([])


# ----------------------------------------------------------------- calibration

def test_ece_perfectly_calibrated_is_zero():
    # confidence == empirical accuracy within each bin -> ECE 0.
    conf = [0.0, 0.0, 1.0, 1.0]   # bin0: 2 wrong (acc 0, conf 0); top bin: 2 right (acc 1, conf 1)
    corr = [0, 0, 1, 1]
    assert em.ece(conf, corr, n_bins=10) == 0.0


def test_ece_overconfident_positive():
    # all predicted 0.9 confidence but only half correct -> ECE ~ 0.4.
    conf = [0.9, 0.9, 0.9, 0.9]
    corr = [1, 0, 1, 0]
    assert math.isclose(em.ece(conf, corr, n_bins=10), 0.4, abs_tol=1e-9)


def test_ece_weights_by_bin_population():
    # Two bins, UNEQUAL populations, different gaps: bin0 (1 item, gap 1.0) and
    # bin5 (3 items, gap 0.5). Population-weighted ECE = 1/4*1.0 + 3/4*0.5 = 0.625;
    # an UNWEIGHTED average would give (1.0+0.5)/2 = 0.75 -> this pins the weight.
    conf = [0.0, 0.5, 0.5, 0.5]
    corr = [1, 1, 1, 1]
    assert math.isclose(em.ece(conf, corr, n_bins=10), 0.625, abs_tol=1e-9)


def test_ece_validates_inputs():
    with pytest.raises(ValueError):
        em.ece([0.5], [1, 0])              # length mismatch
    with pytest.raises(ValueError):
        em.ece([1.5], [1])                 # out of [0,1]
    with pytest.raises(ValueError):
        em.ece([], [])


def test_brier_basics_and_sklearn():
    assert em.brier_binary([1.0, 0.0], [1, 0]) == 0.0           # perfect
    assert em.brier_binary([0.5, 0.5], [1, 0]) == 0.25          # max-uncertain
    assert em.brier_binary([0.0, 1.0], [1, 0]) == 1.0           # confidently wrong
    skm = pytest.importorskip("sklearn.metrics")
    p, y = [0.9, 0.2, 0.7, 0.4], [1, 0, 1, 0]
    assert math.isclose(em.brier_binary(p, y), skm.brier_score_loss(y, p), abs_tol=1e-9)


# --------------------------------------------------------------- contamination

def test_ngram_contamination_full_and_none():
    train = ["the quick brown fox jumps over"]
    idx = em.build_ngram_index(train, n=3)
    # identical eval text -> every 3-gram overlaps -> flagged
    rate, per = em.ngram_contamination(["the quick brown fox jumps over"], idx, n=3, threshold=0.5)
    assert rate == 1.0 and per[0] == 1.0
    # disjoint eval text -> no overlap
    rate, per = em.ngram_contamination(["completely unrelated novel sentence here now"], idx, n=3)
    assert rate == 0.0 and per[0] == 0.0


def test_ngram_short_text_is_empty():
    idx = em.build_ngram_index(["a b c d"], n=3)
    rate, per = em.ngram_contamination(["only two"], idx, n=3)   # < n tokens
    assert per[0] == 0.0 and rate == 0.0


def test_ngram_partial_overlap_threshold():
    idx = em.build_ngram_index(["alpha beta gamma delta"], n=2)  # {alpha beta, beta gamma, gamma delta}
    # eval has 3 bigrams, 1 overlaps ("gamma delta") -> overlap 1/3, below 0.5 -> not flagged
    rate, per = em.ngram_contamination(["gamma delta epsilon zeta"], idx, n=2, threshold=0.5)
    assert math.isclose(per[0], 1 / 3) and rate == 0.0


# --------------------------------------------------------------- robustness

def test_answer_flip_rate():
    # base-correct at idx 0,1,2; perturbed wrong at idx 1 -> flip 1/3.
    assert math.isclose(em.answer_flip_rate([1, 1, 1, 0], [1, 0, 1, 1]), 1 / 3)


def test_answer_flip_none_correct():
    assert em.answer_flip_rate([0, 0], [1, 1]) == 0.0   # nothing could flip


def test_answer_flip_length_mismatch():
    with pytest.raises(ValueError):
        em.answer_flip_rate([1, 1], [1])


# --------------------------------------------------------------- conformal

def test_conformal_halfwidth_known_ranks():
    res = list(range(1, 11))            # 1..10
    # alpha=0.1: rank=ceil(11*0.9)=10 -> 10th smallest = 10
    assert em.conformal_halfwidth(res, alpha=0.1) == 10
    # alpha=0.2: rank=ceil(11*0.8)=9 -> 9th smallest = 9
    assert em.conformal_halfwidth(res, alpha=0.2) == 9


def test_conformal_too_few_points_returns_inf():
    # n=5, alpha=0.01: rank=ceil(6*0.99)=6 > 5 -> coverage impossible -> inf
    assert em.conformal_halfwidth([1, 2, 3, 4, 5], alpha=0.01) == float("inf")


def test_conformal_uses_absolute_residuals():
    # 10 negatives; |.| sorted = 1..10; alpha=0.1 -> rank 10 -> 10.
    assert em.conformal_halfwidth([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], alpha=0.1) == 10


def test_conformal_validates():
    with pytest.raises(ValueError):
        em.conformal_halfwidth([], alpha=0.1)
    with pytest.raises(ValueError):
        em.conformal_halfwidth([1, 2], alpha=1.5)


# --- bits-per-byte + perplexity (the cross-tokenizer comparability fix) --------

def test_bits_per_byte_basic():
    # 100 nats over 200 bytes => (100/ln2)/200 bpb
    import math as _m
    assert em.bits_per_byte(100.0, 200) == pytest.approx((100.0 / _m.log(2)) / 200)
    # fewer bytes for the same loss => higher bpb (harder per byte)
    assert em.bits_per_byte(100.0, 100) > em.bits_per_byte(100.0, 200)
    with pytest.raises(ValueError):
        em.bits_per_byte(1.0, 0)


def test_perplexity_basic():
    assert em.perplexity(0.0, 10) == pytest.approx(1.0)       # perfect model
    assert em.perplexity(10.0, 10) == pytest.approx(math.e)   # mean nll = 1
    with pytest.raises(ValueError):
        em.perplexity(1.0, 0)


# ── interp prerequisites: roc_auc / bootstrap_ci / mcnemar (build_spec §3) ──
import random as _rnd
from eval_metrics import roc_auc, bootstrap_ci, mcnemar

def test_roc_auc_vs_sklearn():
    from sklearn.metrics import roc_auc_score
    r=_rnd.Random(7); s=[r.random() for _ in range(200)]; y=[r.randint(0,1) for _ in range(200)]
    assert abs(roc_auc(s,y)-roc_auc_score(y,s))<1e-9
    assert roc_auc([1,2,3,4],[0,0,1,1])==1.0 and roc_auc([4,3,2,1],[0,0,1,1])==0.0
    assert roc_auc([1,1,0,0],[1,0,1,0])==0.5

def test_bootstrap_ci_reproducible_and_brackets():
    m=lambda a: sum(a)/len(a)
    assert bootstrap_ci(m,[5.0]*30,n=300,seed=1)[1:]==(5.0,5.0)
    a=bootstrap_ci(m,list(range(100)),n=400,seed=1); assert a==bootstrap_ci(m,list(range(100)),n=400,seed=1)
    assert a[1]<49.5<a[2]
    A=[float(i) for i in range(40)]; B=[x+2 for x in A]
    pd,lo,hi=bootstrap_ci(lambda x,y:m(x)-m(y),(A,B),n=300,seed=3,paired=True); assert lo<=-2<=hi

def test_mcnemar_vs_scipy_exact_and_chi2():
    from scipy.stats import binomtest, chi2
    for b,c in [(3,3),(10,2),(40,20),(13,1)]:
        _,p,_=mcnemar(b,c)
        o=binomtest(min(b,c),b+c,0.5,alternative='two-sided').pvalue if b+c<25 else chi2.sf((abs(b-c)-1)**2/(b+c),1)
        assert abs(p-o)<1e-6
    assert mcnemar(5,5)[1]==1.0 and mcnemar(7,7,)[2]=="exact-binomial"
