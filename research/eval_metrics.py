"""Pure evaluation metrics for the hardened eval suite (§C10) — stdlib-only, no
torch, no model, no network, no randomness (so results are deterministic and
unit-testable WITHOUT running a single eval). Companion to research/eval_stats.py
(which owns the cross-run seed-variance significance gate); this module owns the
WITHIN-run quality metrics the audit found missing entirely:

  * bits_per_byte        — the ONLY cross-tokenizer-comparable LM metric (§C10)
  * perplexity           — exp(mean NLL); per-token, NOT comparable across vocabs
  * accuracy_wilson_ci   — capability-benchmark accuracy with a proper interval
  * ece / brier_binary   — calibration (the suite had none)
  * ngram_contamination  — eval<->pretraining overlap (the suite punted this)
  * answer_flip_rate     — robustness under prompt/format perturbation (none before)
  * conformal_halfwidth  — distribution-free split-conformal prediction interval

Senior-RS framing (audit fix, 2026-06-16): a sub-1B suite whose ONLY cross-run
number was PPL cannot make a capability claim. These metrics turn "we measured
loss" into "we measured accuracy with error bars, on uncontaminated data, and
checked calibration + robustness". They are the standard battery (lm-eval-harness
era), deliberately NOT the 2026 arXiv frontier (which targets >>1B instruct
models). All deterministic; cross-validated against statsmodels/sklearn in the
tests where available.
"""
from __future__ import annotations

import math

Z95 = 1.959963984540054  # two-sided 95% normal quantile
_LN2 = math.log(2.0)


# ------------------------------------------------------- language-model quality

def bits_per_byte(total_nll_nats, total_bytes):
    """Bits-per-byte = (Σ NLL in nats / ln2) / (raw UTF-8 byte count).

    The ONLY language-model loss number comparable ACROSS tokenizers (§C10): a
    256k-vocab model packs more bytes per token and looks artificially better in
    token-perplexity, but BPB normalises by the underlying bytes, so a Qwen3 run
    and a SmolLM2 run land on one axis. `total_bytes` MUST count the bytes of the
    exact token span that produced `total_nll_nats` (under overlapping eval
    windows, the byte denominator must match the double-counted token span)."""
    total_bytes = float(total_bytes)
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    return (float(total_nll_nats) / _LN2) / total_bytes


def perplexity(total_nll_nats, total_tokens):
    """exp(mean per-token NLL). Per-TOKEN, so NOT comparable across tokenizers —
    report alongside bits_per_byte, never as the sole cross-run headline."""
    total_tokens = int(total_tokens)
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    return math.exp(float(total_nll_nats) / total_tokens)


# --------------------------------------------------------------- capability + CI

def accuracy_wilson_ci(correct, z=Z95):
    """Accuracy of a benchmark with a Wilson score interval (the right interval
    for a proportion — far better than normal-approx at small n / extreme p, and
    deterministic, unlike a bootstrap). `correct` is an iterable of 0/1 or bool.
    Returns (accuracy, ci_low, ci_high). Raises on empty input."""
    vals = [1 if bool(c) else 0 for c in correct]
    n = len(vals)
    if n == 0:
        raise ValueError("need at least one example")
    k = sum(vals)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - margin), min(1.0, center + margin)


# --------------------------------------------------------------- calibration

def ece(confidences, correct, n_bins=10):
    """Expected Calibration Error: bin predictions by their confidence (the
    model's probability for the answer it chose), then average |accuracy −
    mean-confidence| weighted by bin population. 0 == perfectly calibrated.
    `confidences` in [0,1], `correct` 0/1. Raises on empty / length mismatch."""
    conf = [float(c) for c in confidences]
    corr = [1 if bool(c) else 0 for c in correct]
    if not conf:
        raise ValueError("need at least one prediction")
    if len(conf) != len(corr):
        raise ValueError("confidences and correct must be the same length")
    if any(not (0.0 <= c <= 1.0) for c in conf):
        raise ValueError("confidences must be in [0,1]")
    n = len(conf)
    # assign each prediction to one of n_bins equal-width bins; c==1.0 -> last bin
    bins = [[] for _ in range(n_bins)]
    for i in range(n):
        b = min(int(conf[i] * n_bins), n_bins - 1)
        bins[b].append(i)
    total = 0.0
    for idx in bins:
        if not idx:
            continue
        acc = sum(corr[i] for i in idx) / len(idx)
        avg_conf = sum(conf[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - avg_conf)
    return total


def brier_binary(probs, labels):
    """Brier score for binary outcomes: mean((p − y)^2), p = predicted prob of
    the positive class, y in {0,1}. 0 == perfect, 0.25 == always-0.5, 1 == confidently
    wrong. Lower is better."""
    p = [float(x) for x in probs]
    y = [1 if bool(x) else 0 for x in labels]
    if not p:
        raise ValueError("need at least one prediction")
    if len(p) != len(y):
        raise ValueError("probs and labels must be the same length")
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)


# --------------------------------------------------------------- contamination

def word_ngrams(text, n):
    """Set of word n-grams (lowercased, whitespace-tokenized) in `text`."""
    toks = text.lower().split()
    if len(toks) < n:
        return set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def build_ngram_index(train_texts, n):
    """Union of word n-grams across a (sampled) training corpus — the reference
    set an eval item is checked against for contamination."""
    idx = set()
    for t in train_texts:
        idx |= word_ngrams(t, n)
    return idx


def ngram_contamination(eval_texts, train_ngram_index, n, threshold=0.5):
    """For each eval text, the fraction of its n-grams that also appear in the
    training index; an item is FLAGGED contaminated when that fraction >
    threshold. Returns (flagged_rate, per_item_overlap). The standard n-gram
    train<->test leakage check (n≈8–13 in practice) the suite previously punted."""
    per_item = []
    flagged = 0
    for t in eval_texts:
        grams = word_ngrams(t, n)
        if not grams:
            per_item.append(0.0)
            continue
        overlap = sum(1 for g in grams if g in train_ngram_index) / len(grams)
        per_item.append(overlap)
        if overlap > threshold:
            flagged += 1
    rate = flagged / len(eval_texts) if eval_texts else 0.0
    return rate, per_item


# --------------------------------------------------------------- robustness

def answer_flip_rate(base_correct, perturbed_correct):
    """Robustness probe: of the items the model answered CORRECTLY under the base
    prompt, the fraction that flip to WRONG under a perturbation (reformatted
    prompt / shuffled MCQ options / paraphrase). 0 == fully robust, higher ==
    more brittle. Inputs are aligned 0/1 lists. Raises on length mismatch; if no
    base-correct items, returns 0.0 (nothing could flip)."""
    b = [1 if bool(x) else 0 for x in base_correct]
    p = [1 if bool(x) else 0 for x in perturbed_correct]
    if len(b) != len(p):
        raise ValueError("base_correct and perturbed_correct must be the same length")
    base_right = [i for i in range(len(b)) if b[i] == 1]
    if not base_right:
        return 0.0
    flips = sum(1 for i in base_right if p[i] == 0)
    return flips / len(base_right)


# --------------------------------------------------------------- conformal

def conformal_halfwidth(calib_residuals, alpha=0.05):
    """Split-conformal prediction half-width (distribution-free, finite-sample
    coverage >= 1−alpha): the ceil((n+1)(1−alpha))-th smallest absolute
    calibration residual. Wrap any point prediction as pred ± this to get an
    interval with a coverage GUARANTEE and no normality assumption — complements
    eval_stats' parametric t-CI (standard split-conformal; Papadopoulos et al.
    2002 / Vovk et al., textbook method). If the required rank
    exceeds n, coverage cannot be guaranteed -> returns +inf (honest)."""
    res = sorted(abs(float(r)) for r in calib_residuals)
    n = len(res)
    if n == 0:
        raise ValueError("need at least one calibration residual")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return float("inf")   # too few calibration points for this alpha
    return res[rank - 1]      # 1-indexed rank -> 0-indexed


# ── §C25/interp prerequisites (verified absent tree-wide before adding; build_spec §3) ──
def roc_auc(scores, labels):
    """Rank-based ROC-AUC (Mann-Whitney U, ties averaged) — equals the trapezoid ROC,
    deterministic, no sklearn. `labels` are 0/1; AUC = P(score(pos) > score(neg)), ties = 0.5.
    Returns nan if either class is empty (undefined)."""
    s = [float(x) for x in scores]
    y = [int(v) for v in labels]
    if len(s) != len(y):
        raise ValueError("scores and labels length mismatch")
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(s):                                   # tie-averaged 1-indexed ranks
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(s)) if y[i] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_ci(stat_fn, data, n=2000, alpha=0.05, seed=0, method="percentile", paired=False):
    """Seeded percentile bootstrap CI — byte-identical reruns via random.Random(seed).
    `data`: a sequence (paired=False) or a tuple of ALIGNED sequences (paired=True). For paired,
    ONE resample-index set per replicate is shared across every array, so a `(A−B)` stat keeps
    pairs intact — the CI-disjoint-vs-floor gate in interp.py requires this. (point, lo, hi)."""
    import random as _r
    if method != "percentile":
        raise NotImplementedError("only method='percentile' is implemented (BCa is optional/future)")
    rng = _r.Random(seed)
    if paired:
        arrs = [list(a) for a in data]
        m = len(arrs[0])
        assert all(len(a) == m for a in arrs), "paired arrays must align"
        point = stat_fn(*arrs)
        boots = []
        for _ in range(n):
            idx = [rng.randrange(m) for _ in range(m)]
            boots.append(stat_fn(*[[a[i] for i in idx] for a in arrs]))
    else:
        arr = list(data)
        m = len(arr)
        point = stat_fn(arr)
        boots = []
        for _ in range(n):
            idx = [rng.randrange(m) for _ in range(m)]
            boots.append(stat_fn([arr[i] for i in idx]))
    boots.sort()

    def q(p):
        if not boots:
            return float("nan")
        pos = p * (len(boots) - 1)
        lo = int(pos)
        if lo + 1 >= len(boots):
            return boots[-1]
        return boots[lo] * (1 - (pos - lo)) + boots[lo + 1] * (pos - lo)

    return (point, q(alpha / 2.0), q(1.0 - alpha / 2.0))


def mcnemar(b, c):
    """Paired binary test on the 2x2 discordant counts (b = A-right-B-wrong, c = B-right-A-wrong).
    Exact two-sided binomial when b+c < 25, else continuity-corrected chi-square (df=1).
    Returns (statistic, p_value, method). b==c -> p=1.0."""
    b, c = int(b), int(c)
    ntot = b + c
    if ntot == 0:
        return (0.0, 1.0, "degenerate")
    if ntot < 25:
        k = min(b, c)
        p = 2.0 * sum(math.comb(ntot, i) for i in range(k + 1)) * (0.5 ** ntot)
        return (float(k), min(1.0, p), "exact-binomial")
    chi2 = (abs(b - c) - 1.0) ** 2 / ntot
    return (chi2, math.erfc(math.sqrt(chi2 / 2.0)), "chi2-continuity")   # chi2 df=1 survival
