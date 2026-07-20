"""Pure post-training loss/objective math — stdlib-only, no torch, no model, no
network. The CPU-buildable, unit-testable CORE of the post-training executor
(SFT / DPO / GRPO), mirroring the eval_stats.py / eval_metrics.py pattern: the
subtle math lives here and is proven correct WITHOUT a GPU, so the SKILL is not
"documented but fake" — only the actual training loop (forward/backward on a
checkpoint) is the GPU-staged part.

Lifecycle slot: §C13 `finetune` objective on an EXISTING checkpoint (NOT a
from-scratch build, §C4.2). Methods covered:
  * SFT     — standard next-token CE; helper here is the (optionally
              label-smoothed) per-token NLL from a target log-prob.
  * DPO     — Direct Preference Optimization closed-form loss (no reward model,
              no RL rollout): a function of policy & reference log-probs on a
              chosen/rejected pair.
  * GRPO    — Group-Relative Policy Optimization: group-normalized advantages +
              a PPO-style clipped surrogate + the k3 KL estimator.

Everything operates on plain Python lists of per-example scalar log-probs /
rewards / ratios (whatever the training loop reduces a sequence to), so it is
deterministic and directly testable (test_posttrain_losses.py).
"""
from __future__ import annotations

import math
import statistics


# --------------------------------------------------------------- helpers

def log_sigmoid(x):
    """Numerically-stable log(sigmoid(x)) = -softplus(-x)."""
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


# --------------------------------------------------------------- SFT

def label_smoothed_nll(logp_target, vocab_size, smoothing=0.0):
    """Per-token SFT loss from the model's log-prob of the GOLD token.
    smoothing=0 -> plain negative log-likelihood (-logp_target). With label
    smoothing eps, the target distribution is (1-eps) on the gold token and
    eps/V uniform elsewhere, giving loss = (1-eps)*(-logp_target) + eps*mean_nll,
    approximated (uniform-prior surrogate) as (1-eps)*(-logp) + eps*log(V).
    Raises on an out-of-range smoothing."""
    if not (0.0 <= smoothing < 1.0):
        raise ValueError("smoothing must be in [0,1)")
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    return (1 - smoothing) * (-logp_target) + smoothing * math.log(vocab_size)


# --------------------------------------------------------------- DPO

def dpo_loss(logp_pol_chosen, logp_ref_chosen,
             logp_pol_rejected, logp_ref_rejected, beta=0.1):
    """Direct Preference Optimization loss over a batch of preference pairs.
    Inputs are equal-length lists of SEQUENCE log-probs (sum of token log-probs)
    under the policy and the frozen reference, for the chosen and rejected
    responses. Returns a dict:

      loss         = mean of -log_sigmoid( beta * (chosen_logratio - rejected_logratio) )
      chosen_reward / rejected_reward = mean implicit reward beta*(logp_pol - logp_ref)
      accuracy     = fraction of pairs where chosen_reward > rejected_reward
      margin       = mean(chosen_reward - rejected_reward)

    No reward model, no rollouts — DPO's whole point is this closed form.
    """
    n = len(logp_pol_chosen)
    if not (len(logp_ref_chosen) == len(logp_pol_rejected) == len(logp_ref_rejected) == n):
        raise ValueError("all four log-prob lists must be the same length")
    if n == 0:
        raise ValueError("need at least one preference pair")
    losses, ch_r, rj_r, correct = [], [], [], 0
    for pc, rc, pr, rr in zip(logp_pol_chosen, logp_ref_chosen,
                              logp_pol_rejected, logp_ref_rejected):
        chosen_logratio = pc - rc          # log pi(y_w)/pi_ref(y_w)
        rejected_logratio = pr - rr        # log pi(y_l)/pi_ref(y_l)
        losses.append(-log_sigmoid(beta * (chosen_logratio - rejected_logratio)))
        cr, rj = beta * chosen_logratio, beta * rejected_logratio
        ch_r.append(cr)
        rj_r.append(rj)
        if cr > rj:
            correct += 1
    return {
        "loss": sum(losses) / n,
        "chosen_reward": sum(ch_r) / n,
        "rejected_reward": sum(rj_r) / n,
        "margin": sum(c - r for c, r in zip(ch_r, rj_r)) / n,
        "accuracy": correct / n,
        "n": n,
    }


# --------------------------------------------------------------- GRPO

def group_normalized_advantages(rewards, eps=1e-6):
    """GRPO advantage = (r - mean(r)) / (std(r) + eps) over a sampled GROUP of
    rollouts for one prompt (population std). A group whose rewards are all equal
    yields all-zero advantages (std 0 -> divide by eps -> ~0)."""
    rs = [float(r) for r in rewards]
    if not rs:
        raise ValueError("need at least one reward")
    mu = statistics.fmean(rs)
    sd = statistics.pstdev(rs) if len(rs) > 1 else 0.0
    return [(r - mu) / (sd + eps) for r in rs]


def grpo_clipped_objective(ratios, advantages, clip_eps=0.2):
    """PPO/GRPO clipped surrogate to MAXIMIZE (the training loss is its
    negative). Per sample: min(ratio*A, clip(ratio, 1-eps, 1+eps)*A); returns the
    mean. ratio = pi_new/pi_old. The min correctly handles both advantage signs
    (it caps the gain on positive A and does not reward moving further on the
    clipped side for negative A)."""
    if len(ratios) != len(advantages):
        raise ValueError("ratios and advantages must be the same length")
    if not ratios:
        raise ValueError("need at least one sample")
    lo, hi = 1.0 - clip_eps, 1.0 + clip_eps
    terms = []
    for r, a in zip(ratios, advantages):
        clipped = min(max(r, lo), hi)
        terms.append(min(r * a, clipped * a))
    return sum(terms) / len(terms)


def kl_penalty_k3(logp_pol, logp_ref):
    """Schulman's low-variance, unbiased k3 KL estimator, mean over tokens:
    KL ~= mean( exp(logp_ref - logp_pol) - (logp_ref - logp_pol) - 1 ). Always
    >= 0; exactly 0 when the policy equals the reference. Used as the GRPO KL
    regularizer toward the frozen reference."""
    if len(logp_pol) != len(logp_ref):
        raise ValueError("logp_pol and logp_ref must be the same length")
    if not logp_pol:
        raise ValueError("need at least one token")
    out = []
    for lp, lr in zip(logp_pol, logp_ref):
        d = lr - lp
        out.append(math.exp(d) - d - 1.0)
    return sum(out) / len(out)


def grpo_loss(ratios, advantages, logp_pol, logp_ref, clip_eps=0.2, beta_kl=0.0):
    """The ASSEMBLED GRPO loss to MINIMIZE: the NEGATIVE clipped surrogate (the
    surrogate is a reward to MAXIMIZE) plus a positive KL penalty toward the
    frozen reference. A naive `minimize(grpo_clipped_objective)` ascends the WRONG
    direction and wastes the whole RL run — this helper fixes the sign in one
    place. Returns {loss, surrogate, kl}."""
    surrogate = grpo_clipped_objective(ratios, advantages, clip_eps)
    kl = kl_penalty_k3(logp_pol, logp_ref) if beta_kl else 0.0
    return {"loss": -surrogate + beta_kl * kl, "surrogate": surrogate, "kl": kl}


# --------------------------------------------------------------- SFT masking

def masked_sft_nll(per_token_logp, response_mask, smoothing=0.0, vocab_size=None):
    """SFT loss averaged over RESPONSE tokens only. `per_token_logp[i]` is the
    model's log-prob of the gold token at position i; `response_mask[i]` in {0,1}
    selects completion tokens. Training SFT on PROMPT tokens is objective
    misspecification, so masking is mandatory — `label_smoothed_nll` takes a
    single scalar and delegates the masking to this caller. `vocab_size` is
    required when smoothing > 0."""
    if len(per_token_logp) != len(response_mask):
        raise ValueError("per_token_logp and response_mask must be the same length")
    if smoothing and vocab_size is None:
        raise ValueError("vocab_size required when smoothing > 0")
    num, den = 0.0, 0
    for lp, m in zip(per_token_logp, response_mask):
        if not m:
            continue
        num += label_smoothed_nll(lp, vocab_size, smoothing) if smoothing else (-lp)
        den += 1
    if den == 0:
        raise ValueError("response_mask selects zero tokens")
    return num / den
