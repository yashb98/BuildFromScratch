"""Tests for research/posttrain_losses.py — the SFT/DPO/GRPO math core of the
post-training executor. Pure, deterministic, no model. DPO/GRPO are subtle, so
each property is pinned against a hand-derived closed-form value.
"""
import math

import pytest

import posttrain_losses as pt


# ------------------------------------------------------------------- helpers

def test_log_sigmoid_matches_definition():
    for x in (-5.0, -0.3, 0.0, 0.7, 4.0):
        assert math.isclose(pt.log_sigmoid(x), math.log(1 / (1 + math.exp(-x))), abs_tol=1e-12)


# ------------------------------------------------------------------- SFT

def test_sft_nll_no_smoothing_is_neg_logp():
    assert pt.label_smoothed_nll(-2.0, vocab_size=1000) == 2.0


def test_sft_label_smoothing_interpolates():
    # eps=0.1, logp=-2, V=1000 -> 0.9*2 + 0.1*log(1000)
    expected = 0.9 * 2.0 + 0.1 * math.log(1000)
    assert math.isclose(pt.label_smoothed_nll(-2.0, 1000, smoothing=0.1), expected)


def test_sft_validates():
    with pytest.raises(ValueError):
        pt.label_smoothed_nll(-1.0, 1000, smoothing=1.0)
    with pytest.raises(ValueError):
        pt.label_smoothed_nll(-1.0, vocab_size=1)


# ------------------------------------------------------------------- DPO

def test_dpo_policy_equals_reference_is_log2():
    # policy == reference => logits 0 => loss = -log sigmoid(0) = log 2; reward 0; acc 0.
    r = pt.dpo_loss([-3.0], [-3.0], [-5.0], [-5.0], beta=0.1)
    assert math.isclose(r["loss"], math.log(2), abs_tol=1e-9)
    assert r["chosen_reward"] == 0.0 and r["rejected_reward"] == 0.0
    assert r["accuracy"] == 0.0 and r["margin"] == 0.0


def test_dpo_known_closed_form_value():
    # one pair: pol_chosen=-1, ref_chosen=-2 -> chosen_logratio=+1
    #           pol_rej=-4,   ref_rej=-2     -> rejected_logratio=-2
    # beta=0.5 -> logits = 0.5*(1 - (-2)) = 1.5 ; loss = -log sigmoid(1.5)
    r = pt.dpo_loss([-1.0], [-2.0], [-4.0], [-2.0], beta=0.5)
    assert math.isclose(r["loss"], -pt.log_sigmoid(1.5), abs_tol=1e-12)
    assert math.isclose(r["chosen_reward"], 0.5, abs_tol=1e-12)    # 0.5 * 1
    assert math.isclose(r["rejected_reward"], -1.0, abs_tol=1e-12)  # 0.5 * -2
    assert r["accuracy"] == 1.0 and math.isclose(r["margin"], 1.5)


def test_dpo_strong_preference_drives_loss_down_and_acc_up():
    # policy pushes chosen way up, rejected way down -> large positive logits.
    r = pt.dpo_loss([0.0, 0.0], [-3.0, -3.0], [-9.0, -9.0], [-3.0, -3.0], beta=0.5)
    assert r["loss"] < 0.05 and r["accuracy"] == 1.0


def test_dpo_length_mismatch_raises():
    with pytest.raises(ValueError):
        pt.dpo_loss([-1.0], [-2.0], [-1.0], [-2.0, -3.0])
    with pytest.raises(ValueError):
        pt.dpo_loss([], [], [], [])


# ------------------------------------------------------------------- GRPO

def test_group_advantages_zero_mean_unit_scale():
    adv = pt.group_normalized_advantages([1.0, 2.0, 3.0])
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-9)              # mean-centered
    # pstdev of [1,2,3] = sqrt(2/3); advantages = (-1,0,1)/(sqrt(2/3)+eps).
    # tol allows for the 1e-6 GRPO stabilizer in the denominator.
    s = math.sqrt(2 / 3)
    assert math.isclose(adv[0], -1 / s, abs_tol=1e-3) and math.isclose(adv[2], 1 / s, abs_tol=1e-3)
    assert math.isclose(adv[1], 0.0, abs_tol=1e-9)


def test_group_advantages_all_equal_is_zero():
    assert all(abs(a) < 1e-3 for a in pt.group_normalized_advantages([5.0, 5.0, 5.0]))


def test_grpo_clip_caps_positive_advantage():
    # ratio 2.0, advantage +1, eps 0.2 -> clipped to 1.2*1 = 1.2 (gain capped).
    assert math.isclose(pt.grpo_clipped_objective([2.0], [1.0], clip_eps=0.2), 1.2)


def test_grpo_clip_negative_advantage_uses_unclipped_min():
    # ratio 2.0, advantage -1 -> min(2*-1, 1.2*-1) = min(-2, -1.2) = -2.
    assert math.isclose(pt.grpo_clipped_objective([2.0], [-1.0], clip_eps=0.2), -2.0)


def test_grpo_ratio_one_returns_mean_advantage():
    # ratio==1 everywhere => surrogate == advantage => objective == mean(adv).
    adv = [-0.5, 0.0, 0.5]
    assert math.isclose(pt.grpo_clipped_objective([1.0, 1.0, 1.0], adv), 0.0)


def test_kl_k3_zero_when_equal_and_positive_otherwise():
    assert pt.kl_penalty_k3([-1.0, -2.0], [-1.0, -2.0]) == 0.0
    assert pt.kl_penalty_k3([-1.0, -2.0], [-1.5, -2.5]) > 0.0   # always >= 0


def test_grpo_validations():
    with pytest.raises(ValueError):
        pt.grpo_clipped_objective([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        pt.group_normalized_advantages([])
    with pytest.raises(ValueError):
        pt.kl_penalty_k3([-1.0], [])


# --- assembled GRPO loss (sign-correct) + SFT response masking (audit fix) -----

def test_grpo_loss_decreases_as_policy_improves():
    # advantages favour the chosen direction; a policy that increases its prob on
    # positive-advantage samples (ratio>1) must DRIVE THE ASSEMBLED LOSS DOWN.
    adv = [1.0, 1.0, -1.0]
    worse = pt.grpo_loss(ratios=[1.0, 1.0, 1.0], advantages=adv,
                         logp_pol=[0.0], logp_ref=[0.0])["loss"]
    better = pt.grpo_loss(ratios=[1.3, 1.2, 0.8], advantages=adv,
                          logp_pol=[0.0], logp_ref=[0.0])["loss"]
    assert better < worse


def test_grpo_loss_kl_penalty_raises_loss():
    base = pt.grpo_loss([1.1], [1.0], logp_pol=[-1.0], logp_ref=[-0.5], beta_kl=0.0)
    pen = pt.grpo_loss([1.1], [1.0], logp_pol=[-1.0], logp_ref=[-0.5], beta_kl=0.5)
    assert pen["kl"] > 0 and pen["loss"] > base["loss"]


def test_masked_sft_nll_ignores_prompt_tokens():
    # only the last two tokens are response; the first (huge-loss) prompt token
    # must NOT affect the loss.
    logp = [-9.0, -0.2, -0.4]
    mask = [0, 1, 1]
    got = pt.masked_sft_nll(logp, mask)
    assert got == pytest.approx((0.2 + 0.4) / 2)
    with pytest.raises(ValueError):
        pt.masked_sft_nll([-0.1, -0.2], [0, 0])      # mask selects nothing
