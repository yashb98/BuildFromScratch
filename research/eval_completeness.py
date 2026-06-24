"""§C25 Per-Stage Eval-Completeness Gate — the machine-checkable registry + check.

The durable fix for the founding mistake (the pretraining three-build headline
28.65/23.52/29.54 shipped on n=1 FineWeb val PPL with no downstream / no seed CI).
Given a run's `lifecycle_stage` and the set of eval items it actually recorded, this
decides whether the STAGE-DONE *required* battery ran. A run missing any HARD item is
capped at verdict `directional` (never `win`), stamped `incomplete-eval: <missing>`.

Layered ON TOP of the significance gates (§C10/§C13/§C17/§C18/§C21): completeness asks
"did the right battery run?", significance asks "is the number real?". A run is `win`
only if HARD-complete AND significant.

Source of truth for the human matrix: research/eval/per_stage_eval_batteries.md.
Stdlib-only, pure CPU, no network — headless-safe for the loop (research-loop S8 /
ablation-runner Phase 6). Recency per §C25.7: re-research a stage if RESEARCHED_ON is
older than RECENCY_WINDOW_DAYS.
"""
from __future__ import annotations

REGISTRY_VERSION = "v1"
RESEARCHED_ON = "2026-06-22"
RECENCY_WINDOW_DAYS = 120  # §C25.7.2

STAGES = ("data", "architecture", "scaling", "pretrain-run", "mid-training", "base-eval",
          "sft", "preference", "rlvr", "safety", "interpretability", "systems", "serving")

# Per stage: `required` = always-HARD items (must be in the run's recorded items, or the
# run is capped to `directional`). `conditional` = HARD only when the named condition is
# active (e.g. an architecture change that touches attention requires the KV/latency Pareto).
# Item keys are abstract identifiers the supplying skill stamps into the run's metrics.
REGISTRY: dict[str, dict] = {
    "data": {
        "required": ["provenance_sha_license", "doc_disjoint_split", "decontam_report",
                     "tokenizer_fertility", "downstream_battery_seedci", "second_lr_recheck"],
        "conditional": {"dedup_claim": ["dedup_effect_measured"]},
    },
    "architecture": {
        "required": ["iso_flop", "seed_ci_bpb", "ladder_3rung_trend", "single_axis_isolation", "suite_version"],
        "conditional": {"touches_attention_width_depth": ["kv_ttft_itl_pareto"],
                        "touches_positions": ["length_extrapolation_curve"]},
    },
    "scaling": {
        "required": ["log_space_fit", "log_rmse_r2", "holdout_extrapolation_pctdev", "bootstrap_forecast_ci"],
        "conditional": {"mup_claim": ["mup_coordinate_check", "lr_transfer_across_3_widths"]},
    },
    "pretrain-run": {
        "required": ["predicted_curve_residual", "spike_score", "grad_norm_stationarity",
                     "nan_inf_guard", "per_domain_loss", "throughput_drift", "resume_equality"],
        "conditional": {},
    },
    "mid-training": {
        "required": ["effective_context_length_ruler", "short_ctx_non_regression"],
        "conditional": {"anneal_claim": ["anneal_gain_vs_iso_token_control"]},
    },
    "base-eval": {
        "required": ["heldout_bpb", "contam_resistant_portfolio_ci", "per_task_bpb_or_no_signal", "private_heldout"],
        "conditional": {},
    },
    "sft": {
        "required": ["instruction_following_ifeval", "downstream_gain", "forgetting_retention", "seed_ci"],
        "conditional": {},
    },
    "preference": {
        "required": ["heldout_pref_acc_ci", "kl_frontier", "judge_winrate_length_controlled",
                     "reward_hacking_probe", "forgetting_retention", "seed_ci"],
        "conditional": {"reward_model_in_loop": ["rewardbench2"]},
    },
    "rlvr": {
        "required": ["pass1_wilson_ci", "passk_chen2021", "spurious_reward_control_gate",
                     "verifier_honesty_ipt", "extractor_pinned", "grpo_health", "seed_ci"],
        "conditional": {},
    },
    "safety": {
        "required": ["dangerous_cap_incapacity_caveated", "adaptive_asr_before_after",
                     "over_refusal_counter", "redteam_effort_verdict", "elicitation_validity", "asl_determination"],
        "conditional": {"open_weight_release": ["finetuning_attack_class"]},
    },
    "interpretability": {
        "required": ["control_floor_first", "ci_disjoint_vs_baseline", "suite_stamped"],
        "conditional": {},
    },
    "systems": {
        "required": ["mfu_hfu_peak_provenance", "oi_region", "tokens_per_joule"],
        "conditional": {"multi_gpu": ["distributed_correctness_first"], "kernel": ["kernel_oracle_first", "fast_p_per_category"]},
    },
    "serving": {
        "required": ["p50_p99_swept_to_knee", "median_k_ci", "goodput_at_slo",
                     "harness_integrity", "output_correctness_vs_reference"],
        "conditional": {"quantization": ["quant_ppl_kl_flip_vs_floor"]},
    },
}

# §C26 — every step ships a FIGURE. Global report-only item checked for ALL stages: a run
# without a `figure` artifact is flagged `report_missing: [figure]` and may not be rendered to a
# README/digest until `research/eval_plots.figure_for_run(run_dir)` has produced one. Tracked, not
# a HARD significance cap (a missing plot doesn't make a real result `directional`).
GLOBAL_REPORT_ONLY = ("figure",)

# §C25.7.3 — items that may NEVER be a stage's SOLE headline signal (auto-flagged).
DISALLOWED_SOLE_SIGNAL = frozenset({
    "mmlu_raw", "humaneval_alone", "hellaswag_headline", "gsm8k_original_absolute",
    "glue", "superglue", "leaderboard_rank", "alpacaeval_raw", "throughput_no_slo",
    "datasheet_sparse_fp4_peak", "gpu_util_pct_as_efficiency", "single_needle_niah",
    "advertised_context_window", "ngram_only_decontam", "fertility_only_tokenizer",
    "valppl_n1_stage_headline",  # the founding mistake
})


def check_completeness(lifecycle_stage: str, present_items, conditions=None) -> dict:
    """Return the completeness verdict for one run.

    present_items: iterable of recorded eval-item keys the run actually produced.
    conditions:    iterable of active condition flags (e.g. {"quantization"}).
    """
    present = set(present_items or ())
    conds = set(conditions or ())
    report_missing = [k for k in GLOBAL_REPORT_ONLY if k not in present]  # §C26 figure check (all stages)
    if lifecycle_stage not in REGISTRY:
        return {"stage": lifecycle_stage, "known": False, "complete": False,
                "missing_hard": [], "verdict_cap": "inconclusive", "report_missing": report_missing,
                "reason": f"unknown lifecycle_stage '{lifecycle_stage}' — cannot be win (§C25.1)"}
    spec = REGISTRY[lifecycle_stage]
    required = list(spec["required"])
    for cond, items in spec.get("conditional", {}).items():
        if cond in conds:
            required += items
    missing = [k for k in required if k not in present]
    bad_sole = sorted(present & DISALLOWED_SOLE_SIGNAL) if len(present) == 1 else []
    complete = not missing and not bad_sole
    return {
        "stage": lifecycle_stage, "known": True, "complete": complete,
        "required": required, "present": sorted(present), "missing_hard": missing,
        "disallowed_sole_signal": bad_sole, "report_missing": report_missing,  # §C26: [figure] if no plot
        "verdict_cap": None if complete else "directional",
        "registry_version": REGISTRY_VERSION, "researched_on": RESEARCHED_ON,
    }


def gate_verdict(lifecycle_stage: str, present_items, significance_verdict: str,
                 conditions=None) -> dict:
    """Compose §C25.3: completeness caps significance.
    significance_verdict ∈ {win, loss, inconclusive} from the existing §C13/§C17 gate."""
    c = check_completeness(lifecycle_stage, present_items, conditions)
    if not c["complete"]:
        verdict = c["verdict_cap"]   # "directional" (known but missing HARD) | "inconclusive" (unknown stage, §C25.1)
        why = ("incomplete-eval: " + ", ".join(c["missing_hard"] + c.get("disallowed_sole_signal", []))
               if c["known"] else c["reason"])
    elif significance_verdict == "win":
        verdict, why = "win", "HARD-complete and significant (§C25.3.5)"
    else:
        verdict, why = significance_verdict, "HARD-complete; significance gate decides"
    return {"verdict": verdict, "completeness": c, "significance_verdict": significance_verdict, "why": why}


def _self_test():
    # data stage: all required present + no active condition -> complete
    full = REGISTRY["data"]["required"]
    assert check_completeness("data", full)["complete"], "full data battery should be complete"
    # drop one required -> directional, names the missing item
    part = check_completeness("data", full[:-1])
    assert not part["complete"] and part["verdict_cap"] == "directional"
    assert "second_lr_recheck" in part["missing_hard"], part
    # the founding mistake: a pretrain/base-eval headline on ONLY n=1 val PPL
    bad = check_completeness("base-eval", ["valppl_n1_stage_headline"])
    assert not bad["complete"] and bad["verdict_cap"] == "directional"
    # conditional fires only when active: architecture touching attention needs the Pareto
    arch_req = REGISTRY["architecture"]["required"]
    assert check_completeness("architecture", arch_req)["complete"]                      # no condition
    c_attn = check_completeness("architecture", arch_req, {"touches_attention_width_depth"})
    assert "kv_ttft_itl_pareto" in c_attn["missing_hard"]                                # condition makes it HARD
    # unknown stage cannot win
    assert check_completeness("frobnicate", ["x"])["verdict_cap"] == "inconclusive"
    # gate_verdict composition: complete+significant=win; incomplete caps to directional
    assert gate_verdict("data", full, "win")["verdict"] == "win"
    assert gate_verdict("data", full[:-1], "win")["verdict"] == "directional"
    assert gate_verdict("data", full, "loss")["verdict"] == "loss"
    # every stage is well-formed (no key collisions between required and conditional)
    for st, spec in REGISTRY.items():
        req = set(spec["required"])
        for items in spec.get("conditional", {}).values():
            assert not (set(items) & req), f"{st}: conditional overlaps required"
    print(f"eval_completeness self-test PASS — {len(STAGES)} stages, registry {REGISTRY_VERSION} ({RESEARCHED_ON})")


if __name__ == "__main__":
    _self_test()
