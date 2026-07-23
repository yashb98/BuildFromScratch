"""§C25 Per-Stage Eval-Completeness Gate — the machine-checkable registry + check.

The durable fix for the founding mistake (the pretraining three-build headline
28.65/23.52/29.54 shipped on n=1 FineWeb val PPL with no downstream / no seed CI).
Given a run's `lifecycle_stage` and the set of eval items it actually recorded, this
decides whether the STAGE-DONE *required* battery ran. A run missing any HARD item is
capped BELOW `win` and stamped `incomplete-eval: <missing>`.

Layered ON TOP of the significance gates (§C10/§C13/§C17/§C18/§C21): completeness asks
"did the right battery run?", significance asks "is the number real?". A run is `win`
only if HARD-complete AND significant.

Verdict vocabulary (split 2026-07-22, authority = ledger.py §C8): the cap is no longer
the single word `directional`, which compressed two OPPOSITE realities — "found nothing"
and "found something big, one gate short" — into one token. The cap now preserves that
distinction: a real measured effect held back by a missing HARD item is `promising`, a
measured no-effect is `null`, an uninterpretable contrast is `inconclusive`. `directional`
remains a legal ledger value for historical entries but is NEVER emitted here.

Source of truth for the human matrix: research/eval/per_stage_eval_batteries.md.
Stdlib-only, pure CPU, no network — headless-safe for the loop (research-loop S8 /
ablation-runner Phase 6). Recency per §C25.7: re-research a stage if RESEARCHED_ON is
older than RECENCY_WINDOW_DAYS.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# ── Verdict vocabulary ──────────────────────────────────────────────────────────────
# The ledger is the single source of truth (§C8/§C11); this gate must never invent a word
# `ledger.py` would reject. research/ledger/ is not a package, so load it by absolute path
# (cwd-independent for the headless loop) and deliberately do NOT register it in sys.modules,
# so it cannot shadow the plain `import ledger` used elsewhere (research/tests/test_ledger.py).
_LEDGER_PY = Path(__file__).resolve().parent / "ledger" / "ledger.py"


def _load_ledger_vocabulary():
    spec = importlib.util.spec_from_file_location("_c25_ledger_vocab", _LEDGER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # no import-time side effects: ledger.py's CLI is under __main__
    return frozenset(mod.VERDICTS), frozenset(mod.NEUTRAL_VERDICTS)


VERDICTS, NEUTRAL_VERDICTS = _load_ledger_vocabulary()
DEPRECATED_VERDICTS = frozenset({"directional"})  # legal in old entries; never emitted by this gate
VERDICT_VOCAB = "split-2026-07-22"                # stamped into every result so a v1-era
                                                  # `directional` is never confused with a new call

# Significance verdicts this gate accepts from the §C13/§C17 gate. Anything else (None, a typo,
# the deprecated `directional`) is read as `inconclusive` — fail closed, per §C6: an unreadable
# signal never buys a win.
ACCEPTED_SIGNIFICANCE = frozenset({"win", "promising", "null", "loss", "inconclusive"})

# §C25.3 — what a HARD-incomplete battery downgrades each significance verdict TO. Every value
# is in NEUTRAL_VERDICTS: an incomplete battery can never yield `win`, and can never burn a
# `loss` into never_repeat (ledger.py auto-appends never_repeat[] on verdict=loss).
CAP_WHEN_INCOMPLETE = {
    "win":          "promising",     # real, measured effect — short exactly the missing HARD item(s)
    "promising":    "promising",
    "null":         "null",          # measured no-effect; a missing item cannot manufacture one
    "loss":         "inconclusive",  # measured worse, but an incomplete battery may not condemn it
    "inconclusive": "inconclusive",
}

REGISTRY_VERSION = "v1"
RESEARCHED_ON = "2026-06-22"
RECENCY_WINDOW_DAYS = 120  # §C25.7.2

STAGES = ("data", "architecture", "scaling", "pretrain-run", "mid-training", "base-eval",
          "sft", "preference", "rlvr", "safety", "interpretability", "systems", "serving")

# Per stage: `required` = always-HARD items (must be in the run's recorded items, or the
# run is capped below `win`). `conditional` = HARD only when the named condition is
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
# a HARD significance cap (a missing plot doesn't downgrade a real result's verdict).
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

    `verdict_cap` is the CEILING — the strongest verdict this run may carry — not the final
    call: `None` when the HARD battery is complete (a `win` is permitted), `promising` when a
    HARD item is missing, `inconclusive` when the stage is unknown (§C25.1). gate_verdict()
    picks the actual word inside that ceiling from the significance signal.
    """
    present = set(present_items or ())
    conds = set(conditions or ())
    report_missing = [k for k in GLOBAL_REPORT_ONLY if k not in present]  # §C26 figure check (all stages)
    if lifecycle_stage not in REGISTRY:
        return {"stage": lifecycle_stage, "known": False, "complete": False,
                "missing_hard": [], "verdict_cap": "inconclusive", "report_missing": report_missing,
                "verdict_vocab": VERDICT_VOCAB,
                "reason": f"unknown lifecycle_stage '{lifecycle_stage}' — cannot be win (§C25.1)"}
    spec = REGISTRY[lifecycle_stage]
    required = list(spec["required"])
    for cond, items in spec.get("conditional", {}).items():
        if cond in conds:
            required += items
    missing = [k for k in required if k not in present]
    # §C25.7.3: a disallowed item is the SOLE headline whenever it is present AND — after setting
    # aside report-only artifacts (§C26 `figure`) and the disallowed items themselves — NO admissible
    # effect-measurement signal remains. The old `len(present) == 1` test let a founding-mistake
    # headline (e.g. `valppl_n1_stage_headline`) escape the floor the instant ANY second item was
    # recorded, even a report-only figure — so a confounded n=1 val-PPL run flanked by a plot could
    # still reach `promising`/`win`. Judge sole-ness by what's left after the report-only set, not by count.
    disallowed_present = present & DISALLOWED_SOLE_SIGNAL
    admissible = present - DISALLOWED_SOLE_SIGNAL - set(GLOBAL_REPORT_ONLY)
    bad_sole = sorted(disallowed_present) if (disallowed_present and not admissible) else []
    complete = not missing and not bad_sole
    return {
        "stage": lifecycle_stage, "known": True, "complete": complete,
        "required": required, "present": sorted(present), "missing_hard": missing,
        "disallowed_sole_signal": bad_sole, "report_missing": report_missing,  # §C26: [figure] if no plot
        # ceiling, not the call: a disallowed-sole-signal run has no admissible effect measurement,
        # so its ceiling is `inconclusive` (matching gate_verdict), not `promising` (see docstring).
        "verdict_cap": None if complete else ("inconclusive" if bad_sole else "promising"),
        "registry_version": REGISTRY_VERSION, "researched_on": RESEARCHED_ON,
        "verdict_vocab": VERDICT_VOCAB,
    }


def gate_verdict(lifecycle_stage: str, present_items, significance_verdict: str,
                 conditions=None) -> dict:
    """Compose §C25.3: completeness caps significance.

    significance_verdict: the §C13/§C17 call, one of ACCEPTED_SIGNIFICANCE. Anything else is
    read as `inconclusive` (fail closed). Returns the verdict to record in the ledger — never
    `win` unless the HARD battery is complete, and never the deprecated `directional`.
    """
    sig = significance_verdict if significance_verdict in ACCEPTED_SIGNIFICANCE else "inconclusive"
    c = check_completeness(lifecycle_stage, present_items, conditions)
    if not c["known"]:
        verdict, why = c["verdict_cap"], c["reason"]          # unknown stage → inconclusive (§C25.1)
    elif not c["complete"]:
        why = "incomplete-eval: " + ", ".join(c["missing_hard"] + c["disallowed_sole_signal"])
        if c["disallowed_sole_signal"]:
            # the run's ONLY signal is disallowed as a headline (§C25.7.3) → there is no
            # admissible effect measurement at all, so this cannot even be `promising`.
            verdict = "inconclusive"
        else:
            verdict = CAP_WHEN_INCOMPLETE[sig]
            if sig == "loss":
                why += " — measured worse, but an incomplete battery may not burn a never_repeat loss (§C25.3)"
    elif sig == "win":
        verdict, why = "win", "HARD-complete and significant (§C25.3.5)"
    else:
        verdict, why = sig, "HARD-complete; significance gate decides"
    # Postcondition — the reader that keeps this gate and ledger.VERDICTS in lockstep (§C8/§C11):
    # every emitted word is a current ledger verdict, and an incomplete battery emits only a
    # NEUTRAL one — never `win`, never a `loss` (which auto-appends to never_repeat[]).
    if verdict not in VERDICTS or verdict in DEPRECATED_VERDICTS:
        raise ValueError(f"§C25 gate emitted '{verdict}', not a current ledger verdict")
    if not c["complete"] and verdict not in NEUTRAL_VERDICTS:
        raise ValueError(f"§C25 cap violated: incomplete battery emitted '{verdict}'")
    return {"verdict": verdict, "completeness": c, "significance_verdict": significance_verdict,
            "significance_read_as": sig, "why": why}


def _self_test():
    # data stage: all required present + no active condition -> complete
    full = REGISTRY["data"]["required"]
    assert check_completeness("data", full)["complete"], "full data battery should be complete"
    # drop one required -> capped below win, names the missing item
    part = check_completeness("data", full[:-1])
    assert not part["complete"] and part["verdict_cap"] == "promising"
    assert "second_lr_recheck" in part["missing_hard"], part
    # the founding mistake: a pretrain/base-eval headline on ONLY n=1 val PPL
    bad = check_completeness("base-eval", ["valppl_n1_stage_headline"])
    assert not bad["complete"] and bad["disallowed_sole_signal"] == ["valppl_n1_stage_headline"]
    # conditional fires only when active: architecture touching attention needs the Pareto
    arch_req = REGISTRY["architecture"]["required"]
    assert check_completeness("architecture", arch_req)["complete"]                      # no condition
    c_attn = check_completeness("architecture", arch_req, {"touches_attention_width_depth"})
    assert "kv_ttft_itl_pareto" in c_attn["missing_hard"]                                # condition makes it HARD
    # unknown stage cannot win
    assert check_completeness("frobnicate", ["x"])["verdict_cap"] == "inconclusive"
    # gate_verdict composition: complete+significant=win; incomplete caps below win, and the
    # 2026-07-22 split keeps "found something, one gate short" apart from "found nothing"
    assert gate_verdict("data", full, "win")["verdict"] == "win"
    assert gate_verdict("data", full[:-1], "win")["verdict"] == "promising"
    assert gate_verdict("data", full[:-1], "null")["verdict"] == "null"
    assert gate_verdict("data", full, "loss")["verdict"] == "loss"
    # an incomplete battery may not condemn an arm to never_repeat (§C25.3)
    assert gate_verdict("data", full[:-1], "loss")["verdict"] == "inconclusive"
    # the only signal is §C25.7.3-disallowed -> no admissible effect, so not even `promising`
    assert gate_verdict("base-eval", ["valppl_n1_stage_headline"], "win")["verdict"] == "inconclusive"
    # unreadable significance fails closed
    assert gate_verdict("data", full, "banana")["verdict"] == "inconclusive"
    # the deprecated word is never emitted, from any stage/battery/significance combination
    for st, spec in REGISTRY.items():
        for items in (spec["required"], spec["required"][1:], []):
            for s in sorted(ACCEPTED_SIGNIFICANCE):
                assert gate_verdict(st, items, s)["verdict"] not in DEPRECATED_VERDICTS
    # every stage is well-formed (no key collisions between required and conditional)
    for st, spec in REGISTRY.items():
        req = set(spec["required"])
        for items in spec.get("conditional", {}).values():
            assert not (set(items) & req), f"{st}: conditional overlaps required"
    print(f"eval_completeness self-test PASS — {len(STAGES)} stages, registry {REGISTRY_VERSION} "
          f"({RESEARCHED_ON}), verdict vocab {VERDICT_VOCAB}")


if __name__ == "__main__":
    _self_test()
