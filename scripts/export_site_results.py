#!/usr/bin/env python3
"""Export site-ready results JSONs for yashbishnoi.io.

Reads the committed experiment verdicts/logs and emits one JSON per registered
site claim into Qwen3-0.6B/results/, each conforming to the site contract
(yashbishnoi-io/results-schema/schema.json). Also writes a repo-root
evidence_manifest.json with repo-relative source paths.

Conventions preserved from the paper "Reproduce, Then Attribute":
  - improvement_bpb = baseline_mean - treatment_mean (positive = treatment wins,
    lower BPB is better). Never flip signs.
  - `seeds` arrays on delta entries are PAIRED per-seed deltas
    (baseline_seed_i - treatment_seed_i), ordered seed0, seed1, seed2.
  - `ci` is the Welch-t 95% CI exactly as stored in the source verdict.
  - Evidence never gets promoted: n=1 numbers ship without `seeds` so the site
    renders its "descriptive / n=1" badge automatically.

Run from the repo root:  python3 scripts/export_site_results.py
"""

import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "Qwen3-0.6B"
EXP = Q / "experiments"
OUT = Q / "results"
GENERATED_BY = "scripts/export_site_results.py"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def paired_deltas(baseline, treatment):
    """Per-seed paired deltas, positive = treatment better (lower is better)."""
    assert len(baseline) == len(treatment)
    return [b - t for b, t in zip(baseline, treatment)]


def result_file(id_, kind, *, unit=None, value=None, ci=None, seeds=None,
                series=None, entries=None, rows=None, notes=None,
                derived_from=None, direction=None):
    d = {"id": id_, "kind": kind}
    if unit is not None:
        d["unit"] = unit
    if value is not None:
        d["value"] = value
    if ci is not None:
        d["ci"] = ci
    if seeds is not None:
        d["seeds"] = seeds
    if series is not None:
        d["series"] = series
    if entries is not None:
        d["entries"] = entries
    if rows is not None:
        d["rows"] = rows
    if direction is not None:
        d["direction"] = direction
    d["generated_by"] = GENERATED_BY
    d["generated_at"] = NOW
    if notes is not None:
        d["notes"] = notes
    if derived_from is not None:
        d["derived_from"] = derived_from
    return d


# --------------------------------------------------------------------------
# Per-claim exporters. Each returns (filename, dict).
# --------------------------------------------------------------------------

def export_parity():
    src = Q / "builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/verify.json"
    v = load(src)
    assert v["passed"] and v["max_abs_error"] == 0.0
    return "parity.json", result_file(
        "qwen3.parity", "scalar",
        unit="max |Δlogits|", value=v["max_abs_error"],
        notes=(f"fp32, prompt {v['prompt']!r} (input {v['input_shape'][0]}x"
               f"{v['input_shape'][1]}), vs {v['repo']}; relative error "
               f"{v['relative_error']}, argmax match {v['argmax_match']} "
               f"({v['hf_next_token_text']!r}, id {v['hf_next_token_id']}), "
               f"tolerance {v['tolerance']}, single deterministic CPU probe."),
        derived_from=[rel(src)])


def export_parity_replay():
    src = Q / "builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/verify.json"
    v = load(src)
    runner = Q / "builds/2026-06-08_reproduce-faithful_qwen3-0.6b/verify_run.py"
    lines = [
        f"$ python verify_run.py",
        f"Loading {v['repo']} in fp32 (this downloads ~1.2GB on first run) ...",
        "Building ours and copying weights ...",
        "",
        "=== Verify gate ===",
        f"max |Δlogits| = {v['max_abs_error']:.3e}  (tolerance {v['tolerance']})",
        f"relative      = {v['relative_error']:.3e}",
        f"HF next token : {v['hf_next_token_text']!r}  (id {v['hf_next_token_id']})",
        f"Ours next     : {v['our_next_token_text']!r}  (id {v['our_next_token_id']})",
        f"argmax match  : {v['argmax_match']}",
        "",
        "Wrote results/verify.json",
        "",
        "✓ Verify gate PASSED.",
    ]
    return "parity_replay.json", result_file(
        "qwen3.parity_replay", "table",
        rows=[{"line": ln} for ln in lines],
        notes=("Replay lines reconstructed deterministically from verify_run.py "
               "print statements + the committed results/verify.json values; "
               "per-stage timings omitted (never recorded; total wall time "
               f"{v['total_seconds']}s). Every number is the recorded one; the "
               "output path is shown repo-relative (the script prints it "
               "absolute)."),
        derived_from=[rel(src), rel(runner)])


def export_arch_axis():
    src = EXP / "2026-06-18_qwen3-0.6b_imu1-deconfound-p1/verdict.json"
    v = load(src)
    corpus = v["headline_corpus"]  # wikitext2_val
    order = [("arch", "arch (bundle: vr+ln+hg)"),
             ("wsd", "wsd schedule"),
             ("zloss", "z-loss")]
    entries = []
    for axis, name in order:
        a = v["axes"][axis][corpus]
        entries.append({
            "name": name,
            "value": a["improvement_bpb"],
            "ci": a["ci95"],
            "seeds": paired_deltas(a["baseline_bpb"], a["treatment_bpb"]),
            "significant": a["significant"],
        })
    code = {ax: v["axes"][ax]["code_py"]["improvement_bpb"] for ax, _ in order}
    return "arch_axis.json", result_file(
        "qwen3.arch_axis", "comparison",
        unit="ΔBPB vs faithful baseline (wikitext-2)",
        entries=entries,
        direction="higher_is_better",
        notes=("Phase-1 de-confound: 2000 steps / 131,072,000 tokens per cell, "
               "3 paired seeds, iso-FLOP (ratio 1.00042), text-lm-v2, Welch-t "
               "95% CI. seeds[] are paired per-seed deltas (baseline - treatment). "
               f"Only arch survives; drivers={v['drivers']}. code_py deltas: "
               f"arch +{code['arch']:.4f} (sig), wsd +{code['wsd']:.4f} (n.s.), "
               f"zloss {code['zloss']:+.4f} (n.s.)."),
        derived_from=[rel(src), rel(src.parent / "cohort_bpb.json"),
                      rel(src.parent / "c5_evidence.json")])


def export_arch_subdrill():
    src = EXP / "2026-06-21_qwen3-0.6b_arch-subdrill-p2/verdict.json"
    p1 = EXP / "2026-06-18_qwen3-0.6b_imu1-deconfound-p1/verdict.json"
    v = load(src)
    bundle = load(p1)["axes"]["arch"]
    corpus = v["headline_corpus"]
    names = {"vr": "value-residual", "ln": "LN-scaling", "hg": "head-gating"}
    entries = []
    for axis in ("vr", "ln", "hg"):
        a = v["axes"][axis][corpus]
        entries.append({
            "name": names[axis],
            "value": a["improvement_bpb"],
            "ci": a["ci95"],
            "seeds": paired_deltas(a["baseline_bpb"], a["treatment_bpb"]),
            "significant": a["significant"],
        })
    sub_sum = sum(e["value"] for e in entries)
    measured = bundle[corpus]["improvement_bpb"]
    residual = measured - sub_sum
    sub_sum_code = sum(v["axes"][ax]["code_py"]["improvement_bpb"] for ax in ("vr", "ln", "hg"))
    measured_code = bundle["code_py"]["improvement_bpb"]
    return "arch_subdrill.json", result_file(
        "qwen3.arch_subdrill", "comparison",
        unit="ΔBPB vs faithful baseline (wikitext-2)",
        entries=entries,
        direction="higher_is_better",
        notes=("Phase-2 sub-drill of the Phase-1 arch bundle; same budget, "
               "baseline checkpoints reused and re-scored in-cohort. All three "
               "flags individually significant on both corpora. Additivity: "
               f"sum vr+ln+hg = {sub_sum:.4f} vs measured bundle {measured:.4f} "
               f"(residual {residual:+.4f} = interactions + seed noise); code_py "
               f"sum {sub_sum_code:.4f} vs bundle {measured_code:.4f} "
               f"(residual {measured_code - sub_sum_code:+.4f}). seeds[] are "
               "paired per-seed deltas."),
        derived_from=[rel(src), rel(p1)])


def export_normuon():
    src = EXP / "2026-06-16_qwen3_normuon-vs-adamw/results/verdict.json"
    v = load(src)
    w = v["by_corpus"][v["headline_corpus"]]
    c = v["by_corpus"]["code_py"]
    return "normuon.json", result_file(
        "qwen3.normuon", "scalar",
        unit="ΔBPB, NorMuon vs AdamW (wikitext-2)",
        value=w["improvement_bpb"], ci=w["ci95"],
        seeds=paired_deltas(w["adamw_bpb"], w["normuon_bpb"]),
        notes=("EARLY-TRAINING scope: 640 steps / 41,943,040 tokens per cell "
               "(~28x under the 1.19B full budget), 3 seeds, Welch-t 95% CI. "
               "Defended by an AdamW LR sweep (1.7e-3/3.5e-3/4.8e-3) whose "
               "spread is ~10x smaller than the gap. Not a scale claim; seeds vary "
               "init+shuffle on a fixed data split. "
               f"code_py: +{c['improvement_bpb']:.4f} "
               f"[{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}], significant."),
        derived_from=[rel(src), rel(src.parent / "cohort_bpb.json"),
                      rel(src.parent / "lr_sweep_bpb.json"),
                      rel(src.parent / "verifier_report.json")])


def export_dclm():
    src = EXP / "2026-06-24_qwen3-0.6b_data-dclm-vs-fineweb/verdict.json"
    v = load(src)
    code = v["axes"]["treatment"]["code_py"]
    wt = v["axes"]["treatment"]["wikitext2_val"]
    return "dclm_data.json", result_file(
        "qwen3.dclm_data", "scalar",
        unit="ΔBPB, dclm-edu vs FineWeb-Edu (code_py)",
        value=code["improvement_bpb"], ci=code["ci95"],
        seeds=paired_deltas(code["baseline_bpb"], code["treatment_bpb"]),
        notes=("Largest single attributed effect in the study: code BPB "
               f"{sum(code['baseline_bpb'])/3:.4f} -> "
               f"{sum(code['treatment_bpb'])/3:.4f} at matched tokens "
               "(2000 steps / 131,072,000 per cell, 3 seeds, Welch-t 95% CI). "
               f"English is a null: wikitext-2 {wt['improvement_bpb']:+.4f} "
               f"[{wt['ci95'][0]:.4f}, {wt['ci95'][1]:.4f}], n.s. "
               "Data beat architecture by an order of magnitude."),
        derived_from=[rel(src), rel(src.parent / "cohort_bpb.json")])


def export_mix():
    src = EXP / "2026-06-26_qwen3-0.6b_data-mix-composition/verdict.json"
    v = load(src)
    entries = []
    for axis, name in (("dclm", "dclm-edu (100%)"), ("mix", "50/50 mix")):
        a = v["axes"][axis]["code_py"]
        entries.append({
            "name": name,
            "value": a["improvement_bpb"],
            "ci": a["ci95"],
            "seeds": paired_deltas(a["baseline_bpb"], a["treatment_bpb"]),
            "significant": a["significant"],
        })
    mix_wt = v["axes"]["mix"]["wikitext2_val"]
    dclm_wt = v["axes"]["dclm"]["wikitext2_val"]
    frac = entries[1]["value"] / entries[0]["value"]
    return "mix.json", result_file(
        "qwen3.mix", "comparison",
        unit="ΔBPB vs FineWeb-Edu baseline (code_py)",
        entries=entries,
        direction="higher_is_better",
        notes=("Data-composition curve: the 50/50 dclm+FineWeb mix keeps "
               f"{frac:.0%} of the pure-dclm code win while holding English "
               f"(mix wikitext-2 {mix_wt['improvement_bpb']:+.4f} "
               f"[{mix_wt['ci95'][0]:.4f}, {mix_wt['ci95'][1]:.4f}] n.s. vs "
               f"dclm {dclm_wt['improvement_bpb']:+.4f} n.s./worse). "
               "Best-of-both; this mix is the data carried into mid-training. "
               "3 seeds, Welch-t 95% CI, seeds[] are paired per-seed deltas."),
        derived_from=[rel(src), rel(src.parent / "cohort_bpb.json")])


def export_anneal():
    src = EXP / "2026-06-30_qwen3-0.6b_midtrain-anneal/verdict.json"
    v = load(src)
    code = v["corpora"]["code_py"]
    wt = v["corpora"]["wikitext2_val"]
    return "anneal.json", result_file(
        "qwen3.anneal", "scalar",
        unit="ΔBPB, premium-mix anneal vs iso-token control (code_py)",
        value=code["improvement_bpb"], ci=code["ci95"],
        seeds=paired_deltas(code["control_bpb"], code["treatment_bpb"]),
        notes=("Mid-training anneal (1-sqrt cooldown 2.5e-4 -> 2.5e-5, 2300 "
               "steps x 65,536 tok) on the 50/50 premium mix vs an iso-token "
               "fineweb-only control that absorbs the LR-decay confound: the "
               "data, not the schedule, carries the win. wikitext-2 (the "
               "pre-registered SECONDARY endpoint, expected null) also came out "
               f"significant: {wt['improvement_bpb']:+.4f} "
               f"[{wt['ci95'][0]:.4f}, {wt['ci95'][1]:.4f}]. 3 seeds, "
               f"Welch-t 95% CI. final_verdict: {v['final_verdict']}."),
        derived_from=[rel(src), rel(src.parent / "cohort_bpb.json")])


def export_three_build():
    logs = {
        "faithful": Q / "builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_after.txt",
        "imu1": Q / "builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log",
        "prope25": Q / "builds/2026-06-08_reproduce-exploratory_qwen3-0.6b/results/qwen3_prope25_2tpp_train.log",
    }
    faithful = float(re.search(r"val PPL: [\d.]+ -> ([\d.]+)", logs["faithful"].read_text()).group(1))
    imu1_evals = re.findall(r"\[eval @ (\d+)\] val PPL=([\d.]+)", logs["imu1"].read_text())
    imu1 = float(dict(imu1_evals)["18000"])
    prope25 = float(re.search(r"DONE\s+final val PPL=([\d.]+)", logs["prope25"].read_text()).group(1))
    entries = [
        {"name": "Faithful baseline (AdamW)", "value": faithful},
        {"name": "Modernized (IMU-1 bundle)", "value": imu1},
        {"name": "Exploratory (partial-RoPE 0.25)", "value": prope25},
    ]
    return "three_build.json", result_file(
        "qwen3.three_build", "comparison",
        unit="in-loop val PPL (n=1, not suite-stamped)",
        entries=entries,
        notes=("Three builds at 1,189,478,400 tokens (18,150 steps x 65,536), "
               "single seed each, trainer-internal eval on held-out FineWeb-Edu. "
               f"The -{(1 - imu1 / faithful) * 100:.1f}% IMU-1 margin is "
               "motivation, not a result: the bundle changes 6 things at once "
               "(incl. NorMuon + cautious-WD); attribution lives in the "
               "de-confound experiments. IMU-1 final = eval@18000 (run DONE at "
               "18,150; no after.txt). Fourth arm (partial-RoPE 0.10) died at "
               "step 5,450/18,150."),
        derived_from=[rel(p) for p in logs.values()],
        direction="lower_is_better")


def export_gap_to_original():
    anchor = Q / "builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/original_vs_repro.txt"
    after = Q / "builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_after.txt"
    smoke = Q / "builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_after.txt"
    txt = anchor.read_text()
    original = float(re.search(r"ORIGINAL.*val PPL =\s+([\d.]+)", txt).group(1))
    lr24 = float(re.search(r"lr24.*val PPL =\s+([\d.]+)", txt).group(1))
    faithful = float(re.search(r"val PPL: [\d.]+ -> ([\d.]+)", after.read_text()).group(1))
    smoke_ppl = float(re.search(r"val PPL: [\d.]+ -> ([\d.]+)",
                                smoke.read_text().split("smoke=True")[1]).group(1))
    rows = [
        {"label": "Original Qwen3-0.6B-Base", "tokens": 36_000_000_000_000,
         "ppl": original, "role": "original"},
        {"label": "Smoke probe", "tokens": 65_536_000, "ppl": smoke_ppl,
         "role": "repro"},
        {"label": "Phase-A best (lr 2.4e-3)", "tokens": 131_072_000, "ppl": lr24,
         "gap_label": "3.5x", "data_label": "~275,000x less data", "role": "repro"},
        {"label": "Phase-B faithful (2TPP)", "tokens": 1_189_478_400, "ppl": faithful,
         "gap_label": "2.14x", "data_label": "~30,000x less data", "role": "repro"},
    ]
    return "gap_to_original.json", result_file(
        "qwen3.gap_to_original", "table",
        unit="val PPL (held-out FineWeb-Edu, 204,800 tok, 50x4096 windows)",
        rows=rows,
        notes=("Single-run, in-loop comparison (n=1, descriptive). The ratio "
               "pairs are load-bearing and must never be conflated: 2.14x at "
               "~30,000x less data (1.19B vs 36T tokens) and 3.5x at ~275,000x "
               "less data (131M vs 36T). Computed: "
               f"{faithful}/{original} = {faithful / original:.4f}; "
               f"{lr24}/{original} = {lr24 / original:.4f}."),
        derived_from=[rel(anchor), rel(after), rel(smoke)])


def export_sft_null():
    inloop_src = EXP / "2026-06-27_qwen3-0.6b_sft-3seed/verdict.json"
    heldout_src = EXP / "2026-06-27_qwen3-0.6b_sft-3seed/reasoning_verdict.json"
    il = load(inloop_src)
    ho = load(heldout_src)
    arms = ho["arms"]
    masked = paired_deltas([arms[f"ctrl_seed{i}"]["reasoning_ppl_masked"] for i in range(3)],
                           [arms[f"sft_seed{i}"]["reasoning_ppl_masked"] for i in range(3)])
    fullseq = paired_deltas([arms[f"ctrl_seed{i}"]["reasoning_ppl_full"] for i in range(3)],
                            [arms[f"sft_seed{i}"]["reasoning_ppl_full"] for i in range(3)])
    inloop = paired_deltas(il["ctrl_final_ppl"], il["sft_final_ppl"])
    entries = [
        {"name": "in-loop 'win' (confounded)", "value": il["sft_vs_control"]["improvement_ppl"],
         "ci": il["sft_vs_control"]["ci95"], "seeds": inloop, "significant": True},
        {"name": "held-out, response-masked (corrected)",
         "value": ho["sft_vs_control"]["improvement_ppl"],
         "ci": ho["sft_vs_control"]["ci95"], "seeds": masked,
         "significant": ho["sft_vs_control"]["significant"]},
        {"name": "held-out, full-sequence (cross-check)",
         "value": ho["sft_vs_control_fullseq"]["improvement_ppl"],
         "ci": ho["sft_vs_control_fullseq"]["ci95"], "seeds": fullseq,
         "significant": ho["sft_vs_control_fullseq"]["significant"]},
    ]
    return "sft_null.json", result_file(
        "qwen3.sft_null", "comparison",
        unit="ΔPPL, SFT vs control (positive = SFT better)",
        entries=entries,
        direction="higher_is_better",
        notes=("The Win That Was Not: the in-loop +0.68 PPL 'win' scores SFT "
               "on RESPONSE tokens but the control on ALL tokens (verdict.json "
               "flags the CONFOUND itself). One fixed held-out set collapses "
               "it to +0.009, and the exact full-sequence cross-check flips "
               f"sign (n.s.). overall_verdict: {ho['overall_verdict']!r}. "
               "3 seeds per arm, ~125M math-reasoning tokens each; seeds[] "
               "are paired per-seed deltas."),
        derived_from=[rel(inloop_src), rel(heldout_src)])


def export_grpo_null():
    src = EXP / "2026-07-02_qwen3-0.6b_grpo-phase2/verdict.json"
    passk_grpo = EXP / "2026-07-02_qwen3-0.6b_grpo-phase2/passk_grpo.json"
    phase1 = EXP / "2026-07-01_qwen3-0.6b_rlvr-phase1-passk/phase1_passk.json"
    v = load(src)
    g = next(c for c in v["comparison"] if c["set"] == "gsm8k")
    m = next(c for c in v["comparison"] if c["set"] == "math500_l13")
    entries = [
        {"name": "SFT floor", "value": g["sft_pass1"], "ci": g["sft_pass1_ci"]},
        {"name": "GRPO", "value": g["grpo_pass1"], "ci": g["grpo_pass1_ci"]},
        {"name": "random-reward gate", "value": g["random_pass1"]},
        {"name": "RFT (iso-generation control)", "value": g["rft_pass1"]},
    ]
    return "grpo_null.json", result_file(
        "qwen3.grpo_null", "comparison",
        unit="pass@1, gsm8k (n=100 items, T=0.8, 8 samples)",
        entries=entries,
        direction="higher_is_better",
        notes=("Pre-registered null, confirmed: GRPO beats neither the SFT "
               "floor nor the random-reward gate (both false on gsm8k AND "
               f"math500_l13). pass@8 gsm8k: sft {g['sft_pass8']}, grpo "
               f"{g['grpo_pass8']}, random {g['random_pass8']}, rft "
               f"{g['rft_pass8']}; math500_l13 pass@1: sft {m['sft_pass1']}, "
               f"grpo {m['grpo_pass1']}, random {m['random_pass1']}, rft "
               f"{m['rft_pass1']}. Verdict {v['verdict']!r} (n=1 seed, capped "
               "by pre-registration); proceed_to_phase3="
               f"{v['proceed_to_phase3']}. At ~1% reward the learning signal "
               "is too sparse at 0.6B. RFT sits nominally above the floor but CIs "
               "overlap (n.s.)."),
        derived_from=[rel(src), rel(passk_grpo), rel(phase1)])


def export_grpo_rewards():
    src = EXP / "2026-07-02_qwen3-0.6b_grpo-phase2/health_grpo_seed0.jsonl"
    steps = [json.loads(ln) for ln in src.read_text().splitlines() if ln.strip()]
    series = [{"x": s["step"], "y": s["mean_reward"]} for s in steps]
    assert len(series) <= 500
    rewards = [s["mean_reward"] for s in steps]
    mean = sum(rewards) / len(rewards)
    kept = sum(s["kept_groups"] for s in steps) / len(steps)
    return "grpo_rewards.json", result_file(
        "qwen3.grpo_rewards", "curve",
        unit="fraction verifier-correct per step",
        series=series,
        notes=(f"GRPO per-step mean reward over {len(series)} steps (16 prompts "
               f"x G=8, T=0.8, lr 1e-6): flat at ~0.9% (overall {mean:.4f}, "
               f"first-50 {sum(rewards[:50]) / 50:.4f} vs last-50 "
               f"{sum(rewards[-50:]) / 50:.4f}). mean_reward is the fraction "
               "verifier-correct (the 0.1 parseable-wrong shaping term is not "
               f"logged). DAPO kept_groups mean {kept:.1f}/16. Random-reward "
               "contrast arm sits at ~0.50. n=1 seed."),
        derived_from=[rel(src)])


def export_passkey_ladder():
    src = EXP / "2026-06-30_qwen3-0.6b_midtrain-anneal/verdict.json"
    detail = EXP / "2026-06-30_qwen3-0.6b_midtrain-anneal/ecl_ladder.json"
    e = load(src)["ecl_ladder"]
    n_cell = e["n_per_rung_per_cell"]
    rows = []
    for rung, acc in e["base"]["per_rung_accuracy"].items():
        lo, hi = e["base"]["per_rung_wilson95"][rung]
        rows.append({"arm": "base", "rung": int(rung), "acc": acc,
                     "wilson_lo": lo, "wilson_hi": hi, "n": n_cell})
    for arm in ("fineweb", "mix"):
        for rung, cell in e["arms"][arm]["per_rung_pooled"].items():
            rows.append({"arm": arm, "rung": int(rung), "acc": cell["acc"],
                         "wilson_lo": cell["wilson95"][0],
                         "wilson_hi": cell["wilson95"][1], "n": n_cell * 3})
    return "passkey_ladder.json", result_file(
        "qwen3.passkey_ladder", "table",
        unit="passkey retrieval accuracy",
        rows=rows,
        notes=("Long-context passkey ladder (ecl-ladder-v1), rungs 512-8192, "
               "5 depths x 8 keys = 40 probes per rung per cell; anneal arms "
               "pooled over 3 seeds (n=120/rung), base is a single checkpoint "
               "(n=40/rung). Wilson 95% CIs. The 4096 dip is BASE-ONLY (0.025 "
               "at its own trained length; both annealed arms hold 0.400 "
               "there) and unexplained — see paper §13. All three lines bump "
               "at 6144. Per-seed detail in ecl_ladder.json."),
        derived_from=[rel(src), rel(detail)])


def export_loss_curves():
    fcsv = Q / "builds/2026-06-08_reproduce-faithful_qwen3-0.6b/results/qwen3_baseline2tpp_train.csv"
    ilog = Q / "builds/2026-06-08_reproduce-modernized_qwen3-0.6b/results/qwen3_imu1_2tpp_train.log"
    with open(fcsv) as f:
        faithful = [{"x": int(r["tok_seen"]), "y": float(r["loss"])}
                    for r in csv.DictReader(f) if int(r["step"]) % 50 == 0]
    ce_lines = re.findall(r"step\s+(\d+)/18150\s+ce ([\d.]+)", ilog.read_text())
    imu1 = [{"x": int(s) * 65536, "y": float(ce)} for s, ce in ce_lines]
    assert len(faithful) <= 500 and len(imu1) <= 500, (len(faithful), len(imu1))
    shared = ("Train CE loss vs tokens seen, 1.19B-token build (n=1, "
              "in-loop, descriptive). Sampled every 50 steps. Overlay partner: ")
    out_f = result_file(
        "qwen3.loss_curve_faithful", "curve", unit="train CE loss",
        series=faithful,
        notes=shared + "qwen3.loss_curve_imu1 (IMU-1 bundle, same token grid).",
        derived_from=[rel(fcsv)])
    out_i = result_file(
        "qwen3.loss_curve_imu1", "curve", unit="train CE loss",
        series=imu1,
        notes=shared + "qwen3.loss_curve_faithful (AdamW baseline, same token "
              "grid). Parsed from the log's every-50-step 'ce' lines.",
        derived_from=[rel(ilog)])
    return [("loss_curve_faithful.json", out_f), ("loss_curve_imu1.json", out_i)]


def export_evidence_manifest():
    src = ROOT / "research/papers/qwen3-0.6b-study/evidence_manifest.json"
    m = load(src)
    prefix = str(ROOT) + "/"
    for cluster in m["clusters"].values():
        for claim in cluster:
            if claim.get("source_file", "").startswith(prefix):
                claim["source_file"] = claim["source_file"][len(prefix):]
    m["note"] = (m.get("note", "") +
                 " Repo-root copy with repo-relative source paths, written by "
                 + GENERATED_BY + " for yashbishnoi.io; source of truth: "
                 + rel(src) + ".")
    return m


# --------------------------------------------------------------------------
# Validation against the site contract.
# --------------------------------------------------------------------------

def validate(name, d):
    errs = []
    for k in ("id", "kind", "generated_by", "generated_at"):
        if k not in d:
            errs.append(f"missing {k}")
    kind = d.get("kind")
    if kind == "scalar" and not isinstance(d.get("value"), (int, float)):
        errs.append("scalar needs numeric value")
    if kind == "curve":
        if not isinstance(d.get("series"), list) or not d["series"]:
            errs.append("curve needs series")
        elif len(d["series"]) > 500:
            errs.append(f"series too long ({len(d['series'])})")
    if kind == "comparison":
        if not isinstance(d.get("entries"), list) or not d["entries"]:
            errs.append("comparison needs entries")
        else:
            for e in d["entries"]:
                if "name" not in e or not isinstance(e.get("value"), (int, float)):
                    errs.append(f"bad entry {e}")
                if "ci" in e and (len(e["ci"]) != 2 or e["ci"][0] > e["ci"][1]):
                    errs.append(f"bad ci on {e['name']}")
                if "seeds" in e and not all(isinstance(s, (int, float)) for s in e["seeds"]):
                    errs.append(f"bad seeds on {e['name']}")
    if kind == "table" and not isinstance(d.get("rows"), list):
        errs.append("table needs rows")
    if "ci" in d and kind == "scalar" and (len(d["ci"]) != 2 or d["ci"][0] > d["ci"][1]):
        errs.append("bad root ci")
    for p in d.get("derived_from", []):
        if not (ROOT / p).exists():
            errs.append(f"derived_from missing: {p}")
    if errs:
        raise SystemExit(f"VALIDATION FAILED for {name}: {errs}")


def tracked(path_rel: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", path_rel],
                       cwd=ROOT, capture_output=True)
    return r.returncode == 0


def main():
    OUT.mkdir(exist_ok=True)
    outputs = [
        export_parity(), export_parity_replay(), export_arch_axis(),
        export_arch_subdrill(), export_normuon(), export_dclm(), export_mix(),
        export_anneal(), export_three_build(), export_gap_to_original(),
        export_sft_null(), export_grpo_null(), export_grpo_rewards(),
        export_passkey_ladder(),
    ]
    outputs += export_loss_curves()

    for name, d in outputs:
        validate(name, d)
        (OUT / name).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
        untracked = [p for p in d.get("derived_from", []) if not tracked(p)]
        flag = f"  [WARN: untracked sources: {untracked}]" if untracked else ""
        summary = d.get("value")
        if summary is None and d.get("entries"):
            summary = ", ".join(f"{e['name']}={e['value']:.4g}" for e in d["entries"])
        elif summary is None and d.get("series"):
            summary = f"{len(d['series'])} points"
        elif summary is None and d.get("rows"):
            summary = f"{len(d['rows'])} rows"
        print(f"✓ {name:28s} {d['kind']:10s} {summary}{flag}")

    manifest = export_evidence_manifest()
    (ROOT / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    n_claims = sum(len(c) for c in manifest["clusters"].values())
    print(f"✓ evidence_manifest.json (repo root, {n_claims} claims, repo-relative paths)")
    print(f"\n{len(outputs)} result files -> {rel(OUT)}/")


if __name__ == "__main__":
    main()
