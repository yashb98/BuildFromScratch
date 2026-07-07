#!/usr/bin/env python3
"""Phase-2 exploratory RLVR verdict builder (research/rlvr/plan.md).

Reads the paired pass@k scores — SFT floor (from Phase-1) + the GRPO arm + the random-reward
control — all scored on the SAME band (identical items + per-item generation seeds), and answers
the two questions the plan poses:
  (1) does GRPO beat the SFT floor on held-out math EXACT-MATCH pass@1 / pass@8?  (the RL effect)
  (2) does GRPO beat the RANDOM-reward arm?  (the Spurious-Rewards gate — if not, any apparent
      movement is prior elicitation, not reasoning, and must NOT be called a win)
"Beat" = the arm's pass@1 Wilson CI_low exceeds the comparator's CI_high (non-overlapping,
conservative marginal test). Verdict is CAPPED at `directional` regardless — this is n=1 seed and
the §C25 rlvr battery needs seed_ci (≥3 seeds); the point here is the go/no-go for a Phase-3 cohort,
not a headline. Writes verdict.json and updates the ledger run.
"""
import json, pathlib, subprocess

HERE = pathlib.Path(__file__).parent
ROOT = pathlib.Path("/home/yashb98/Downloads/BuildFromScratch")
P1 = ROOT / "Qwen3-0.6B/experiments/2026-07-01_qwen3-0.6b_rlvr-phase1-passk/phase1_passk.json"
RUN_ID = "2026-07-02_qwen3-0.6b_grpo-phase2"
SETS = ["gsm8k", "math500_l13"]


def load(p):
    return json.loads(pathlib.Path(p).read_text())


def arm(path, label):
    return load(path)["checkpoints"][label] if pathlib.Path(path).exists() else None


def main():
    sft = load(P1)["checkpoints"]["sft_seed0"]
    grpo = arm(HERE / "passk_grpo.json", "grpo")
    rnd = arm(HERE / "passk_random.json", "random")
    rft = arm(HERE / "passk_rft.json", "rft")   # expected == SFT floor (0 completions distilled)
    if grpo is None:
        print("no passk_grpo.json — cannot score"); return 1

    rows, proceed = [], False
    for s in SETS:
        a_sft, a_grpo = sft[s]["pass1_wilson_ci"], grpo[s]["pass1_wilson_ci"]
        a_rnd = rnd[s]["pass1_wilson_ci"] if rnd else None
        beats_sft = a_grpo["ci_low"] > a_sft["ci_high"]
        beats_rnd = (a_rnd is not None) and a_grpo["ci_low"] > a_rnd["ci_high"]
        row = {
            "set": s,
            "sft_pass1": round(a_sft["acc"], 4), "grpo_pass1": round(a_grpo["acc"], 4),
            "random_pass1": round(a_rnd["acc"], 4) if a_rnd else None,
            "rft_pass1": round(rft[s]["pass1_wilson_ci"]["acc"], 4) if rft else None,
            "sft_pass8": sft[s]["passk_chen2021"].get("8"),
            "grpo_pass8": grpo[s]["passk_chen2021"].get("8"),
            "random_pass8": rnd[s]["passk_chen2021"].get("8") if rnd else None,
            "rft_pass8": rft[s]["passk_chen2021"].get("8") if rft else None,
            "grpo_pass1_ci": [round(a_grpo["ci_low"], 4), round(a_grpo["ci_high"], 4)],
            "sft_pass1_ci": [round(a_sft["ci_low"], 4), round(a_sft["ci_high"], 4)],
            "grpo_beats_sft_floor": beats_sft,      # non-overlapping Wilson CIs
            "grpo_beats_random_gate": beats_rnd,
        }
        rows.append(row)
        if beats_sft and beats_rnd:
            proceed = True

    if proceed:
        conclusion = ("GRPO clears the SFT floor AND the random-reward gate on >=1 set (non-overlapping "
                      "pass@1 CIs) -> a Phase-3 multi-seed paired-CI cohort is JUSTIFIED.")
    else:
        why = []
        if not any(r["grpo_beats_sft_floor"] for r in rows):
            why.append("GRPO does not beat the SFT floor")
        if rnd is not None and not any(r["grpo_beats_random_gate"] for r in rows):
            why.append("GRPO does not beat the random-reward gate")
        conclusion = ("PREDICTED NULL CONFIRMED: " + "; ".join(why) +
                      ". At ~1% base pass@1 there is nothing for RL to sharpen (GRPO training reward flat at ~0.9% "
                      "correct with no trend [first-50 0.0091 vs last-50 0.0080]; RFT's 352 collected "
                      "completions, ~0.9% of 38,400 rollouts, were too few to separate from the floor). "
                      "Do NOT spend the multi-seed cohort; "
                      "the reasoning gain lives in SFT/distillation, not RL at this scale.")

    verdict = {
        "run_id": RUN_ID, "lifecycle_stage": "rlvr", "seeds": 1,
        "verdict": "directional",   # §C25 rlvr HARD-incomplete (n=1 seed, no seed_ci) caps here
        "verdict_capped_reason": "n=1 seed; §C25 rlvr requires seed_ci (>=3 paired seeds) for a non-directional call",
        "extractor": "math-acc-v1",
        "comparison": rows,
        "grpo_reward_trajectory": ("flat at ~0.9% correct over 300 steps (health_grpo_seed0.jsonl: overall 0.0092, "
                                   "first-50 0.0091 vs last-50 0.0080 — no learning trend). Gradient DID flow "
                                   "(13.5/16 groups kept by DAPO, mostly format-reward variance); correctness never moved."),
        "rft_arm": ("iso-generation control: 352 verifier-correct completions collected (~0.9% of 38,400 rollouts; "
                    "train_rft_seed0.log 14:52) -> masked-SFT pass over them. Scores nominally above the floor "
                    "(gsm8k pass@8 0.11 vs sft 0.07) but CIs overlap — n.s. NOTE: an earlier draft misread the "
                    "July-2 SMOKE log line ('0 collected') as the real run; corrected 2026-07-06.") if rft
                   else "not scored",
        "random_reward_gate": "scored" if rnd else "not scored",
        "proceed_to_phase3": proceed,
        "conclusion": conclusion,
    }
    (HERE / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"verdict": verdict["verdict"], "proceed_to_phase3": proceed,
                      "conclusion": conclusion, "rows": rows}, indent=2))

    # ledger (never hand-edit the JSON — go through ledger.py)
    try:
        subprocess.run(["python3", str(ROOT / "research/ledger/ledger.py"), "update-run", RUN_ID,
                        "--set", "status=done", "--set", "verdict=directional",
                        "--set", f"metrics={json.dumps({k: verdict[k] for k in ('proceed_to_phase3','conclusion','comparison')})}"],
                       check=True)
        print("ledger updated")
    except Exception as e:
        print("ledger update failed (record manually):", repr(e)[:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
