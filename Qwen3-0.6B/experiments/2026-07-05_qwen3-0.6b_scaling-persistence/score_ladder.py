#!/usr/bin/env python3
"""Score the scaling-persistence ladder → the verdict on whether NorMuon's +0.474 BPB win
PERSISTS or CONVERGES with token budget (closes the IMU-1 RESULT.md Limitation #3).

Per token budget: compute the AdamW−NorMuon BPB gap on wikitext-2 + code (text-lm-v2, reusing the
IMU-1 score_cohort scorer so numbers are byte-identical/comparable), with across-seed CI via the
tested eval_stats.seed_delta_significant; the 42M rung REUSES the existing IMU-1 cohort_bpb.json (no
retrain); then scaling_ladder.fit_gap_trend() over log10(tokens) → {PERSISTS, WIDENS, CONVERGES, FLAT}.

Runs on the GB10 AFTER the ladder cells finish (run_ladder.sh calls it; GPU is free by then). Robust:
scores whatever budgets already have >=2 seeds/arm, so it can also be run mid-cohort for an early read.
Writes verdict.json + registers the ledger run. safe_cuda-guarded; sequential (parallel GPU evals crash).
"""
from __future__ import annotations
import json, math, pathlib, re, shlex, subprocess, sys, time

ROOT = pathlib.Path("/home/yashb98/Downloads/BuildFromScratch")
IMU1 = ROOT / "Qwen3-0.6B/experiments/2026-06-16_qwen3_normuon-vs-adamw"
LDIR = ROOT / "Qwen3-0.6B/experiments/2026-07-05_qwen3-0.6b_scaling-persistence"
RES = IMU1 / "results"                         # train_ablation writes checkpoint_persist_*.pt here
for p in (IMU1, ROOT, ROOT / "research", ROOT / "Qwen3-0.6B"):
    sys.path.insert(0, str(p))
import safe_cuda  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from model import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from eval_stats import seed_delta_significant  # noqa: E402
from scaling_ladder import fit_gap_trend  # noqa: E402
import score_cohort as sc  # noqa: E402  (reuse score() + load_corpora(): text-lm-v2 SEQ/STRIDE/bpb)

BUDGETS = [42_000_000, 168_000_000, 420_000_000, 840_000_000]
SEEDS = {42_000_000: [0, 1, 2], 168_000_000: [0, 1, 2], 420_000_000: [0, 1], 840_000_000: [0, 1]}
CORPORA = ["wikitext2_val", "code_py"]
tagM = lambda t: f"{t // 1_000_000}M"
LEDGER = ROOT / "research/ledger/ledger.py"


def _ladder_started():
    """Ladder start = the first DATED `START <cell>` line in run_ladder.log. Without it
    `add-run` defaults `started` to TODAY — exactly how this run's entry ended up with
    started(2026-07-13) AFTER ended(2026-07-12). Returns None if unreadable."""
    try:
        for line in (LDIR / "run_ladder.log").read_text(errors="replace").splitlines():
            m = re.match(r"\[(\d{4}-\d{2}-\d{2})[ T][^\]]*\]\s+START\b", line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return None


def _ladder_ended():
    """Run end = the mtime of `ladder.done`, which run_ladder.sh touches only after the
    completion gate sees every cell's .done. Falls back to today (scoring date)."""
    try:
        return time.strftime("%Y-%m-%d",
                             time.localtime((LDIR / "ladder.done").stat().st_mtime))
    except OSError:
        return time.strftime("%Y-%m-%d")


def sync_ledger(verdict, ledger_verdict, metrics) -> bool:
    """Land the scored run in the ledger through ledger.py (§C11 — never hand-edit).

    Four things this gets right that the previous one-liner did not:
      * `--type scaling-fit`. A token-budget ladder IS a scaling-fit run;
        `pretrain-ablation` is an OBJECTIVE, not a run type, so passing it as --type made
        argparse reject the whole call (exit 2) and NOTHING was ever written.
      * The failure is LOUD. The old call passed `check=False`, so a non-zero exit raised
        nothing and printed nothing — the write had been failing on every scoring pass,
        invisibly, while the script still returned 0. That silence is why a broken ledger
        call survived a multi-day ladder.
      * Idempotent. `add-run` REJECTS a duplicate run_id (exit 2), so a re-scored ladder
        falls back to `update-run` rather than losing the write.
      * Real dates + terminal state. `started`/`ended` come from the log and the
        `ladder.done` marker instead of defaulting to today, and the run is marked
        `status=done` (add-run defaults to `launched`).
    """
    started, ended = _ladder_started(), _ladder_ended()
    sets = ["--set", "lifecycle_stage=scaling",
            "--set", "status=done",
            "--set", f"ended={ended}",
            "--set", f"suite_version={verdict['suite_version']}",
            "--set", f"verdict={ledger_verdict}",
            "--set", f"metrics={json.dumps(metrics)}"]
    if started:
        sets += ["--set", f"started={started}"]

    proc = subprocess.run(
        ["python3", str(LEDGER), "add-run", "--run-id", verdict["run_id"],
         "--type", "scaling-fit", "--model-dir", "Qwen3-0.6B",
         "--objective", verdict["objective"]] + sets,
        capture_output=True, text=True)
    if proc.returncode == 2 and "already exists" in (proc.stdout + proc.stderr):
        proc = subprocess.run(
            ["python3", str(LEDGER), "update-run", verdict["run_id"]] + sets,
            capture_output=True, text=True)

    if proc.returncode != 0:
        bar = "!" * 78
        print(f"\n{bar}\n"
              f"LEDGER WRITE FAILED — verdict.json IS on disk, the ledger is NOT updated.\n"
              f"  exit : {proc.returncode}\n"
              f"  cmd  : {' '.join(shlex.quote(c) for c in proc.args)}\n"
              f"  error: {(proc.stderr or proc.stdout).strip()[:400]}\n"
              f"  fix  : re-run this scorer, or land it by hand via research/ledger/ledger.py\n"
              f"{bar}\n", flush=True)
        return False
    print(f"ledger: {verdict['run_id']} verdict={ledger_verdict} "
          f"started={started or 'today'} ended={ended}", flush=True)
    return True


def score_ckpt(model, ckpt, corpora, tok):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return {c: sc.score(model, ids, tok)["bpb"] for c, ids in corpora.items()}


def main():
    safe_cuda.guard(0.85)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    corpora = sc.load_corpora(tok)
    model = Qwen3ForCausalLM(Qwen3Config()).to(sc.DEVICE, sc.DTYPE).eval()
    reuse = json.loads((RES / "cohort_bpb.json").read_text())["cells"]   # 42M rung

    # collect per-budget BPBs (skip a budget until both arms have >=2 seeds)
    per_budget = {}
    for t in BUDGETS:
        got = {"adamw": {}, "normuon": {}}
        for arm in ("adamw", "normuon"):
            for s in SEEDS[t]:
                if t == 42_000_000:
                    cell = f"{arm}_seed{s}"
                    if cell in reuse:
                        got[arm][s] = {c: reuse[cell][c]["bpb"] for c in CORPORA}
                else:
                    ck = RES / f"checkpoint_persist_{tagM(t)}_{arm}_s{s}.pt"
                    if ck.exists():
                        t0 = time.time()
                        got[arm][s] = score_ckpt(model, ck, corpora, tok)
                        print(f"  scored persist_{tagM(t)}_{arm}_s{s}: "
                              f"wt={got[arm][s]['wikitext2_val']:.4f} code={got[arm][s]['code_py']:.4f} "
                              f"({time.time()-t0:.0f}s)", flush=True)
        if len(got["adamw"]) >= 2 and len(got["normuon"]) >= 2:
            per_budget[t] = got

    # per-budget gap + CI, per corpus
    gap_by_corpus = {}
    for c in CORPORA:
        pts = []
        for t in sorted(per_budget):
            a = [per_budget[t]["adamw"][s][c] for s in sorted(per_budget[t]["adamw"])]
            n = [per_budget[t]["normuon"][s][c] for s in sorted(per_budget[t]["normuon"])]
            r = seed_delta_significant(a, n, direction="lower_is_better")
            pts.append({"tokens": t, "budget": tagM(t), "gap_bpb": r["improvement"],
                        "ci95": r["ci95"], "significant": r["significant"],
                        "adamw_mean": sum(a) / len(a), "normuon_mean": sum(n) / len(n),
                        "n_seeds": [len(a), len(n)], "warning": r["warning"]})
        gap_by_corpus[c] = pts

    # Fit the trend on BOTH corpora (a headline needs BPB on >=2 corpora, §C10/§C16 — the old
    # code fit wikitext ONLY, so a divergent code trend was silently missed). Noise floor = the
    # MAX per-cell CI half-width across rungs, NOT the median: the median discards the noisiest
    # (here the decisive n=2 420M top) rung and yields an anti-conservative "resolved" verdict.
    def _fit_corpus(pts):
        if len(pts) < 2:
            return {"verdict": "descriptive_only", "verdict_code": "descriptive_only",
                    "reason": "need >=2 budgets scored"}
        hw = [(p["ci95"][1] - p["ci95"][0]) / 2 for p in pts
              if p["ci95"] and all(isinstance(x, (int, float)) and math.isfinite(x) for x in p["ci95"])]
        noise = max(hw) if hw else 0.0
        tr = fit_gap_trend([p["tokens"] for p in pts], [p["gap_bpb"] for p in pts], gap_noise=noise)
        tr["gap_noise_rule"] = "max per-cell CI half-width (conservative; keeps the n=2 top rung's uncertainty)"
        return tr

    trend_by_corpus = {c: _fit_corpus(gap_by_corpus[c]) for c in CORPORA}
    trend = trend_by_corpus["wikitext2_val"]                 # wikitext-2 = the headline corpus
    v = trend.get("verdict_code", trend.get("verdict", "?"))

    # §C17/§C25 CAP: the "at scale" claim rests on the LARGEST scored budget. If that rung has
    # <3 seeds in EITHER arm (the 420M rung is n=2), the headline caps to DIRECTIONAL regardless
    # of what the 3-point OLS says — a trend fit cannot out-rank the seeds>=3 rule.
    scored_budgets = sorted(per_budget)
    top_t = scored_budgets[-1] if scored_budgets else None
    top_seeds = ([len(per_budget[top_t]["adamw"]), len(per_budget[top_t]["normuon"])]
                 if top_t is not None else [0, 0])
    headline_capped = (min(top_seeds) < 3) if top_t is not None else True
    codes = {trend_by_corpus[c].get("verdict_code") for c in CORPORA}
    codes.discard("descriptive_only"); codes.discard(None)
    corpora_agree = len(codes) <= 1

    concl_map = {
        "PERSISTS": "The NorMuon 2D-weight advantage PERSISTS with budget (edge at the top rung > noise, slope flat).",
        "WIDENS": "The NorMuon advantage WIDENS with budget.",
        "CONVERGES": "The NorMuon advantage CONVERGES toward 0 with budget — an early-training speedup, as the IMU-1 RESULT.md Limitation #3 predicted it might.",
        "FLAT": "Inconclusive: slope within noise and no resolvable edge at the largest budget.",
        "descriptive_only": "Descriptive only — not enough budgets scored yet for a trend verdict.",
    }
    concl = concl_map.get(v, f"trend verdict={v}")
    if headline_capped and v not in ("descriptive_only", "FLAT"):
        concl = (f"DIRECTIONAL ({v} shape on wikitext-2): the top budget ({tagM(top_t)}) rung is "
                 f"n={min(top_seeds)} seeds (<3, §C17), so this is a directional trend, NOT a headline "
                 f"win — add a 3rd 420M seed (and the 840M rung) to earn more. Shape: {concl}")
    if not corpora_agree:
        concl += (f" WARNING: corpora disagree — wikitext-2={trend_by_corpus['wikitext2_val'].get('verdict_code')} "
                  f"vs code_py={trend_by_corpus['code_py'].get('verdict_code')}; treat with extra caution.")

    # Ledger verdict: DIRECTIONAL unless a FULL-strength trend (>=3 budgets, top rung n>=3, both
    # corpora agree, and a resolvable non-FLAT code). With 420M at n=2 this stays directional.
    full_strength = (not headline_capped and corpora_agree
                     and v not in ("descriptive_only", "FLAT", "?"))
    ledger_verdict = v.lower() if full_strength else "directional"

    verdict = {
        "run_id": "2026-07-05_qwen3-0.6b_scaling-persistence", "lifecycle_stage": "scaling",
        "objective": "pretrain-ablation", "suite_version": "text-lm-v2",
        "question": "Does NorMuon's +0.474 BPB win over AdamW (2D weights, fixed N=596M) persist or converge with token budget?",
        "budgets_scored": [tagM(t) for t in scored_budgets],
        "top_budget": (tagM(top_t) if top_t is not None else None),
        "top_budget_seeds": top_seeds,
        "headline_capped_to_directional": headline_capped,
        "cap_reason": (f"top budget {tagM(top_t)} rung is n={min(top_seeds)} (<3 seeds, §C17)"
                       if headline_capped else "top rung has >=3 seeds"),
        "corpora_agree": corpora_agree,
        "gap_by_corpus": gap_by_corpus,
        "trend_by_corpus": trend_by_corpus,
        "trend_wikitext2": trend, "trend_verdict": v,
        "ledger_verdict": ledger_verdict,
        "conclusion": concl,
        "honesty": ("Achieved tok/s only, never %-of-peak MFU (GB10 peak estimated, §C24). A measured "
                    "CONVERGES is a PASSING result. Caps to DIRECTIONAL while the 420M top rung is n=2 "
                    "(<§C17 seeds>=3) and 840M is absent. Noise floor = MAX per-cell CI half-width (keeps "
                    "the n=2 rung, not the median). Gaps use the same unpaired-Welch test as IMU-1's 42M "
                    "headline (byte-comparable); the design is paired-by-seed, so a paired-t on per-seed "
                    "diffs is the stricter test (disclosed caveat). Inherited confound: AdamW/NorMuon LRs "
                    "tuned at 42M, not re-tuned per horizon (RESULT.md Limitation #1)."),
    }
    (LDIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    print(f"\nTREND wikitext-2={v} | code_py={trend_by_corpus['code_py'].get('verdict_code')} "
          f"| corpora_agree={corpora_agree} | headline_capped={headline_capped} -> ledger={ledger_verdict}",
          flush=True)
    print(concl, flush=True)
    print("budgets scored:", verdict["budgets_scored"], "top_seeds:", top_seeds, flush=True)

    # ledger (via ledger.py; never hand-edit)
    try:
        metrics = {"trend_verdict_wikitext": v,
                   "trend_code_py": trend_by_corpus["code_py"].get("verdict_code"),
                   "budgets": verdict["budgets_scored"], "top_budget_seeds": top_seeds,
                   "headline_capped": headline_capped, "corpora_agree": corpora_agree,
                   "conclusion": concl}
        sync_ledger(verdict, ledger_verdict, metrics)
    except Exception as e:                       # OSError etc.; a bad exit is handled inside
        print("ledger write note:", repr(e)[:200], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
