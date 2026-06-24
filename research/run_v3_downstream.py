"""Execute the text-lm-v3 downstream battery (the §C25 capability axis) across ALL
our checkpoints — the three 1.19B builds + the Phase-1 de-confound cohort + the Phase-2
arch sub-drill cohort — so we have one comparable downstream table.

For each checkpoint: zero-shot accuracy (acc + acc_norm, Wilson CI via eval_metrics) +
LAMBADA last-token + **per-task BPB-on-gold** (continuous → the only discriminator at
596M/1.19B-tok near-chance scale; computed here from eval_downstream.loglikelihood, so the
tested eval_downstream module is untouched). Writes one result JSON per checkpoint +
a comparison CSV. Sequential on the single GB10 (§C4.5), safe_cuda-guarded.

HONEST SCALE NOTE (§C25.6): the 131M-token PROXY cohort cells are deeply undertrained →
MC accuracy is pure chance (reported `no-signal` when the Wilson CI overlaps the chance
line); BPB-on-gold and LAMBADA carry what little signal exists. The 1.19B builds are the
meaningful downstream comparison.

Usage:  python research/run_v3_downstream.py [--limit-builds 500 --limit-cohort 200 --only builds|cohorts|all --smoke]
"""
from __future__ import annotations
import sys, json, pathlib, time, argparse, importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[1]
MD = ROOT / "Qwen3-0.6B"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
import safe_cuda  # noqa
import torch
import eval_downstream as ED
from eval_downstream import TASKS, load_task, score_multiple_choice, score_lambada
from eval_metrics import accuracy_wilson_ci, bits_per_byte

FAITHFUL = next(MD.glob("builds/*_reproduce-faithful_*"))
MODERN = next(MD.glob("builds/*_reproduce-modernized_*"))
EXPLOR = next(MD.glob("builds/*_reproduce-exploratory_*"))
sys.path.insert(0, str(MODERN))   # model_imu1.py lives here (cohorts + faithful-off + imu1-on)
P1 = MD / "experiments/2026-06-18_qwen3-0.6b_imu1-deconfound-p1"
P2 = MD / "experiments/2026-06-21_qwen3-0.6b_arch-subdrill-p2"
OUT = ROOT / "research/eval/downstream_v3"; OUT.mkdir(parents=True, exist_ok=True)
DEVICE, DTYPE = torch.device("cuda"), torch.bfloat16


def _imu1(vr=False, ln=False, hg=False):
    import model_imu1 as M
    cfg = M.Qwen3Config(use_value_residual=vr, use_layernorm_scaling=ln, use_head_gating=hg)
    return M.Qwen3ForCausalLM(cfg)


def _partialrope(factor):
    spec = importlib.util.spec_from_file_location("model_partialrope", EXPLOR / "model_partialrope.py")
    m = importlib.util.module_from_spec(spec); sys.modules["model_partialrope"] = m; spec.loader.exec_module(m)
    # construct with default config; set the partial fraction if the config exposes it
    cfg = m.Qwen3Config()
    assert hasattr(cfg, "partial_rotary_factor"), "model_partialrope config lacks partial_rotary_factor"
    cfg.partial_rotary_factor = factor   # the REAL attr (default 1.0=full RoPE); 0.25/0.10 for the builds
    return m.Qwen3ForCausalLM(cfg)


def manifest(args):
    M = []
    if args.only in ("all", "builds"):
        M += [
            dict(name="build_faithful", tokens="1.19B", kind="imu1", flags=(0, 0, 0),
                 ckpt=FAITHFUL / "checkpoint_qwen3_baseline2tpp.pt"),
            dict(name="build_imu1", tokens="1.19B", kind="imu1", flags=(1, 1, 1),
                 ckpt=MODERN / "results/checkpoint_imu1_2tpp_step12000.pt"),
            dict(name="build_prope25", tokens="1.19B", kind="prope", factor=0.25,
                 ckpt=EXPLOR / "results/checkpoint_prope25_2tpp_.pt"),
            dict(name="build_prope10", tokens="0.36B", kind="prope", factor=0.10,
                 ckpt=EXPLOR / "results/checkpoint_prope10_2tpp_.pt"),
        ]
    ARCH = {"baseline": (0, 0, 0), "wsd": (0, 0, 0), "zloss": (0, 0, 0), "arch": (1, 1, 1),
            "vr": (1, 0, 0), "ln": (0, 1, 0), "hg": (0, 0, 1)}
    if args.only in ("all", "cohorts"):
        for arm in ("baseline", "wsd", "zloss", "arch"):
            for s in (0, 1, 2):
                M.append(dict(name=f"p1_{arm}_seed{s}", tokens="131M", kind="imu1", flags=ARCH[arm],
                              ckpt=P1 / f"checkpoint_{arm}_seed{s}.pt"))
        for arm in ("vr", "ln", "hg"):
            for s in (0, 1, 2):
                M.append(dict(name=f"p2_{arm}_seed{s}", tokens="131M", kind="imu1", flags=ARCH[arm],
                              ckpt=P2 / f"checkpoint_{arm}_seed{s}.pt"))
    if args.smoke:
        M = [m for m in M if m["name"] in ("build_faithful", "p2_ln_seed0")]
    return M


def load_model(entry):
    if entry["kind"] == "imu1":
        model = _imu1(*[bool(x) for x in entry["flags"]])
    else:
        model = _partialrope(entry["factor"])
    ck = torch.load(entry["ckpt"], map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device=DEVICE, dtype=DTYPE).eval()


@torch.no_grad()
def score_ckpt(entry, tok, corpora, limit):
    model = load_model(entry)
    def logits_fn(ids): return model(input_ids=ids.to(DEVICE))["logits"]
    res = {"name": entry["name"], "tokens": entry["tokens"], "tasks": {}}
    for task, items in corpora.items():
        items = items[:limit]
        if TASKS[task]["kind"] == "lambada":
            r = score_lambada(logits_fn, tok, items)
            acc = accuracy_wilson_ci(r["correct"]) if r["correct"] else (float("nan"),) * 3
            gnll = sum(-ED.loglikelihood(logits_fn, ED._enc(tok, it["text"].strip().rsplit(" ", 1)[0]),
                       ED._enc(tok, " " + it["text"].strip().rsplit(" ", 1)[1]))[0]
                       for it in items if " " in it["text"].strip())
            gby = sum(len((" " + it["text"].strip().rsplit(" ", 1)[1]).encode("utf-8"))
                      for it in items if " " in it["text"].strip())
            res["tasks"][task] = {"metric": "lambada_acc", "acc": acc[0], "ci": [acc[1], acc[2]],
                                  "chance": 0.0, "bpb_gold": bits_per_byte(gnll, gby), "n": r["n"]}
        else:
            r = score_multiple_choice(logits_fn, tok, items)
            a = accuracy_wilson_ci(r["correct"]); an = accuracy_wilson_ci(r["correct_norm"])
            gnll = gby = 0.0
            for it in items:
                ctx = ED._enc(tok, it["context"]) or [tok.eos_token_id or 0]
                g = it["gold"]; ct = it["choices"][g]; ct = ct if ct.startswith(" ") else " " + ct
                gnll += -ED.loglikelihood(logits_fn, ctx, ED._enc(tok, ct))[0]
                gby += len(ct.encode("utf-8"))
            ch = TASKS[task]["chance"]
            res["tasks"][task] = {"metric": "acc_norm", "acc": a[0], "acc_ci": [a[1], a[2]],
                                  "acc_norm": an[0], "acc_norm_ci": [an[1], an[2]], "chance": ch,
                                  "signal": not (an[1] <= ch <= an[2]), "bpb_gold": bits_per_byte(gnll, gby), "n": r["n"]}
    del model; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-builds", type=int, default=500)
    ap.add_argument("--limit-cohort", type=int, default=200)
    ap.add_argument("--only", choices=["all", "builds", "cohorts"], default="all")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    safe_cuda.guard(0.85)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    lim = max(args.limit_builds, args.limit_cohort)
    print("loading benchmarks (once)...", flush=True)
    corpora = {t: load_task(t, limit=lim) for t in TASKS if TASKS[t]["revision"]}
    entries = manifest(args)
    print(f"scoring {len(entries)} checkpoints on {list(corpora)}", flush=True)
    allres = []
    for i, e in enumerate(entries, 1):
        t0 = time.time()
        limit = args.limit_builds if e["tokens"] in ("1.19B", "0.36B") else args.limit_cohort
        try:
            r = score_ckpt(e, tok, corpora, limit)
            (OUT / f"{e['name']}.json").write_text(json.dumps(r, indent=2))
            allres.append(r)
            lam = r["tasks"].get("lambada", {})
            print(f"[{i}/{len(entries)}] {e['name']:22s} lambada_acc={lam.get('acc',float('nan')):.3f} "
                  f"bpb_gold(wiki-style avg)~{sum(t['bpb_gold'] for t in r['tasks'].values())/len(r['tasks']):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as ex:
            print(f"[{i}/{len(entries)}] {e['name']:22s} FAILED: {str(ex)[:120]}", flush=True)
            (OUT / f"{e['name']}.error.txt").write_text(str(ex))
    (OUT / "comparison.json").write_text(json.dumps(allres, indent=2))
    print(f"DONE -> {OUT}/comparison.json ({len(allres)} scored)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
