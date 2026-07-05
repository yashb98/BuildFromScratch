#!/usr/bin/env python3
"""RLVR Phase-2 exploratory cohort (research/rlvr/plan.md §2-3, ACCEPTED 2026-07-02):
ONE Dr.GRPO seed vs its two matched controls, from the SFT policy init.

Arms (--arm):
  grpo    Dr.GRPO: G=8 rollouts/prompt @ T=0.8 cap 256, reward = pinned math-acc-v1
          verifier (1.0 correct / 0.1 parseable-wrong / 0.0 unparseable), advantages
          mean-centered WITHOUT std division (Dr.GRPO fix; constant divisor 1),
          DAPO dynamic sampling (drop zero-variance groups), token-level global-mean
          clipped surrogate (eps 0.2) + beta*k3-KL to the frozen SFT reference
          (beta 3e-3, INFERRED mid of plan band), AdamW lr 1e-6, 1 update/rollout
          batch (on-policy, mu=1), 300 steps x 16 prompts.
  random  Spurious-Rewards gate (REQUIRED §C25): identical pipeline, reward ~
          Bernoulli(0.5) per completion, content-blind. If random ~= grpo on the
          decision metric, the "gain" is prior elicitation -> refuse the win.
  rft     Iso-generation-compute control: the SAME 300x16xG rollout budget from the
          FROZEN policy, keep verifier-correct completions, one masked-SFT pass
          (prompt masked), AdamW lr 1e-5 (INFERRED). Isolates on-policy advantage
          vs best-of-n SFT.

Correctness: per-token logp/ratio/KL tensor math is ORACLE-GATED every 50 steps
against the unit-tested research/posttrain_losses.py scalar forms (grpo_clipped_
objective, kl_penalty_k3) on a random sequence — mismatch >1e-3 aborts the run.
Rewards for grpo/rft go through research/eval_math_acc.py (extract_answer +
is_equiv, extractor math-acc-v1) — the same verifier as the decision metric.

Box safety (§C1/§C5): safe_cuda before torch; logits materialized only per
micro-batch (<=8 x ~400 x V fp32 ~ 1.9 GB); ckpt+rng saved every 25 steps
(--resume); detached launch + sentinel watch handled by run_arms.sh.
"""
from __future__ import annotations
import argparse, json, math, pathlib, random, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MD = ROOT / "Qwen3-0.6B"
MODERN = next(MD.glob("builds/*_reproduce-modernized_*"))
for p in (str(ROOT), str(ROOT / "research"), str(MD), str(MODERN)):
    sys.path.insert(0, p)
import safe_cuda  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import model_imu1 as M  # noqa: E402
sys.path.insert(0, str(ROOT))
from research.eval_math_acc import extract_answer, is_equiv, EXTRACTOR_VERSION  # noqa: E402
from research.posttrain_losses import grpo_clipped_objective, kl_penalty_k3  # noqa: E402

SFT_CKPT = MD / "experiments" / "2026-06-27_qwen3-0.6b_sft-3seed" / "checkpoint_sft_seed0.pt"
PROMPTS = ROOT / "research" / "datasets" / "grpo-math-prompts-v1"
G, PROMPTS_PER_STEP, MAX_NEW, TEMP = 8, 16, 256, 0.8
CLIP_EPS, BETA_KL, LR_GRPO, LR_RFT = 0.2, 3e-3, 1e-6, 1e-5
MICRO_SEQS = 8
ORACLE_EVERY, CKPT_EVERY = 50, 25


def log(msg, fp):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fp.write(line + "\n"); fp.flush()


def load_prompts(seed):
    gsm = [json.loads(l) for l in (PROMPTS / "gsm8k_train_clean.jsonl").open()]
    mth = [json.loads(l) for l in (PROMPTS / "math_l13_train_clean.jsonl").open()]
    rng = random.Random(seed)
    rng.shuffle(gsm); rng.shuffle(mth)
    # pre-registered mix: 2/3 gsm8k, 1/3 MATH L1-3 (mixed difficulty, Learning-from-Less)
    out, gi, mi = [], 0, 0
    while gi < len(gsm) or mi < len(mth):
        for _ in range(2):
            if gi < len(gsm):
                out.append(gsm[gi]); gi += 1
        if mi < len(mth):
            out.append(mth[mi]); mi += 1
    return out


def build_model(device="cuda"):
    cfg = M.Qwen3Config(use_value_residual=False, use_layernorm_scaling=False,
                        use_head_gating=False)
    m = M.Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    sd = torch.load(SFT_CKPT, map_location="cpu", weights_only=False)["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m


@torch.no_grad()
def rollout(model, tok, prompt, g, max_new, gen_rng):
    """g sampled completions for one prompt. Returns (full_ids rows, prompt_len,
    completion token lists, sampling-time per-token logps)."""
    eos = tok.eos_token_id
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0).repeat(g, 1)
    done = torch.zeros(g, dtype=torch.bool, device="cuda")
    logps = [[] for _ in range(g)]
    for _ in range(max_new):
        logits = model(input_ids=x)["logits"][:, -1, :].float()
        lsm = F.log_softmax(logits / TEMP, dim=-1)
        nxt = torch.multinomial(lsm.exp(), 1, generator=gen_rng)
        lp = lsm.gather(1, nxt).squeeze(1)
        for i in range(g):
            if not done[i]:
                logps[i].append(float(lp[i]))
        if eos is not None:
            nxt[done] = eos
            done |= nxt.squeeze(1) == eos
        x = torch.cat([x, nxt], dim=1)
        if bool(done.all()):
            break
    comps = []
    for row in x[:, len(ids):].tolist():
        if eos is not None and eos in row:
            row = row[:row.index(eos) + 1]          # keep eos as trained stop token
        comps.append(row)
    logps = [lp[:len(c)] for lp, c in zip(logps, comps)]
    return ids, comps, logps


def reward_of(text, gold, arm, rew_rng):
    if arm == "random":
        return 1.0 if rew_rng.random() < 0.5 else 0.0
    pred = extract_answer(text)
    if pred is None:
        return 0.0
    return 1.0 if is_equiv(pred, gold) else 0.1


def per_token_logp(model, full_ids, prompt_len, need_grad):
    """Per-token logp of the completion tokens (list of tensors, one per seq),
    micro-batched; logits only ever materialized per micro-batch."""
    outs = []
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        for i in range(0, len(full_ids), MICRO_SEQS):
            batch = full_ids[i:i + MICRO_SEQS]
            L = max(len(b) for b in batch)
            pad = torch.full((len(batch), L), 0, dtype=torch.long, device="cuda")
            for j, b in enumerate(batch):
                pad[j, :len(b)] = torch.tensor(b, device="cuda")
            logits = model(input_ids=pad)["logits"].float()
            lsm = F.log_softmax(logits[:, :-1], dim=-1)
            tgt = pad[:, 1:]
            lp = lsm.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            for j, b in enumerate(batch):
                s, e = prompt_len - 1, len(b) - 1     # completion targets
                outs.append(lp[j, s:e])
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["grpo", "random", "rft"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    safe_cuda.guard(0.85)
    torch.manual_seed(args.seed)
    gen_rng = torch.Generator(device="cuda"); gen_rng.manual_seed(args.seed + 1)
    rew_rng = random.Random(args.seed + 2)
    name = f"{args.arm}_seed{args.seed}"
    logf = open(HERE / f"train_{name}.log", "a")
    health = open(HERE / f"health_{name}.jsonl", "a")

    steps = 1 if args.smoke else args.steps
    n_prompts, g, max_new = (4, 4, 128) if args.smoke else (PROMPTS_PER_STEP, G, MAX_NEW)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    prompts = load_prompts(args.seed)
    log(f"ARM {name}: steps={steps} prompts/step={n_prompts} G={g} cap={max_new} "
        f"beta_kl={BETA_KL} clip={CLIP_EPS} lr={LR_GRPO if args.arm != 'rft' else LR_RFT} "
        f"extractor={EXTRACTOR_VERSION} n_prompt_pool={len(prompts)}", logf)

    policy = build_model()
    ref = None
    if args.arm in ("grpo", "random"):
        ref = build_model().eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    lr = LR_RFT if args.arm == "rft" else LR_GRPO
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    start = 0
    if args.resume and pathlib.Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cuda", weights_only=False)
        policy.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start = ck["step"]
        torch.set_rng_state(ck["rng"].cpu()); gen_rng.set_state(ck["gen_rng"].cpu())
        rew_rng.setstate(ck["rew_rng"])
        log(f"RESUMED @ step {start}", logf)

    rft_buffer = []                                    # (full_ids, prompt_len) of correct comps
    pi = start * n_prompts
    for step in range(start + 1, steps + 1):
        t0 = time.time()
        batch = [prompts[(pi + j) % len(prompts)] for j in range(n_prompts)]
        pi += n_prompts
        groups = []                                     # per kept seq: dict
        n_correct = n_total = kept_groups = 0
        sum_len = 0
        policy.eval()
        for item in batch:
            ids, comps, old_lps = rollout(policy, tok, item["prompt"], g, max_new, gen_rng)
            texts = [tok.decode(c) for c in comps]
            rs = [reward_of(t, item["gold"], args.arm, rew_rng) for t in texts]
            n_correct += sum(1 for r in rs if r == 1.0); n_total += len(rs)
            sum_len += sum(len(c) for c in comps)
            if args.arm == "rft":
                for c, r in zip(comps, rs):
                    if r == 1.0:
                        rft_buffer.append((ids + c, len(ids)))
                continue
            if max(rs) == min(rs):                      # DAPO dynamic sampling
                continue
            kept_groups += 1
            mu = sum(rs) / len(rs)
            for c, r, olp in zip(comps, rs, old_lps):
                if len(c) == 0:
                    continue
                groups.append({"full": ids + c, "plen": len(ids), "adv": r - mu, "old_lp": olp})

        stats = {"step": step, "arm": args.arm, "mean_reward": round(n_correct / max(1, n_total), 4),
                 "kept_groups": kept_groups, "mean_len": round(sum_len / max(1, n_total), 1)}

        if args.arm in ("grpo", "random") and groups:
            policy.train()
            # per-seq exact computation (prompt lengths vary within a step)
            total_tok = sum(len(s["full"]) - s["plen"] for s in groups)
            loss_sum = kl_sum = 0.0
            opt.zero_grad(set_to_none=True)
            for i in range(0, len(groups), MICRO_SEQS):
                sub = groups[i:i + MICRO_SEQS]
                mb_loss = 0.0
                for s in sub:
                    lp = per_token_logp(policy, [s["full"]], s["plen"], need_grad=True)[0]
                    n_c = lp.shape[0]
                    old = torch.tensor(s["old_lp"][:n_c], device="cuda", dtype=torch.float32)
                    ratio = torch.exp(lp - old)
                    a = float(s["adv"])
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * a
                    surr = torch.minimum(unclipped, clipped)
                    with torch.no_grad():
                        rlp = per_token_logp(ref, [s["full"]], s["plen"], need_grad=False)[0]
                    d = rlp - lp
                    kl = (torch.exp(d) - d - 1.0)
                    seq_loss = (-(surr) + BETA_KL * kl).sum() / total_tok
                    mb_loss = mb_loss + seq_loss
                    kl_sum += float(kl.detach().mean()) * n_c
                    if step % ORACLE_EVERY == 1 and s is sub[0]:
                        o_s = grpo_clipped_objective([float(r) for r in ratio.detach().tolist()],
                                                     [a] * n_c, CLIP_EPS)
                        t_s = float(surr.detach().mean())
                        o_k = kl_penalty_k3([float(x) for x in lp.detach().tolist()],
                                            [float(x) for x in rlp.tolist()])
                        t_k = float(kl.detach().mean())
                        assert abs(o_s - t_s) < 1e-3 and abs(o_k - t_k) < 1e-3, \
                            f"ORACLE MISMATCH surr {o_s} vs {t_s} | kl {o_k} vs {t_k}"
                mb_loss.backward()
                loss_sum += float(mb_loss)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            stats.update({"loss": round(loss_sum, 6), "kl": round(kl_sum / max(1, total_tok), 6)})

        stats["sec"] = round(time.time() - t0, 1)
        health.write(json.dumps(stats) + "\n"); health.flush()
        log(f"step {step}/{steps} reward={stats['mean_reward']} kept={stats.get('kept_groups')} "
            f"len={stats['mean_len']} kl={stats.get('kl', '-')} ({stats['sec']}s)", logf)

        if step % CKPT_EVERY == 0 or step == steps:
            torch.save({"model": policy.state_dict(), "opt": opt.state_dict(), "step": step,
                        "rng": torch.get_rng_state(), "gen_rng": gen_rng.get_state(),
                        "rew_rng": rew_rng.getstate()},
                       HERE / f"checkpoint_{name}.pt")

    if args.arm == "rft":
        log(f"RFT: {len(rft_buffer)} verifier-correct completions collected; masked-SFT pass", logf)
        policy.train()
        rng = random.Random(args.seed + 3); rng.shuffle(rft_buffer)
        opt.zero_grad(set_to_none=True)
        for i, (full, plen) in enumerate(rft_buffer, 1):
            lp = per_token_logp(policy, [full], plen, need_grad=True)[0]
            (-(lp.sum()) / max(1, lp.shape[0])).backward()
            if i % MICRO_SEQS == 0 or i == len(rft_buffer):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            if i % 200 == 0:
                log(f"RFT sft-pass {i}/{len(rft_buffer)}", logf)
        torch.save({"model": policy.state_dict(), "opt": opt.state_dict(), "step": steps,
                    "rng": torch.get_rng_state(), "gen_rng": gen_rng.get_state(),
                    "rew_rng": rew_rng.getstate()},
                   HERE / f"checkpoint_{name}.pt")

    log("DONE", logf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
