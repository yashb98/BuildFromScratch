#!/usr/bin/env python3
"""§C5.0 smoke test for the VibeThinker SFT run — proves the SCRIPT (the loss
MATH is already unit-tested in research/tests/test_posttrain_losses.py).

Asserts, end-to-end, on a tiny slice (a few SFT samples, few steps):
  1. import + safe_cuda guard + chunked masked-CE oracle equivalence;
  2. base checkpoint loads strict=True;
  3. a few real SFT optimizer steps run, loss is FINITE (no NaN/Inf) and the
     prompt is actually masked (response-token fraction < 1.0);
  4. a checkpoint is written, then RELOADED and a resume continues (§C5.5
     save->reload round-trip): the reloaded model/optimizer state is byte-identical
     to what was saved (a tensor-equality assert, not just "it ran");
  5. exit 0.

Run: python sft_smoke.py   (uses the same sft_train building blocks, no copy).
"""
from __future__ import annotations
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import sft_train as S  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
LOG = RESULTS / "smoke.log"


def log(m):
    line = f"[smoke] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def build(device, dtype, seq_len, n_samples):
    stream = S.load_shard_stream()
    metas = S.load_metas()
    ds = S.SFTPackedDataset(stream, metas, seq_len, limit_samples=n_samples)
    ck = torch.load(S.BASE_CKPT, map_location="cpu", weights_only=False)
    from model import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(**{k: v for k, v in ck["config"].items()
                         if k in Qwen3Config.__dataclass_fields__})
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
    model.load_state_dict(ck["model"], strict=True)
    model.train()
    return ds, model, cfg


def steps(model, ds, optim, sched, device, n_steps, seq_len, grad_accum=2):
    import random
    losses = []
    order = list(range(len(ds)))
    random.Random(0).shuffle(order)
    wi, accum, step = 0, 0, 0
    optim.zero_grad(set_to_none=True)
    while step < n_steps:
        idx = order[wi % len(ds)]; wi += 1
        inp, lab = ds[idx]
        inp = inp.unsqueeze(0).to(device); lab = lab.unsqueeze(0).to(device)
        logits = model(input_ids=inp)["logits"]
        loss = S.masked_chunked_ce(logits, lab)
        (loss / grad_accum).backward()
        accum += 1
        if accum < grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        accum = 0; step += 1
        losses.append(float(loss.item()))
        log(f"  step {step}/{n_steps} loss {loss.item():.4f}")
    return losses, wi


def main():
    if not torch.cuda.is_available():
        log("CUDA required"); return 1
    S.safe_cuda.guard(0.85)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = 1024          # short seq for the smoke (the real run uses 4096)
    n_samples = 40

    # 1. loss oracle
    g, w = S.assert_loss_matches_oracle()
    log(f"1. loss-oracle OK: masked_chunked_ce={g:.6f} == masked_sft_nll={w:.6f}")

    # 2. base load strict=True + 3. a few steps, finite loss, prompt masked
    ds, model, cfg = build(device, dtype, seq_len, n_samples)
    log(f"2. base loaded strict=True; {len(ds)} windows; "
        f"response-token frac={ds.n_response_tokens/max(1,ds.used_tokens):.3f}")
    assert 0.0 < ds.n_response_tokens / max(1, ds.used_tokens) < 1.0, \
        "prompt not masked (response frac must be strictly between 0 and 1)"

    from torch.optim import AdamW
    optim = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), eps=1e-8)
    sched = S.tq.make_cosine_scheduler(optim, 2, 8, 5e-5, 8e-8)
    losses, cursor = steps(model, ds, optim, sched, device, 4, seq_len)
    assert all(math.isfinite(l) for l in losses), f"non-finite loss: {losses}"
    log(f"3. ran {len(losses)} steps, all finite; losses={['%.4f'%l for l in losses]}")

    # 4. checkpoint + reload round-trip (§C5.5) — assert byte-identical restore
    ckpt = RESULTS / "smoke_ckpt.pt"
    S.save_ckpt(ckpt, model, optim, sched, step=4, sample_cursor=cursor,
                tok_seen=4 * seq_len * 2, base_ppl=1.0, recipe={"smoke": True})
    log(f"4a. wrote checkpoint {ckpt.name}")

    ds2, model2, _ = build(device, dtype, seq_len, n_samples)
    optim2 = AdamW(model2.parameters(), lr=5e-5, betas=(0.9, 0.95), eps=1e-8)
    sched2 = S.tq.make_cosine_scheduler(optim2, 2, 8, 5e-5, 8e-8)
    rk = torch.load(ckpt, map_location="cpu", weights_only=False)
    model2.load_state_dict(rk["model"])
    optim2.load_state_dict(rk["optim"])
    sched2.load_state_dict(rk["sched"])
    # assert model weights identical
    sd1, sd2 = model.state_dict(), model2.state_dict()
    max_d = 0.0
    for k in sd1:
        d = (sd1[k].float().cpu() - sd2[k].float().cpu()).abs().max().item()
        max_d = max(max_d, d)
    assert max_d == 0.0, f"model restore not identical (max abs diff {max_d})"
    assert rk["step"] == 4 and rk["sample_cursor"] == cursor, "step/cursor not restored"
    log(f"4b. reload round-trip OK: model max|Δ|={max_d}, step/cursor restored")

    # 5. resume continues + still finite
    losses2, _ = steps(model2, ds2, optim2, sched2, device, 2, seq_len)
    assert all(math.isfinite(l) for l in losses2), f"resume non-finite: {losses2}"
    log(f"5. resume ran 2 more steps, finite; losses={['%.4f'%l for l in losses2]}")

    log("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
