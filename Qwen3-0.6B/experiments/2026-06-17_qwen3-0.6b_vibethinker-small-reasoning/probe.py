#!/usr/bin/env python3
"""§C5.3 measured throughput + peak-memory probe at the EXACT launch config
(seq_len 4096, micro_batch 4, grad_accum 32 -> global batch 128). Runs a handful
of real optimizer steps on the base checkpoint with the masked chunked CE, then
reports tokens/sec and torch.cuda.max_memory_allocated(). The plan must fit
< 60% of the 119 GB pool (~71 GB). Writes probe.json for the ledger c5_evidence.
"""
from __future__ import annotations
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import sft_train as S  # noqa: E402

RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)

SEQ_LEN, MICRO_BATCH, GRAD_ACCUM = 4096, 4, 32
N_OPT_STEPS = 3                              # measured steps (after a warmup step)


def main():
    if not torch.cuda.is_available():
        print("CUDA required"); return 1
    S.safe_cuda.guard(0.85)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    stream = S.load_shard_stream()
    metas = S.load_metas()
    ds = S.SFTPackedDataset(stream, metas, SEQ_LEN, limit_samples=400)
    del stream
    ck = torch.load(S.BASE_CKPT, map_location="cpu", weights_only=False)
    from model import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(**{k: v for k, v in ck["config"].items()
                         if k in Qwen3Config.__dataclass_fields__})
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype)
    model.load_state_dict(ck["model"], strict=True)
    model.train()
    from torch.optim import AdamW
    optim = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), eps=1e-8)
    tmodel = torch.compile(model)

    tok_per_step = SEQ_LEN * MICRO_BATCH * GRAD_ACCUM
    order = list(range(len(ds)))
    wi = 0

    def one_opt_step():
        nonlocal wi
        optim.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            # micro_batch windows stacked
            inps, labs = [], []
            for _ in range(MICRO_BATCH):
                inp, lab = ds[order[wi % len(ds)]]; wi += 1
                inps.append(inp); labs.append(lab)
            inp = torch.stack(inps).to(device); lab = torch.stack(labs).to(device)
            logits = tmodel(input_ids=inp)["logits"]
            loss = S.masked_chunked_ce(logits, lab)
            (loss / GRAD_ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        return float(loss.item())

    print(f"[probe] warmup (compile) step @ seq{SEQ_LEN} mb{MICRO_BATCH} ga{GRAD_ACCUM} "
          f"tok/step={tok_per_step:,}...", flush=True)
    l0 = one_opt_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    last = l0
    for i in range(N_OPT_STEPS):
        last = one_opt_step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tps = N_OPT_STEPS * tok_per_step / dt

    out = {
        "seq_len": SEQ_LEN, "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
        "tok_per_step": tok_per_step, "measured_opt_steps": N_OPT_STEPS,
        "elapsed_s": round(dt, 2), "tokens_per_sec": round(tps, 1),
        "peak_mem_gb": round(peak_gb, 2), "pool_gb": 119,
        "plan_cap_gb": round(0.60 * 119, 1), "fits": peak_gb < 0.60 * 119,
        "warmup_loss": round(l0, 4), "last_loss": round(last, 4),
    }
    (RESULTS / "probe.json").write_text(json.dumps(out, indent=2))
    print(f"[probe] {json.dumps(out)}", flush=True)
    print(f"[probe] tok/s={tps:,.0f}  peak={peak_gb:.1f}GB  "
          f"fits<71GB={out['fits']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
