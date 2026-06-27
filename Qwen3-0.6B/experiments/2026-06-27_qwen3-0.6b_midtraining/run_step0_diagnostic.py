"""Step 0 — long-context effective-context diagnostic on the faithful base (mid-training plan §3).
GATES context-extension: does the base USE its trained 4096 window, or collapse before it?
Eval-only (no training); safe_cuda-guarded; residuals only."""
import sys, json, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
MD = ROOT / "Qwen3-0.6B"
MODERN = next(MD.glob("builds/*_reproduce-modernized_*"))
FAITHFUL = next(MD.glob("builds/*_reproduce-faithful_*"))
for p in (str(ROOT), str(ROOT / "research"), str(MD), str(MODERN), str(FAITHFUL)):
    sys.path.insert(0, p)
import safe_cuda; safe_cuda.guard(0.85)
import torch
import model_imu1 as M
from transformers import AutoTokenizer
from eval_longcontext import run_diagnostic

DEV, DT = torch.device("cuda"), torch.bfloat16
HERE = pathlib.Path(__file__).resolve().parent


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    cfg = M.Qwen3Config(use_value_residual=False, use_layernorm_scaling=False, use_head_gating=False)
    model = M.Qwen3ForCausalLM(cfg).to(device=DEV, dtype=DT).eval()
    ck = torch.load(FAITHFUL / "checkpoint_qwen3_baseline2tpp.pt", map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck.get("model", ck).items()}
    model.load_state_dict(sd, strict=True)
    print(f"loaded faithful base ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)

    @torch.no_grad()
    def generate(ids, max_new_tokens=8):
        x = torch.tensor(ids, dtype=torch.long, device=DEV).unsqueeze(0)
        out = []
        for _ in range(max_new_tokens):
            logits = model(input_ids=x)["logits"][0, -1]
            nxt = int(logits.argmax())
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=DEV)], dim=1)
            if x.shape[1] > 8400:    # safety
                break
        return tok.decode(out)

    t0 = time.time()
    # ladder spans + exceeds the trained 4096; base CAN forward to 8192 (RoPE cache → 40960)
    res = run_diagnostic(generate, tok, lengths=(512, 1024, 2048, 4096, 6144, 8192),
                         trained_len=4096, threshold=0.5)
    res["wall_clock_s"] = round(time.time() - t0)
    (HERE / "step0_diagnostic.json").write_text(json.dumps(res, indent=2))
    print("\n=== EFFECTIVE-CONTEXT DIAGNOSTIC (faithful base) ===", flush=True)
    for L, a in res["per_length_accuracy"].items():
        bar = "█" * int(a * 20)
        print(f"  len {L:5d}: passkey acc {a:.2f}  {bar}", flush=True)
    print(f"\n  effective context length (acc>=0.5): {res['effective_context_length']}", flush=True)
    print(f"  accuracy at trained_len 4096: {res['accuracy_at_trained_len']:.2f}", flush=True)
    print(f"  GATE: {res['gate']} -> {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
