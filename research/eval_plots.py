"""§C26 — every step ships a FIGURE. The reusable plotter the loop calls so that every
eval / verdict / ablation / training run produces a graph from its on-disk results, not just
a table. "Show what we did, don't just say it."

CPU-only (matplotlib Agg), reads results files, writes PNGs. The dispatcher `figure_for_run`
auto-detects what a run produced (verdict.json / cohort_bpb.json / train CSVs / a downstream
comparison) and renders the right figure — so eval-harness / ablation-runner Phase 6 call ONE
function after scoring. Every figure traces to a file (verifiable-accuracy); nothing invented.
"""
from __future__ import annotations
import json, pathlib, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2a9d8f", "#e9c46a", "#e76f51", "#264653", "#8ab17d", "#287271", "#bdbdbd"]


def bar_with_ci(labels, values, ci_low=None, ci_high=None, *, title="", ylabel="", out,
                zero_line=True, baseline=None):
    """Generic bar chart with optional 95% CI error bars — the verdict workhorse."""
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(labels)), 4.2))
    err = None
    if ci_low is not None and ci_high is not None:
        err = [[v - lo for v, lo in zip(values, ci_low)], [hi - v for v, hi in zip(values, ci_high)]]
    ax.bar(labels, values, yerr=err, capsize=5, color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))])
    if zero_line:
        ax.axhline(0, color="k", lw=.8)
    if baseline is not None:
        ax.axhline(baseline, color="k", ls="--", lw=1, label=f"baseline {baseline:.3f}"); ax.legend()
    ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return str(out)


def grouped_bars(labels, series: dict, *, title="", ylabel="", out):
    """series = {name: [values]} aligned to labels — comparison panels."""
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 4.2))
    n = len(series); w = 0.8 / max(n, 1); x = np.arange(len(labels))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return str(out)


def lines(x, series: dict, *, title="", xlabel="step", ylabel="", out, logy=False):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(x[:len(ys)], ys, label=name, color=PALETTE[i % len(PALETTE)])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return str(out)


def seed_ci_attribution(verdict_path, out, headline_corpus="wikitext2_val"):
    """From a verdict.json (axes -> {corpus -> improvement_bpb + ci95}) render the per-axis
    Δbpb with 95% CI — the standard ablation-verdict figure."""
    d = json.loads(pathlib.Path(verdict_path).read_text())
    axes_ = list(d["axes"]); corpora = list(next(iter(d["axes"].values())))
    fig, axs = plt.subplots(1, len(corpora), figsize=(5 * len(corpora), 4.2), squeeze=False)
    for j, corpus in enumerate(corpora):
        imp = [d["axes"][a][corpus]["improvement_bpb"] for a in axes_]
        lo = [d["axes"][a][corpus]["ci95"][0] for a in axes_]
        hi = [d["axes"][a][corpus]["ci95"][1] for a in axes_]
        sig = [d["axes"][a][corpus]["significant"] for a in axes_]
        err = [[i - l for i, l in zip(imp, lo)], [h - i for i, h in zip(imp, hi)]]
        cols = ["#2a9d8f" if s else "#bdbdbd" for s in sig]
        ax = axs[0][j]; ax.bar(axes_, imp, yerr=err, capsize=5, color=cols)
        ax.axhline(0, color="k", lw=.8); ax.set_title(f"{corpus}"); ax.set_ylabel("Δ BPB vs baseline (+ = better)")
    fig.suptitle(f"{d.get('overall_verdict','verdict')} — per-axis BPB (3 seeds, 95% CI; grey = n.s.)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return str(out)


def figure_for_run(run_dir, out_dir=None):
    """Dispatcher: detect what the run produced and render the right figure(s).
    Returns the list of PNG paths written. Loop callers invoke THIS after scoring."""
    run_dir = pathlib.Path(run_dir)
    out_dir = pathlib.Path(out_dir) if out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    vj = run_dir / "verdict.json"
    if vj.exists():
        try:
            written.append(seed_ci_attribution(vj, out_dir / "verdict_bpb.png"))
        except Exception as e:
            print(f"  [eval_plots] verdict figure skipped: {str(e)[:60]}")
    # training curves from a *_train.csv (step, ce / val_ppl) if present
    for csv in glob.glob(str(run_dir / "*train*.csv"))[:1]:
        try:
            import csv as _c
            rows = list(_c.DictReader(open(csv)))
            if rows and "step" in rows[0]:
                step = [float(r["step"]) for r in rows]
                ycol = next((c for c in ("ce", "loss", "val_ppl") if c in rows[0]), None)
                if ycol:
                    written.append(lines(step, {ycol: [float(r[ycol]) for r in rows]},
                                         title=f"{run_dir.name} — {ycol}", ylabel=ycol,
                                         out=out_dir / "train_curve.png"))
        except Exception as e:
            print(f"  [eval_plots] curve skipped: {str(e)[:60]}")
    # interp_metrics.json -> Loss-Recovered-vs-L0 Pareto (SAE vs PCA/random floors) + probe AUROC-vs-floor bars
    im = run_dir / "interp_metrics.json"
    if im.exists():
        try:
            d = json.loads(im.read_text())
            if d.get("pareto"):  # [{l0, sae, pca, random}] — error vs sparsity, floors overlaid
                xs = [p["l0"] for p in d["pareto"]]
                series = {k: [p[k] for p in d["pareto"]] for k in ("sae", "pca", "random") if k in d["pareto"][0]}
                written.append(lines(xs, series, title=f"{run_dir.name} — recon error vs L0 (SAE vs floors)",
                                     xlabel="L0 (active features)", ylabel="recon error", out=out_dir / "interp_pareto.png"))
            if d.get("probe"):  # {auroc, vs_random_init, vs_length_confound}
                pr = d["probe"]; labs = [k for k in ("auroc", "vs_random_init", "vs_length_confound") if k in pr]
                written.append(bar_with_ci(labs, [pr[k] for k in labs], title=f"{run_dir.name} — probe AUROC vs floors",
                                           ylabel="AUROC", out=out_dir / "interp_probe.png", zero_line=False, baseline=0.5))
        except Exception as e:
            print(f"  [eval_plots] interp figure skipped: {str(e)[:60]}")
    if not written:
        print(f"  [eval_plots] no plottable artifact in {run_dir} (need verdict.json / *train*.csv / interp_metrics.json)")
    return written


def _self_test():
    import tempfile, os
    d = pathlib.Path(tempfile.mkdtemp())
    # synthetic verdict.json -> seed_ci_attribution
    vj = {"overall_verdict": "attributed",
          "axes": {"A": {"wikitext2_val": {"improvement_bpb": 0.1, "ci95": [0.05, 0.15], "significant": True},
                          "code_py": {"improvement_bpb": 0.2, "ci95": [0.1, 0.3], "significant": True}},
                   "B": {"wikitext2_val": {"improvement_bpb": 0.01, "ci95": [-0.02, 0.04], "significant": False},
                          "code_py": {"improvement_bpb": 0.0, "ci95": [-0.05, 0.05], "significant": False}}}}
    (d / "verdict.json").write_text(json.dumps(vj))
    p = seed_ci_attribution(d / "verdict.json", d / "v.png")
    assert os.path.getsize(p) > 1000, "empty verdict figure"
    bar_with_ci(["x", "y"], [0.1, 0.2], [0.05, 0.1], [0.15, 0.3], title="t", ylabel="b", out=d / "b.png")
    grouped_bars(["a", "b"], {"s1": [1, 2], "s2": [2, 1]}, title="g", out=d / "g.png")
    lines([1, 2, 3], {"ce": [3, 2, 1]}, out=d / "l.png")
    got = figure_for_run(d)
    assert got and os.path.getsize(got[0]) > 1000, "dispatcher produced no figure"
    print("eval_plots self-test PASS — verdict CI bars, grouped bars, lines, dispatcher all render")


if __name__ == "__main__":
    _self_test()
