"""§C26 eval_plots tests — CPU, verify figures render non-empty (no model, no network)."""
import sys, pathlib, tempfile, json, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import eval_plots as P

def test_self_test():
    P._self_test()

def test_bar_and_grouped_and_lines(tmp_path):
    assert os.path.getsize(P.bar_with_ci(["a","b"],[.1,.2],[.05,.1],[.15,.3],title="t",ylabel="y",out=tmp_path/"a.png"))>800
    assert os.path.getsize(P.grouped_bars(["a","b"],{"s":[1,2]},title="t",out=tmp_path/"g.png"))>800
    assert os.path.getsize(P.lines([1,2,3],{"y":[3,2,1]},out=tmp_path/"l.png"))>800

def test_dispatcher_needs_an_artifact(tmp_path):
    assert P.figure_for_run(tmp_path) == []   # empty dir -> no figure, no crash
    (tmp_path/"verdict.json").write_text(json.dumps({"overall_verdict":"x","axes":{"A":{"c":{"improvement_bpb":.1,"ci95":[.05,.15],"significant":True}}}}))
    assert len(P.figure_for_run(tmp_path))>=1
