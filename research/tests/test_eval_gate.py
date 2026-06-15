"""Tests for research/ci/eval_gate.py — the §C23.4 eval-regression deploy gate.
Stdlib-only (CI-safe)."""
import json
import sys
from pathlib import Path

import pytest

# eval_gate lives at research/ci/, which conftest does not add to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import eval_gate as eg


def _res(metric, value, suite="text-lm-v2"):
    return {"suite_version": suite, "metrics": {metric: {"mean": value}}}


def test_improvement_passes():
    v = eg.check(_res("ppl", 23.52), _res("ppl", 28.65), "ppl", floor_abs=0.5)
    assert v["verdict"] == "pass" and v["delta"] == pytest.approx(-5.13)


def test_regression_beyond_floor_fails():
    v = eg.check(_res("ppl", 30.0), _res("ppl", 28.65), "ppl", floor_abs=0.5)
    assert v["verdict"] == "regress"


def test_within_floor_is_noise_and_passes():
    v = eg.check(_res("ppl", 28.9), _res("ppl", 28.65), "ppl", floor_abs=0.5)
    assert v["verdict"] == "pass"


def test_higher_better_metric_direction():
    # accuracy dropped 0.80 -> 0.75, floor 0.01 => regression
    v = eg.check(_res("acc", 0.75), _res("acc", 0.80), "acc", floor_abs=0.01,
                 higher_better=True)
    assert v["verdict"] == "regress"


def test_bare_number_metric_accepted():
    cand = {"suite_version": "v", "metrics": {"ppl": 10.0}}
    base = {"suite_version": "v", "metrics": {"ppl": 10.0}}
    v = eg.check(cand, base, "ppl", floor_abs=0.1)
    assert v["verdict"] == "pass"


def test_missing_metric_raises():
    with pytest.raises(KeyError):
        eg.check(_res("ppl", 1.0), _res("ppl", 1.0), "nope", floor_abs=0.1)


def test_main_exit_codes(tmp_path):
    cand = tmp_path / "cand.json"
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_res("ppl", 28.65)))
    # passing candidate -> exit 0
    cand.write_text(json.dumps(_res("ppl", 23.52)))
    assert eg.main(["--candidate", str(cand), "--baseline", str(base),
                    "--metric", "ppl", "--floor-abs", "0.5"]) == 0
    # regressed candidate -> exit 1
    cand.write_text(json.dumps(_res("ppl", 31.0)))
    assert eg.main(["--candidate", str(cand), "--baseline", str(base),
                    "--metric", "ppl", "--floor-abs", "0.5"]) == 1
