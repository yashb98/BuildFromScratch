"""Tests for the system-upgrade ledger schema extension: the new run types
(§C20-§C23), the triton framework (§C-systems), and that additive run keys
(parent_id, is_remote, run_type, metrics.train_flops, confound_check) ride
through --set and validate. Stdlib-only (CI-safe); uses the shared `ledger_path`
fixture + the same exit-code contract as test_ledger.py."""
import json

import ledger


def run(ledger_path, *argv):
    try:
        rc = ledger.main([*argv, "--ledger", str(ledger_path)])
        return 0 if rc is None else rc
    except SystemExit as e:
        return e.code


def reload(ledger_path):
    return json.loads(ledger_path.read_text())


NEW_TYPES = ["scaling-fit", "replication", "serving-bench", "rag-eval",
             "safeguards-round", "sweep", "serve"]


def test_new_run_types_accepted(ledger_path):
    for i, t in enumerate(NEW_TYPES):
        rid = f"2026-06-15_qwen3_{t.replace('-', '')}{i}"
        assert run(ledger_path, "add-run", "--run-id", rid, "--type", t) == 0
    d = reload(ledger_path)
    assert {r["type"] for r in d["runs"]} == set(NEW_TYPES)


def test_triton_framework_accepted(ledger_path):
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_k1",
               "--type", "ablation", "--framework", "triton") == 0
    assert reload(ledger_path)["runs"][0]["framework"] == "triton"


def test_bogus_framework_and_type_rejected_via_set(ledger_path):
    # --set bypasses argparse choices, exercising validate()
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_m_a",
               "--type", "ablation", "--set", "framework=fortran") == 2
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_m_b",
               "--type", "ablation", "--set", "type=nonsense") == 2


def test_additive_keys_stored(ledger_path):
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_arm1",
               "--type", "ablation",
               "--set", "parent_id=2026-06-15_qwen3_sweep0",
               "--set", "is_remote=true",
               "--set", "run_type=arm",
               "--set", 'metrics={"train_flops":1.2e18,"mfu":0.42}',
               "--set", 'confound_check={"n_vars":1,"iso_flop":true}') == 0
    r = reload(ledger_path)["runs"][0]
    assert r["parent_id"] == "2026-06-15_qwen3_sweep0"
    assert r["is_remote"] is True
    assert r["metrics"]["train_flops"] == 1.2e18
    assert r["confound_check"] == {"n_vars": 1, "iso_flop": True}


def test_sweep_parent_then_child(ledger_path):
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_sweep0",
               "--type", "sweep") == 0
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_seed0",
               "--type", "ablation",
               "--set", "parent_id=2026-06-15_qwen3_sweep0") == 0
    d = reload(ledger_path)
    child = [r for r in d["runs"] if r["run_id"].endswith("seed0")][0]
    assert child["parent_id"] == "2026-06-15_qwen3_sweep0"
