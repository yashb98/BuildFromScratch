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


# --- §C18 single-variable WIN gate (the confounded-headline run-waster) ---------

def test_win_requires_single_variable_iso_flop(ledger_path):
    assert run(ledger_path, "add-technique", "--slug", "norm", "--title", "N") == 0
    # a recorded win with NO confound_check is rejected (confounded / non-comparable)
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_w0",
               "--type", "ablation", "--technique-slug", "norm",
               "--set", "verdict=win") == 2
    # a win on a multi-variable BUNDLE is rejected as a bare win (cf. IMU-1 honesty)
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_w1",
               "--type", "ablation", "--technique-slug", "norm", "--set", "verdict=win",
               "--set", 'confound_check={"n_vars":6,"iso_flop":true}') == 2
    # a single-variable, FLOP-matched win is accepted
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_w2",
               "--type", "ablation", "--technique-slug", "norm", "--set", "verdict=win",
               "--set", 'confound_check={"n_vars":1,"iso_flop":true}') == 0
    # a bundle recorded as INCONCLUSIVE is fine (the honest path stays open)
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_w3",
               "--type", "ablation", "--technique-slug", "norm",
               "--set", "verdict=inconclusive",
               "--set", 'confound_check={"n_vars":6,"iso_flop":true}') == 0


def test_orphan_launch_run_rejected(ledger_path):
    # §C8 referential integrity: a launch-bearing run with an unknown technique -> 3
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_o0",
               "--type", "ablation", "--technique-slug", "ghost") == 3
    # a free-floating eval with an unknown slug still only warns (exit 0)
    assert run(ledger_path, "add-run", "--run-id", "2026-06-15_qwen3_o1",
               "--type", "eval", "--technique-slug", "ghost") == 0


def test_predicted_win_prob_range(ledger_path):
    assert run(ledger_path, "add-technique", "--slug", "p", "--title", "P") == 0
    assert run(ledger_path, "update-technique", "p", "--set", "predicted_win_prob=1.5") == 2
    assert run(ledger_path, "update-technique", "p", "--set", "predicted_win_prob=0.42") == 0
    t = [t for t in reload(ledger_path)["techniques"] if t["slug"] == "p"][0]
    assert t["predicted_win_prob"] == 0.42


def test_daily_backup_snapshot(ledger_path):
    run(ledger_path, "add-technique", "--slug", "b", "--title", "B")
    run(ledger_path, "add-technique", "--slug", "b2", "--title", "B2")
    snaps = list((ledger_path.parent / "backups").glob(f"{ledger_path.stem}-*.json"))
    assert len(snaps) == 1  # ONE per-day snapshot, not one per mutation


def test_ledger_lock_serializes(ledger_path):
    import threading
    fd1 = ledger.acquire_lock(ledger_path)
    assert fd1 is not None
    got = []

    def grab():
        fd2 = ledger.acquire_lock(ledger_path)  # must BLOCK until fd1 releases
        got.append(fd2)
        ledger.release_lock(fd2)

    t = threading.Thread(target=grab)
    t.start()
    t.join(timeout=0.5)
    assert t.is_alive()              # still blocked: the lock is exclusive
    ledger.release_lock(fd1)
    t.join(timeout=2.0)
    assert not t.is_alive() and got and got[0] is not None
