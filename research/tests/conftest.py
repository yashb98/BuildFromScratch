"""pytest wiring for the research-loop infrastructure tests.

These tests exercise the two pieces of code that keep the GB10 box alive and
hold the loop's memory — sentinel.py (the OOM kill-switch) and ledger.py (the
single source of experiment truth). They are stdlib-only, CPU-only, never load
a model, never touch the real ledger, and are safe to run in CI on any Linux.
"""
import json
import sys
from pathlib import Path

import pytest

# sentinel.py lives at the repo root; ledger.py at research/ledger/;
# eval_stats.py / eval_metrics.py / posttrain_losses.py at research/;
# the harness-search framework at research/harness_search/.
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "research", REPO_ROOT / "research" / "ledger",
           REPO_ROOT / "research" / "harness_search"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# A freshly-seeded, empty ledger — the canonical starting shape (§C8).
SEED_LEDGER = {
    "schema_version": 1,
    "techniques": [],
    "runs": [],
    "proposals": [],
    "never_repeat": [],
}


@pytest.fixture
def ledger_path(tmp_path):
    """An isolated, seeded ledger.json in a temp dir. The real ledger is never
    touched: every test passes --ledger <this path>."""
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(SEED_LEDGER, indent=2) + "\n")
    return p
