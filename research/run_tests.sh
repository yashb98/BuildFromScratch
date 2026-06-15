#!/usr/bin/env bash
# Local mirror of the CI gate (.github/workflows/research-loop-ci.yml). Run this
# before committing any change to sentinel.py or research/ledger/ledger.py.
# Stdlib-only, CPU-only, no GPU/model — safe to run anytime, including while a
# trainer is active. Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[run_tests] py_compile sentinel.py + ledger.py"
python3 -m py_compile sentinel.py research/ledger/ledger.py

echo "[run_tests] pytest research/tests"
python3 -m pytest research/tests -q

echo "[run_tests] OK"
