#!/usr/bin/env bash
# Start the tool-bridge server on http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"
exec python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
