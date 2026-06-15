#!/usr/bin/env python3
"""research/provenance.py — append-only provenance trail for the autonomous loop
(§C22). One JSONL span per S0–S9 stage boundary (and per spend/ledger mutation),
under OpenTelemetry GenAI-style keys, so every autonomous decision is replayable
and auditable.

Stdlib-only, CPU-only, no network. Append-only and FAIL-OPEN: a tracing failure
must never take down the loop, so write errors are swallowed (the loop's job is
the experiment, not the trace). The JSONL file is the source of truth; an
optional OTLP/Langfuse export rides on top elsewhere.

Named `provenance` (NOT `trace`) on purpose: `trace` is a stdlib module and this
file sits on sys.path, so the stdlib name must not be shadowed.

A span is one line:
    {"ts": "...", "schema": "gen_ai.loop/0.1", "stage": "S4", "decision": "...",
     "gen_ai.operation.name": "select", "technique_slug": "...", ...}
`ts` is supplied by the caller (the loop derives it at runtime, §C2) — this
module never calls a wall clock so it stays deterministic and testable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "gen_ai.loop/0.1"  # pinned; bump on key changes (§C22)


def append_span(path, stage: str, ts: str, **fields) -> bool:
    """Append one span. Returns True on success, False on any failure
    (fail-open — the caller must not branch the loop on tracing). Creates the
    parent dir if needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        span = {"ts": ts, "schema": SCHEMA, "stage": stage}
        span.update(fields)
        line = json.dumps(span, sort_keys=True)
        with open(p, "a") as f:
            f.write(line + "\n")
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_spans(path) -> list:
    """Read all spans; tolerate (skip) a truncated/corrupt trailing line so a
    crash mid-write never makes the trail unreadable (fail-open on read too)."""
    out = []
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip a partial line rather than failing the whole read
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--path", required=True)
    e.add_argument("--stage", required=True)
    e.add_argument("--ts", required=True, help="ISO timestamp, derived at runtime (§C2)")
    e.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    r = sub.add_parser("read")
    r.add_argument("--path", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "emit":
        fields = {}
        for kv in a.set:
            if "=" not in kv:
                print(json.dumps({"error": f"--set expects key=value: {kv!r}"}),
                      file=sys.stderr)
                return 2
            k, v = kv.split("=", 1)
            try:
                fields[k] = json.loads(v)
            except json.JSONDecodeError:
                fields[k] = v
        ok = append_span(a.path, a.stage, a.ts, **fields)
        return 0 if ok else 1
    if a.cmd == "read":
        print(json.dumps(read_spans(a.path), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
