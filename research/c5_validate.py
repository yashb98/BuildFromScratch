#!/usr/bin/env python3
"""c5_validate.py — machine-checkable §C5 bounded-auto-run evidence (upgrade-plan #11).

The audit found c5_evidence.json is schema-free and its keys drift per run: the
flagship win has NO c5 file, one full 3-seed cohort has none, and the key naming
varies so nothing could machine-check §C5 compliance. This turns "smoke: pass" from
prose into a CHECKABLE contract: a launch (or the adopted-run protocol for a manual
launch) calls validate_c5() and REFUSES to spawn if a required §C5 item is missing.

It accepts the several historical namings for each item (both the numbered
`c5_0_smoke` style and the flat `smoke` style), because the point is to check that
the EVIDENCE exists in a machine-readable field — not to bikeshed the key name. An
item recorded only in free prose under an unrecognized key reads as MISSING, which is
the correct signal: §C5 evidence belongs in a checkable field.

CLI:  python3 research/c5_validate.py <path/to/c5_evidence.json>
        exit 0 = all required items present; exit 1 = missing items (printed); exit 2 = bad file.
Import: from c5_validate import validate_c5 ; report = validate_c5(evidence_dict)
"""
from __future__ import annotations
import json
import sys

# Each §C5 item -> the set of accepted keys that satisfy it (any one present = satisfied).
# §C5.1 (concurrency) is a launch-instant check, legitimately transient, so it is NOT a
# required stored field; the rest (0,2,3,4,5,6,7) must be recorded before launch.
C5_ITEMS = {
    "c5.0_smoke":     ("smoke", "c5_0_smoke"),
    "c5.2_budget":    ("budget", "c5_2_budget"),
    "c5.3_probe":     ("probe", "c5_3_probe"),
    "c5.4_eta":       ("eta_hours", "c5_4_eta_hours"),
    "c5.5_resume":    ("c5_5_resume", "resume_roundtrip", "resume"),
    "c5.6_sentinel":  ("c5_6_sentinel", "sentinel"),
    "c5.7_guards":    ("c5_7_guards", "guards"),
}
# Identity fields every c5 file must carry regardless of style.
REQUIRED_IDENTITY = ("run_id", "objective", "framework")


def _present(evidence, keys):
    """A key satisfies an item if present AND its value is not None/empty."""
    for k in keys:
        if k in evidence and evidence[k] not in (None, "", {}, []):
            return k
    return None


def validate_c5(evidence: dict) -> dict:
    """Return {ok, missing_items, missing_identity, satisfied} for a c5_evidence dict.
    `ok` is True only when every required §C5 item AND every identity field is present."""
    if not isinstance(evidence, dict):
        return {"ok": False, "missing_identity": list(REQUIRED_IDENTITY),
                "missing_items": list(C5_ITEMS), "satisfied": {},
                "error": "evidence is not a JSON object"}
    satisfied, missing_items = {}, []
    for item, keys in C5_ITEMS.items():
        hit = _present(evidence, keys)
        if hit:
            satisfied[item] = hit
        else:
            missing_items.append(item)
    missing_identity = [k for k in REQUIRED_IDENTITY
                        if evidence.get(k) in (None, "", {}, [])]
    return {"ok": not missing_items and not missing_identity,
            "missing_items": missing_items, "missing_identity": missing_identity,
            "satisfied": satisfied}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: c5_validate.py <c5_evidence.json>", file=sys.stderr)
        return 2
    try:
        evidence = json.loads(open(argv[0]).read())
    except (OSError, json.JSONDecodeError) as e:
        print(f"[c5] cannot read {argv[0]}: {e}", file=sys.stderr)
        return 2
    r = validate_c5(evidence)
    if r["ok"]:
        print(f"[c5] PASS — all §C5 items present ({len(r['satisfied'])}/7): "
              f"{', '.join(sorted(r['satisfied']))}")
        return 0
    if r.get("error"):
        print(f"[c5] FAIL — {r['error']}")
    if r["missing_identity"]:
        print(f"[c5] FAIL — missing identity fields: {', '.join(r['missing_identity'])}")
    if r["missing_items"]:
        print(f"[c5] FAIL — missing §C5 evidence: {', '.join(r['missing_items'])}")
        print("      (record each in a machine-readable field before launch — §C5)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
