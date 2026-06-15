"""Tests for research/provenance.py — append-only JSONL spans, fail-open on bad
input, and corrupt-line tolerance on read. Stdlib-only (CI-safe)."""
import provenance as pv


def test_append_and_read_roundtrip(tmp_path):
    p = tmp_path / "trace.jsonl"
    assert pv.append_span(p, "S4", ts="2026-06-15T00:00:00",
                          decision="select", technique_slug="x") is True
    spans = pv.read_spans(p)
    assert len(spans) == 1
    assert spans[0]["stage"] == "S4"
    assert spans[0]["decision"] == "select"
    assert spans[0]["schema"] == pv.SCHEMA


def test_creates_missing_parent_dir(tmp_path):
    p = tmp_path / "deep" / "nested" / "trace.jsonl"
    assert pv.append_span(p, "S0", ts="t") is True
    assert p.exists()


def test_fail_open_on_unserializable(tmp_path):
    p = tmp_path / "trace.jsonl"
    # a set is not JSON-serialisable -> append must return False, NOT raise
    assert pv.append_span(p, "S1", ts="t", bad={1, 2, 3}) is False


def test_read_skips_corrupt_trailing_line(tmp_path):
    p = tmp_path / "trace.jsonl"
    pv.append_span(p, "S2", ts="t", ok=1)
    with open(p, "a") as f:
        f.write("{this is a truncated/corrupt line\n")  # crash-mid-write residue
    spans = pv.read_spans(p)
    assert len(spans) == 1 and spans[0]["stage"] == "S2"


def test_read_missing_file_is_empty(tmp_path):
    assert pv.read_spans(tmp_path / "nope.jsonl") == []
