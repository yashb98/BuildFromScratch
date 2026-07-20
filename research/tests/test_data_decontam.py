"""Tests for research/data_decontam.py — document-disjoint splitting + n-gram
decontamination (the fix for the leak-suspect sequential val split). Pure,
deterministic, stdlib-only."""
import data_decontam as dd


def test_split_is_deterministic_and_order_independent():
    docs = [f"document number {i} with some words" for i in range(200)]
    a = [d for d in docs if dd.is_val_doc(d, seed=0, val_fraction=0.2)]
    b = [d for d in reversed(docs) if dd.is_val_doc(d, seed=0, val_fraction=0.2)]
    # same set regardless of order; a given doc always lands in the same bucket
    assert set(a) == set(b)
    # roughly val_fraction of docs (loose bound for n=200)
    assert 0.1 < len(a) / len(docs) < 0.3


def test_seed_changes_assignment():
    docs = [f"doc {i}" for i in range(300)]
    s0 = {d for d in docs if dd.is_val_doc(d, seed=0, val_fraction=0.3)}
    s1 = {d for d in docs if dd.is_val_doc(d, seed=1, val_fraction=0.3)}
    assert s0 != s1


def test_decontaminate_drops_overlapping_val_doc():
    train = ["the quick brown fox jumps over the lazy dog again and again today"]
    clean = "an entirely unrelated sentence about marine biology and coral reefs ok"
    leaked = train[0]  # identical to a train doc -> must be dropped
    kept, dropped, overlap = dd.decontaminate_val([clean, leaked], train,
                                                  n=5, threshold=0.5)
    assert leaked not in kept and clean in kept
    assert dropped == [1] and overlap[1] > overlap[0]


def test_split_and_decontaminate_report_shape():
    docs = [f"unique sentence alpha {i} bravo charlie delta echo foxtrot" for i in range(120)]
    train, val, report = dd.split_and_decontaminate(docs, seed=7, val_fraction=0.25,
                                                     n=13, threshold=0.8)
    assert set(train).isdisjoint(set(val))                 # document-disjoint
    assert report["split_seed"] == 7 and report["ngram_n"] == 13
    assert report["docs_dropped"] >= 0
    assert report["n_val_docs_kept"] == report["n_val_docs_raw"] - report["docs_dropped"]
    # fixed-split lineage: base seed == effective seed, randomized flag off
    assert report["randomized_split"] is False
    assert report["base_split_seed"] == 7 and report["split_seed"] == 7


# --- randomized per-cell split (the RS full-variance option) -----------------

def test_cell_split_seed_fixed_is_base_seed_for_every_cell():
    # fixed: every cell seed maps back to the SAME base split seed
    for cs in range(10):
        assert dd.cell_split_seed(42, cs, randomized=False) == 42


def test_cell_split_seed_randomized_differs_per_cell_and_is_deterministic():
    seeds = {dd.cell_split_seed(42, cs, randomized=True) for cs in range(8)}
    # a different split seed per cell (no collisions across 8 cells)
    assert len(seeds) == 8
    # all non-negative and distinct from the trivial base seed
    assert all(s >= 0 and s != 42 for s in seeds)
    # deterministic: same (base, cell) -> same value
    assert dd.cell_split_seed(42, 3, randomized=True) == dd.cell_split_seed(42, 3, randomized=True)
    # base seed also mixes in: same cell seed, different base -> different split
    assert dd.cell_split_seed(42, 3, randomized=True) != dd.cell_split_seed(43, 3, randomized=True)


def test_randomized_split_yields_different_partitions_across_cells():
    docs = [f"corpus document {i} with assorted lexical content here" for i in range(300)]
    _, val0, r0 = dd.split_and_decontaminate(docs, seed=5, val_fraction=0.2,
                                             n=13, threshold=0.8,
                                             cell_seed=0, randomized=True)
    _, val1, r1 = dd.split_and_decontaminate(docs, seed=5, val_fraction=0.2,
                                             n=13, threshold=0.8,
                                             cell_seed=1, randomized=True)
    # different cell seeds -> different doc-disjoint partitions (corpus resampled)
    assert set(val0) != set(val1)
    assert r0["split_seed"] != r1["split_seed"]
    assert r0["randomized_split"] is True and r0["base_split_seed"] == 5
    assert r0["cell_seed"] == 0 and r1["cell_seed"] == 1
    assert "randomized" in r0["method"]


def test_fixed_split_identical_partition_across_cells():
    docs = [f"corpus document {i} with assorted lexical content here" for i in range(300)]
    _, val0, _ = dd.split_and_decontaminate(docs, seed=5, val_fraction=0.2,
                                            cell_seed=0, randomized=False)
    _, val1, _ = dd.split_and_decontaminate(docs, seed=5, val_fraction=0.2,
                                            cell_seed=9, randomized=False)
    # fixed split: membership is held constant regardless of cell seed
    assert set(val0) == set(val1)


def test_randomized_requires_cell_seed():
    import pytest
    with pytest.raises(ValueError):
        dd.split_and_decontaminate(["a b c", "d e f"], seed=1, val_fraction=0.3,
                                   randomized=True)  # cell_seed missing
