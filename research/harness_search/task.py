"""Bin-packing task for the harness-search experiment (a GPU-free, deterministic
replication harness for arXiv 2603.28052 'Meta-Harness'). A 'harness' is a
pack(items, capacity) -> list[bins] function; the reward is mean fill efficiency
(sum of item sizes / number of bins used; capacity 1.0, so 1.0 == perfect). The
`search` and `heldout` splits use different seeds — we optimize ONLY on search
and report on heldout, exactly like the paper. Pure stdlib, no model, no GPU.
"""
import random
from collections import Counter

CAPACITY = 1.0
N_INSTANCES = 40
N_ITEMS = 40

# Distinct seeds per split (+ per replicate for CIs). search != heldout, always.
SPLIT_SEED = {"search": 1, "heldout": 2}


def make_instances(seed):
    """Deterministic instances. A mix of LARGE (0.5-0.7) and SMALL (0.1-0.4)
    items: first-fit wastes space here, sorting (FFD) helps, and pairing
    large+small or small-clusters helps more — so there is real headroom over a
    naive hand-designed heuristic, and the per-bin waste in the trace reveals it."""
    rng = random.Random(seed)
    insts = []
    for _ in range(N_INSTANCES):
        items = []
        for _ in range(N_ITEMS):
            if rng.random() < 0.4:
                items.append(round(rng.uniform(0.5, 0.7), 3))   # large
            else:
                items.append(round(rng.uniform(0.1, 0.4), 3))   # small
        insts.append(items)
    return insts


def validate(items, bins, capacity=CAPACITY):
    """A packing is valid iff every input item is placed EXACTLY once and no bin
    exceeds capacity. Anything else scores 0 for that instance (no cheating by
    dropping items or overstuffing)."""
    placed = []
    for b in bins:
        if sum(b) > capacity + 1e-9:
            return False
        placed.extend(b)
    return Counter(round(x, 6) for x in placed) == Counter(round(x, 6) for x in items)


def efficiency(items, bins):
    """Fill efficiency = total item size / bins used (capacity 1). Higher is
    better; 1.0 is a perfect pack. 0 if no non-empty bins."""
    nb = len([b for b in bins if b])
    return (sum(items) / nb) if nb else 0.0
