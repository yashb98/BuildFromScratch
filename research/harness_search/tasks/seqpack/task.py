"""Sequence-packing task — the real `from-scratch-build` DATA harness, deterministic
and GPU-free. Pack WHOLE documents (token lengths <= L) into fixed-length training
sequences of capacity L, minimizing PADDING. Reward = real_tokens / (n_seqs * L) =
the exact training-efficiency metric (1.0 = zero padding; higher = less wasted
compute every step). Documents that already fit <= L; over-length docs are assumed
pre-chunked upstream (a fixed step), so the harness only decides ASSIGNMENT — the
bin-packing core our experiment validated, on real token-length distributions.

search and heldout use different seeds; we optimize ONLY on search.
"""
import random
from collections import Counter

SEQ_LEN = 2048          # training context length L (the bin capacity)
N_INSTANCES = 30        # each instance = one shard of documents to pack
N_DOCS = 200            # documents per shard

SPLIT_SEED = {"search": 1, "heldout": 2}


def make_instances(seed):
    """Heavy-tailed doc lengths mimicking web text: mostly short, some medium, a
    few near full-context. First-fit leaves padding here; sorting/best-fit packs
    tighter — real headroom, and the per-sequence padding in the trace reveals it."""
    rng = random.Random(seed)
    insts = []
    for _ in range(N_INSTANCES):
        docs = []
        for _ in range(N_DOCS):
            r = rng.random()
            if r < 0.70:
                length = rng.randint(50, 400)          # short
            elif r < 0.95:
                length = rng.randint(400, 1200)        # medium
            else:
                length = rng.randint(1200, SEQ_LEN)    # long (up to full context)
            docs.append(length)
        insts.append(docs)
    return insts


def validate(docs, seqs, seq_len=SEQ_LEN):
    """Valid iff every document is placed EXACTLY once and no sequence exceeds L
    (integer token counts, strict). No dropping docs, no overflowing context."""
    placed = []
    for s in seqs:
        if sum(s) > seq_len:
            return False
        placed.extend(s)
    return Counter(placed) == Counter(docs)


def efficiency(docs, seqs, seq_len=SEQ_LEN):
    """real_tokens / (n_sequences * L). 1.0 == no padding. This IS the metric a
    pretraining team cares about: (1 - efficiency) is wasted forward/backward FLOPs."""
    n = len([s for s in seqs if s])
    return (sum(docs) / (n * seq_len)) if n else 0.0
