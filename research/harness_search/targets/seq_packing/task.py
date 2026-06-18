"""Sequence-packing task for harness-search — the GB10-relevant analog of
bin-packing. A 'harness' is a `pack(doc_lengths, capacity) -> list[sequences]`
function that packs WHOLE document chunks (token counts) into fixed-length
training sequences (`capacity` = seq_len) WITHOUT splitting a chunk across
sequences, to minimise PADDING. Reward = token efficiency = real tokens /
(n_sequences × capacity): 1.0 == zero padding.

Why this matters on a single-GPU box: every padding token is a wasted FLOP. A
packer that hits 0.95 efficiency vs first-fit's 0.85 trains ~12% more real tokens
per GPU-hour — pure throughput, no extra hardware. (Document-boundary packing,
unlike naive concatenate-and-chunk, also avoids the cross-document contamination
the treatise flags — so efficiency is the only free lever left.)

search != heldout seeds, always. Pure stdlib, no model, no GPU.
"""
import random
from collections import Counter

CAPACITY = 2048          # seq_len (tokens)
N_INSTANCES = 40
N_DOCS = 60              # document chunks per instance

SPLIT_SEED = {"search": 11, "heldout": 22}


def make_instances(seed):
    """Deterministic instances of realistic doc-chunk token-lengths. A FineWeb-like
    mix: many SHORT docs (32–512), some MEDIUM (512–1536), and tail chunks from
    long docs (near or at capacity). First-fit wastes padding here; best-fit /
    sort-descending packs tighter — real headroom over a 2-minute heuristic."""
    rng = random.Random(seed)
    insts = []
    for _ in range(N_INSTANCES):
        docs = []
        for _ in range(N_DOCS):
            r = rng.random()
            if r < 0.55:
                docs.append(rng.randint(32, 512))        # short
            elif r < 0.85:
                docs.append(rng.randint(512, 1536))      # medium
            else:
                docs.append(rng.randint(1536, CAPACITY)) # long tail-chunk
        insts.append(docs)
    return insts


def validate(docs, sequences, capacity=CAPACITY):
    """Valid iff every doc-chunk is placed EXACTLY once and no sequence exceeds
    capacity. No cheating by dropping a chunk (loses data) or overflowing
    (corrupts the sequence)."""
    placed = []
    for s in sequences:
        if sum(s) > capacity:
            return False
        placed.extend(s)
    return Counter(placed) == Counter(docs)


def efficiency(docs, sequences, capacity=CAPACITY):
    """Token efficiency = real tokens / (n_sequences × capacity). Higher is
    better; 1.0 == no padding. 0 if no non-empty sequence."""
    nb = len([s for s in sequences if s])
    return (sum(docs) / (nb * capacity)) if nb else 0.0
