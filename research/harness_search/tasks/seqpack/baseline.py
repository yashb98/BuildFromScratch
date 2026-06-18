"""HAND-DESIGNED baseline sequence packer: first-fit in arrival order — a
reasonable thing a person writes (close to the common 'greedy concatenation'
real teams use). The harness search must beat THIS on held-out padding efficiency.
"""


def pack(docs, seq_len):
    seqs = []
    for d in docs:
        for s in seqs:
            if sum(s) + d <= seq_len:
                s.append(d)
                break
        else:
            seqs.append([d])
    return seqs
