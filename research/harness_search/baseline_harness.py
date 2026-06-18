"""The HAND-DESIGNED baseline harness: plain first-fit (the control arm — what a
person writes in 2 minutes). The Meta-Harness search must beat THIS to support
the paper's 'automated > manual' claim."""


def pack(items, capacity):
    bins = []
    for it in items:
        for b in bins:
            if sum(b) + it <= capacity + 1e-9:
                b.append(it)
                break
        else:
            bins.append([it])
    return bins
