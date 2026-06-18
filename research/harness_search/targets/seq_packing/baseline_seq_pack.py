"""The HAND-DESIGNED baseline sequence-packer: plain first-fit — place each
document chunk in the first sequence it fits, else open a new one. This is what
a naive document-boundary packer does (and roughly what the build's
PackedTextDataset approximates). The Meta-Harness search must beat THIS to
support the 'automated > manual' claim on a real, GB10-relevant component."""


def pack(doc_lengths, capacity):
    sequences = []
    for n in doc_lengths:
        for s in sequences:
            if sum(s) + n <= capacity:
                s.append(n)
                break
        else:
            sequences.append([n])
    return sequences
