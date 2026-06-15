#!/usr/bin/env python3
"""research/taxonomy_gap.py — the §C15.3 combinatorial gap-finder.

Novelty lives in the STRUCTURAL HOLES of the taxonomy: an empty cell adjacent to
full cells (emptiness = novel; adjacency to dense regions = plausible, not crank).
This module scores a candidate cell of the §C12 taxonomy matrix
(architecture × modality × objective × technique × scale-band …) for that
"empty-amid-full" property, so the loop can GENERATE where to look, not just rank
what trended.

Pure stdlib, CPU-only, deterministic — CI-safe and unit-tested.

`occupancy` maps a cell (a tuple of axis values) → count of known works/runs in
it (0 = empty). `axes` is the list of allowed value-lists per axis. Neighbours of
a cell are the cells that differ in EXACTLY ONE axis coordinate.
"""
from __future__ import annotations


def neighbors(cell, axes):
    """All cells differing from `cell` in exactly one axis coordinate."""
    out = []
    for i, axis_values in enumerate(axes):
        for v in axis_values:
            if v == cell[i]:
                continue
            out.append(cell[:i] + (v,) + cell[i + 1:])
    return out


def _emptiness(count):
    # an empty cell is maximally novel (1.0); a populated cell decays toward 0 as
    # 1/(1+count), so a SPARSE cell still scores some novelty.
    return 1.0 / (1.0 + count)


def neighbor_density(cell, occupancy, axes):
    """Fraction of a cell's neighbours that are occupied (count > 0). High density
    = the empty cell sits amid explored work (plausible); 0 = isolated (novel but
    likely irrelevant)."""
    nbrs = neighbors(cell, axes)
    if not nbrs:
        return 0.0
    occ = sum(1 for n in nbrs if occupancy.get(n, 0) > 0)
    return occ / len(nbrs)


def gap_score(cell, occupancy, axes) -> float:
    """Structural-hole score ∈ [0,1] = emptiness(cell) × neighbour_density(cell).
    Maximal for an EMPTY cell whose neighbours are ALL occupied; ~0 for a full
    cell OR for an empty-but-isolated cell."""
    own = _emptiness(occupancy.get(cell, 0))
    return own * neighbor_density(cell, occupancy, axes)


def rank_gaps(occupancy, axes, top=None):
    """Score every cell in the taxonomy grid and return the biggest structural
    holes first — i.e. the most promising unexplored-but-adjacent combinations."""
    grid = [()]
    for axis_values in axes:
        grid = [c + (v,) for c in grid for v in axis_values]
    scored = [{"cell": c, "gap": gap_score(c, occupancy, axes),
               "count": occupancy.get(c, 0)} for c in grid]
    scored.sort(key=lambda r: r["gap"], reverse=True)
    return scored[:top] if top else scored
