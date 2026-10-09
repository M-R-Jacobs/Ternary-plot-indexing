"""
Scores that make the trade-offs between schemes measurable.

All scores are computed against the *camps* defined in geometry.camp: a neutral
zone in the middle plus three kite-shaped dominance regions, one per corner.

coverage          Fraction of cells that received an index at all.
continuity        For walks: fraction of consecutive indices (k, k+1) that are
                  neighboring cells. 1.0 means the walk never jumps.
mean_tear         Over every pair of neighboring cells: the index difference,
                  as a fraction of the full index range, averaged.
max_tear          The worst such difference. Some tearing is unavoidable for any
                  2-D -> 1-D map (see README "Why it can never be perfect").
zone_runs         Same idea, but against the scheme's OWN top-level zones (rows,
                  rings, recursion quadrants, ...). Shows whether a scheme keeps
                  its own promise even when its zones aren't the camps.
camp_runs         Walk the cells in index order and count how many unbroken
                  stretches of the same camp occur. The ideal equals the number
                  of camps (each camp one block).
same_camp_steps   Fraction of consecutive indices that stay in the same camp
                  ("are 90 and 91 in the same camp?").
intensity_rank    Spearman rank correlation between index and intensity inside
                  each camp, size-weighted. 1.0 means: within a camp, a higher
                  index always means further from the center.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import geometry as geo
from .lattice import Lattice
from .schemes import SchemeResult


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass
class Scores:
    coverage: float
    continuity: float
    mean_tear: float
    max_tear: float
    zone_runs: int
    zone_runs_ideal: int
    camp_runs: int
    camp_runs_ideal: int
    same_camp_steps: float
    intensity_rank: float

    def as_dict(self):
        return asdict(self)


def evaluate(lat: Lattice, res: SchemeResult, neutral_radius: float = 0.15,
             metric: str = "euclidean") -> Scores:
    idx = res.index
    ok = ~np.isnan(idx)
    coverage = ok.mean()

    order = [c for c in res.order if ok[c]]
    pairs = list(zip(order, order[1:]))
    continuity = np.mean([lat.adjacent(a, b) for a, b in pairs]) if pairs else float("nan")

    span = np.nanmax(idx) - np.nanmin(idx) if ok.any() else 1.0
    span = span or 1.0
    tears = [abs(idx[i] - idx[j]) / span for i, j in lat.edges if ok[i] and ok[j]]
    mean_tear = float(np.mean(tears)) if tears else float("nan")
    max_tear = float(np.max(tears)) if tears else float("nan")

    zseq = [res.zone[c] for c in order]
    zone_runs = 1 + sum(1 for a, b in zip(zseq, zseq[1:]) if a != b) if zseq else 0
    zone_ideal = len(set(zseq))

    camps = geo.camp(lat.weights, neutral_radius, metric)
    seq = [camps[c] for c in order]
    runs = 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b) if seq else 0
    ideal = len(set(seq))
    same = np.mean([a == b for a, b in zip(seq, seq[1:])]) if len(seq) > 1 else float("nan")

    inten = geo.intensity(lat.weights, metric)
    num, den = 0.0, 0
    for c in (1, 2, 3):
        members = [i for i in range(len(lat)) if ok[i] and camps[i] == c]
        rho = spearman(idx[members], inten[members])
        if not np.isnan(rho):
            num += rho * len(members)
            den += len(members)
    intensity_rank = num / den if den else float("nan")

    return Scores(float(coverage), float(continuity), mean_tear, max_tear,
                  int(zone_runs), int(zone_ideal), int(runs),
                  int(ideal), float(same), float(intensity_rank))
