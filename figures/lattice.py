"""
The discrete ternary grid ("lattice") that every scheme in this package indexes.

A ternary plot shows three non-negative quantities (A, B, C) that always add up
to a fixed total (100 %). Pick a resolution n and keep only the points whose
three coordinates are whole multiples of 1/n:

        (a, b, c)  with  a, b, c >= 0 integers  and  a + b + c = n

There are (n + 1)(n + 2) / 2 such points. Each one is drawn as a small hexagon
("cell"), and two cells are *neighbors* when they share a hexagon edge, which
happens exactly when you can move one unit from one coordinate to another:

        (a, b, c) -> (a + 1, b - 1, c), (a - 1, b + 1, c), ... (6 directions)

Interior cells have 6 neighbors, edge cells 4, corner cells 2.

Screen layout used everywhere: A at the top, B bottom-left, C bottom-right.
"""

from __future__ import annotations

import numpy as np

# Six unit moves on the lattice. Each moves one unit between two coordinates.
DIRECTIONS = [(1, -1, 0), (-1, 1, 0), (1, 0, -1), (-1, 0, 1), (0, 1, -1), (0, -1, 1)]

# 2-D screen positions of the three pure corners (apexes).
VERTICES = np.array([[0.5, np.sqrt(3) / 2],   # A: top
                     [0.0, 0.0],              # B: bottom-left
                     [1.0, 0.0]])             # C: bottom-right

CENTROID = np.array([1 / 3, 1 / 3, 1 / 3])


class Lattice:
    """All cells of a resolution-n ternary grid, plus their neighbor lists.

    Attributes
    ----------
    n         : resolution (number of steps along one edge)
    points    : list of integer triplets (a, b, c), in "reading order"
                (rows from the A apex downward, each row left to right)
    cell_of   : dict mapping a triplet to its position in `points`
    weights   : (N, 3) array of fractions (a, b, c) / n, each row sums to 1
    xy        : (N, 2) array of screen positions
    neighbors : list of neighbor-id lists
    edges     : list of (i, j) pairs with i < j for every adjacent pair
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("resolution n must be >= 1")
        self.n = n
        self.points = [(a, n - a - c, c) for a in range(n, -1, -1) for c in range(n - a + 1)]
        self.cell_of = {p: i for i, p in enumerate(self.points)}
        self.weights = np.array(self.points, dtype=float) / n
        self.xy = self.weights @ VERTICES
        self.neighbors = []
        for p in self.points:
            nb = []
            for d in DIRECTIONS:
                q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                if q in self.cell_of:
                    nb.append(self.cell_of[q])
            self.neighbors.append(nb)
        self.edges = [(i, j) for i, nb in enumerate(self.neighbors) for j in nb if i < j]

    def __len__(self) -> int:
        return len(self.points)

    @property
    def cell_spacing(self) -> float:
        """Screen distance between neighboring cell centers."""
        return 1.0 / self.n

    def adjacent(self, i: int, j: int) -> bool:
        return j in self.neighbors[i]

    def nearest_cell(self, w) -> int:
        """Cell closest to a continuous triplet w (fractions summing to 1).

        Uses largest-remainder rounding so the rounded integers still sum to n.
        """
        w = np.clip(np.asarray(w, dtype=float), 0, None)
        w = w / w.sum()
        raw = w * self.n
        base = np.floor(raw).astype(int)
        short = self.n - base.sum()
        for k in np.argsort(-(raw - base))[:short]:
            base[k] += 1
        return self.cell_of[tuple(int(v) for v in base)]

    def cell_at_xy(self, x: float, y: float) -> int:
        """Cell whose center is closest to a screen position (for mouse clicks)."""
        d = np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)
        return int(np.argmin(d))


def to_xy(w) -> np.ndarray:
    """Screen position of a continuous triplet (or an (N, 3) array of them)."""
    return np.asarray(w, dtype=float) @ VERTICES
