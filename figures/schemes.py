"""
Every way of turning the 2-D ternary grid into a 1-D index that this project
explored, in the order they were tried.

Each scheme returns a SchemeResult:

    index : float array, one entry per cell. NaN means "this cell got no index".
    order : list of cell ids sorted by index (the 1-D sequence).
    is_path : True if the scheme intends consecutive indices to be neighboring
              cells (a "walk" through the grid). False for the vector key,
              which is not a walk at all.
    zone  : int array labeling the scheme's own top-level regions (rows, rings,
            corner clusters, recursion zones, camps). Used only for drawing.

Schemes
-------
raster            Read rows like a book, always left to right.
snake             Read rows alternating direction (boustrophedon).
corner_clusters   Index each corner cluster separately, leave the middle out.
recursive_corner  Fractal split into 4 sub-triangles, corner-to-corner walk.
recursive_center  Same idea, but the walk starts at the center.
ring_spiral       Concentric triangular rings from the center outward.
vector_key        Not a walk: index = camp * 100 + intensity of each cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo
from .lattice import Lattice


@dataclass
class SchemeResult:
    name: str
    index: np.ndarray
    order: list
    is_path: bool
    zone: np.ndarray
    description: str = ""
    params: dict = field(default_factory=dict)

    @property
    def indexed(self) -> np.ndarray:
        return ~np.isnan(self.index)


def _from_order(lat: Lattice, order, name, zone, description, params=None) -> SchemeResult:
    index = np.full(len(lat), np.nan)
    for k, cell in enumerate(order):
        index[cell] = k
    return SchemeResult(name, index, list(order), True, np.asarray(zone), description,
                        params or {})


# ---------------------------------------------------------------------------
# 1. Raster: rows like lines of a book
# ---------------------------------------------------------------------------
def raster(lat: Lattice) -> SchemeResult:
    """Rows from the A apex down, each read left to right.

    Lattice.points is already stored in this order, so the order is 0..N-1.
    Horizontal neighbors get consecutive numbers, but stepping from the end of
    one row to the start of the next jumps across the whole triangle, and
    vertical/diagonal neighbors are a full row-length apart.
    """
    order = list(range(len(lat)))
    zone = np.array([lat.n - p[0] for p in lat.points])          # row number
    return _from_order(lat, order, "Raster rows", zone,
                       "Rows read left to right like lines of a book.")


# ---------------------------------------------------------------------------
# 2. Snake: rows alternating direction
# ---------------------------------------------------------------------------
def snake(lat: Lattice) -> SchemeResult:
    """Rows alternate direction, so every step is to a neighboring cell.

    Fixes the row-end jump of `raster`, but the index still sweeps across all
    three corners every row, so nothing groups by corner.
    """
    order = []
    for row in range(lat.n + 1):
        a = lat.n - row
        cells = [lat.cell_of[(a, lat.n - a - c, c)] for c in range(row + 1)]
        order.extend(cells if row % 2 == 0 else cells[::-1])
    zone = np.array([lat.n - p[0] for p in lat.points])
    return _from_order(lat, order, "Snake rows", zone,
                       "Rows alternate direction; every step touches the last.")


# ---------------------------------------------------------------------------
# 3. Corner clusters: each corner indexed separately, middle left out
# ---------------------------------------------------------------------------
def corner_clusters(lat: Lattice, threshold: float = 0.5) -> SchemeResult:
    """Cells with one coordinate above `threshold` form that corner's cluster.

    Each cluster is read in raster order, A cluster first, then B, then C.
    Cells near the middle belong to no cluster and get no index at all -
    exactly the balanced states one most wants to describe.
    """
    w = lat.weights
    zone = np.full(len(lat), -1)
    order = []
    for k in range(3):
        members = [i for i in range(len(lat)) if zone[i] == -1 and w[i, k] > threshold]
        for i in members:
            zone[i] = k
        order.extend(sorted(members))          # raster order inside the cluster
    return _from_order(lat, order, "Corner clusters", zone,
                       f"Corners (coordinate > {threshold:.0%}) indexed one at a time; "
                       "center left unindexed.", {"threshold": threshold})


# ---------------------------------------------------------------------------
# 4 & 5. Recursive (fractal) curves
# ---------------------------------------------------------------------------
#
# All recursive paths are built in *local* coordinates (x, y, z), x + y + z = s,
# for a triangle with corners X = (s,0,0), Y = (0,s,0), Z = (0,0,s). A child
# triangle is described by its three corners in the parent's coordinates; a
# child point with local coords (u, v, t) on a side-k child maps to
#       (u * child_X + v * child_Y + t * child_Z) / k
# which is exact integer arithmetic because the corners are lattice points.
#
# Splitting a side-s triangle:
#   s = 2m + 1 (odd):  corners of side m at X, Y, Z; inverted center of side m-1
#   s = 2m     (even): one corner of side m, two of side m-1, center of side m-1
# The corner zones touch each other only at single cell pairs near the edge
# midpoints; the center touches all three along whole edges.


def _place(k, cx, cy, cz, sub):
    cx, cy, cz = (np.array(v) for v in (cx, cy, cz))
    for (u, v, t), addr in sub:
        p = cx if k == 0 else (u * cx + v * cy + t * cz) // k
        yield tuple(int(q) for q in p), addr


def corner_path_local(s: int):
    """Walk from corner X to corner Y covering every point, in local coords.

    Visit order of the four children:  X-corner -> Z-corner -> center -> Y-corner.
    Each child walk ends on the cell that touches the first cell of the next
    child, and each child is itself a corner-to-corner walk, so it recurses.
    Returns a list of (point, address); address is the string of child letters
    (X, Y, Z for corners, M for middle) from the top level down.
    """
    if s == 0:
        return [((0, 0, 0), "")]
    if s == 1:
        return [((1, 0, 0), "X"), ((0, 0, 1), "Z"), ((0, 1, 0), "Y")]
    if s % 2:
        m = (s - 1) // 2
        kids = [("X", m,     (s, 0, 0),     (m + 1, 0, m), (m + 1, m, 0)),
                ("Z", m,     (m, 0, m + 1), (0, m, m + 1), (0, 0, s)),
                ("M", m - 1, (1, m, m),     (m, m, 1),     (m, 1, m)),
                ("Y", m,     (m, m + 1, 0), (0, s, 0),     (0, m + 1, m))]
    else:
        m = s // 2
        kids = [("X", m,     (s, 0, 0),         (m, 0, m),         (m, m, 0)),
                ("Z", m - 1, (m - 1, 0, m + 1), (0, m - 1, m + 1), (0, 0, s)),
                ("M", m - 1, (0, m, m),         (m - 1, m, 1),     (m - 1, 1, m)),
                ("Y", m - 1, (m - 1, m + 1, 0), (0, s, 0),         (0, m + 1, m - 1))]
    out = []
    for letter, k, cx, cy, cz in kids:
        out += [(p, letter + a) for p, a in _place(k, cx, cy, cz, corner_path_local(k))]
    return out


def center_path_local(s: int, target=None):
    """Walk from the cell nearest the centroid out to corner X, in local coords.

    Visit order:  center (recursively, starting at ITS center) -> Z-corner ->
    Y-corner -> X-corner. The center child's walk is the same function, so the
    walk always begins at the innermost cell.

    The center child can be placed in two mirror-image orientations (its own
    Y and Z corners swapped); both connect correctly to the Z-corner. We try
    both and keep the one whose first cell lands closest to `target` (the true
    centroid by default). This puts index 0 on the centroid cell whenever the
    construction allows it, and never more than one cell away otherwise.
    """
    if target is None:
        target = np.full(3, s / 3)
    target = np.asarray(target, dtype=float)
    if s == 0:
        return [((0, 0, 0), "")]
    if s == 1:
        options = [[((0, 1, 0), "Y"), ((0, 0, 1), "Z"), ((1, 0, 0), "X")],
                   [((0, 0, 1), "Z"), ((0, 1, 0), "Y"), ((1, 0, 0), "X")]]
        return min(options, key=lambda o: np.sum((np.array(o[0][0]) - target) ** 2))
    if s % 2:
        m = (s - 1) // 2
        centre = (m - 1, (m, 1, m), (1, m, m), (m, m, 1))
        corners = [("Z", m, (m, 0, m + 1), (0, m, m + 1), (0, 0, s)),
                   ("Y", m, (0, m + 1, m), (m, m + 1, 0), (0, s, 0)),
                   ("X", m, (m + 1, m, 0), (s, 0, 0),     (m + 1, 0, m))]
    else:  # even: the big corner is Y so the corners can chain Z -> Y -> X
        m = s // 2
        centre = (m - 1, (m, 0, m), (m, m - 1, 1), (1, m - 1, m))
        corners = [("Z", m - 1, (m - 1, 0, m + 1), (0, m - 1, m + 1), (0, 0, s)),
                   ("Y", m,     (0, m, m),         (m, m, 0),         (0, s, 0)),
                   ("X", m - 1, (m + 1, m - 1, 0), (s, 0, 0),         (m + 1, 0, m - 1))]
    k, cx, cy, cz = centre
    best, best_d = None, np.inf
    for a, b in ((cy, cz), (cz, cy)):
        if k == 0:
            sub = [((0, 0, 0), "")]
        else:
            basis = np.column_stack([cx, a, b]).astype(float)
            sub = center_path_local(k, k * np.linalg.solve(basis, target))
        placed = list(_place(k, cx, a, b, sub))
        d = np.sum((np.array(placed[0][0]) - target) ** 2)
        if d < best_d - 1e-9:
            best, best_d = placed, d
    out = [(p, "M" + a) for p, a in best]
    for letter, kk, ax, ay, az in corners:
        out += [(p, letter + a) for p, a in _place(kk, ax, ay, az, corner_path_local(kk))]
    return out


def _local_to_global(seq, axis_x, axis_y, axis_z):
    out = []
    for (x, y, z), addr in seq:
        g = [0, 0, 0]
        g[axis_x], g[axis_y], g[axis_z] = x, y, z
        out.append((tuple(g), addr))
    return out


def _addresses_to_result(lat, seq, name, description, params):
    order = [lat.cell_of[p] for p, _ in seq]
    addr = {lat.cell_of[p]: a for p, a in seq}
    tops = sorted({a[:1] for a in addr.values()}, key=lambda t: [a[:1] for _, a in seq].index(t))
    zone = np.array([tops.index(addr[i][:1]) for i in range(len(lat))])
    res = _from_order(lat, order, name, zone, description, params)
    res.params["addresses"] = [addr[i] for i in range(len(lat))]
    return res


def recursive_corner(lat: Lattice, start: int = 0, end: int = 1) -> SchemeResult:
    """Fractal corner-to-corner walk from apex `start` to apex `end` (0=A,1=B,2=C)."""
    if start == end:
        raise ValueError("start and end apex must differ")
    other = 3 - start - end
    seq = _local_to_global(corner_path_local(lat.n), start, end, other)
    names = "ABC"
    return _addresses_to_result(
        lat, seq, "Recursive (corner start)",
        f"4-way fractal split, walk from apex {names[start]} to apex {names[end]}.",
        {"start": start, "end": end})


def recursive_center(lat: Lattice, end: int = 0, mirror: bool = False) -> SchemeResult:
    """Fractal walk starting at the center cell and ending at apex `end`.

    After the center zone it sweeps the other two corner zones and finishes in
    the `end` corner. `mirror` reverses which of the other two comes first.
    """
    others = [k for k in range(3) if k != end]
    first, second = (others[1], others[0]) if not mirror else (others[0], others[1])
    # local Z is visited before local Y
    seq = _local_to_global(center_path_local(lat.n), end, second, first)
    return _addresses_to_result(
        lat, seq, "Recursive (center start)",
        "4-way fractal split, walk from the center outward, ending at apex "
        f"{'ABC'[end]}.", {"end": end, "mirror": mirror})


# ---------------------------------------------------------------------------
# 6. Concentric ring spiral
# ---------------------------------------------------------------------------
def ring_spiral(lat: Lattice) -> SchemeResult:
    """Rings of equal min(a, b, c), innermost first, each walked all the way round.

    Ring k is the outline of a smaller triangle (all cells whose smallest
    coordinate equals k). Walking each outline in the same rotational direction,
    and starting each new ring next to where the previous one ended, gives a
    continuous spiral. The index now tracks distance from the center well, but
    every ring passes through all three corners' colors.
    """
    ring = np.min(np.array(lat.points), axis=1)
    cx, cy = 0.5, np.sqrt(3) / 6                           # screen position of the centroid
    angle = np.arctan2(lat.xy[:, 1] - cy, lat.xy[:, 0] - cx)
    order = []
    for k in range(int(ring.max()), -1, -1):
        cells = [i for i in range(len(lat)) if ring[i] == k]
        cells.sort(key=lambda i: angle[i])                 # counter-clockwise cycle
        if order:
            prev = order[-1]
            touching = [c for c in cells if lat.adjacent(prev, c)]
            start = cells.index(touching[0])
            cells = cells[start:] + cells[:start]
        order.extend(cells)
    return _from_order(lat, order, "Ring spiral", ring.max() - ring,
                       "Concentric rings from the center outward, each walked "
                       "counter-clockwise.")


# ---------------------------------------------------------------------------
# 7. Vector key: no walk, index derived from the cell's own coordinates
# ---------------------------------------------------------------------------
def vector_key(lat: Lattice, neutral_radius: float = 0.15, metric: str = "euclidean",
               band: int = 100) -> SchemeResult:
    """index = camp * band + floor(band * intensity)  (see geometry.key).

    Not a walk: two cells with the same key are equally intense in the same
    camp, and consecutive keys need not be neighbors. What it guarantees
    instead is that the leading digit always names the camp and the remainder
    always measures intensity.
    """
    w = lat.weights
    k = geo.key(w, neutral_radius, metric, band).astype(float)
    angle = np.arctan2(lat.xy[:, 1] - np.sqrt(3) / 6, lat.xy[:, 0] - 0.5)
    order = sorted(range(len(lat)), key=lambda i: (k[i], angle[i]))
    return SchemeResult("Vector key", k, order, False, geo.camp(w, neutral_radius, metric),
                        "Index derived from each cell's own camp and intensity; no walk.",
                        {"neutral_radius": neutral_radius, "metric": metric, "band": band})


SCHEMES = {
    "Raster rows": raster,
    "Snake rows": snake,
    "Corner clusters": corner_clusters,
    "Recursive (corner start)": recursive_corner,
    "Recursive (center start)": recursive_center,
    "Ring spiral": ring_spiral,
    "Vector key": vector_key,
}


def build(name: str, lat: Lattice, **kwargs) -> SchemeResult:
    """Build a scheme by display name, passing only the kwargs it accepts."""
    import inspect
    fn = SCHEMES[name]
    accepted = inspect.signature(fn).parameters
    return fn(lat, **{k: v for k, v in kwargs.items() if k in accepted})
