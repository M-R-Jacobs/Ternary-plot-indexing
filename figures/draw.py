"""Shared matplotlib drawing helpers used by demo.py and scripts/render_figures.py."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon, RegularPolygon
from matplotlib import colormaps

from . import geometry as geo
from .lattice import Lattice, VERTICES
from .schemes import SchemeResult

UNINDEXED = (0.86, 0.86, 0.86)
LETTERS = "ABC"


def cell_colors(lat: Lattice, res: SchemeResult | None, mode: str = "position",
                metric: str = "euclidean") -> np.ndarray:
    """'position': radial ternary color.  'index': viridis by index value."""
    if mode == "index" and res is not None:
        idx = res.index
        ok = ~np.isnan(idx)
        lo, hi = np.nanmin(idx), np.nanmax(idx)
        t = np.zeros(len(lat)) if hi == lo else (np.nan_to_num(idx) - lo) / (hi - lo)
        cols = colormaps["viridis"](t)[:, :3]
    else:
        cols = np.atleast_2d(geo.color(lat.weights, metric))
        ok = np.ones(len(lat), bool) if res is None else ~np.isnan(res.index)
    cols = cols.copy()
    cols[~ok] = UNINDEXED
    return cols


def shared_edge(p, q):
    """The hexagon edge shared by neighboring cell centers p and q."""
    mid, d = (p + q) / 2, q - p
    perp = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    half = np.linalg.norm(d) / (2 * np.sqrt(3))
    return [mid - perp * half, mid + perp * half]


def draw_triangle(ax, lat: Lattice, res: SchemeResult | None = None, *,
                  color_mode="position", metric="euclidean", show_walk=True,
                  show_labels=None, show_zones=True, camp_overlay=False,
                  neutral_radius=0.15, label_corners=True, title=None):
    """Draw cells, zone borders, the walk and optional index labels on `ax`."""
    ax.clear()
    h = lat.cell_spacing
    cols = cell_colors(lat, res, color_mode, metric)
    hexes = [RegularPolygon(q, 6, radius=h / np.sqrt(3), orientation=0) for q in lat.xy]
    hatch_mask = np.zeros(len(lat), bool) if res is None else np.isnan(res.index)
    ax.add_collection(PatchCollection(hexes, facecolor=cols, edgecolor=cols * 0.8 + 0.2,
                                      lw=0.4, zorder=1))
    if hatch_mask.any():
        hatched = [RegularPolygon(lat.xy[i], 6, radius=h / np.sqrt(3))
                   for i in np.flatnonzero(hatch_mask)]
        ax.add_collection(PatchCollection(hatched, facecolor="none", edgecolor="#9a9a9a",
                                          hatch="///", lw=0, zorder=1.5))

    if res is not None and show_zones:
        segs = [shared_edge(lat.xy[i], lat.xy[j]) for i, j in lat.edges
                if res.zone[i] != res.zone[j]]
        ax.add_collection(LineCollection(segs, colors="white", lw=2.6, zorder=2))

    if camp_overlay:
        camps = geo.camp(lat.weights, neutral_radius, metric)
        segs = [shared_edge(lat.xy[i], lat.xy[j]) for i, j in lat.edges
                if camps[i] != camps[j]]
        ax.add_collection(LineCollection(segs, colors="black", lw=1.6, linestyles=(0, (3, 2)),
                                         zorder=2.5))

    if res is not None and show_walk and res.is_path:
        order = [c for c in res.order if not np.isnan(res.index[c])]
        steps, jumps = [], []
        for a, b in zip(order, order[1:]):
            (steps if lat.adjacent(a, b) else jumps).append([lat.xy[a], lat.xy[b]])
        ax.add_collection(LineCollection(steps, colors="#111", lw=1.4, alpha=0.85, zorder=3))
        ax.add_collection(LineCollection(jumps, colors="#d62728", lw=1.2, alpha=0.75,
                                         linestyles=(0, (2, 2)), zorder=3))
        if order:
            ax.scatter(*lat.xy[order[0]], s=60, c="white", ec="black", lw=1.5, zorder=5)
            ax.scatter(*lat.xy[order[-1]], s=60, c="black", marker="s", zorder=5)

    if show_labels is None:
        show_labels = len(lat) <= 120
    if res is not None and show_labels:
        size = max(4.0, min(7.5, 70 / lat.n))
        for i, (x, y) in enumerate(lat.xy):
            if not np.isnan(res.index[i]):
                ax.text(x, y + h * 0.22, f"{int(res.index[i])}", fontsize=size, ha="center",
                        va="center", color="#111", zorder=4)

    ax.add_patch(Polygon(VERTICES, closed=True, fill=False, ec="#555", lw=0.8, zorder=0))
    if label_corners:
        offs = [(0, 0.075), (-0.045, -0.045), (0.045, -0.045)]
        for k in range(3):
            ax.text(*(VERTICES[k] + offs[k]), LETTERS[k], ha="center", va="center",
                    fontsize=14, weight="bold", color=geo.COLORS[k])
    pad = 0.1
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, np.sqrt(3) / 2 + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)


def draw_strip(ax, lat: Lattice, res: SchemeResult, metric="euclidean"):
    """The 1-D index line: every indexed cell's position color, in index order."""
    ax.clear()
    order = [c for c in res.order if not np.isnan(res.index[c])]
    cols = np.atleast_2d(geo.color(lat.weights[order], metric))
    ax.imshow(cols[None, :, :], aspect="auto", extent=(-0.5, len(order) - 0.5, 0, 1),
              interpolation="nearest")
    for k, (a, b) in enumerate(zip(order, order[1:]), start=1):
        if res.zone[a] != res.zone[b]:
            ax.axvline(k - 0.5, color="white", lw=1.5)
        if res.is_path and not lat.adjacent(a, b):
            ax.plot([k - 0.5], [1.12], marker="v", ms=3, color="#d62728", clip_on=False)
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.tick_params(labelsize=7)
