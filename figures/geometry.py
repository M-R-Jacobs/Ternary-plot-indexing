"""
Quantities derived from a single ternary triplet w = (A, B, C), sum 1.

Everything here is a pure function of w. Nothing depends on any path or
ordering, which is the whole point of the "vector" approach (see README).

    magnitude(w)       Euclidean distance from the centroid, scaled so the
                       three apexes are exactly 1. Edge midpoints are 0.5.
    edge_intensity(w)  1 - 3 * min(w). Distance from the centroid scaled so
                       the *entire* outer edge is 1. Edge midpoints are 1.
    camp(w)            Which corner region ("kite") w lies in: the largest
                       coordinate, or 0 = neutral if w is close to the center.
    lean(w)            0 when w points straight at its camp's apex, 1 when it
                       sits on the border with the neighboring camp.
    key(w)             A single sortable integer: camp * 100 + intensity.
    color(w)           Radial color map: grey center, pure hue at apexes.
"""

from __future__ import annotations

import numpy as np

from .lattice import CENTROID

MAGNITUDE_METRICS = ("euclidean", "edge")

# Default palette: A red, B blue, C green; neutral grey in the middle.
COLORS = np.array([[0.80, 0.16, 0.14],
                   [0.13, 0.33, 0.78],
                   [0.16, 0.62, 0.26]])
NEUTRAL = np.array([0.60, 0.60, 0.60])


def _as_w(w) -> np.ndarray:
    return np.asarray(w, dtype=float)


def magnitude(w) -> np.ndarray | float:
    """Euclidean distance from the centroid, normalized so apexes equal 1.

    Derivation: the squared screen distance between w and the centroid is
    proportional to sum((w_i - 1/3)^2). At an apex, e.g. (1, 0, 0), that sum is
    (2/3)^2 + (1/3)^2 + (1/3)^2 = 2/3. Dividing by 2/3 (multiplying by 1.5) and
    taking the square root gives 1 at every apex.

        centroid  (1/3, 1/3, 1/3) -> 0
        midpoint  (1/2, 1/2, 0)   -> 0.5
        apex      (1, 0, 0)       -> 1
    """
    w = _as_w(w)
    return np.sqrt(1.5 * np.sum((w - 1 / 3) ** 2, axis=-1))


def edge_intensity(w) -> np.ndarray | float:
    """Distance from centroid scaled so every point on the outer edge equals 1.

    Along the ray from the centroid through w, the first coordinate to hit 0 is
    the smallest one; 1 - 3 * min(w) is the fraction of that ray covered.

        centroid -> 0,  midpoint -> 1,  apex -> 1
    """
    w = _as_w(w)
    return 1 - 3 * np.min(w, axis=-1)


def intensity(w, metric: str = "euclidean"):
    if metric == "euclidean":
        return magnitude(w)
    if metric == "edge":
        return edge_intensity(w)
    raise ValueError(f"unknown magnitude metric {metric!r}; use one of {MAGNITUDE_METRICS}")


def camp(w, neutral_radius: float = 0.15, metric: str = "euclidean") -> np.ndarray | int:
    """0 = neutral (intensity below neutral_radius), else 1/2/3 for A/B/C.

    The three non-neutral camps are the "dominance regions": the set of points
    where that coordinate is the largest. Their borders are the three straight
    lines from the centroid to the edge midpoints, which cut the triangle into
    three kite-shaped quadrilaterals. Ties on a border go to the earlier letter.
    """
    w = _as_w(w)
    c = np.argmax(w, axis=-1) + 1
    return np.where(intensity(w, metric) < neutral_radius, 0, c)


def lean(w) -> np.ndarray | float:
    """How far w leans from its own apex line toward the neighboring camp.

    (second largest - smallest) / (largest - smallest)
      0 -> on the straight line from the centroid to the camp's apex
      1 -> exactly on the border with the next camp (two leaders tied)
    Undefined at the exact centroid; returns 0 there.
    """
    w = np.sort(_as_w(w), axis=-1)
    lo, mid, hi = w[..., 0], w[..., 1], w[..., 2]
    span = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(span > 1e-12, (mid - lo) / np.where(span > 1e-12, span, 1), 0.0)
    return out


def second_camp(w) -> np.ndarray | int:
    """The camp w leans toward (second-largest coordinate), 1/2/3."""
    return np.argsort(_as_w(w), axis=-1)[..., 1] + 1


def key(w, neutral_radius: float = 0.15, metric: str = "euclidean", band: int = 100):
    """camp * band + floor(band * intensity), capped inside the band.

    Same camp -> same leading digit(s). Within a camp the key rises
    monotonically with intensity. Example with band=100:
        neutral cell, intensity 0.07 -> 7
        A camp, intensity 0.91      -> 191
    """
    w = _as_w(w)
    m = intensity(w, metric)
    # small epsilon: exact band edges must not floor down due to rounding (C# matches)
    level = np.minimum(np.floor(band * m + 1e-4), band - 1).astype(int)
    return camp(w, neutral_radius, metric) * band + level


def color(w, metric: str = "euclidean") -> np.ndarray:
    """Radial color map.

    Find where the ray from the centroid through w meets the outer edge; the
    hue there is a straight blend of the two corners on that edge (pure at an
    apex, 50/50 at a midpoint). Then fade from NEUTRAL grey at the center to
    that hue in proportion to the chosen intensity metric.
    """
    w = np.atleast_2d(_as_w(w))
    r = np.maximum(edge_intensity(w), 1e-12)[:, None]
    edge_point = 1 / 3 + (w - 1 / 3) / r
    hue = edge_point @ COLORS
    t = intensity(w, metric)[:, None]
    out = NEUTRAL + t * (hue - NEUTRAL)
    return np.clip(out, 0, 1).squeeze()


def dominance_region(w) -> np.ndarray | int:
    """0/1/2 index of the largest coordinate (no neutral zone)."""
    return np.argmax(_as_w(w), axis=-1)


__all__ = ["magnitude", "edge_intensity", "intensity", "camp", "lean", "second_camp",
           "key", "color", "dominance_region", "COLORS", "NEUTRAL", "CENTROID",
           "MAGNITUDE_METRICS"]
