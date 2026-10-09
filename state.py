"""
A moving point in the triangle: the "vector" approach.

The only stored truth is the triplet w = (A, B, C) and how many actions have
been applied. Everything else (camp, intensity, lean, key) is derived on demand
by the functions in geometry.py.

Update rule
-----------
Every action names a *target* point in the triangle and pulls w part of the
way toward it:

    w  <-  w + alpha * (target - w)
    alpha = step / (seed + actions + 1)          (capped at 1)

  * target = an apex   (1,0,0), (0,1,0), (0,0,1)   -> a "pure" action
  * target = any mix,  e.g. (0.5, 0.5, 0)          -> a blended action
  * target = the centroid (1/3, 1/3, 1/3)         -> a balancing action

Why this rule:
  * It can never leave the triangle. w' is a weighted average of two points
    inside the triangle, and the triangle is convex.
  * alpha shrinks as actions accumulate, so early choices move the point a lot
    and late choices move it a little: built-in inertia / reputation.
  * With step = 1 it is *exactly* tallying: start with `seed` imaginary actions
    split evenly, add one count per action toward its target, and divide by the
    total. Proof: counts c, total N = seed + actions, w = c / N. One more action
    toward t gives (c + t) / (N + 1) = w + (t - w) / (N + 1).
  * seed sets early-game volatility: seed = 0 makes the very first action jump
    all the way to its target; a large seed makes the start sluggish.
  * step > 1 makes everything more responsive; step < 1 more stubborn.
  * Balancing actions pull toward the center through the same formula, so
    staying neutral is something you do, not just something you fail to leave.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from . import geometry as geo

APEXES = np.eye(3)
CENTROID = np.full(3, 1 / 3)


@dataclass
class TernaryState:
    w: np.ndarray = field(default_factory=lambda: CENTROID.copy())
    actions: int = 0
    step: float = 1.0
    seed: float = 4.0
    neutral_radius: float = 0.15
    metric: str = "euclidean"
    on_camp_change: list = field(default_factory=list)   # callbacks(old_camp, new_camp)

    # ---- update -----------------------------------------------------------
    @property
    def alpha(self) -> float:
        return float(min(1.0, self.step / (self.seed + self.actions + 1)))

    def apply(self, target, weight: float = 1.0) -> "TernaryState":
        """Pull toward `target` (a triplet). `weight` scales this one action."""
        target = np.asarray(target, dtype=float)
        target = target / target.sum()
        before = self.camp
        a = min(1.0, self.alpha * weight)
        self.w = self.w + a * (target - self.w)
        self.w = np.clip(self.w, 0, None)
        self.w /= self.w.sum()                      # guard against float drift
        self.actions += 1
        after = self.camp
        if after != before:
            for cb in self.on_camp_change:
                cb(before, after)
        return self

    def toward(self, corner: int, weight: float = 1.0) -> "TernaryState":
        return self.apply(APEXES[corner], weight)

    def blend(self, i: int, j: int, weight: float = 1.0) -> "TernaryState":
        return self.apply((APEXES[i] + APEXES[j]) / 2, weight)

    def balance(self, weight: float = 1.0) -> "TernaryState":
        return self.apply(CENTROID, weight)

    def reset(self) -> "TernaryState":
        self.w = CENTROID.copy()
        self.actions = 0
        return self

    # ---- derived (never stored) ------------------------------------------
    @property
    def magnitude(self) -> float:
        return float(geo.magnitude(self.w))

    @property
    def edge_intensity(self) -> float:
        return float(geo.edge_intensity(self.w))

    @property
    def intensity(self) -> float:
        return float(geo.intensity(self.w, self.metric))

    @property
    def camp(self) -> int:
        return int(geo.camp(self.w, self.neutral_radius, self.metric))

    @property
    def lean(self) -> float:
        return float(geo.lean(self.w))

    @property
    def leaning_toward(self) -> int:
        return int(geo.second_camp(self.w))

    @property
    def key(self) -> int:
        return int(geo.key(self.w, self.neutral_radius, self.metric))

    def describe(self) -> str:
        names = "NABC"
        return (f"w=({self.w[0]:.3f}, {self.w[1]:.3f}, {self.w[2]:.3f})  "
                f"camp={names[self.camp]}  intensity={self.intensity:.3f}  "
                f"lean={self.lean:.2f}->{names[self.leaning_toward]}  key={self.key}  "
                f"actions={self.actions}  next alpha={self.alpha:.3f}")
