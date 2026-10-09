"""
Interactive explorer for ternary-plot indexing schemes.

    python demo.py            (needs a desktop matplotlib backend, e.g. TkAgg/QtAgg)

LEFT   choose a scheme, toggle overlays, and change the grid / camp parameters.
MIDDLE the triangle (every cell drawn as a hexagon) and, below it, the 1-D index
       line: each cell's color laid out in index order.
RIGHT  live scores for the current scheme, details of the cell you clicked, and
       an "agent": a point that moves around the triangle when you press the
       action buttons. Its history plot shows what the current scheme's index
       does while the point moves smoothly - the clearest way to see why a
       walk-based index misbehaves and a vector-derived key does not.

Mouse:     click a cell to inspect it and its neighbors.
Keyboard:  a / b / c  push the agent toward A / B / C
           n          balancing action (pull toward the center)
           space      ten random actions
           r          reset the agent
"""

from __future__ import annotations

import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from ternary_index import Lattice, TernaryState, build, evaluate, geometry as geo
from ternary_index.draw import draw_strip, draw_triangle
from ternary_index.lattice import to_xy
from ternary_index.schemes import SCHEMES

NAMES = "NABC"   # camp 0 is neutral


class Demo:
    def __init__(self, n: int = 12):
        self.fig = plt.figure(figsize=(16, 9.5), facecolor="white")
        try:
            self.fig.canvas.manager.set_window_title("Ternary indexing explorer")
        except AttributeError:
            pass

        # ---- layout -------------------------------------------------------
        self.ax_tri = self.fig.add_axes([0.20, 0.17, 0.46, 0.78])
        self.ax_strip = self.fig.add_axes([0.22, 0.075, 0.42, 0.04])
        self.ax_info = self.fig.add_axes([0.675, 0.40, 0.32, 0.57])
        self.ax_info.axis("off")
        self.ax_hist = self.fig.add_axes([0.71, 0.17, 0.27, 0.19])

        self.fig.text(0.01, 0.965, "Scheme", weight="bold")
        self.radio = RadioButtons(self.fig.add_axes([0.005, 0.66, 0.17, 0.30]),
                                  list(SCHEMES), active=4)
        self.fig.text(0.01, 0.635, "Overlays", weight="bold")
        self.checks = CheckButtons(self.fig.add_axes([0.005, 0.47, 0.17, 0.16]),
                                   ["Show walk", "Index numbers", "Color by index",
                                    "Camp borders"], [True, True, False, False])
        self.fig.text(0.01, 0.445, "Intensity measure", weight="bold")
        self.metric_radio = RadioButtons(self.fig.add_axes([0.005, 0.36, 0.17, 0.08]),
                                         ["euclidean (apex = 1)", "edge (whole rim = 1)"])

        def slider(y, label, lo, hi, val, step=None):
            ax = self.fig.add_axes([0.075, y, 0.10, 0.022])
            return Slider(ax, label, lo, hi, valinit=val, valstep=step)

        self.fig.text(0.01, 0.325, "Parameters", weight="bold")
        self.s_n = slider(0.29, "grid n", 2, 36, n, 1)
        self.s_thr = slider(0.25, "corner %", 0.34, 0.9, 0.5)
        self.s_neu = slider(0.21, "neutral r", 0.0, 0.5, 0.15)
        self.fig.text(0.01, 0.17, "Agent dynamics", weight="bold")
        self.s_step = slider(0.135, "step", 0.1, 4.0, 1.0)
        self.s_seed = slider(0.095, "seed T0", 0.0, 30.0, 4.0)

        bx = [0.675 + i * 0.081 for i in range(4)]
        def button(x, y, label, cb):
            b = Button(self.fig.add_axes([x, y, 0.076, 0.035]), label)
            b.on_clicked(cb)
            return b
        self.buttons = [
            button(bx[0], 0.075, "→ A", lambda e: self.act("A")),
            button(bx[1], 0.075, "→ B", lambda e: self.act("B")),
            button(bx[2], 0.075, "→ C", lambda e: self.act("C")),
            button(bx[3], 0.075, "Balance", lambda e: self.act("N")),
            button(bx[0], 0.03, "→ A+B", lambda e: self.act("AB")),
            button(bx[1], 0.03, "→ B+C", lambda e: self.act("BC")),
            button(bx[2], 0.03, "Random ×10", lambda e: self.random_walk(10)),
            button(bx[3], 0.03, "Reset", lambda e: self.reset_agent()),
        ]

        # ---- state --------------------------------------------------------
        self.rng = np.random.default_rng(7)
        self.agent = TernaryState()
        self.trail = [self.agent.w.copy()]
        self.events = []
        self.agent.on_camp_change.append(
            lambda a, b: self.events.append(f"camp change {NAMES[a]} → {NAMES[b]} "
                                            f"at action {self.agent.actions}"))
        self.selected = None
        self.lat = None
        self.res = None

        for w in (self.radio, self.metric_radio):
            w.on_clicked(lambda _: self.rebuild())
        self.checks.on_clicked(lambda _: self.redraw())
        for s in (self.s_n, self.s_thr, self.s_neu):
            s.on_changed(lambda _: self.rebuild())
        for s in (self.s_step, self.s_seed):
            s.on_changed(lambda _: self.sync_agent())
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.rebuild()

    # ---- parameters ---------------------------------------------------------
    @property
    def metric(self):
        return "euclidean" if self.metric_radio.value_selected.startswith("eu") else "edge"

    @property
    def neutral_radius(self):
        return float(self.s_neu.val)

    def sync_agent(self):
        self.agent.step = float(self.s_step.val)
        self.agent.seed = float(self.s_seed.val)
        self.agent.metric = self.metric
        self.agent.neutral_radius = self.neutral_radius
        self.draw_info()
        self.fig.canvas.draw_idle()

    def rebuild(self):
        n = int(self.s_n.val)
        if self.lat is None or self.lat.n != n:
            self.lat = Lattice(n)
            self.selected = None
        self.res = build(self.radio.value_selected, self.lat, threshold=float(self.s_thr.val),
                         neutral_radius=self.neutral_radius, metric=self.metric)
        self.scores = evaluate(self.lat, self.res, self.neutral_radius, self.metric)
        self.sync_agent()
        self.redraw()

    # ---- drawing ------------------------------------------------------------
    def redraw(self):
        walk, labels, by_index, camps = self.checks.get_status()
        draw_triangle(self.ax_tri, self.lat, self.res, metric=self.metric,
                      color_mode="index" if by_index else "position", show_walk=walk,
                      show_labels=labels and len(self.lat) <= 200, camp_overlay=camps,
                      neutral_radius=self.neutral_radius)
        self.draw_agent()
        if self.selected is not None:
            i = self.selected
            h = self.lat.cell_spacing
            self.ax_tri.add_patch(plt.Circle(self.lat.xy[i], h * 0.45, fill=False, ec="black",
                                             lw=2.2, zorder=6))
            for j in self.lat.neighbors[i]:
                self.ax_tri.add_patch(plt.Circle(self.lat.xy[j], h * 0.38, fill=False,
                                                 ec="#ffcc00", lw=1.8, zorder=6))
        draw_strip(self.ax_strip, self.lat, self.res, self.metric)
        self.ax_strip.set_xlabel("1-D index order (white = scheme zone border, "
                                 "red ▼ = jump to a non-neighbor)", fontsize=8)
        self.draw_info()
        self.draw_history()
        self.fig.canvas.draw_idle()

    def draw_agent(self):
        xy = to_xy(np.array(self.trail))
        self.ax_tri.plot(xy[:, 0], xy[:, 1], color="white", lw=2.2, zorder=7, alpha=0.9)
        self.ax_tri.plot(xy[:, 0], xy[:, 1], color="#222", lw=1.0, zorder=7.1)
        self.ax_tri.scatter(*xy[-1], s=140, c=[geo.color(self.agent.w, self.metric)],
                            ec="white", lw=2.5, zorder=8)

    def agent_index(self, w):
        if self.res.name == "Vector key":
            return float(geo.key(w, self.neutral_radius, self.metric))
        return float(self.res.index[self.lat.nearest_cell(w)])

    def draw_history(self):
        ax = self.ax_hist
        ax.clear()
        vals = np.array([self.agent_index(w) for w in self.trail])
        if self.res.name == "Vector key":
            # camp is the hundreds digit (shown by the background tint);
            # the line shows the level part, key % 100
            norm = (vals % 100) / 100
            label = "key level (key % 100)"
        else:
            lo, hi = np.nanmin(self.res.index), np.nanmax(self.res.index)
            norm = (vals - lo) / (hi - lo or 1)
            label = "index (scaled)"

        inten = [geo.intensity(w, self.metric) for w in self.trail]
        x = np.arange(len(self.trail))
        camp_cols = np.vstack([geo.NEUTRAL, geo.COLORS])
        for k, w in enumerate(self.trail):
            c = int(geo.camp(w, self.neutral_radius, self.metric))
            ax.axvspan(k - 0.5, k + 0.5, color=camp_cols[c], alpha=0.16, lw=0)
        ax.plot(x, inten, ls="--", color="#888", lw=1.2, label="intensity")
        ax.step(x, norm, where="post", color="#222", lw=1.4, label=label)
        cols = np.atleast_2d(geo.color(np.array(self.trail), self.metric))
        ax.scatter(x, norm, c=cols, s=18, zorder=3, ec="none")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, max(10, len(x)) - 0.5)
        ax.set_xlabel("action #", fontsize=8)
        ax.set_title("Agent: index under this scheme vs. intensity\n"
                     "(background = agent's camp)", fontsize=9, pad=16)
        ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 0.98), ncol=2,
                  frameon=False)
        ax.tick_params(labelsize=7)

    def draw_info(self):
        if self.res is None:
            return
        s, lat, res = self.scores, self.lat, self.res
        L = []
        L.append(f"{res.name}")
        L += textwrap.wrap(res.description, 52)
        L.append("")
        L.append(f"Scores   n = {lat.n}, {len(lat)} cells")
        L.append(f"  coverage              {s.coverage:6.1%}")
        walk = f"{s.continuity:6.1%}" if res.is_path else "   n/a  (not a walk)"
        L.append(f"  walk continuity       {walk}")
        L.append(f"  neighbor tear mean    {s.mean_tear:6.3f}   max {s.max_tear:.3f}")
        L.append(f"  own zones in blocks   {s.zone_runs:3d}  (ideal {s.zone_runs_ideal})")
        L.append(f"  camps in blocks       {s.camp_runs:3d}  (ideal {s.camp_runs_ideal})")
        L.append(f"  same-camp next index  {s.same_camp_steps:6.1%}")
        L.append(f"  index~intensity rank  {s.intensity_rank:6.2f}  (1 = perfect)")
        L.append("")
        a = self.agent
        L.append(f"Agent  (actions {a.actions}, next step size α = {a.alpha:.3f})")
        L.append(f"  w = A {a.w[0]:.2f}  B {a.w[1]:.2f}  C {a.w[2]:.2f}")
        L.append(f"  intensity {a.intensity:.3f}  camp {NAMES[a.camp]}  "
                 f"lean {a.lean:.2f}→{NAMES[a.leaning_toward]}")
        L.append(f"  index here {self.agent_index(a.w):.0f}    vector key {a.key}")
        if self.events:
            L.append(f"  last: {self.events[-1]}")
        if self.selected is not None:
            i = self.selected
            w = lat.weights[i]
            L.append("")
            L.append(f"Clicked cell  A {w[0]:.0%}  B {w[1]:.0%}  C {w[2]:.0%}")
            L.append(f"  index {self._fmt(res.index[i])}   intensity "
                     f"{geo.intensity(w, self.metric):.2f}   camp "
                     f"{NAMES[int(geo.camp(w, self.neutral_radius, self.metric))]}")
            parts = []
            for j in lat.neighbors[i]:
                d = res.index[j] - res.index[i]
                parts.append(f"{self._fmt(res.index[j])}({'+' if d >= 0 else ''}"
                             f"{self._fmt(d)})")
            L += textwrap.wrap("  neighbors: " + "  ".join(parts), 54,
                               subsequent_indent="    ")
        self.ax_info.clear()
        self.ax_info.axis("off")
        self.ax_info.text(0, 1, "\n".join(L), va="top", ha="left", family="monospace",
                          fontsize=8.6, transform=self.ax_info.transAxes)

    @staticmethod
    def _fmt(v):
        return "—" if np.isnan(v) else f"{v:.0f}"

    # ---- events -------------------------------------------------------------
    def on_click(self, event):
        if event.inaxes is not self.ax_tri or event.xdata is None:
            return
        self.selected = self.lat.cell_at_xy(event.xdata, event.ydata)
        self.redraw()

    def on_key(self, event):
        keys = {"a": "A", "b": "B", "c": "C", "n": "N"}
        if event.key in keys:
            self.act(keys[event.key])
        elif event.key == " ":
            self.random_walk(10)
        elif event.key == "r":
            self.reset_agent()

    def act(self, what: str):
        a = self.agent
        if what == "N":
            a.balance()
        elif len(what) == 1:
            a.toward("ABC".index(what))
        else:
            a.blend("ABC".index(what[0]), "ABC".index(what[1]))
        self.trail.append(a.w.copy())
        self.redraw()

    def random_walk(self, k: int):
        favorite = int(self.rng.integers(3))
        for _ in range(k):
            r = self.rng.random()
            if r < 0.55:
                self.agent.toward(favorite)
            elif r < 0.75:
                self.agent.toward(int(self.rng.integers(3)))
            elif r < 0.9:
                i, j = self.rng.choice(3, 2, replace=False)
                self.agent.blend(int(i), int(j))
            else:
                self.agent.balance()
            self.trail.append(self.agent.w.copy())
        self.redraw()

    def reset_agent(self):
        self.agent.reset()
        self.trail = [self.agent.w.copy()]
        self.events.clear()
        self.redraw()


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=12, help="initial grid resolution (default 12)")
    args = p.parse_args()
    Demo(args.n)
    plt.show()


if __name__ == "__main__":
    main()
