"""Live dashboard — watch the fly brain while it runs.

Four panels:

    left / right eye   the ommatidial lattice, one hexagon per column, shaded
                       by what that column currently sees
    population rates   a bar per monitored cell type, so you can see the signal
                       climb the visual hierarchy in real time
    steering trace     the DNa02 left-right differential and the decoded turn

The same dashboard serves the synthetic rig (M3/M4) and ViZDoom later; only the
source of `luminance` changes. Nothing here touches the simulation — it reads
state and draws it, so `--live` can be dropped without changing a result.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

SQRT3 = math.sqrt(3.0)


class LiveDashboard:
    def __init__(
        self,
        retina,
        populations: list[str],
        history_s: float = 3.0,
        dt: float = 5e-4,
        title: str = "flydoom",
        rate_ceiling: float = 60.0,
    ) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        self.plt = plt
        self.populations = populations
        self.rate_ceiling = rate_ceiling
        self.n_hist = max(16, int(history_s / max(dt, 1e-9) / 20))
        self.t_hist: deque = deque(maxlen=self.n_hist)
        self.d_hist: deque = deque(maxlen=self.n_hist)
        self.turn_hist: deque = deque(maxlen=self.n_hist)

        plt.rcParams.update({
            "figure.facecolor": "#12140f",
            "axes.facecolor": "#12140f",
            "axes.edgecolor": "#3a4034",
            "axes.labelcolor": "#c9cfc0",
            "xtick.color": "#8a9180",
            "ytick.color": "#8a9180",
            "text.color": "#c9cfc0",
            "font.size": 9,
        })

        self.fig = plt.figure(figsize=(13, 6.2))
        self.fig.canvas.manager.set_window_title(title)
        gs = self.fig.add_gridspec(
            2, 3, width_ratios=[1, 1, 1.25], height_ratios=[1.35, 1],
            hspace=0.30, wspace=0.24,
            left=0.045, right=0.985, top=0.90, bottom=0.09,
        )

        # --- eyes -------------------------------------------------------
        self.eye_ax, self.eye_art = {}, {}
        cols = retina.column_arrays()
        for k, side in enumerate(("left", "right")):
            ax = self.fig.add_subplot(gs[0, k])
            p, q, az, el = cols[side]
            x = p + q / 2.0
            y = q * SQRT3 / 2.0
            art = ax.scatter(
                x, y, c=np.full(len(x), 0.5), cmap="magma",
                vmin=0, vmax=1, s=13, marker="h", linewidths=0,
            )
            ax.set_aspect("equal")
            ax.set_title(f"{side} eye   {len(x)} columns", color="#c9cfc0")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            self.eye_ax[side] = ax
            self.eye_art[side] = art

        # --- populations ------------------------------------------------
        ax = self.fig.add_subplot(gs[1, :2])
        self.pop_bars = ax.bar(
            range(len(populations)), [0] * len(populations),
            color="#7fa650", edgecolor="none",
        )
        ax.set_xticks(range(len(populations)))
        ax.set_xticklabels(populations, rotation=0, fontsize=8)
        ax.set_ylim(0, rate_ceiling)
        ax.set_ylabel("rate (Hz)")
        ax.set_title("population firing rate", loc="left", color="#c9cfc0")
        ax.grid(axis="y", color="#2a2f24", lw=0.6)
        ax.set_axisbelow(True)
        self.pop_ax = ax
        self.pop_labels = [
            ax.text(i, 0, "", ha="center", va="bottom", fontsize=7.5,
                    color="#c9cfc0")
            for i in range(len(populations))
        ]

        # --- steering ---------------------------------------------------
        ax = self.fig.add_subplot(gs[:, 2])
        (self.diff_line,) = ax.plot([], [], color="#e0703f", lw=1.6,
                                    label="DNa02  L − R")
        (self.turn_line,) = ax.plot([], [], color="#56aaae", lw=1.2, ls="--",
                                    label="decoded turn")
        ax.axhline(0, color="#3a4034", lw=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Hz   /   turn")
        ax.set_title("steering signal", loc="left", color="#c9cfc0")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.grid(color="#2a2f24", lw=0.6)
        ax.set_axisbelow(True)
        self.turn_ax = ax

        self.banner = self.fig.text(
            0.045, 0.955, "", fontsize=11, color="#e0703f", family="monospace",
        )

        self.fig.show()
        self.plt.pause(0.001)

    # -- updating --------------------------------------------------------

    def update(
        self,
        t: float,
        luminance: dict[str, np.ndarray],
        pop_rates: dict[str, float],
        diff: float,
        turn: float,
        banner: str = "",
    ) -> bool:
        """Redraw. Returns False once the user closes the window."""
        if not self.plt.fignum_exists(self.fig.number):
            return False

        for side, art in self.eye_art.items():
            if side in luminance:
                art.set_array(np.asarray(luminance[side], dtype=float))

        top = max(self.rate_ceiling, max(pop_rates.values(), default=0) * 1.1)
        self.pop_ax.set_ylim(0, top)
        for bar, label, name in zip(self.pop_bars, self.pop_labels,
                                    self.populations):
            r = float(pop_rates.get(name, 0.0))
            bar.set_height(r)
            label.set_position((label.get_position()[0], r))
            label.set_text(f"{r:.0f}" if r >= 0.5 else "")

        self.t_hist.append(t)
        self.d_hist.append(diff)
        self.turn_hist.append(turn)
        self.diff_line.set_data(self.t_hist, self.d_hist)
        self.turn_line.set_data(self.t_hist, self.turn_hist)
        if len(self.t_hist) > 1:
            self.turn_ax.set_xlim(self.t_hist[0], max(self.t_hist[-1], 1e-3))
            lim = max(5.0, max(abs(v) for v in self.d_hist) * 1.25)
            self.turn_ax.set_ylim(-lim, lim)

        if banner:
            self.banner.set_text(banner)

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        return True

    def hold(self, message: str = "close the window to continue") -> None:
        self.banner.set_text(message)
        self.plt.show(block=True)

    def close(self) -> None:
        self.plt.close(self.fig)


def have_display() -> bool:
    import os
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
