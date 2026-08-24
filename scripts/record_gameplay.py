#!/usr/bin/env python3
"""Record a short clip of the connectome playing Doom.

Composites, per Doom tic, the rendered frame the agent is looking at beside
what the fly actually receives -- both ommatidial lattices, the monitored
population rates, and the steering signal -- and pipes the result to ffmpeg.

    python scripts/record_gameplay.py                    # 12 s mp4 + gif
    python scripts/record_gameplay.py --seconds 15 --no-gif

Needs ffmpeg on PATH. Runs headless (no Doom window, Agg backend), so it works
over SSH and in CI.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import vizdoom as vzd  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.agent import AgentConfig, FlyDoomAgent  # noqa: E402
from flydoom.doom import DoomConfig  # noqa: E402
from flydoom.olfaction import FOOD_NAMES, THREAT_NAMES  # noqa: E402

TICS_PER_SECOND = 35
SQRT3 = np.sqrt(3.0)

# Same palette as the live dashboard, so the clip and the tool look alike.
BG = "#12140f"
EDGE = "#3a4034"
FG = "#c9cfc0"
MUTED = "#8a9180"
GREEN = "#7fa650"
RUST = "#e0703f"
TEAL = "#56aaae"

LUM_WARMUP = 20
"""Frames used to calibrate the eye colour scale before recording starts.

The absolute luminance range depends on the whole input configuration --
linearising gamma alone drops the mean from 0.196 to 0.048 -- so a hardcoded
ceiling silently goes wrong whenever that changes and the eyes render as flat
black. We take a percentile over a short warmup and then FREEZE it, because a
scale that keeps re-fitting per frame makes the panels flicker and hides the
very contrast changes they exist to show."""

POPULATIONS = ["LC4", "LPLC2", "LC11", "BPN", "MDN",
               "DNa02_L", "DNa02_R", "DNp01_L", "DNp01_R"]


def _encoder_args() -> list[str]:
    """Pick an H.264 encoder this ffmpeg actually has.

    Distributions that ship ffmpeg without the non-free codecs (Fedora's
    `ffmpeg-free`, most notably) have no libx264, so hardcoding it makes the
    script fail on a clean machine with an unhelpful broken-pipe error.
    """
    try:
        have = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        have = ""
    if "libx264" in have:
        return ["-vcodec", "libx264", "-crf", "20", "-preset", "slow"]
    if "libopenh264" in have:
        return ["-vcodec", "libopenh264", "-b:v", "2200k"]
    return ["-vcodec", "mpeg4", "-q:v", "3"]


class Recorder:
    """One matplotlib figure, redrawn per tic and pushed to ffmpeg as raw RGB."""

    def __init__(self, retina, width: int, height: int, fps: int,
                 out: Path, history_s: float = 4.0):
        self.fps = fps
        self.n_hist = int(history_s * fps)
        self.t_hist: list[float] = []
        self.d_hist: list[float] = []
        self.turn_hist: list[float] = []

        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": BG,
            "axes.edgecolor": EDGE, "axes.labelcolor": FG,
            "xtick.color": MUTED, "ytick.color": MUTED,
            "text.color": FG, "font.size": 9,
        })

        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        # 3.35/1/1/1/1 against a 1760x640 canvas leaves the Doom cell at
        # almost exactly 4:3, so the frame fills it rather than floating in
        # letterbox.
        gs = self.fig.add_gridspec(
            2, 5, width_ratios=[3.35, 1.0, 1.0, 1.0, 1.0],
            height_ratios=[1.0, 0.70],
            hspace=0.42, wspace=0.34,
            left=0.010, right=0.960, top=0.875, bottom=0.125,
        )

        # --- what Doom draws -------------------------------------------
        ax = self.fig.add_subplot(gs[:, 0])
        # Doom renders at 320x240 because that is what the retina samples;
        # upscaling here is display only and changes nothing the fly sees.
        self.screen = ax.imshow(np.zeros((240, 320, 3), np.uint8),
                                interpolation="bilinear", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(EDGE)
        ax.set_title("what Doom draws", loc="left", color=FG, pad=6)

        # --- what the fly sees ------------------------------------------
        cols = retina.column_arrays()
        self.eye_art = {}
        # The eyes share the whole top-right band rather than sitting in two
        # of the three bottom-row columns, which left a hole between them.
        eyes_gs = gs[0, 1:5].subgridspec(1, 2, wspace=0.10)
        for k, side in enumerate(("left", "right")):
            a = self.fig.add_subplot(eyes_gs[0, k])
            p_, q_, _, _ = cols[side]
            x = p_ + q_ / 2.0
            y = q_ * SQRT3 / 2.0
            # Column luminance lives in 0-0.42, not 0-1; scaling to the real
            # range is the difference between visible structure and flat wash.
            self.eye_art[side] = a.scatter(
                x, y, c=np.full(len(x), 0.0), cmap="magma",
                vmin=0.0, vmax=1.0, s=7, marker="h", linewidths=0,
            )
            a.set_aspect("equal")
            a.set_title(f"{side} eye \u00b7 {len(x)} columns", color=FG,
                        fontsize=8.5, pad=4)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)

        # --- population rates -------------------------------------------
        a = self.fig.add_subplot(gs[1, 1])
        self.bars = a.bar(range(len(POPULATIONS)), [0] * len(POPULATIONS),
                          color=GREEN, edgecolor="none")
        a.set_xticks(range(len(POPULATIONS)))
        a.set_xticklabels(POPULATIONS, rotation=55, fontsize=6.5, ha="right")
        a.set_ylim(0, 60)
        a.set_ylabel("Hz", fontsize=8)
        a.set_title("descending + visual rates", loc="left", color=FG,
                    fontsize=8.5, pad=4)
        a.grid(axis="y", color="#2a2f24", lw=0.6)
        a.set_axisbelow(True)
        self.pop_ax = a

        # --- steering ----------------------------------------------------
        a = self.fig.add_subplot(gs[1, 2])
        (self.diff_line,) = a.plot([], [], color=RUST, lw=1.5)
        a.set_xlabel("time (s)", fontsize=8)
        a.set_ylabel("DNa02 L\u2212R (Hz)", fontsize=7.5, color=RUST)
        a.tick_params(axis="y", colors=RUST, labelsize=7)
        a.tick_params(axis="x", labelsize=7)
        a.set_title("steering", loc="left", color=FG, fontsize=8.5, pad=4)
        a.grid(color="#2a2f24", lw=0.6)
        a.set_axisbelow(True)
        # The standing L/R asymmetry is ~100 Hz and the decoded turn is ~1,
        # so they need separate scales or one of them is a flat line.
        a2 = a.twinx()
        (self.turn_line,) = a2.plot([], [], color=TEAL, lw=1.1, ls="--")
        a2.set_ylabel("turn", fontsize=7.5, color=TEAL)
        a2.tick_params(axis="y", colors=TEAL, labelsize=7)
        a2.set_facecolor("none")
        for sp in a2.spines.values():
            sp.set_edgecolor(EDGE)
        self.turn_ax, self.turn_ax2 = a, a2

        # --- what it smells ----------------------------------------------
        # Doom renders no odour, so without this panel the channel is
        # invisible in the clip even when it is driving the descending
        # neurons harder than vision is.
        a = self.fig.add_subplot(gs[1, 3])
        (self.food_line,) = a.plot([], [], color=GREEN, lw=1.5, label="food")
        (self.threat_line,) = a.plot([], [], color=RUST, lw=1.5, ls="--",
                                     label="rival")
        a.set_ylim(-0.03, 1.03)
        a.set_xlabel("time (s)", fontsize=8)
        a.set_ylabel("odour", fontsize=8)
        a.set_title("what it smells", loc="left", color=FG, fontsize=8.5,
                    pad=4)
        a.legend(loc="upper right", frameon=False, fontsize=6.5, ncol=2)
        a.grid(color="#2a2f24", lw=0.6)
        a.set_axisbelow(True)
        self.smell_ax = a
        self.food_hist: list[float] = []
        self.threat_hist: list[float] = []

        # --- where it actually went --------------------------------------
        # The panel the first cut of this clip was missing. Rates and a
        # steering trace show the decoder working; only a map shows whether
        # any of it adds up to going somewhere.
        a = self.fig.add_subplot(gs[1, 4])
        # Drawn in the fly's own frame, rotated so it always faces up. A path
        # in world coordinates is just a squiggle; heading-up turns it into
        # something you can read against what the Doom panel is showing.
        (self.path_line,) = a.plot([], [], color=GREEN, lw=1.3, alpha=0.95)
        (self.path_old,) = a.plot([], [], color=GREEN, lw=1.0, alpha=0.45)
        # clip_on matters here: an object beyond the fixed span would
        # otherwise be drawn outside the axes box, floating in the figure.
        self.food_pts = a.scatter([], [], s=22, marker="o", c=GREEN,
                                  edgecolors="none", alpha=0.9, clip_on=True)
        self.threat_pts = a.scatter([], [], s=26, marker="X", c=RUST,
                                    edgecolors="none", alpha=0.9, clip_on=True)
        a.plot([0], [0], marker="^", ms=7, color=FG, ls="none", zorder=5)
        a.set_aspect("equal")
        a.set_title("map \u00b7 heading up", loc="left", color=FG,
                    fontsize=8.5, pad=4)
        a.set_xticks([]); a.set_yticks([])
        a.grid(color="#2a2f24", lw=0.6)
        a.set_axisbelow(True)
        for sp in a.spines.values():
            sp.set_edgecolor(EDGE)
        self.path_ax = a
        self.px: list[float] = []
        self.py: list[float] = []
        self.pang: list[float] = []

        self.title = self.fig.text(
            0.012, 0.950,
            "flydoom \u2014 a frozen FlyWire connectome plays Doom",
            fontsize=12.5, color=FG, family="monospace")
        self.banner = self.fig.text(
            0.012, 0.912, "", fontsize=8.5, color=RUST, family="monospace")
        self.footer = self.fig.text(
            0.955, 0.025,
            "139,255 neurons \u00b7 2.7M edges \u00b7 nothing trained",
            fontsize=7.5, color=MUTED, family="monospace", ha="right")

        self.lum_scale = None
        self._warm: list[float] = []

        self.fig.canvas.draw()
        h, w = np.asarray(self.fig.canvas.buffer_rgba()).shape[:2]
        self.size = (w, h)
        self.ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-",
             "-an", *_encoder_args(), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE,
        )

    def draw(self, frame, luminance, rates, diff, turn, t, banner,
             pos=None, angle=None, objects=(), smell=None):
        if frame is not None:
            self.screen.set_data(frame)
        if self.lum_scale is None:
            self._warm.append(float(np.percentile(
                np.concatenate([np.asarray(v, float)
                                for v in luminance.values()]), 99)))
            if len(self._warm) >= LUM_WARMUP:
                self.lum_scale = max(float(np.mean(self._warm)), 1e-3)
                for art in self.eye_art.values():
                    art.set_clim(0.0, self.lum_scale)
        for side, art in self.eye_art.items():
            if side in luminance:
                art.set_array(np.asarray(luminance[side], float))

        top = max(60.0, max(rates.values(), default=0.0) * 1.1)
        self.pop_ax.set_ylim(0, top)
        for bar, name in zip(self.bars, POPULATIONS):
            bar.set_height(float(rates.get(name, 0.0)))

        self.t_hist.append(t); self.d_hist.append(diff)
        self.turn_hist.append(turn)
        self.t_hist = self.t_hist[-self.n_hist:]
        self.d_hist = self.d_hist[-self.n_hist:]
        self.turn_hist = self.turn_hist[-self.n_hist:]
        self.diff_line.set_data(self.t_hist, self.d_hist)
        self.turn_line.set_data(self.t_hist, self.turn_hist)
        if len(self.t_hist) > 1:
            self.turn_ax.set_xlim(self.t_hist[0], max(self.t_hist[-1], 1e-3))
            self.turn_ax2.set_xlim(*self.turn_ax.get_xlim())
            lim = max(5.0, max(abs(v) for v in self.d_hist) * 1.25)
            self.turn_ax.set_ylim(-lim, lim)
            tlim = max(0.5, max(abs(v) for v in self.turn_hist) * 1.3)
            self.turn_ax2.set_ylim(-tlim, tlim)
        if smell is not None:
            self.food_hist.append(float(smell[0]))
            self.threat_hist.append(float(smell[1]))
            self.food_hist = self.food_hist[-self.n_hist:]
            self.threat_hist = self.threat_hist[-self.n_hist:]
            th = self.t_hist[-len(self.food_hist):]
            self.food_line.set_data(th, self.food_hist)
            self.threat_line.set_data(th, self.threat_hist)
            if len(th) > 1:
                self.smell_ax.set_xlim(th[0], max(th[-1], 1e-3))

        if pos is not None and angle is not None:
            self.px.append(pos[0]); self.py.append(pos[1])
            self.pang.append(angle)
            # Rotate the whole trail into the CURRENT heading frame, so the
            # fly sits at the origin pointing up and the world moves past it.
            ca = np.cos(np.radians(90.0 - angle))
            sa = np.sin(np.radians(90.0 - angle))
            dx = np.asarray(self.px) - pos[0]
            dy = np.asarray(self.py) - pos[1]
            rx = dx * ca - dy * sa
            ry = dx * sa + dy * ca
            keep = 240        # tics of trail drawn bright
            self.path_line.set_data(rx[-keep:], ry[-keep:])
            self.path_old.set_data(rx[:-keep], ry[:-keep])

            fx, fy, tx, ty = [], [], [], []
            for o in objects:
                # threats() gives azimuth relative to gaze, positive to the
                # left, which is already the heading-up frame.
                r_, th_ = o["distance"], np.radians(o["azimuth_deg"])
                ox, oy = -r_ * np.sin(th_), r_ * np.cos(th_)
                (fx, fy) if o["kind"] == "food" else (tx, ty)
                if o["kind"] == "food":
                    fx.append(ox); fy.append(oy)
                else:
                    tx.append(ox); ty.append(oy)
            self.food_pts.set_offsets(np.c_[fx, fy] if fx else np.empty((0, 2)))
            self.threat_pts.set_offsets(np.c_[tx, ty] if tx else np.empty((0, 2)))

            # Fixed span so distances stay comparable between moments, with
            # the fly low in the frame because what is ahead matters more
            # than what is behind.
            span = 420.0
            self.path_ax.set_xlim(-span, span)
            self.path_ax.set_ylim(-span * 0.55, span * 1.45)

        self.banner.set_text(banner)

        self.fig.canvas.draw()
        self.ff.stdin.write(self.fig.canvas.buffer_rgba())

    def new_episode(self) -> None:
        """Drop the trail. A respawn is elsewhere on the map, and carrying the
        old path across would draw a line through geometry it never walked."""
        self.px.clear(); self.py.clear(); self.pang.clear()

    def close(self):
        if self.ff.stdin:
            self.ff.stdin.close()
        self.ff.wait()
        plt.close(self.fig)


def per_column_luminance(agent):
    lum = agent.last_luminance.detach().cpu().numpy()
    out, off = {}, 0
    for side, eye in agent.retina.eyes.items():
        n = eye.neuron_idx.size
        col = np.full(eye.n_columns, 0.5)
        if n:
            col[eye.neuron_column] = lum[off:off + n]
            off += n
        out[side] = col
    return out


def make_gif(mp4: Path, gif: Path, fps: int, width: int,
             colors: int = 96, seconds: float | None = None) -> None:
    """Two-pass palette GIF -- one-pass output is both larger and uglier."""
    palette = gif.with_suffix(".palette.png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    trim = ["-t", str(seconds)] if seconds else []
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *trim, "-i", str(mp4),
                    "-vf", f"{vf},palettegen=stats_mode=diff:max_colors={colors}",
                    str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *trim, "-i", str(mp4),
                    "-i", str(palette), "-lavfi",
                    f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)
    palette.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="record flydoom gameplay")
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="clip length in game seconds")
    ap.add_argument("--warmup", type=int, default=25,
                    help="tics to run before recording, so the rate filters "
                         "have converged and the first frame is not blank")
    ap.add_argument("--scenario", default="deathmatch",
                    help="deathmatch is an open map with rooms, items and "
                         "monsters, so movement is visible. "
                         "defend_the_center pins the player in a bare "
                         "circular room and shows nothing.")
    ap.add_argument("--out", type=Path, default=Path("media/flydoom.mp4"))
    ap.add_argument("--width", type=int, default=1760)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--no-smell", action="store_true",
                    help="record without the odour channel")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--gif-seconds", type=float, default=13.0,
                    help="length of the GIF excerpt. The mp4 keeps the full "
                         "clip; a GIF of the whole thing is several times the "
                         "size for a file that autoplays in a README.")
    ap.add_argument("--gif-colors", type=int, default=96,
                    help="GIF palette size. Every pixel changes as the view\n                         turns, so length and size drive the file more than\n                         dithering does; 640px/10fps/96 keeps 13 s under 5 MB.")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tics = int(args.seconds * TICS_PER_SECOND)

    # Smell ON. It is off by default because M6/M7 are only valid without it,
    # but a clip recorded without it shows a fly with no nose, and the odour
    # channel drives the descending neurons harder than vision does.
    agent = FlyDoomAgent(AgentConfig(
        doom=DoomConfig(scenario=args.scenario, window=False,
                        labels=not args.no_smell),
        smell=not args.no_smell,
        device=args.device,
    ))
    print(agent.summary())
    print(f"\nrecording {tics} tics ({args.seconds:.0f} s) "
          f"after {args.warmup} warmup tics -> {args.out}")

    rec = Recorder(agent.retina, args.width, args.height, TICS_PER_SECOND,
                   args.out)
    print(f"canvas {rec.size[0]}x{rec.size[1]}")

    state = {"n": 0, "episode": 0}

    def on_tic(ag, r):
        if r.tic < args.warmup:
            return True
        d = r.rates.get("DNa02_L", 0.0) - r.rates.get("DNa02_R", 0.0)
        t = state["n"] / TICS_PER_SECOND
        g = ag.doom.game
        pos = (g.get_game_variable(vzd.GameVariable.POSITION_X),
               g.get_game_variable(vzd.GameVariable.POSITION_Y))
        ang = g.get_game_variable(vzd.GameVariable.ANGLE)
        objs = [{"distance": o["distance"], "azimuth_deg": o["azimuth_deg"],
                 "kind": ("food" if o["name"] in FOOD_NAMES else "threat")}
                for o in ag.doom.threats()
                if o["name"] in FOOD_NAMES or o["name"] in THREAT_NAMES]
        sm = ((ag.smell.sensed["food"], ag.smell.sensed["threat"])
              if ag.smell is not None else None)
        smelling = ("" if sm is None else
                    f"   smell f{sm[0]:.2f} r{sm[1]:.2f}")
        rec.draw(ag.doom.frame(), per_column_luminance(ag), r.rates, d,
                 r.action["TURN_LEFT_RIGHT_DELTA"] / 12.0, t,
                 f"t {t:5.1f}s   health {r.health:3.0f}   "
                 f"yaw {r.action['TURN_LEFT_RIGHT_DELTA']:+6.2f}   "
                 f"walk {r.action['MOVE_FORWARD_BACKWARD_DELTA']:+6.2f}"
                 + smelling
                 + ("" if state["episode"] == 0
                    else f"   life {state['episode'] + 1}"),
                 pos=pos, angle=ang, objects=objs, smell=sm)
        state["n"] += 1
        if state["n"] % 35 == 0:
            print(f"  {state['n'] / TICS_PER_SECOND:4.1f} s recorded")
        return state["n"] < tics

    try:
        # The agent dies, and agent.run() stops when the episode does. Keep
        # starting fresh episodes until the requested length is recorded,
        # rather than silently returning a clip shorter than asked for.
        while state["n"] < tics:
            before = state["n"]
            agent.run(tics - state["n"] + args.warmup + 5, on_tic=on_tic)
            if state["n"] == before:      # made no progress; stop rather than spin
                print("  episode produced no frames; stopping")
                break
            if state["n"] < tics:
                state["episode"] += 1
                rec.new_episode()
                print(f"  died at {state['n'] / TICS_PER_SECOND:.1f} s "
                      f"— starting episode {state['episode'] + 1}")
    finally:
        agent.close()
        rec.close()

    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    if not args.no_gif:
        gif = args.out.with_suffix(".gif")
        make_gif(args.out, gif, args.gif_fps, args.gif_width,
                 args.gif_colors, args.gif_seconds)
        print(f"wrote {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
