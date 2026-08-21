#!/usr/bin/env python3
"""M5 — closed loop. A frozen connectome plays Doom.

This is the first experiment where the fly's own output changes what it sees
next. Spec 8 asks for: it runs, it does not crash, the buttons do not chatter,
and the movement is stable.

Stability is the real content. Everything before M5 was open loop, where a bad
gain shows up as a wrong number; in closed loop it shows up as the agent
spinning on the spot forever, because its own turn feeds back into its own
retina. That failure mode does not exist until the loop is closed, which is
why this milestone is worth having even though M3 and M4 do not pass.

READ THIS BEFORE INTERPRETING ANYTHING THE AGENT DOES
-----------------------------------------------------
M3 shows direction selectivity ~50x weaker than a real fly, and M4 shows no
looming selectivity at all. So visually-driven STEERING and ESCAPE are not
working, and any turning you see is driven by raw contrast asymmetry between
the eyes rather than by motion vision. What IS working: the taste pathway (M2,
fully validated), contrast transduction through the lamina, and conduction to
the descending neurons. Watch it with that in mind.

    python experiments/m5_closed_loop.py --live       # watch it play
    python experiments/m5_closed_loop.py --tics 400   # headless, for the check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.agent import AgentConfig, FlyDoomAgent  # noqa: E402
from flydoom.doom import DoomConfig  # noqa: E402
from flydoom.motor import MotorConfig  # noqa: E402

USE_COLOR = sys.stdout.isatty()


def paint(t: str, c: str) -> str:
    return f"\033[{c}m{t}\033[0m" if USE_COLOR else t


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, ok, name, detail=""):
        self.rows.append((bool(ok), name, detail))
        return bool(ok)

    def render(self):
        for ok, name, detail in self.rows:
            mark = paint("PASS", "32") if ok else paint("FAIL", "1;31")
            print(f"  {mark}  {name:<46} {paint(detail, '90')}")
        return all(o for o, _, _ in self.rows)


def chatter_rate(series: list[float], deadzone: float = 1e-6) -> float:
    """Fraction of consecutive tics where a binary signal flips.

    Spec 6.2's whole reason for a Schmitt trigger. Above ~0.25 the button is
    rattling at tic rate and the agent looks like it is having a seizure.
    """
    flips = sum(
        1 for a, b in zip(series, series[1:])
        if (a > deadzone) != (b > deadzone)
    )
    return flips / max(len(series) - 1, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M5 — closed loop")
    ap.add_argument("--tics", type=int, default=350)
    ap.add_argument("--live", action="store_true",
                    help="live dashboard alongside the Doom window")
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--no-window", action="store_true",
                    help="headless; the Doom window is shown by default")
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--yaw-gain", type=float, default=MotorConfig.yaw_gain)
    ap.add_argument("--spiking", action="store_true",
                    help="all-spiking optic lobe (the paper's model)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(paint("flydoom M5 — closed loop", "1"))
    print(paint("=" * 76, "90"))

    cfg = AgentConfig(
        doom=DoomConfig(scenario=args.scenario, window=not args.no_window),
        motor=MotorConfig(yaw_gain=args.yaw_gain),
        graded=not args.spiking,
        bias_mv=args.bias,
        device=args.device,
    )
    t0 = time.perf_counter()
    agent = FlyDoomAgent(cfg)
    print(agent.summary())
    print(f"\nbuilt in {time.perf_counter() - t0:.1f} s")

    dash = None
    if args.live:
        from flydoom.viz import LiveDashboard, have_display
        if have_display():
            pops = ["LC4", "LPLC2", "LC11", "BPN", "MDN",
                    "DNa02_L", "DNa02_R", "DNp01_L", "DNp01_R"]
            dash = LiveDashboard(agent.retina, pops, dt=config.DT,
                                 title="flydoom — a fly plays Doom")
        else:
            print(paint("no display; ignoring --live", "33"))

    print(f"\n{paint('RUNNING', '1;36')}  {args.tics} tics "
          f"({args.tics / 35:.1f} s of game time)")

    log = {"yaw": [], "fwd": [], "attack": [], "use": [], "health": []}
    wall = time.perf_counter()

    def on_tic(ag, rec):
        log["yaw"].append(rec.action["TURN_LEFT_RIGHT_DELTA"])
        log["fwd"].append(rec.action["MOVE_FORWARD_BACKWARD_DELTA"])
        log["attack"].append(rec.action["ATTACK"])
        log["use"].append(rec.action["USE"])
        log["health"].append(rec.health)
        if rec.tic % 35 == 0:
            print(f"  t={rec.tic / 35:5.1f}s  hp={rec.health:5.1f}  "
                  f"yaw={rec.action['TURN_LEFT_RIGHT_DELTA']:+6.2f}  "
                  f"fwd={rec.action['MOVE_FORWARD_BACKWARD_DELTA']:+6.2f}  "
                  f"DNa02 L/R={rec.rates.get('DNa02_L', 0):6.1f}/"
                  f"{rec.rates.get('DNa02_R', 0):6.1f}")
        if dash is not None and rec.tic % 2 == 0:
            lum = ag.last_luminance.detach().cpu().numpy()
            per_col = {}
            off = 0
            for side, eye in ag.retina.eyes.items():
                n = eye.neuron_idx.size
                col = np.full(eye.n_columns, 0.5)
                if n:
                    col[eye.neuron_column] = lum[off:off + n]
                    off += n
                per_col[side] = col
            d = rec.rates.get("DNa02_L", 0) - rec.rates.get("DNa02_R", 0)
            if not dash.update(rec.tic / 35.0, per_col, rec.rates, d,
                               rec.action["TURN_LEFT_RIGHT_DELTA"] / 12.0,
                               f"tic {rec.tic}   hp {rec.health:.0f}"):
                return False
        return True

    try:
        hist = agent.run(args.tics, on_tic=on_tic)
    finally:
        elapsed = time.perf_counter() - wall
        agent.close()

    if not hist:
        print(paint("no tics ran", "31"))
        return 1

    game_s = len(hist) / 35.0
    print(f"\nran {len(hist)} tics ({game_s:.1f} s game) in {elapsed:.1f} s wall"
          f"  ->  {game_s / elapsed:.2f}x realtime")

    # ---------------- acceptance ----------------
    print(f"\n{paint('ACCEPTANCE', '1')}")
    c = Checks()

    c.check(len(hist) >= min(args.tics, 50) * 0.5, "the loop runs without crashing",
            f"{len(hist)} tics")

    dn = max(max(r.rates.get("DNa02_L", 0), r.rates.get("DNa02_R", 0))
             for r in hist)
    c.check(dn > 0.5, "descending neurons are active", f"peak {dn:.1f} Hz")

    atk_chatter = chatter_rate(log["attack"])
    use_chatter = chatter_rate(log["use"])
    c.check(atk_chatter < 0.25, "ATTACK does not chatter",
            f"{atk_chatter * 100:.0f}% of tics flip")
    c.check(use_chatter < 0.25, "USE does not chatter",
            f"{use_chatter * 100:.0f}% of tics flip")

    yaw = np.array(log["yaw"])
    c.check(np.abs(yaw).max() <= cfg.motor.yaw_max_deg + 1e-6,
            "yaw stays inside its clamp",
            f"max |yaw| {np.abs(yaw).mean():.2f} deg/tic mean")

    # Spinning is THE closed-loop failure mode: a constant-sign yaw means the
    # agent's own turn is feeding its own retina and never settling.
    if len(yaw) > 20:
        same_sign = float(np.mean(np.sign(yaw[10:]) == np.sign(yaw[10:]).mean()))
        spin = abs(float(np.mean(np.sign(yaw[10:]))))
        c.check(spin < 0.98, "not locked in a constant spin",
                f"mean sign {np.mean(np.sign(yaw[10:])):+.2f}")

    c.check(np.isfinite(yaw).all() and np.isfinite(log["fwd"]).all(),
            "no NaN or inf reached the actuators")

    ok = c.render()
    if dash is not None:
        dash.hold("done — close to exit")

    print(f"\n{paint('=' * 76, '90')}")
    if ok:
        print(paint("VERDICT: M5 PASS — the loop is closed and stable.", "1;32"))
        print(paint("Remember what is and is not driving the behaviour: see the "
                    "module docstring.", "90"))
        return 0
    print(paint("VERDICT: M5 FAIL", "1;31"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
