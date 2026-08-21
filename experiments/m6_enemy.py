#!/usr/bin/env python3
"""M6 — enemy present. Does the fly flee, without being told to?

Spec 8's pass condition is that the agent flees on approach with no hand-coded
flee rule anywhere. Spec 7 is explicit about why that matters: the fly already
implements Doom's threat model, and if escape appears it must EMERGE from the
connectome rather than from a heuristic we wrote.

So the discipline here is strict. ViZDoom's label buffer gives true enemy
positions, and this experiment reads them -- but ONLY to score the result. The
agent's control path never sees them. Everything it knows about the world
arrives through 1,581 ommatidia.

WHAT TO EXPECT, stated up front so a positive is not read into noise: M4 found
no looming selectivity at all (LPLC2 fires identically for expansion and
contraction) and M3 found direction selectivity ~50x weaker than a real fly.
Escape is built on exactly those computations, so M6 is expected to fail. It is
worth running anyway, because "fails, and here is the correlation coefficient"
is a far stronger statement than "we did not try".

    python experiments/m6_enemy.py --live
    python experiments/m6_enemy.py --tics 900 --shuffled   # control arm
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.agent import AgentConfig, FlyDoomAgent  # noqa: E402
from flydoom.doom import DoomConfig  # noqa: E402

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
            print(f"  {mark}  {name:<48} {paint(detail, '90')}")
        return all(o for o, _, _ in self.rows)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M6 — enemy present")
    ap.add_argument("--tics", type=int, default=700)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--shuffled", action="store_true",
                    help="shuffled connectome. NOTE: this is NOT a valid "
                         "control here -- see the warning it prints. Use "
                         "m4_looming.py --shuffled instead.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(paint("flydoom M6 — enemy present", "1"))
    print(paint("=" * 76, "90"))

    cfg = AgentConfig(
        doom=DoomConfig(scenario=args.scenario, window=not args.no_window,
                        labels=True),
        device=args.device,
    )
    agent = FlyDoomAgent(cfg)

    if args.shuffled:
        print(paint("""SHUFFLED CONNECTOME -- AND A WARNING ABOUT READING IT

This is not a valid control in CLOSED LOOP, and the numbers below should not
be compared against the intact run. Measured: the intact agent saw an enemy on
186 of 500 tics and finished at full health; the shuffled one saw an enemy on
489 of 492 and finished at 16. A brain that behaves differently generates a
different stimulus distribution, so the two correlations are computed over
different worlds and their difference means nothing.

A shuffle control has to hold the stimulus fixed. That is what
`m4_looming.py --shuffled` does -- identical looming stimulus to both arms,
open loop, nothing the brain does can change what it sees.""", "1;31"))
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(agent.graph.post_idx))
        agent.graph.post_idx = agent.graph.post_idx[perm]
        agent.net = type(agent.net).from_graph(
            agent.graph, device=args.device, seed=0,
            edge_delay=agent.graph.edge_delay_steps(
                agent.ann, config.DT, t_slow=cfg.slow_delay_s),
            graded=agent.graph.graded_mask(agent.ann),
        )

    print(agent.summary())

    dash = None
    if args.live:
        from flydoom.viz import LiveDashboard, have_display
        if have_display():
            dash = LiveDashboard(
                agent.retina,
                ["LC4", "LPLC2", "LC11", "BPN", "MDN",
                 "DNa02_L", "DNa02_R", "DNp01_L", "DNp01_R"],
                dt=config.DT, title="flydoom M6 — enemy present",
            )

    rec = {k: [] for k in ("size", "dist", "azim", "dnp01", "lplc2", "lc4",
                           "lateral", "backward", "yaw", "health")}
    print(f"\n{paint('RUNNING', '1;36')}  {args.tics} tics")

    def on_tic(ag, r):
        threats = ag.doom.threats()          # MEASUREMENT ONLY
        if threats:
            t = threats[0]
            rec["size"].append(t["half_size_deg"])
            rec["dist"].append(t["distance"])
            rec["azim"].append(t["azimuth_deg"])
        else:
            rec["size"].append(0.0)
            rec["dist"].append(9999.0)
            rec["azim"].append(0.0)
        rec["dnp01"].append(max(r.rates.get("DNp01_L", 0.0),
                                r.rates.get("DNp01_R", 0.0)))
        rec["lplc2"].append(r.rates.get("LPLC2", 0.0))
        rec["lc4"].append(r.rates.get("LC4", 0.0))
        rec["lateral"].append(abs(r.action["MOVE_LEFT_RIGHT_DELTA"]))
        rec["backward"].append(-min(0.0, r.action["MOVE_FORWARD_BACKWARD_DELTA"]))
        rec["yaw"].append(r.action["TURN_LEFT_RIGHT_DELTA"])
        rec["health"].append(r.health)
        if r.tic % 100 == 0:
            near = rec["size"][-1]
            print(f"  t={r.tic / 35:5.1f}s hp={r.health:5.1f} "
                  f"nearest theta/2={near:5.1f}deg "
                  f"DNp01={rec['dnp01'][-1]:6.1f} LPLC2={rec['lplc2'][-1]:6.1f}")
        if dash is not None and r.tic % 2 == 0:
            lum = ag.last_luminance.detach().cpu().numpy()
            per_col, off = {}, 0
            for side, eye in ag.retina.eyes.items():
                n = eye.neuron_idx.size
                col = np.full(eye.n_columns, 0.5)
                if n:
                    col[eye.neuron_column] = lum[off:off + n]
                    off += n
                per_col[side] = col
            d = r.rates.get("DNa02_L", 0) - r.rates.get("DNa02_R", 0)
            if not dash.update(r.tic / 35.0, per_col, r.rates, d,
                               r.action["TURN_LEFT_RIGHT_DELTA"] / 12.0,
                               f"theta/2 {rec['size'][-1]:.0f} deg"):
                return False
        return True

    try:
        hist = agent.run(args.tics, on_tic=on_tic)
    finally:
        agent.close()

    arr = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    seen = arr["size"] > 0
    n_seen = int(seen.sum())

    print(f"\n{paint('THREAT EXPOSURE', '1;36')}")
    print(f"  tics with an enemy visible   {n_seen} of {len(hist)}")
    if n_seen:
        s = arr["size"][seen]
        print(f"  angular half-size            {s.min():.1f} to {s.max():.1f} deg"
              f"  (median {np.median(s):.1f})")
        print(f"  nearest approach             {arr['dist'][seen].min():.0f} map units")
    print(f"  health {arr['health'][0]:.0f} -> {arr['health'][-1]:.0f}")

    print(f"\n{paint('DID THE FLY RESPOND?', '1;36')}")
    print(paint("  Pearson r against the enemy's true angular size. A real "
                "escape\n  system gives a strong positive r for the looming "
                "detectors.", "90"))
    print(f"\n  {'signal':>10} {'r vs angular size':>19}")
    corrs = {}
    if n_seen >= 20:
        for k in ("lplc2", "lc4", "dnp01", "lateral", "backward"):
            corrs[k] = pearson(arr["size"][seen], arr[k][seen])
            print(f"  {k:>10} {corrs[k]:+19.3f}")

    print(f"\n{paint('ACCEPTANCE', '1')}")
    c = Checks()
    c.check(len(hist) > 50, "the loop ran with enemies present",
            f"{len(hist)} tics")
    c.check(n_seen >= 20, "enemies were actually visible",
            f"{n_seen} tics with a target")
    c.check(corrs.get("lplc2", 0.0) > 0.3,
            "LOOMING DETECTORS track approaching enemies",
            f"r = {corrs.get('lplc2', 0.0):+.3f}, need > 0.3")
    c.check(corrs.get("dnp01", 0.0) > 0.3,
            "GIANT FIBER tracks approaching enemies",
            f"r = {corrs.get('dnp01', 0.0):+.3f}, need > 0.3")
    c.check(max(corrs.get("lateral", 0.0), corrs.get("backward", 0.0)) > 0.2,
            "THE AGENT FLEES as enemies approach",
            f"lateral r = {corrs.get('lateral', 0.0):+.3f}, "
            f"backward r = {corrs.get('backward', 0.0):+.3f}")

    ok = c.render()
    if dash is not None:
        dash.hold("done — close to exit")

    print(f"\n{paint('=' * 76, '90')}")
    if ok:
        print(paint("VERDICT: M6 PASS — escape emerged from the connectome.", "1;32"))
        return 0
    print(paint("VERDICT: M6 FAIL", "1;31"))
    print(paint("""
Expected, and traceable to a cause already measured rather than a mystery.

Escape needs looming selectivity, and M4 established there is none: LPLC2
fires 0.42 Hz for an expanding disc and 0.42 Hz for a contracting one -- it
detects that something changed, not which way. LC4 responds to dark AREA
instead, so a big distant wall reads the same as a close enemy.

So the null here is not "the connectome lacks an escape circuit". The circuit
is present and correctly wired -- LC4 and LPLC2 resolve, they project to
DNp01, and the geometry is textbook. What is missing is the temporal-order
computation that would let them tell approach from retreat, and that is a
property of the MODEL, not of the wiring. See the M3/M4 diagnoses.""", "33"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
