#!/usr/bin/env python3
"""M7 — does the fly know which side the target is on?

Everything that failed in this project failed for one reason: the model cannot
recover the ORDER in which neighbouring ommatidia fire. Direction selectivity
(M3) and looming (M4) are both temporal-order computations, and both came back
empty.

Fixation is not. Working out WHICH SIDE a small object sits on needs only
retinotopy -- a spatial fact -- and the retinotopy is verified intact end to
end (L1's response phase advances at -20.0 deg/deg against an expected 20.0,
and survives to T4a). So LC10a may work where T4/T5 and LPLC2 do not.

The measurement, structured exactly like M6: the agent is driven only by its
eyes, and ViZDoom's label buffer supplies true enemy positions FOR SCORING
ONLY. Nothing in the control path sees them.

    LC10a_left - LC10a_right   vs   true azimuth of the nearest enemy

A real fixation signal gives a strong positive correlation. Controls: the same
correlation against DISTANCE (should be far weaker -- the claim is about
direction, not proximity) and against LC4/LPLC2, which are not fixation cells
and should not track azimuth as well.

    python experiments/m7_fixation.py [--tics 1400] [--live]
"""

from __future__ import annotations

import argparse
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


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M7 — object fixation")
    ap.add_argument("--tics", type=int, default=1400)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(paint("flydoom M7 — object fixation", "1"))
    print(paint("=" * 74, "90"))

    agent = FlyDoomAgent(AgentConfig(
        doom=DoomConfig(scenario=args.scenario, window=args.window, labels=True),
        device=args.device,
    ))

    # LC10a split by hemisphere -- the whole measurement is a left/right compare
    pops = {
        "LC10a_L": agent._idx("LC10a", side="left"),
        "LC10a_R": agent._idx("LC10a", side="right"),
        "LC11_L": agent._idx("LC11", side="left"),
        "LC11_R": agent._idx("LC11", side="right"),
        "LC4_L": agent._idx("LC4", side="left"),
        "LC4_R": agent._idx("LC4", side="right"),
    }
    print("populations " + "  ".join(f"{k}={len(v)}" for k, v in pops.items()))
    missing = [k for k, v in pops.items() if not len(v)]
    if missing:
        print(paint(f"unresolved: {missing} — cannot run", "1;31"))
        return 2
    pt = {k: torch.as_tensor(v.astype(np.int64), device=args.device)
          for k, v in pops.items()}

    # half the Doom viewport: beyond this the enemy is simply not on screen
    half_fov = agent.cfg.doom.fov_deg / 2.0
    print(f"viewport    +/-{half_fov:.0f} deg; samples outside are discarded\n")

    rec = {k: [] for k in ("az", "dist", "d10a", "d11", "d4", "n")}

    def on_tic(a, r):
        threats = [t for t in a.doom.threats()          # MEASUREMENT ONLY
                   if abs(t["azimuth_deg"]) <= half_fov]
        if not threats:
            return True
        t = threats[0]
        f = a.motor._filt
        rate = lambda k: float(f[pt[k]].mean()) if f is not None else 0.0
        rec["az"].append(t["azimuth_deg"])
        rec["dist"].append(t["distance"])
        rec["d10a"].append(rate("LC10a_L") - rate("LC10a_R"))
        rec["d11"].append(rate("LC11_L") - rate("LC11_R"))
        rec["d4"].append(rate("LC4_L") - rate("LC4_R"))
        rec["n"].append(len(threats))
        if r.tic % 200 == 0:
            print(f"  t={r.tic/35:5.1f}s  nearest enemy at {t['azimuth_deg']:+6.1f} deg,"
                  f" {t['distance']:5.0f} units   LC10a L-R = {rec['d10a'][-1]:+7.2f}")
        return True

    try:
        agent.run(args.tics, on_tic=on_tic)
    finally:
        agent.close()

    a = {k: np.asarray(v, float) for k, v in rec.items()}
    n = len(a["az"])
    print(f"\n{paint('SAMPLE', '1;36')}")
    print(f"  usable tics (enemy on screen)  {n}")
    if n < 40:
        print(paint("  too few samples to conclude anything", "1;31"))
        return 1
    print(f"  azimuth spread   {a['az'].min():+.0f} to {a['az'].max():+.0f} deg"
          f"   (sd {a['az'].std():.1f})")
    print(f"  distance spread  {a['dist'].min():.0f} to {a['dist'].max():.0f} units")

    r_az = pearson(a["az"], a["d10a"])
    print(f"\n{paint('DOES LC10a TRACK WHICH SIDE THE TARGET IS ON?', '1;36')}")
    print(f"  {'signal':>22} {'r vs azimuth':>14} {'r vs distance':>15}")
    for key, lbl in (("d10a", "LC10a  L-R"), ("d11", "LC11   L-R"),
                     ("d4", "LC4    L-R")):
        print(f"  {lbl:>22} {pearson(a['az'], a[key]):+14.3f}"
              f" {pearson(a['dist'], a[key]):+15.3f}")

    print(f"\n{paint('VERDICT', '1')}")
    strong = abs(r_az) > 0.3
    right_sign = r_az > 0
    if strong and right_sign:
        print(paint(f"  FIXATION WORKS — r = {r_az:+.3f}", "1;32"))
        print("  LC10a tracks target azimuth with the correct sign: an object to")
        print("  the fly's left drives the left population harder. This is the")
        print("  first visual computation in this project that does what the")
        print("  biology says it should.")
        print("\n  It is a SPATIAL computation, which is why it survives where")
        print("  direction selectivity and looming did not — no temporal order")
        print("  is required, only retinotopy, and retinotopy is intact.")
        return 0
    if strong and not right_sign:
        print(paint(f"  TRACKS, BUT INVERTED — r = {r_az:+.3f}", "1;33"))
        print("  The magnitude is real but the sign is backwards, which means a")
        print("  left/right convention is flipped somewhere between the eye")
        print("  splay in doom.py and the side filter in cells.py. Worth fixing:")
        print("  the signal exists.")
        return 1
    print(paint(f"  NO FIXATION SIGNAL — r = {r_az:+.3f}", "1;31"))
    print("  LC10a does not track which side the target is on. Since retinotopy")
    print("  is verified intact, the failure is in the pooling from retinotopic")
    print("  columns onto LC10a rather than in the input. Aiming is not")
    print("  available, so an ATTACK gate would fire blind.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
