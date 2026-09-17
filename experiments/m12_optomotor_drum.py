#!/usr/bin/env python3
"""M12 — the optomotor drum, run in closed loop.

WHY THIS EXPERIMENT AND NOT THE OTHERS
--------------------------------------
Every behavioural measurement in this project so far scores the model on
collecting medkits in a corridor shooter, a task the animal has no circuitry
for, with a seed-to-seed standard deviation that swamps the effects being
tested. The canonical experiment for the pathway we care about is the
opposite in every respect: impose a rotation on the visual world and measure
whether the animal turns to cancel it. A fly in a rotating striped drum does
this reflexively; it is about as universal as insect behaviour gets.

The imposition is added to the agent's action AFTER the brain has committed
its own command, so the brain never sees the perturbation as an efference
copy. It sees only its consequence: the world rotating past the eyes. The
measurable is the brain's own yaw command, which for a working optomotor
response should OPPOSE the imposed rotation and grow with it.

    imposed +4 deg/tic (world sweeps left)  ->  brain should command right
    imposed -4 deg/tic                      ->  brain should command left

WHY THE DRUM ALTERNATES
-----------------------
The motor decoder subtracts a tau = 3 s baseline from the yaw channel, which
makes the readout a high-pass with a corner near 0.05 Hz. A CONSTANT imposed
rotation therefore produces a sustained counter-turn that is exactly the DC
component the filter removes: the experiment would return a null for a reason
that has nothing to do with vision. The drum reverses on a square wave instead,
at a period well inside the passband, and the measurable is the modulation
locked to that reversal.

Sign and depth, not magnitude. Commanded yaw during the +imposed half minus
commanded yaw during the -imposed half should be NEGATIVE, and grow with drum
speed. That is the whole test.

CONTROLS
--------
mirror  flips the retinal sampling grid, reversing every horizontal optic-flow
        signal while leaving rates, contrast and wiring untouched. A genuine
        optomotor slope must reverse sign.
blind   freezes the retina on one frame. The slope must go to zero: with no
        visual consequence there is nothing to stabilise against.

Neither control changes the imposed rotation, so any residual slope under them
is the motor system responding to its own dynamics rather than to vision.

    python experiments/m12_optomotor_drum.py --device cpu --seeds 6
"""

from __future__ import annotations

import argparse
import json
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402


def run_one(imposed: float, seed: int, tics: int, device: str,
            mirror: bool = False, blind: bool = False,
            yaw_source: str = "DNp15", optic_gain: float = 16.0,
            scenario: str = "defend_the_center",
            half_period: int = 70, eye_map: str = "lattice",
            wide: bool = False, fixed_turn: bool = False,
            open_loop: bool = False) -> dict:
    """One episode with a SQUARE-WAVE rotation added to the agent's action.

    `half_period` is in tics; 70 at 35 tics/s is a 4 s cycle, 0.25 Hz, which
    clears the decoder's 0.05 Hz high-pass corner with room to spare.
    """
    from flydoom.agent import FlyDoomAgent, AgentConfig
    from flydoom.doom import DoomConfig
    from flydoom.motor import MotorConfig

    from m9_behaviour import WIDE_EYE
    cfg = AgentConfig(
        device=device, seed=seed, spiking_t4=True, optic_gain=optic_gain,
        eye_map=eye_map,
        motor=MotorConfig(yaw_source=yaw_source, fixed_turn_sign=fixed_turn),
        doom=DoomConfig(labels=True, seed=seed, scenario=scenario,
                        **(WIDE_EYE if wide else {})),
    )
    agent = FlyDoomAgent(cfg)
    if mirror:
        agent.vision.mirror()
    agent.reset()

    if blind:
        first = {}
        raw = agent.doom.frame

        def frozen():
            if "f" not in first:
                f = raw()
                first["f"] = None if f is None else np.array(f, copy=True)
            return first["f"]
        agent.doom.frame = frozen

    # The imposition is applied to the action the brain has already decided,
    # so the brain sees the rotation as a visual consequence and not as a
    # command it issued.
    orig_step = agent.doom.step
    commanded, phase = [], []
    tick = {"t": 0}

    def step(actions, skip):
        # actions is ordered by DoomSession.BUTTONS; index 0 is yaw delta
        sign = 1.0 if (tick["t"] // half_period) % 2 == 0 else -1.0
        commanded.append(float(actions[0]))
        phase.append(sign)
        actions = list(actions)
        if open_loop:
            # Tethered: the brain's commands are recorded and discarded, and
            # the body only turns as the drum dictates. No walking either,
            # so the retinal image is the drum's alone.
            actions = [0.0] * len(actions)
            actions[0] = imposed * sign
        else:
            actions[0] = actions[0] + imposed * sign
        tick["t"] += 1
        return orig_step(actions, skip)
    agent.doom.step = step

    try:
        for t in range(tics):
            if agent.tic(t) is None:
                break
    finally:
        agent.close()          # one ViZDoom process per agent; see m13
    warm = min(70, len(commanded) // 4)      # drop the first cycle
    c = np.asarray(commanded[warm:], float)
    ph = np.asarray(phase[warm:], float)
    pos = c[ph > 0]
    neg = c[ph < 0]
    depth = (float(pos.mean()) - float(neg.mean())
             if len(pos) and len(neg) else 0.0)
    return {"imposed": imposed, "seed": seed,
            "modulation": depth,
            "yaw_pos": float(pos.mean()) if len(pos) else 0.0,
            "yaw_neg": float(neg.mean()) if len(neg) else 0.0,
            "n": len(c)}


def run_retry(*a, attempts: int = 3, **kw) -> dict:
    """run_one, retried. Under load a ViZDoom engine occasionally exits on
    new_episode and takes the whole sweep with it; the run is deterministic
    per seed, so a retry costs one episode rather than the remaining hours."""
    for k in range(attempts):
        try:
            return run_one(*a, **kw)
        except Exception as e:                       # noqa: BLE001
            if k == attempts - 1:
                raise
            print(f"    engine failed ({type(e).__name__}), retry {k + 1}",
                  flush=True)
            time.sleep(5)


def slope(xs, ys):
    """Least-squares slope of y on x, and its standard error."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(x) < 3 or x.std() == 0:
        return float("nan"), float("nan")
    b, a = np.polyfit(x, y, 1)
    resid = y - (b * x + a)
    se = float(np.sqrt((resid ** 2).sum() / max(len(x) - 2, 1)
                       / max(((x - x.mean()) ** 2).sum(), 1e-9)))
    return float(b), se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imposed", default="0,2,4,8",
                    help="drum speeds, deg per tic (amplitude of the square wave)")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--tics", type=int, default=350)
    ap.add_argument("--half-period", type=int, default=70,
                    help="tics per half cycle; 70 = 4 s at 35 tics/s")
    ap.add_argument("--scenario", default="defend_the_center",
                    help="defend_the_center is a bare circular room, which is "
                         "the closest thing the installed scenarios have to an "
                         "optomotor drum.")
    ap.add_argument("--yaw-source", default="DNp15",
                    choices=["DNa02", "DNp15"])
    ap.add_argument("--optic-gain", type=float, default=16.0)
    ap.add_argument("--eye-map", default="lattice",
                    choices=["lattice", "anatomical"])
    ap.add_argument("--wide", action="store_true",
                    help="the full eye; see m9_behaviour.WIDE_EYE")
    ap.add_argument("--fixed-turn", action="store_true",
                    help="steer toward the more active side; see "
                         "MotorConfig.fixed_turn_sign")
    ap.add_argument("--open-loop", action="store_true",
                    help="tethered fly: the brain's commands are recorded "
                         "but not executed, and it does not walk. Removes "
                         "the closed-loop asymmetry in which a counter-turn "
                         "cancels its own stimulus and a wrong-way turn "
                         "amplifies it.")
    ap.add_argument("--controls", action="store_true",
                    help="also run mirrored and blind arms")
    ap.add_argument("--arm", default=None,
                    choices=["intact", "mirrored", "blind"],
                    help="run a single arm, so the three can go in parallel")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                                         or "cuda"))
    args = ap.parse_args()

    imposed = [float(x) for x in args.imposed.split(",") if x.strip()]
    allarms = {"intact": (False, False), "mirrored": (True, False),
               "blind": (False, True)}
    if args.arm:
        arms = [(args.arm, *allarms[args.arm])]
    else:
        arms = [("intact", False, False)]
        if args.controls:
            arms += [("mirrored", True, False), ("blind", False, True)]

    out = {"eye_map": args.eye_map, "wide": args.wide,
           "fixed_turn": args.fixed_turn, "yaw_source": args.yaw_source,
           "open_loop": args.open_loop,
           "seeds": args.seeds, "tics": args.tics, "argv": sys.argv[1:]}
    for name, mir, bl in arms:
        xs, ys, rows = [], [], []
        print(f"\n=== {name} ===")
        for imp in imposed:
            per = []
            for s in range(args.seeds):
                r = run_retry(imp, s, args.tics, args.device, mir, bl,
                            args.yaw_source, args.optic_gain,
                            args.scenario, args.half_period,
                            args.eye_map, args.wide, args.fixed_turn,
                            args.open_loop)
                per.append(r["modulation"])
                xs.append(imp)
                ys.append(r["modulation"])
                rows.append(r)
            print(f"  drum {imp:+6.1f} deg/tic -> yaw modulation "
                  f"{np.mean(per):+8.4f} +- {np.std(per):.4f}")
        b, se = slope(xs, ys)
        out[name] = {"slope": b, "se": se, "rows": rows}
        verdict = ("OPPOSES (optomotor sign)" if b < -2 * se
                   else "follows" if b > 2 * se else "no slope")
        print(f"  slope {b:+.4f} +- {se:.4f}   {verdict}")

    if "intact" in out and "mirrored" in out:
        bi, bm = out["intact"]["slope"], out["mirrored"]["slope"]
        print(f"\n  mirror reverses the slope? "
              f"{'YES' if bi * bm < 0 else 'NO'}  ({bi:+.4f} -> {bm:+.4f})")
    if "intact" in out and "blind" in out:
        print(f"  blind abolishes the slope?  "
              f"{'YES' if abs(out['blind']['slope']) < abs(out['intact']['slope'])/3 else 'NO'}"
              f"  ({out['intact']['slope']:+.4f} -> {out['blind']['slope']:+.4f})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
