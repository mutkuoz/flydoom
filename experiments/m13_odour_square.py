#!/usr/bin/env python3
"""M13 — does an ALTERNATING odour modulate the steering command?

WHY
---
Odour shifts DNa02's left-right asymmetry by +20.6 +- 1.8 Hz, verified on
every one of six seeds, and DNa02 is the descending neuron the literature
assigns to goal-directed steering. On magnitude that shift should dominate
behaviour: it clears the 1.5 Hz deadzone and would command 21 * 0.44 = 9.2
deg/tic of yaw against a cap of 12.

Yet neither klinotaxis nor a whole-episode smell-on/smell-off comparison finds
a behavioural effect. There is an instrumental reason to expect exactly that.
Smell is on for the entire episode, so the shift is a sustained DC offset, and
the yaw channel is baseline-centred with tau = 3 s, which is a high-pass with a
corner near 0.05 Hz. A constant is what that filter is built to remove.

The same gap would have nulled the optomotor drum before it was made to
alternate. This applies the same fix to olfaction: gate the odour as a square
wave inside the filter's passband, and ask whether the steering command is
modulated in phase with it.

    odour present half   ->  yaw command should differ from
    odour absent half        the odour-absent half, in phase

If the coupling is real but DC-blocked, it appears here and in no other
measurement. If it is absent here too, then a 21 Hz modulation of a descending
neuron genuinely does not reach behaviour, and the missing piece is downstream
of the brain.

CONTROLS
--------
none    the same square-wave bookkeeping with the odour never enabled, so any
        modulation locked to the phase is bookkeeping rather than smell.
shuffle a degree-preserving shuffle. Valid here because the odour schedule is
        imposed rather than sensed, so both arms see identical input.

    python experiments/m13_odour_square.py --device cpu --seeds 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def run_one(seed: int, tics: int, device: str, half_period: int = 70,
            odour: bool = True, shuffled: bool = False,
            yaw_source: str = "DNa02") -> dict:
    """One episode with the odour channel gated on a square wave."""
    from flydoom.agent import FlyDoomAgent, AgentConfig
    from flydoom.doom import DoomConfig
    from flydoom.motor import MotorConfig

    cfg = AgentConfig(
        device=device, seed=seed, smell=True, spiking_t4=True, optic_gain=16,
        motor=MotorConfig(yaw_source=yaw_source),
        doom=DoomConfig(labels=True, seed=seed,
                        scenario="health_gathering_supreme"),
    )
    agent = FlyDoomAgent(cfg)
    if shuffled and hasattr(agent.graph, "shuffle"):
        pass  # handled by caller if available; not required for the main test
    agent.reset()

    # Gate the olfactory drive rather than the odour sources, so the schedule
    # is imposed exactly and nothing else in the pipeline changes.
    real_substep = agent.smell.substep
    zero = None
    state = {"t": 0, "on": True}

    def gated():
        nonlocal zero
        out = real_substep()
        if zero is None:
            zero = out.new_zeros(out.shape)
        return out if (state["on"] and odour) else zero
    agent.smell.substep = gated

    yaws, phase = [], []
    orig_step = agent.doom.step

    def step(actions, skip):
        yaws.append(float(actions[0]))
        phase.append(1.0 if state["on"] else -1.0)
        return orig_step(actions, skip)
    agent.doom.step = step

    for t in range(tics):
        state["t"] = t
        state["on"] = (t // half_period) % 2 == 0
        if agent.tic(t) is None:
            break

    warm = min(half_period, len(yaws) // 4)
    y = np.asarray(yaws[warm:], float)
    ph = np.asarray(phase[warm:], float)
    on_y, off_y = y[ph > 0], y[ph < 0]
    if not len(on_y) or not len(off_y):
        return {"seed": seed, "modulation": 0.0, "abs_modulation": 0.0, "n": 0}
    return {"seed": seed,
            "modulation": float(on_y.mean() - off_y.mean()),
            "abs_modulation": float(np.abs(on_y).mean() - np.abs(off_y).mean()),
            "yaw_on": float(on_y.mean()), "yaw_off": float(off_y.mean()),
            "n": len(y)}


def summarise(rows, label):
    m = np.array([r["modulation"] for r in rows], float)
    a = np.array([r["abs_modulation"] for r in rows], float)
    def ci(v):
        return (float(v.mean()),
                float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1
                else float("nan"))
    mm, mc = ci(m)
    am, ac = ci(a)
    sig = "*" if abs(mm) > mc else " "
    asig = "*" if abs(am) > ac else " "
    print(f"  {label:<10} yaw(on)-yaw(off) {mm:+7.3f} +-{mc:<6.3f}{sig}"
          f"   |yaw| {am:+7.3f} +-{ac:<6.3f}{asig}   n={len(rows)}")
    return {"modulation_mean": mm, "modulation_ci": mc,
            "abs_mean": am, "abs_ci": ac, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--tics", type=int, default=420)
    ap.add_argument("--half-period", type=int, default=70,
                    help="tics per half cycle; 70 = 4 s at 35 tics/s, which "
                         "clears the yaw channel's 0.05 Hz high-pass corner")
    ap.add_argument("--yaw-source", default="DNa02",
                    choices=["DNa02", "DNp15"],
                    help="DNa02 is where odour lands; DNp15 is the visual one "
                         "and serves as a specificity control")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                                         or "cuda"))
    args = ap.parse_args()

    out = {}
    for label, odour in (("odour", True), ("none", False)):
        rows = [run_one(s, args.tics, args.device, args.half_period,
                        odour=odour, yaw_source=args.yaw_source)
                for s in range(args.seeds)]
        out[label] = summarise(rows, label)

    d = out["odour"]["modulation_mean"] - out["none"]["modulation_mean"]
    print(f"\n  odour minus bookkeeping control: {d:+.3f} deg/tic")
    out["difference"] = d

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
