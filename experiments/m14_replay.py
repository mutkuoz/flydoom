#!/usr/bin/env python3
"""M14 — trajectory-matched replay: a valid control for closed-loop arms.

THE PROBLEM THIS SOLVES
-----------------------
Every closed-loop comparison in this project has the same defect. Change
anything that affects behaviour -- odour, a readout, the wiring -- and the
agent goes somewhere different, sees something different, and every downstream
measurement moves for reasons that have nothing to do with the manipulation.
The paper already records this for degree-preserving shuffles. The
alternating-odour test hit it again: the phase-locked yaw modulation appeared
equally through a readout odour barely reaches, because gating the odour
changed the trajectory and therefore changed vision.

Freezing the retina removes the confound but also removes the behaviour, and a
frozen frame is not what the animal would see anyway.

THE FIX
-------
Run the agent once under condition A and record the exact action sequence.
Then REPLAY that sequence under condition B, forcing the same actions, so the
agent traverses an identical path through an identical world and receives an
identical visual stream. The brain still computes its own commands; they are
simply not used to drive. Comparing those commands isolates the manipulation.

    pass 1   condition A, closed loop, record actions
    pass 2   condition B, actions forced from pass 1, record the commands
             the brain WOULD have issued

Identical sensory input by construction, realistic visual dynamics unlike a
frozen frame, and the only difference is the manipulation. What it cannot tell
you is whether the difference would have produced different behaviour had it
been allowed to act; it measures the command, not the consequence.

    python experiments/m14_replay.py --device cpu --seeds 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def _agent(seed: int, device: str, smell: bool, yaw_source: str):
    from flydoom.agent import FlyDoomAgent, AgentConfig
    from flydoom.doom import DoomConfig
    from flydoom.motor import MotorConfig
    return FlyDoomAgent(AgentConfig(
        device=device, seed=seed, smell=smell, spiking_t4=True, optic_gain=16,
        motor=MotorConfig(yaw_source=yaw_source),
        doom=DoomConfig(labels=True, seed=seed,
                        scenario="health_gathering_supreme"),
    ))


def record(seed: int, tics: int, device: str, smell: bool,
           yaw_source: str) -> tuple[list, list]:
    """Closed loop. Returns (actions actually taken, commands issued)."""
    agent = _agent(seed, device, smell, yaw_source)
    agent.reset()
    taken, commanded = [], []
    orig = agent.doom.step

    def step(actions, skip):
        taken.append(list(actions))
        commanded.append(float(actions[0]))
        return orig(actions, skip)
    agent.doom.step = step
    try:
        for t in range(tics):
            if agent.tic(t) is None:
                break
    finally:
        agent.close()
    return taken, commanded


def replay(seed: int, device: str, smell: bool, yaw_source: str,
           script: list) -> list:
    """Replay `script` exactly; return the commands the brain would issue.

    The brain runs normally and its motor decoder produces a command every
    tic. That command is recorded and then DISCARDED: the action actually
    sent to the engine comes from the script, so the trajectory, and hence the
    visual stream, is identical to the recording pass.
    """
    agent = _agent(seed, device, smell, yaw_source)
    agent.reset()
    would = []
    orig = agent.doom.step
    k = {"i": 0}

    def step(actions, skip):
        would.append(float(actions[0]))
        i = k["i"]
        k["i"] += 1
        forced = script[i] if i < len(script) else actions
        return orig(forced, skip)
    agent.doom.step = step
    try:
        for t in range(len(script)):
            if agent.tic(t) is None:
                break
    finally:
        agent.close()
    return would


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else 0.0, float("nan")
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--tics", type=int, default=300)
    ap.add_argument("--yaw-source", default="DNa02",
                    choices=["DNa02", "DNp15"])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                                         or "cuda"))
    args = ap.parse_args()

    diffs, on_means, off_means = [], [], []
    for s in range(args.seeds):
        # pass 1: odour present, closed loop, record the path
        script, cmd_on = record(s, args.tics, args.device, True,
                                args.yaw_source)
        # pass 2: odour absent, same path forced
        cmd_off = replay(s, args.device, False, args.yaw_source, script)
        n = min(len(cmd_on), len(cmd_off))
        warm = min(40, n // 4)
        a = np.asarray(cmd_on[warm:n], float)
        b = np.asarray(cmd_off[warm:n], float)
        d = float(a.mean() - b.mean())
        diffs.append(d)
        on_means.append(float(a.mean()))
        off_means.append(float(b.mean()))
        print(f"  seed {s}: yaw command  odour {a.mean():+7.4f}   "
              f"no odour {b.mean():+7.4f}   diff {d:+7.4f}   n={n}")

    m, c = ci95(diffs)
    print(f"\n  odour minus no-odour, identical trajectory: "
          f"{m:+.4f} +- {c:.4f} {'*' if abs(m) > c else ''}")
    print(f"  readout {args.yaw_source}; vision identical by construction")

    out = {"yaw_source": args.yaw_source, "seeds": args.seeds,
           "tics": args.tics, "diff_mean": m, "diff_ci95": c,
           "per_seed": diffs, "on": on_means, "off": off_means}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
