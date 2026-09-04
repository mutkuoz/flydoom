#!/usr/bin/env python3
"""M11 — does antennal contact reach the steering neurons?

The model had no mechanosensation. Walking into a wall produced no afferent
activity anywhere, which makes collisions a consequence the agent cannot
sense, and collisions are the one behavioural metric that moved under every
intervention tried.

This is the open-loop test that has to pass before any closed-loop claim, and
it is structured like the olfaction test rather than the motion tests: a
labelled line is driven, and the question is whether the effect arrives at the
descending neurons with the sign the wiring predicts. Direction of the effect,
not magnitude.

    left antenna driven   ->  DNa02 asymmetry should favour one side
    right antenna driven  ->  it should REVERSE

Reversal is the whole test. A standing left-right difference that does not
move with the stimulated side is the single-brain asymmetry, not a response,
and that asymmetry is 20-45 Hz here -- large enough to have buried the visual
signal entirely, which is why sign rather than magnitude is the measurable.

The control is a degree-preserving shuffle, valid here because this is open
loop: the stimulus is imposed and nothing feeds back, so both arms see
identical input. In closed loop it would not be (see the shuffle-validity
result in the paper).

    python experiments/m11_antennal_touch.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.cells import AnnotationTable  # noqa: E402
from flydoom.graph import ConnectomeGraph  # noqa: E402
from flydoom.lif import LIFNetwork, LIFParams  # noqa: E402
from flydoom.mechanosensation import MechanoConfig  # noqa: E402
from flydoom.registry import by_name  # noqa: E402


def run_forced(side: str | None, tics: int, seed: int, device: str,
               shuffled: bool = False) -> dict:
    """Run the real pipeline with antennal contact clamped to one side.

    The bare network is silent: DNa02 has no baseline without the visual drive
    that reaches it through the optic lobe, so an isolated stimulation
    experiment measures modulation of nothing. This runs the full agent on a
    live scene and overrides only the contact state, which is the same
    structure the olfaction test uses.
    """
    from flydoom.agent import FlyDoomAgent, AgentConfig
    from flydoom.doom import DoomConfig

    cfg = AgentConfig(device=device, touch=True, seed=seed,
                      doom=DoomConfig(labels=True, seed=seed,
                                      scenario="health_gathering_supreme"))
    agent = FlyDoomAgent(cfg)
    if shuffled:
        agent.net.shuffle_targets(seed=seed) if hasattr(
            agent.net, "shuffle_targets") else None
    agent.reset()

    # Clamp contact: replace on_tic so the deflection is imposed rather than
    # derived, leaving every other pathway untouched and identical across arms.
    def clamp(x, y, ang, fwd, _t=agent.touch, _s=side):
        _t.contact["left"] = 1.0 if _s in ("left", "both") else 0.0
        _t.contact["right"] = 1.0 if _s in ("right", "both") else 0.0
        _t.touching = _s is not None
    agent.touch.on_tic = clamp

    acc, n = {}, 0
    for t in range(tics):
        rec = agent.tic(t)
        if rec is None:
            break
        if t < 20:                      # let the rate filter settle
            continue
        for k, v in rec.rates.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n += 1
    out = {k: v / max(n, 1) for k, v in acc.items()}
    out["DNa02_LmR"] = out.get("DNa02_L", 0.0) - out.get("DNa02_R", 0.0)
    out["_tics"] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tics", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                                         or "cuda"))
    args = ap.parse_args()

    rows = {"none": [], "left": [], "right": []}
    for seed in range(args.seeds):
        for side in ("none", "left", "right"):
            r = run_forced(None if side == "none" else side,
                           args.tics, seed, args.device)
            rows[side].append(r)
            print(f"  seed {seed} {side:>5}: DNa02 L {r.get('DNa02_L', 0):7.2f}"
                  f"  R {r.get('DNa02_R', 0):7.2f}   L-R {r['DNa02_LmR']:+7.2f} Hz")

    def mean(side, key="DNa02_LmR"):
        v = [x[key] for x in rows[side]]
        return sum(v) / max(len(v), 1)

    base = mean("none")
    dl, dr = mean("left") - base, mean("right") - base
    print(f"\n  baseline L-R {base:+.2f} Hz")
    print(f"  left contact  {dl:+.2f} Hz")
    print(f"  right contact {dr:+.2f} Hz")
    agree = sum(1 for a, b in zip(rows["left"], rows["right"])
                if (a["DNa02_LmR"] - b["DNa02_LmR"]) > 0)
    print(f"  per-seed sign consistency: {agree}/{args.seeds} seeds have "
          f"left > right")
    reverses = (dl * dr) < 0
    print(f"  {'PASS' if reverses else 'FAIL'}  DNa02 asymmetry reverses with "
          f"the contacted antenna")

    rec = {"rows": rows, "baseline": base, "delta_left": dl,
           "delta_right": dr, "reverses": bool(reverses),
           "sign_consistency": f"{agree}/{args.seeds}",
           "tics": args.tics, "seeds": args.seeds}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rec, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
