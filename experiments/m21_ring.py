#!/usr/bin/env python3
"""M21 -- is the central complex silent for lack of drive, or structurally?

WHERE THIS COMES FROM

M20 localised this project's repeated failures to one junction. Vision leaves
the optic lobe without difficulty -- visual projection neurons are the most
active population in the model, LC11 at 365 Hz, LC4 at 242 -- and then stops.
ER, the ring neurons that carry visual input into the ellipsoid body, fire at
0.00 Hz; Delta7 at 0.00; EPG at 0.30. The compass (M19) is not a circuit that
computes the wrong thing, it is a circuit that is never driven.

The connectome says why. ER receives excitation 24,903 and inhibition 104,173,
a ratio of 4.18 to 1, and the inhibition is overwhelmingly ER onto ER: the
mutual inhibition that makes a ring attractor a ring attractor. Its excitatory
drive is TuBu, +14,560, which is the anterior optic tubercle relay -- the real
MeTu -> TuBu -> ER -> EPG pathway into the central complex. At a uniform
synaptic gain that ratio silences the population.

THE QUESTION, which is worth asking precisely because it has two useful
answers. Scale the excitatory drive onto ER and sweep it. If ER wakes and a
heading signal appears in EPG, the central complex here is starved rather than
broken, and this project's three failures share a curable cause. If ER wakes
and EPG still carries nothing, the silence was never the whole story.

THIS IS A MANIPULATION AND IS REPORTED AS ONE. It is a scalar on one pathway,
the same class of thing as the regional inhibitory scale the paper already
reports, and it is swept rather than fitted: the whole curve is printed, not
the best point on it. Two controls keep it honest:

  random   the same scalar on a random set of excitatory synapses of the same
           count, elsewhere in the brain. If that also produces a compass, the
           result is about adding drive and not about this junction.
  cost     T4/T5, DNa02 and DNp15 rates at every gain, because a manipulation
           that wakes the compass by deranging the rest of the brain has not
           bought anything.

    python experiments/m21_ring.py --gains 1 4 16 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

WATCH = ("ER", "EPG", "PEN", "Delta7", "TuBu", "T4a", "T5a", "LC4",
         "DNa02", "DNp15")


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M21 -- the ring junction")
    ap.add_argument("--gains", type=float, nargs="+",
                    default=[1.0, 4.0, 16.0, 64.0])
    ap.add_argument("--tics", type=int, default=400)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--target", nargs="+", default=["ER"],
                    help="which populations' excitatory input to scale. ER is "
                         "the visual gateway; EPG PEN is the recurrent loop "
                         "that has to sustain a bump.")
    ap.add_argument("--control", action="store_true",
                    help="scale LC4's excitatory input instead -- a "
                         "comparably sized visual-projection population that "
                         "is already firing. If a compass appears there too, "
                         "the result is about drive and not this junction.")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device", default=os.environ.get("FLYDOOM_DEVICE", "cuda"))
    args = ap.parse_args()

    import torch
    from flydoom.agent import AgentConfig, FlyDoomAgent
    from flydoom.doom import DoomConfig
    from flydoom.motor import MotorConfig
    from flydoom.mechanosensation import MechanoConfig
    from flydoom import config as C
    from m9_behaviour import RENDER
    from m19_compass import decode, circ_corr  # noqa: F401

    rec = {"gains": {}, "control": bool(args.control)}
    print("M21 -- scaling the excitatory drive onto the ring neurons\n")
    print(f"{'gain':>6}" + "".join(f"{t:>9}" for t in WATCH)
          + f"{'heading r':>11}{'null':>8}")

    for gain in args.gains:
        pw = () if gain == 1.0 else tuple(
            (t, gain) for t in (["LC4"] if args.control else args.target))
        agent = FlyDoomAgent(AgentConfig(
            doom=DoomConfig(scenario="health_gathering_fly", window=False,
                            seed=args.seed, labels=True, **RENDER["fast"]),
            motor=MotorConfig(yaw_source="DNp15", fixed_turn_sign=True,
                              phasic_mdn=True, forward_gain=0.16),
            mechano=MechanoConfig(front_only=True), pathway_gain=pw,
            eye_map="anatomical", seed=args.seed, optic_gain=16.0,
            spiking_t4=True, touch=True, device=args.device))

        d = agent.ann.df
        pos = {int(r): i for i, r in enumerate(agent.graph.root_ids)}

        def idx_of(prefix):
            f = d.filter(pl.col("primary_type").str.starts_with(prefix))
            return np.array(sorted(pos[int(r)] for r in f["root_id"].to_list()
                                   if int(r) in pos), dtype=np.int64)

        epg_i = idx_of("EPG")
        agent.reset()
        idx = {t: torch.as_tensor(idx_of(t), device=agent.net.device)
               for t in WATCH}
        epg_i = idx_of("EPG")
        tot = torch.zeros(agent.net.n, dtype=torch.float64,
                          device=agent.net.device)
        steps = {"n": 0}
        orig = agent.net.step

        def counting(*a, **kw):
            o = orig(*a, **kw)
            tot.add_(agent.net.out.to(torch.float64))
            steps["n"] += 1
            return o
        agent.net.step = counting

        rates_epg, head = [], []
        try:
            for t in range(args.tics):
                if agent.tic(t) is None:
                    break
                f = agent.motor._filt
                rates_epg.append(f[torch.as_tensor(epg_i, device=f.device)]
                                 .detach().cpu().numpy() if f is not None
                                 else np.zeros(len(epg_i)))
                head.append(np.radians(agent.doom.pose()[2]))
        finally:
            agent.close()

        hz = (tot / max(steps["n"], 1) / C.DT).cpu().numpy()
        dec = decode(np.asarray(rates_epg), np.asarray(head))
        row = f"{gain:>6.0f}"
        got = {}
        for t in WATCH:
            v = float(hz[idx_of(t)].mean()) if len(idx_of(t)) else 0.0
            got[t] = v
            row += f"{v:>9.2f}"
        row += f"{dec.get('test', 0.0):>+11.3f}{dec.get('shifted', 0.0):>+8.3f}"
        print(row)
        rec["gains"][str(gain)] = {"rates": got,
                                   "heading_r": dec.get("test", 0.0),
                                   "shifted": dec.get("shifted", 0.0)}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rec, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
