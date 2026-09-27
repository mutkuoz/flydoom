#!/usr/bin/env python3
"""M20 -- where does the activity stop?

THE OBSERVATION THIS EXISTS TO TEST

Three circuits have now been asked to do their job and could not, and each time
the proximate cause was the same:

    M7   fixation   LC10a fires 0.00 Hz
    M4   looming    LPLC2 fires 0.00 Hz
    M19  compass    EPG   fires 0.44 Hz, against 149 Hz for random optic cells

Three circuit-specific stories were written for these. One measurement would be
better, and the hypothesis is simple enough to state in a sentence: activity in
this model concentrates in the optic lobe and does not arrive at the deep
populations that would have to do the computing.

If that is right it is a more useful description of the model's limit than any
of the three, because it predicts the next failure as well as explaining the
last three, and it points where the study already points -- at the per-cell-type
gains the connectome does not specify.

THE MEASUREMENT. One closed-loop episode in the configuration the behaviour
tables use. Every neuron's rate, accumulated per SIMULATION SUBSTEP rather than
per Doom tic, because sampling once a tic undercounts by up to the substep
count and would manufacture the very silence being tested for. Then the
distribution by super class, and by the named populations that failed.

    python experiments/m20_silence.py --tics 400
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

WATCH = ["LC10a", "LPLC2", "EPG", "LC4", "LC11", "T4a", "T5a", "Mi1", "L1",
         "DNa02", "DNp15", "DNp01", "BPN", "MDN", "PEN", "ER", "Delta7"]


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M20 -- the silence census")
    ap.add_argument("--tics", type=int, default=400)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--scenario", default="health_gathering_fly")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device", default=os.environ.get("FLYDOOM_DEVICE", "cuda"))
    args = ap.parse_args()

    import torch
    from flydoom.agent import AgentConfig, FlyDoomAgent
    from flydoom.doom import DoomConfig
    from flydoom.motor import MotorConfig
    from flydoom.mechanosensation import MechanoConfig
    from m9_behaviour import RENDER

    agent = FlyDoomAgent(AgentConfig(
        doom=DoomConfig(scenario=args.scenario, window=False, seed=args.seed,
                        labels=True, **RENDER["fast"]),
        motor=MotorConfig(yaw_source="DNp15", fixed_turn_sign=True,
                          phasic_mdn=True, forward_gain=0.16),
        mechano=MechanoConfig(front_only=True),
        eye_map="anatomical", seed=args.seed, optic_gain=16.0,
        spiking_t4=True, touch=True, device=args.device))
    agent.reset()

    n = agent.net.n
    total = torch.zeros(n, dtype=torch.float64, device=agent.net.device)
    steps = {"n": 0}
    orig = agent.net.step

    def counting(*a, **kw):
        out = orig(*a, **kw)
        total.add_(agent.net.out.to(torch.float64))
        steps["n"] += 1
        return out
    agent.net.step = counting
    try:
        for t in range(args.tics):
            if agent.tic(t) is None:
                break
    finally:
        agent.close()

    from flydoom import config as C
    hz = (total / max(steps["n"], 1) / C.DT).cpu().numpy()
    print(f"\n{steps['n']:,} substeps ({steps['n'] / 57:.0f} tics), "
          f"{n:,} neurons\n")

    d = agent.ann.df
    pos = {int(r): i for i, r in enumerate(agent.graph.root_ids)}
    rec = {"substeps": steps["n"], "by_class": {}, "by_type": {}}

    print(f"{'super class':<22}{'cells':>8}{'mean Hz':>10}{'median':>9}"
          f"{'% silent':>10}{'% over 1 Hz':>13}")
    for sc in ("sensory", "optic", "visual_projection", "central",
               "visual_centrifugal", "ascending", "descending", "motor"):
        ids = d.filter(pl.col("super_class") == sc)["root_id"].to_list()
        idx = np.array([pos[int(r)] for r in ids if int(r) in pos], dtype=np.int64)
        if not len(idx):
            continue
        v = hz[idx]
        rec["by_class"][sc] = {"n": int(len(v)), "mean": float(v.mean()),
                               "median": float(np.median(v)),
                               "silent": float((v < 0.01).mean()),
                               "over1": float((v > 1.0).mean())}
        print(f"{sc:<22}{len(v):>8,}{v.mean():10.2f}{np.median(v):9.2f}"
              f"{(v < 0.01).mean():10.1%}{(v > 1.0).mean():13.1%}")

    print(f"\n{'population':<12}{'cells':>7}{'mean Hz':>10}   what it failed at")
    why = {"LC10a": "fixation (M7)", "LPLC2": "looming (M4)",
           "EPG": "compass (M19)", "PEN": "compass (M19)",
           "ER": "compass, visual anchor", "Delta7": "compass, bump sharpening",
           "T4a": "direction selectivity (M3)", "T5a": "direction selectivity"}
    for t in WATCH:
        f = d.filter((pl.col("primary_type") == t)
                     | (pl.col("primary_type").str.starts_with(t)))
        ids = f["root_id"].to_list()
        idx = np.array([pos[int(r)] for r in ids if int(r) in pos], dtype=np.int64)
        if not len(idx):
            continue
        v = hz[idx]
        rec["by_type"][t] = {"n": int(len(v)), "mean": float(v.mean())}
        print(f"{t:<12}{len(v):>7}{v.mean():10.2f}   {why.get(t, '')}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rec, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
