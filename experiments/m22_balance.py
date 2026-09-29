#!/usr/bin/env python3
"""M22 -- a bump is a ratio, not an amount. Can the ratio be set?

WHAT M21 ESTABLISHED, AND WHAT IT LEFT OPEN

M21 swept a scalar on one pathway over 256 fold, twice, and could not produce a
heading signal. Driving the visual gateway silenced the compass, because ER is
EPG's inhibition and turning up the brakes stops the car. Driving the recurrent
loop woke the whole central complex at once and saturated it: EPG at 383 Hz,
the refractory ceiling, every cell flat out.

The conclusion was that a bump is not an AMOUNT of activity but a
RELATIONSHIP -- activity somewhere, suppression everywhere else -- and that no
single number per pathway can create a relationship. That is a claim, and it
has an obvious test it has not been given: set the RATIO.

THE MANIPULATION. Excitation onto EPG and PEN scaled up by g_e, inhibition onto
EPG scaled by g_i, and the pair swept independently. Ring attractor theory says
what is wanted -- enough recurrent excitation to sustain a bump, enough
patterned inhibition to keep it from spreading -- so the shape of the
prediction is fixed before the run:

    a bump should appear over a RANGE of ratios and fail outside it, high and
    low, which is a signature no magnitude sweep can produce

If instead every ratio is either silent or saturated, the claim strengthens:
the balance a ring attractor needs is not reachable by any per-population
scalar, and the missing quantity really is per-cell-type.

READ-OUTS, and the bump one matters most.

    heading r   preferred directions fitted on half an episode, tested on the
                other half, against a rolled-heading null
    bump        NOT the population-vector sharpness M19 used, which reads 1.0
                both when the population is silent and when it is saturated
                and is therefore useless at exactly the two extremes this
                sweep visits. Here it is the fraction of EPG cells that are
                QUIET while the population as a whole is active -- a bump means
                most of the ring is off. Flat activity scores 0.

    python experiments/m22_balance.py --ge 4 16 64 --gi 0.25 1 4
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


FLOOR_HZ = 5.0      # below this the population is not doing anything


def bump_index(rates: np.ndarray) -> float:
    """Fraction of the ring quiet while the ring is MEANINGFULLY active.

    A bump is localised: a few cells loud, most silent, so the fraction below a
    fixed share of the peak is the natural measure. Saturation scores near
    zero, correctly.

    The floor is not optional and the first version of this function lacked it.
    Without one, a population where a single cell fires at 0.2 Hz has 50 of 51
    cells "quiet" and scores 0.98 -- the same degeneracy that made M19's
    population-vector sharpness read 1.000 for a silent ring. Any sharpness
    measure is meaningless when there is nothing to be sharp, so a population
    under FLOOR_HZ returns nan and is reported as such rather than as a bump.
    """
    if rates.size == 0 or rates.mean() < FLOOR_HZ:
        return float("nan")
    live = rates.sum(axis=1) > 1e-6
    if not live.any():
        return float("nan")
    r = rates[live]
    peak = r.max(axis=1, keepdims=True)
    quiet = (r < 0.2 * np.maximum(peak, 1e-9)).mean(axis=1)
    return float(quiet.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M22 -- E/I balance")
    ap.add_argument("--ge", type=float, nargs="+", default=[4.0, 16.0, 64.0],
                    help="excitation onto EPG and PEN")
    ap.add_argument("--gi", type=float, nargs="+", default=[0.25, 1.0, 4.0],
                    help="inhibition onto EPG")
    ap.add_argument("--tics", type=int, default=400)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device", default=os.environ.get("FLYDOOM_DEVICE", "cuda"))
    args = ap.parse_args()

    import torch
    from flydoom.agent import AgentConfig, FlyDoomAgent
    from flydoom.doom import DoomConfig
    from flydoom.motor import MotorConfig
    from flydoom.mechanosensation import MechanoConfig
    from m9_behaviour import RENDER
    from m19_compass import decode

    print("M22 -- sweeping the excitation/inhibition RATIO at the compass\n")
    print("  bump = fraction of the ring quiet while the ring is active.")
    print("  A localised bump scores high; silence and saturation both score "
          "low.\n")
    print(f"{'g_e':>6}{'g_i':>6}{'EPG Hz':>9}{'PEN Hz':>9}{'ER Hz':>8}"
          f"{'bump':>7}{'heading r':>11}{'null':>8}")
    rec = {}
    for ge in args.ge:
        for gi in args.gi:
            agent = FlyDoomAgent(AgentConfig(
                doom=DoomConfig(scenario="health_gathering_fly", window=False,
                                seed=args.seed, labels=True, **RENDER["fast"]),
                motor=MotorConfig(yaw_source="DNp15", fixed_turn_sign=True,
                                  phasic_mdn=True, forward_gain=0.16),
                mechano=MechanoConfig(front_only=True),
                pathway_gain=(("EPG", ge), ("PEN", ge)),
                inhib_gain=(("EPG", gi),),
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
            rates, head = [], []
            try:
                for t in range(args.tics):
                    if agent.tic(t) is None:
                        break
                    f = agent.motor._filt
                    rates.append(f[torch.as_tensor(epg_i, device=f.device)]
                                 .detach().cpu().numpy() if f is not None
                                 else np.zeros(len(epg_i)))
                    head.append(np.radians(agent.doom.pose()[2]))
                pen = float(agent.motor._filt[
                    torch.as_tensor(idx_of("PEN"), device=f.device)].mean())
                er = float(agent.motor._filt[
                    torch.as_tensor(idx_of("ER"), device=f.device)].mean())
            finally:
                agent.close()
            R = np.asarray(rates)
            dec = decode(R, np.asarray(head))
            b = bump_index(R[100:])
            print(f"{ge:>6.0f}{gi:>6.2f}{R[100:].mean():9.2f}{pen:9.2f}"
                  f"{er:8.2f}{b:7.3f}{dec.get('test', 0.0):>+11.3f}"
                  f"{dec.get('shifted', 0.0):>+8.3f}")
            rec[f"ge{ge}_gi{gi}"] = {"epg": float(R[100:].mean()), "pen": pen,
                                     "er": er, "bump": b,
                                     "heading_r": dec.get("test", 0.0),
                                     "shifted": dec.get("shifted", 0.0)}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rec, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
