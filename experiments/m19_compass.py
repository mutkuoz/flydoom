#!/usr/bin/env python3
"""M19 -- is there a heading signal in the central complex?

WHY THIS ONE

Asked how a fly finds its way, the honest answer is that it mostly does not do
it by seeing food. It stabilises its course on optic flow, moves in straight
bouts between saccades, avoids collisions by expansion, finds food by casting
in wind -- and it holds a direction with a COMPASS. The ellipsoid body carries
a ring attractor: a single bump of EPG activity whose position encodes heading,
updated by self-motion and anchored to visual landmarks. Walking a constant
angle to that bump is menotaxis, and it is how an insect travels in a straight
line across ground that gives it no other reference.

Every ingredient is in this connectome and none has ever been driven or read
here: 51 EPG, 42 PEN, 20 PEG, 278 ER ring neurons carrying visual input, and 42
Delta7, the inhibition that sharpens a bump into a single peak.

Two of this project's failures do not obviously apply. Direction selectivity
and looming are computations on temporal ORDER, which this model loses. A
compass is a persistent state, which is a different thing to ask of a network.

THE MEASUREMENT, and it is built to be hard to fool.

Per-neuron EPG rates and the true heading are logged over a closed-loop
episode. The first half estimates each cell's preferred heading as the circular
mean of headings weighted by its own rate. The SECOND half is then decoded from
those preferences by population vector and compared with the truth it never
saw. Fitting 51 preferred directions on 350 tics and testing on another 350 is
the whole guard against reading structure into noise.

CONTROLS, each of which should break it if the signal is real:

  shifted   the same rates against a circularly shifted heading trace. Every
            marginal statistic is preserved and only the pairing is destroyed,
            so this is what overfitting alone scores.
  shuffled  a degree-preserving shuffled connectome. If decoding survives this,
            it is not about the wiring.
  frozen    retina held on one frame. A fly's compass runs on self-motion too,
            so this need not collapse -- but the visual anchor is gone, and the
            bump should drift.

    python experiments/m19_compass.py --seeds 3
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


def circ_mean(angles_rad, weights=None):
    w = np.ones_like(angles_rad) if weights is None else np.asarray(weights)
    return np.angle(np.sum(w * np.exp(1j * angles_rad)))


def circ_corr(a, b):
    """Circular correlation between two angle series (Jammalamadaka)."""
    a, b = np.asarray(a), np.asarray(b)
    am, bm = circ_mean(a), circ_mean(b)
    sa, sb = np.sin(a - am), np.sin(b - bm)
    d = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    return float(np.sum(sa * sb) / d) if d > 0 else 0.0


def cells_of(graph, ann, prefix):
    d = ann.df
    f = d.filter(pl.col("primary_type").str.starts_with(prefix))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array(sorted(pos[int(r)] for r in f["root_id"].to_list()
                           if int(r) in pos), dtype=np.int64)


def run_episode(seed, tics, device, mirror=False, blind=False, shuffled=False,
                cells="EPG", scenario="health_gathering_fly"):
    """Returns (rates [tics, n_epg], heading [tics] in radians)."""
    from flydoom.agent import AgentConfig, FlyDoomAgent
    from flydoom.doom import DoomConfig
    from flydoom.motor import MotorConfig
    from flydoom.mechanosensation import MechanoConfig
    from m9_behaviour import RENDER

    agent = FlyDoomAgent(AgentConfig(
        doom=DoomConfig(scenario=scenario, window=False, seed=seed,
                        labels=True, **RENDER["fast"]),
        motor=MotorConfig(yaw_source="DNp15", fixed_turn_sign=True,
                          phasic_mdn=True, forward_gain=0.16),
        mechano=MechanoConfig(front_only=True),
        eye_map="anatomical", seed=seed, optic_gain=16.0, spiking_t4=True,
        touch=True, shuffle_graph=shuffled, device=device))
    if mirror:
        agent.vision.mirror()
    agent.reset()
    if blind:
        raw, first = agent.doom.frame, {}

        def frozen():
            if "f" not in first:
                f = raw()
                first["f"] = None if f is None else np.array(f, copy=True)
            return first["f"]
        agent.doom.frame = frozen

    epg = cells_of(agent.graph, agent.ann, "EPG")
    if cells != "EPG":
        # SIZE-MATCHED CONTROL, and the one this experiment turned out to need.
        # In a fixed arena heading determines what is on the retina, so ANY
        # visually driven cell correlates with heading and decoding one proves
        # nothing about a compass. The question is whether EPG does it BETTER
        # than an arbitrary population of the same size.
        rng = np.random.default_rng(1000 + seed)
        if cells == "optic":
            pool = np.flatnonzero(agent.graph.graded_mask(agent.ann)) \
                if hasattr(agent.graph, "graded_mask") else None
            if pool is None or len(pool) < len(epg):
                pool = np.arange(agent.graph.n_neurons)
        else:
            pool = np.arange(agent.graph.n_neurons)
        epg = rng.choice(pool, size=len(epg), replace=False)
    import torch
    idx = torch.as_tensor(epg, device=agent.net.device)
    rates, head = [], []
    try:
        for t in range(tics):
            if agent.tic(t) is None:
                break
            f = agent.motor._filt
            rates.append((f[idx].detach().cpu().numpy() if f is not None
                          else np.zeros(len(epg))))
            head.append(np.radians(agent.doom.pose()[2]))
    finally:
        agent.close()
    return np.asarray(rates), np.asarray(head), epg


def decode(rates, head, warm=100):
    """Fit preferred headings on the first half, decode the second."""
    r, h = rates[warm:], head[warm:]
    if len(r) < 100:
        return {}
    half = len(r) // 2
    tr_r, tr_h, te_r, te_h = r[:half], h[:half], r[half:], h[half:]
    # preferred heading of each cell, from the training half only
    num = (tr_r * np.exp(1j * tr_h)[:, None]).sum(axis=0)
    pref = np.angle(num)
    live = tr_r.sum(axis=0) > 0
    out = {}
    for tag, (rr, hh) in (("test", (te_r, te_h)),):
        if not live.any():
            out[tag] = 0.0
            continue
        dec = np.angle((rr[:, live] * np.exp(1j * pref[live])[None, :]).sum(1))
        out[tag] = circ_corr(dec, hh)
    # the null: same rates, heading rolled so only the pairing dies
    rolled = np.roll(te_h, len(te_h) // 3)
    dec = np.angle((te_r[:, live] * np.exp(1j * pref[live])[None, :]).sum(1))
    out["shifted"] = circ_corr(dec, rolled)
    out["n_live"] = int(live.sum())
    out["mean_rate"] = float(r.mean())
    # is the population a BUMP or a blur? ratio of the vector length to the sum
    vec = np.abs((te_r * np.exp(1j * pref)[None, :]).sum(1))
    out["bump"] = float(np.mean(vec / np.maximum(te_r.sum(1), 1e-9)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M19 -- the compass")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tics", type=int, default=700)
    ap.add_argument("--arms", nargs="+",
                    default=["intact", "rand_optic", "rand_any",
                             "frozen", "shuffled"])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device", default=os.environ.get("FLYDOOM_DEVICE", "cuda"))
    args = ap.parse_args()

    ARMS = {"intact": {}, "mirrored": {"mirror": True},
            "frozen": {"blind": True}, "shuffled": {"shuffled": True},
            "rand_optic": {"cells": "optic"}, "rand_any": {"cells": "any"}}
    print("M19 -- is heading decodable from EPG?\n")
    print("  preferred headings fitted on the first half of each episode and")
    print("  tested on the second. `shifted` is the same decode against a")
    print("  rolled heading trace: that is what fitting alone scores.\n")
    print(f"{'arm':<12}{'seeds':>6}{'circ corr (held out)':>24}"
          f"{'shifted null':>16}{'bump':>8}{'Hz':>9}")
    record = {}
    for arm in args.arms:
        rows = []
        for s in range(args.seeds):
            rates, head, epg = run_episode(40 + s, args.tics, args.device,
                                           **ARMS[arm])
            d = decode(rates, head)
            if d:
                rows.append(d)
        if not rows:
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))       # noqa: E731
        sd = lambda k: float(np.std([r[k] for r in rows]))       # noqa: E731
        record[arm] = {"n": len(rows), "test": m("test"), "shifted": m("shifted"),
                       "bump": m("bump"), "mean_rate": m("mean_rate"),
                       "test_sd": sd("test")}
        print(f"{arm:<12}{len(rows):>6}{m('test'):+14.3f} +-{sd('test'):5.3f}"
              f"{m('shifted'):+16.3f}{m('bump'):8.3f}{m('mean_rate'):9.2f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
