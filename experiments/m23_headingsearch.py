#!/usr/bin/env python3
"""M23 -- is heading represented ANYWHERE in this brain?

M19 asked one population whether it encodes heading, because EPG is where a fly
keeps its compass. It does not. That leaves an obvious question unasked: does
any OTHER population carry heading, and is the compass simply in the wrong
place in this model, or is the quantity absent altogether?

This asks every named cell type at once, with the same discipline M19 needed.

  fitted then tested   each cell's preferred heading comes from the first half
                       of an episode and is scored on the second, so nothing is
                       read off the data it was fitted to
  size matched         every type is compared against RANDOM cells drawn in the
                       same number, because in a fixed arena heading determines
                       what is on the retina and any visually driven cell
                       correlates with heading. Without this the answer is
                       "everything encodes heading", which is true and useless
  rolled null          the same decode against a shifted heading trace: what
                       fitting alone scores

The interesting outcome is not a long list of types that beat zero. It is
whether ANY type beats size-matched random cells by a margin that holds across
seeds -- that is, whether heading is represented anywhere better than it leaks
into everything.

    python experiments/m23_headingsearch.py --seeds 3 --min-cells 20
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


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M23 -- heading, anywhere")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tics", type=int, default=600)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device", default=os.environ.get("FLYDOOM_DEVICE", "cuda"))
    args = ap.parse_args()

    import torch
    from flydoom.agent import AgentConfig, FlyDoomAgent
    from flydoom.doom import DoomConfig
    from flydoom.motor import MotorConfig
    from flydoom.mechanosensation import MechanoConfig
    from m9_behaviour import RENDER
    from m19_compass import circ_corr

    def decode_from(rates, head, warm=100):
        """Fit preferred headings on the first half, score on the second."""
        r, h = rates[warm:], head[warm:]
        if len(r) < 100:
            return 0.0, 0.0
        half = len(r) // 2
        pref = np.angle((r[:half] * np.exp(1j * h[:half])[:, None]).sum(0))
        live = r[:half].sum(0) > 0
        if not live.any():
            return 0.0, 0.0
        dec = np.angle((r[half:][:, live]
                        * np.exp(1j * pref[live])[None, :]).sum(1))
        rolled = np.roll(h[half:], len(h[half:]) // 3)
        return circ_corr(dec, h[half:]), circ_corr(dec, rolled)

    per_type = {}
    for s in range(args.seeds):
        seed = 40 + s
        agent = FlyDoomAgent(AgentConfig(
            doom=DoomConfig(scenario="health_gathering_fly", window=False,
                            seed=seed, labels=True, **RENDER["fast"]),
            motor=MotorConfig(yaw_source="DNp15", fixed_turn_sign=True,
                              phasic_mdn=True, forward_gain=0.16),
            mechano=MechanoConfig(front_only=True),
            eye_map="anatomical", seed=seed, optic_gain=16.0,
            spiking_t4=True, touch=True, device=args.device))
        d = agent.ann.df
        pos = {int(r): i for i, r in enumerate(agent.graph.root_ids)}
        counts = (d.group_by("primary_type").len()
                  .filter(pl.col("len") >= args.min_cells))
        types = [t for t in counts["primary_type"].to_list() if t]
        idx = {}
        for t in types:
            f = d.filter(pl.col("primary_type") == t)
            v = np.array(sorted(pos[int(r)] for r in f["root_id"].to_list()
                                if int(r) in pos), dtype=np.int64)
            if len(v) >= args.min_cells:
                idx[t] = v
        if s == 0:
            print(f"{len(idx)} cell types with >= {args.min_cells} cells, "
                  f"{args.seeds} seeds, {args.tics} tics\n")

        agent.reset()
        allrates, head = [], []
        try:
            for t in range(args.tics):
                if agent.tic(t) is None:
                    break
                f = agent.motor._filt
                allrates.append(f.detach().cpu().numpy() if f is not None
                                else np.zeros(agent.net.n))
                head.append(np.radians(agent.doom.pose()[2]))
        finally:
            agent.close()
        A = np.asarray(allrates)
        H = np.asarray(head)
        rng = np.random.default_rng(100 + seed)
        for t, v in idx.items():
            r, null = decode_from(A[:, v], H)
            # size-matched random control, drawn fresh per type per seed
            rv = rng.choice(A.shape[1], size=len(v), replace=False)
            rr, _ = decode_from(A[:, rv], H)
            e = per_type.setdefault(t, {"n": len(v), "r": [], "null": [],
                                        "rand": []})
            e["r"].append(r); e["null"].append(null); e["rand"].append(rr)

    rows = []
    for t, e in per_type.items():
        r = float(np.mean(e["r"])); rand = float(np.mean(e["rand"]))
        rows.append((r - rand, r, rand, float(np.mean(e["null"])), e["n"], t))
    rows.sort(reverse=True)
    print(f"{'type':<14}{'cells':>6}{'heading r':>11}{'random':>9}"
          f"{'margin':>9}{'null':>8}")
    for margin, r, rand, null, n, t in rows[:args.top]:
        print(f"{t:<14}{n:>6}{r:>+11.3f}{rand:>+9.3f}{margin:>+9.3f}{null:>+8.3f}")
    print("\n... and the worst few, for the shape of the distribution")
    for margin, r, rand, null, n, t in rows[-3:]:
        print(f"{t:<14}{n:>6}{r:>+11.3f}{rand:>+9.3f}{margin:>+9.3f}{null:>+8.3f}")
    best = rows[0]
    print(f"\nbest margin over size-matched random: {best[5]} at {best[0]:+.3f}")
    print(f"median margin across {len(rows)} types: "
          f"{np.median([x[0] for x in rows]):+.3f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {t: {"n": e["n"], "r": float(np.mean(e["r"])),
                 "rand": float(np.mean(e["rand"])),
                 "null": float(np.mean(e["null"]))}
             for t, e in per_type.items()}, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
