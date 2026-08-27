#!/usr/bin/env python3
"""M3d — is each correlator arm a POINT sample, or a blur?

A Reichardt correlator multiplies a signal from one point by a delayed signal
from a NEIGHBOURING point. Both arms must sample narrowly. In the fly, one T4
cell takes Mi1 from its own medulla column and Mi9 from one neighbour.

If our T4a instead receives Mi1 from many columns spanning the grating cycle,
that arm's summed input averages toward a constant -- and a constant carries no
phase. Two smeared arms cannot correlate however well balanced, delayed or
gain-matched they are. Every intervention tried so far (depression, per-type
gains, CT1 ablation) acts on amplitude, and none of them can un-smear a
spatial average.

Measured per T4a cell, for each arm:
    fan-in          how many distinct presynaptic cells
    column spread   how many distinct ommatidial columns they occupy
    coherence       |mean unit phasor| of those inputs' responses.
                    1.0 = all in phase (a point sample). 0.0 = spread over the
                    whole cycle (a constant, no phase information).

    python experiments/m3d_fanin.py
"""
from __future__ import annotations

import argparse
import os
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.cells import AnnotationTable  # noqa: E402
from flydoom.graph import ConnectomeGraph  # noqa: E402
from flydoom.lif import LIFNetwork  # noqa: E402
from flydoom.retina import Retina  # noqa: E402
from m3_optomotor import GratingRig, _bias_vector, paint  # noqa: E402


def cells_of_type(graph, ann, name):
    d = ann.df
    f = d.filter((pl.col("primary_type") == name)
                 | (pl.col("visual_type") == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                     if int(x) in pos], dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=15.0)
    ap.add_argument("--tf", type=float, default=4.0)
    ap.add_argument("--bias", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=1.5)
    ap.add_argument("--device",
                    default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
    args = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    retina = Retina.build(g, ann, site=("L1", "L2", "L3"))
    graded = g.graded_mask(ann)
    ed = g.edge_delay_steps(ann, config.DT, t_slow=config.T_DLY_SLOW)
    net = LIFNetwork.from_graph(g, device=args.device, seed=0,
                                edge_delay=ed, graded=graded)
    gext = _bias_vector(net, g, ann, args.bias, args.device)
    rig = GratingRig(net, retina, args.device, args.period, 150.0)

    ARMS = {"Mi1": "fast exc", "Tm3": "fast exc",
            "Mi9": "slow inh", "Mi4": "slow inh"}
    tgt = cells_of_type(g, ann, "T4a")
    arm_cells = {k: cells_of_type(g, ann, k) for k in ARMS}
    watch = np.unique(np.concatenate(list(arm_cells.values())))
    wt = torch.as_tensor(watch, device=args.device)
    col_of = {int(v): i for i, v in enumerate(watch)}

    dt = config.DT
    n = int(round(args.duration / dt))
    trace = np.zeros((n, len(watch)), dtype=np.float32)
    net.reset()
    for step in range(n):
        net.step(g_ext=gext,
                 out_set=rig.out_set(step * dt, args.tf, +1,
                                     net.p.graded_max_rate, dt))
        trace[step] = net.out[wt].detach().cpu().numpy()

    skip = int(0.4 / dt)
    tt = np.arange(n)[skip:] * dt
    ref = np.exp(-2j * math.pi * args.tf * tt)[:, None]
    comp = (trace[skip:] * ref).mean(axis=0)
    unit = comp / np.maximum(np.abs(comp), 1e-12)      # phase only

    # which ommatidial column each watched cell belongs to
    ca = pl.read_csv(Path(config.RAW_DIR) / "column_assignment.csv.gz")
    rid2col = {int(r): int(c) for r, c in zip(ca["root_id"].to_list(),
                                              ca["column_id"].to_list())}
    idx2col = {}
    for i, rid in enumerate(g.root_ids):
        c = rid2col.get(int(rid))
        if c is not None:
            idx2col[i] = c

    pre, post, syn = g.pre_idx, g.post_idx, np.abs(g.signed_syn)
    tgtset = set(tgt.tolist())
    per = {int(t): {k: {"cells": [], "w": []} for k in ARMS} for t in tgt}
    setmap = {k: set(v.tolist()) for k, v in arm_cells.items()}
    for p_, q_, w_ in zip(pre, post, syn):
        q_ = int(q_)
        if q_ not in tgtset:
            continue
        p_ = int(p_)
        for k, s in setmap.items():
            if p_ in s:
                per[q_][k]["cells"].append(p_)
                per[q_][k]["w"].append(float(w_))
                break

    print(paint("M3d — is each correlator arm a point sample or a blur?", "1"))
    print(paint("=" * 74, "90"))
    print(f"grating period {args.period:.0f} deg, {args.tf:.0f} Hz\n")
    print(f"  {'arm':>5} {'role':>9} {'cells w/ arm':>13} {'fan-in':>8} "
          f"{'columns':>8} {'coherence':>10}")
    for k, role in ARMS.items():
        fan, cols, coh = [], [], []
        for t in tgt:
            e = per[int(t)][k]
            if not e["cells"]:
                continue
            fan.append(len(e["cells"]))
            cols.append(len({idx2col.get(c, -1) for c in e["cells"]}))
            w = np.asarray(e["w"])
            z = np.array([unit[col_of[c]] for c in e["cells"]])
            coh.append(abs((w * z).sum() / max(w.sum(), 1e-12)))
        if not fan:
            continue
        print(f"  {k:>5} {role:>9} {len(fan):13d} {np.median(fan):8.1f} "
              f"{np.median(cols):8.1f} {np.median(coh):10.3f}")

    print(paint("""
  COHERENCE is the number that matters. A Reichardt arm must sample ONE point:
  coherence near 1 means all of that arm's inputs arrive in phase, so the arm
  carries a clean phase the other arm can be correlated against. Coherence near
  0 means the arm's inputs are spread across the stimulus cycle and sum to a
  constant -- and a constant has no phase, so there is nothing to correlate,
  whatever the delay or the gains.

  Note what this would explain that amplitude fixes cannot: depression,
  per-type gains and CT1 ablation all rescale inputs, and no rescaling can
  recover phase that spatial averaging has already destroyed.""", "90"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
