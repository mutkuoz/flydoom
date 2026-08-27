#!/usr/bin/env python3
"""M3b — does the correlator's delayed arm carry a signal at all?

A correlator multiplies a signal by a DELAYED copy of a neighbouring signal.
That requires both arms to be modulated by the stimulus, and to be modulated
with a phase offset between them. If the delayed arm is flat, there is nothing
to correlate and no amount of rebalancing or gain can help -- you cannot
correlate against a constant.

This measures, during a drifting grating, for each arm of the T4a detector:

    modulation depth   how much the cell's rate swings with the stimulus
    phase              when in the stimulus cycle it peaks

A working correlator needs both arms modulated and separated in phase. The
hand-built control in this project reaches DSI -0.79 with two modulated,
phase-shifted inputs, so the machinery is capable; this asks whether the real
wiring delivers that input.

    python experiments/m3b_arms.py [--period 15] [--tf 4]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.cells import AnnotationTable  # noqa: E402
from flydoom.graph import ConnectomeGraph  # noqa: E402
from flydoom.lif import LIFNetwork  # noqa: E402
from flydoom.retina import Retina  # noqa: E402
import polars as pl  # noqa: E402
from m3_optomotor import GratingRig, _bias_vector, paint  # noqa: E402


def cells_of_type(graph, ann, name):
    """Indices of a cell type by name. The registry only knows the handles
    this project reads out; the correlator's arms are plain type names."""
    d = ann.df
    f = d.filter((pl.col('primary_type') == name)
                 | (pl.col('visual_type') == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f['root_id'].unique().to_list()
                     if int(x) in pos], dtype=np.int64)

ARMS = ["L1", "L2", "L3", "Mi1", "Tm3", "Mi9", "Mi4", "CT1",
        "Tm1", "Tm2", "Tm9", "T4a", "T4b", "T5a"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=15.0)
    ap.add_argument("--tf", type=float, default=4.0)
    ap.add_argument("--bias", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--site", default="L1+L2+L3")
    ap.add_argument("--optic-gain", type=float, default=1.0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    if args.optic_gain != 1.0:
        from flydoom.gains import optic_gain_multipliers
        g.signed_syn = (g.signed_syn
                        * optic_gain_multipliers(g, ann, args.optic_gain)
                        ).astype(np.float32)
    retina = Retina.build(g, ann, site=tuple(args.site.split("+")))
    graded = g.graded_mask(ann)
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=config.T_DLY_SLOW)
    net = LIFNetwork.from_graph(g, device=args.device, seed=0,
                                edge_delay=edge_delay, graded=graded)
    gext = _bias_vector(net, g, ann, args.bias, args.device)
    rig = GratingRig(net, retina, args.device, args.period, args.rate)

    idx = {t: cells_of_type(g, ann, t) for t in ARMS}
    idx = {t: torch.as_tensor(v, device=args.device)
           for t, v in idx.items() if len(v)}

    dt = config.DT
    n_steps = int(round(args.duration / dt))
    # PER CELL, not the population mean. A grating puts neighbouring columns at
    # different phases, so averaging a type's cells before measuring modulation
    # cancels the very signal being measured -- the population mean of a
    # spatially periodic stimulus is flat by construction.
    SAMPLE = 60
    sub = {k: v[:SAMPLE] for k, v in idx.items()}
    trace = {k: np.zeros((n_steps, len(v))) for k, v in sub.items()}
    net.reset()
    for step in range(n_steps):
        t = step * dt
        net.step(g_ext=gext,
                 out_set=rig.out_set(t, args.tf, +1,
                                     net.p.graded_max_rate, dt))
        for k, v in sub.items():
            trace[k][step] = net.out[v].detach().cpu().numpy() / dt

    print(paint("M3b — is each arm modulated by the stimulus?", "1"))
    print(paint("=" * 74, "90"))
    print(f"grating period {args.period:.0f} deg, {args.tf:.1f} Hz, "
          f"bias {args.bias:.1f} mV\n")
    print(f"  {'cell':>6} {'mean Hz':>9} {'per-cell depth':>15} "
          f"{'pop-mean depth':>15} {'phase spread':>13}")

    # Fourier component at the stimulus frequency: amplitude and phase.
    tt = np.arange(n_steps) * dt
    ref = np.exp(-2j * math.pi * args.tf * tt)
    skip = int(0.5 / dt)          # drop the onset transient
    for k in ARMS:
        if k not in trace:
            continue
        X = trace[k][skip:]                      # [time, cell]
        r = ref[skip:][:, None]
        comp = (X * r).mean(axis=0)              # per cell
        amp = 2.0 * np.abs(comp)
        mean = X.mean(axis=0)
        depth = amp / np.maximum(mean, 1e-9)
        phases = np.degrees(np.angle(comp)) % 360
        # circular spread of preferred phase across cells
        spread = 1.0 - np.abs(np.exp(1j * np.radians(phases)).mean())
        popmean = X.mean(axis=1)
        pamp = 2.0 * np.abs((popmean * ref[skip:]).mean())
        pdepth = pamp / max(popmean.mean(), 1e-9)
        print(f"  {k:>6} {mean.mean():9.2f} {np.median(depth):14.1%} "
              f"{pdepth:15.1%} {spread:13.2f}")

    print(paint("""
  A correlator needs BOTH arms modulated and separated in phase. Read the
  'depth' column: it is the fraction of each cell's own rate that swings with
  the stimulus. If the delayed (inhibitory) arm -- Mi9, Mi4, CT1 -- has depth
  near zero while the fast arm does not, then the delayed arm is a constant,
  there is nothing to correlate against, and neither rebalancing the arms nor
  changing their gains can produce direction selectivity. That would explain
  every negative result in this project with one measurement.""", "90"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
