#!/usr/bin/env python3
"""M3i — is the fast arm rectified against its floor?

Step 0 (in-network ablation) removed the last competing-input explanation:
stripping T4a/T4b to nothing but Mi1 and Mi9 does not raise direction
selectivity, and neither does adding any single partner back. The failure is
not in WHICH inputs T4 receives. What is left is the arms themselves.

A graded unit emits

    out = clamp((v - v_rest) / threshold_distance, 0, 1) * graded_max_rate

so its output is CLIPPED at both ends. Mi1 is held down by glutamatergic L1
(-77.3 synapses, inhibitory) and sits at 2-9 Hz against a 200 Hz ceiling. If
its activation spends most of the cycle pinned at the lower clamp, then the
excitatory arm delivers a rectified, half-wave-shaped signal whose ABSOLUTE
modulation is tiny however large its RELATIVE depth looks -- and a correlator
multiplies absolute quantities, not percentages. That would reconcile two
measurements that otherwise look contradictory: M3b found Mi1 modulating at
125% of its own mean, while the realised conductances differ 37x.

Reported per arm:
  mean Hz          the operating level
  floor%           fraction of timesteps within 1% of the lower clamp
  ceil%            fraction within 1% of the upper clamp
  F1 abs           modulation amplitude at the stimulus frequency, in Hz
  F1 rel           the same as a fraction of the arm's own mean

Read F1 ABS across arms, not F1 rel. Relative depth is what a rectified arm
inflates.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    """Direct type lookup: these are optic-lobe types, not registry handles."""
    d = ann.df
    f = d.filter((pl.col("primary_type") == name)
                 | (pl.col("visual_type") == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                     if int(x) in pos], dtype=np.int64)

ARMS = ("Mi1", "Tm3", "Mi9", "Mi4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=15.0)
    ap.add_argument("--tf", type=float, default=4.0)
    ap.add_argument("--biases", default="1.0,2.0,4.0,7.5")
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--site", default="L1+L2+L3")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    retina = Retina.build(g, ann, site=tuple(args.site.split("+")))
    graded = g.graded_mask(ann)
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=config.T_DLY_SLOW)
    net = LIFNetwork.from_graph(g, device=args.device, seed=0,
                                edge_delay=edge_delay, graded=graded)
    rig = GratingRig(net, retina, args.device, args.period, args.rate)

    idx = {t: cells_of_type(g, ann, t) for t in ARMS}
    idx = {t: torch.as_tensor(v[:60], device=args.device)
           for t, v in idx.items() if len(v)}

    dt = config.DT
    n_steps = int(round(args.duration / dt))
    ceiling = net.p.graded_max_rate
    tt = np.arange(n_steps) * dt
    ref = np.exp(-2j * math.pi * args.tf * tt)
    skip = int(0.5 / dt)

    print(paint("M3i — is the fast arm rectified against its floor?", "1"))
    print(paint("=" * 78, "90"))
    print(f"grating {args.period:.0f} deg, {args.tf:.1f} Hz; graded ceiling "
          f"{ceiling:.0f} Hz; {n_steps - skip} samples per cell\n")

    record = {"period": args.period, "tf": args.tf, "ceiling_hz": ceiling,
              "points": []}
    for bias in [float(b) for b in args.biases.split(",")]:
        gext = _bias_vector(net, g, ann, bias, args.device)
        trace = {k: np.zeros((n_steps, len(v))) for k, v in idx.items()}
        net.reset()
        for step in range(n_steps):
            net.step(g_ext=gext,
                     out_set=rig.out_set(step * dt, args.tf, +1, ceiling, dt))
            for k, v in idx.items():
                trace[k][step] = net.out[v].detach().cpu().numpy() / dt

        print(paint(f"  bias {bias:.1f} mV", "1;36"))
        print(f"  {'arm':>5} {'mean Hz':>9} {'floor%':>8} {'ceil%':>7}"
              f" {'F1 abs Hz':>11} {'F1 rel':>9}")
        for k in ARMS:
            if k not in trace:
                continue
            X = trace[k][skip:]
            mean = X.mean()
            floor = float((X < 0.01 * ceiling).mean())
            ceil = float((X > 0.99 * ceiling).mean())
            comp = (X * ref[skip:][:, None]).mean(axis=0)
            f1 = float(np.median(2.0 * np.abs(comp)))
            rel = f1 / max(mean, 1e-9)
            print(f"  {k:>5} {mean:9.2f} {floor:8.1%} {ceil:7.1%}"
                  f" {f1:11.3f} {rel:9.1%}")
            record["points"].append(
                {"bias_mv": bias, "arm": k, "mean_hz": mean,
                 "floor_frac": floor, "ceil_frac": ceil,
                 "f1_abs_hz": f1, "f1_rel": rel})
        print()

    print(paint("""  A correlator multiplies the two arms. What it can use is F1 ABS -- the
  amplitude in Hz that actually arrives. If Mi1 sits at a high floor% with an
  F1 abs far below Mi9's, the excitatory arm is rectified: its relative depth
  is an artefact of dividing a small modulation by a smaller mean.""", "90"))

    if args.json:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
