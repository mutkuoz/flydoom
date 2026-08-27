#!/usr/bin/env python3
"""M3e — is the direction information present in the arms themselves?

Ten structural explanations for the motion-vision failure have been tested and
eliminated: pooling, arm imbalance (twice), input dilution, unmodulated arms,
CT1 saturation, absent phase offset, smeared sampling, same-column sampling,
opposing-flank cancellation, and graded-versus-spiking output. The wiring is
textbook -- coherent point-sampled arms one column apart on opposite flanks,
both strongly modulated, correct delays.

So this stops testing components and isolates the question. It records the
REAL Mi1 and Mi9 partners of real T4a cells during a grating in both
directions, then drives a single isolated LIF cell with ONLY those two inputs,
using their real synapse counts and real delays.

    high DSI  -> the two arms DO carry direction information, and something in
                 the rest of T4a's input destroys it downstream.
    zero DSI  -> the arms as measured carry no direction information despite
                 their phase difference, and the problem is upstream of T4.

Either answer localises the failure, which testing components one at a time
has not.

    python experiments/m3e_isolate.py
"""
from __future__ import annotations

import argparse
import os
import collections
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
from flydoom.lif import LIFNetwork, LIFParams  # noqa: E402
from flydoom.retina import Retina  # noqa: E402
from m3_optomotor import GratingRig, _bias_vector, paint  # noqa: E402


def _replay(pairs, rec, wcol, gain, fast_d, slow_d, dt, n, device, duration):
    """Drive one isolated LIF cell per T4a with only its two real arms."""
    m = len(pairs)
    p = LIFParams(dt=dt)
    pre = torch.arange(2 * m, dtype=torch.long)
    post = torch.cat([torch.arange(2 * m, 3 * m), torch.arange(2 * m, 3 * m)])
    w = torch.tensor([pr[1][1] for pr in pairs] + [pr[2][1] for pr in pairs],
                     dtype=torch.float32) * gain
    delay = np.array([fast_d] * m + [slow_d] * m, dtype=np.int64)
    iso = LIFNetwork(3 * m, pre, post, w, p, device, 0, edge_delay=delay)
    fi = [wcol[pr[1][0]] for pr in pairs]
    si = [wcol[pr[2][0]] for pr in pairs]
    out = {}
    for direction in (+1, -1):
        iso.reset()
        cnt = np.zeros(m)
        src = rec[direction]
        for step in range(n):
            os_ = torch.full((3 * m,), -1.0, device=device)
            os_[:m] = torch.as_tensor(src[step][fi], device=device)
            os_[m:2 * m] = torch.as_tensor(src[step][si], device=device)
            iso.step(out_set=os_)
            cnt += iso.out[2 * m:].detach().cpu().numpy()
        out[direction] = cnt / duration
    return out[+1], out[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=15.0)
    ap.add_argument("--tf", type=float, default=4.0)
    ap.add_argument("--bias", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=2.5)
    ap.add_argument("--n-cells", type=int, default=200)
    ap.add_argument("--gain", type=float, default=0.0,
                    help="scale the two real weights so the isolated cell "
                         "reaches a firing regime comparable to the real T4a. "
                         "0 = sweep automatically. This is a readout "
                         "calibration, not a model change: two synapses cannot "
                         "reach threshold alone, and a silent cell answers "
                         "nothing. The RATIO between the arms is preserved "
                         "exactly, which is what the question is about.")
    ap.add_argument("--device",
                    default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
    args = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    d = ann.df
    pos = {int(r): i for i, r in enumerate(g.root_ids)}

    def cells(name):
        f = d.filter((pl.col("primary_type") == name)
                     | (pl.col("visual_type") == name))
        return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                         if int(x) in pos], dtype=np.int64)

    T4A, MI1, MI9 = cells("T4a"), set(cells("Mi1").tolist()), set(cells("Mi9").tolist())
    part = collections.defaultdict(lambda: {"Mi1": [], "Mi9": []})
    t4set = set(T4A.tolist())
    for p_, q_, w in zip(g.pre_idx, g.post_idx, g.signed_syn):
        q_ = int(q_)
        if q_ not in t4set:
            continue
        p_ = int(p_)
        if p_ in MI1:
            part[q_]["Mi1"].append((p_, float(w)))
        elif p_ in MI9:
            part[q_]["Mi9"].append((p_, float(w)))

    pairs = [(t, max(v["Mi1"], key=lambda x: abs(x[1])),
              max(v["Mi9"], key=lambda x: abs(x[1])))
             for t, v in part.items() if v["Mi1"] and v["Mi9"]][:args.n_cells]
    if not pairs:
        print("no T4a cell has both arms")
        return 1

    watch = sorted({p[1][0] for p in pairs} | {p[2][0] for p in pairs})
    wt = torch.as_tensor(np.array(watch), device=args.device)
    wcol = {v: i for i, v in enumerate(watch)}

    # --- record the real arms, both directions
    retina = Retina.build(g, ann, site=("L1", "L2", "L3"))
    graded = g.graded_mask(ann)
    ed = g.edge_delay_steps(ann, config.DT, t_slow=config.T_DLY_SLOW)
    net = LIFNetwork.from_graph(g, device=args.device, seed=0,
                                edge_delay=ed, graded=graded)
    gext = _bias_vector(net, g, ann, args.bias, args.device)
    rig = GratingRig(net, retina, args.device, args.period, 150.0)

    dt = config.DT
    n = int(round(args.duration / dt))
    rec = {}
    for direction in (+1, -1):
        net.reset()
        buf = np.zeros((n, len(watch)), dtype=np.float32)
        for step in range(n):
            net.step(g_ext=gext,
                     out_set=rig.out_set(step * dt, args.tf, direction,
                                         net.p.graded_max_rate, dt))
            buf[step] = net.out[wt].detach().cpu().numpy()
        rec[direction] = buf

    # --- replay into isolated cells fed ONLY by their two real arms
    def run_iso(gain):
        return _iso(pairs, rec, wcol, m_pairs=len(pairs), gain=gain,
                    dt=dt, n=n, device=args.device, duration=args.duration)

    fast_d = max(1, int(round(config.T_DLY / dt)))
    slow_d = max(1, int(round(config.T_DLY_SLOW / dt)))
    gains = [args.gain] if args.gain > 0 else [1, 3, 10, 30, 100, 300]
    best = None
    print(paint("M3e — direction information in the arms alone", "1"))
    print(paint("=" * 74, "90"))
    print(f"{len(pairs)} T4a cells, each driven ONLY by its own strongest Mi1 "
          f"and Mi9,\nreal synapse counts, real delays "
          f"({config.T_DLY*1e3:.0f} / {config.T_DLY_SLOW*1e3:.0f} ms)\n")
    print(f"  {'gain':>6} {'live':>6} {'mean Hz':>9} {'mean |DSI|':>11} "
          f"{'>0.1':>7} {'>0.5':>7} {'signed':>9}")
    for gain in gains:
        a, b = _replay(pairs, rec, wcol, gain, fast_d, slow_d, dt, n,
                       args.device, args.duration)
        tot = a + b
        live = tot > 1.0
        if not live.any():
            print(f"  {gain:6.0f} {0:6d} {0.0:9.2f}        (silent)")
            continue
        dsi = np.where(tot > 1e-9, (a - b) / np.maximum(tot, 1e-9), 0.0)
        dl = np.abs(dsi[live])
        print(f"  {gain:6.0f} {int(live.sum()):6d} {tot[live].mean()/2:9.2f} "
              f"{dl.mean():11.4f} {(dl>0.1).mean():6.1%} {(dl>0.5).mean():6.1%} "
              f"{dsi[live].mean():+9.4f}")
        if best is None or abs(tot[live].mean()/2 - 22.0) < best[0]:
            best = (abs(tot[live].mean()/2 - 22.0), gain, dl, dsi[live])
    if best:
        _, gain, dl, sd = best
        print(f"\n  at the gain matching the real T4a rate (~22 Hz), gain={gain:.0f}:")
        print(f"    mean |DSI| {dl.mean():.4f}   median {np.median(dl):.4f}   "
              f"max {dl.max():.4f}")
        print(f"    fraction >0.1 {(dl>0.1).mean():.1%}   "
              f">0.5 {(dl>0.5).mean():.1%}   signed mean {sd.mean():+.4f}")

    print(paint("""
  Compare against the whole-network T4a, which reads |DSI| ~ 0.002. If these
  isolated cells are selective, the two arms carry direction information and it
  is destroyed downstream by the rest of T4a's input. If they are not, the
  information was never in the arms, and every downstream fix was doomed.""",
                "90"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
