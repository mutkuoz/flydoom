#!/usr/bin/env python3
"""M3f — which input destroys the direction signal?

M3e established the key fact: a T4a cell driven by ONLY its own real Mi1 and
Mi9 -- real synapse counts, real delays -- reaches mean |DSI| 0.30 with 20% of
cells above the experimental selection threshold. The same cells in the whole
network read 0.002. The correlator works; something in the rest of T4a's input
destroys it, a ~150x loss.

This adds the other inputs back one at a time to find which one does it. Same
isolated-replay method as M3e, same gain, only the input set changes.

    python experiments/m3f_addback.py
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

# Which input types to test, and whether each is fast or slow-delayed.
SLOW = {"Mi9", "Mi4", "CT1", "Tm9"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=15.0)
    ap.add_argument("--tf", type=float, default=4.0)
    ap.add_argument("--bias", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=2.5)
    ap.add_argument("--n-cells", type=int, default=200)
    ap.add_argument("--gain", type=float, default=10.0)
    ap.add_argument("--device",
                    default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
    args = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    d = ann.df
    pos = {int(r): i for i, r in enumerate(g.root_ids)}
    tname = np.full(g.n_neurons, "", dtype=object)
    for r, a, b in zip(d["root_id"].to_list(), d["primary_type"].to_list(),
                       d["visual_type"].to_list()):
        i = pos.get(int(r))
        if i is not None:
            tname[i] = a or b or ""

    t4 = np.array([i for i in range(g.n_neurons) if tname[i] == "T4a"])
    t4set = set(t4.tolist())

    # every presynaptic input to each T4a, grouped by source type
    inputs = collections.defaultdict(lambda: collections.defaultdict(list))
    for p_, q_, w in zip(g.pre_idx, g.post_idx, g.signed_syn):
        q_ = int(q_)
        if q_ in t4set:
            inputs[q_][tname[int(p_)] or "?"].append((int(p_), float(w)))

    have = [t for t in t4 if inputs[int(t)].get("Mi1") and inputs[int(t)].get("Mi9")]
    cells = have[:args.n_cells]
    if not len(cells):
        print("no cell has both arms")
        return 1

    # record every presynaptic partner of the chosen cells, both directions
    watch = sorted({p for t in cells for lst in inputs[int(t)].values()
                    for p, _ in lst})
    wcol = {v: i for i, v in enumerate(watch)}
    wt = torch.as_tensor(np.array(watch), device=args.device)

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

    fast_d = max(1, int(round(config.T_DLY / dt)))
    slow_d = max(1, int(round(config.T_DLY_SLOW / dt)))

    def replay(include, graded_target=False, gain=None, bias_mv=0.0):
        """include: set of type names, or None for ALL inputs.

        graded_target: make the isolated T4a a GRADED unit instead of spiking.
        This is the one difference between the isolated replay and the network:
        a graded unit's output is rectified-LINEAR, and the mean of a linear
        function of summed sinusoids does not depend on their relative phase.
        """
        pre_l, post_l, w_l, dly_l = [], [], [], []
        src_idx = []
        m = len(cells)
        for k, t in enumerate(cells):
            for typ, lst in inputs[int(t)].items():
                if include is not None and typ not in include:
                    continue
                for p_, w_ in lst:
                    src_idx.append(wcol[p_])
                    pre_l.append(len(src_idx) - 1)
                    post_l.append(k)
                    w_l.append(w_ * (args.gain if gain is None else gain))
                    dly_l.append(slow_d if typ in SLOW else fast_d)
        ns = len(src_idx)
        if ns == 0:
            return None, None
        pre = torch.tensor(pre_l, dtype=torch.long)
        post = torch.tensor([p + ns for p in post_l], dtype=torch.long)
        w = torch.tensor(w_l, dtype=torch.float32)
        gm = None
        if graded_target:
            gm = np.zeros(ns + m, dtype=bool)
            gm[ns:] = True            # only the target is graded
        iso = LIFNetwork(ns + m, pre, post, w, LIFParams(dt=dt), args.device, 0,
                         edge_delay=np.array(dly_l, dtype=np.int64), graded=gm)
        out = {}
        for direction in (+1, -1):
            iso.reset()
            cnt = np.zeros(m)
            src = rec[direction]
            ge = None
            if bias_mv:
                ge = torch.zeros(ns + m, dtype=torch.float32,
                                 device=args.device)
                ge[ns:] = bias_mv * 1e-3
            skip_steps = int(0.5 / dt)   # whole cycles only; see the
            #                                 windowing note in main()
            for step in range(n):
                os_ = torch.full((ns + m,), -1.0, device=args.device)
                os_[:ns] = torch.as_tensor(src[step][src_idx],
                                           device=args.device)
                iso.step(out_set=os_, g_ext=ge)
                if step >= skip_steps:
                    cnt += iso.out[ns:].detach().cpu().numpy()
            out[direction] = cnt / (args.duration - 0.5)
        return out[+1], out[-1]

    print(paint("M3f — which input destroys the direction signal?", "1"))
    print(paint("=" * 78, "90"))
    print(f"{len(cells)} T4a cells, isolated replay, gain {args.gain:.0f}, "
          f"period {args.period:.0f} deg, {args.tf:.0f} Hz\n")
    print(f"  {'inputs included':<34} {'live':>5} {'Hz':>7} {'mean|DSI|':>10} "
          f"{'>0.1':>7} {'>0.5':>7}")
    CASES = [
        ("Mi1 + Mi9  (the correlator)", {"Mi1", "Mi9"}),
        ("  + Tm3   (2nd fast exc)", {"Mi1", "Mi9", "Tm3"}),
        ("  + Mi4   (2nd slow inh)", {"Mi1", "Mi9", "Mi4"}),
        ("  + CT1   (saturated inh)", {"Mi1", "Mi9", "CT1"}),
        ("all five correlator types", {"Mi1", "Mi9", "Tm3", "Mi4", "CT1"}),
        ("EVERY input T4a receives", None),
    ]
    for label, inc in CASES:
        a, b = replay(inc)
        if a is None:
            print(f"  {label:<34} (no inputs)")
            continue
        tot = a + b
        live = tot > 1.0
        if not live.any():
            print(f"  {label:<34} {0:5d}  (silent)")
            continue
        dsi = np.where(tot > 1e-9, (a - b) / np.maximum(tot, 1e-9), 0.0)
        dl = np.abs(dsi[live])
        print(f"  {label:<34} {int(live.sum()):5d} {tot[live].mean()/2:7.1f} "
              f"{dl.mean():10.4f} {(dl>0.1).mean():6.1%} {(dl>0.5).mean():6.1%}")

    print(f"\n  THE ONE REMAINING DIFFERENCE: spiking vs graded target cell")
    print(f"  {'target type':<34} {'live':>5} {'Hz':>7} {'mean|DSI|':>10} "
          f"{'>0.1':>7} {'>0.5':>7}")
    for label, gt in (("spiking (as in the isolated test)", False),
                      ("GRADED (as in the network)", True)):
        a, b = replay(None, graded_target=gt)
        tot = a + b
        live = tot > 1.0
        if not live.any():
            print(f"  {label:<34} {0:5d}  (silent)"); continue
        dsi = np.where(tot > 1e-9, (a - b) / np.maximum(tot, 1e-9), 0.0)
        dl = np.abs(dsi[live])
        print(f"  {label:<34} {int(live.sum()):5d} {tot[live].mean()/2:7.1f} "
              f"{dl.mean():10.4f} {(dl>0.1).mean():6.1%} {(dl>0.5).mean():6.1%}")

    print(f"\n  HOW THE CELL REACHES THRESHOLD: input gain vs tonic bias")
    print(f"  {'route to ~20-50 Hz':<34} {'live':>5} {'Hz':>7} {'mean|DSI|':>10} "
          f"{'>0.1':>7} {'>0.5':>7}")
    for label, gain, bias, gt in (
            ("input gain x10, no bias", 10.0, 0.0, True),
            ("real weights + bias 2 mV", 1.0, 2.0, True),
            ("real weights + bias 4 mV", 1.0, 4.0, True),
            ("real weights + bias 7 mV", 1.0, 7.0, True)):
        a, b = replay(None, graded_target=gt, gain=gain, bias_mv=bias)
        tot = a + b
        live = tot > 1.0
        if not live.any():
            print(f"  {label:<34} {0:5d}  (silent)"); continue
        dsi = np.where(tot > 1e-9, (a - b) / np.maximum(tot, 1e-9), 0.0)
        dl = np.abs(dsi[live])
        print(f"  {label:<34} {int(live.sum()):5d} {tot[live].mean()/2:7.1f} "
              f"{dl.mean():10.4f} {(dl>0.1).mean():6.1%} {(dl>0.5).mean():6.1%}")

    # SINGLE-INPUT CONTROL. One input CANNOT be direction selective: a single
    # sampling point sees the same waveform whichever way the grating moves,
    # only time-shifted, so the mean rate must be identical. Any |DSI| here is
    # manufactured by the replay itself and invalidates every other row.
    print(f"\n  SINGLE-INPUT CONTROL (must be ~0 -- one point cannot be selective)")
    for only in ("Mi1", "Mi9"):
        a, b = replay({only})
        if a is None:
            print(f"  {only+' alone':<34} (no inputs)"); continue
        tot = a + b
        live = tot > 1.0
        if not live.any():
            print(f"  {only+' alone':<34} {0:5d}  (silent)"); continue
        dsi = np.where(tot > 1e-9, (a - b) / np.maximum(tot, 1e-9), 0.0)
        dl = np.abs(dsi[live])
        print(f"  {only+' alone':<34} {int(live.sum()):5d} {tot[live].mean()/2:7.1f} "
              f"{dl.mean():10.4f} {(dl>0.1).mean():6.1%} {(dl>0.5).mean():6.1%}")

    # NULL CONTROL. A DSI computed from finite spike counts is nonzero by
    # chance. The honest null runs the SAME direction twice with different
    # noise and computes "DSI" between those, which must be ~0 for any real
    # effect to mean anything.
    print(f"\n  NULL: same direction twice, different noise")
    a1, _ = replay(None, graded_target=True)
    rec[-1], keep = rec[+1], rec[-1]          # both replays see +1 traces
    a2, _ = replay(None, graded_target=True)
    rec[-1] = keep
    tot = a1 + a2
    live = tot > 1.0
    if live.any():
        nd = np.abs(np.where(tot > 1e-9, (a1 - a2) / np.maximum(tot, 1e-9), 0.0))[live]
        print(f"  {'null (same stimulus)':<34} {int(live.sum()):5d} "
              f"{tot[live].mean()/2:7.1f} {nd.mean():10.4f} "
              f"{(nd>0.1).mean():6.1%} {(nd>0.5).mean():6.1%}")
        print(f"  -> a real effect must exceed this null by a clear margin")

    print(paint("""
  Read down the column. The row where mean |DSI| collapses names the input that
  destroys direction selectivity. If it survives all five correlator types and
  dies only on the last row, the culprit is outside the correlator pathway --
  and that pathway is only 14-20% of T4a's synapses.""", "90"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
