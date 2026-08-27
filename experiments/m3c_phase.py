#!/usr/bin/env python3
"""M3c — do a T4 cell's two correlator arms arrive with a phase difference?

A correlator multiplies a signal by a DELAYED copy of a signal from a
NEIGHBOURING point. Both conditions matter. This project has verified the
delay (T_DLY_SLOW on the slow lines) and, in m3b, that both arms are strongly
modulated by the stimulus. What has never been checked is the third
requirement: that the two arms feeding ONE T4 cell sample different points, so
their signals differ in phase.

If a cell's Mi1 and Mi9 inputs arrive in phase, the cell is summing two copies
of the same waveform and there is no correlator, however well balanced or
delayed the arms are. That single fact would explain every negative result in
this project.

Measured per T4 cell: the synapse-weighted circular mean phase of its Mi1
inputs, the same for its Mi9 inputs, and the difference.

    python experiments/m3c_phase.py [--period 15] [--tf 4]
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--target", default="T4a")
    ap.add_argument("--fast", default="Mi1")
    ap.add_argument("--slow", default="Mi9")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
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

    tgt = cells_of_type(g, ann, args.target)
    fast = cells_of_type(g, ann, args.fast)
    slow = cells_of_type(g, ann, args.slow)
    watch = np.unique(np.concatenate([fast, slow]))
    wt = torch.as_tensor(watch, device=args.device)
    col = {int(v): i for i, v in enumerate(watch)}

    dt = config.DT
    n = int(round(args.duration / dt))
    tgt_t = torch.as_tensor(tgt, device=args.device)
    trace = np.zeros((n, len(watch)), dtype=np.float32)
    # Per-cell DSI of the TARGET needs both directions, measured at the same
    # operating point as the phases -- otherwise the two cannot be compared.
    tgt_rate = {}
    for direction in (+1, -1):
        net.reset()
        acc_t = np.zeros(len(tgt), dtype=np.float64)
        for step in range(n):
            net.step(g_ext=gext,
                     out_set=rig.out_set(step * dt, args.tf, direction,
                                         net.p.graded_max_rate, dt))
            if direction > 0:
                trace[step] = net.out[wt].detach().cpu().numpy()
            if step >= int(0.4 / dt):
                acc_t += net.out[tgt_t].detach().cpu().numpy()
        tgt_rate[direction] = acc_t / (args.duration - 0.4)

    skip = int(0.4 / dt)
    tt = np.arange(n)[skip:] * dt
    ref = np.exp(-2j * math.pi * args.tf * tt)[:, None]
    comp = (trace[skip:] * ref).mean(axis=0)          # complex, per watched cell
    amp = np.abs(comp)

    # per target cell, synapse-weighted circular mean of each arm's phase
    fastset, slowset = set(fast.tolist()), set(slow.tolist())
    pre, post, syn = g.pre_idx, g.post_idx, np.abs(g.signed_syn)
    acc = {int(t): {"f": 0j, "s": 0j, "fw": 0.0, "sw": 0.0} for t in tgt}
    for p_, q_, w_ in zip(pre, post, syn):
        a = acc.get(int(q_))
        if a is None:
            continue
        p_ = int(p_)
        if p_ in fastset:
            a["f"] += w_ * comp[col[p_]] / max(amp[col[p_]], 1e-12)
            a["fw"] += w_
        elif p_ in slowset:
            a["s"] += w_ * comp[col[p_]] / max(amp[col[p_]], 1e-12)
            a["sw"] += w_

    tpos = {int(v): i for i, v in enumerate(tgt)}
    ra, rb = tgt_rate[+1], tgt_rate[-1]
    diffs, both, dsis = [], 0, []
    for t, a in acc.items():
        if a["fw"] <= 0 or a["sw"] <= 0:
            continue
        both += 1
        diffs.append(math.degrees(np.angle(a["s"] / a["f"])) % 360)
        i = tpos[t]
        tot = ra[i] + rb[i]
        dsis.append((ra[i] - rb[i]) / tot if tot > 1e-9 else 0.0)
    diffs = np.array(diffs); dsis = np.array(dsis)

    print(paint(f"M3c — phase between {args.fast} and {args.slow} arms "
                f"of {args.target}", "1"))
    print(paint("=" * 74, "90"))
    print(f"grating period {args.period:.0f} deg, {args.tf:.1f} Hz; "
          f"{both} of {len(tgt)} {args.target} cells receive both arms\n")
    if not len(diffs):
        print("  no cell receives both arms — nothing to correlate")
        return 1

    # circular statistics
    z = np.exp(1j * np.radians(diffs)).mean()
    print(f"  mean phase difference   {math.degrees(np.angle(z)) % 360:7.1f} deg")
    print(f"  concentration (0=random, 1=identical)  {abs(z):.3f}")
    print(f"  median |offset from 0|  {np.median(np.minimum(diffs, 360-diffs)):7.1f} deg")
    for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 180),
                   (180, 240), (240, 300), (300, 360)):
        k = int(((diffs >= lo) & (diffs < hi)).sum())
        bar = "#" * int(40 * k / max(len(diffs), 1))
        print(f"    {lo:3d}-{hi:3d} deg  {k:5d}  {bar}")

    # THE TEST: does a cell's phase difference predict its own selectivity?
    print(f"\n  per-cell DSI binned by that cell's phase difference:")
    print(f"  {'phase bin':>14} {'n':>6} {'mean DSI':>10} {'mean |DSI|':>11} "
          f"{'frac |DSI|>0.1':>15}")
    for lo, hi in ((0, 45), (45, 90), (90, 135), (135, 180),
                   (180, 225), (225, 270), (270, 315), (315, 360)):
        k = (diffs >= lo) & (diffs < hi)
        if k.sum() < 5:
            continue
        print(f"  {lo:5d}-{hi:3d} deg {int(k.sum()):6d} {dsis[k].mean():+10.4f} "
              f"{np.abs(dsis[k]).mean():11.4f} {(np.abs(dsis[k])>0.1).mean():14.1%}")
    # a correlator predicts DSI ~ sin(phase difference): opposite sign either
    # side of 180 deg. Correlate signed DSI against sin of the phase.
    # A correlator's output varies as sin(phase + offset). The offset is not
    # zero: the 80 ms slow-line delay and the membrane filtering both add lag,
    # so we FIT it rather than assume it, and report the best correlation
    # together with a null from shuffling the pairing.
    best_r, best_off = 0.0, 0
    for off in range(0, 360, 5):
        pred = np.sin(np.radians(diffs + off))
        if dsis.std() > 1e-12 and pred.std() > 1e-12:
            r = float(np.corrcoef(pred, dsis)[0, 1])
            if abs(r) > abs(best_r):
                best_r, best_off = r, off
    rng = np.random.default_rng(0)
    null = []
    for _ in range(200):
        d2 = rng.permutation(dsis)
        m = 0.0
        for off in range(0, 360, 15):
            pred = np.sin(np.radians(diffs + off))
            r = float(np.corrcoef(pred, d2)[0, 1])
            if abs(r) > abs(m):
                m = r
        null.append(abs(m))
    null = np.array(null)
    print(f"\n  best fit: corr = {best_r:+.4f} at phase offset {best_off} deg")
    print(f"  null (200 shuffles, same fitting freedom): "
          f"mean |r| {null.mean():.4f}, 95th pct {np.percentile(null,95):.4f}")
    print(f"  -> {'ABOVE' if abs(best_r) > np.percentile(null,95) else 'WITHIN'}"
          f" the null. A real correlator signal must beat its own null.")
    print(f"  overall: mean DSI {dsis.mean():+.4f}, "
          f"mean |DSI| {np.abs(dsis).mean():.4f}, "
          f"max |DSI| {np.abs(dsis).max():.4f}")

    np.save("/tmp/claude-1000/-home-mutkuoz-Documents-flydoom/phase_dsi.npy",
            np.stack([diffs, dsis]))
    delay_phase = (config.T_DLY_SLOW * args.tf * 360.0) % 360
    print(f"\n  the {config.T_DLY_SLOW*1e3:.0f} ms slow-line delay alone contributes "
          f"{delay_phase:.0f} deg at {args.tf:.0f} Hz")
    print(paint("""
  A correlator needs the two arms separated in phase. Near 0 or 360 deg means
  the arms carry the same waveform and the cell is summing two copies of one
  signal -- no spatial offset survives into the retinal mapping, and no
  rebalancing, gain change or delay can create selectivity from that. Near
  90-180 deg means the offset is present and the failure lies elsewhere.""",
                "90"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
