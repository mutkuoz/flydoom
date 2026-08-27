#!/usr/bin/env python3
"""M3m — feed the REAL Mi1/Mi9 signals into a correlator that provably works.

WHERE THIS SITS
---------------
M3l built a correlator from the same neuron model and walked it toward T4 one
property at a time. Real synapse counts make the cell silent (rung 1) -- which
is the drive problem -- but once drive is restored, EVERY remaining T4 property
is harmless: fan-in, spatial spread across columns, and the clamped graded
transfer all leave it selective, reaching DSI -1.0.

So the cell is not the problem, the weights are not the problem (given drive),
the fan-in is not, the spread is not, the transfer function is not. The only
thing the network has that the ladder does not is THE ACTUAL SIGNALS: Mi1 and
Mi9 activity produced by the retina and lamina, rather than clean sinusoids.

This swaps exactly that one thing. The correlator is unchanged; only its input
waveforms are replaced by traces recorded from the running network.

  synthetic   matched sinusoids            -> must reproduce strong DSI
  real        recorded Mi1/Mi9 traces      -> the measurement
  single arm  real Mi1 only, no inhibition -> must give ~0

THE THIRD ROW IS NOT OPTIONAL. An earlier isolated-replay experiment in this
project returned |DSI| 0.12 for a single input, where the geometry makes
direction selectivity physically impossible, because a threshold unit amplifies
whatever weak selectivity its one input already carries. Any real-trace result
must clear that floor to mean anything.
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
from flydoom.lif import LIFNetwork, LIFParams  # noqa: E402
from flydoom.retina import Retina  # noqa: E402
from m3_optomotor import GratingRig, _bias_vector, paint  # noqa: E402


def cells_of_type(graph, ann, name):
    d = ann.df
    f = d.filter((pl.col("primary_type") == name) | (pl.col("visual_type") == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                     if int(x) in pos], dtype=np.int64)


def toy(w_exc, w_inh, delay_inh_ms, device, graded_target=False):
    """One spiking target fed by len(w_exc) excitatory and len(w_inh) inhibitory
    graded inputs, with the caller's real synapse counts as weights."""
    ne, ni = len(w_exc), len(w_inh)
    n = ne + ni + 1
    tgt = n - 1
    pre = np.arange(n - 1, dtype=np.int64)
    post = np.full(n - 1, tgt, dtype=np.int64)
    syn = np.concatenate([np.asarray(w_exc, np.float32),
                          -np.asarray(w_inh, np.float32)])
    dt_ms = config.DT * 1e3
    delay = np.concatenate([np.full(ne, 2), np.full(ni, max(1, round(delay_inh_ms / dt_ms)))])
    graded = np.zeros(n, dtype=bool); graded[:n - 1] = True
    # The network models T4 as a GRADED non-spiking unit, whose output is a
    # clamped LINEAR function of membrane voltage. A correlator needs a
    # nonlinearity to turn a phase difference into a rate difference; a
    # threshold supplies one, a clamped ramp does not.
    if graded_target:
        graded[n - 1] = True
    net = LIFNetwork(n, torch.as_tensor(pre, device=device),
                     torch.as_tensor(post, device=device),
                     torch.as_tensor(syn, device=device),
                     LIFParams(), device, 0,
                     edge_delay=delay.astype(np.int64), graded=graded)
    return net, tgt


def replay(net, tgt, drive, device):
    """drive: [steps, n_inputs] in the same units net.out produces."""
    steps, n_in = drive.shape
    out_set = torch.full((net.n,), -1.0, dtype=torch.float32, device=device)
    dt = config.DT
    acc, cnt = 0.0, 0
    skip = min(int(0.5 / dt), steps // 4)
    net.reset()
    for s in range(steps):
        out_set[:n_in] = torch.as_tensor(drive[s], dtype=torch.float32, device=device)
        net.step(out_set=out_set)
        if s >= skip:
            acc += float(net.out[tgt]) / dt; cnt += 1
    return acc / max(cnt, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=float, default=2.0)
    ap.add_argument("--period", type=float, default=15.0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--gain", type=float, default=2.0,
                    help="drive multiplier; rung 1 of M3l shows the real "
                         "weights alone leave the cell silent.")
    ap.add_argument("--n-cells", type=int, default=12)
    ap.add_argument("--optic-gain", type=float, default=1.0,
                    help="gain applied to the NETWORK before recording. At 1.0 "
                         "with bias 0 the whole visual pathway is near silent, "
                         "so the recorded traces are empty and the replay says "
                         "nothing. x2 is where T4 first fires from its inputs.")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
    a = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    if a.optic_gain != 1.0:
        from flydoom.gains import optic_gain_multipliers
        g.signed_syn = (g.signed_syn
                        * optic_gain_multipliers(g, ann, a.optic_gain)
                        ).astype(np.float32)
    retina = Retina.build(g, ann)
    graded = g.graded_mask(ann)
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=config.T_DLY_SLOW)
    net = LIFNetwork.from_graph(g, device=a.device, seed=0,
                                edge_delay=edge_delay, graded=graded)
    gext = _bias_vector(net, g, ann, 0.0, a.device)     # bias OFF
    rig = GratingRig(net, retina, a.device, a.period, 150.0)

    mi1 = set(cells_of_type(g, ann, "Mi1").tolist())
    mi9 = set(cells_of_type(g, ann, "Mi9").tolist())
    t4a = cells_of_type(g, ann, "T4a")

    # pick the T4a cell with the most complete correlator, and take ITS inputs
    best, bw = None, -1
    for c in t4a[:400]:
        c = int(c)
        m = g.post_idx == c
        pres, ws = g.pre_idx[m], np.abs(g.signed_syn[m])
        e = [(int(p), float(w)) for p, w in zip(pres, ws) if int(p) in mi1]
        i = [(int(p), float(w)) for p, w in zip(pres, ws) if int(p) in mi9]
        if e and i and (len(e) + len(i)) > bw:
            best, bw = (c, e, i), len(e) + len(i)
    if best is None:
        print("no T4a cell with both arms"); return 1
    cell, exc, inh = best
    exc = sorted(exc, key=lambda x: -x[1])[:a.n_cells]
    inh = sorted(inh, key=lambda x: -x[1])[:a.n_cells]
    idx = [p for p, _ in exc] + [p for p, _ in inh]
    we = [w * a.gain for _, w in exc]
    wi = [w * a.gain for _, w in inh]

    print(paint("M3m — real Mi1/Mi9 signals in a correlator that provably works", "1"))
    print(paint("=" * 78, "90"))
    print(f"T4a cell {cell}: {len(exc)} Mi1 inputs (sum w {sum(we)/a.gain:.1f}), "
          f"{len(inh)} Mi9 inputs (sum w {sum(wi)/a.gain:.1f}); gain x{a.gain:g}\n")

    dt = config.DT
    steps = int(round(a.seconds / dt))
    wt = torch.as_tensor(np.asarray(idx), device=a.device)
    traces = {}
    for sign, nm in ((+1, "rightward"), (-1, "leftward")):
        net.reset()
        tr = np.zeros((steps, len(idx)), dtype=np.float32)
        for s in range(steps):
            net.step(g_ext=gext,
                     out_set=rig.out_set(s * dt, a.tf, sign,
                                         net.p.graded_max_rate, dt))
            tr[s] = net.out[wt].detach().cpu().numpy()
        traces[nm] = tr

    ceiling = net.p.graded_max_rate
    ne, ni = len(we), len(wi)
    # what amplitude do the real arms actually deliver, against the ceiling a
    # synthetic drive uses? This is the comparison the replay depends on.
    for nm in ("rightward",):
        tr = traces[nm]
        e_hz = tr[:, :ne].mean() / config.DT
        i_hz = tr[:, ne:].mean() / config.DT
        e_f1 = (tr[:, :ne].max() - tr[:, :ne].min()) / config.DT
        i_f1 = (tr[:, ne:].max() - tr[:, ne:].min()) / config.DT
    print(f"  real arm amplitudes: Mi1 mean {e_hz:6.2f} Hz (swing {e_f1:6.2f}), "
          f"Mi9 mean {i_hz:6.2f} Hz (swing {i_f1:6.2f})")
    print(f"  synthetic drive uses the full {ceiling:.0f} Hz range\n")
    # synthetic control matched to the real traces in mean and modulation
    ph_e = np.zeros(ne)
    ph_i = np.full(ni, math.radians(120.0))
    tt = np.arange(steps) * dt
    syn = {}
    for sign, nm in ((+1, "rightward"), (-1, "leftward")):
        ph = np.concatenate([ph_e, ph_i]) * sign
        syn[nm] = ((0.5 + 0.5 * np.sin(2 * math.pi * a.tf * tt[:, None] + ph))
                   * ceiling * dt).astype(np.float32)

    rows = []
    for label, src, keep_inh, gt in (("synthetic", syn, True, False),
                                     ("REAL traces", traces, True, False),
                                     ("REAL, graded T4", traces, True, True),
                                     ("real, Mi1 only", traces, False, False)):
        if keep_inh:
            tnet, tgt = toy(we, wi, config.T_DLY_SLOW * 1e3, a.device,
                            graded_target=gt)
            R = replay(tnet, tgt, src["rightward"], a.device)
            L = replay(tnet, tgt, src["leftward"], a.device)
        else:
            tnet, tgt = toy(we, [], config.T_DLY_SLOW * 1e3, a.device,
                            graded_target=gt)
            R = replay(tnet, tgt, src["rightward"][:, :ne], a.device)
            L = replay(tnet, tgt, src["leftward"][:, :ne], a.device)
        d = (R - L) / max(R + L, 1e-12)
        rows.append((label, d, R, L))

    print(f"  {'input':<16}{'DSI':>10}{'R_pref':>10}{'R_null':>10}{'|R-L| Hz':>11}")
    print("  " + "-" * 57)
    for label, d, R, L in rows:
        print(f"  {label:<16}{d:>10.4f}{R:>10.2f}{L:>10.2f}{abs(R-L):>11.3f}")
    print(paint("""
  Row 1 validates the instrument. Row 3 is the floor: it must be near zero,
  because one arm cannot compute direction. If row 2 sits at the floor while
  row 1 is large, the correlator is fine and the SIGNALS arriving from the
  retina and lamina are what lack direction information -- which places the
  failure upstream of T4 entirely.""", "90"))
    if a.json:
        import json
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(
            {"cell": cell, "tf": a.tf, "period": a.period, "gain": a.gain,
             "rows": [{"input": l, "dsi": d, "r": R, "l": L} for l, d, R, L in rows]},
            indent=1))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
