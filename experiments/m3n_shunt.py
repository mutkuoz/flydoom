#!/usr/bin/env python3
"""M3n — is the cell in its LINEAR regime? Then no correlator can work.

A Reichardt detector needs a MULTIPLICATIVE interaction between a delayed and
an undelayed arm. In a conductance-based neuron the only place that can come
from is the denominator:

    v_inf = (V_rest + g_e*E_e + g_i*E_i) / (1 + g_e + g_i)

Inhibition DIVIDES -- it shunts -- and that division is what makes the effect
of excitation depend on inhibition. But the division only bites if g_e + g_i
is comparable to 1. If both are << 1 the denominator is ~1 and the equation
collapses to

    v_inf ~ V_rest + g_e*E_e + g_i*E_i

which is purely ADDITIVE. There is no interaction term at all, and no amount of
correct wiring can produce direction selectivity from it: threshold(A + B) is
symmetric in A and B.

This script MEASURES the realised conductances on T4/T5 during a drifting
grating in the regime where the cell actually works (spiking T4, optic gain 16,
bias 0), then quantifies how nonlinear the cell is there:

    d(v_inf)/d(g_e) = (E_e - v_inf) / g_tot

Evaluated at the measured g_i, and again at g_i = 0. If the two derivatives are
nearly equal, excitation and inhibition are not interacting and the cell is an
adder. The ratio, as a percentage, is the size of the multiplicative term the
correlator has to work with.

It also solves for the g_i that WOULD give a stated change in that derivative,
so the distance from the linear regime is a number and not an impression.

    python experiments/m3n_shunt.py --bias 0 --optic-gain 16 --spiking-t4 \
        --period 30 --tf 2.0 --device cpu
"""
from __future__ import annotations

import argparse
import os
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

WATCH = ("T4a", "T4b", "T5a", "T5b", "Mi1", "Mi9", "Tm1", "Tm9")


def cells_of_type(graph, ann, name):
    d = ann.df
    f = d.filter((pl.col("primary_type") == name) | (pl.col("visual_type") == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                     if int(x) in pos], dtype=np.int64)


def v_inf(ge, gi, p):
    return (p.v_rest + ge * p.e_exc + gi * p.e_inh) / (1.0 + ge + gi)


def dvinf_dge(ge, gi, p):
    """d(v_inf)/d(g_e) = (E_e - v_inf) / g_tot. Exact, not numerical."""
    return (p.e_exc - v_inf(ge, gi, p)) / (1.0 + ge + gi)


def gi_for_ratio(ge, target, p, hi=1e4):
    """Smallest g_i at which the excitatory derivative is `target` times its
    value at g_i = 0. Bisection; the function is monotone decreasing in g_i."""
    lo = 0.0
    base = dvinf_dge(ge, 0.0, p)
    if dvinf_dge(ge, hi, p) / base > target:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dvinf_dge(ge, mid, p) / base > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--period", type=float, default=30.0)
    ap.add_argument("--tf", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--optic-gain", type=float, default=16.0)
    ap.add_argument("--inh-scale", type=float, default=1.0,
                    help="one global scalar over every inhibitory synapse")
    ap.add_argument("--spiking-t4", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE") or "cpu"))
    args = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    if args.optic_gain != 1.0:
        from flydoom.gains import optic_gain_multipliers
        g.signed_syn = (g.signed_syn
                        * optic_gain_multipliers(g, ann, args.optic_gain)
                        ).astype(np.float32)
    if args.inh_scale != 1.0:
        neg = g.signed_syn < 0
        sw = g.signed_syn.astype(np.float32).copy()
        sw[neg] *= args.inh_scale
        g.signed_syn = sw
        print(f"inhibitory conductance scaled x{args.inh_scale:g}")
    retina = Retina.build(g, ann)

    graded = g.graded_mask(ann)
    if args.spiking_t4:
        d_ = ann.df
        for t_ in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"):
            f_ = d_.filter((pl.col("primary_type") == t_)
                           | (pl.col("visual_type") == t_))
            ids = f_["root_id"].unique().to_list()
            if ids:
                graded[g.index_of(ids)] = False
        print("T4/T5 made SPIKING")
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=config.T_DLY_SLOW)
    net = LIFNetwork.from_graph(g, device=args.device, seed=0,
                                edge_delay=edge_delay, graded=graded)
    gext = _bias_vector(net, g, ann, args.bias, args.device)
    rig = GratingRig(net, retina, args.device, args.period, 150.0)

    idx = {t: cells_of_type(g, ann, t) for t in WATCH}
    idx = {t: v for t, v in idx.items() if len(v)}
    watch = np.unique(np.concatenate(list(idx.values())))
    wpos = {int(c): k for k, c in enumerate(watch)}
    wt = torch.as_tensor(watch, device=args.device)

    dt = config.DT
    n_steps = int(round(args.duration / dt))
    skip = int(0.5 / dt)
    n_rec = n_steps - skip
    GE = np.zeros((n_rec, len(watch)), dtype=np.float32)
    GI = np.zeros((n_rec, len(watch)), dtype=np.float32)
    V = np.zeros((n_rec, len(watch)), dtype=np.float32)
    OUT = np.zeros(len(watch))

    # net.step() DECAYS g after using it, so the value left behind is the one
    # that acted, times syn_decay. Undo that so the numbers reported are the
    # conductances the membrane equation actually saw.
    undo = 1.0 / net.syn_decay

    net.reset()
    for step in range(n_steps):
        net.step(g_ext=gext,
                 out_set=rig.out_set(step * dt, args.tf, +1,
                                     net.p.graded_max_rate, dt))
        if step >= skip:
            k = step - skip
            GE[k] = net.g_exc[wt].detach().cpu().numpy() * undo
            GI[k] = net.g_inh[wt].detach().cpu().numpy() * undo
            V[k] = net.v[wt].detach().cpu().numpy()
            OUT += net.out[wt].detach().cpu().numpy() / dt
    OUT /= n_rec

    p = net.p
    print(paint("M3n — realised conductances, and how nonlinear the cell is "
                "there", "1"))
    print(paint("=" * 86, "90"))
    print(f"bias {args.bias} mV, optic gain {args.optic_gain:g}, inh scale "
          f"{args.inh_scale:g}, period {args.period:.0f} deg, tf {args.tf:.1f} Hz, "
          f"{args.duration:.0f} s\n")
    print(f"  E_exc {p.e_exc * 1e3:.0f} mV   E_inh {p.e_inh * 1e3:.0f} mV   "
          f"V_rest {p.v_rest * 1e3:.0f} mV   V_thresh {p.v_thresh * 1e3:.0f} mV\n")
    print(f"  {'type':>5} {'n':>5} {'rate Hz':>8} | "
          f"{'g_e med':>9} {'g_e p90':>9} | {'g_i med':>9} {'g_i p90':>9} | "
          f"{'g_tot med':>10} {'g_tot p90':>10}")

    record = {"bias_mv": args.bias, "optic_gain": args.optic_gain,
              "inh_scale": args.inh_scale, "period_deg": args.period,
              "tf_hz": args.tf, "e_exc": p.e_exc, "e_inh": p.e_inh,
              "v_rest": p.v_rest, "types": {}}

    for t in WATCH:
        if t not in idx:
            continue
        cols = np.array([wpos[int(c)] for c in idx[t]])
        ge = GE[:, cols].ravel()
        gi = GI[:, cols].ravel()
        gt = 1.0 + ge + gi
        row = {
            "n": int(len(cols)), "rate_hz": float(OUT[cols].mean()),
            "g_exc_median": float(np.median(ge)),
            "g_exc_p90": float(np.percentile(ge, 90)),
            "g_exc_mean": float(ge.mean()),
            "g_inh_median": float(np.median(gi)),
            "g_inh_p90": float(np.percentile(gi, 90)),
            "g_inh_mean": float(gi.mean()),
            "g_tot_median": float(np.median(gt)),
            "g_tot_p90": float(np.percentile(gt, 90)),
            "g_tot_p99": float(np.percentile(gt, 99)),
            "v_median_mv": float(np.median(V[:, cols]) * 1e3),
        }
        record["types"][t] = row
        print(f"  {t:>5} {row['n']:>5} {row['rate_hz']:>8.2f} | "
              f"{row['g_exc_median']:>9.4f} {row['g_exc_p90']:>9.4f} | "
              f"{row['g_inh_median']:>9.4f} {row['g_inh_p90']:>9.4f} | "
              f"{row['g_tot_median']:>10.4f} {row['g_tot_p90']:>10.4f}")

    # ---- how much interaction is available at those conductances? ---------
    print()
    print(paint("  Multiplicative interaction available", "1"))
    print(paint("  d(v_inf)/d(g_e) = (E_e - v_inf)/g_tot, at the measured g_i "
                "versus at g_i = 0", "90"))
    print(f"  {'type':>5} {'g_e':>8} {'g_i':>8} {'dV/dge @gi':>12} "
          f"{'dV/dge @0':>11} {'ratio':>8} {'interaction':>12}")
    for t in WATCH:
        if t not in record["types"]:
            continue
        r = record["types"][t]
        ge, gi = r["g_exc_median"], r["g_inh_median"]
        d_at = dvinf_dge(ge, gi, p)
        d_0 = dvinf_dge(ge, 0.0, p)
        ratio = d_at / d_0
        r["dvinf_dge_at_gi"] = d_at
        r["dvinf_dge_at_zero"] = d_0
        r["derivative_ratio"] = float(ratio)
        r["interaction_pct"] = float((1.0 - ratio) * 100.0)
        # and what g_i would be needed for a real interaction
        for tgt, key in ((0.5, "gi_for_2x"), (0.9, "gi_for_10pct")):
            r[key] = float(gi_for_ratio(ge, tgt, p))
        print(f"  {t:>5} {ge:>8.4f} {gi:>8.4f} {d_at:>12.5f} {d_0:>11.5f} "
              f"{ratio * 100:>7.1f}% {(1 - ratio) * 100:>11.1f}%")

    print()
    print(paint("  How far from a real interaction?", "1"))
    print(f"  {'type':>5} {'g_i now':>9} {'g_i for 2x':>11} {'shortfall':>11} "
          f"{'g_tot now':>10} {'g_tot needed':>13}")
    for t in WATCH:
        if t not in record["types"]:
            continue
        r = record["types"][t]
        ge, gi = r["g_exc_median"], r["g_inh_median"]
        need = r["gi_for_2x"]
        print(f"  {t:>5} {gi:>9.4f} {need:>11.4f} "
              f"{need / max(gi, 1e-12):>10.1f}x {1 + ge + gi:>10.4f} "
              f"{1 + ge + need:>13.4f}")

    # ---- assumption-free version of the same question --------------------
    # The derivative comparison above is a local statement at the median. This
    # is the global one: take the ACTUAL joint trajectory of (g_e, g_i) the
    # cell visited, compute the v_inf it implies, and ask how much of that a
    # purely ADDITIVE model  v = a + b*g_e + c*g_i  can reproduce. Whatever it
    # cannot is the entire nonlinear budget -- the multiplicative interaction
    # included. Adding the bilinear term g_e*g_i then says how much of that
    # budget is the interaction rather than plain curvature.
    print()
    print(paint("  How much of the cell's own trajectory is ADDITIVE?", "1"))
    print(f"  {'type':>5} {'sd(v_inf)':>10} {'additive R2':>12} "
          f"{'resid mV':>9} {'+g_e*g_i R2':>12} {'gain from x':>12}")
    for t in WATCH:
        if t not in idx:
            continue
        cols = np.array([wpos[int(c)] for c in idx[t]])
        ge = GE[:, cols].ravel().astype(np.float64)
        gi = GI[:, cols].ravel().astype(np.float64)
        y = (p.v_rest + ge * p.e_exc + gi * p.e_inh) / (1.0 + ge + gi)
        one = np.ones_like(ge)
        sst = float(((y - y.mean()) ** 2).sum())

        def r2(X):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ beta
            return 1.0 - float((res ** 2).sum()) / sst, float(np.sqrt((res ** 2).mean()))

        r2_add, rms_add = r2(np.stack([one, ge, gi], axis=1))
        r2_bil, _ = r2(np.stack([one, ge, gi, ge * gi], axis=1))
        rec = record["types"][t]
        rec["v_inf_sd_mv"] = float(y.std() * 1e3)
        rec["additive_r2"] = r2_add
        rec["additive_resid_mv"] = rms_add * 1e3
        rec["bilinear_r2"] = r2_bil
        print(f"  {t:>5} {y.std() * 1e3:>9.2f}m {r2_add:>12.4f} "
              f"{rms_add * 1e3:>9.3f} {r2_bil:>12.4f} "
              f"{(r2_bil - r2_add) * 100:>11.2f}%")

    print(paint("""
  Read the 'interaction' column. It is the fraction by which the presence of
  inhibition changes the cell's sensitivity to excitation -- the entire
  multiplicative term a Reichardt detector has to build direction selectivity
  out of. Near zero means the neuron is an ADDER, and the wiring cannot be the
  binding constraint.""", "90"))

    if args.json:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
