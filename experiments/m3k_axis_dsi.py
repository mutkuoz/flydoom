#!/usr/bin/env python3
"""M3k — within a subtype, does a cell's arm SEPARATION predict its selectivity?

M3j found that across the four measured subtypes, the spatial separation of the
excitatory and inhibitory centroids tracks direction selectivity almost
perfectly (r = +0.98). That is four points, and axis length and consistency are
collinear, so it is suggestive and no more.

This is the same question at n ~ 500 per subtype instead of n = 4. Each cell
gets its OWN arm separation and its OWN DSI, both measured directly, and the
correlation is taken WITHIN a subtype -- so it cannot be explained by anything
that differs between subtypes.

The prediction is specific rather than generic. The grating drifts
HORIZONTALLY, so the component of the axis that should matter is the AZIMUTHAL
one: a cell whose two arms are displaced vertically has no horizontal baseline
and should show no horizontal direction selectivity however large its total
offset. So |DSI| should track |axis_azimuth|, and track it better than it
tracks total axis length. If instead the elevation component predicts just as
well, the relationship is not the geometric one claimed and something commoner
(fan-in, weight, noise) is driving both.

Geometry is in DEGREES OF VISUAL ANGLE, not lattice steps, taken from the
retina's own azimuth/elevation map, because that is the space the stimulus
moves in.
"""
from __future__ import annotations

import argparse
import os
import math
import sys
from collections import defaultdict
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

# Mi9 and Mi4 sit on OPPOSITE flanks; pooling them averages two opposite
# vectors and manufactures incoherence (m3t: R rises 0.19-0.47 -> 0.67-0.93
# when Mi9 is taken alone, and the mirror pairs go from an apparent 97-114 deg
# defect to a correct 182-194 deg). The null-side inhibition alone defines the
# correlator axis.
ARMS = {"T4": {"exc": ("Mi1",), "inh": ("Mi9",)},
        "T5": {"exc": ("Tm1",), "inh": ("Tm9",)}}
SUBTYPES = ("T4a", "T4b", "T5a", "T5b")


def cells_of_type(graph, ann, name):
    d = ann.df
    f = d.filter((pl.col("primary_type") == name) | (pl.col("visual_type") == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                     if int(x) in pos], dtype=np.int64)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return 0.0
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, default=1.0,
                    help="lowest bias the grid found unsaturated; also where "
                         "arm modulation is healthiest.")
    ap.add_argument("--period", type=float, default=15.0)
    ap.add_argument("--tf", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--spiking-t4", action="store_true",
                    help="restore the threshold nonlinearity on T4/T5. A graded "
                         "unit is a clamped LINEAR ramp and costs 10.6x (m3m).")
    ap.add_argument("--optic-gain", type=float, default=1.0,
                    help="T4's summed weight sits at the threshold for any "
                         "output at all, so at gain 1 with bias 0 the cell "
                         "fires on nothing.")
    ap.add_argument("--inh-scale", type=float, default=1.0,
                    help="ONE GLOBAL SCALAR over every inhibitory synapse in "
                         "the brain. Not a per-cell-type gain: it multiplies "
                         "g_inh everywhere and so moves the whole model along "
                         "the linear -> shunting axis. g_tot = 1 + g_e + g_i "
                         "is where any multiplicative interaction has to come "
                         "from; at g_i << 1 the cell is additive.")
    ap.add_argument("--min-rate", type=float, default=20.0,
                    help="minimum R+L in Hz for a cell to enter the "
                         "correlation. Below this, DSI is a ratio of two "
                         "near-zero numbers and carries no information.")
    ap.add_argument("--e-inh", type=float, default=None,
                    help="inhibitory reversal potential in mV, one global "
                         "constant. Default -70 sits 18 mV BELOW V_rest, so "
                         "inhibition here both hyperpolarises and shunts. "
                         "Setting it to V_rest (-52) makes it PURELY DIVISIVE "
                         "and separates the two: if raising inhibition still "
                         "helps at -52, the mechanism is the shunt; if it "
                         "stops helping, it was the hyperpolarisation working "
                         "against the spike threshold.")
    ap.add_argument("--dump-cells", action="store_true",
                    help="write per-cell (index, axis azimuth, right Hz, left "
                         "Hz) into the JSON, so two runs can be compared on "
                         "the SAME cells rather than on each run's survivors.")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
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
        print(f"inhibitory conductance scaled x{args.inh_scale:g} "
              f"({int(neg.sum()):,} edges)")
    retina = Retina.build(g, ann)

    # column -> visual direction, per eye
    colxy, colside = {}, {}
    for side, eye in retina.eyes.items():
        for cid, az, el in zip(eye.column_ids, eye.azimuth_deg, eye.elevation_deg):
            colxy[(side, int(cid))] = (float(az), float(el))
    path = Path(config.RAW_DIR) / "column_assignment.csv.gz"
    ca = pl.read_csv(path)
    pos = {int(r): i for i, r in enumerate(g.root_ids)}
    cell_pt = {}
    for rid, h, cid in zip(ca["root_id"], ca["hemisphere"], ca["column_id"]):
        i = pos.get(int(rid))
        if i is None:
            continue
        pt = colxy.get((str(h), int(cid)))
        if pt is not None:
            cell_pt[i] = pt
            colside[i] = str(h)

    pre, post, w = g.pre_idx, g.post_idx, np.abs(g.signed_syn)
    inputs = defaultdict(list)
    for a, b, c in zip(pre, post, w):
        if c > 0:
            inputs[int(b)].append((int(a), float(c)))

    graded = g.graded_mask(ann)
    if args.spiking_t4:
        import polars as _pl
        d_ = ann.df
        for t_ in ("T4a","T4b","T4c","T4d","T5a","T5b","T5c","T5d"):
            f_ = d_.filter((_pl.col("primary_type")==t_)|(_pl.col("visual_type")==t_))
            ids = f_["root_id"].unique().to_list()
            if ids:
                graded[g.index_of(ids)] = False
        print("T4/T5 made SPIKING")
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=config.T_DLY_SLOW)
    params = None
    if args.e_inh is not None:
        from flydoom.lif import LIFParams
        params = LIFParams(e_inh=args.e_inh * 1e-3)
        print(f"E_inh set to {args.e_inh:g} mV "
              f"(V_rest {config.V_REST * 1e3:.0f} mV)")
    net = LIFNetwork.from_graph(g, params=params, device=args.device, seed=0,
                                edge_delay=edge_delay, graded=graded)
    gext = _bias_vector(net, g, ann, args.bias, args.device)
    rig = GratingRig(net, retina, args.device, args.period, 150.0)

    cells = {s: cells_of_type(g, ann, s) for s in SUBTYPES}
    watch = np.unique(np.concatenate(list(cells.values())))
    widx = {int(c): k for k, c in enumerate(watch)}
    wt = torch.as_tensor(watch, device=args.device)

    dt = config.DT
    n_steps = int(round(args.duration / dt))
    skip = int(0.5 / dt)
    rates = {}
    for sign, name in ((+1, "rightward"), (-1, "leftward")):
        net.reset()
        acc = np.zeros(len(watch)); n = 0
        for step in range(n_steps):
            net.step(g_ext=gext,
                     out_set=rig.out_set(step * dt, args.tf, sign,
                                         net.p.graded_max_rate, dt))
            if step >= skip:
                acc += net.out[wt].detach().cpu().numpy() / dt; n += 1
        rates[name] = acc / max(n, 1)

    print(paint("M3k — does arm separation predict selectivity WITHIN a subtype?", "1"))
    print(paint("=" * 78, "90"))
    print(f"bias {args.bias} mV, period {args.period:.0f} deg, tf {args.tf:.1f} Hz, "
          f"{args.duration:.0f} s per direction\n")
    print(f"  {'subtype':>8} {'n':>5} {'r(|az|,|DSI|)':>15} {'r(|el|,|DSI|)':>15}"
          f" {'r(len,|DSI|)':>14} {'mean|az| deg':>13}")

    record = {"bias_mv": args.bias, "period_deg": args.period, "tf_hz": args.tf,
              "inh_scale": args.inh_scale, "optic_gain": args.optic_gain,
              "subtypes": {}}
    pooled_proj, pooled_diff, pooled_tot = [], [], []
    pooled_sgn, pooled_raw = [], []
    for s in SUBTYPES:
        group = ARMS["T4" if s.startswith("T4") else "T5"]
        exc = set().union(*[set(cells_of_type(g, ann, t).tolist()) for t in group["exc"]])
        inh = set().union(*[set(cells_of_type(g, ann, t).tolist()) for t in group["inh"]])
        AZ, EL, LEN, DSI, TOT, PROJ, DIFF, SGN, RAW = (
            [], [], [], [], [], [], [], [], [])
        CELL = []
        for c in cells[s]:
            c = int(c)
            if c not in cell_pt or c not in widx:
                continue
            x0, y0 = cell_pt[c]
            cen = {}
            for role, members in (("exc", exc), ("inh", inh)):
                sx = sy = sw = 0.0
                for a, ww in inputs.get(c, ()):
                    if a in members and a in cell_pt:
                        ax, ay = cell_pt[a]
                        sx += (ax - x0) * ww; sy += (ay - y0) * ww; sw += ww
                if sw > 0:
                    cen[role] = (sx / sw, sy / sw)
            if len(cen) != 2:
                continue
            dx = cen["inh"][0] - cen["exc"][0]
            dy = cen["inh"][1] - cen["exc"][1]
            k = widx[c]
            r_, l_ = rates["rightward"][k], rates["leftward"][k]
            if r_ + l_ < 1e-6:
                continue
            AZ.append(abs(dx)); EL.append(abs(dy)); TOT.append(r_ + l_)
            LEN.append(math.hypot(dx, dy)); DSI.append(abs((r_ - l_) / (r_ + l_)))
            # AXIS-PROJECTED, and SIGNED. |DSI| is a magnitude, so noise scores
            # positive on it; a correlator population has to agree on a
            # DIRECTION. Each cell's own exc->inh azimuthal offset says which
            # way it ought to prefer, so project its signed DSI onto that
            # offset. Noise averages to zero here. A real detector does not.
            sgn = 1.0 if dx >= 0 else -1.0
            PROJ.append(sgn * (r_ - l_) / (r_ + l_))
            DIFF.append(sgn * (r_ - l_))
            SGN.append(sgn)
            RAW.append((r_ - l_) / (r_ + l_))
            CELL.append((c, dx, float(r_), float(l_)))
        if len(AZ) < 10:
            continue
        # DSI is a ratio and inflates without bound as its denominator vanishes;
        # a cell firing 0.01 Hz against 0.02 Hz scores 0.33 on no signal at all.
        # Correlating geometry against that measures noise. Restrict to cells
        # that are actually driven, as the per-cell control in M3 does.
        AZ, EL, LEN, DSI, TOT, PROJ, DIFF, SGN, RAW = map(
            np.asarray, (AZ, EL, LEN, DSI, TOT, PROJ, DIFF, SGN, RAW))
        keep = TOT > args.min_rate
        n_all = len(AZ)
        AZ, EL, LEN, DSI = AZ[keep], EL[keep], LEN[keep], DSI[keep]
        PROJ, DIFF, TOTK = PROJ[keep], DIFF[keep], TOT[keep]
        SGN, RAW = SGN[keep], RAW[keep]
        pooled_proj.append(PROJ); pooled_diff.append(DIFF); pooled_tot.append(TOTK)
        pooled_sgn.append(SGN); pooled_raw.append(RAW)
        if len(AZ) < 10:
            print(f"  {s:>8} {0:>5}   -- no cells above {args.min_rate} Hz total")
            continue
        r_az, r_el, r_len = pearson(AZ, DSI), pearson(EL, DSI), pearson(LEN, DSI)
        print(f"  {s:>8} {len(AZ):>5} {r_az:>15.3f} {r_el:>15.3f} {r_len:>14.3f}"
              f" {np.mean(AZ):>13.2f}   ({len(AZ)}/{n_all} driven, "
              f"mean|DSI| {DSI.mean():.4f})")
        # DSI is a ratio and inflates without bound as its denominator falls,
        # so it is printed beside the two numbers that cannot be inflated: the
        # firing rate, and the rate DIFFERENCE in Hz.
        print(f"  {'':>8} {'':>5} projected DSI {np.mean(PROJ):>+8.4f}"
              f"   rate {np.mean(TOTK) / 2:>7.2f} Hz"
              f"   dRate {np.mean(DIFF):>+7.3f} Hz")
        record["subtypes"][s] = {"n": len(AZ), "r_azimuth": r_az,
                                 "r_elevation": r_el, "r_length": r_len,
                                 "mean_abs_az_deg": float(np.mean(AZ)),
                                 "mean_abs_el_deg": float(np.mean(EL)),
                                 "mean_dsi": float(np.mean(DSI)), "n_driven": int(len(DSI)),
                                 "projected_dsi": float(np.mean(PROJ)),
                                 "global_dsi": float(np.mean(RAW)),
                                 "sign_balance": float(np.mean(SGN)),
                                 "geometry_linked_dsi": float(
                                     np.mean(PROJ) - np.mean(SGN) * np.mean(RAW)),
                                 "rate_hz": float(np.mean(TOTK) / 2),
                                 "rate_diff_hz": float(np.mean(DIFF)),
                                 "min_rate_hz": args.min_rate}
        if args.dump_cells:
            # PER CELL, so a later comparison can be made on the SAME cells.
            # Raising inhibition silences part of the population, and a
            # measurement taken on the survivors is a different measurement;
            # this is what makes the matched comparison possible at all.
            record["subtypes"][s]["cells"] = [
                [int(a), float(b), float(cc), float(dd)] for a, b, cc, dd in CELL]
    if pooled_proj:
        P = np.concatenate(pooled_proj)
        D = np.concatenate(pooled_diff)
        T = np.concatenate(pooled_tot)
        S = np.concatenate(pooled_sgn)
        R = np.concatenate(pooled_raw)
        se = float(P.std(ddof=1) / math.sqrt(len(P))) if len(P) > 1 else 0.0
        # CONTROL. A projected mean can be manufactured without any geometry:
        # if the whole population simply responds more to one direction (a
        # stimulus or boundary asymmetry) and the sign labels happen to be
        # unbalanced, sgn*DSI has a nonzero mean on no per-cell structure at
        # all. Split it: the part explained by the global bias alone is
        # mean(sgn)*mean(DSI); what is left is the covariance between a cell's
        # OWN geometry and its OWN preference, which is the only part a
        # correlator account predicts. That residual is exactly the mean of a
        # sign-shuffled null subtracted off.
        glob = float(S.mean() * R.mean())
        geo = float(P.mean() - glob)
        # sign-shuffle null: permute the geometry labels, keep the responses
        rng = np.random.default_rng(0)
        null = np.array([float((rng.permutation(S) * R).mean())
                         for _ in range(200)])
        print(f"\n  POOLED  n={len(P)}  projected DSI {P.mean():+.4f} "
              f"+- {se:.4f} (s.e.)   rate {T.mean() / 2:.2f} Hz   "
              f"dRate {D.mean():+.3f} Hz")
        print(f"          global bias {R.mean():+.4f} x sign balance "
              f"{S.mean():+.3f} = {glob:+.4f}   ->  GEOMETRY-LINKED "
              f"{geo:+.4f}   (shuffle null {null.mean():+.4f} "
              f"+- {null.std():.4f})")
        record["pooled"] = {"n": int(len(P)), "projected_dsi": float(P.mean()),
                            "projected_dsi_se": se,
                            "rate_hz": float(T.mean() / 2),
                            "rate_diff_hz": float(D.mean()),
                            "global_dsi": float(R.mean()),
                            "sign_balance": float(S.mean()),
                            "geometry_linked_dsi": geo,
                            "shuffle_null_mean": float(null.mean()),
                            "shuffle_null_sd": float(null.std())}
    print(paint("""
  The grating moves horizontally. If the geometric account is right,
  r(|azimuth|, |DSI|) should be clearly positive AND larger than
  r(|elevation|, |DSI|), because only the horizontal baseline can encode
  horizontal direction. If both are equal, something other than geometry is
  driving the relation and the n=4 result was a coincidence.""", "90"))
    if args.json:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
