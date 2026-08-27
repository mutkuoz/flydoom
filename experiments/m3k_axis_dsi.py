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

ARMS = {"T4": {"exc": ("Mi1", "Tm3"), "inh": ("Mi9", "Mi4")},
        "T5": {"exc": ("Tm1", "Tm2"), "inh": ("Tm9", "CT1")}}
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
    ap.add_argument("--min-rate", type=float, default=20.0,
                    help="minimum R+L in Hz for a cell to enter the "
                         "correlation. Below this, DSI is a ratio of two "
                         "near-zero numbers and carries no information.")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
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
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=config.T_DLY_SLOW)
    net = LIFNetwork.from_graph(g, device=args.device, seed=0,
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
              "subtypes": {}}
    for s in SUBTYPES:
        group = ARMS["T4" if s.startswith("T4") else "T5"]
        exc = set().union(*[set(cells_of_type(g, ann, t).tolist()) for t in group["exc"]])
        inh = set().union(*[set(cells_of_type(g, ann, t).tolist()) for t in group["inh"]])
        AZ, EL, LEN, DSI, TOT = [], [], [], [], []
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
        if len(AZ) < 10:
            continue
        # DSI is a ratio and inflates without bound as its denominator vanishes;
        # a cell firing 0.01 Hz against 0.02 Hz scores 0.33 on no signal at all.
        # Correlating geometry against that measures noise. Restrict to cells
        # that are actually driven, as the per-cell control in M3 does.
        AZ, EL, LEN, DSI, TOT = map(np.asarray, (AZ, EL, LEN, DSI, TOT))
        keep = TOT > args.min_rate
        n_all = len(AZ)
        AZ, EL, LEN, DSI = AZ[keep], EL[keep], LEN[keep], DSI[keep]
        if len(AZ) < 10:
            print(f"  {s:>8} {0:>5}   -- no cells above {args.min_rate} Hz total")
            continue
        r_az, r_el, r_len = pearson(AZ, DSI), pearson(EL, DSI), pearson(LEN, DSI)
        print(f"  {s:>8} {len(AZ):>5} {r_az:>15.3f} {r_el:>15.3f} {r_len:>14.3f}"
              f" {np.mean(AZ):>13.2f}   ({len(AZ)}/{n_all} driven, "
              f"mean|DSI| {DSI.mean():.4f})")
        record["subtypes"][s] = {"n": len(AZ), "r_azimuth": r_az,
                                 "r_elevation": r_el, "r_length": r_len,
                                 "mean_abs_az_deg": float(np.mean(AZ)),
                                 "mean_abs_el_deg": float(np.mean(EL)),
                                 "mean_dsi": float(np.mean(DSI)), "n_driven": int(len(DSI)),
                                 "min_rate_hz": args.min_rate}
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
