#!/usr/bin/env python3
"""M3j — do the four T4 subtypes point in four different directions?

WHY THIS IS THE RIGHT QUESTION TO ASK BEFORE ANY MORE DYNAMICS
--------------------------------------------------------------
T4a/b/c/d are four detectors for four directions. What distinguishes them is
not their cell type but their GEOMETRY: where each one's delayed inhibitory arm
(Mi9, Mi4) sits in the retinotopic lattice relative to its fast excitatory arm
(Mi1, Tm3). That displacement vector IS the direction the cell detects, and
mirror-pair subtypes must carry OPPOSITE vectors.

This project has measured the arm offset pooled over all T4 cells (a mean phase
difference of 154 deg, which looks healthy) but never per subtype. The
distinction matters: a pooled offset can be large while the four subtypes all
point the SAME way, and in that case there is no four-direction system in the
extracted wiring at all. No amount of delay, gain, compartmentalisation or
operating point can produce mirrored outputs from unmirrored inputs -- so this
sits upstream of every hypothesis eliminated so far.

There is already a hint. Across the operating grid the mirror pairs oppose
BELOW chance: T4a and T4b agree on ~78% of points where a correlator requires
them to disagree. Unmirrored input geometry would produce exactly that.

METHOD
------
Pure arithmetic on the connectome and column_assignment.csv.gz; nothing is
simulated. For each cell: take its own ommatidial column, take the
synapse-count-weighted centroid of each arm's presynaptic columns, and subtract.
The correlator axis is centroid(inhibitory) - centroid(excitatory).

Hex lattice (p,q) is converted to Cartesian as x = p + q/2, y = q*sqrt(3)/2, so
angles are comparable up to one global rotation -- which cancels in every
comparison made here.

LEFT AND RIGHT ARE MIRROR IMAGES, so hemispheres are reported separately.
Pooling them would cancel the azimuthal component and manufacture the null
result this test exists to look for.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.cells import AnnotationTable  # noqa: E402
from flydoom.graph import ConnectomeGraph  # noqa: E402

T4_ARMS = {"exc": ("Mi1", "Tm3"), "inh": ("Mi9", "Mi4")}
T5_ARMS = {"exc": ("Tm1", "Tm2"), "inh": ("Tm9", "CT1")}
GROUPS = [("T4", ("T4a", "T4b", "T4c", "T4d"), T4_ARMS),
          ("T5", ("T5a", "T5b", "T5c", "T5d"), T5_ARMS)]


def cells_of_type(graph, ann, name):
    d = ann.df
    f = d.filter((pl.col("primary_type") == name) | (pl.col("visual_type") == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                     if int(x) in pos], dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)

    path = Path(config.RAW_DIR) / "column_assignment.csv.gz"
    ca = pl.read_csv(path)
    xy, hemi = {}, {}
    pos = {int(r): i for i, r in enumerate(g.root_ids)}
    for rid, h, p_, q_ in zip(ca["root_id"], ca["hemisphere"], ca["p"], ca["q"]):
        i = pos.get(int(rid))
        if i is None:
            continue
        xy[i] = (float(p_) + float(q_) / 2.0, float(q_) * math.sqrt(3.0) / 2.0)
        hemi[i] = str(h)

    pre, post, w = g.pre_idx, g.post_idx, np.abs(g.signed_syn)
    # inputs grouped by target, once
    inputs = defaultdict(list)
    for a, b, c in zip(pre, post, w):
        if c > 0:
            inputs[int(b)].append((int(a), float(c)))

    record = {"groups": {}}
    for gname, subtypes, arms in GROUPS:
        armidx = {role: set().union(*[set(cells_of_type(g, ann, t).tolist())
                                      for t in names]) or set()
                  for role, names in arms.items()}
        print(f"\n\033[1m{gname}: correlator axis = centroid(inhibitory) - "
              f"centroid(excitatory)\033[0m")
        print(f"  arms: exc={arms['exc']}  inh={arms['inh']}")
        print(f"  {'subtype':>8} {'side':>6} {'n':>5} {'axis angle':>11} "
              f"{'length':>8} {'consistency':>12}")
        record["groups"][gname] = {}
        for st in subtypes:
            cells = cells_of_type(g, ann, st)
            per_side = defaultdict(list)
            for c in cells:
                c = int(c)
                if c not in xy:
                    continue
                x0, y0 = xy[c]
                cen = {}
                for role in ("exc", "inh"):
                    sx = sy = sw = 0.0
                    for a, wt in inputs.get(c, ()):
                        if a in armidx[role] and a in xy:
                            ax, ay = xy[a]
                            sx += (ax - x0) * wt; sy += (ay - y0) * wt; sw += wt
                    if sw > 0:
                        cen[role] = (sx / sw, sy / sw)
                if len(cen) == 2:
                    per_side[hemi.get(c, "?")].append(
                        (cen["inh"][0] - cen["exc"][0],
                         cen["inh"][1] - cen["exc"][1]))
            for side, vecs in sorted(per_side.items()):
                if not vecs:
                    continue
                V = np.array(vecs)
                mean = V.mean(axis=0)
                ang = math.degrees(math.atan2(mean[1], mean[0])) % 360
                length = float(np.hypot(*mean))
                # how aligned are individual cells with the subtype mean?
                units = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
                consist = float(np.linalg.norm(units.mean(axis=0)))
                print(f"  {st:>8} {side:>6} {len(V):>5} {ang:>10.1f}deg "
                      f"{length:>8.3f} {consist:>12.2f}")
                record["groups"][gname].setdefault(st, {})[side] = {
                    "angle_deg": ang, "length": length,
                    "consistency": consist, "n": len(V)}

    print("\n\033[1mMIRROR CHECK — pairs must differ by ~180 deg\033[0m")
    for gname, subtypes, _ in GROUPS:
        d = record["groups"][gname]
        for a, b in ((subtypes[0], subtypes[1]), (subtypes[2], subtypes[3])):
            for side in ("left", "right"):
                if a in d and b in d and side in d[a] and side in d[b]:
                    diff = abs(d[a][side]["angle_deg"] - d[b][side]["angle_deg"]) % 360
                    diff = min(diff, 360 - diff)
                    verdict = "OK" if diff > 120 else ("WEAK" if diff > 60 else "NOT MIRRORED")
                    print(f"  {a} vs {b:<5} {side:>6}: {diff:6.1f} deg apart   {verdict}")
    print("\n  A four-direction system needs the four axes spread around the circle")
    print("  and each mirror pair near 180 deg apart. Consistency is the fraction")
    print("  of cells agreeing with their subtype's mean direction (1.0 = all).")

    if args.json:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
