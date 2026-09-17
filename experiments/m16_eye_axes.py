#!/usr/bin/env python3
"""M16 — which way does each eye face? Read it off the wiring.

WHY
---
The retina maps each ommatidial column to a viewing direction by stretching
the lattice's raw Cartesian coordinates to 170 x 150 degrees. Plotted in
visual space that gives each eye a slanted parallelogram, the same slant on
both sides, where a real right eye is the mirror image of the left. FlyWire's
column_assignment carries lattice topology, not an eye map, so the
orientation was never checked. The connectome can check it.

WHAT THE WIRING KNOWS
---------------------
T4/T5 subtypes detect four directions on the eye: a front-to-back, b
back-to-front, c upward, d downward (Maisak et al. 2013). Each dendrite points
AGAINST its preferred direction, with the delayed arm at the tip and the
null-side inhibition at the base (Takemura et al. 2017; Shinomiya et al.
2019): Mi9 at the tip of T4, Mi4 and C3 at its base; Tm9 at the tip of T5.
So, per cell, relative to its own column,

    T4 preferred direction  ~  offset(Mi4, C3) - offset(Mi9)
    T5 preferred direction  ~  offset(Tm1, Tm2) - offset(Tm9)

T4 and T5 are independent estimates of the same four directions. Where they
agree, the lattice direction of "front-to-back" and "up" is settled per eye,
and with it the mirror. CT1 is not columnar, so it cannot enter.

Two more checks, both independent of motion:
  * dorsal: the dorsal rim area. R7/R8 columns that synapse onto DmDRA1/2
    must sit at the dorsal margin, on the side the T4c/T5c "up" points to.
  * shape: FlyWire's (p, q) are axial hex coordinates, but axial coordinates
    come in two conventions, axes 60 or 120 degrees apart, and both describe a
    valid hex lattice. The retina assumed 60. The real eye is roughly round,
    so the convention that makes it round, and makes the four detector
    directions roughly perpendicular, is the physical one.

Pure arithmetic on the connectome; nothing is simulated.

    python experiments/m16_eye_axes.py --json paper/data/eye_axes.json
"""
from __future__ import annotations

import argparse
import json
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

ARMS = {
    "T4": {"base": ("Mi4", "C3"), "tip": ("Mi9",)},
    "T5": {"base": ("Tm1", "Tm2"), "tip": ("Tm9",)},
}
SUBTYPES = {"a": "front-to-back", "b": "back-to-front",
            "c": "upward", "d": "downward"}


def to_xy(dp, dq, convention):
    """Axial lattice offset -> Cartesian, axes `convention` degrees apart."""
    c = math.cos(math.radians(convention))
    s = math.sin(math.radians(convention))
    return dp + dq * c, dq * s


def ang(v):
    return math.degrees(math.atan2(v[1], v[0])) % 360


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    pos = {int(r): i for i, r in enumerate(g.root_ids)}
    ca = pl.read_csv(Path(config.RAW_DIR) / "column_assignment.csv.gz")

    col = {}                                   # neuron idx -> (p, q, side)
    for rid, h, p_, q_ in zip(ca["root_id"], ca["hemisphere"], ca["p"], ca["q"]):
        i = pos.get(int(rid))
        if i is not None:
            col[i] = (int(p_), int(q_), str(h))

    def of_type(*names):
        d = ann.df
        f = d.filter(pl.col("primary_type").is_in(names)
                     | pl.col("visual_type").is_in(names))
        return {pos[int(x)] for x in f["root_id"].unique().to_list()
                if int(x) in pos}

    inputs = defaultdict(list)
    for a, b, c in zip(g.pre_idx, g.post_idx, np.abs(g.signed_syn)):
        if c > 0:
            inputs[int(b)].append((int(a), float(c)))

    record = {"shape": {}, "directions": {}, "dorsal_rim": {}}

    # ---- shape under both conventions -----------------------------------
    print("\n\033[1mEYE SHAPE\033[0m  (principal axes of the column positions, "
          "in ommatidial spacings)")
    per_side = defaultdict(set)
    for p_, q_, h in col.values():
        per_side[h].add((p_, q_))
    for side, pts in sorted(per_side.items()):
        P = np.array(sorted(pts), float)
        for conv in (60, 120):
            X = np.stack(to_xy(P[:, 0], P[:, 1], conv), 1)
            X -= X.mean(0)
            ev, evec = np.linalg.eigh(np.cov(X.T))
            # full extent along each principal axis
            ext = [float(np.ptp(X @ evec[:, k])) for k in (1, 0)]
            long_ang = ang(evec[:, 1]) % 180
            print(f"  {side:>5} {conv:>3} deg axes: {len(P)} columns, "
                  f"{ext[0]:5.1f} x {ext[1]:5.1f} (aspect {ext[0] / ext[1]:.2f}), "
                  f"long axis at {long_ang:5.1f} deg")
            record["shape"].setdefault(side, {})[conv] = {
                "extent": ext, "aspect": ext[0] / ext[1], "long_axis_deg": long_ang}

    # ---- preferred directions -------------------------------------------
    print("\n\033[1mPREFERRED DIRECTIONS\033[0m  base-minus-tip input offset, "
          "mean over cells, as a lattice (dp, dq)")
    for grp, arms in ARMS.items():
        base, tip = of_type(*arms["base"]), of_type(*arms["tip"])
        for letter, meaning in SUBTYPES.items():
            cells = of_type(f"{grp}{letter}")
            vecs = defaultdict(list)
            for c in cells:
                if c not in col:
                    continue
                p0, q0, side = col[c]
                cen = {}
                for role, pool in (("base", base), ("tip", tip)):
                    sp = sq = sw = 0.0
                    for a, w in inputs.get(c, ()):
                        if a in pool and a in col and col[a][2] == side:
                            sp += (col[a][0] - p0) * w
                            sq += (col[a][1] - q0) * w
                            sw += w
                    if sw:
                        cen[role] = (sp / sw, sq / sw)
                if len(cen) == 2:
                    vecs[side].append((cen["base"][0] - cen["tip"][0],
                                       cen["base"][1] - cen["tip"][1]))
            for side, V in sorted(vecs.items()):
                V = np.array(V)
                m = V.mean(0)
                entry = {"n": len(V), "dp": float(m[0]), "dq": float(m[1])}
                line = f"  {grp}{letter} {meaning:>13} {side:>5} n={len(V):4d}  (dp,dq)=({m[0]:+.2f},{m[1]:+.2f})"
                for conv in (60, 120):
                    Xc = np.stack(to_xy(V[:, 0], V[:, 1], conv), 1)
                    u = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
                    mc = Xc.mean(0)
                    entry[conv] = {"angle_deg": ang(mc),
                                   "consistency": float(np.linalg.norm(u.mean(0)))}
                    line += f"   {conv}: {ang(mc):5.1f} deg (cons {entry[conv]['consistency']:.2f})"
                print(line)
                record["directions"].setdefault(f"{grp}{letter}", {})[side] = entry

    print("\n\033[1mAGREEMENT\033[0m  T4 vs T5, same subtype and eye "
          "(angle between them), and axis perpendicularity")
    for conv in (60, 120):
        for side in ("left", "right"):
            gaps, dirs = [], {}
            for letter in SUBTYPES:
                d4 = record["directions"].get(f"T4{letter}", {}).get(side)
                d5 = record["directions"].get(f"T5{letter}", {}).get(side)
                if d4 and d5:
                    a4, a5 = d4[conv]["angle_deg"], d5[conv]["angle_deg"]
                    gap = abs((a4 - a5 + 180) % 360 - 180)
                    gaps.append(gap)
                    # average the two as unit vectors
                    vx = math.cos(math.radians(a4)) + math.cos(math.radians(a5))
                    vy = math.sin(math.radians(a4)) + math.sin(math.radians(a5))
                    dirs[letter] = math.degrees(math.atan2(vy, vx)) % 360
            if len(dirs) == 4:
                h = (dirs["a"] - dirs["b"] + 180) % 360 - 180
                horiz = (dirs["b"] + h / 2 + 180) % 360     # front-to-back axis
                vtt = (dirs["c"] - dirs["d"] + 180) % 360 - 180
                vert = (dirs["d"] + vtt / 2 + 180) % 360    # upward axis
                perp = abs((horiz - vert + 180) % 360 - 180)
                print(f"  {conv:>3} deg {side:>5}: T4-T5 gaps "
                      f"{', '.join(f'{x:.0f}' for x in gaps)} deg; a/b {abs(h):.0f} "
                      f"apart, c/d {abs(vtt):.0f} apart; front-to-back at "
                      f"{horiz:.0f}, up at {vert:.0f}, {perp:.0f} deg between them")
                record.setdefault("axes", {}).setdefault(side, {})[conv] = {
                    "front_to_back_deg": horiz, "up_deg": vert,
                    "between_deg": perp, "t4_t5_gaps": gaps}

    # ---- dorsal rim --------------------------------------------------------
    print("\n\033[1mDORSAL RIM\033[0m  R7/R8 columns presynaptic to DmDRA1/2, "
          "relative to the eye centre")
    dra = of_type("DmDRA1", "DmDRA2")
    photo = of_type("R7", "R8")
    rim = defaultdict(lambda: defaultdict(float))
    for c in dra:
        for a, w in inputs.get(c, ()):
            if a in photo and a in col:
                p_, q_, side = col[a]
                rim[side][(p_, q_)] += w
    for side, cols in sorted(rim.items()):
        P = np.array(sorted(per_side[side]), float)
        ctr = P.mean(0)
        R = np.array(list(cols.keys()), float)
        W = np.array(list(cols.values()))
        top = R[np.argsort(-W)[:max(5, len(R) // 4)]]      # the strongest quarter
        d = top.mean(0) - ctr
        entry = {"n_columns": len(R), "dp": float(d[0]), "dq": float(d[1])}
        line = f"  {side:>5}: {len(R)} columns, strongest at (dp,dq)=({d[0]:+.1f},{d[1]:+.1f})"
        for conv in (60, 120):
            v = to_xy(d[0], d[1], conv)
            entry[conv] = ang(v)
            up = record.get("axes", {}).get(side, {}).get(conv, {}).get("up_deg")
            off = "" if up is None else f" ({abs((ang(v) - up + 180) % 360 - 180):.0f} from 'up')"
            line += f"   {conv}: {ang(v):5.1f} deg{off}"
        print(line)
        record["dorsal_rim"][side] = entry

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
