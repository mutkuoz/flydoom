#!/usr/bin/env python3
"""M3t — is a T4 subtype one population or a mixture?

M3j reported a MEAN axis per subtype. A mean is only meaningful if the thing
averaged is unimodal, and two facts say it may not be here:

  * T4a and T4b sit 97-114 deg apart. Mirror subtypes must be 180.
  * The four T4 axes fall at 20, 156, 216, 261 deg -- gaps of 136, 60, 45 and
    119, where four cardinal detectors should be spread ~90 apart.
  * Per-cell consistency is 0.19-0.32 for T4 against 0.58-0.86 for T5.

A low consistency has two very different explanations. Either each cell's
geometry is genuinely noisy, or the subtype is a MIXTURE of two populations
pointing opposite ways -- which would make the mean an average of two peaks
and put it somewhere neither population actually points. That second case is
not a modelling failure at all; it is a labelling or lattice problem upstream
of every simulation, and it would invalidate the mean axis, the mirror
comparison, and any conclusion drawn from either.

The two are easy to tell apart: look at the DISTRIBUTION, not the mean.

  unimodal + wide   -> noisy geometry, mean is meaningful but imprecise
  bimodal ~180 apart -> a mixture; the subtype label groups two opposite cells

Reported per subtype and hemisphere: the circular histogram of per-cell axis
angles, the resultant length R (1 = all identical), and a Rayleigh test against
uniform, plus the two-peak split that best explains the data.
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

# Mi9 and Mi4 sit on OPPOSITE flanks of T4 (Mi9 at (-0.76,+0.26), Mi4 at
# (+0.39,-0.60)). Averaging them into one "inhibitory centroid" averages two
# opposite vectors, so the result flips with their relative weight per cell and
# manufactures bimodality that is an artefact of the arm definition, not of the
# wiring. The correlator axis is Mi9 -- the null-side inhibition -- against the
# centred excitation, so it is taken alone by default.
ARMS = {"T4": {"exc": ("Mi1",), "inh": ("Mi9",)},
        "T5": {"exc": ("Tm1",), "inh": ("Tm9",)}}
ARMS_POOLED = {"T4": {"exc": ("Mi1", "Tm3"), "inh": ("Mi9", "Mi4")},
               "T5": {"exc": ("Tm1", "Tm2"), "inh": ("Tm9", "CT1")}}
GROUPS = [("T4", ("T4a", "T4b", "T4c", "T4d")),
          ("T5", ("T5a", "T5b", "T5c", "T5d"))]


def cells_of_type(graph, ann, name):
    d = ann.df
    f = d.filter((pl.col("primary_type") == name) | (pl.col("visual_type") == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                     if int(x) in pos], dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", action="store_true",
                    help="use the pooled arm definition (Mi9+Mi4 together), "
                         "which averages two opposite flanks and is what "
                         "produced the apparent bimodality.")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    ca = pl.read_csv(Path(config.RAW_DIR) / "column_assignment.csv.gz")
    pos = {int(r): i for i, r in enumerate(g.root_ids)}
    xy, hemi = {}, {}
    for rid, h, p_, q_ in zip(ca["root_id"], ca["hemisphere"], ca["p"], ca["q"]):
        i = pos.get(int(rid))
        if i is None:
            continue
        xy[i] = (float(p_) + float(q_) / 2.0, float(q_) * math.sqrt(3.0) / 2.0)
        hemi[i] = str(h)

    pre, post, w = g.pre_idx, g.post_idx, np.abs(g.signed_syn)
    inputs = defaultdict(list)
    for x, y, c in zip(pre, post, w):
        if c > 0:
            inputs[int(y)].append((int(x), float(c)))

    record = {}
    for gname, subtypes in GROUPS:
        arms = (ARMS_POOLED if a.pooled else ARMS)[gname]
        members = {r: set().union(*[set(cells_of_type(g, ann, t).tolist())
                                    for t in names])
                   for r, names in arms.items()}
        print(f"\n\033[1m{gname} — per-cell axis distribution\033[0m")
        print(f"  {'subtype':>8} {'side':>6} {'n':>5} {'R':>6} {'Rayleigh p':>11}"
              f" {'mean':>7} {'2-peak split':>22} {'bimodal?':>9}")
        for st in subtypes:
            for side in ("left", "right"):
                ang = []
                for c in cells_of_type(g, ann, st):
                    c = int(c)
                    if c not in xy or hemi.get(c) != side:
                        continue
                    x0, y0 = xy[c]
                    cen = {}
                    for role in ("exc", "inh"):
                        sx = sy = sw = 0.0
                        for p_, ww in inputs.get(c, ()):
                            if p_ in members[role] and p_ in xy:
                                ax, ay = xy[p_]
                                sx += (ax - x0) * ww; sy += (ay - y0) * ww; sw += ww
                        if sw > 0:
                            cen[role] = (sx / sw, sy / sw)
                    if len(cen) == 2:
                        dx = cen["inh"][0] - cen["exc"][0]
                        dy = cen["inh"][1] - cen["exc"][1]
                        if dx or dy:
                            ang.append(math.atan2(dy, dx))
                if len(ang) < 20:
                    continue
                A = np.asarray(ang)
                C, S = np.cos(A).mean(), np.sin(A).mean()
                R = math.hypot(C, S)
                n = len(A)
                # Rayleigh test for non-uniformity
                Z = n * R * R
                p_ray = math.exp(-Z) * (1 + (2 * Z - Z * Z) / (4 * n))
                mean = math.degrees(math.atan2(S, C)) % 360
                # best two-peak split: does splitting the circle in half
                # explain the cells better than one mean?
                best = None
                for cut in np.arange(0, math.pi, math.pi / 36):
                    side1 = A[np.cos(A - cut) >= 0]
                    side2 = A[np.cos(A - cut) < 0]
                    if len(side1) < 5 or len(side2) < 5:
                        continue
                    r1 = math.hypot(np.cos(side1).mean(), np.sin(side1).mean())
                    r2 = math.hypot(np.cos(side2).mean(), np.sin(side2).mean())
                    score = (len(side1) * r1 + len(side2) * r2) / n
                    if best is None or score > best[0]:
                        m1 = math.degrees(math.atan2(np.sin(side1).mean(),
                                                     np.cos(side1).mean())) % 360
                        m2 = math.degrees(math.atan2(np.sin(side2).mean(),
                                                     np.cos(side2).mean())) % 360
                        best = (score, m1, m2, len(side1), len(side2))
                gain = best[0] - R if best else 0.0
                sep = abs(best[1] - best[2]) % 360 if best else 0
                sep = min(sep, 360 - sep)
                bim = "YES" if gain > 0.15 and sep > 120 else ("maybe" if gain > 0.10 else "no")
                print(f"  {st:>8} {side:>6} {n:>5} {R:>6.2f} {p_ray:>11.1e}"
                      f" {mean:>6.0f}d {best[1]:>7.0f}d/{best[2]:<5.0f}d"
                      f" ({best[3]:>4}/{best[4]:<4}) {bim:>9}")
                record[f"{st}_{side}"] = {
                    "n": n, "R": R, "rayleigh_p": p_ray, "mean_deg": mean,
                    "peak1_deg": best[1], "peak2_deg": best[2],
                    "n1": best[3], "n2": best[4], "split_gain": gain,
                    "peak_sep_deg": sep, "bimodal": bim}
    print("""
  R is the resultant length: 1.0 = every cell points the same way, 0 = uniform.
  'split gain' is how much better two peaks describe the cells than one mean;
  a large gain with peaks ~180 apart means the subtype LABEL is grouping two
  opposite populations, and its mean axis is an artefact of averaging them.""")
    if a.json:
        import json
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
