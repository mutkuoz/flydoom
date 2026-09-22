#!/usr/bin/env python3
"""M17 — give the detector a shape. Does an ordered dendrite recover direction?

WHY THIS ONE
------------
Every failure in this project's motion pathway comes back to one measured fact:
the cell adds where it has to multiply. Shunting inhibition, which is the
multiplication, cancels itself in a point neuron -- opening the drain also
lowers the level, so each tap counts for less (/1.77) while pushing harder
(x1.85), for a net 1.06.

A point neuron is the assumption under that. Every input lands on the same
node, so WHERE on the cell an input arrives is not representable, and direction
is exactly that information: the same inputs in the order A-then-B must sum
differently from B-then-A. A real T4 orders its inputs along a cable, Mi9 at
the distal tip, Mi1 in the middle, Mi4 proximal, and that ordering is what
makes the arrangement a filter rather than a sum.

flydoom/compartments_chain.py builds exactly that and had never been run: the
integrator only expressed a PAIR of compartments, which cannot carry an order.
It now takes an edge list (tests/test_axial_chain.py checks a two-node chain
against the pair model it generalises).

WHAT IS MEASURED
----------------
Direction selectivity of each subtype under a drifting grating, for:

    point     the model as published, every input on one node
    chain     inputs placed along a cable by their own retinotopic offset,
              projected onto that cell's own correlator axis. No cell type is
              named in the placement rule.
    flipped   CONTROL. The same cable, distal and proximal exchanged. A result
              that survives this is not about dendritic ordering.
    shuffled  CONTROL. Each cell keeps the same inputs and the same number in
              each compartment; only which input lands where is permuted.
              Whatever survives this is not geometry either.

T4a/T4b and T5a/T5b are the horizontal pair the grating tests; c/d are the
vertical subtypes and should stay near zero on a horizontal grating, which is
a further check that the measurement is reading direction rather than drive.

    python experiments/m17_dendrite.py --spiking-t4 --optic-gain 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
from flydoom.lif import LIFNetwork, LIFParams  # noqa: E402
from flydoom.retina import Retina  # noqa: E402
from flydoom import compartments_chain as CC  # noqa: E402

from m3_optomotor import GratingRig  # noqa: E402

SUBTYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
# The cell's own correlator axis: its delayed arm against its fast one. Mi9
# alone for T4 and Tm9 alone for T5, never pooled with the proximal inhibition,
# which sits on the opposite flank and would cancel the axis (m3t).
AXIS_ARMS = {"T4": ("Mi9", "Mi1"), "T5": ("Tm9", "Tm1")}


def cells_of_type(graph, ann, name):
    d = ann.df
    f = d.filter((pl.col("primary_type") == name) | (pl.col("visual_type") == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                     if int(x) in pos], dtype=np.int64)


def visual_positions(graph, retina):
    """cell index -> (azimuth, elevation), and cell index -> eye."""
    colxy = {}
    for side, eye in retina.eyes.items():
        for cid, az, el in zip(eye.column_ids, eye.azimuth_deg,
                               eye.elevation_deg):
            colxy[(side, int(cid))] = (float(az), float(el))
    ca = pl.read_csv(Path(config.RAW_DIR) / "column_assignment.csv.gz")
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    out, side = {}, {}
    for rid, h, cid in zip(ca["root_id"], ca["hemisphere"], ca["column_id"]):
        i = pos.get(int(rid))
        pt = colxy.get((str(h), int(cid)))
        if i is not None and pt is not None:
            out[i] = pt
            side[i] = str(h)
    return out, side


def cell_axes(graph, ann, cell_pt):
    """cell -> (dx, dy): where its delayed arm sits relative to its fast one."""
    inputs = defaultdict(list)
    w = np.abs(graph.signed_syn)
    for a, b, c in zip(graph.pre_idx, graph.post_idx, w):
        if c > 0:
            inputs[int(b)].append((int(a), float(c)))
    axes = {}
    for st in SUBTYPES:
        slow, fast = AXIS_ARMS["T4" if st.startswith("T4") else "T5"]
        S = set(cells_of_type(graph, ann, slow).tolist())
        F = set(cells_of_type(graph, ann, fast).tolist())
        for c in cells_of_type(graph, ann, st):
            c = int(c)
            if c not in cell_pt:
                continue
            cen = {}
            for role, pool in (("slow", S), ("fast", F)):
                sx = sy = sw = 0.0
                for a, ww in inputs.get(c, ()):
                    if a in pool and a in cell_pt:
                        ax, ay = cell_pt[a]
                        sx += (ax - cell_pt[c][0]) * ww
                        sy += (ay - cell_pt[c][1]) * ww
                        sw += ww
                if sw > 0:
                    cen[role] = (sx / sw, sy / sw)
            if len(cen) == 2:
                axes[c] = (cen["slow"][0] - cen["fast"][0],
                           cen["slow"][1] - cen["fast"][1])
    return axes


def run_grating(net, rig, gext, duration, tf, device, watch, settle=0.5):
    dt = config.DT
    steps = int(round(duration / dt))
    skip = int(settle / dt)
    out = {}
    for sign, name in ((+1, "rightward"), (-1, "leftward")):
        net.reset()
        acc = np.zeros(len(watch))
        n = 0
        for step in range(steps):
            net.step(g_ext=gext,
                     out_set=rig.out_set(step * dt, tf, sign,
                                         net.p.graded_max_rate, dt))
            if step >= skip:
                acc += net.out[watch].detach().cpu().numpy() / dt
                n += 1
        out[name] = acc / max(n, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--tf", type=float, default=2.0)
    ap.add_argument("--period", type=float, default=30.0)
    ap.add_argument("--optic-gain", type=float, default=16.0)
    ap.add_argument("--spiking-t4", action="store_true", default=True)
    ap.add_argument("--n-comp", type=int, default=3)
    ap.add_argument("--g-axial", type=float, default=1.0)
    ap.add_argument("--nmda", type=float, default=0.0, metavar="FRAC",
                    help="voltage-dependent excitation, the preferred-direction "
                         "half of a correlator: g_e is multiplied by "
                         "(1 + FRAC * sigmoid((v - v_half)/k)), per compartment. "
                         "See config.NMDA_FRAC.")
    ap.add_argument("--shuffle-seeds", type=int, default=1, metavar="N",
                    help="how many independent shuffled cables to run. The "
                         "shuffle is the control that says whether retinotopic "
                         "ORDER matters, and one draw of it is one sample.")
    ap.add_argument("--nmda-k", type=float, default=None, metavar="VOLTS",
                    help="slope of the voltage dependence. A LARGE value (1.0) "
                         "pins the sigmoid at 0.5 whatever the voltage, which "
                         "is the drive-matched control: the same mean "
                         "amplification with no coincidence detection.")
    ap.add_argument("--slow-filter", type=float, default=None, metavar="TAU_MS",
                    help="deliver the slow arm as a one-pole low pass of this "
                         "time constant instead of a conduction delay. Delays "
                         "have been swept; the SHAPE of the arm has not.")
    ap.add_argument("--eye-map", default="anatomical",
                    choices=["lattice", "anatomical"])
    ap.add_argument("--json", type=Path)
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
    retina = Retina.build(g, ann, eye_map=args.eye_map)
    cell_pt, cell_side = visual_positions(g, retina)
    axes = cell_axes(g, ann, cell_pt)
    print(f"{len(axes):,} T4/T5 cells have a measurable correlator axis")

    graded = g.graded_mask(ann)
    if args.spiking_t4:
        for t in SUBTYPES:
            idx = cells_of_type(g, ann, t)
            if idx.size:
                graded[idx] = False
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=config.T_DLY_SLOW)
    pre_t, post_t, w_t = g.to_torch(args.device)

    watch_idx = {t: cells_of_type(g, ann, t) for t in SUBTYPES}
    watch = np.unique(np.concatenate(list(watch_idx.values())))
    wt = torch.as_tensor(watch, device=args.device)
    where = {int(c): k for k, c in enumerate(watch)}

    slow_kw = {}
    if args.slow_filter:
        slow_kw = dict(slow_filter_tau=args.slow_filter * 1e-3,
                       slow_delay_steps=int(round(config.T_DLY_SLOW / config.DT)))
        print(f"slow arm is a low pass, tau {args.slow_filter:g} ms, in place "
              f"of the {config.T_DLY_SLOW * 1e3:.0f} ms delay")

    kw_p = {"nmda_frac": args.nmda}
    if args.nmda_k:
        kw_p["nmda_k"] = args.nmda_k
    params = LIFParams(**kw_p) if args.nmda else None
    if args.nmda:
        k = args.nmda_k or config.NMDA_K
        print(f"supralinear dendrite: g_e x (1 + {args.nmda:g} * sigmoid), "
              f"half at {config.NMDA_V_HALF * 1e3:.0f} mV, slope {k * 1e3:g} mV"
              + ("   [FLAT: drive-matched control]" if k > 0.05 else ""))

    def build(plan):
        if plan is None:
            return LIFNetwork.from_graph(g, params=params, device=args.device,
                                         seed=0, edge_delay=edge_delay,
                                         graded=graded, **slow_kw)
        return LIFNetwork(plan["n_total"], pre_t,
                          torch.as_tensor(plan["post_idx"], device=args.device),
                          w_t, params, args.device, 0, edge_delay=edge_delay,
                          graded=CC.extend_graded(graded, plan),
                          axial_edges=plan["axial_edges"],
                          axial_edge_g=plan["axial_edge_g"], **slow_kw)

    arms = {"point": None}
    shuffles = [(f"shuffled{k}" if args.shuffle_seeds > 1 else "shuffled",
                 {"shuffle": k}) for k in range(args.shuffle_seeds)]
    for name, kw in [("chain", {}), ("flipped", {"flip": True})] + shuffles:
        arms[name] = CC.build_chain(g, ann, axes, cell_pt, n_comp=args.n_comp,
                                    g_axial=args.g_axial, **kw)
    shuf_names = [n for n, _ in shuffles]
    print(f"chain: {arms['chain']['n_cells']:,} cells x {args.n_comp} "
          f"compartments, {arms['chain']['n_moved']:,} of "
          f"{arms['chain']['n_edges_onto_targets']:,} inputs placed off centre; "
          f"per compartment {arms['chain']['per_compartment']}")

    record = {"n_comp": args.n_comp, "g_axial": args.g_axial,
              "nmda": args.nmda, "nmda_k": args.nmda_k,
              "optic_gain": args.optic_gain, "tf": args.tf,
              "period": args.period, "eye_map": args.eye_map, "arms": {}}
    print(f"\n{'arm':>9} " + "".join(f"{t:>9}" for t in SUBTYPES))
    for name, plan in arms.items():
        net = build(plan)
        rig = GratingRig(net, retina, args.device, args.period, 150.0)
        rates = run_grating(net, rig, None, args.duration, args.tf,
                            args.device, wt)
        row, dsis = f"{name:>9} ", {}
        for t in SUBTYPES:
            # PER EYE. Under the anatomical map the eyes are mirrored, so a
            # grating drifting rightward in the world runs front-to-back on
            # one eye and back-to-front on the other: pooling the two cancels
            # exactly the quantity being measured. The left eye's sign is
            # flipped back before averaging, so the number means "selectivity
            # along the cell's own eye" as it did under the lattice map.
            per_side = {}
            for sd in ("left", "right"):
                idx = [where[int(c)] for c in watch_idx[t]
                       if int(c) in where and cell_side.get(int(c)) == sd]
                if not idx:
                    continue
                r = rates["rightward"][idx].mean()
                l = rates["leftward"][idx].mean()
                d = (r - l) / (r + l) if (r + l) > 1e-9 else 0.0
                per_side[sd] = -d if sd == "left" else d
            dsi = float(np.mean(list(per_side.values()))) if per_side else 0.0
            dsis[t] = dsi
            row += f"{dsi:>+9.4f}"
        print(row)
        record["arms"][name] = dsis
        del net, rig

    print("\nthe horizontal pair is what a horizontal grating tests:")
    for group in ("T4", "T5"):
        a, b = f"{group}a", f"{group}b"
        print(f"  {group}: " + "   ".join(
            f"{n} {record['arms'][n][a] - record['arms'][n][b]:+.4f}"
            for n in arms))
    print("  (a minus b; a correlator pair must separate, and the controls "
          "must not)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
