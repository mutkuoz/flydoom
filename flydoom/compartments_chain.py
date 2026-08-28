"""Two-compartment T4/T5 dendrites.

WHY
---
A point neuron makes every synapse electrically identical. Inhibition in a
conductance model divides rather than subtracts -- that part is right, and it
is what makes a product-like interaction available at all -- but it divides the
WHOLE cell. Null-direction suppression in T4 is local: the inhibitory arm must
divide the branch carrying the excitatory arm, and nothing else. A cell that
shunts globally cannot express that, however correct its wiring.

Two compartments are the minimum model of the distinction: a soma that spikes
and a passive dendrite, joined by an axial conductance (see LIFNetwork's
`axial_partner`). At g_ax = 0 they are independent; as g_ax grows they merge
back into the point neuron, so the point neuron is the large-g_ax limit of this
model rather than a different one.

WHAT ASSIGNS A SYNAPSE TO A COMPARTMENT
---------------------------------------
Retinotopic position, taken from `column_assignment.csv.gz`, and nothing else:

    presynaptic cell in the SAME column as the T4  ->  soma
    presynaptic cell in a DIFFERENT column         ->  dendrite

This is deliberately a geometric rule and not a list of cell types. It happens
to place the centred excitatory arm (Mi1, Tm3) at the soma and the flanking
inhibitory arm (Mi9, Mi4) on the dendrite, but that is a CONSEQUENCE of the
measured geometry rather than an input to it -- which matters, because choosing
per-type placement by hand would be fitting the answer.

Inputs whose presynaptic cell has no column assignment (central-brain feedback,
mostly) stay at the soma: absent evidence for placing them distally, the
conservative choice is the one that changes nothing.

HONEST LIMIT
------------
FlyWire publishes per-synapse coordinates, but `coordinates.csv.gz` in this
distribution carries one point per NEURON, not per synapse. So the compartment
boundary here is inferred from retinotopy, not read off the morphology. Two
compartments with a single coupling constant is also the coarsest possible
dendrite. Both are approximations to state, not results to hide.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from flydoom import config

T4T5 = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def cells_of_type(graph, ann, name: str) -> np.ndarray:
    d = ann.df
    f = d.filter((pl.col("primary_type") == name)
                 | (pl.col("visual_type") == name))
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    return np.array([pos[int(x)] for x in f["root_id"].unique().to_list()
                     if int(x) in pos], dtype=np.int64)


def _index_to_column(graph, raw_dir) -> dict[int, int]:
    path = Path(raw_dir) / "column_assignment.csv.gz"
    if not path.exists():
        path = Path(raw_dir) / "column_assignment.csv"
    ca = pl.read_csv(path)
    rid2col = {int(r): int(c) for r, c in zip(ca["root_id"].to_list(),
                                              ca["column_id"].to_list())}
    out = {}
    for i, rid in enumerate(graph.root_ids):
        c = rid2col.get(int(rid))
        if c is not None:
            out[i] = c
    return out


def build(graph, ann, types: tuple[str, ...] = T4T5, g_axial: float = 1.0,
          raw_dir=None) -> dict:
    """Split `types` into soma + dendrite and re-route off-column inputs.

    Dendrites are appended AFTER every existing neuron, so all pre-existing
    indices -- readouts, motor populations, retina injection sites -- keep
    meaning exactly what they meant before.
    """
    raw_dir = raw_dir or config.RAW_DIR
    n = graph.n_neurons
    idx2col = _index_to_column(graph, raw_dir)

    targets = np.unique(np.concatenate(
        [cells_of_type(graph, ann, t) for t in types])) if types else np.zeros(0, np.int64)
    targets = np.array([t for t in targets if int(t) in idx2col], dtype=np.int64)
    if targets.size == 0:
        raise ValueError("no target cells with a column assignment")

    dend_of = np.full(n, -1, dtype=np.int64)
    dend_of[targets] = np.arange(n, n + targets.size, dtype=np.int64)
    n_total = int(n + targets.size)

    pre, post = graph.pre_idx.astype(np.int64), graph.post_idx.astype(np.int64)
    col = np.full(n, -1, dtype=np.int64)
    for i, c in idx2col.items():
        col[i] = c

    is_target = dend_of[post] >= 0
    known = (col[pre] >= 0) & (col[post] >= 0)
    off_column = col[pre] != col[post]
    move = is_target & known & off_column

    post_new = post.copy()
    post_new[move] = dend_of[post[move]]

    axial_partner = np.arange(n_total, dtype=np.int64)
    axial_partner[targets] = dend_of[targets]
    axial_partner[dend_of[targets]] = targets
    g_ax = np.zeros(n_total, dtype=np.float32)
    g_ax[targets] = g_axial
    g_ax[dend_of[targets]] = g_axial

    return {
        "n_total": n_total,
        "post_idx": post_new.astype(np.int64),
        "axial_partner": axial_partner,
        "g_ax": g_ax,
        "soma_idx": targets,
        "dend_idx": dend_of[targets],
        "n_moved": int(move.sum()),
        "n_edges_onto_targets": int(is_target.sum()),
        "n_cells": int(targets.size),
    }


def extend_graded(graded: np.ndarray | None, plan: dict) -> np.ndarray:
    """Dendrites are passive: they integrate and conduct, they do not spike."""
    n_total, dend = plan["n_total"], plan["dend_idx"]
    out = np.zeros(n_total, dtype=bool)
    if graded is not None:
        out[:len(graded)] = graded
    out[dend] = True
    return out


# ==========================================================================
# A CHAIN dendrite: position along the cable encodes position along the
# cell's own motion axis.
# ==========================================================================
#
# WHY THIS EXISTS AND WHAT WAS WRONG WITH TWO
# -------------------------------------------
# `build` above lumps EVERY off-column input into one dendrite. Mi9 arriving
# from the left flank and Mi9 arriving from the right flank land in the same
# compartment and shunt the same node, so that compartment carries no
# information about WHICH SIDE an input came from. Direction is exactly that
# information, so a two-compartment cell cannot express it however strong the
# coupling.
#
# A real T4 dendrite orders its inputs along its length: Mi9 at the distal
# tip, Mi1 in the middle, Mi4 proximal. That ordering is what makes the
# arrangement a spatial filter rather than a sum. Here the ordering is
# RECOVERED rather than imposed -- each input's compartment is its column
# offset projected onto that cell's own correlator axis -- and what comes out
# of the geometry is
#
#     Mi9 median +3.5 deg    Mi1 median 0.0 deg    Mi4 median -2.3 deg
#
# at a measured column spacing of 3.66 deg: one column distal, centre, one
# column proximal. No cell type is named anywhere in the placement rule, so
# the agreement is a result rather than an input.
#
# The axis is the cell's OWN Mi9-against-Mi1 vector (Tm9-against-Tm1 for T5),
# the same one m3k projects its DSI onto. Mi9 alone, never pooled with Mi4:
# they sit on opposite flanks and averaging them corrupts the axis (m3t).
#
# The soma sits at the PROXIMAL end (most negative projection, the Mi4 side)
# and is the only compartment that spikes; the distal tip is the Mi9 side.
# `flip=True` reverses that, and is the control: a result that survives the
# reversal is not about dendritic ordering.


def build_chain(graph, ann, axis: dict, cell_pt: dict, n_comp: int = 3,
                g_axial: float = 1.0, bin_deg: float = 3.66,
                flip: bool = False, shuffle: int | None = None) -> dict:
    """Split each cell in `axis` into a chain of `n_comp` compartments.

    axis     cell index -> (dx, dy), the cell's correlator axis in degrees of
             visual angle. Only cells appearing here are split; a cell whose
             axis is not measurable stays a point neuron.
    cell_pt  cell index -> (azimuth, elevation) in degrees.
    bin_deg  compartment width along the axis. The default is the measured
             inter-column spacing, so one compartment is one column.

    Compartment 0 is the spiking soma at the proximal end; compartment
    n_comp-1 is the distal tip. Returns the same shape of plan as `build`,
    with `axial_edges` in place of `axial_partner`, because a chain node has
    two neighbours where a pair has one.
    """
    if n_comp < 3:
        raise ValueError("n_comp must be at least 3; 2 is `build` above")
    if n_comp % 2 == 0:
        raise ValueError("n_comp must be ODD so one compartment sits at "
                         "offset zero, where the centred excitatory arm is")
    half = (n_comp - 1) // 2
    n = graph.n_neurons

    targets = np.array(sorted(int(c) for c in axis
                              if int(c) in cell_pt
                              and math.hypot(*axis[int(c)]) > 0),
                       dtype=np.int64)
    if targets.size == 0:
        raise ValueError("no target cells with a measurable axis")

    # comp[t, p]: index of compartment p of target t. p = 0 IS the original
    # neuron and the dendrites are appended after every existing neuron, so
    # all pre-existing indices -- readouts, motor populations, retina
    # injection sites -- keep meaning exactly what they meant before.
    comp = np.zeros((targets.size, n_comp), dtype=np.int64)
    comp[:, 0] = targets
    extra = np.arange(n, n + targets.size * (n_comp - 1), dtype=np.int64)
    comp[:, 1:] = extra.reshape(targets.size, n_comp - 1)
    n_total = int(n + targets.size * (n_comp - 1))

    row_of = np.full(n, -1, dtype=np.int64)
    row_of[targets] = np.arange(targets.size)

    ux = np.zeros(targets.size); uy = np.zeros(targets.size)
    x0 = np.zeros(targets.size); y0 = np.zeros(targets.size)
    for r_, t in enumerate(targets):
        dx, dy = axis[int(t)]
        L = math.hypot(dx, dy)
        ux[r_], uy[r_] = dx / L, dy / L
        x0[r_], y0[r_] = cell_pt[int(t)]

    has_pt = np.zeros(n, dtype=bool)
    px = np.zeros(n); py = np.zeros(n)
    for i, (a_, b_) in cell_pt.items():
        if 0 <= int(i) < n:
            has_pt[int(i)] = True
            px[int(i)], py[int(i)] = a_, b_

    pre, post = graph.pre_idx.astype(np.int64), graph.post_idx.astype(np.int64)
    rows = row_of[post]
    sel = rows >= 0
    r = rows[sel]
    p = pre[sel]
    known = has_pt[p]

    # signed distance along the cell's own axis, in degrees of visual angle
    s = np.zeros(r.size)
    s[known] = ((px[p[known]] - x0[r[known]]) * ux[r[known]]
                + (py[p[known]] - y0[r[known]]) * uy[r[known]])
    # An input whose presynaptic cell has no column (central feedback, mostly)
    # carries no retinotopic evidence. It goes to the soma, which is what the
    # two-compartment model does with it: absent evidence, change nothing.
    k = np.clip(np.rint(s / bin_deg), -half, half).astype(np.int64)
    slot = (half - k) if flip else (k + half)
    slot[~known] = 0

    if shuffle is not None:
        # CONTROL. Permute each cell's slots AMONG ITS OWN INPUTS. Every cell
        # keeps exactly the same number of synapses in exactly the same
        # compartments, and every compartment keeps its electrical position;
        # only the correspondence between an input's retinotopic offset and
        # where it lands is destroyed. Whatever survives this is not geometry.
        rng = np.random.default_rng(shuffle)
        by_cell = np.lexsort((np.arange(r.size), r))
        randomly = np.lexsort((rng.random(r.size), r))
        shuffled = np.empty_like(slot)
        shuffled[randomly] = slot[by_cell]
        slot = shuffled

    post_new = post.copy()
    post_new[sel] = comp[r, slot]

    # axial edges: consecutive compartments only, one conductance throughout
    a_lo = comp[:, :-1].reshape(-1)
    a_hi = comp[:, 1:].reshape(-1)

    counts = np.bincount(slot, minlength=n_comp)
    return {
        "n_total": n_total,
        "post_idx": post_new.astype(np.int64),
        "axial_edges": np.stack([a_lo, a_hi]),
        "axial_edge_g": np.full(a_lo.size, float(g_axial), dtype=np.float32),
        "soma_idx": targets,
        "dend_idx": extra,
        "comp": comp,
        "n_comp": int(n_comp),
        "bin_deg": float(bin_deg),
        "g_axial": float(g_axial),
        "flip": bool(flip),
        "shuffle": shuffle,
        "n_cells": int(targets.size),
        "n_edges_onto_targets": int(sel.sum()),
        "n_no_column": int((~known).sum()),
        "n_moved": int((slot != 0).sum()),
        "per_compartment": counts.tolist(),
        "proj_deg": s,
        "slot": slot,
        "edge_pre": p,
    }


def chain_placement_report(plan: dict, graph, ann,
                           types=("Mi1", "Mi9", "Mi4", "Tm1", "Tm9",
                                  "Tm2")) -> str:
    """Where did each input type actually land?

    Placement is geometric and type-blind, so this is a RESULT, not a setting:
    it is the check that the geometry reproduces the published Mi9-distal /
    Mi1-middle / Mi4-proximal order without being told to.
    """
    lines = [f"  chain: {plan['n_comp']} compartments, bin {plan['bin_deg']:.2f} deg,"
             f" g_ax {plan['g_axial']:g}, soma at the "
             f"{'DISTAL end (FLIPPED control)' if plan['flip'] else 'proximal end'}"
             + (f", SHUFFLED (seed {plan['shuffle']}): compartment counts "
                f"kept, geometry destroyed" if plan.get("shuffle") is not None
                else ""),
             f"  {plan['n_cells']:,} cells split; {plan['n_edges_onto_targets']:,}"
             f" input edges, {plan['n_no_column']:,} with no column -> soma;"
             f" per compartment {plan['per_compartment']}"]
    idx2type = {}
    for t in types:
        for c in cells_of_type(graph, ann, t):
            idx2type[int(c)] = t
    p, slot, s = plan["edge_pre"], plan["slot"], plan["proj_deg"]
    tvec = np.array([idx2type.get(int(x), "") for x in p])
    for t in types:
        m = tvec == t
        if not m.any():
            continue
        hist = np.bincount(slot[m], minlength=plan["n_comp"])
        lines.append(f"    {t:>4} n={int(m.sum()):>6}  median s "
                     f"{np.median(s[m]):+5.2f} deg  compartments {hist.tolist()}")
    return "\n".join(lines)
