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
