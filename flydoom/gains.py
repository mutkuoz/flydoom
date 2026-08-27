"""Per-cell-type synaptic gains, transplanted from an optimized model.

THIS IS AN ABLATION, NOT PART OF THE MODEL. Everything else in this project
holds the connectome frozen and fits exactly one scalar, W_SYN, on a non-visual
behaviour. That is the point: it measures how much behaviour the wiring alone
implies. The measured obstacle to visual motion is a RATIO -- T4a's fast
excitatory arm starved 37:1, LPLC2 held below rest by inhibition roughly 10:1 --
and every input-side scale we can reach multiplies both arms and cancels out of
a ratio. What would not cancel is a gain that differs BETWEEN cell-type pairs.

The connectome does not supply those. Lappalainen et al. (2024) fit them: one
scaling factor per type-to-type connection, 604 of them, by gradient descent on
optic flow. Their trained ensemble is public, so rather than assert that
per-type gain is the missing quantity we can transplant theirs and re-measure.

WHAT THIS CAN AND CANNOT SHOW
-----------------------------
It is a transplant between datasets. Their gains come from FIB-25/FIB-19, ours
from FlyWire FAFB, matched on shared cell-type names -- 49 of 59 types after
aliasing, covering 24% of optic-lobe edges and 30% of optic-lobe synapses. The
correlator pathway itself is fully covered (Mi1, Tm3, Mi9, Mi4, CT1 onto T4a,
and L1 onto Mi1), which is what the hypothesis is about.

So a POSITIVE result is strong: if direction selectivity appears, relative
per-type gain was the missing ingredient and we have identified it precisely. A
NEGATIVE result is weaker, because partial coverage and a cross-dataset mapping
could each explain it. Both are worth reporting; only the first is conclusive.

We normalise the transplanted gains to a synapse-weighted mean of 1 over the
edges they cover, so the ablation changes the RELATIVE balance between type
pairs and leaves the global gain at the W_SYN fitted on taste. One variable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


def type_names(graph, ann) -> np.ndarray:
    """Cell type per neuron index, empty string where unknown."""
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    out = np.full(graph.n_neurons, "", dtype=object)
    d = ann.df
    for r, primary, visual in zip(d["root_id"].to_list(),
                                  d["primary_type"].to_list(),
                                  d["visual_type"].to_list()):
        i = pos.get(int(r))
        if i is not None:
            out[i] = primary or visual or ""
    return out


def edge_multipliers(graph, ann, gains_path: Path | str) -> tuple[np.ndarray, dict]:
    """Per-edge gain multiplier, 1.0 where no transplanted gain applies.

    Returns (multiplier[E], report).
    """
    rec = json.loads(Path(gains_path).read_text())
    gains = {k: v["k"] for k, v in rec["gains"].items()}

    tname = type_names(graph, ann)
    pre_t = tname[graph.pre_idx]
    post_t = tname[graph.post_idx]

    mult = np.ones(len(graph.pre_idx), dtype=np.float32)
    covered = np.zeros(len(graph.pre_idx), dtype=bool)
    for i, (a, b) in enumerate(zip(pre_t, post_t)):
        if not a or not b:
            continue
        k = gains.get(f"{a}->{b}")
        if k is not None:
            mult[i] = k
            covered[i] = True

    # Normalise over the covered edges only, weighted by synapse count, so the
    # transplant is a pure redistribution and not a global gain change.
    syn = np.abs(graph.signed_syn)
    if covered.any():
        wmean = float((mult[covered] * syn[covered]).sum() / syn[covered].sum())
        if wmean > 0:
            mult[covered] /= wmean

    report = {
        "source": rec.get("source"),
        "connectome": rec.get("connectome"),
        "n_gains": len(gains),
        "edges_covered": int(covered.sum()),
        "edges_total": int(len(covered)),
        "synapses_covered_frac": float(syn[covered].sum() / syn.sum()),
        "multiplier_min": float(mult[covered].min()) if covered.any() else 1.0,
        "multiplier_max": float(mult[covered].max()) if covered.any() else 1.0,
    }
    return mult, report


def optic_gain_multipliers(graph, ann, gain: float) -> np.ndarray:
    """Per-edge multiplier that scales synapses ONTO the visual populations.

    Applied to exactly the set that used to receive the tonic bias
    (config.BIASED_SUPER_CLASSES), so this is a swap of one global scalar for
    another rather than an extra degree of freedom. See config.OPTIC_GAIN.
    """
    from . import config
    cls = pl.read_csv(Path(config.RAW_DIR) / "classification.csv.gz",
                      infer_schema_length=50_000)
    ids = cls.filter(
        pl.col("super_class").is_in(list(config.BIASED_SUPER_CLASSES))
    )["root_id"].to_list()
    pos = {int(r): i for i, r in enumerate(graph.root_ids)}
    target = np.zeros(graph.n_neurons, dtype=bool)
    for r in ids:
        i = pos.get(int(r))
        if i is not None:
            target[i] = True
    mult = np.ones(len(graph.pre_idx), dtype=np.float32)
    mult[target[graph.post_idx]] = float(gain)
    return mult


def gain_onto_types(graph, ann, types, gain: float) -> np.ndarray:
    """Per-edge multiplier scaling synapses onto NAMED cell types only.

    ABLATION, and a more permissive one than optic_gain_multipliers: selecting
    which cell types get the boost is a step toward the per-type gains this
    project exists to measure the absence of. It is used to ask a diagnostic
    question -- does direction selectivity return if only the motion detectors
    are driven harder? -- and never as part of the reported model.
    """
    tname = type_names(graph, ann)
    want = set(types)
    target = np.array([t in want for t in tname], dtype=bool)
    mult = np.ones(len(graph.pre_idx), dtype=np.float32)
    mult[target[graph.post_idx]] = float(gain)
    return mult
