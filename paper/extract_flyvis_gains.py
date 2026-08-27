#!/usr/bin/env python3
"""Extract per-cell-type-pair synaptic gains from the flyvis pretrained ensemble.

The gains this project deliberately does NOT fit are the ones Lappalainen et al.
(2024) DO fit -- one scaling factor per type-to-type connection, 604 of them.
Their trained ensemble is public, so the hypothesis "per-type gain is what is
missing" can be tested rather than asserted: transplant their relative gains
into the frozen model and re-measure direction selectivity.

Provenance and alignment, both verified rather than assumed:

  * source: results_pretrained_models.zip from the flyvis Google Drive folder,
    sha256 71c78d40...81db1, matching the checksum in their own downloader.
  * the parameter vectors are length 604 while the connectome JSON lists 605
    edges. The missing one is index 50, `L2 -> R1`: a feedback edge onto a
    photoreceptor, which is an input unit in their model. Verified by finding
    the unique single-element deletion that makes the JSON's expected signs
    (alpha == -1) agree with the checkpoint's `edges_sign` element for element:
    228 negatives on both sides.

Their connectome is FIB-25/FIB-19, not FlyWire, so this transplants gains
BETWEEN datasets on the strength of shared cell-type names. That is a real
limitation and is reported as one.

    python paper/extract_flyvis_gains.py --zip <pretrained.zip> --json <conn.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DROP_EDGE_INDEX = 50          # L2 -> R1; see module docstring

# flyvis names some cells at a finer grain than FlyWire's consolidated typing.
# Collapsing them is a naming fix, not a modelling choice: the photoreceptors
# R1-R6 are one spectral class and FlyWire types them as a unit, and CT1 is a
# single cell whose two arborisations flyvis tracks separately. Where several
# flyvis pairs collapse onto one FlyWire pair the gains are averaged.
ALIAS = {
    **{f"R{i}": "R1-6" for i in range(1, 7)},
    "CT1(Lo1)": "CT1",
    "CT1(M10)": "CT1",
}


def load(conn_json: Path, ckpt_dir: Path) -> dict:
    conn = json.loads(conn_json.read_text())
    edges = [e for i, e in enumerate(conn["edges"]) if i != DROP_EDGE_INDEX]
    pairs = [(e["src"], e["tar"]) for e in edges]

    ckpts = sorted(ckpt_dir.glob("*/best_chkpt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {ckpt_dir}")
    S = np.stack([
        torch.load(c, map_location="cpu",
                   weights_only=False)["network"]["edges_syn_strength"].numpy()
        for c in ckpts
    ])
    if S.shape[1] != len(pairs):
        raise SystemExit(f"alignment failed: {S.shape[1]} params, {len(pairs)} pairs")

    # Verify the sign alignment rather than trusting the index.
    sign = torch.load(ckpts[0], map_location="cpu",
                      weights_only=False)["network"]["edges_sign"].numpy()
    expect = np.array([-1.0 if e.get("alpha") == -1 else 1.0 for e in edges])
    if not np.array_equal(sign, expect):
        raise SystemExit("sign alignment failed; edge order is not what we assume")

    return {"pairs": pairs, "strength": S, "n_models": len(ckpts)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn-json", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data" / "flyvis_gains.json")
    args = ap.parse_args()

    d = load(args.conn_json, args.ckpt_dir)
    S, pairs = d["strength"], d["pairs"]
    med = np.median(S, axis=0)
    spread = S.std(axis=0) / (np.abs(med) + 1e-12)

    # Normalise to mean 1 so the transplant changes the RELATIVE balance between
    # type pairs and not the global gain. Global gain is W_SYN, which stays
    # fitted on the taste circuit -- the whole point is to vary one thing.
    k = med / med.mean()

    # collapse aliased names, averaging gains that land on the same pair
    merged: dict[str, list[float]] = {}
    merged_sp: dict[str, list[float]] = {}
    for (src, tar), kk, sp in zip(pairs, k, spread):
        name = f"{ALIAS.get(src, src)}->{ALIAS.get(tar, tar)}"
        merged.setdefault(name, []).append(float(kk))
        merged_sp.setdefault(name, []).append(float(sp))

    rec = {
        "source": "Lappalainen et al. 2024 flyvis pretrained ensemble",
        "connectome": "fib25-fib19_v2.2 (NOT FlyWire)",
        "n_models": d["n_models"],
        "n_pairs": len(pairs),
        "note": "k is median syn_strength normalised to mean 1; spread is "
                "sd/|median| across the ensemble",
        "n_pairs_after_alias": len(merged),
        "alias": ALIAS,
        "gains": {name: {"k": float(np.mean(v)),
                         "spread": float(np.mean(merged_sp[name])),
                         "n_merged": len(v)}
                  for name, v in merged.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1))

    print(f"{d['n_models']} models, {len(pairs)} raw pairs -> "
          f"{len(merged)} after aliasing -> {args.out}")
    print(f"k: min {k.min():.3f}  median {np.median(k):.3f}  max {k.max():.3f}")
    print(f"ensemble disagreement sd/|median|: median {np.median(spread):.2f}, "
          f"90th pct {np.percentile(spread, 90):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
