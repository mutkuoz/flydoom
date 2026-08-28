#!/usr/bin/env python3
"""Recover the per-edge neurotransmitter label for the processed graph.

WHY THIS EXISTS. `edges.parquet` stores only `signed_syn`, so the transmitter
survives as a SIGN and nothing more: GABA and glutamate are both negative and
become indistinguishable once loaded. That was harmless while every synapse
carried the same conductance. It stops being harmless the moment conductance
differs BY receptor -- and published physiology says it does: GABA-A (Rdl) is
~2.5x cholinergic, measured in two unrelated preparations (Su & O'Dowd 2003
quantal events in the same cells, 2.5; Groschner et al. 2022 adult T4 in vivo,
2.6). See config.G_SYN_RATIO.

This writes an ADDITIVE file, `nt_type.parquet`, aligned row-for-row with
`edges.parquet`. Nothing existing is modified, so runs already in flight
against data/processed are unaffected.

Alignment is the whole risk here, so it is checked rather than assumed: the
join is replicated exactly as ConnectomeGraph.build does it, and the resulting
signs are verified against the signs already stored in edges.parquet. If a
single row disagrees the script refuses to write.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl

from flydoom import config, sources
from flydoom.cells import AnnotationTable
from flydoom.graph import VALID_NT, _find, _photoreceptor_root_ids


def main() -> int:
    raw = Path(config.RAW_DIR)
    out = Path(config.PROCESSED_DIR)
    ids = pl.read_parquet(out / "id_map.parquet")
    ref = pl.read_parquet(out / "edges.parquet")
    ann = AnnotationTable.load(raw)

    path = _find(raw, sources.CONNECTIONS)
    if path is None:
        raise FileNotFoundError(sources.CONNECTIONS)
    e = pl.read_csv(path, infer_schema_length=50_000)
    e = e.filter(pl.col("syn_count") >= config.SYN_THRESHOLD).with_columns(
        pl.col("pre_root_id").cast(pl.Int64),
        pl.col("post_root_id").cast(pl.Int64),
        pl.col("nt_type").cast(pl.Utf8).fill_null("UNK"),
    )
    id_map = ids.rename({"root_id": "root_id", "idx": "idx"})
    e = (e.join(id_map.rename({"root_id": "pre_root_id", "idx": "pre_idx"}),
                on="pre_root_id", how="inner")
           .join(id_map.rename({"root_id": "post_root_id", "idx": "post_idx"}),
                 on="post_root_id", how="inner"))

    sign = pl.col("nt_type").replace_strict(config.SIGN, default=1,
                                            return_dtype=pl.Int8)
    e = e.with_columns(sign.alias("sign"))

    # histamine override: photoreceptors are inhibitory regardless of label
    photo = set(_photoreceptor_root_ids(ann))
    if photo:
        e = e.with_columns(
            pl.when(pl.col("pre_root_id").is_in(list(photo)))
              .then(pl.lit(config.HISTAMINE_SIGN, dtype=pl.Int8))
              .otherwise(pl.col("sign")).alias("sign"))

    e = e.with_columns((pl.col("sign") * pl.col("syn_count")).alias("signed_syn"))

    if e.height != ref.height:
        print(f"REFUSING TO WRITE: {e.height:,} rows rebuilt vs "
              f"{ref.height:,} in edges.parquet")
        return 1
    a = e["signed_syn"].to_numpy().astype(np.float32)
    b = ref["signed_syn"].to_numpy().astype(np.float32)
    bad = int((a != b).sum())
    if bad:
        print(f"REFUSING TO WRITE: {bad:,} of {len(a):,} signed_syn values "
              f"disagree -- the row order does not match edges.parquet")
        return 1

    e.select("nt_type").write_parquet(out / "nt_type.parquet")
    counts = e.group_by("nt_type").len().sort("len", descending=True)
    print(f"verified row-aligned with edges.parquet ({len(a):,} edges, "
          f"0 signed_syn disagreements)")
    print(counts)
    print(f"\nwrote {out / 'nt_type.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
