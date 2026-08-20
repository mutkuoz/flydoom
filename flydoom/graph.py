"""Connectome ETL — CSVs to three flat arrays.

The graph holds only MEASURED quantities: for each edge, a source index, a
target index, and a signed synapse count. The global gain W_SYN is applied by
the simulator at step time, not baked in here. That keeps this module free of
unverified constants and makes a W_SYN sweep cost nothing.

Representation is three flat arrays rather than a sparse matrix, per spec 3:

    pre_idx:  int32[E]
    post_idx: int32[E]
    signed_syn: float32[E]      sign * syn_count

and propagation is a single scatter-add:

    current.index_add_(0, post_idx, W_SYN * signed_syn * spiked[pre_idx])

`torch.sparse` is deliberately not used — index_add_ is faster for this access
pattern and avoids CSR rebuild costs.

Index space is defined by classification.csv (139,255 neurons) so that indices
are stable and every other module — cells.py above all — keys off the same map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import config, sources
from .cells import AnnotationTable
from .registry import by_name

VALID_NT = frozenset({"ACH", "GABA", "GLUT", "DA", "OCT", "SER", "UNK"})


@dataclass
class BuildReport:
    """Everything that happened during the build, for the M1 verdict."""

    source_file: str
    n_neurons: int
    n_edges_raw: int
    n_edges_thresholded: int
    n_edges_kept: int
    n_synapses: int
    n_edges_dropped_unmapped: int
    n_neurons_connected: int
    nt_counts: dict[str, int]
    n_unknown_nt: int
    n_photoreceptor_neurons: int
    n_photoreceptor_edges_flipped: int
    n_excitatory: int
    n_inhibitory: int
    build_seconds: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ConnectomeGraph:
    pre_idx: np.ndarray      # int32[E]
    post_idx: np.ndarray     # int32[E]
    signed_syn: np.ndarray   # float32[E]  = sign * syn_count
    root_ids: np.ndarray     # int64[N]    idx -> root_id
    report: BuildReport | None = None

    @property
    def n_neurons(self) -> int:
        return len(self.root_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre_idx)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.pre_idx, self.post_idx, self.signed_syn))

    # -- build -----------------------------------------------------------

    @classmethod
    def build(
        cls,
        raw_dir: Path | str = config.RAW_DIR,
        connections_file: str | None = None,
        threshold: int = config.SYN_THRESHOLD,
        apply_histamine_override: bool = True,
    ) -> ConnectomeGraph:
        import time

        t0 = time.perf_counter()
        raw = Path(raw_dir)

        ann = AnnotationTable.load(raw)

        # --- index space: classification.csv defines it, so cells.py and
        # --- graph.py can never disagree about what index 12345 means.
        root_ids = (
            ann.df["root_id"].drop_nulls().unique().sort().to_numpy().astype(np.int64)
        )
        id_map = pl.DataFrame(
            {"root_id": root_ids, "idx": np.arange(len(root_ids), dtype=np.int32)}
        )

        # --- edges
        name = connections_file or sources.CONNECTIONS
        path = _find(raw, name)
        if path is None:
            raise FileNotFoundError(f"{name} not found in {raw}")

        edges = pl.read_csv(path, infer_schema_length=50_000)
        n_raw = edges.height

        edges = edges.filter(pl.col("syn_count") >= threshold)
        n_thresholded = edges.height

        edges = edges.with_columns(
            pl.col("pre_root_id").cast(pl.Int64),
            pl.col("post_root_id").cast(pl.Int64),
            pl.col("nt_type").cast(pl.Utf8).fill_null("UNK"),
        )

        unknown = set(edges["nt_type"].unique().to_list()) - VALID_NT
        if unknown:
            raise ValueError(
                f"unexpected nt_type values in {name}: {sorted(unknown)}; "
                f"expected a subset of {sorted(VALID_NT)}"
            )
        nt_counts = {
            str(r[0]): int(r[1])
            for r in edges.group_by("nt_type").len().iter_rows()
        }

        # --- map root_ids to dense indices, dropping edges that touch a
        # --- neuron outside the classification universe
        edges = (
            edges.join(id_map.rename({"root_id": "pre_root_id", "idx": "pre_idx"}),
                       on="pre_root_id", how="inner")
                 .join(id_map.rename({"root_id": "post_root_id", "idx": "post_idx"}),
                       on="post_root_id", how="inner")
        )
        n_kept = edges.height

        # --- sign
        sign = pl.col("nt_type").replace_strict(config.SIGN, default=1, return_dtype=pl.Int8)
        edges = edges.with_columns(sign.alias("sign"))

        n_flipped = 0
        n_photo = 0
        if apply_histamine_override:
            photo = _photoreceptor_root_ids(ann)
            n_photo = len(photo)
            if photo:
                is_photo = pl.col("pre_root_id").is_in(list(photo))
                # Count only edges whose sign actually changes, so the report
                # states the real damage rather than the population size.
                n_flipped = int(
                    edges.filter(is_photo & (pl.col("sign") != config.HISTAMINE_SIGN))
                    .height
                )
                edges = edges.with_columns(
                    pl.when(is_photo)
                    .then(pl.lit(config.HISTAMINE_SIGN, dtype=pl.Int8))
                    .otherwise(pl.col("sign"))
                    .alias("sign")
                )

        edges = edges.with_columns(
            (pl.col("sign").cast(pl.Float32) * pl.col("syn_count").cast(pl.Float32))
            .alias("signed_syn")
        )

        pre = edges["pre_idx"].to_numpy().astype(np.int32)
        post = edges["post_idx"].to_numpy().astype(np.int32)
        signed = edges["signed_syn"].to_numpy().astype(np.float32)

        report = BuildReport(
            source_file=path.name,
            n_neurons=len(root_ids),
            n_edges_raw=n_raw,
            n_edges_thresholded=n_thresholded,
            n_edges_kept=n_kept,
            n_synapses=int(edges["syn_count"].sum()),
            n_edges_dropped_unmapped=n_thresholded - n_kept,
            n_neurons_connected=int(np.unique(np.concatenate([pre, post])).size),
            nt_counts=nt_counts,
            n_unknown_nt=nt_counts.get("UNK", 0),
            n_photoreceptor_neurons=n_photo,
            n_photoreceptor_edges_flipped=n_flipped,
            n_excitatory=int((signed > 0).sum()),
            n_inhibitory=int((signed < 0).sum()),
            build_seconds=time.perf_counter() - t0,
        )
        return cls(pre, post, signed, root_ids, report)

    # -- persistence -----------------------------------------------------

    def save(self, out_dir: Path | str = config.PROCESSED_DIR) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"root_id": self.root_ids,
             "idx": np.arange(self.n_neurons, dtype=np.int32)}
        ).write_parquet(out / "id_map.parquet")
        pl.DataFrame(
            {"pre_idx": self.pre_idx,
             "post_idx": self.post_idx,
             "signed_syn": self.signed_syn}
        ).write_parquet(out / "edges.parquet")
        if self.report is not None:
            (out / "build_report.json").write_text(
                json.dumps(self.report.to_dict(), indent=2)
            )
        return out

    @classmethod
    def load(cls, out_dir: Path | str = config.PROCESSED_DIR) -> ConnectomeGraph:
        out = Path(out_dir)
        ids = pl.read_parquet(out / "id_map.parquet")
        e = pl.read_parquet(out / "edges.parquet")
        return cls(
            e["pre_idx"].to_numpy().astype(np.int32),
            e["post_idx"].to_numpy().astype(np.int32),
            e["signed_syn"].to_numpy().astype(np.float32),
            ids["root_id"].to_numpy().astype(np.int64),
        )

    # -- gpu -------------------------------------------------------------

    def to_torch(self, device: str = "cuda"):
        """Move the three arrays to a torch device. Returns (pre, post, signed)."""
        import torch

        return (
            torch.from_numpy(self.pre_idx).to(device),
            torch.from_numpy(self.post_idx).to(device),
            torch.from_numpy(self.signed_syn).to(device),
        )

    # -- lookups ---------------------------------------------------------

    def index_of(self, root_ids) -> np.ndarray:
        """root_id -> dense index. Raises if any id is outside the universe."""
        want = np.asarray(list(root_ids), dtype=np.int64)
        pos = np.searchsorted(self.root_ids, want)
        pos = np.clip(pos, 0, len(self.root_ids) - 1)
        missing = self.root_ids[pos] != want
        if missing.any():
            raise KeyError(
                f"{int(missing.sum())} root_id(s) not in the index space, "
                f"first: {want[missing][0]}"
            )
        return pos.astype(np.int32)


def _find(raw: Path, name: str) -> Path | None:
    for c in (raw / name, raw / name.removesuffix(".gz")):
        if c.exists():
            return c
    return None


def _photoreceptor_root_ids(ann: AnnotationTable) -> set[int]:
    """Resolve R1-6, R7 and R8 through the tested resolver, not by hand."""
    out: set[int] = set()
    for handle in config.PHOTORECEPTOR_HANDLES:
        res = ann.resolve(by_name(handle))
        out.update(res.root_ids)
    return out
