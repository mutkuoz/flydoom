"""Resolve named cell populations to FlyWire root IDs.

The contract, from spec 4 and 10: **never guess**. Every lookup returns a
Resolution recording not just what was found but *how* it was found — exact
type match, prefix match, or fuzzy substring — plus the distinct type strings
that matched, so a human can eyeball whether the match is honest.

An unresolved handle is reported, never silently empty.

The annotation universe is assembled from four files, because no single one has
what we need — see sources.py for the schema archaeology. In particular
`classification.csv` has **no cell_type column**; types live in
`consolidated_cell_types.csv` and `visual_neuron_types.csv`, and a few
populations (BPN, aIPg) exist only as free-text community labels.

Nothing here touches torch or the connection graph. This module answers one
question: does population X exist in this snapshot, and which neurons is it?
Mapping root_id -> dense index is graph.py's job (M1).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import polars as pl

from . import sources
from .registry import ALL_HANDLES, Handle

# Type columns we search, in priority order. Assembled by the loader from
# several files; not all are guaranteed to exist, so we intersect with reality.
TYPE_COLUMNS = (
    "primary_type",      # consolidated_cell_types.csv — the main source
    "visual_type",       # visual_neuron_types.csv — optic lobe
    "additional_types",  # consolidated_cell_types.csv — secondary typing
    "sub_class",         # classification.csv — e.g. "sugar/water", "bitter"
    "class",             # classification.csv — e.g. "gustatory", "olfactory"
    "hemilineage",
    "label",             # labels.csv — free-text, fuzzy only
)

FUZZY_ONLY_COLUMNS = frozenset({"label"})
"""Free-text columns. Exact and prefix matching against a sentence like
"Type 1 BPN (Bolt Protocerebral Neuron)" is meaningless, so these participate
in substring search only — and any hit is reported as a WEAK match."""

# Two separate guards, because prefix and substring carry different risk.
#
# Prefix matching is anchored, so a 2-char pattern like "KC" is safe and in fact
# necessary -- it is the only way to gather KCg-m + KCab + KCg-d as one
# population. Only a 1-char pattern is dangerous.
MIN_PREFIX_LEN = 2
#
# Substring matching is unanchored and much greedier, so it stays conservative.
MIN_FUZZY_LEN = 3

DOWNLOAD_HINT = f"""\
Connectome files not found.

FlyWire data requires a manual download (Google sign-in, and the terms must be
accepted in person -- it cannot be scripted):

  1. Open https://codex.flywire.ai/?dataset=fafb
  2. Sign in, then go to Info -> Download Data
  3. Select snapshot v783 -- NOT the live materialization, which drifts
  4. Download at least:
       {sources.CLASSIFICATION}
       {sources.CELL_TYPES}
       {sources.VISUAL_TYPES}
       {sources.LABELS}
       {sources.CONNECTIONS}
  5. Place them in data/raw/  (.gz is fine, they are read compressed)

See scripts/fetch_data.sh, and data/DATA_LICENSE.md for the CC BY-NC 4.0 terms
that govern this data.
"""


class MatchKind(Enum):
    EXACT = "exact"
    PREFIX = "prefix"
    FUZZY = "fuzzy"
    NONE = "none"


class Status(Enum):
    OK = "ok"
    COUNT_OFF = "count_off"
    WEAK_MATCH = "weak_match"
    UNILATERAL = "unilateral"
    MISSING = "missing"


@dataclass
class Resolution:
    handle: Handle
    status: Status
    match_kind: MatchKind = MatchKind.NONE
    root_ids: list[int] = field(default_factory=list)
    matched_types: list[str] = field(default_factory=list)
    matched_column: str | None = None
    matched_pattern: str | None = None
    side_counts: dict[str, int] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.root_ids)

    @property
    def is_fatal(self) -> bool:
        return self.handle.required and self.status is Status.MISSING


def _read(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=50_000)


class AnnotationTable:
    """The neuron annotation universe, loaded once and queried many times."""

    def __init__(self, frame: pl.DataFrame, sources_used: list[str]) -> None:
        self.df = frame
        self.sources = sources_used
        self.type_columns = [c for c in TYPE_COLUMNS if c in frame.columns]
        if not self.type_columns:
            raise ValueError(
                f"no usable type columns in {sources_used}; "
                f"expected one of {TYPE_COLUMNS}, found {frame.columns}"
            )
        self.has_side = "side" in frame.columns
        self._vocab: list[str] | None = None

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, raw_dir: Path | str = "data/raw") -> AnnotationTable:
        raw = Path(raw_dir)

        def find(name: str) -> Path | None:
            for candidate in (raw / name, raw / name.removesuffix(".gz")):
                if candidate.exists():
                    return candidate
            return None

        base = find(sources.CLASSIFICATION)
        if base is None:
            raise FileNotFoundError(DOWNLOAD_HINT)

        df = _read(base).with_columns(pl.col("root_id").cast(pl.Int64, strict=False))
        used = [base.name]

        # consolidated_cell_types.csv — the primary source of cell types, since
        # classification.csv has no cell_type column at all.
        if (p := find(sources.CELL_TYPES)) is not None:
            ct = _read(p)
            cols = [pl.col("root_id").cast(pl.Int64, strict=False)]
            if "primary_type" in ct.columns:
                cols.append(pl.col("primary_type").cast(pl.Utf8))
            extra = next(
                (c for c in ct.columns if c.startswith("additional_type")), None
            )
            if extra:
                cols.append(pl.col(extra).cast(pl.Utf8).alias("additional_types"))
            df = df.join(
                ct.select(cols).unique(subset=["root_id"], keep="first"),
                on="root_id", how="left",
            )
            used.append(p.name)

        # visual_neuron_types.csv — optic lobe typing for ~95k neurons.
        if (p := find(sources.VISUAL_TYPES)) is not None:
            vt = _read(p)
            col = next(
                (c for c in ("type", "cell_type", "visual_type") if c in vt.columns),
                None,
            )
            if col and "root_id" in vt.columns:
                df = df.join(
                    vt.select(
                        pl.col("root_id").cast(pl.Int64, strict=False),
                        pl.col(col).cast(pl.Utf8).alias("visual_type"),
                    ).unique(subset=["root_id"], keep="first"),
                    on="root_id", how="left",
                )
                used.append(p.name)

        # labels.csv — free-text community annotations, many per neuron.
        # The ONLY place BPN and aIPg appear in this snapshot.
        if (p := find(sources.LABELS)) is not None:
            lb = _read(p)
            if {"root_id", "label"} <= set(lb.columns):
                df = df.join(
                    lb.select(
                        pl.col("root_id").cast(pl.Int64, strict=False),
                        pl.col("label").cast(pl.Utf8),
                    )
                    .group_by("root_id")
                    .agg(pl.col("label").str.join(" | ").alias("label")),
                    on="root_id", how="left",
                )
                used.append(p.name)

        return cls(df, used)

    # -- vocabulary ------------------------------------------------------

    @property
    def vocab(self) -> list[str]:
        """Distinct type strings, for near-miss suggestions.

        Excludes free-text columns — suggesting whole sentences is noise.
        """
        if self._vocab is None:
            seen: set[str] = set()
            for col in self.type_columns:
                if col in FUZZY_ONLY_COLUMNS:
                    continue
                seen.update(
                    str(v)
                    for v in self.df[col].drop_nulls().unique().to_list()
                    if v not in (None, "")
                )
            self._vocab = sorted(seen)
        return self._vocab

    # -- matching --------------------------------------------------------

    def _select(self, col: str, expr: pl.Expr) -> pl.DataFrame:
        return self.df.filter(expr & pl.col(col).is_not_null())

    def _cols(self, *, structured_only: bool) -> list[str]:
        if structured_only:
            return [c for c in self.type_columns if c not in FUZZY_ONLY_COLUMNS]
        return self.type_columns

    # Within one column, every pattern is UNIONED rather than tried in turn.
    # Handles that list sibling subtypes -- pC1a..pC1e, aIPg1/3/4, T4a..T4d --
    # need all of them, and returning only the first match silently drops the
    # rest. pC1 came back as 2 neurons instead of 10 before this was fixed.
    # Columns are still tried in priority order, so alternative spellings that
    # live in different columns (MN9 vs proboscis_motor_neuron) do not merge.

    @staticmethod
    def _any_of(exprs: list[pl.Expr]) -> pl.Expr:
        out = exprs[0]
        for e in exprs[1:]:
            out = out | e
        return out

    def _try_exact(self, patterns: tuple[str, ...]):
        wanted = [p.lower() for p in patterns]
        for col in self._cols(structured_only=True):
            low = pl.col(col).cast(pl.Utf8).str.to_lowercase()
            hit = self._select(col, low.is_in(wanted))
            if hit.height:
                got = set(hit[col].cast(pl.Utf8).str.to_lowercase().to_list())
                used = [p for p in patterns if p.lower() in got]
                return hit, col, " + ".join(used)
        return None, None, None

    def _try_prefix(self, patterns: tuple[str, ...]):
        elig = [p for p in patterns if len(p) >= MIN_PREFIX_LEN]
        if not elig:
            return None, None, None
        for col in self._cols(structured_only=True):
            low = pl.col(col).cast(pl.Utf8).str.to_lowercase()
            hit = self._select(
                col, self._any_of([low.str.starts_with(p.lower()) for p in elig])
            )
            if hit.height:
                return hit, col, " + ".join(elig)
        return None, None, None

    def _try_fuzzy(self, patterns: tuple[str, ...], aliases: tuple[str, ...]):
        """Patterns are authoritative; aliases are a last resort.

        Tried as two separate tiers so a speculative alias never widens an
        otherwise clean pattern match.
        """
        for tier in (patterns, aliases):
            elig = [t for t in tier if len(t) >= MIN_FUZZY_LEN]
            if not elig:
                continue
            for col in self._cols(structured_only=False):
                low = pl.col(col).cast(pl.Utf8).str.to_lowercase()
                hit = self._select(
                    col,
                    self._any_of(
                        [low.str.contains(t.lower(), literal=True) for t in elig]
                    ),
                )
                if hit.height:
                    return hit, col, " + ".join(elig)
        return None, None, None

    # -- the public API --------------------------------------------------

    def resolve(self, handle: Handle, side: str | None = None) -> Resolution:
        hit, col, pat = self._try_exact(handle.patterns)
        kind = MatchKind.EXACT

        # exact_only guards handles whose own name is a proper prefix of a
        # sibling's -- ORN_D would otherwise gather ORN_DA1..ORN_DL3, and "V"
        # would gather most of the antennal lobe. Better to report MISSING and
        # make a human look than to return a confidently wrong population.
        if hit is None and not handle.exact_only:
            hit, col, pat = self._try_prefix(handle.patterns)
            kind = MatchKind.PREFIX

        if hit is None and not handle.exact_only:
            hit, col, pat = self._try_fuzzy(handle.patterns, handle.aliases)
            kind = MatchKind.FUZZY

        if hit is None:
            return Resolution(
                handle=handle,
                status=Status.MISSING,
                candidates=self._near_misses(handle),
            )

        if side is not None and self.has_side:
            hit = hit.filter(
                pl.col("side").cast(pl.Utf8).str.to_lowercase() == side.lower()
            )

        root_ids = hit["root_id"].drop_nulls().unique().to_list()

        if col in FUZZY_ONLY_COLUMNS:
            matched_types = [f"<free-text label containing {pat!r}>"]
        else:
            matched_types = sorted(
                {str(v) for v in hit[col].drop_nulls().unique().to_list()}
            )

        side_counts: dict[str, int] = {}
        if self.has_side:
            side_counts = {
                str(r[0]): int(r[1])
                for r in hit.group_by("side").len().iter_rows()
                if r[0] is not None
            }

        return Resolution(
            handle=handle,
            status=self._grade(handle, len(root_ids), kind, side_counts, col),
            match_kind=kind,
            root_ids=root_ids,
            matched_types=matched_types,
            matched_column=col,
            matched_pattern=pat,
            side_counts=side_counts,
        )

    def resolve_all(self, handles: list[Handle] | None = None) -> list[Resolution]:
        return [self.resolve(h) for h in (handles or ALL_HANDLES)]

    # -- grading ---------------------------------------------------------

    @staticmethod
    def _grade(
        handle: Handle,
        count: int,
        kind: MatchKind,
        side_counts: dict[str, int],
        column: str,
    ) -> Status:
        if count == 0:
            return Status.MISSING

        lo, hi = handle.expect_min, handle.expect_max
        if (lo is not None and count < lo) or (hi is not None and count > hi):
            return Status.COUNT_OFF

        if handle.bilateral and side_counts:
            lateral = {
                s: n for s, n in side_counts.items()
                if s.lower() in ("left", "right", "l", "r")
            }
            if len(lateral) == 1:
                return Status.UNILATERAL

        # A free-text hit is never trustworthy on its own, regardless of kind.
        if kind is not MatchKind.EXACT or column in FUZZY_ONLY_COLUMNS:
            return Status.WEAK_MATCH

        return Status.OK

    def _near_misses(self, handle: Handle, n: int = 6) -> list[str]:
        out: list[str] = []
        for pat in handle.patterns + handle.aliases:
            out.extend(difflib.get_close_matches(pat, self.vocab, n=n, cutoff=0.6))
            if len(pat) >= MIN_FUZZY_LEN:
                low = pat.lower()
                out.extend(v for v in self.vocab if low in v.lower())
        seen: set[str] = set()
        return [v for v in out if not (v in seen or seen.add(v))][:n]
