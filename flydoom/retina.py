"""Ommatidial lattice and visual input.

The fly's retinotopy is not something we invent — FlyWire ships it. The file
`column_assignment.csv` gives, for 45,528 optic lobe neurons, the ommatidial
column they belong to plus hexagonal axial coordinates (p, q). That is 796
columns per eye, which is the real lattice.

    hex axial (p, q)  ->  cartesian  ->  (azimuth, elevation)

Two things the file does NOT give us:

  * R1-6 has no column assignment. We derive it: each photoreceptor's strongest
    columned target names its column. This is not a hack — in neural
    superposition optics the six photoreceptors of one ommatidium each project
    to a DIFFERENT lamina cartridge, so the target cartridge IS that cell's
    visual direction. Recovered 4,147 of 8,456 R1-6 (49%), median 5.3 per
    column, which is the expected ~6.
  * absolute angular scale. We set it from the published interommatidial angle
    (~5 deg for Drosophila) and report the resulting field of view so the
    number can be sanity-checked rather than trusted.

Injection sites, selectable:

    "L1"/"L2"  DEFAULT. Lamina monopolars, native column assignment, one cell
               per column per type, and crucially SYMMETRIC between the eyes:
               783/785 left and 789/796 right columns covered.
    "R1-6"     The true input layer, one synapse earlier, and the site the
               histamine override in graph.py exists for. But its derived
               retinotopy is BADLY ASYMMETRIC -- 451 of 785 left columns
               against 749 of 796 right, an artefact of FAFB's uneven optic
               lobe proofreading. M3 measures a LEFT-RIGHT DIFFERENCE, so a
               57%-vs-94% coverage asymmetry is a built-in bias in exactly the
               quantity under test. Use it only with that in mind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from . import config
from .cells import AnnotationTable
from .graph import ConnectomeGraph
from .registry import by_name

# ---- the angular scale, and why it is a calibration rather than a measurement
#
# MEASURED on this lattice: nearest-neighbour distance is exactly 1.000 axial
# units, so the hex conversion is right. But one eye spans 46.5 x 29.4 units --
# an aspect ratio of 1.58, while the real Drosophila eye covers roughly
# 170 x 150 deg, an aspect of 1.13.
#
# The lattice axes are therefore NOT isotropic in visual angle. Applying a
# single 5 deg/ommatidium gives a 233 x 145 deg eye, ~35% too wide. FlyWire's
# column_assignment.csv carries lattice topology, not an eye map, and there is
# no eye map in the download to do better with.
#
# So we scale each axis independently to land on the published field of view,
# and report the implied per-ommatidium angle. This is honest for anything
# that depends on the SIGN or the ORDERING of visual position -- M3 above all.
# It is a real limitation for anything depending on absolute angle: the
# looming crossover of spec 7 inherits this calibration directly, and should be
# reported with these two constants beside it.
EYE_FOV_AZIMUTH_DEG = 170.0
EYE_FOV_ELEVATION_DEG = 150.0
INTEROMMATIDIAL_DEG = 5.0   # nominal, for reporting only

SQRT3_2 = math.sqrt(3.0) / 2.0


def hex_to_cartesian(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axial hex coordinates to cartesian, in units of one ommatidium."""
    return p + q / 2.0, q * SQRT3_2


@dataclass
class Eye:
    """One eye's ommatidial lattice."""

    side: str
    column_ids: np.ndarray          # int32[C]
    p: np.ndarray                   # int32[C]
    q: np.ndarray                   # int32[C]
    azimuth_deg: np.ndarray         # float32[C]  negative = toward the midline
    elevation_deg: np.ndarray       # float32[C]

    # neuron -> column mapping for the chosen injection site
    neuron_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    neuron_column: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    """neuron_column[k] is a position into column_ids, not a column_id."""

    @property
    def n_columns(self) -> int:
        return len(self.column_ids)

    @property
    def fov_deg(self) -> tuple[float, float]:
        return (
            float(self.azimuth_deg.max() - self.azimuth_deg.min()),
            float(self.elevation_deg.max() - self.elevation_deg.min()),
        )

    def cartesian(self) -> tuple[np.ndarray, np.ndarray]:
        return hex_to_cartesian(self.p.astype(np.float64), self.q.astype(np.float64))


class Retina:
    """Maps a visual scene onto per-neuron drive, via the real lattice."""

    LAMINA_SITES = ("L1", "L2", "L3")

    def __init__(self, eyes: dict[str, Eye], site: str, n_neurons: int) -> None:
        self.eyes = eyes
        self.site = site
        self.n_neurons = n_neurons

    @property
    def inverts(self) -> bool:
        """Whether drive must be inverted relative to luminance.

        Photoreceptors DEPOLARISE to light. They release histamine, which
        HYPERPOLARISES L1/L2 -- the sign inversion at the first visual synapse.

        Injecting at R1-6 lets graph.py's histamine override perform that
        inversion for us, so drive rises with luminance. Injecting at L1/L2
        BYPASSES that synapse, so we must invert here instead: a lamina
        monopolar depolarises to light OFF. Getting this backwards makes the
        whole optic lobe read a negative image and reverses the optomotor
        response, which is exactly what M3 would catch.
        """
        return self.site in self.LAMINA_SITES

    # -- construction ----------------------------------------------------

    @classmethod
    def build(
        cls,
        graph: ConnectomeGraph,
        ann: AnnotationTable | None = None,
        raw_dir: Path | str = config.RAW_DIR,
        site: str = "L1",
    ) -> Retina:
        raw = Path(raw_dir)
        ann = ann or AnnotationTable.load(raw)

        path = raw / "column_assignment.csv.gz"
        if not path.exists():
            path = raw / "column_assignment.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"column_assignment.csv not found in {raw}; it carries the "
                "ommatidial lattice and there is no substitute for it"
            )
        ca = pl.read_csv(path, infer_schema_length=50_000).with_columns(
            pl.col("root_id").cast(pl.Int64)
        )

        eyes: dict[str, Eye] = {}
        for side in ("left", "right"):
            sub = (
                ca.filter(pl.col("hemisphere") == side)
                .group_by("column_id")
                .agg(pl.col("p").first(), pl.col("q").first())
                .sort("column_id")
            )
            p = sub["p"].to_numpy().astype(np.int32)
            q = sub["q"].to_numpy().astype(np.int32)
            cx, cy = hex_to_cartesian(p.astype(np.float64), q.astype(np.float64))
            # anisotropic scale, calibrated to the published eye FOV -- see the
            # note at the top of this module
            sx = EYE_FOV_AZIMUTH_DEG / max(cx.max() - cx.min(), 1e-9)
            sy = EYE_FOV_ELEVATION_DEG / max(cy.max() - cy.min(), 1e-9)
            # centre each eye on its own mean so azimuth is relative to gaze
            az = (cx - cx.mean()) * sx
            el = (cy - cy.mean()) * sy
            eyes[side] = Eye(
                side=side,
                column_ids=sub["column_id"].to_numpy().astype(np.int32),
                p=p, q=q,
                azimuth_deg=az.astype(np.float32),
                elevation_deg=el.astype(np.float32),
            )

        # ---- neuron -> column ----
        if site == "R1-6":
            mapping = cls._derive_photoreceptor_columns(graph, ann, ca)
        else:
            mapping = cls._native_columns(graph, ann, ca, site)

        for side, eye in eyes.items():
            pos = {int(c): i for i, c in enumerate(eye.column_ids)}
            idx, col = [], []
            for neuron, (hemi, column) in mapping.items():
                if hemi == side and column in pos:
                    idx.append(neuron)
                    col.append(pos[column])
            eye.neuron_idx = np.asarray(idx, dtype=np.int32)
            eye.neuron_column = np.asarray(col, dtype=np.int32)

        return cls(eyes, site, graph.n_neurons)

    @staticmethod
    def _native_columns(graph, ann, ca, site) -> dict[int, tuple[str, int]]:
        want = set(ann.resolve(by_name(site)).root_ids)
        sub = ca.filter(pl.col("root_id").is_in(list(want)))
        if not sub.height:
            raise ValueError(f"{site!r} has no rows in column_assignment.csv")
        out = {}
        for rid, hemi, colid in zip(sub["root_id"].to_list(),
                                    sub["hemisphere"].to_list(),
                                    sub["column_id"].to_list()):
            try:
                out[int(graph.index_of([rid])[0])] = (hemi, int(colid))
            except KeyError:
                continue
        return out

    @staticmethod
    def _derive_photoreceptor_columns(graph, ann, ca) -> dict[int, tuple[str, int]]:
        """Assign each R1-6 the column of its strongest columned target."""
        col = {
            int(r): (h, int(c))
            for r, h, c in zip(ca["root_id"].to_list(),
                               ca["hemisphere"].to_list(),
                               ca["column_id"].to_list())
        }
        photo = ann.resolve(by_name("R1-6")).root_ids
        photo_idx = set(graph.index_of(photo).tolist())

        mask = np.isin(graph.pre_idx, np.fromiter(photo_idx, dtype=np.int32))
        pre = graph.pre_idx[mask]
        post = graph.post_idx[mask]
        wgt = np.abs(graph.signed_syn[mask])

        best: dict[int, tuple[float, tuple[str, int]]] = {}
        roots = graph.root_ids
        for a, b, w in zip(pre, post, wgt):
            c = col.get(int(roots[b]))
            if c is None:
                continue
            a = int(a)
            if a not in best or w > best[a][0]:
                best[a] = (float(w), c)
        return {k: v[1] for k, v in best.items()}

    # -- stimuli ---------------------------------------------------------

    def grating(
        self,
        t: float,
        spatial_period_deg: float = 30.0,
        temporal_freq_hz: float = 2.0,
        direction: int = +1,
        contrast: float = 1.0,
        vertical: bool = False,
    ) -> dict[str, np.ndarray]:
        """A drifting square-wave grating, per eye, in [0, 1].

        direction  +1 drifts toward increasing azimuth (rightward in the fly's
                   frame), -1 leftward. This is the sign M3 tests.
        """
        out = {}
        for side, eye in self.eyes.items():
            coord = eye.elevation_deg if vertical else eye.azimuth_deg
            phase = (coord / spatial_period_deg
                     - direction * temporal_freq_hz * t)
            lum = 0.5 + 0.5 * contrast * np.sign(np.sin(2 * math.pi * phase))
            out[side] = lum.astype(np.float32)
        return out

    def uniform(self, level: float = 0.5) -> dict[str, np.ndarray]:
        return {s: np.full(e.n_columns, level, np.float32)
                for s, e in self.eyes.items()}

    def disc(
        self,
        radius_deg: float,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
        dark: bool = True,
    ) -> dict[str, np.ndarray]:
        """A disc of a given angular RADIUS -- the M4 looming stimulus."""
        out = {}
        for side, eye in self.eyes.items():
            d = np.hypot(eye.azimuth_deg - azimuth_deg,
                         eye.elevation_deg - elevation_deg)
            inside = d <= radius_deg
            lum = np.full(eye.n_columns, 1.0 if dark else 0.0, np.float32)
            lum[inside] = 0.0 if dark else 1.0
            out[side] = lum
        return out

    # -- drive -----------------------------------------------------------

    def drive(
        self,
        luminance: dict[str, np.ndarray],
        max_rate_hz: float = 150.0,
        baseline_hz: float = 0.0,
    ) -> np.ndarray:
        """Per-neuron Poisson rate vector from per-column luminance.

        Photoreceptors depolarise with light, so rate rises with luminance.
        The SIGN INVERSION to the ON/OFF pathways happens at the R->L synapse,
        which graph.py has already set to inhibitory (histamine). We do not
        impose it here.
        """
        rates = np.zeros(self.n_neurons, dtype=np.float32)
        for side, eye in self.eyes.items():
            if not eye.neuron_idx.size:
                continue
            lum = luminance[side][eye.neuron_column]
            if self.inverts:
                lum = 1.0 - lum
            rates[eye.neuron_idx] = baseline_hz + max_rate_hz * lum
        return rates

    # -- gpu-side stimulus ------------------------------------------------

    def to_torch(self, device: str = "cuda"):
        """Per-neuron visual coordinates, so stimuli can be built on the GPU.

        Returns (neuron_idx, azimuth, elevation, side_sign) as tensors. Doing
        the grating in numpy every 0.5 ms step and copying it across the bus
        dominates the run; this keeps it a single fused tensor op.
        """
        import torch

        idx, az, el, sgn = [], [], [], []
        for side, eye in self.eyes.items():
            if not eye.neuron_idx.size:
                continue
            idx.append(eye.neuron_idx)
            az.append(eye.azimuth_deg[eye.neuron_column])
            el.append(eye.elevation_deg[eye.neuron_column])
            sgn.append(np.full(eye.neuron_idx.size,
                               -1.0 if side == "left" else 1.0, np.float32))
        t = lambda a, d: torch.as_tensor(np.concatenate(a), dtype=d, device=device)
        return (t(idx, None).long() if False else
                torch.as_tensor(np.concatenate(idx), dtype=torch.long, device=device),
                t(az, torch.float32), t(el, torch.float32), t(sgn, torch.float32))

    def column_arrays(self):
        """(side, p, q, azimuth, elevation) per eye, for plotting."""
        return {s: (e.p, e.q, e.azimuth_deg, e.elevation_deg)
                for s, e in self.eyes.items()}

    # -- reporting -------------------------------------------------------

    def summary(self) -> str:
        lines = [f"injection site: {self.site}"
                 f"   (drive {'INVERTED' if self.inverts else 'follows'} luminance)"]
        for side, eye in self.eyes.items():
            az, el = eye.fov_deg
            covered = len(np.unique(eye.neuron_column)) if eye.neuron_idx.size else 0
            lines.append(
                f"  {side:<5} {eye.n_columns:>4} columns   "
                f"{eye.neuron_idx.size:>5} neurons over {covered:>4} columns   "
                f"FOV {az:.0f} x {el:.0f} deg"
            )
        lines.append(
            f"  FOV calibrated to {EYE_FOV_AZIMUTH_DEG:.0f} x "
            f"{EYE_FOV_ELEVATION_DEG:.0f} deg per eye (anisotropic scale; "
            f"absolute angles inherit this)")
        return "\n".join(lines)
