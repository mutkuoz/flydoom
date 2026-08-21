"""The closed loop: Doom frame in, connectome in the middle, buttons out.

One Doom tic is 28.57 ms and one LIF step is 0.5 ms, so each frame is held
across 57 substeps. Holding rather than blanking matters -- see doom.py.

Everything the agent does per tic:

    frame ---> ommatidial luminance ---> contrast ---> graded lamina drive
                                                             |
                                      taste (health/damage) --+
                                                             v
                                            57 x LIF substep on the connectome
                                                             |
                                     descending neuron rates <+
                                                             |
                                      Schmitt / delta decode --> action
                                                             |
                                                          Doom tic

Nothing in this file decides anything. There is no policy, no value function
and no learning; the only thing between the retina and the buttons is the
connectome and two decoders whose parameters are fixed before the run starts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import config
from .cells import AnnotationTable
from .doom import DoomConfig, DoomSession, DoomVision
from .graph import ConnectomeGraph
from .interocept import Interoception, InteroceptConfig
from .lif import LIFNetwork, LIFParams
from .motor import MotorConfig, MotorDecoder
from .registry import by_name
from .retina import Retina


@dataclass
class AgentConfig:
    doom: DoomConfig = field(default_factory=DoomConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    intero: InteroceptConfig = field(default_factory=InteroceptConfig)
    site: tuple[str, ...] = ("L1", "L2", "L3")
    graded: bool = True
    slow_delay_s: float = config.T_DLY_SLOW
    bias_mv: float = 0.0
    """Tonic optic-lobe drive. Graded units need far less of this than spiking
    ones did -- M3's best results are at zero -- but it is left exposed because
    it is the first knob to try if the optic lobe goes silent."""
    device: str = "cuda"
    seed: int = 0


@dataclass
class TicRecord:
    tic: int
    rates: dict[str, float]
    action: dict[str, float]
    health: float
    damage: float
    sweet: float
    foul: float


class FlyDoomAgent:
    """A frozen connectome wired to a video game."""

    READOUTS = ("BPN", "MDN", "DNa02_L", "DNa02_R", "DNp01_L", "DNp01_R",
                "MN9", "LC4", "LPLC2", "LC11", "aggression")

    def __init__(self, cfg: AgentConfig | None = None) -> None:
        self.cfg = cfg or AgentConfig()
        c = self.cfg
        self.graph = ConnectomeGraph.load()
        self.ann = AnnotationTable.load(config.RAW_DIR)
        self.retina = Retina.build(self.graph, self.ann, site=c.site)

        edge_delay = self.graph.edge_delay_steps(
            self.ann, config.DT, t_slow=c.slow_delay_s
        )
        graded = self.graph.graded_mask(self.ann) if c.graded else None
        self.net = LIFNetwork.from_graph(
            self.graph, params=LIFParams(), device=c.device, seed=c.seed,
            edge_delay=edge_delay, graded=graded,
        )
        self.graded = graded is not None

        self.doom = DoomSession(c.doom)
        self.vision = DoomVision(self.retina, c.doom, config.DT, c.device)

        self.motor = MotorDecoder(self._motor_populations(), config.DT,
                                  c.motor, c.device)
        self.intero = Interoception(
            self._idx("sugar_GRN"), self._idx("bitter_GRN"),
            self.net.n, config.DT, c.intero, c.device,
        )

        self.gext = None
        if c.bias_mv:
            import polars as pl
            cls = pl.read_csv(config.RAW_DIR / "classification.csv.gz",
                              infer_schema_length=50_000)
            ids = cls.filter(pl.col("super_class").is_in(
                list(config.BIASED_SUPER_CLASSES)))["root_id"].to_list()
            self.gext = torch.zeros(self.net.n, dtype=torch.float32,
                                    device=c.device)
            self.gext[torch.as_tensor(self.graph.index_of(ids).astype(np.int64),
                                      device=c.device)] = c.bias_mv * 1e-3

        self.substeps = config.SUBSTEPS_PER_TIC
        self.history: list[TicRecord] = []
        self._probe = {k: torch.as_tensor(v.astype(np.int64), device=c.device)
                       for k, v in self._motor_populations().items() if len(v)}

    # -- population resolution -------------------------------------------

    def _idx(self, handle: str, side: str | None = None) -> np.ndarray:
        try:
            res = self.ann.resolve(by_name(handle), side=side)
        except KeyError:
            return np.zeros(0, np.int32)
        if not res.root_ids:
            return np.zeros(0, np.int32)
        return self.graph.index_of(res.root_ids)

    def _motor_populations(self) -> dict[str, np.ndarray]:
        pops = {
            "BPN": self._idx("BPN"),
            "MDN": self._idx("MDN"),
            "DNa02_L": self._idx("DNa02", side="left"),
            "DNa02_R": self._idx("DNa02", side="right"),
            "DNp01_L": self._idx("DNp01", side="left"),
            "DNp01_R": self._idx("DNp01", side="right"),
            "MN9": self._idx("MN9"),
            "LC4": self._idx("LC4"),
            "LPLC2": self._idx("LPLC2"),
            "LC11": self._idx("LC11"),
        }
        # The ATTACK gate. Spec 7 wants P1, but FAFB is a FEMALE brain and P1
        # is male-specific, so pC1 is the homologue that actually exists.
        agg = self._idx("pC1")
        if not len(agg):
            agg = self._idx("aIPg")
        pops["aggression"] = agg
        return pops

    # -- the loop --------------------------------------------------------

    def reset(self) -> None:
        self.doom.new_episode()
        self.net.reset()
        self.vision.reset()
        self.motor.reset()
        self.intero.reset()
        self.history.clear()

    def tic(self, tic_index: int) -> TicRecord | None:
        """Advance one Doom tic: 57 LIF substeps, then one action."""
        frame = self.doom.frame()
        if frame is None:
            return None

        out_set, column_lum = self.vision.drive(
            frame, self.net.n, self.net.p.graded_max_rate, config.DT
        )
        self.last_luminance = column_lum

        # ---- 57 substeps, frame HELD ----
        for _ in range(self.substeps):
            taste = self.intero.substep()
            forced = None
            if self.intero.active:
                forced = (torch.rand(self.net.n, generator=self.net.gen,
                                     device=self.net.device)
                          < (taste * config.DT).clamp(0, 1))
            self.net.step(g_ext=self.gext,
                          out_set=out_set if self.graded else None,
                          forced=forced)
            self.motor.observe(self.net)

        rates = self.motor.sample()
        action = self.motor.decode()
        result = self.doom.step(
            [action.get(b, 0.0) for b in DoomSession.BUTTONS],
            self.cfg.doom.frame_skip,
        )
        self.intero.on_tic(result["healed"], result["damage_taken"])

        rec = TicRecord(
            tic=tic_index, rates=rates, action=action,
            health=result["health"], damage=result["damage_taken"],
            sweet=self.intero.sweet, foul=self.intero.foul,
        )
        self.history.append(rec)
        return rec

    def run(self, tics: int, on_tic=None) -> list[TicRecord]:
        self.reset()
        for i in range(tics):
            if self.doom.finished:
                break
            rec = self.tic(i)
            if rec is None:
                break
            if on_tic is not None and on_tic(self, rec) is False:
                break
        return self.history

    def close(self) -> None:
        self.doom.close()

    # -- reporting -------------------------------------------------------

    def summary(self) -> str:
        parts = [
            self.retina.summary(),
            self.vision.summary(),
            self.motor.summary(),
            self.intero.summary(),
            f"synapses: {self.net.conductance_summary}",
            f"delays:   {self.net.delay_summary}",
            f"graded:   {int(self.net.graded.sum()):,} non-spiking neurons"
            if self.graded else "graded:   off (all cells spike)",
            f"timing:   {self.substeps} LIF substeps per Doom tic "
            f"({config.DT * 1e3:.1f} ms each, frame held throughout)",
        ]
        return "\n".join(parts)
