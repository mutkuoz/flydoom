"""Smell: a modality Doom does not render.

Doom draws light and nothing else. A real fly in a real room with a large
animal in it genuinely smells that animal, so simulating only vision does not
make the model more honest — it makes it impoverished in a way the animal never
is. This module stands in for a sensor the engine lacks, the same way you would
compute wind or humidity from the geometry.

Two channels, both real labelled lines in the fly's nose:

    ORN_DA1   cVA, a fly pheromone. Means "another fly is here", and in the
              right context, a rival. Reaches the pC1 aggression population in
              three hops, touching 8 of its 10 cells -- measured on this
              connectome, and matching the documented cVA -> aggression path.
    ORN_DM1   vinegar. The strongest attractive odour a fly has. Food.

WHAT KEEPS THIS A NOSE AND NOT A TARGETING ORACLE
-------------------------------------------------
A fly's antennae are ~0.3 mm apart, which is far too close to triangulate on;
real flies locate sources by CASTING, zigzagging across a plume. So:

  * NO AZIMUTH. Left and right receive identical drive. This is the single
    most important line in the file. The moment the two sides differ, this
    stops being a nose and becomes a direction sensor that hands the fly the
    answer the visual system is supposed to work out.
  * SATURATING. Receptors have a ceiling; a crowd does not smell N times as
    loud as one.
  * SLOW. Odour has to physically arrive, and receptors are not instant.
  * PATCHY. Turbulent plumes are intermittent -- bursts, not a smooth ramp.
  * PERSISTENT. MEASURED: ViZDoom's label buffer contains only on-screen
    objects (421 sightings tested, none beyond 59 deg of a 130 deg viewport),
    so occlusion is already handled and nothing is smelled through a wall. But
    strict line-of-sight would be a LIGHT model, not an air model -- odour
    drifts around corners and lingers after its source is hidden. The plume
    time constant restores that without needing map geometry.

WHAT REMAINS A CHEAT, PLAINLY
-----------------------------
Distances come from the label buffer, so the game is telling us where things
are. The defence is not that this is free; it is that the sensor we build from
it is deliberately WEAK. It can say "something is around". It cannot say where,
so it cannot substitute for the computation under test.

An agent that reacts to enemies with this enabled is not evidence that the
connectome detects enemies. We told it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Doom actor names. Anything not listed is ignored rather than guessed at.
FOOD_NAMES = frozenset({
    "Medikit", "Stimpack", "HealthBonus", "ArmorBonus", "GreenArmor",
    "BlueArmor", "Backpack", "Berserk", "Soulsphere", "Megasphere",
})
NOT_A_SOURCE = frozenset({"DoomPlayer", "BulletPuff", "Blood", "Clip",
                          "Shell", "RocketAmmo", "Cell"})


@dataclass
class OlfactionConfig:
    r_half: float = 280.0
    """Map units at which a single source is half strength.

    Grounded in measurement rather than taste: across the M6 and M7 runs
    enemies sat between 48 and 764 map units away, nearest approach 54. At 280
    a close encounter is near saturation, mid-range is clearly detectable, and
    the far end is faint but present."""

    falloff_power: float = 2.0
    max_rate_hz: float = 100.0
    """Peak ORN drive. M2 calibrated this model at 100 Hz sensory stimulation,
    so this sits inside the validated range."""

    tau_transport: float = 0.20
    """Seconds. Onset lag — the odour has to physically get here."""

    tau_plume: float = 1.5
    """Seconds. How long the smell lingers after its source goes out of sight.
    This is what lets the fly still smell something that has stepped behind a
    pillar, which strict line-of-sight would wrongly extinguish."""

    intermittency_hz: float = 3.0
    duty: float = 0.6
    """Turbulent plumes arrive in bursts. Without this the channel is a
    suspiciously clean proximity readout."""

    food_gain: float = 1.0
    threat_gain: float = 1.0


class Olfaction:
    """Doom object positions to bilateral ORN drive."""

    def __init__(
        self,
        threat_idx: np.ndarray,
        food_idx: np.ndarray,
        n_neurons: int,
        dt: float,
        cfg: OlfactionConfig | None = None,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        import torch

        self.torch = torch
        self.cfg = cfg or OlfactionConfig()
        self.dt = dt
        self.threat = torch.as_tensor(np.asarray(threat_idx, dtype=np.int64),
                                      device=device)
        self.food = torch.as_tensor(np.asarray(food_idx, dtype=np.int64),
                                    device=device)
        self.rate = torch.zeros(n_neurons, dtype=torch.float32, device=device)
        self.rng = np.random.default_rng(seed)

        self.decay_transport = math.exp(-dt / self.cfg.tau_transport)
        self.decay_plume = math.exp(-dt / self.cfg.tau_plume)

        # raw concentration at the source (updated per tic), the lingering
        # plume, and the filtered value the receptors actually see
        self.raw = {"threat": 0.0, "food": 0.0}
        self.plume = {"threat": 0.0, "food": 0.0}
        self.sensed = {"threat": 0.0, "food": 0.0}
        self._burst = {"threat": 1.0, "food": 1.0}
        self._burst_steps = max(1, int(round(1.0 / self.cfg.intermittency_hz / dt)))
        self._since_burst = 0

    def reset(self) -> None:
        for d in (self.raw, self.plume, self.sensed):
            for k in d:
                d[k] = 0.0
        self.rate.zero_()

    # -- per Doom tic ----------------------------------------------------

    def _concentration(self, distances: list[float]) -> float:
        """Summed, saturating concentration from a set of sources."""
        c = self.cfg
        total = 0.0
        for r in distances:
            total += 1.0 / (1.0 + (max(r, 1.0) / c.r_half) ** c.falloff_power)
        return min(total, 1.0)

    def on_tic(self, objects: list[dict]) -> None:
        """Update from this tic's visible objects.

        `objects` is what DoomSession.threats() returns — name, distance,
        azimuth. The azimuth is deliberately IGNORED here; see the module
        docstring for why that is the point rather than an oversight.
        """
        threats, foods = [], []
        for o in objects:
            name = o.get("name", "")
            if name in NOT_A_SOURCE:
                continue
            (foods if name in FOOD_NAMES else threats).append(o["distance"])
        self.raw["threat"] = self._concentration(threats) * self.cfg.threat_gain
        self.raw["food"] = self._concentration(foods) * self.cfg.food_gain

    # -- per simulation substep ------------------------------------------

    def substep(self):
        """Advance the plume and return the per-neuron rate vector."""
        c = self.cfg

        self._since_burst += 1
        if self._since_burst >= self._burst_steps:
            self._since_burst = 0
            for k in self._burst:
                self._burst[k] = 1.0 if self.rng.random() < c.duty else 0.0

        for k in ("threat", "food"):
            # the plume decays slowly toward whatever the source currently
            # emits, so it lingers when the source drops out of sight
            target = self.raw[k]
            self.plume[k] = (self.decay_plume * self.plume[k]
                             + (1 - self.decay_plume) * target)
            arriving = max(self.plume[k], target) * self._burst[k]
            self.sensed[k] = (self.decay_transport * self.sensed[k]
                              + (1 - self.decay_transport) * arriving)

        self.rate.zero_()
        if self.threat.numel() and self.sensed["threat"] > 1e-4:
            self.rate[self.threat] = c.max_rate_hz * self.sensed["threat"]
        if self.food.numel() and self.sensed["food"] > 1e-4:
            self.rate[self.food] = c.max_rate_hz * self.sensed["food"]
        return self.rate

    @property
    def active(self) -> bool:
        return max(self.sensed.values()) > 1e-4

    def summary(self) -> str:
        return (
            f"smell: {self.threat.numel()} threat ORNs (ORN_DA1, cVA), "
            f"{self.food.numel()} food ORNs (ORN_DM1, vinegar)\n"
            f"  half-strength at {self.cfg.r_half:.0f} map units, "
            f"plume lingers {self.cfg.tau_plume:.1f} s, "
            f"bursts at {self.cfg.intermittency_hz:.0f} Hz\n"
            f"  BILATERALLY SYMMETRIC — carries no direction by construction"
        )
