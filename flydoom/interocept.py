"""Health and damage as taste.

Spec 6.3 is emphatic and worth restating, because the shape of this file looks
like reinforcement learning and is not. There is no reward, no return, no
credit assignment, and nothing learns. Picking up health puts something sweet
in the fly's mouth; taking damage puts something foul in it. What the brain
does next is whatever the connectome already does with sugar and bitter -- the
same pathway M2 validated.

Two properties matter and neither is decorative:

* the injection PERSISTS after the event. A real taste lingers; a one-tic
  impulse would be gone before the SEZ finished responding, since the sugar to
  MN9 path takes tens of milliseconds.
* damage and healing are graded by magnitude, because M2 established that the
  dose-response is graded rather than all-or-none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class InteroceptConfig:
    sugar_rate_hz: float = 120.0
    """Peak GRN drive for a full-strength pickup. M2 calibrated the model at
    100 Hz sugar stimulation, so this sits in the validated range."""

    bitter_rate_hz: float = 120.0

    tau_taste: float = 0.35
    """Seconds. How long a taste lingers after the event."""

    health_scale: float = 25.0
    """Health points that count as a full-strength stimulus."""

    damage_scale: float = 20.0


class Interoception:
    """Turns Doom's health changes into gustatory drive."""

    def __init__(
        self,
        sugar_idx: np.ndarray,
        bitter_idx: np.ndarray,
        n_neurons: int,
        dt: float,
        cfg: InteroceptConfig | None = None,
        device: str = "cuda",
    ) -> None:
        import torch

        self.torch = torch
        self.cfg = cfg or InteroceptConfig()
        self.dt = dt
        self.decay = math.exp(-dt / self.cfg.tau_taste)
        self.sugar = torch.as_tensor(np.asarray(sugar_idx, dtype=np.int64),
                                     device=device)
        self.bitter = torch.as_tensor(np.asarray(bitter_idx, dtype=np.int64),
                                      device=device)
        self.rate = torch.zeros(n_neurons, dtype=torch.float32, device=device)
        self.sweet = 0.0
        self.foul = 0.0

    def reset(self) -> None:
        self.rate.zero_()
        self.sweet = 0.0
        self.foul = 0.0

    def on_tic(self, healed: float, damage: float) -> None:
        """Register this tic's events. Call once per Doom tic."""
        c = self.cfg
        if healed > 0:
            self.sweet = min(1.0, self.sweet + healed / c.health_scale)
        if damage > 0:
            self.foul = min(1.0, self.foul + damage / c.damage_scale)

    def substep(self):
        """Decay the lingering taste and return the per-neuron rate vector."""
        self.sweet *= self.decay
        self.foul *= self.decay
        self.rate.zero_()
        if self.sweet > 1e-4 and self.sugar.numel():
            self.rate[self.sugar] = self.cfg.sugar_rate_hz * self.sweet
        if self.foul > 1e-4 and self.bitter.numel():
            self.rate[self.bitter] = self.cfg.bitter_rate_hz * self.foul
        return self.rate

    @property
    def active(self) -> bool:
        return self.sweet > 1e-4 or self.foul > 1e-4

    def summary(self) -> str:
        return (f"taste: {self.sugar.numel()} sugar GRNs, "
                f"{self.bitter.numel()} bitter GRNs, "
                f"tau {self.cfg.tau_taste * 1e3:.0f} ms")
