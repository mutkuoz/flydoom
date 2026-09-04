"""Antennal mechanosensation: wall contact to bilateral afferent drive.

WHY THIS EXISTS
---------------
The model had vision, smell and taste, and nothing else. Walking into a wall
produced no afferent activity anywhere in the brain, which makes collisions --
the one behavioural metric that moved under every intervention tried -- a
consequence the agent cannot sense. Doom knows about contact the same way it
knows about pickups, so this stands in for a sensor the engine lacks, on the
same argument that justifies the olfactory channel.

WHY THE ANTENNAE AND NOT THE BRISTLES
-------------------------------------
The obvious target is wrong. FAFB carries 1,113 eye bristles and 304 head
bristles, one synapse from 109 descending neurons, which looks ideal until you
read which ones: DNg15, DNg35, DNg48, DNg84, DNg85. The DNg family is largely
grooming, and that is correct biology -- deflect a fly's head bristles and it
grooms its head rather than steering away. Driving wall contact into bristles
would ask the model to wash its face.

Flies wall-follow by ANTENNAL contact, and the antennal afferents are two
synapses from DNa02, the steering neuron this project decodes, against roughly
six for the visual pathway. That path also does not depend on the
multiplicative interaction that fails in this neuron model, which is why it is
worth testing separately from vision.

WHAT THE ENVIRONMENT SUPPLIES
-----------------------------
Position and heading, from which contact is derived rather than read: an agent
commanding forward motion that does not achieve it is against something. The
side comes from the slide. A body pushed into an angled surface slides along
it, and the component of actual displacement to the left of the commanded
heading means the obstruction lies to the right. Head-on contact produces no
slide and drives both antennae, which is what a fly meeting a wall squarely
would feel.

WHAT THIS IS NOT
----------------
Not a claim that Doom's collision geometry resembles an insect's mechanical
world, and not a tuned channel: the gain is one number, set so that a firm
contact reaches the same afferent rate the olfactory channel uses for a near
source, and never fitted against a behavioural outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MechanoConfig:
    """Antennal channel parameters. None is fitted on a behavioural result."""

    max_rate_hz: float = 120.0
    """Afferent rate at full deflection. Johnston's organ afferents fire
    briskly; this matches the ceiling the olfactory channel uses so the two
    sensory channels enter the network on comparable terms."""

    tau_adapt: float = 0.25
    """Mechanoreceptors are phasic. Sustained contact decays toward
    `sustained_frac` with this time constant, so a wall that is merely leaned
    against stops shouting while a fresh impact does not."""

    sustained_frac: float = 0.25
    """Floor the adapted response decays to, rather than zero: antennal
    afferents retain a tonic component under maintained deflection."""

    tau_release: float = 0.05
    """Decay once contact ends. Fast, because the antenna springs back."""

    stuck_cmd: float = 4.0
    """Commanded walk, in map units per tic, above which the agent counts as
    pushing. Matches the collision detector in the behavioural harness so the
    two agree about what a collision is."""

    stuck_move: float = 1.5
    """Achieved movement below which a pushing agent counts as obstructed."""

    slide_frac: float = 0.35
    """Lateral slide, as a fraction of the achieved step, above which contact
    is treated as one-sided. Below it the contact is head-on and bilateral."""


class Antennae:
    """Wall contact to bilateral antennal afferent drive."""

    def __init__(
        self,
        left_idx: np.ndarray,
        right_idx: np.ndarray,
        n_neurons: int,
        dt: float,
        cfg: MechanoConfig | None = None,
        device: str = "cuda",
    ) -> None:
        import torch

        self.torch = torch
        self.cfg = cfg or MechanoConfig()
        self.dt = dt
        self.left = torch.as_tensor(np.asarray(left_idx, dtype=np.int64),
                                    device=device)
        self.right = torch.as_tensor(np.asarray(right_idx, dtype=np.int64),
                                     device=device)
        self.rate = torch.zeros(n_neurons, dtype=torch.float32, device=device)

        self.decay_adapt = math.exp(-dt / self.cfg.tau_adapt)
        self.decay_release = math.exp(-dt / self.cfg.tau_release)

        self._prev = None          # (x, y) at the previous tic
        self.contact = {"left": 0.0, "right": 0.0}   # target deflection 0..1
        self.sensed = {"left": 0.0, "right": 0.0}    # after adaptation
        self.touching = False
        self.n_contacts = 0

    def reset(self) -> None:
        self._prev = None
        for d in (self.contact, self.sensed):
            for k in d:
                d[k] = 0.0
        self.touching = False
        self.n_contacts = 0
        self.rate.zero_()

    # -- per Doom tic ----------------------------------------------------

    def on_tic(self, x: float, y: float, angle_deg: float,
               commanded_fwd: float) -> None:
        """Derive contact and its side from motion against the command.

        `commanded_fwd` is the forward component the motor decoder asked for
        this tic, in the same map units the position is measured in.
        """
        c = self.cfg
        prev, self._prev = self._prev, (float(x), float(y))
        if prev is None:
            return

        dx, dy = float(x) - prev[0], float(y) - prev[1]
        step = math.hypot(dx, dy)
        pushing = abs(float(commanded_fwd)) >= c.stuck_cmd
        was_touching = self.touching
        self.touching = bool(pushing and step < c.stuck_move)

        if not self.touching:
            self.contact["left"] = self.contact["right"] = 0.0
            return
        if not was_touching:
            self.n_contacts += 1

        # How hard: the fraction of the commanded step that did not happen.
        want = max(abs(float(commanded_fwd)), 1e-6)
        deflection = min(1.0, max(0.0, (want - step) / want))

        # Which side: the component of actual displacement to the LEFT of the
        # heading. Sliding left means the obstruction is on the right.
        a = math.radians(float(angle_deg))
        hx, hy = math.cos(a), math.sin(a)
        slide = hx * dy - hy * dx            # z of heading x displacement
        lateral = abs(slide) / max(step, 1e-6) if step > 1e-6 else 0.0

        if lateral < c.slide_frac:
            self.contact["left"] = self.contact["right"] = deflection
        elif slide > 0.0:                     # sliding left -> wall on right
            self.contact["right"] = deflection
            self.contact["left"] = 0.0
        else:
            self.contact["left"] = deflection
            self.contact["right"] = 0.0

    # -- per simulation substep ------------------------------------------

    def substep(self):
        """Advance adaptation and return the per-neuron rate vector."""
        c = self.cfg
        self.rate.zero_()
        for side, idx in (("left", self.left), ("right", self.right)):
            target = self.contact[side]
            cur = self.sensed[side]
            if target > cur:
                # onset is immediate: the antenna is deflected the moment it
                # touches, and only then begins to adapt
                cur = target
            elif target > 0.0:
                floor = target * c.sustained_frac
                cur = floor + (cur - floor) * self.decay_adapt
            else:
                cur = cur * self.decay_release
            self.sensed[side] = cur
            if cur > 0.0 and len(idx):
                self.rate[idx] = cur * c.max_rate_hz
        return self.rate

    @property
    def active(self) -> bool:
        return max(self.sensed.values()) > 1e-6

    def summary(self) -> str:
        return (f"touch:    antennal, {len(self.left)}L/{len(self.right)}R "
                f"afferents, {self.n_contacts} contacts")
