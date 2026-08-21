"""Descending neuron activity to ViZDoom actions.

Spec 6.2 asks for a Schmitt trigger per button, and that is right for the
binary ones. But steering is not binary: `rate(DNa02_L) - rate(DNa02_R)` is a
graded quantity, and the fly's real yaw command is proportional to it. Squashing
it into TURN_LEFT / TURN_RIGHT throws away the signal and reintroduces exactly
the chatter hysteresis exists to prevent.

So we use ViZDoom's DELTA buttons for the continuous axes and Schmitt-triggered
binary buttons for the discrete ones:

    TURN_LEFT_RIGHT_DELTA        <- DNa02 L-R differential, proportional
    MOVE_FORWARD_BACKWARD_DELTA  <- BPN forward, MDN backward
    MOVE_LEFT_RIGHT_DELTA        <- DNp01 giant fiber, lateral escape
    ATTACK / USE                 <- Schmitt triggered

Rate estimation is an exponential leaky filter over each population's output.
Note it reads `net.out`, not `net.spiked`: for a spiking cell those are the
same, but graded cells never spike, and a decoder built on spike counts would
read exactly zero from them. Descending neurons do spike, so this only matters
if the graded boundary ever moves -- but it costs nothing to be correct now.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class SchmittTrigger:
    """Two-threshold latch. Prevents chatter when a rate hovers at threshold.

    A single threshold on a noisy rate toggles every tic; that is the button
    rattle spec 6.2 warns about.
    """

    on_at: float
    off_at: float
    state: bool = False

    def __post_init__(self) -> None:
        if self.off_at > self.on_at:
            raise ValueError(
                f"off_at ({self.off_at}) must be <= on_at ({self.on_at}); "
                "an inverted Schmitt trigger oscillates by construction"
            )

    def update(self, value: float) -> bool:
        if self.state:
            if value < self.off_at:
                self.state = False
        elif value > self.on_at:
            self.state = True
        return self.state

    def reset(self) -> None:
        self.state = False


@dataclass
class MotorConfig:
    tau_rate: float = 0.08
    """Seconds. Spec 6.2 says 50-100 ms; 80 ms spans ~3 Doom tics."""

    yaw_gain: float = 0.35
    """Degrees of turn per tic per Hz of DNa02 differential. A free scalar --
    fly saccades reach hundreds of deg/s and Doom's turn rate does not, so this
    must be tuned or the agent oscillates. Spec 6.2 flags it explicitly."""

    yaw_max_deg: float = 12.0
    """Clamp per tic. Doom turns feel unusable past roughly this."""

    forward_gain: float = 0.9
    forward_max: float = 22.0
    """Doom map units per tic. ~22 is a normal run speed."""

    lateral_gain: float = 2.5
    lateral_max: float = 20.0

    attack_on: float = 25.0
    attack_off: float = 15.0
    use_on: float = 20.0
    use_off: float = 12.0

    tau_baseline: float = 3.0
    """Seconds. Time constant for adapting out a CONSTANT steering offset.

    MEASURED, and this is why it exists. DNa02_R sits about 2x above DNa02_L
    permanently (255 vs 130 Hz), because the two cells' input wiring is not
    symmetric -- FAFB is one real brain and its two halves are not mirror
    images. A fixed differential means a fixed turn command, and the agent
    spins forever.

    A real fly does not have this problem because the optomotor reflex closes
    the loop: it turns, sees the world slip the other way, and corrects. Our
    optomotor response is ~50x too weak (M3), so nothing corrects it. The spin
    is M3's failure made visible in behaviour.

    Adapting out the slow component is the standard motor-control answer and
    the direct counterpart of the retina's luminance adaptation: keep the
    transients, discard the DC. Set to 0 to disable and watch it spin."""

    warmup_tics: int = 12
    """Tics to stay still before steering at all.

    The rate filter needs a few tau_rate to converge from zero. Seeding the
    yaw baseline before it has means seeding from a stale value, and as the
    filter catches up the differential drifts away from that baseline and
    produces a hard turn -- measured at the yaw clamp (12 deg/tic) for about a
    second at episode start.

    The window must be several tau_rate, not one: at 2.1 tau the filter is only
    88% converged and the residual 12% still drifted into a 3.6 deg turn. 12
    tics is 343 ms, about 4.3 tau, which leaves under 2%."""

    deadzone_hz: float = 1.5
    """Ignore differentials smaller than this. Two DNa02 cells (one per side)
    firing stochastically produce a nonzero difference from noise alone."""


class MotorDecoder:
    """Turns descending-neuron output into a ViZDoom action vector."""

    def __init__(
        self,
        populations: dict[str, np.ndarray],
        dt: float,
        cfg: MotorConfig | None = None,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg or MotorConfig()
        self.dt = dt
        self.device = device
        self.pop = {
            k: torch.as_tensor(np.asarray(v, dtype=np.int64), device=device)
            for k, v in populations.items()
            if len(v)
        }
        self.missing = sorted(set(populations) - set(self.pop))
        self.decay = float(np.exp(-dt / self.cfg.tau_rate))
        self.rates: dict[str, float] = {k: 0.0 for k in self.pop}
        self.attack = SchmittTrigger(self.cfg.attack_on, self.cfg.attack_off)
        self.use = SchmittTrigger(self.cfg.use_on, self.cfg.use_off)
        self._filt: torch.Tensor | None = None
        # slow baseline of the yaw differential, updated once per tic
        self.yaw_baseline = 0.0
        self._baseline_decay = (
            float(np.exp(-1.0 / 35.0 / self.cfg.tau_baseline))
            if self.cfg.tau_baseline > 0 else 0.0
        )
        self._seen_tics = 0

    def reset(self) -> None:
        self.rates = {k: 0.0 for k in self.pop}
        self.yaw_baseline = 0.0
        self._seen_tics = 0
        self.attack.reset()
        self.use.reset()
        self._filt = None

    # -- per-substep -----------------------------------------------------

    def observe(self, net) -> None:
        """Accumulate one simulation substep. Call every LIF step."""
        if self._filt is None:
            self._filt = torch.zeros(net.n, dtype=torch.float32,
                                     device=net.out.device)
        # Exponential rate estimate in Hz from spike-equivalent output.
        #
        # `out` is rate*dt per step, so recovering a RATE means dividing by dt,
        # not by tau. Dividing by tau instead scales every reported rate by
        # dt/tau -- a factor of 160 at dt=0.5 ms and tau=80 ms -- which made a
        # descending neuron firing at 141 Hz read out as 0.9 Hz and left the
        # agent permanently inside its deadzone.
        self._filt.mul_(self.decay).add_(net.out * (1.0 - self.decay) / self.dt)

    def sample(self) -> dict[str, float]:
        """Read the filtered rate of each population. Call once per Doom tic."""
        if self._filt is None:
            return dict(self.rates)
        for k, idx in self.pop.items():
            self.rates[k] = float(self._filt[idx].mean())
        return dict(self.rates)

    # -- decoding --------------------------------------------------------

    def _deadzone(self, x: float) -> float:
        dz = self.cfg.deadzone_hz
        if abs(x) <= dz:
            return 0.0
        return x - dz if x > 0 else x + dz

    def decode(self) -> dict[str, float]:
        """Named action values for the current filtered rates."""
        c = self.cfg
        r = self.rates

        # --- yaw: the L-R differential IS the steering command.
        # Sign convention: ViZDoom's TURN_LEFT_RIGHT_DELTA is positive for a
        # LEFT turn. A fly turns toward the side whose DNa02 is more active,
        # so left-minus-right maps straight through.
        raw = r.get("DNa02_L", 0.0) - r.get("DNa02_R", 0.0)
        self._seen_tics += 1
        warming = self._seen_tics <= c.warmup_tics
        if c.tau_baseline > 0:
            if warming:
                # track the raw value while the rate filter converges, so the
                # baseline we finish with reflects a settled estimate
                self.yaw_baseline = raw
            else:
                self.yaw_baseline = (self._baseline_decay * self.yaw_baseline
                                     + (1 - self._baseline_decay) * raw)
            raw = raw - self.yaw_baseline
        if warming:
            raw = 0.0
        diff = self._deadzone(raw)
        yaw = float(np.clip(diff * c.yaw_gain, -c.yaw_max_deg, c.yaw_max_deg))

        # --- forward / backward. BPN drives walking, MDN drives moonwalking;
        # they oppose, so the net is their difference rather than two buttons
        # that can both be held at once.
        fwd = r.get("BPN", 0.0) - r.get("MDN", 0.0)
        forward = float(np.clip(fwd * c.forward_gain,
                                -c.forward_max, c.forward_max))

        # --- lateral escape. The giant fiber is an all-or-nothing escape, and
        # its direction is set by which side fired.
        gf = r.get("DNp01_L", 0.0) - r.get("DNp01_R", 0.0)
        if "DNp01_L" not in r and "DNp01" in r:
            gf = 0.0
        lateral = float(np.clip(gf * c.lateral_gain,
                                -c.lateral_max, c.lateral_max))

        return {
            "TURN_LEFT_RIGHT_DELTA": yaw,
            "MOVE_FORWARD_BACKWARD_DELTA": forward,
            "MOVE_LEFT_RIGHT_DELTA": lateral,
            "ATTACK": float(self.attack.update(r.get("aggression", 0.0))),
            "USE": float(self.use.update(r.get("MN9", 0.0))),
        }

    def action_vector(self, button_names: list[str]) -> list[float]:
        """Order the decoded values to match a ViZDoom button list."""
        d = self.decode()
        return [d.get(name, 0.0) for name in button_names]

    # -- reporting -------------------------------------------------------

    def summary(self) -> str:
        lines = [f"rate filter tau {self.cfg.tau_rate * 1e3:.0f} ms, "
                 f"yaw gain {self.cfg.yaw_gain}"]
        for k, idx in sorted(self.pop.items()):
            lines.append(f"  {k:<12} {len(idx):>4} cells")
        for k in self.missing:
            lines.append(f"  {k:<12}    - unresolved, contributes nothing")
        return "\n".join(lines)
