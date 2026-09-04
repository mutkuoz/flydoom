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

    # ---- the three graded gains -------------------------------------
    #
    # These map Hz of descending-neuron differential onto Doom command units,
    # and there is no biological number to copy: a fly saccades at hundreds of
    # deg/s and Doom's turn rate does not. Spec 6.2 flags the scalar and leaves
    # it to be tuned, which invites tuning each channel until the agent "looks
    # right" -- three unfalsifiable knobs.
    #
    # What each channel needs turns out to follow from WHAT IT IS, and the two
    # cases are not the same:
    #
    # BILATERAL PAIRS -- yaw (DNa02_L/_R) and lateral (DNp01_L/_R). These read
    # a difference between the same cell type on the two sides, so a standing
    # difference is a reconstruction artifact rather than a command: FAFB is
    # one real brain and its halves are not mirror images. Measured, DNa02 sits
    # at 150/257 Hz and DNp01 at 110/206. Both get the DC removed
    # (see tau_baseline) and then a gain set so three standard deviations of
    # what is left reach the clamp.
    #
    # NOT A PAIR -- forward (BPN - MDN). Two different cell types that oppose
    # each other, so a positive standing difference is a real tonic walk
    # command, and centring it would be deleting locomotion rather than
    # deleting an artifact. It keeps its DC; the gain instead places that DC at
    # a cruise speed the clamp does not cut off.
    #
    # MEASURED over 400 tics x 3 scenarios (deathmatch,
    # health_gathering_supreme, defend_the_center), skipping 40 settling tics:
    #
    #     channel   centred s.d.   across scenarios   clamp   rule       gain
    #     yaw           9.16 Hz     8.69 - 9.74        12     3 sigma    0.44
    #     lateral       6.12 Hz     5.95 - 6.36        20     3 sigma    1.09
    #     forward       1.27 Hz     1.22 - 1.31        22     DC at 50%  0.40
    #
    # The spreads are tight across three very different maps, so neither rule
    # is fitted to one scenario. As a check on the 3-sigma rule itself: yaw was
    # hand-tuned to 0.35 long before this, and the rule independently lands on
    # 0.44 -- within 25% of a value already known to be stable.
    #
    # Worth stating plainly, because it bounds what the forward channel can
    # mean: BPN - MDN is +27.6 +- 1.3 Hz, so its modulation is 4.6% of its own
    # mean. Whatever gain it gets, this agent walks at a near-constant speed.
    # The forward channel carries a tonic command and almost no scene.

    yaw_gain: float = 0.44
    """Degrees of turn per tic per Hz of the yaw differential."""

    yaw_max_deg: float = 12.0
    """Clamp per tic. Doom turns feel unusable past roughly this."""

    forward_gain: float = 0.40
    forward_max: float = 22.0
    """Doom map units per tic. ~22 is a normal run speed."""

    lateral_gain: float = 1.1
    lateral_max: float = 20.0

    centre_channels: tuple[str, ...] = ("yaw", "lateral")
    """Which graded channels get their DC removed. See the block above: the
    bilateral pairs do, the walk command does not."""

    attack_on: float = 25.0
    attack_off: float = 15.0
    use_on: float = 20.0
    use_off: float = 12.0

    yaw_source: str = "DNa02"
    """Which bilateral descending pair supplies yaw: "DNa02" (the reported
    model) or "DNp15". See MotorDecoder.yaw_pair for why the choice matters."""

    tau_baseline: float = 3.0
    """Seconds. Time constant for adapting out a CONSTANT command offset.

    MEASURED, and this is why it exists. DNa02_R sits about 2x above DNa02_L
    permanently (255 vs 130 Hz), because the two cells' input wiring is not
    symmetric -- FAFB is one real brain and its two halves are not mirror
    images. A fixed differential means a fixed turn command, and the agent
    spins forever.

    The SAME asymmetry sits on the other two graded channels, and leaving it
    there is worse than the spin because it is silent. Measured over 300 tics
    of deathmatch:

        BPN - MDN     = +26.8 +- 2.8 Hz  -> x0.9 = 24.2 against a clamp of 22
        DNp01_L - _R  = -99.0 +- 21  Hz  -> x2.5 = -247 against a clamp of 20

    Both sit past their clamps, so forward ran at full speed and lateral
    strafed hard left on 89% of tics, and the +-2.8 Hz of actual modulation
    was clipped off the top. The channels looked alive and carried nothing.
    Removing the DC is what makes the modulation visible; it is the same
    keep-the-transients-discard-the-DC move as the retina's Weber adaptation.

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
        # slow baseline per graded channel, updated once per tic
        self.baseline = {"yaw": 0.0, "forward": 0.0, "lateral": 0.0}
        self._baseline_decay = (
            float(np.exp(-1.0 / 35.0 / self.cfg.tau_baseline))
            if self.cfg.tau_baseline > 0 else 0.0
        )
        self._seen_tics = 0

    def reset(self) -> None:
        self.rates = {k: 0.0 for k in self.pop}
        self.baseline = {k: 0.0 for k in self.baseline}
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

    def _centre(self, name: str, raw: float, warming: bool) -> float:
        """Subtract this channel's slow baseline, tracking it while warming.

        Channels outside `centre_channels` keep their DC and are only gated
        during warmup, while the rate filter converges.
        """
        if self.cfg.tau_baseline <= 0 or name not in self.cfg.centre_channels:
            return 0.0 if warming else raw
        if warming:
            # track the raw value while the rate filter converges, so the
            # baseline we finish with reflects a settled estimate
            self.baseline[name] = raw
            return 0.0
        d = self._baseline_decay
        self.baseline[name] = d * self.baseline[name] + (1 - d) * raw
        return raw - self.baseline[name]

    def yaw_pair(self) -> tuple[str, str]:
        """Which bilateral pair supplies the steering differential.

        DNa02 is a documented steering neuron, but for goal-directed walking:
        it is targeted directly by central-complex output (PFL3), sits two
        synapses from the head-direction system, and in this connectome draws
        2.3% of its input from visual populations against 97.7% central. It is
        the navigation pathway, not the optomotor one.

        DNp15 (DNHS1) is the optomotor pathway: HS cells drive it for yaw
        rotations, and it is the strongest descending target of the horizontal
        system here (32.3% of HS descending output, against 7.1% to DNa02).
        Its axons reach the neck motor system, so it rotates gaze rather than
        body -- which is what this environment's yaw control actually is,
        since the camera is the head.

        Selected by published function and by connectivity, not by which
        readout carried the signal we were looking for. See yaw_source.
        """
        return (("DNp15_L", "DNp15_R") if self.cfg.yaw_source == "DNp15"
                else ("DNa02_L", "DNa02_R"))

    def decode(self) -> dict[str, float]:
        """Named action values for the current filtered rates."""
        c = self.cfg
        r = self.rates

        # --- yaw: the L-R differential IS the steering command.
        # Sign convention: ViZDoom's TURN_LEFT_RIGHT_DELTA is positive for a
        # LEFT turn. A fly turns toward the side whose DNa02 is more active,
        # so left-minus-right maps straight through.
        self._seen_tics += 1
        warming = self._seen_tics <= c.warmup_tics

        yl, yr = self.yaw_pair()
        raw = self._centre("yaw", r.get(yl, 0.0) - r.get(yr, 0.0), warming)
        diff = self._deadzone(raw)
        yaw = float(np.clip(diff * c.yaw_gain, -c.yaw_max_deg, c.yaw_max_deg))

        # --- forward / backward. BPN drives walking, MDN drives moonwalking;
        # they oppose, so the net is their difference rather than two buttons
        # that can both be held at once. NOT centred -- see the gain block in
        # MotorConfig for why this DC is a command and not an artifact.
        fwd = self._centre("forward",
                           r.get("BPN", 0.0) - r.get("MDN", 0.0), warming)
        # no deadzone here: it is applied to centred channels to reject
        # two-cell counting noise around zero, and this channel's operating
        # point is +27.6 Hz, not zero.
        forward = float(np.clip(fwd * c.forward_gain,
                                -c.forward_max, c.forward_max))

        # --- lateral escape. The giant fiber is an all-or-nothing escape, and
        # its direction is set by which side fired. Same standing asymmetry as
        # DNa02, and twice as large.
        gf = r.get("DNp01_L", 0.0) - r.get("DNp01_R", 0.0)
        if "DNp01_L" not in r and "DNp01" in r:
            gf = 0.0
        gf = self._centre("lateral", gf, warming)
        lateral = float(np.clip(self._deadzone(gf) * c.lateral_gain,
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
