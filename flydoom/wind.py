"""Airflow: the cue a fly actually steers by when it is following a smell.

WHY THIS EXISTS

Measured in this project: strengthening the odour channel fivefold, so that
every medkit in the level reaches the animal with no line of sight, bought a
blind model -0.07 against its matched control. Odour alone does not find food
here, and the reason is structural rather than a tuning failure.
flydoom/olfaction.py is bilaterally symmetric by construction -- the
concentration is a scalar and every azimuth is discarded, because a
direction-carrying odour input would be the confound the study exists to avoid.
A scalar cannot steer. No amount of it can.

A real fly has exactly this problem and does not solve it with its nose. In
turbulent air an odour gradient is nearly useless: concentration flickers and
points nowhere. So the animal uses odour as a GATE and the WIND as the steering
cue -- it casts across the flow, and on contacting a plume it surges upwind.
Direction comes from the antennae, which is a mechanosensory channel, not a
chemical one.

This module supplies that missing half, and it supplies it as PHYSICS rather
than as a hint:

  * The plume decides WHETHER a source can be smelled at all. Odour travels
    downwind; a fly upwind of a medkit smells nothing, however close. That is a
    property of air, and it is what makes going upwind worth doing.

  * The antennae carry WHERE THE FLOW COMES FROM, and they carry it the way the
    animal's do -- as a difference in deflection between two sensors placed
    either side of the midline, driven into Johnston's organ afferents that
    already exist in the connectome and are already wired into the model for
    touch. Nothing is told to the brain; two numbers arrive at two populations
    and the wiring makes of them what it can.

WHAT WOULD COUNT, and it is worth stating before running rather than after.
Health should rise only when odour AND wind are both present. Wind alone would
mean the model has found a compass, not a food source -- useful, and a
different claim. Odour alone is already measured, and it is nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WindConfig:
    """Airflow over the arena. Environment, not brain."""

    direction_deg: float = 0.0
    """The direction the wind BLOWS TOWARD, in Doom's world frame, degrees
    counter-clockwise from +x. Odour therefore travels this way, and a fly
    wanting the source must go the other way."""

    speed: float = 1.0
    """Antennal deflection at a full headwind, 0 to 1, on the same scale the
    touch channel uses. 0 disables the channel, which is the default state of
    every result before this."""

    antenna_splay_deg: float = 40.0
    """Half-angle between the two antennae. This is what converts a single
    flow direction into a LEFT-RIGHT DIFFERENCE, and so it is the only reason
    the channel carries direction at all. Drosophila's arista sit roughly this
    far off the midline."""

    plume_half_angle_deg: float = 35.0
    """How wide the downwind plume spreads. Wider is a weaker cue, because more
    of the arena smells of everything."""

    plume_length: float = 1200.0
    """How far downwind a source can still be smelled, in map units. Beyond it
    the plume has dispersed."""

    meander_deg: float = 0.0
    """How far the wind swings either side of `direction_deg`. 0 is a
    steady wind, which is what the first run used and why it produced nothing.

    MEASURED, and it is the same trap M12 fell into with the optomotor drum.
    The airflow signal reaches DNp15 at about 8 Hz of left-right differential,
    consistently across seeds -- the channel works. But the motor decoder
    removes a 3 s running baseline from the yaw command, so a CONSTANT
    differential is exactly the DC it exists to delete. A fly settled on a
    heading in a steady wind therefore commands nothing, however clear the
    signal at the neuron. M12 solved this for the drum by reversing it on a
    square wave inside the decoder's passband; real wind meanders on its own,
    so this is physics rather than a contrivance."""

    meander_period_s: float = 4.0
    """Seconds per swing. 4 s is 0.25 Hz, well clear of the decoder's
    0.05 Hz corner, which is the band M12 used for the same reason."""

    plume_floor: float = 0.05
    """What a source contributes when the fly is NOT in its plume. Not zero:
    still air and eddies leave a little odour everywhere, and a hard zero would
    make the channel a cleaner cue than air ever is."""


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def direction_at(t_s: float, cfg: WindConfig) -> float:
    """Where the wind blows toward at time `t_s`, meander included."""
    if not cfg.meander_deg or cfg.meander_period_s <= 0:
        return cfg.direction_deg
    return cfg.direction_deg + cfg.meander_deg * math.sin(
        2.0 * math.pi * t_s / cfg.meander_period_s)


def antennal_deflection(heading_deg: float, cfg: WindConfig,
                        t_s: float = 0.0) -> tuple[float, float]:
    """(left, right) antennal deflection, 0 to 1, for a fly facing `heading_deg`.

    The wind comes FROM `direction_deg + 180`. Each antenna is exposed by the
    component of that flow along its own axis, and shielded when the flow comes
    from behind it, so a headwind deflects both equally, a wind from the left
    deflects the left, and a tailwind deflects neither.
    """
    if cfg.speed <= 0.0:
        return 0.0, 0.0
    comes_from = _wrap(direction_at(t_s, cfg) + 180.0 - heading_deg)
    s = math.radians(cfg.antenna_splay_deg)
    b = math.radians(comes_from)
    left = max(0.0, math.cos(b - s)) * cfg.speed
    right = max(0.0, math.cos(b + s)) * cfg.speed
    return min(left, 1.0), min(right, 1.0)


def plume_weight(bearing_deg: float, distance: float, cfg: WindConfig,
                 t_s: float = 0.0) -> float:
    """How much of a source's odour reaches a fly that sees it at
    `bearing_deg` (world frame, fly to source) and `distance` away.

    The fly is in the plume when the vector FROM THE SOURCE TO ITSELF points
    along the wind. Upwind of a source, it smells almost nothing however close
    it is, which is the whole point: it makes going upwind informative.
    """
    if cfg.speed <= 0.0:
        return 1.0                      # no wind: the old, isotropic odour
    if distance > cfg.plume_length:
        return cfg.plume_floor
    off = abs(_wrap(bearing_deg + 180.0 - direction_at(t_s, cfg)))
    if off >= cfg.plume_half_angle_deg:
        return cfg.plume_floor
    # cosine taper across the plume rather than a hard edge
    t = math.cos(math.radians(off) * 90.0 / cfg.plume_half_angle_deg)
    return cfg.plume_floor + (1.0 - cfg.plume_floor) * max(0.0, t)
