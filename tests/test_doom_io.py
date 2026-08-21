"""Tests for the Doom I/O layer that do NOT need Doom running.

The vision path is pure geometry and signal processing over a numpy array, so
it can be tested against synthetic frames. The parts that genuinely need a
running Doom process (episode handling, label buffer) are covered by M5 and M6
themselves; duplicating them here would trade a slow test for a slower one.

What these pin down is the stuff that was silently wrong at some point:
mean-fill outside the viewport, the eye splay that makes the two eyes differ,
Gaussian acceptance rather than nearest-neighbour, and adaptation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from flydoom.interocept import Interoception, InteroceptConfig  # noqa: E402
from flydoom.retina import LuminanceAdaptation  # noqa: E402

DEV = "cpu"
DT = 5e-4


# ==========================================================================
# Luminance adaptation
# ==========================================================================


def _adapt(n=4, **kw):
    return LuminanceAdaptation({"left": n}, dt=DT, device=DEV, **kw)


def test_adaptation_removes_a_constant_level():
    """Weber coding: a steady scene must converge to the neutral midpoint,
    whatever its absolute brightness. This is what makes the retina work in
    both a bright and a dark room."""
    for level in (0.15, 0.5, 0.9):
        a = _adapt()
        lum = torch.full((4,), level, device=DEV)
        for _ in range(int(2.0 / DT)):
            out = a.drive("left", lum, invert=True)
        assert float(out.mean()) == pytest.approx(0.5, abs=0.02), (
            f"level {level} did not adapt to neutral"
        )


def test_adaptation_responds_to_change():
    a = _adapt()
    bright = torch.ones(4, device=DEV)
    for _ in range(int(2.0 / DT)):
        a.drive("left", bright, invert=True)
    dark = a.drive("left", torch.zeros(4, device=DEV), invert=True)
    assert float(dark.mean()) > 0.9, "a step to dark should drive hard"


def test_adaptation_is_per_column():
    """Each ommatidium adapts independently -- a bright patch must not
    desensitise its neighbours."""
    a = _adapt()
    lum = torch.tensor([0.1, 0.9, 0.1, 0.9], device=DEV)
    for _ in range(int(2.0 / DT)):
        out = a.drive("left", lum, invert=True)
    assert float(out.std()) < 0.05, "columns did not adapt independently"


def test_adaptation_inversion_flips_the_sign():
    """L1/L2 depolarise to light OFF; getting this backwards inverts the whole
    optic lobe and reverses the optomotor response."""
    a1, a2 = _adapt(), _adapt()
    for _ in range(int(1.0 / DT)):
        a1.drive("left", torch.full((4,), 0.5, device=DEV), invert=True)
        a2.drive("left", torch.full((4,), 0.5, device=DEV), invert=False)
    dark = torch.zeros(4, device=DEV)
    assert float(a1.drive("left", dark, invert=True).mean()) > 0.5
    assert float(a2.drive("left", dark, invert=False).mean()) < 0.5


def test_adaptation_output_is_bounded():
    a = _adapt()
    for lum in (torch.zeros(4, device=DEV), torch.ones(4, device=DEV)):
        for _ in range(50):
            out = a.drive("left", lum, invert=True)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ==========================================================================
# Interoception -- health and damage as taste
# ==========================================================================


def _intero(n=10, **kw):
    return Interoception(np.array([0, 1]), np.array([2, 3]), n, DT,
                         InteroceptConfig(**kw), device=DEV)


def test_taste_is_silent_with_no_events():
    i = _intero()
    for _ in range(100):
        r = i.substep()
    assert float(r.max()) == 0.0
    assert not i.active


def test_healing_drives_sugar_not_bitter():
    i = _intero()
    i.on_tic(healed=25.0, damage=0.0)
    r = i.substep()
    assert float(r[0]) > 0 and float(r[1]) > 0        # sugar GRNs
    assert float(r[2]) == 0.0 and float(r[3]) == 0.0  # bitter GRNs


def test_damage_drives_bitter_not_sugar():
    i = _intero()
    i.on_tic(healed=0.0, damage=20.0)
    r = i.substep()
    assert float(r[2]) > 0 and float(r[3]) > 0
    assert float(r[0]) == 0.0 and float(r[1]) == 0.0


def test_taste_lingers_then_decays():
    """A real taste outlasts the event. A one-tic impulse would be gone before
    the SEZ finished responding -- sugar to MN9 takes tens of ms."""
    cfg_tau = 0.35
    i = _intero(tau_taste=cfg_tau)
    i.on_tic(healed=25.0, damage=0.0)
    first = float(i.substep()[0])
    for _ in range(int(cfg_tau / DT)):        # one tau later
        r = i.substep()
    later = float(r[0])
    assert 0 < later < first
    assert later == pytest.approx(first * np.exp(-1.0), rel=0.15)
    for _ in range(int(4 * cfg_tau / DT)):
        r = i.substep()
    assert float(r[0]) < 0.05 * first


def test_taste_is_graded_by_magnitude():
    """M2 established the dose-response is graded, so the injection must be."""
    small, big = _intero(), _intero()
    small.on_tic(healed=5.0, damage=0.0)
    big.on_tic(healed=25.0, damage=0.0)
    assert float(big.substep()[0]) > float(small.substep()[0]) * 2


def test_taste_saturates():
    i = _intero()
    for _ in range(20):
        i.on_tic(healed=100.0, damage=0.0)
    assert i.sweet <= 1.0
    assert float(i.substep()[0]) <= i.cfg.sugar_rate_hz


def test_sugar_and_bitter_coexist():
    """M2's suppression result depends on both arriving together."""
    i = _intero()
    i.on_tic(healed=25.0, damage=20.0)
    r = i.substep()
    assert float(r[0]) > 0 and float(r[2]) > 0


def test_reset_clears_taste():
    i = _intero()
    i.on_tic(healed=25.0, damage=25.0)
    i.substep()
    i.reset()
    assert not i.active
    assert float(i.substep().max()) == 0.0


def test_missing_gustatory_populations_do_not_crash():
    """M0 can legitimately fail to resolve a handle; that must degrade, not
    explode, in the middle of a Doom run."""
    i = Interoception(np.array([], dtype=np.int64), np.array([2]), 10, DT,
                      InteroceptConfig(), device=DEV)
    i.on_tic(healed=25.0, damage=25.0)
    r = i.substep()
    assert float(r[2]) > 0


# ==========================================================================
# Doom viewport geometry
# ==========================================================================


def test_vertical_fov_follows_from_aspect_ratio():
    from flydoom.doom import DoomConfig

    c = DoomConfig(width=320, height=240, fov_deg=130.0)
    # 4:3 viewport, so vertical is narrower than horizontal but not by 3/4 --
    # the relation is through the tangent, not linear in the angle.
    assert 100.0 < c.vfov_deg < 125.0
    wide = DoomConfig(width=640, height=240, fov_deg=130.0)
    assert wide.vfov_deg < c.vfov_deg


def test_eye_splay_makes_the_two_eyes_differ():
    """Without splay both eyes map azimuth-zero to screen centre, see the same
    image, and the DNa02 differential is zero BY CONSTRUCTION -- the agent can
    never turn. This is the geometry that fixes it."""
    from flydoom.retina import EYE_SPLAY_DEG

    assert EYE_SPLAY_DEG > 10.0, "splay too small to separate the eyes"
    left_gaze, right_gaze = -EYE_SPLAY_DEG, EYE_SPLAY_DEG
    assert abs(left_gaze - right_gaze) > 20.0
