"""Olfaction tests.

The load-bearing one is `test_smell_carries_no_direction`. Everything else in
this module is ordinary correctness; that one is what stops the channel
degenerating into a targeting oracle and invalidating every visual result in
the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from flydoom.olfaction import Olfaction, OlfactionConfig  # noqa: E402

DEV = "cpu"
DT = 5e-4
THREAT = np.array([0, 1, 2, 3])     # pretend: 2 left, 2 right
FOOD = np.array([4, 5])
N = 8


def make(**kw):
    cfg = OlfactionConfig(**kw)
    return Olfaction(THREAT, FOOD, N, DT, cfg, device=DEV, seed=1)


def settle(o, seconds=2.0):
    for _ in range(int(seconds / DT)):
        r = o.substep()
    return r


def enemy(dist, name="Demon"):
    return {"name": name, "distance": dist, "azimuth_deg": 0.0}


# -- THE property that matters ---------------------------------------------


def test_smell_carries_no_direction():
    """Every threat receptor must receive IDENTICAL drive.

    A fly's antennae are ~0.3 mm apart and cannot triangulate. If left and
    right ever differ here, the channel becomes a direction sensor and hands
    the agent the answer its visual system is supposed to compute — which
    would silently invalidate M6 and M7.
    """
    o = make()
    o.on_tic([{"name": "Demon", "distance": 120.0, "azimuth_deg": -55.0}])
    r = settle(o)
    vals = r[THREAT].tolist()
    assert len(set(vals)) == 1, f"receptors differ: {vals}"
    assert vals[0] > 0


def test_azimuth_is_ignored_entirely():
    """Same distance, wildly different bearings — identical response."""
    out = []
    for az in (-64.0, 0.0, +64.0):
        o = make()
        o.on_tic([{"name": "Demon", "distance": 200.0, "azimuth_deg": az}])
        out.append(float(settle(o)[THREAT[0]]))
    assert max(out) - min(out) < 1e-6, f"azimuth leaked into the signal: {out}"


# -- distance ---------------------------------------------------------------


def test_closer_smells_stronger():
    near, far = make(), make()
    near.on_tic([enemy(60.0)])
    far.on_tic([enemy(900.0)])
    assert float(settle(near)[THREAT[0]]) > float(settle(far)[THREAT[0]]) * 3


def test_half_strength_at_r_half():
    o = make(r_half=300.0, intermittency_hz=1e-6, duty=1.0)
    o.on_tic([enemy(300.0)])
    at_half = float(settle(o)[THREAT[0]])
    o2 = make(r_half=300.0, intermittency_hz=1e-6, duty=1.0)
    o2.on_tic([enemy(1.0)])
    at_zero = float(settle(o2)[THREAT[0]])
    assert at_half == pytest.approx(at_zero * 0.5, rel=0.15)


def test_sources_sum_but_saturate():
    """Three enemies smell stronger than one, but not three times — receptors
    have a ceiling."""
    one, three = make(), make()
    one.on_tic([enemy(300.0)])
    three.on_tic([enemy(300.0), enemy(300.0), enemy(300.0)])
    a = float(settle(one)[THREAT[0]])
    b = float(settle(three)[THREAT[0]])
    assert b > a
    assert b < 3 * a
    assert b <= OlfactionConfig().max_rate_hz * 1.001


# -- plume dynamics ---------------------------------------------------------


def test_smell_lingers_after_the_source_disappears():
    """MEASURED: ViZDoom labels contain only on-screen objects, so an enemy
    stepping behind a pillar vanishes instantly. Strict line-of-sight would be
    a LIGHT model; odour drifts around corners and persists."""
    o = make(tau_plume=1.5, intermittency_hz=1e-6, duty=1.0)
    o.on_tic([enemy(150.0)])
    peak = float(settle(o, 2.0)[THREAT[0]])
    o.on_tic([])                       # source occluded
    after = float(settle(o, 0.5)[THREAT[0]])
    assert after > 0.25 * peak, "smell vanished the instant it went out of sight"
    gone = float(settle(o, 8.0)[THREAT[0]])
    assert gone < 0.02 * peak, "smell never faded"


def test_onset_is_not_instant():
    """Odour has to physically arrive."""
    o = make(intermittency_hz=1e-6, duty=1.0)
    o.on_tic([enemy(100.0)])
    first = float(o.substep()[THREAT[0]])
    later = float(settle(o, 1.5)[THREAT[0]])
    assert first < 0.2 * later


# -- channels ---------------------------------------------------------------


def test_food_and_threat_use_different_receptors():
    o = make(intermittency_hz=1e-6, duty=1.0)
    o.on_tic([{"name": "Medikit", "distance": 100.0, "azimuth_deg": 0.0}])
    r = settle(o)
    assert float(r[FOOD[0]]) > 0
    assert float(r[THREAT[0]]) == 0.0


def test_both_channels_can_be_active_together():
    o = make(intermittency_hz=1e-6, duty=1.0)
    o.on_tic([enemy(150.0),
              {"name": "Stimpack", "distance": 150.0, "azimuth_deg": 20.0}])
    r = settle(o)
    assert float(r[THREAT[0]]) > 0 and float(r[FOOD[0]]) > 0


def test_non_sources_are_ignored():
    """Bullet puffs and the player itself are not smells."""
    o = make()
    o.on_tic([{"name": "DoomPlayer", "distance": 0.0, "azimuth_deg": 0.0},
              {"name": "BulletPuff", "distance": 50.0, "azimuth_deg": 0.0}])
    assert float(settle(o).max()) == 0.0


def test_unknown_actor_emits_nothing_and_does_not_crash():
    """An unrecognised actor must degrade silently, not become a rival fly.

    This used to fall back to THREAT, and the fallback was the bug: cVA means
    "another fly is here", so classifying every unlisted object that way made
    ammo crates smell like rivals (~3,900 sightings in deathmatch) and, because
    `health_gathering_supreme` names its pickups `CustomMedikit`, turned the
    one health-collection map's 802 medkits into pheromone with zero food.
    Both channels are explicit allowlists now.
    """
    o = make(intermittency_hz=1e-6, duty=1.0)
    o.on_tic([{"name": "SomeModdedMonster", "distance": 100.0, "azimuth_deg": 0}])
    assert float(settle(o).max()) == 0.0


def test_health_pickups_of_every_scenario_smell_like_food():
    """The names really used by the shipped scenarios, not a guess at them."""
    from flydoom.olfaction import FOOD_NAMES, THREAT_NAMES
    for name in ("Medikit", "CustomMedikit", "Stimpack", "GreenArmor"):
        assert name in FOOD_NAMES, name
        assert name not in THREAT_NAMES, name


def test_inanimate_objects_are_not_rivals():
    """cVA is a fly pheromone; a crate cannot emit it."""
    from flydoom.olfaction import FOOD_NAMES, THREAT_NAMES
    for name in ("ClipBox", "ShellBox", "RocketBox", "Chainsaw",
                 "RocketLauncher"):
        assert name not in THREAT_NAMES, name
        assert name not in FOOD_NAMES, name
    o = make(intermittency_hz=1e-6, duty=1.0)
    o.on_tic([{"name": "ClipBox", "distance": 100.0, "azimuth_deg": 0}])
    assert float(settle(o).max()) == 0.0


def test_monsters_are_rivals():
    o = make(intermittency_hz=1e-6, duty=1.0)
    o.on_tic([{"name": "DoomImp", "distance": 100.0, "azimuth_deg": 0}])
    assert float(settle(o)[THREAT[0]]) > 0


# -- hygiene ----------------------------------------------------------------


def test_silent_with_nothing_around():
    o = make()
    o.on_tic([])
    assert float(settle(o).max()) == 0.0
    assert not o.active


def test_reset_clears_the_plume():
    o = make()
    o.on_tic([enemy(80.0)])
    settle(o)
    o.reset()
    assert float(o.rate.max()) == 0.0
    assert not o.active


def test_missing_receptor_population_does_not_crash():
    """M0 can legitimately fail to resolve a glomerulus."""
    o = Olfaction(np.array([], dtype=np.int64), FOOD, N, DT,
                  OlfactionConfig(intermittency_hz=1e-6, duty=1.0),
                  device=DEV, seed=1)
    o.on_tic([enemy(100.0),
              {"name": "Medikit", "distance": 100.0, "azimuth_deg": 0.0}])
    assert float(settle(o)[FOOD[0]]) > 0
