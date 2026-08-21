"""Motor decoder tests.

These exist because of a real bug. The rate filter divided by `tau` where it
should divide by `dt`, scaling every reported rate by dt/tau -- a factor of 160
at dt=0.5 ms and tau=80 ms. A descending neuron firing at 141 Hz read out as
0.9 Hz and the agent sat inside its deadzone doing nothing, while every
milestone before M5 still passed because none of them used the decoder.

The lesson: a unit conversion between two modules needs a test at the boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from flydoom.motor import MotorConfig, MotorDecoder, SchmittTrigger  # noqa: E402

DEV = "cpu"
DT = 5e-4


class FakeNet:
    """Minimal stand-in: just the `out` vector the decoder reads."""

    def __init__(self, n=8):
        self.n = n
        self.out = torch.zeros(n, device=DEV)

    def fire(self, idx, rate_hz, dt=DT):
        """Set steady spike-equivalent output for a given rate."""
        self.out.zero_()
        self.out[idx] = rate_hz * dt


# -- Schmitt trigger --------------------------------------------------------


def test_schmitt_needs_the_high_threshold_to_switch_on():
    t = SchmittTrigger(on_at=10.0, off_at=5.0)
    assert not t.update(9.9)
    assert t.update(10.1)


def test_schmitt_holds_between_thresholds():
    """The entire point: a value drifting in the band does not toggle."""
    t = SchmittTrigger(on_at=10.0, off_at=5.0)
    t.update(20.0)
    for v in (9.0, 6.0, 8.0, 5.5, 7.0):
        assert t.update(v), "dropped out inside the hysteresis band"
    assert not t.update(4.9)


def test_schmitt_rejects_inverted_thresholds():
    """off_at above on_at oscillates by construction; refuse it loudly."""
    with pytest.raises(ValueError, match="oscillates"):
        SchmittTrigger(on_at=5.0, off_at=10.0)


def test_schmitt_suppresses_chatter_on_a_noisy_signal():
    # Noise must fit INSIDE the band or hysteresis cannot help -- 8.5/11.5
    # against a band of 8..12. A band narrower than the noise flips anyway,
    # which is a property of the signal, not a bug in the trigger.
    noisy = [10.0 + 1.5 * ((-1) ** i) for i in range(40)]   # 8.5 <-> 11.5
    hyst = SchmittTrigger(on_at=12.0, off_at=8.0)
    plain_flips = sum(1 for a, b in zip(noisy, noisy[1:])
                      if (a > 10) != (b > 10))
    hyst.update(20.0)                     # latch on first
    states = [hyst.update(v) for v in noisy]
    hyst_flips = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert plain_flips > 30, "the raw signal should chatter"
    assert hyst_flips == 0, "hysteresis should hold through in-band noise"


# -- rate estimation: the bug that motivated this file ----------------------


def _decoder(pops, cfg=None):
    return MotorDecoder(pops, DT, cfg or MotorConfig(), device=DEV)


def test_rate_filter_converges_to_the_true_rate():
    """`out` is rate*dt per step, so recovering Hz means dividing by dt.

    Dividing by tau instead under-reports by dt/tau = 160x.
    """
    net = FakeNet()
    d = _decoder({"X": np.array([0, 1])})
    net.fire([0, 1], 140.0)
    for _ in range(int(1.0 / DT)):            # 1 s, >> tau
        d.observe(net)
    assert d.sample()["X"] == pytest.approx(140.0, rel=0.02)


@pytest.mark.parametrize("hz", [5.0, 60.0, 250.0])
def test_rate_filter_is_linear_in_rate(hz):
    net = FakeNet()
    d = _decoder({"X": np.array([0])})
    net.fire([0], hz)
    for _ in range(int(1.0 / DT)):
        d.observe(net)
    assert d.sample()["X"] == pytest.approx(hz, rel=0.02)


def test_rate_filter_tracks_a_change_within_a_few_tau():
    net = FakeNet()
    cfg = MotorConfig(tau_baseline=0.0)
    d = _decoder({"X": np.array([0])}, cfg)
    net.fire([0], 200.0)
    for _ in range(int(0.5 / DT)):
        d.observe(net)
    net.fire([0], 0.0)
    for _ in range(int(5 * cfg.tau_rate / DT)):
        d.observe(net)
    assert d.sample()["X"] < 5.0


# -- steering ---------------------------------------------------------------


def _steer(left_hz, right_hz, cfg=None, tics=None):
    """Steer after the warmup window has passed.

    The decoder deliberately outputs zero yaw for its first `warmup_tics` while
    the rate filter converges, so any test of steering must run past that.
    """
    cfg = cfg or MotorConfig(tau_baseline=0.0)
    tics = tics if tics is not None else cfg.warmup_tics + 3
    net = FakeNet()
    d = _decoder({"DNa02_L": np.array([0]), "DNa02_R": np.array([1])}, cfg)
    for _ in range(tics):
        net.out.zero_()
        net.out[0] = left_hz * DT
        net.out[1] = right_hz * DT
        for _ in range(int(0.2 / DT)):
            d.observe(net)
        d.sample()
        act = d.decode()
    return act["TURN_LEFT_RIGHT_DELTA"]


def test_yaw_follows_the_left_right_differential():
    assert _steer(100.0, 20.0) > 0      # left louder -> turn left
    assert _steer(20.0, 100.0) < 0
    assert _steer(60.0, 60.0) == pytest.approx(0.0, abs=1e-6)


def test_yaw_is_clamped():
    cfg = MotorConfig(tau_baseline=0.0)
    assert abs(_steer(5000.0, 0.0, cfg)) == pytest.approx(cfg.yaw_max_deg)


def test_yaw_deadzone_ignores_tiny_differentials():
    """Two single-cell populations produce a nonzero difference from noise
    alone; the deadzone stops that from becoming a steering command."""
    cfg = MotorConfig(tau_baseline=0.0, deadzone_hz=5.0)
    assert _steer(52.0, 50.0, cfg) == pytest.approx(0.0, abs=1e-6)
    assert abs(_steer(80.0, 50.0, cfg)) > 0


def test_baseline_adaptation_cancels_a_constant_offset():
    """THE closed-loop fix. A permanent DNa02 asymmetry -- which this
    connectome genuinely has, R sits ~2x above L -- would otherwise be a
    permanent turn command and the agent spins forever."""
    cfg = MotorConfig(tau_baseline=1.0)
    net = FakeNet()
    d = _decoder({"DNa02_L": np.array([0]), "DNa02_R": np.array([1])}, cfg)
    yaws = []
    for _ in range(200):                       # ~6 s of tics
        net.out.zero_()
        net.out[0] = 130.0 * DT
        net.out[1] = 255.0 * DT
        for _ in range(57):
            d.observe(net)
        d.sample()
        yaws.append(d.decode()["TURN_LEFT_RIGHT_DELTA"])
    # The baseline is SEEDED from the first tics, so a differential already
    # present at startup reads as neutral -- the decoder auto-zeroes like any
    # sensor. What must never happen is a constant offset producing a sustained
    # turn command.
    assert abs(yaws[0]) < 0.5, "startup should be neutral, not a lurch"
    assert abs(yaws[-1]) < 0.5, "constant offset became a standing turn"
    # No startup lurch: a purely constant input must never produce a turn, and
    # before the warmup fix this clamped at 12 deg/tic for about a second.
    assert max(abs(y) for y in yaws) < 2.0, "constant input caused a transient"


def test_warmup_suppresses_steering_until_the_filter_settles():
    cfg = MotorConfig(tau_baseline=1.0, warmup_tics=6)
    net = FakeNet()
    d = _decoder({"DNa02_L": np.array([0]), "DNa02_R": np.array([1])}, cfg)
    out = []
    for _ in range(10):
        net.out.zero_(); net.out[0] = 50.0 * DT; net.out[1] = 400.0 * DT
        for _ in range(57):
            d.observe(net)
        d.sample()
        out.append(d.decode()["TURN_LEFT_RIGHT_DELTA"])
    assert all(y == 0.0 for y in out[:cfg.warmup_tics]), "steered while warming"


def test_baseline_adaptation_preserves_transients():
    """It must remove the DC without eating real signal."""
    cfg = MotorConfig(tau_baseline=1.0, deadzone_hz=0.0)
    net = FakeNet()
    d = _decoder({"DNa02_L": np.array([0]), "DNa02_R": np.array([1])}, cfg)
    for _ in range(200):                       # settle on a constant offset
        net.out.zero_(); net.out[0] = 100.0 * DT; net.out[1] = 200.0 * DT
        for _ in range(57):
            d.observe(net)
        d.sample(); d.decode()
    settled = abs(d.decode()["TURN_LEFT_RIGHT_DELTA"])
    net.out.zero_(); net.out[0] = 300.0 * DT; net.out[1] = 200.0 * DT
    for _ in range(57):
        d.observe(net)
    d.sample()
    assert abs(d.decode()["TURN_LEFT_RIGHT_DELTA"]) > settled + 1.0


# -- forward / backward and buttons ----------------------------------------


def test_forward_and_backward_oppose():
    """BPN walks, MDN moonwalks. They cannot both be held at once."""
    cfg = MotorConfig(tau_baseline=0.0)
    net = FakeNet()
    d = _decoder({"BPN": np.array([0]), "MDN": np.array([1])}, cfg)
    def run(bpn, mdn):
        d.reset()                       # otherwise the filter carries history
        net.out.zero_(); net.out[0] = bpn * DT; net.out[1] = mdn * DT
        for _ in range(int(1.0 / DT)):
            d.observe(net)
        d.sample()
        return d.decode()["MOVE_FORWARD_BACKWARD_DELTA"]
    assert run(80.0, 0.0) > 0
    assert run(0.0, 80.0) < 0
    assert run(50.0, 50.0) == pytest.approx(0.0, abs=1e-6)


def test_action_vector_matches_button_order():
    d = _decoder({"BPN": np.array([0])})
    names = ["ATTACK", "TURN_LEFT_RIGHT_DELTA", "USE"]
    v = d.action_vector(names)
    assert len(v) == 3 and all(isinstance(x, float) for x in v)


def test_unresolved_populations_are_reported_not_crashed():
    d = _decoder({"BPN": np.array([0]), "GHOST": np.array([], dtype=np.int64)})
    assert "GHOST" in d.missing
    d.sample()
    assert d.decode()["TURN_LEFT_RIGHT_DELTA"] == 0.0


def test_reset_clears_baseline_and_triggers():
    cfg = MotorConfig(tau_baseline=1.0)
    net = FakeNet()
    d = _decoder({"DNa02_L": np.array([0]), "DNa02_R": np.array([1])}, cfg)
    net.fire([0], 200.0)
    for _ in range(100):
        d.observe(net)
    d.sample()
    d.reset()
    assert d.yaw_baseline == 0.0
    assert all(v == 0.0 for v in d.rates.values())
