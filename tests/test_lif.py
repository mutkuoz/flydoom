"""LIF integrator tests.

These check the discretisation against closed-form solutions of the equations
in Shiu et al.'s Methods. They are independent of whether the parameter VALUES
are biologically right — they validate the maths, not the biology.

The heavier version with printed traces is experiments/m15_lif.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

from flydoom.lif import LIFNetwork, LIFParams, poisson_rate_vector  # noqa: E402

DEV = "cpu"  # deterministic and fast; the maths is device-independent


@pytest.fixture
def p() -> LIFParams:
    return LIFParams()


# -- closed-form helpers ---------------------------------------------------


def test_peak_psp_is_a_small_fraction_of_w(p):
    """W_SYN is NOT the voltage one synapse delivers.

    Because tau_syn < tau_mem the membrane never reaches V_rest + g. At the
    nominal taus the peak is ~16% of w. Reasoning about gain with W_SYN
    directly overestimates by ~6x.
    """
    frac = p.peak_psp_fraction()
    assert 0.10 < frac < 0.25
    assert p.psp_peak_time() > p.tau_syn      # peak lags the synaptic input


def test_rheobase_is_the_threshold_distance(p):
    assert p.analytic_rate(p.rheobase() * 0.999) == 0.0
    assert p.analytic_rate(p.rheobase() * 1.001) > 0.0


def test_analytic_rate_is_monotonic(p):
    rates = [p.analytic_rate(m * p.threshold_distance) for m in (1.1, 2, 5, 20)]
    assert rates == sorted(rates)


# -- subthreshold ----------------------------------------------------------


def test_steady_state_is_rest_plus_g(p):
    g = 0.5 * p.threshold_distance
    net = LIFNetwork(1, params=p, device=DEV)
    for _ in range(int(10 * p.tau_mem / p.dt)):
        net.step(g_ext=g)
    assert float(net.v[0]) == pytest.approx(p.v_rest + g, rel=1e-3)


def test_no_spike_below_rheobase(p):
    net = LIFNetwork(1, params=p, device=DEV)
    res = net.run(1.0, g_ext=0.99 * p.threshold_distance)
    assert float(res["counts"][0]) == 0


# -- synaptic transmission -------------------------------------------------


def _psp_trace(p, syn_count, sign=1.0, seconds=0.1):
    net = LIFNetwork(
        2,
        pre_idx=torch.tensor([0]), post_idx=torch.tensor([1]),
        signed_syn=torch.tensor([sign * syn_count]),
        params=p, device=DEV,
    )
    net.step(forced=torch.tensor([True, False], device=DEV))
    for _ in range(p.delay_steps - 1):      # spike is still in flight
        net.step()
    out = []
    for _ in range(int(seconds / p.dt)):
        net.step()
        out.append(float(net.v[1]) - p.v_rest)
    return out


def test_psp_matches_closed_form(p):
    """The two-neuron test: one spike in, exact PSP out."""
    trace = _psp_trace(p, syn_count=10)
    w = p.w_syn * 10
    peak = max(trace)
    assert peak == pytest.approx(w * p.peak_psp_fraction(), rel=0.01)
    # and the whole time course, not just the peak
    for i, meas in enumerate(trace):
        pred = p.analytic_psp(w, (i + 1) * p.dt)
        if pred > 0.05 * peak:
            assert meas == pytest.approx(pred, rel=0.02)


def test_inhibitory_synapse_hyperpolarises(p):
    """A negative weight must push v DOWN. If this fails the sign convention
    is inverted somewhere between graph.py and lif.py."""
    assert min(_psp_trace(p, 10, sign=-1.0)) < 0
    assert max(_psp_trace(p, 10, sign=+1.0)) > 0


def test_psp_scales_linearly_with_synapse_count(p):
    a = max(_psp_trace(p, 10))
    b = max(_psp_trace(p, 20))
    assert b == pytest.approx(2 * a, rel=1e-3)


def test_spikes_arrive_after_the_conduction_delay(p):
    """Table 1 gives T_dly = 1.8 ms -- a parameter the spec omits entirely.
    A spike must be invisible to its target until that delay has elapsed."""
    net = LIFNetwork(
        2,
        pre_idx=torch.tensor([0]), post_idx=torch.tensor([1]),
        signed_syn=torch.tensor([10.0]),
        params=p, device=DEV,
    )
    net.step(forced=torch.tensor([True, False], device=DEV))
    for i in range(p.delay_steps - 1):
        net.step()
        assert float(net.v[1]) == pytest.approx(p.v_rest, rel=1e-6), (
            f"arrived early, at step {i + 1} of {p.delay_steps}"
        )
    net.step()
    assert float(net.v[1]) > p.v_rest


def test_delay_is_at_least_one_step(p):
    """A spike can never affect its target within the same timestep, however
    small T_dly is."""
    from dataclasses import replace
    assert replace(p, t_dly=0.0).delay_steps == 1
    assert replace(p, t_dly=1e-9).delay_steps == 1


def test_delay_quantisation_is_visible(p):
    """1.8 ms / 0.5 ms = 3.6 -> 4 steps = 2.0 ms. Must not be silent."""
    assert p.t_dly_effective == p.delay_steps * p.dt
    assert p.t_dly_effective >= p.t_dly


# -- firing ----------------------------------------------------------------


@pytest.mark.parametrize("mult", [1.2, 2.0, 5.0])
def test_fi_curve_matches_closed_form(p, mult):
    g = mult * p.threshold_distance
    net = LIFNetwork(1, params=p, device=DEV)
    res = net.run(2.0, g_ext=g)
    # tolerance is set by spike-time quantisation: the ISI is a whole number
    # of dt, so relative error scales as dt/ISI.
    assert float(res["rates_hz"][0]) == pytest.approx(p.analytic_rate(g), rel=0.05)


def test_refractory_ceiling_is_never_exceeded(p):
    net = LIFNetwork(1, params=p, device=DEV)
    res = net.run(1.0, g_ext=1e6 * p.threshold_distance)
    assert float(res["rates_hz"][0]) <= p.max_rate * 1.001


def test_forced_spikes_respect_refractoriness(p):
    net = LIFNetwork(1, params=p, device=DEV, seed=0)
    rate = poisson_rate_vector(1, [0], 5000.0, device=DEV)
    res = net.run(1.0, forced_rate=rate)
    assert float(res["rates_hz"][0]) <= p.max_rate * 1.001


def test_refractory_quantisation_is_reported(p):
    """dt must divide t_refrac or the effective period is wrong. 2.2/0.5 = 4.4
    quantises to 2.0 ms. We do not fix it, but it must not be silent."""
    assert p.t_refrac_effective == p.refrac_steps * p.dt
    if abs(p.t_refrac / p.dt - round(p.t_refrac / p.dt)) > 1e-9:
        assert p.refrac_quantisation_error > 0


def test_poisson_input_is_dead_time_limited(p):
    """Poisson drive at rate r is depressed by the refractory period to
    r / (1 + r*T) -- the standard dead-time correction. Requesting 100 Hz
    delivers ~83 Hz, so the M2 calibration must stimulate against the
    DELIVERED rate, not the requested one."""
    r = 100.0
    net = LIFNetwork(1, params=p, device=DEV, seed=7)
    res = net.run(5.0, forced_rate=poisson_rate_vector(1, [0], r, device=DEV))
    expected = r / (1.0 + r * p.t_refrac_effective)
    assert float(res["rates_hz"][0]) == pytest.approx(expected, rel=0.10)
    assert expected < r


# -- state hygiene ---------------------------------------------------------


def test_reset_clears_everything(p):
    net = LIFNetwork(1, params=p, device=DEV)
    net.run(0.2, g_ext=5 * p.threshold_distance)
    net.reset()
    assert float(net.v[0]) == pytest.approx(p.v_rest, rel=1e-6)
    assert float(net.g[0]) == 0.0
    assert int(net.refrac[0]) == 0
    assert not bool(net.spiked[0])
    assert net.t == 0.0


def test_empty_graph_is_allowed(p):
    """M1.5 runs on isolated neurons, so a network with no edges must work."""
    net = LIFNetwork(3, params=p, device=DEV)
    res = net.run(0.1, g_ext=2 * p.threshold_distance)
    assert all(c > 0 for c in res["counts"].tolist())
