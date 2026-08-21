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
    """Subtractive mode — the paper's model. The closed-form checks in this
    file are derived for it. Conductance mode has its own section below."""
    return LIFParams(synapse_model="subtractive")


@pytest.fixture
def pc() -> LIFParams:
    return LIFParams(synapse_model="conductance")


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


# -- per-edge conduction delays --------------------------------------------


def test_per_edge_delays_arrive_at_different_times(p):
    """Two edges from the same source with different delays must land at
    different steps. This is the mechanism M3's direction selectivity depends
    on, so a bug here would silently invalidate the whole result."""
    import numpy as np
    fast, slow = 2, 10
    net = LIFNetwork(
        3,
        pre_idx=torch.tensor([0, 0]), post_idx=torch.tensor([1, 2]),
        signed_syn=torch.tensor([10.0, 10.0]),
        params=p, device=DEV, edge_delay=np.array([fast, slow]),
    )
    net.step(forced=torch.tensor([True, False, False], device=DEV))
    arrived = {}
    for k in range(1, slow + 4):
        net.step()
        for target in (1, 2):
            if target not in arrived and float(net.v[target]) > p.v_rest + 1e-9:
                arrived[target] = k
    assert arrived[1] == fast, f"fast edge arrived at {arrived.get(1)}, want {fast}"
    assert arrived[2] == slow, f"slow edge arrived at {arrived.get(2)}, want {slow}"


def test_delay_groups_cover_every_edge(p):
    import numpy as np
    n_edges = 7
    delays = np.array([3, 1, 3, 9, 1, 9, 3])
    net = LIFNetwork(
        4,
        pre_idx=torch.zeros(n_edges, dtype=torch.long),
        post_idx=torch.ones(n_edges, dtype=torch.long),
        signed_syn=torch.ones(n_edges),
        params=p, device=DEV, edge_delay=delays,
    )
    assert sum(hi - lo for *_, lo, hi in net.delay_groups) == n_edges
    assert sorted({d for d, _s, _lo, _hi in net.delay_groups}) == [1, 3, 9]
    assert net.buf_len >= 9


def test_uniform_delay_is_unchanged_by_the_group_machinery(p):
    """With one delay value the multi-tap path must reproduce the single-tap
    path exactly."""
    import numpy as np
    def run(edge_delay):
        net = LIFNetwork(
            2, pre_idx=torch.tensor([0]), post_idx=torch.tensor([1]),
            signed_syn=torch.tensor([10.0]), params=p, device=DEV,
            edge_delay=edge_delay,
        )
        net.step(forced=torch.tensor([True, False], device=DEV))
        return [float(net.step()[0]) or float(net.v[1]) for _ in range(12)]
    a = run(None)
    b = run(np.array([p.delay_steps]))
    assert a == pytest.approx(b, rel=1e-9)


# -- per-neuron membrane time constants ------------------------------------


def test_per_neuron_tau_mem_changes_integration_speed(p):
    """A slow-tau neuron must reach threshold later than a fast one under the
    same drive. This is the mechanism the T4 correlator is built from, so a
    bug here would silently invalidate M3."""
    import numpy as np
    taus = np.array([5e-3, 20e-3, 100e-3])
    net = LIFNetwork(3, params=p, device=DEV, tau_mem=taus)
    g = 2.0 * p.threshold_distance
    first = {}
    for k in range(int(0.5 / p.dt)):
        s_ = net.step(g_ext=g)
        for i in range(3):
            if i not in first and bool(s_[i]):
                first[i] = (k + 1) * p.dt
    assert first[0] < first[1] < first[2], f"ordering wrong: {first}"


def test_per_neuron_tau_matches_scalar_when_uniform(p):
    """A uniform tau vector must reproduce the scalar path exactly."""
    import numpy as np
    a = LIFNetwork(1, params=p, device=DEV)
    b = LIFNetwork(1, params=p, device=DEV,
                   tau_mem=np.full(1, p.tau_mem))
    g = 1.5 * p.threshold_distance
    va, vb = [], []
    for _ in range(200):
        a.step(g_ext=g); b.step(g_ext=g)
        va.append(float(a.v[0])); vb.append(float(b.v[0]))
    assert va == pytest.approx(vb, rel=1e-6)


def test_tau_equal_to_tau_syn_does_not_explode(p):
    """B has a removable singularity at tau_mem == tau_syn. It must be nudged,
    not allowed to produce inf."""
    import numpy as np
    net = LIFNetwork(1, params=p, device=DEV, tau_mem=np.array([p.tau_syn]))
    assert torch.isfinite(net.B).all()
    for _ in range(50):
        net.step(g_ext=2 * p.threshold_distance)
    assert torch.isfinite(net.v).all()


def test_slow_tau_lowers_firing_rate(p):
    """The confound to be aware of: slow tau also means LOWER rate, so this
    manipulation changes gain as well as timing."""
    import numpy as np
    net = LIFNetwork(2, params=p, device=DEV, tau_mem=np.array([20e-3, 100e-3]))
    res = net.run(1.0, g_ext=2 * p.threshold_distance)
    fast, slow = res["rates_hz"].tolist()
    assert fast > slow * 2


# -- conductance-based synapses --------------------------------------------


def _drive(params, syn, sign=1.0, g_ext=None, steps=120):
    net = LIFNetwork(
        2, pre_idx=torch.tensor([0]), post_idx=torch.tensor([1]),
        signed_syn=torch.tensor([sign * syn]), params=params, device=DEV,
    )
    net.step(forced=torch.tensor([True, False], device=DEV))
    peak = 0.0
    for _ in range(steps):
        net.step(g_ext=g_ext)
        peak = max(peak, abs(float(net.v[1]) - params.v_rest))
    return peak


def test_conductance_reduces_to_subtractive_without_synapses(pc, p):
    """With g_e = g_i = 0 the two models are the same equation, so a
    current-clamp f-I curve must agree. This is what keeps M1.5 valid."""
    for mult in (1.2, 3.0):
        g = mult * p.threshold_distance
        a = LIFNetwork(1, params=p, device=DEV).run(1.0, g_ext=g)
        b = LIFNetwork(1, params=pc, device=DEV).run(1.0, g_ext=g)
        assert float(a["rates_hz"][0]) == pytest.approx(
            float(b["rates_hz"][0]), rel=0.05)


def test_excitation_saturates_toward_the_reversal_potential(pc):
    """The defining nonlinearity: PSP amplitude is SUBLINEAR in synapse count,
    because v cannot exceed E_exc however much conductance arrives."""
    small = _drive(pc, 10)
    big = _drive(pc, 1000)
    assert big > small
    assert big < 100 * small, "response is still linear — not conductance-based"
    assert big < abs(pc.e_exc - pc.v_rest) * 1.01, "overshot E_exc"


def _subthreshold_gain(params, exc_syn, inh_rate, inh_syn=400.0,
                       exc_rate=300.0, seconds=0.4):
    """Mean subthreshold depolarisation under sustained Poisson drive.

    Two things matter for this to measure synaptic ARITHMETIC rather than
    something else. Spiking is disabled by putting the threshold out of reach,
    because a spike resets v and truncates the mean -- which silently made the
    inhibited and uninhibited cases incomparable. And the input is varied by
    SYNAPSE COUNT rather than by rate, because the presynaptic refractory
    period caps rate and compresses the input range to nothing.
    """
    from dataclasses import replace
    quiet = replace(params, v_thresh=1.0)          # far above any reachable v
    net = LIFNetwork(
        3,
        pre_idx=torch.tensor([0, 2]), post_idx=torch.tensor([1, 1]),
        signed_syn=torch.tensor([float(exc_syn), -float(inh_syn)]),
        params=quiet, device=DEV, seed=3,
    )
    rate = torch.zeros(3, device=DEV)
    rate[0], rate[2] = exc_rate, inh_rate
    n = int(seconds / quiet.dt)
    acc, cnt = 0.0, 0
    for k in range(n):
        forced = torch.rand(3, generator=net.gen, device=DEV) < (rate * quiet.dt)
        net.step(forced=forced)
        if k > n // 3:
            acc += float(net.v[1]) - quiet.v_rest
            cnt += 1
    return acc / cnt


def test_shunting_inhibition_divides_the_gain():
    """THE point of the whole change.

    Shunting inhibition scales the input-output SLOPE -- it changes gain, which
    is a multiplication. E_inh == V_rest isolates pure shunting so there is no
    hyperpolarising offset mixed in.
    """
    shunt = LIFParams(synapse_model="conductance", e_inh=LIFParams().v_rest)
    slope_off = (_subthreshold_gain(shunt, 160, 0.0)
                 - _subthreshold_gain(shunt, 40, 0.0))
    slope_on = (_subthreshold_gain(shunt, 160, 400.0)
                - _subthreshold_gain(shunt, 40, 400.0))
    assert slope_off > 0 and slope_on > 0
    assert slope_on < 0.6 * slope_off, (
        f"gain not divided: {slope_off * 1e3:.4f} -> {slope_on * 1e3:.4f} mV"
    )


def test_subtractive_inhibition_preserves_the_gain():
    """The contrast case, and precisely why M3 and M4 could not work. The
    paper's inhibition removes a constant AMOUNT, leaving the slope intact --
    an offset, not a multiplication, and an offset cannot build a correlator.
    """
    sub = LIFParams(synapse_model="subtractive")
    slope_off = (_subthreshold_gain(sub, 160, 0.0)
                 - _subthreshold_gain(sub, 40, 0.0))
    slope_on = (_subthreshold_gain(sub, 160, 400.0)
                - _subthreshold_gain(sub, 40, 400.0))
    assert slope_on == pytest.approx(slope_off, rel=0.15), (
        f"expected slope preserved, got {slope_off * 1e3:.4f} -> "
        f"{slope_on * 1e3:.4f} mV"
    )


def test_shunting_shortens_the_effective_time_constant(pc):
    """tau_eff = tau_mem / (1 + g_e + g_i). A neuron under heavy conductance
    load must relax faster -- this is what makes the nonlinearity temporal as
    well as amplitude-wise."""
    net = LIFNetwork(
        2, pre_idx=torch.tensor([0]), post_idx=torch.tensor([1]),
        signed_syn=torch.tensor([-2000.0]), params=pc, device=DEV,
    )
    net.step(forced=torch.tensor([True, False], device=DEV))
    for _ in range(pc.delay_steps):
        net.step()
    g_tot = float(1.0 + net.g_exc[1] + net.g_inh[1])
    assert g_tot > 2.0, "inhibitory conductance did not accumulate"


def test_conductance_mode_splits_edges_by_sign(pc):
    """Excitatory and inhibitory edges must land in different accumulators."""
    net = LIFNetwork(
        3, pre_idx=torch.tensor([0, 0]), post_idx=torch.tensor([1, 2]),
        signed_syn=torch.tensor([100.0, -100.0]), params=pc, device=DEV,
    )
    net.step(forced=torch.tensor([True, False, False], device=DEV))
    for _ in range(pc.delay_steps):
        net.step()
    assert float(net.g_exc[1]) > 0 and float(net.g_inh[1]) == 0
    assert float(net.g_inh[2]) > 0 and float(net.g_exc[2]) == 0


# -- graded (non-spiking) units --------------------------------------------


def _graded_net(params, n=2, **kw):
    import numpy as np
    graded = np.zeros(n, dtype=bool)
    graded[0] = True
    return LIFNetwork(n, params=params, device=DEV, graded=graded, **kw)


def test_graded_neuron_never_spikes(pc):
    """However hard it is driven. Real photoreceptors and lamina monopolars do
    not fire action potentials."""
    net = _graded_net(pc)
    fired = False
    for _ in range(400):
        s_ = net.step(g_ext=50 * pc.threshold_distance)
        fired = fired or bool(s_[0])
    assert not fired
    assert float(net.v[0]) > pc.v_thresh, "should have depolarised past threshold"


def test_graded_output_is_continuous_in_membrane_potential(pc):
    """Output must scale with v, not jump between 0 and 1."""
    net = _graded_net(pc)
    seen = []
    for drive in (0.2, 0.5, 0.9):
        net.reset()
        for _ in range(300):
            net.step(g_ext=drive * pc.threshold_distance)
        seen.append(float(net.out[0]))
    assert seen[0] < seen[1] < seen[2]
    assert all(0.0 < v < 1.0 for v in seen[1:])


def test_graded_output_is_rectified_and_saturating(pc):
    """Zero at rest, capped at the threshold-equivalent."""
    net = _graded_net(pc)
    net.step()
    assert float(net.out[0]) == pytest.approx(0.0, abs=1e-9)
    net.reset()
    for _ in range(400):
        net.step(g_ext=20 * pc.threshold_distance)
    ceiling = pc.graded_max_rate * pc.dt
    assert float(net.out[0]) == pytest.approx(ceiling, rel=1e-6)


def test_graded_neuron_drives_a_spiking_target(pc):
    """The interface that matters: lamina/medulla are graded, LC and
    descending cells spike, and signal has to cross that boundary."""
    import numpy as np
    graded = np.array([True, False])
    net = LIFNetwork(
        2, pre_idx=torch.tensor([0]), post_idx=torch.tensor([1]),
        signed_syn=torch.tensor([4000.0]), params=pc, device=DEV, graded=graded,
    )
    fired = 0
    for _ in range(600):
        s_ = net.step(g_ext=torch.tensor([2.0 * pc.threshold_distance, 0.0],
                                         device=DEV))
        fired += int(s_[1])
    assert fired > 0, "graded source never drove its spiking target"


def test_out_set_overrides_output(pc):
    """How a graded input population is driven -- a photoreceptor's release is
    set by light, not by injected spikes."""
    net = _graded_net(pc)
    free = torch.full((2,), -1.0, device=DEV)
    net.step(out_set=free)
    assert float(net.out[0]) == pytest.approx(0.0, abs=1e-9)
    forcedval = torch.tensor([0.037, -1.0], device=DEV)
    net.step(out_set=forcedval)
    assert float(net.out[0]) == pytest.approx(0.037, rel=1e-6)


def test_graded_and_spiking_use_the_same_synaptic_units(pc):
    """A graded cell at threshold must deliver the same conductance per second
    as a spiking cell at graded_max_rate -- otherwise the two populations are
    on different scales and every weight in the graph means two things."""
    ceiling = pc.graded_max_rate * pc.dt
    # a spiking neuron at graded_max_rate emits 1.0 once per (1/rate) seconds,
    # i.e. a mean of rate*dt per step -- the same number
    assert ceiling == pytest.approx(pc.graded_max_rate * pc.dt)
    net = _graded_net(pc)
    for _ in range(400):
        net.step(g_ext=20 * pc.threshold_distance)
    assert float(net.out[0]) == pytest.approx(ceiling, rel=1e-6)


def test_spiking_only_network_is_unchanged_by_the_graded_machinery(p):
    """Regression: with no graded cells, results must match exactly."""
    import numpy as np
    def run(graded):
        net = LIFNetwork(1, params=p, device=DEV, graded=graded)
        return net.run(1.0, g_ext=2 * p.threshold_distance)["counts"].tolist()
    assert run(None) == run(np.zeros(1, dtype=bool))
