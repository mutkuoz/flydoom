"""Voltage-dependent excitation must be supralinear, and must stay put.

config.NMDA_FRAC multiplies the excitatory conductance by a sigmoid of the
compartment's own voltage, which is the preferred-direction half of a
correlator: two inputs arriving together on one branch should do more than the
same two arriving apart. These check that it does, that it is off by default,
and that being regenerative has not made it run away.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _net(frac=0.0, n=4):
    from flydoom.lif import LIFNetwork, LIFParams
    pre = np.array([0, 1], np.int32)
    post = np.array([2, 2], np.int32)     # both land on the same node
    syn = np.array([4.0, 4.0], np.float32)
    p = LIFParams(nmda_frac=frac, noise_std=0.0)
    return LIFNetwork(n, torch.as_tensor(pre), torch.as_tensor(post),
                      torch.as_tensor(syn), p, "cpu", 0)


def _drive(net, spikes, steps=60):
    """`spikes` maps step -> which presynaptic cells fire. Returns the peak
    depolarisation of the target above rest."""
    peak = -np.inf
    for t in range(steps):
        out = torch.zeros(net.n)
        for i in spikes.get(t, ()):
            out[i] = 1.0
        net.step(out_set=torch.where(out > 0, out, torch.full_like(out, -1.0)))
        peak = max(peak, float(net.v[2]) - net.p.v_rest)
    return peak


def test_off_by_default():
    from flydoom.lif import LIFParams
    assert LIFParams().nmda_frac == 0.0


def test_coincident_input_is_supralinear():
    """Together minus apart, with and without the nonlinearity."""
    together = {5: (0, 1)}
    apart = {5: (0,), 35: (1,)}
    lin = _net(0.0)
    gain = _net(1.0)
    lin_gap = _drive(lin, together) - _drive(_net(0.0), apart)
    nl_gap = _drive(gain, together) - _drive(_net(1.0), apart)
    assert nl_gap > lin_gap > 0, (lin_gap, nl_gap)


def test_amplifies_but_does_not_run_away():
    """A regenerative term evaluated at the entering voltage must still settle:
    with no input the cell stays at rest, and with input it stays below the
    excitatory reversal."""
    net = _net(2.0)
    for _ in range(200):
        net.step(out_set=torch.full((net.n,), -1.0))
    assert abs(float(net.v[2]) - net.p.v_rest) < 1e-7   # float32 rest
    peak = _drive(_net(2.0), {5: (0, 1), 6: (0, 1), 7: (0, 1)}, steps=120)
    assert 0.0 < peak < abs(net.p.e_exc - net.p.v_rest)
