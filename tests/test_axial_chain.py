"""A two-node chain must be the pair model, and a chain must conduct.

flydoom/compartments_chain.py orders a cell's inputs along a cable, which is
the arrangement that makes a dendrite a spatial filter rather than a sum. It
needs more than one neighbour per node, which `axial_partner` cannot express,
so lif.py gained an edge list. These check the generalisation against the
implementation it generalises.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _net(**kw):
    from flydoom.lif import LIFNetwork, LIFParams
    n = 6
    pre = np.array([0, 1], np.int32)
    post = np.array([2, 3], np.int32)
    syn = np.array([5.0, -5.0], np.float32)
    return LIFNetwork(n, torch.as_tensor(pre), torch.as_tensor(post),
                      torch.as_tensor(syn), LIFParams(), "cpu", 0, **kw)


def test_two_node_chain_equals_the_pair_model():
    partner = np.arange(6)
    partner[0], partner[1] = 1, 0            # 0 <-> 1 coupled, rest alone
    g = np.zeros(6, np.float32)
    g[0] = g[1] = 0.7
    pair = _net(axial_partner=partner, g_axial=g)
    chain = _net(axial_edges=np.array([[0], [1]]), axial_edge_g=0.7)

    drive = torch.full((6,), -1.0)
    drive[0] = 0.9
    for _ in range(40):
        pair.step(out_set=drive)
        chain.step(out_set=drive)
    assert torch.allclose(pair.v, chain.v, atol=1e-6), (pair.v, chain.v)


def test_a_chain_conducts_along_its_length():
    """Excite one end; the voltage must fall along the cable and not leave it.

    Driven through a synapse, because out_set overrides a cell's OUTPUT and
    would leave its membrane untouched.
    """
    chain = _net(axial_edges=np.array([[2, 3, 4], [3, 4, 5]]),
                 axial_edge_g=1.0)
    drive = torch.full((6,), -1.0)
    drive[0] = 0.9                            # neuron 0 excites node 2
    for _ in range(200):
        chain.step(out_set=drive)
    v = chain.v.numpy()
    assert v[2] > v[3] > v[4] > v[5], f"no gradient along the cable: {v}"
    # the uncoupled neuron 1 receives nothing and stays at rest
    assert abs(v[1] - chain.p.v_rest) < 1e-6, "an uncoupled node moved"
