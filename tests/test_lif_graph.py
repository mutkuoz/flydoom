"""The CUDA-graph step must be the eager step, to the last bit.

flydoom/lif_graph.py rewrites how state is stored so a graph can replay it:
state is updated in place on persistent buffers and the delay ring is indexed
with a GPU tensor rather than a Python int. Those are representation changes
and must not be arithmetic changes. Its docstring promised a verify() that
checks exactly that; this is it.

Skipped without CUDA, since a graph cannot be captured on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA graphs need a GPU")


def _small_net(seed=0):
    """A network small enough to build in a test, with the real machinery:
    conductance synapses, mixed delays, graded units."""
    from flydoom.lif import LIFNetwork, LIFParams

    rng = np.random.default_rng(seed)
    n, e = 400, 6000
    pre = rng.integers(0, n, e).astype(np.int32)
    post = rng.integers(0, n, e).astype(np.int32)
    syn = (rng.integers(1, 6, e) * rng.choice([1, -1], e)).astype(np.float32)
    delay = rng.choice([4, 160], e).astype(np.int64)      # fast and slow taps
    graded = torch.zeros(n, dtype=torch.bool, device="cuda")
    graded[: n // 4] = True
    net = LIFNetwork(n, torch.as_tensor(pre, device="cuda"),
                     torch.as_tensor(post, device="cuda"),
                     torch.as_tensor(syn, device="cuda"),
                     LIFParams(), "cuda", seed,
                     edge_delay=delay, graded=graded.cpu().numpy())
    return net


def test_graph_step_is_bit_identical():
    """Two identically built networks, one stepped eagerly and one by graph
    replay, must stay identical.

    Comparing two networks rather than saving and restoring one, because the
    eager step REBINDS net.v each call while a captured graph writes to the
    buffer it captured: restoring by rebinding would silently point the two
    paths at different memory, which is the failure this module exists to
    avoid.
    """
    from flydoom.lif_graph import GraphedLIF

    a, b = _small_net(), _small_net()
    assert torch.equal(a.v, b.v) and torch.equal(a.delay_buf, b.delay_buf)

    rng = np.random.default_rng(1)
    drives = []
    for _ in range(12):
        d = torch.full((a.n,), -1.0, device="cuda")
        idx = torch.as_tensor(rng.choice(a.n, 40, replace=False), device="cuda")
        d[idx] = torch.as_tensor(rng.random(40).astype("float32"), device="cuda")
        drives.append(d)

    g = GraphedLIF(b)
    g.capture()
    # capture() runs `warmup` steps and then the captured one, so the graphed
    # net is four steps ahead; give the eager net the same four.
    for _ in range(4):
        a.step()

    for i, drive in enumerate(drives):
        a.step(out_set=drive)
        g.replay(out_set=drive)
        assert torch.equal(b.v, a.v), (
            f"v differs at step {i}: max |diff| "
            f"{(b.v - a.v).abs().max().item():.3e}")
        assert torch.equal(b.out, a.out), f"out differs at step {i}"
        assert torch.equal(b.spiked, a.spiked), f"spiked differs at step {i}"
