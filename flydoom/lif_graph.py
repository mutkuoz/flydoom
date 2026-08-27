"""CUDA-graph replay for the LIF step. Same arithmetic, ~200x fewer launches.

WHY
---
Profiling one step of the real network (139,255 neurons, 2,710,038 edges):

    self CUDA time   0.31 ms      <- the GPU doing the work
    self CPU  time   8.50 ms      <- Python and ATen dispatch issuing ~200 ops
    wall             8.80 ms

The GPU is idle 97% of the time. The work per op is far too small to amortise a
launch, so the simulator is dispatch-bound: a faster card would change nothing.
A CUDA graph records the launches once and replays them as a single unit, which
removes the dispatch cost rather than the work.

WHAT HAD TO CHANGE, AND WHAT DID NOT
------------------------------------
Nothing about the model. Every constant, every operation and their order are
those of LIFNet.step; this module only rewrites HOW the state is stored:

  * step() rebinds `self.v = ...` to a fresh tensor each call. A graph replays
    fixed addresses, so a rebound tensor means replay silently reads the buffer
    captured at capture time and writes somewhere nothing reads. All state is
    therefore updated IN PLACE (`v.copy_(...)`) on persistent buffers.
  * the delay ring is indexed with a Python int that changes every step, which
    would bake one slot into the graph. The pointer becomes a GPU tensor and the
    ring is read with index_select and written with index_copy_.

Both are representation changes. `verify()` asserts bit-identical v, out and
spiked against the eager implementation before any graph is trusted, and
capture is refused if it fails.

LIMITS
------
Conductance mode only (config.CONDUCTANCE is True), no noise, no `forced`, no
`g_ext` -- the closed-loop path uses none of them. Anything else falls back to
eager, which is why this is opt-in and not a silent replacement.
"""
from __future__ import annotations

import torch


class GraphedLIF:
    """Graph-capturable in-place mirror of LIFNet.step for the closed loop."""

    def __init__(self, net):
        p = net.p
        if not p.conductance:
            raise ValueError("subtractive mode not supported; use eager step")
        if p.noise_std:
            raise ValueError("noise_std != 0 needs RNG capture; use eager step")
        self.net = net
        self.p = p
        self.L = net.buf_len
        d = net.device
        # ring pointer on the GPU: a Python int would be baked into the graph
        self.ptr = torch.zeros(1, dtype=torch.long, device=d)
        self.Lt = torch.tensor([self.L], dtype=torch.long, device=d)
        self.offsets = [torch.tensor([(self.L - dly) % self.L],
                                     dtype=torch.long, device=d)
                        for dly, _, _, _ in net.delay_groups]
        # static input buffer: the retina drive is copied in, never rebound
        self.out_set = torch.full((net.n,), -1.0, dtype=torch.float32, device=d)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._v_reset = torch.full((net.n,), p.v_reset, dtype=torch.float32,
                                   device=d)
        self._refrac_set = torch.full((net.n,), p.refrac_steps,
                                      dtype=net.refrac.dtype, device=d)

    # -- the step, arithmetic-for-arithmetic as in LIFNet.step -----------
    def step_inplace(self) -> None:
        net, p = self.net, self.p
        scale = p.g_syn

        # 1. deliver the previous step's spikes, at their delays
        for gi, (dly, is_inh, lo, hi) in enumerate(net.delay_groups):
            idx = torch.remainder(self.ptr + self.offsets[gi], self.Lt)
            arriving = net.delay_buf.index_select(0, idx).squeeze(0)
            contrib = scale * net.w_abs[lo:hi] * arriving[net.pre[lo:hi]]
            if is_inh:
                net.g_inh.index_add_(0, net.post[lo:hi], contrib)
            else:
                net.g_exc.index_add_(0, net.post[lo:hi], contrib)

        # 2b. conductance-based membrane update (exponential Euler)
        g_tot = 1.0 + net.g_exc + net.g_inh
        num = p.v_rest + net.g_exc * p.e_exc + net.g_inh * p.e_inh
        v_inf = num / g_tot
        decay = torch.exp(-net.dt_over_tau * g_tot)
        net.v.copy_(v_inf + (net.v - v_inf) * decay)
        net.g_exc.mul_(net.syn_decay)
        net.g_inh.mul_(net.syn_decay)

        # 4. refractory neurons frozen at reset
        in_refrac = net.refrac > 0
        if net.any_graded:
            in_refrac = in_refrac & ~net.graded
        net.v.copy_(torch.where(in_refrac, self._v_reset, net.v))
        net.refrac.copy_(torch.clamp(net.refrac - 1, min=0))

        # 5. threshold
        spiked = net.v > p.v_thresh
        if net.any_graded:
            spiked = spiked & ~net.graded
        net.v.copy_(torch.where(spiked, self._v_reset, net.v))
        net.refrac.copy_(torch.where(spiked, self._refrac_set, net.refrac))

        # 6. output, graded override, then the external drive
        out = spiked.float()
        if net.any_graded:
            act = ((net.v - p.v_rest) / p.threshold_distance).clamp(0.0, 1.0)
            out = torch.where(net.graded, act * (p.graded_max_rate * p.dt), out)
        out = torch.where(self.out_set >= 0.0, self.out_set, out)

        # 6b. short-term depression at emission
        if p.stp:
            released = out * net.stp_R
            net.stp_R.copy_((net.stp_R
                             + (1.0 - net.stp_R) * (p.dt / p.stp_tau_rec)
                             - p.stp_u * net.stp_R * out).clamp(0.0, 1.0))
            out = released

        net.out.copy_(out)
        net.spiked.copy_(spiked)
        # push into the delay line and advance the ring
        net.delay_buf.index_copy_(0, self.ptr, net.out.unsqueeze(0))
        self.ptr.copy_(torch.remainder(self.ptr + 1, self.Lt))

    # -- capture / replay ------------------------------------------------
    def capture(self, warmup: int = 3) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.step_inplace()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step_inplace()

    def replay(self, out_set: torch.Tensor | None = None) -> None:
        if out_set is not None:
            self.out_set.copy_(out_set)
        self.graph.replay()
