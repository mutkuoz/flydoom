"""Leaky integrate-and-fire simulator.

Implements the model form verified against Shiu et al. (bioRxiv 2023.05.02.539144,
Methods):

    tau_mem * dv/dt = (V_rest - v) + g       on presynaptic spike:  g += w
    tau_syn * dg/dt = -g                     w = sign * syn_count * W_SYN

`g` is a voltage offset in volts, not a conductance. It moves the potential the
neuron decays *toward*, and itself decays to zero with tau_syn. On firing, v
resets to V_rest and is frozen for the refractory period.

Note what this means for W_SYN: it is NOT the peak depolarisation caused by one
synapse. Because tau_syn < tau_mem the membrane never reaches V_rest + g, and
the actual peak is `peak_psp_fraction()` of w — about 16% at the nominal taus.
The paper's prose ("how much the downstream membrane potential changes as a
result of a single synapse") is loose about this. It matters when reasoning
about gain by hand.

Everything is vectorised over all neurons; there is no Python loop over cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import config
from .graph import ConnectomeGraph


@dataclass
class LIFParams:
    v_rest: float = config.V_REST
    v_thresh: float = config.V_THRESH
    v_reset: float = config.V_RESET
    tau_mem: float = config.TAU_MEM
    tau_syn: float = config.TAU_SYN
    t_refrac: float = config.T_REFRAC
    w_syn: float = config.W_SYN
    t_dly: float = config.T_DLY
    dt: float = config.DT
    noise_std: float = config.NOISE_STD

    @property
    def threshold_distance(self) -> float:
        """How far above rest the membrane must climb to fire."""
        return self.v_thresh - self.v_rest

    @property
    def refrac_steps(self) -> int:
        return int(round(self.t_refrac / self.dt))

    @property
    def delay_steps(self) -> int:
        """Conduction delay in timesteps. At least 1 -- a spike cannot affect
        its target within the same step."""
        return max(1, int(round(self.t_dly / self.dt)))

    @property
    def t_dly_effective(self) -> float:
        return self.delay_steps * self.dt

    @property
    def t_refrac_effective(self) -> float:
        """The refractory period the simulation ACTUALLY uses.

        dt must divide t_refrac for these to agree, and 2.2 ms / 0.5 ms = 4.4
        does not: it quantises to 4 steps = 2.0 ms, 9% short, which raises the
        maximum firing rate from 370 Hz to 400 Hz. Harmless at the rates most
        neurons run at; it matters if anything saturates.
        """
        return self.refrac_steps * self.dt

    @property
    def refrac_quantisation_error(self) -> float:
        if not self.t_refrac:
            return 0.0
        return abs(self.t_refrac_effective - self.t_refrac) / self.t_refrac

    @property
    def max_rate(self) -> float:
        """Hard ceiling: one spike per refractory period plus the step it fires on."""
        return 1.0 / (self.t_refrac_effective + self.dt)

    # -- closed-form predictions, used by M1.5 to validate the integrator ----

    def analytic_rate(self, g_const: float) -> float:
        """Steady firing rate for a constant voltage offset g, in Hz.

        With v decaying toward V_rest + g from V_reset == V_rest, the interval
        between spikes is

            T = tau_mem * ln( g / (g - dV) ),      dV = V_thresh - V_rest

        and the rate is 1 / (T + t_refrac). Returns 0 below rheobase.
        """
        dv = self.threshold_distance
        if g_const <= dv:
            return 0.0
        t_spike = self.tau_mem * math.log(g_const / (g_const - dv))
        # the EFFECTIVE refractory period, since that is what the sim enforces
        return 1.0 / (t_spike + self.t_refrac_effective)

    def rheobase(self) -> float:
        """Smallest constant g that fires at all."""
        return self.threshold_distance

    def analytic_psp(self, w: float, t: float) -> float:
        """Membrane deflection at time t after a single presynaptic spike.

            u(t) = w * tau_syn/(tau_syn - tau_mem) * (e^-t/tau_syn - e^-t/tau_mem)
        """
        ts, tm = self.tau_syn, self.tau_mem
        return w * ts / (ts - tm) * (math.exp(-t / ts) - math.exp(-t / tm))

    def psp_peak_time(self) -> float:
        ts, tm = self.tau_syn, self.tau_mem
        return ts * tm / (tm - ts) * math.log(tm / ts)

    def peak_psp_fraction(self) -> float:
        """Peak PSP as a fraction of w. ~0.16 at the nominal taus.

        This is the number to reason with when asking "is the gain sane?",
        not W_SYN itself.
        """
        return self.analytic_psp(1.0, self.psp_peak_time())


class LIFNetwork:
    """State and dynamics for one brain."""

    def __init__(
        self,
        n_neurons: int,
        pre_idx: torch.Tensor | None = None,
        post_idx: torch.Tensor | None = None,
        signed_syn: torch.Tensor | None = None,
        params: LIFParams | None = None,
        device: str = "cuda",
        seed: int | None = None,
    ) -> None:
        self.n = n_neurons
        self.p = params or LIFParams()
        self.device = device
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)

        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        empty_f = torch.zeros(0, dtype=torch.float32, device=device)
        self.pre = pre_idx.long().to(device) if pre_idx is not None else empty_i
        self.post = post_idx.long().to(device) if post_idx is not None else empty_i
        self.w = signed_syn.to(device) if signed_syn is not None else empty_f

        # Precomputed step coefficients for EXACT integration of the linear
        # system over one dt. Brian2 -- which the paper used -- integrates
        # linear ODEs exactly (method='exact'), so forward Euler would not match
        # the reference implementation. Measured on the single-spike PSP, plain
        # forward Euler was 6% low at dt=0.5 ms; this is exact to machine
        # precision.
        #
        #   u(dt) = u0*A + g0*B + g_ext*(1-A),   u = v - V_rest
        #   A = e^(-dt/tau_mem)
        #   B = tau_syn/(tau_syn-tau_mem) * (e^(-dt/tau_syn) - e^(-dt/tau_mem))
        self.syn_decay = math.exp(-self.p.dt / self.p.tau_syn)
        self.A = math.exp(-self.p.dt / self.p.tau_mem)
        ts, tm = self.p.tau_syn, self.p.tau_mem
        self.B = ts / (ts - tm) * (math.exp(-self.p.dt / ts) - self.A)
        self.alpha = self.p.dt / self.p.tau_mem  # kept for reference only

        self.reset()

    @classmethod
    def from_graph(
        cls,
        graph: ConnectomeGraph,
        params: LIFParams | None = None,
        device: str = "cuda",
        seed: int | None = None,
    ) -> LIFNetwork:
        pre, post, w = graph.to_torch(device)
        return cls(graph.n_neurons, pre, post, w, params, device, seed)

    # -- state -----------------------------------------------------------

    def reset(self) -> None:
        d, n = self.device, self.n
        self.v = torch.full((n,), self.p.v_rest, dtype=torch.float32, device=d)
        self.g = torch.zeros(n, dtype=torch.float32, device=d)
        self.refrac = torch.zeros(n, dtype=torch.int16, device=d)
        self.spiked = torch.zeros(n, dtype=torch.bool, device=d)
        # Conduction delay line: a ring buffer of the last `delay_steps` spike
        # vectors. Reading the oldest entry delivers spikes T_dly in the past.
        # With delay_steps == 1 this reduces to "use the previous step".
        self.delay_buf = torch.zeros(
            (self.p.delay_steps, n), dtype=torch.bool, device=d
        )
        self.delay_ptr = 0
        self.t = 0.0

    # -- one timestep ----------------------------------------------------

    def step(
        self,
        g_ext: torch.Tensor | float | None = None,
        forced: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance dt and return the boolean spike vector for this step.

        g_ext   constant voltage offset added to the drive but NOT decayed --
                a current clamp, used for f-I characterisation and tonic drive.
        forced  neurons made to spike this step regardless of their membrane,
                used to drive sensory populations. Respects refractoriness, so
                a forced rate above 1/t_refrac cannot be achieved.
        """
        p = self.p

        # 1. deliver the previous step's spikes into g, BEFORE advancing, so
        #    they act over the whole interval.
        #    Spikes arrive T_dly late, via the delay ring buffer. The paper
        #    gives T_dly = 1.8 ms (Table 1); a uniform delay on every
        #    connection is an approximation -- real fly delays vary with axon
        #    length -- and uniformity makes synchronous oscillation more likely.
        arriving = self.delay_buf[self.delay_ptr]
        if self.pre.numel():
            self.g.index_add_(
                0, self.post, p.w_syn * self.w * arriving[self.pre].float()
            )

        # 2. advance the membrane exactly over dt, given g decaying across it
        u = (self.v - p.v_rest) * self.A + self.g * self.B
        if g_ext is not None:
            u = u + g_ext * (1.0 - self.A)
        self.v = u + p.v_rest

        # 3. now decay g to its value at the end of the interval
        self.g.mul_(self.syn_decay)

        if p.noise_std:
            self.v.add_(
                torch.randn(self.n, generator=self.gen, device=self.device)
                * p.noise_std
            )

        # 4. refractory neurons are frozen at reset, so discard their update
        in_refrac = self.refrac > 0
        self.v = torch.where(in_refrac, torch.full_like(self.v, p.v_reset), self.v)
        self.refrac = torch.clamp(self.refrac - 1, min=0)

        # 5. threshold
        spiked = self.v > p.v_thresh
        if forced is not None:
            spiked = spiked | (forced & ~in_refrac)

        self.v = torch.where(spiked, torch.full_like(self.v, p.v_reset), self.v)
        self.refrac = torch.where(
            spiked,
            torch.full_like(self.refrac, p.refrac_steps),
            self.refrac,
        )

        self.spiked = spiked
        # push this step's spikes into the delay line, overwriting the slot we
        # just consumed, and advance the ring
        self.delay_buf[self.delay_ptr] = spiked
        self.delay_ptr = (self.delay_ptr + 1) % p.delay_steps
        self.t += p.dt
        return spiked

    # -- running ---------------------------------------------------------

    def run(
        self,
        duration_s: float,
        g_ext: torch.Tensor | float | None = None,
        forced_rate: torch.Tensor | None = None,
        record: torch.Tensor | None = None,
    ) -> dict:
        """Run for duration_s and return spike counts.

        forced_rate  per-neuron Poisson rate in Hz (0 for most neurons). This
                     is how the paper stimulates: Poisson input, not a constant
                     current.
        record       indices to return a full spike raster for. None = counts
                     only, which is what almost everything needs.
        """
        n_steps = int(round(duration_s / self.p.dt))
        counts = torch.zeros(self.n, dtype=torch.int32, device=self.device)
        raster = [] if record is not None else None

        p_fire = None
        if forced_rate is not None:
            p_fire = (forced_rate * self.p.dt).clamp(0, 1).to(self.device)

        for _ in range(n_steps):
            forced = None
            if p_fire is not None:
                forced = (
                    torch.rand(self.n, generator=self.gen, device=self.device)
                    < p_fire
                )
            s = self.step(g_ext=g_ext, forced=forced)
            counts += s
            if raster is not None:
                raster.append(s[record].clone())

        out = {
            "counts": counts,
            "duration_s": duration_s,
            "rates_hz": counts.float() / duration_s,
        }
        if raster is not None:
            out["raster"] = torch.stack(raster)
        return out

    # -- convenience -----------------------------------------------------

    def rate_of(self, result: dict, idx) -> float:
        """Mean firing rate in Hz across a population."""
        i = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=self.device)
        if i.numel() == 0:
            return 0.0
        return float(result["rates_hz"][i].mean())


def poisson_rate_vector(
    n_neurons: int, idx, rate_hz: float, device: str = "cuda"
) -> torch.Tensor:
    """A per-neuron rate vector that is `rate_hz` on idx and zero elsewhere."""
    r = torch.zeros(n_neurons, dtype=torch.float32, device=device)
    i = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device)
    r[i] = rate_hz
    return r
