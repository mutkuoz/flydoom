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
    synapse_model: str = config.SYNAPSE_MODEL
    e_exc: float = config.E_EXC
    e_inh: float = config.E_INH
    g_syn: float = config.G_SYN
    graded_max_rate: float = config.GRADED_MAX_RATE

    @property
    def conductance(self) -> bool:
        return self.synapse_model == "conductance"

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
        edge_delay: "np.ndarray | None" = None,
        tau_mem: "np.ndarray | None" = None,
        graded: "np.ndarray | None" = None,
    ) -> None:
        self.n = n_neurons
        self.p = params or LIFParams()
        self.device = device
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)

        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        empty_f = torch.zeros(0, dtype=torch.float32, device=device)
        pre_np = pre_idx.long().to(device) if pre_idx is not None else empty_i
        post_np = post_idx.long().to(device) if post_idx is not None else empty_i
        w_np = signed_syn.to(device) if signed_syn is not None else empty_f

        # ---- delay groups -------------------------------------------------
        # Edges are sorted by conduction delay so each distinct delay is one
        # contiguous slice. The step then does one scatter-add per GROUP rather
        # than per edge, reading a different tap of the ring buffer for each.
        # With a uniform delay this is a single group and costs nothing.
        import numpy as _np
        E = int(pre_np.numel())
        if E == 0:
            self.pre, self.post, self.w = pre_np, post_np, w_np
            self.delay_groups = []
            self.buf_len = max(1, self.p.delay_steps)
        else:
            ed = (_np.full(E, self.p.delay_steps, dtype=_np.int32)
                  if edge_delay is None
                  else _np.asarray(edge_delay, dtype=_np.int32))
            if ed.size != E:
                raise ValueError(f"edge_delay has {ed.size} entries for {E} edges")
            # Group by (delay, sign). In conductance mode excitatory and
            # inhibitory edges land in DIFFERENT accumulators, so they must be
            # separable; sorting once here keeps every group a contiguous slice
            # and costs no extra scatter-adds.
            sign_bit = (w_np < 0).to(torch.int8).cpu().numpy()
            order = _np.lexsort((sign_bit, ed))
            ed, sign_bit = ed[order], sign_bit[order]
            ot = torch.as_tensor(order, dtype=torch.long, device=device)
            self.pre, self.post = pre_np[ot], post_np[ot]
            self.w = w_np[ot]
            key = ed.astype(_np.int64) * 2 + sign_bit
            bounds = _np.flatnonzero(_np.diff(key)) + 1
            starts = _np.concatenate([[0], bounds])
            ends = _np.concatenate([bounds, [key.size]])
            self.delay_groups = [
                (int(ed[a]), bool(sign_bit[a]), int(a), int(b))
                for a, b in zip(starts, ends)
            ]
            self.buf_len = max(1, int(ed.max()))
        # magnitude only; the sign now lives in which accumulator it targets
        self.w_abs = self.w.abs()

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
        ts, dt_ = self.p.tau_syn, self.p.dt

        if tau_mem is None:
            tm = self.p.tau_mem
            self.A = math.exp(-dt_ / tm)
            self.B = ts / (ts - tm) * (math.exp(-dt_ / ts) - self.A)
            self.tau_mem_vec = None
        else:
            # PER-NEURON membrane time constant. Cell types differ in how fast
            # they integrate -- Mi9/Mi4 are slow and sustained, Mi1/Tm3 fast and
            # transient -- and that difference is what a motion correlator is
            # built out of. Shiu et al. use one global value; this is a
            # deliberate departure, see config.TAU_MEM_BY_TYPE.
            import numpy as _np
            tmv = _np.asarray(tau_mem, dtype=_np.float64).copy()
            if tmv.size != n_neurons:
                raise ValueError(
                    f"tau_mem has {tmv.size} entries for {n_neurons} neurons"
                )
            # B has a removable singularity at tau_mem == tau_syn; nudge off it
            # rather than emitting inf.
            close = _np.abs(tmv - ts) < 1e-9
            tmv[close] = ts * 1.001
            A = _np.exp(-dt_ / tmv)
            B = ts / (ts - tmv) * (_np.exp(-dt_ / ts) - A)
            self.A = torch.as_tensor(A, dtype=torch.float32, device=device)
            self.B = torch.as_tensor(B, dtype=torch.float32, device=device)
            self.tau_mem_vec = torch.as_tensor(tmv, dtype=torch.float32,
                                               device=device)
        # dt/tau_mem, scalar or per-neuron. The conductance branch needs this
        # explicitly -- it cannot reuse A, because its decay depends on the
        # instantaneous total conductance.
        self.dt_over_tau = (
            self.p.dt / self.p.tau_mem if self.tau_mem_vec is None
            else self.p.dt / self.tau_mem_vec
        )
        self.alpha = self.p.dt / self.p.tau_mem  # kept for reference only

        # Which neurons signal continuously instead of by spikes. See
        # config.GRADED_SUPER_CLASSES.
        if graded is None:
            self.graded = torch.zeros(n_neurons, dtype=torch.bool, device=device)
        else:
            import numpy as _np
            self.graded = torch.as_tensor(
                _np.asarray(graded, dtype=bool), device=device
            )
        self.any_graded = bool(self.graded.any())
        self.reset()

    @classmethod
    def from_graph(
        cls,
        graph: ConnectomeGraph,
        params: LIFParams | None = None,
        device: str = "cuda",
        seed: int | None = None,
        edge_delay: "np.ndarray | None" = None,
        tau_mem: "np.ndarray | None" = None,
        graded: "np.ndarray | None" = None,
    ) -> LIFNetwork:
        pre, post, w = graph.to_torch(device)
        return cls(graph.n_neurons, pre, post, w, params, device, seed,
                   edge_delay=edge_delay, tau_mem=tau_mem, graded=graded)

    @property
    def delay_summary(self) -> str:
        if not self.delay_groups:
            return "no edges"
        agg: dict[int, int] = {}
        for d, _is_inh, lo, hi in self.delay_groups:
            agg[d] = agg.get(d, 0) + (hi - lo)
        return "  ".join(f"{n:,}@{d * self.p.dt * 1e3:.1f}ms"
                         for d, n in sorted(agg.items()))

    @property
    def conductance_summary(self) -> str:
        p = self.p
        if not p.conductance:
            return "subtractive (Shiu et al.)"
        return (f"conductance  E_exc {p.e_exc * 1e3:.0f} mV  "
                f"E_inh {p.e_inh * 1e3:.0f} mV  G_syn {p.g_syn:.5f}")

    # -- state -----------------------------------------------------------

    def reset(self) -> None:
        d, n = self.device, self.n
        self.v = torch.full((n,), self.p.v_rest, dtype=torch.float32, device=d)
        self.g = torch.zeros(n, dtype=torch.float32, device=d)
        self.g_exc = torch.zeros(n, dtype=torch.float32, device=d)
        self.g_inh = torch.zeros(n, dtype=torch.float32, device=d)
        self.refrac = torch.zeros(n, dtype=torch.int16, device=d)
        self.spiked = torch.zeros(n, dtype=torch.bool, device=d)
        # Conduction delay ring buffer, `buf_len` steps deep. Reading tap d
        # delivers the spikes emitted d steps ago.
        # float, not bool: a graded neuron's output is a continuous value.
        # A spiking neuron contributes exactly 1.0 on the step it fires.
        self.delay_buf = torch.zeros((self.buf_len, n), dtype=torch.float32,
                                     device=d)
        self.out = torch.zeros(n, dtype=torch.float32, device=d)
        self.delay_ptr = 0
        self.t = 0.0

    # -- one timestep ----------------------------------------------------

    def step(
        self,
        g_ext: torch.Tensor | float | None = None,
        forced: torch.Tensor | None = None,
        out_set: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance dt and return the boolean spike vector for this step.

        out_set  per-neuron output override, or negative to leave free. This is
                 how a GRADED input population is driven: a photoreceptor does
                 not emit Poisson spikes, it releases transmitter in proportion
                 to light, so the retina sets its output directly.

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
        L = self.buf_len
        cond = p.conductance
        scale = p.g_syn if cond else p.w_syn
        for d, is_inh, lo, hi in self.delay_groups:
            arriving = self.delay_buf[(self.delay_ptr + L - d) % L]
            contrib = scale * self.w_abs[lo:hi] * arriving[self.pre[lo:hi]]
            if not cond:
                # subtractive: one signed accumulator, as in the paper
                self.g.index_add_(0, self.post[lo:hi],
                                  -contrib if is_inh else contrib)
            elif is_inh:
                self.g_inh.index_add_(0, self.post[lo:hi], contrib)
            else:
                self.g_exc.index_add_(0, self.post[lo:hi], contrib)

        if not cond:
            # 2a. exact update of the linear system (see A, B above)
            u = (self.v - p.v_rest) * self.A + self.g * self.B
            if g_ext is not None:
                u = u + g_ext * (1.0 - self.A)
            self.v = u + p.v_rest
            self.g.mul_(self.syn_decay)
        else:
            # 2b. conductance-based. Rearranging
            #     tau dv/dt = (V_rest-v) + g_e(E_e-v) + g_i(E_i-v) + I_ext
            # gives  tau dv/dt = -g_tot (v - v_inf)  with
            #     g_tot = 1 + g_e + g_i
            #     v_inf = (V_rest + g_e E_e + g_i E_i + I_ext) / g_tot
            # so the membrane relaxes toward v_inf with an EFFECTIVE time
            # constant tau/g_tot. Inhibition therefore divides rather than
            # subtracts, which is the whole point.
            g_tot = 1.0 + self.g_exc + self.g_inh
            num = p.v_rest + self.g_exc * p.e_exc + self.g_inh * p.e_inh
            if g_ext is not None:
                num = num + g_ext
            v_inf = num / g_tot
            # exponential Euler: exact for g held over the step, and g moves
            # only ~10% within dt at tau_syn=5 ms.
            decay = torch.exp(-self.dt_over_tau * g_tot)
            self.v = v_inf + (self.v - v_inf) * decay
            self.g_exc.mul_(self.syn_decay)
            self.g_inh.mul_(self.syn_decay)

        if p.noise_std:
            self.v.add_(
                torch.randn(self.n, generator=self.gen, device=self.device)
                * p.noise_std
            )

        # 4. refractory neurons are frozen at reset, so discard their update
        in_refrac = self.refrac > 0
        if self.any_graded:
            in_refrac = in_refrac & ~self.graded
        self.v = torch.where(in_refrac, torch.full_like(self.v, p.v_reset), self.v)
        self.refrac = torch.clamp(self.refrac - 1, min=0)

        # 5. threshold -- spiking neurons only
        spiked = self.v > p.v_thresh
        if forced is not None:
            spiked = spiked | (forced & ~in_refrac)
        if self.any_graded:
            spiked = spiked & ~self.graded

        self.v = torch.where(spiked, torch.full_like(self.v, p.v_reset), self.v)
        self.refrac = torch.where(
            spiked,
            torch.full_like(self.refrac, p.refrac_steps),
            self.refrac,
        )

        # 6. output. A spike contributes 1.0 for one step; a graded neuron
        # contributes a continuous value every step, scaled so that a graded
        # cell held at threshold delivers the same conductance per second as a
        # spiking cell firing at graded_max_rate.
        out = spiked.float()
        if self.any_graded:
            act = ((self.v - p.v_rest) / p.threshold_distance).clamp(0.0, 1.0)
            out = torch.where(
                self.graded, act * (p.graded_max_rate * p.dt), out
            )
        if out_set is not None:
            out = torch.where(out_set >= 0.0, out_set, out)
        self.out = out

        self.spiked = spiked
        # push this step's spikes into the delay line, overwriting the slot we
        # just consumed, and advance the ring
        self.delay_buf[self.delay_ptr] = self.out
        self.delay_ptr = (self.delay_ptr + 1) % self.buf_len
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
