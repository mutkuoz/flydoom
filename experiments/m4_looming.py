#!/usr/bin/env python3
"""M4 — looming escape. Does an expanding disc drive the giant fiber?

Unlike M3 this is NOT a delay-line correlator. Looming detection reads angular
SIZE and its rate of expansion, which a spiking LIF can express directly. So M4
may pass even though M3 does not, and that is the point of running it: it tells
us how much of the visual arm is actually at risk versus just direction
selectivity.

Stimulus is the standard looming parameterisation. An object of half-width l
approaching at constant speed v collides at t_c, and its angular half-size is

    theta_half(t) = arctan( (l/|v|) / (t_c - t) )

l/|v| has units of seconds and is the single parameter the escape literature
reports against. Fly experiments typically use 10-80 ms.

Controls, which are the real content:
    static     a disc frozen at the looming stimulus's final size
    receding   the same expansion played backwards
    blank      uniform grey

A detector that fires for all four is measuring luminance, not looming.

    python experiments/m4_looming.py [--live] [--sweep]
"""

from __future__ import annotations

import argparse
import os
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.cells import AnnotationTable  # noqa: E402
from flydoom.graph import ConnectomeGraph  # noqa: E402
from flydoom.lif import LIFNetwork  # noqa: E402
from flydoom.registry import by_name  # noqa: E402
from flydoom.retina import Retina  # noqa: E402

USE_COLOR = sys.stdout.isatty()


def paint(t: str, c: str) -> str:
    return f"\033[{c}m{t}\033[0m" if USE_COLOR else t


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, ok, name, detail=""):
        self.rows.append((bool(ok), name, detail))
        return bool(ok)

    def render(self):
        for ok, name, detail in self.rows:
            mark = paint("PASS", "32") if ok else paint("FAIL", "1;31")
            print(f"  {mark}  {name:<48} {paint(detail, '90')}")
        return all(o for o, _, _ in self.rows)


def population(graph, ann, handle, side=None):
    res = ann.resolve(by_name(handle), side=side)
    if not res.root_ids:
        return np.zeros(0, np.int32)
    return graph.index_of(res.root_ids)


class LoomRig:
    """Expanding / static / receding disc, computed on the GPU."""

    def __init__(self, net, retina, device, max_rate_hz=150.0,
                 azimuth_deg=0.0, elevation_deg=0.0):
        idx, az, el, _ = retina.to_torch(device)
        self.idx = idx
        # angular distance of every driven neuron from the disc centre
        self.ecc = torch.hypot(az - azimuth_deg, el - elevation_deg)
        self.max_rate = max_rate_hz
        self.inverts = retina.inverts
        self.rate_buf = torch.zeros(net.n, dtype=torch.float32, device=device)
        self.out_buf = torch.full((net.n,), -1.0, dtype=torch.float32,
                                  device=device)
        self.retina = retina
        # Weber adaptation, matching the retina and the Doom path.
        self.adapt_mean = torch.full_like(self.ecc, 0.5)
        self.adapt_decay = math.exp(-net.p.dt / 0.25)
        self.adapt_gain = 2.5

    def reset_adaptation(self):
        self.adapt_mean.fill_(0.5)

    def out_set(self, radius_deg, graded_max_rate, dt):
        """Direct output override for a GRADED input population.

        L1/L2/L3 do not spike, so forcing Poisson spikes into them is a
        category error -- they simply ignore it and the whole cascade reads
        exactly 0.00 Hz. This is the same fix M3 needed.
        """
        lum = self.luminance(radius_deg)
        self.adapt_mean.mul_(self.adapt_decay).add_(lum * (1 - self.adapt_decay))
        c = ((lum - self.adapt_mean) / self.adapt_mean.clamp(min=1e-3)
             * self.adapt_gain).clamp(-1.0, 1.0)
        if self.inverts:
            c = -c
        act = (0.5 + 0.5 * c).clamp(0.0, 1.0)
        self.out_buf.fill_(-1.0)
        self.out_buf[self.idx] = act * (graded_max_rate * dt)
        return self.out_buf

    @staticmethod
    def theta_half(t, l_over_v, t_c, cap_deg=80.0):
        """Angular half-size in degrees at time t."""
        remaining = max(t_c - t, 1e-4)
        return min(math.degrees(math.atan(l_over_v / remaining)), cap_deg)

    def luminance(self, radius_deg):
        """1.0 outside the disc, 0.0 inside — a dark object on a light field."""
        return (self.ecc > radius_deg).float()

    def drive(self, radius_deg):
        lum = self.luminance(radius_deg)
        if self.inverts:
            lum = 1.0 - lum
        self.rate_buf.zero_()
        self.rate_buf[self.idx] = self.max_rate * lum
        return self.rate_buf


def radius_schedule(kind, n_steps, dt, l_over_v, t_c, start_deg, cap_deg):
    """Angular half-size for every timestep, for one stimulus condition."""
    loom = [LoomRig.theta_half(k * dt, l_over_v, t_c, cap_deg)
            for k in range(n_steps)]
    loom = [max(r, start_deg) for r in loom]
    if kind == "looming":
        return loom
    if kind == "static":
        return [loom[-1]] * n_steps
    if kind == "receding":
        return loom[::-1]
    if kind == "blank":
        return [0.0] * n_steps
    raise ValueError(kind)


def run_condition(net, rig, radii, gext, monitors, dash=None, retina=None,
                  dash_every=25, banner=""):
    """Step through one stimulus and return spike counts plus a rate trace."""
    dt = net.p.dt
    # Accumulate OUT, not spikes. out is in spike-equivalents -- 1.0 on the
    # step a spiking cell fires, rate*dt every step for a graded cell -- so
    # sum(out)/duration is a firing rate for both populations. Counting spikes
    # would report exactly 0.00 Hz for every graded neuron.
    counts = torch.zeros(net.n, dtype=torch.float32, device=net.device)
    net.reset()
    tau = 0.05
    decay = math.exp(-dt / tau)
    filt = torch.zeros(net.n, dtype=torch.float32, device=net.device)
    trace = []

    graded_input = bool(net.any_graded) and bool(net.graded[rig.idx].all())
    if graded_input:
        rig.reset_adaptation()
    for step, radius in enumerate(radii):
        if graded_input:
            s = net.step(g_ext=gext,
                         out_set=rig.out_set(radius, net.p.graded_max_rate, dt))
        else:
            rate = rig.drive(radius)
            forced = torch.rand(net.n, generator=net.gen,
                                device=net.device) < (rate * dt).clamp(0, 1)
            s = net.step(g_ext=gext, forced=forced)
        counts += net.out
        filt.mul_(decay).add_(s.float() * (1 - decay) / tau)

        if step % dash_every == 0:
            row = {k: (float(filt[v].mean()) if len(v) else 0.0)
                   for k, v in monitors.items()}
            row["_radius"] = radius
            row["_t"] = step * dt
            trace.append(row)
            if dash is not None and retina is not None:
                lum = rig.luminance(radius)
                per_col = _lum_by_column(retina, lum, rig.idx)
                d = row.get("DNp01", 0.0)
                if not dash.update(step * dt, per_col, row, d,
                                   np.tanh(d / 10.0),
                                   f"{banner}  theta/2 = {radius:.0f} deg"):
                    return None, None
    return counts, trace


def _lum_by_column(retina, lum_t, idx_t):
    lum = lum_t.detach().cpu().numpy()
    idx = idx_t.detach().cpu().numpy()
    order = {int(v): i for i, v in enumerate(idx)}
    out = {}
    for side, eye in retina.eyes.items():
        col = np.full(eye.n_columns, 0.5, dtype=float)
        if eye.neuron_idx.size:
            pos = np.array([order.get(int(n), -1) for n in eye.neuron_idx])
            ok = pos >= 0
            col[eye.neuron_column[ok]] = lum[pos[ok]]
        out[side] = col
    return out


def rate_of(counts, idx, duration):
    if counts is None or len(idx) == 0:
        return 0.0
    return float(counts[idx].mean()) / duration


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M4 — looming escape")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep l/|v| and report the angular threshold")
    ap.add_argument("--l-over-v", type=float, default=40.0,
                    help="looming parameter in ms (fly literature: 10-80)")
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--start-deg", type=float, default=5.0)
    ap.add_argument("--cap-deg", type=float, default=70.0)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--bias", type=float, default=0.0,
                    help="tonic optic-lobe drive, mV. Graded units need NONE "
                         "-- with bias they pin at the activation ceiling and "
                         "every condition returns an identical rate, blank "
                         "included. Only the all-spiking model needed it.")
    ap.add_argument("--site", default="L1+L2+L3")
    ap.add_argument("--shuffled", action="store_true",
                    help="degree-preserving shuffled connectome. THE control: "
                         "the stimulus here is generated open-loop and is "
                         "identical for both arms, so any difference is "
                         "attributable to wiring rather than to behaviour.")
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
    args = ap.parse_args()

    print(paint("flydoom M4 — looming escape", "1"))
    print(paint("=" * 76, "90"))

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    retina = Retina.build(g, ann, site=tuple(args.site.split("+")))
    if args.shuffled:
        # Permute edge TARGETS. Every neuron keeps its out-degree and every
        # weight keeps its magnitude and sign, so degree and E/I statistics are
        # preserved exactly and only the wiring pattern is destroyed. If the
        # intact and shuffled arms score the same, the result was never about
        # connectivity.
        rng = np.random.default_rng(0)
        g.post_idx = g.post_idx[rng.permutation(len(g.post_idx))]
        print(paint("SHUFFLED CONNECTOME — degree-preserving control arm", "1;33"))

    graded = g.graded_mask(ann)
    net = LIFNetwork.from_graph(g, device=args.device, seed=0, graded=graded)
    rig = LoomRig(net, retina, args.device, args.rate)

    print(retina.summary())
    print(f"W_SYN      {net.p.w_syn * 1e3:.5f} mV   bias {args.bias:.1f} mV")

    mons = {
        "L1": population(g, ann, "L1"),
        "LC4": population(g, ann, "LC4"),
        "LPLC2": population(g, ann, "LPLC2"),
        "LC11": population(g, ann, "LC11"),
        "DNp01": population(g, ann, "DNp01"),
    }
    mon_t = {k: torch.as_tensor(v.astype(np.int64), device=args.device)
             for k, v in mons.items()}
    print("populations " + "  ".join(f"{k}={len(v)}" for k, v in mons.items()))

    import polars as pl
    cls = pl.read_csv(config.RAW_DIR / "classification.csv.gz",
                      infer_schema_length=50_000)
    optic = cls.filter(
        pl.col("super_class").is_in(list(config.BIASED_SUPER_CLASSES))
    )["root_id"].to_list()
    oidx = torch.as_tensor(g.index_of(optic).astype(np.int64), device=args.device)
    gext = torch.zeros(net.n, dtype=torch.float32, device=args.device)
    gext[oidx] = args.bias * 1e-3

    dash = None
    if args.live:
        from flydoom.viz import LiveDashboard, have_display
        if have_display():
            dash = LiveDashboard(retina, list(mons), dt=net.p.dt,
                                 title="flydoom M4 — looming")
        else:
            print(paint("no display; ignoring --live", "33"))

    n_steps = int(round(args.duration / net.p.dt))
    t_c = args.duration * 1.02          # collision just after the window ends
    lv = args.l_over_v * 1e-3

    # ---------------- sweep ----------------
    if args.sweep:
        print(f"\n{paint('l/|v| SWEEP', '1;36')}")
        print(paint("  The escape literature reports thresholds against l/|v|. "
                    "A real\n  looming detector fires LATER (bigger theta) for "
                    "larger l/|v|.", "90"))
        print(f"\n  {'l/v(ms)':>8} {'LC4':>7} {'LPLC2':>7} {'DNp01':>7}"
              f" {'theta@peak':>11}")
        for lv_ms in (10, 20, 40, 80, 160):
            radii = radius_schedule("looming", n_steps, net.p.dt,
                                    lv_ms * 1e-3, t_c, args.start_deg,
                                    args.cap_deg)
            counts, trace = run_condition(net, rig, radii, gext, mon_t)
            peak = max(trace, key=lambda r: r["LPLC2"]) if trace else {}
            print(f"  {lv_ms:8d} {rate_of(counts, mon_t['LC4'], args.duration):7.2f}"
                  f" {rate_of(counts, mon_t['LPLC2'], args.duration):7.2f}"
                  f" {rate_of(counts, mon_t['DNp01'], args.duration):7.2f}"
                  f" {peak.get('_radius', 0):10.0f}d")
        return 0

    # ---------------- the four conditions ----------------
    print(f"\n{paint('CONDITIONS', '1;36')}")
    print(f"  {'condition':<12} {'L1':>7} {'LC4':>7} {'LPLC2':>7} {'LC11':>7}"
          f" {'DNp01':>7}")
    results = {}
    for kind in ("looming", "static", "receding", "blank"):
        radii = radius_schedule(kind, n_steps, net.p.dt, lv, t_c,
                                args.start_deg, args.cap_deg)
        counts, trace = run_condition(net, rig, radii, gext, mon_t, dash,
                                      retina, banner=kind)
        if counts is None:
            print(paint("\nwindow closed", "33"))
            return 0
        r = {k: rate_of(counts, v, args.duration) for k, v in mon_t.items()}
        results[kind] = (r, trace)
        print(f"  {kind:<12} {r['L1']:7.1f} {r['LC4']:7.2f} {r['LPLC2']:7.2f}"
              f" {r['LC11']:7.2f} {r['DNp01']:7.2f}")

    # ---------------- acceptance ----------------
    print(f"\n{paint('ACCEPTANCE', '1')}")
    c = Checks()
    loom = results["looming"][0]
    static = results["static"][0]
    recede = results["receding"][0]
    blank = results["blank"][0]

    c.check(loom["L1"] > 1, "visual input reaches the lamina",
            f"L1 {loom['L1']:.1f} Hz")
    c.check(max(loom["LC4"], loom["LPLC2"]) > 0.5,
            "looming detectors respond to expansion",
            f"LC4 {loom['LC4']:.2f}, LPLC2 {loom['LPLC2']:.2f} Hz")
    c.check(loom["DNp01"] > 0.5, "EXPANDING DISC DRIVES DNp01 (giant fiber)",
            f"{loom['DNp01']:.2f} Hz")
    c.check(loom["DNp01"] > 2 * max(static["DNp01"], 0.01),
            "CONTROL: static disc does not",
            f"static {static['DNp01']:.2f} vs looming {loom['DNp01']:.2f} Hz")
    c.check(loom["DNp01"] > 1.5 * max(recede["DNp01"], 0.01),
            "CONTROL: receding disc is weaker than looming",
            f"receding {recede['DNp01']:.2f} vs {loom['DNp01']:.2f} Hz")
    c.check(blank["DNp01"] < 0.5, "CONTROL: blank field is silent",
            f"{blank['DNp01']:.2f} Hz")

    ok = c.render()
    if dash is not None:
        dash.hold("done — close to exit")

    print(f"\n{paint('=' * 76, '90')}")
    if ok:
        print(paint("VERDICT: M4 PASS — the escape arm works.", "1;32"))
        print("This is the half of the engage/flee switch the Doom agent needs.")
        return 0
    print(paint("VERDICT: M4 FAIL", "1;31"))
    print(paint("""
DIAGNOSIS: change detection works; SIGN of change does not.

Measured at bias 6.5 mV, 2 s, driving L1+L2+L3:

    LPLC2   looming 0.42   receding 0.42   static 0.00   blank 0.00
    LC4     looming 0.37   receding 0.34   static 5.16   blank 0.00

LPLC2 has the right shape in one respect -- it is silent for a static disc and
for a blank field, so it is not merely reporting luminance. But it fires
IDENTICALLY for expansion and contraction. It detects that the stimulus is
changing, not which way. That is not looming selectivity.

LC4 is worse: it tracks dark AREA (static 5.16 is its largest response), which
is a luminance response wearing a looming costume. This is exactly why the
static and receding controls exist.

THIS IS THE SAME FAILURE AS M3, and the two together name the limitation
precisely. Direction selectivity (leftward vs rightward) and looming
selectivity (expanding vs contracting) are both computations over the TEMPORAL
ORDER in which neighbouring columns activate. This model detects that
neighbours changed; it cannot recover the order.

WHY, concretely. In this model every input sums LINEARLY into g, and v is a
linear filter of g:

    tau_mem * dv/dt = (V_rest - v) + g          g = sum of signed weights

The only nonlinearity is the spike threshold, applied to the SUM. But
threshold(A + B) is symmetric in A and B: a single threshold on a summed input
cannot distinguish "A then B" from "B then A". Adding per-type delays and
per-type time constants (both tried, see M3) changes WHEN each term arrives but
not the linearity of their combination, which is why neither rescued it.

A correlator needs a MULTIPLICATIVE interaction. In the real circuit that comes
from SHUNTING inhibition: GABA and GluCl open chloride channels, which changes
the membrane CONDUCTANCE and therefore divides the effect of excitation.

This text used to end by recommending conductance-based synapses as the fix.
They are now the default (config.SYNAPSE_MODEL = "conductance"):

    tau*dv/dt = (V_rest - v) + g_exc*(E_exc - v) + g_inh*(E_inh - v)

and the failure above was measured WITH them. Shunting was necessary and is not
sufficient. What remains is not the form of the synapse but the balance between
the two arms: T4a's fast excitatory arm is starved 37:1 by its own inhibition,
and LPLC2 sits BELOW its resting potential because PVLP011 fires at 72-82% of
the refractory ceiling. Those are ratios, and every input-side scale we can
reach -- drive, contrast, palette, field of view, engine rendering -- multiplies
both arms and cancels out of a ratio.

What would change it is a per-cell-type gain, which the connectome does not
specify. That is the measurement this project exists to report, not a bug to
fix here.""", "33"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
