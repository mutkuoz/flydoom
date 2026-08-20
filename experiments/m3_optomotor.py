#!/usr/bin/env python3
"""M3 — optomotor response. The cheapest full-pipeline test.

Put any insect in a rotating striped drum and it turns to follow the stripes.
This is about as universal as insect behaviour gets, and it exercises the whole
chain — retina, optic lobe, descending neurons, steering readout — without
ViZDoom anywhere near it. Spec 8 is explicit: build this before touching Doom.

The test: a drifting grating should make the DNa02 pair go asymmetric, and the
asymmetry must REVERSE when the grating reverses. Sign, not magnitude.

    python experiments/m3_optomotor.py --live      # watch it
    python experiments/m3_optomotor.py --sweep     # temporal frequency tuning
"""

from __future__ import annotations

import argparse
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
TWO_PI = 2.0 * math.pi


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
            print(f"  {mark}  {name:<46} {paint(detail, '90')}")
        return all(o for o, _, _ in self.rows)


# --------------------------------------------------------------------------


def population_indices(graph, ann, handle, side=None):
    res = ann.resolve(by_name(handle), side=side)
    if not res.root_ids:
        return np.zeros(0, np.int32)
    return graph.index_of(res.root_ids)


class GratingRig:
    """Drives the network with a drifting grating, entirely on the GPU."""

    def __init__(self, net, retina, device, spatial_period_deg=30.0,
                 max_rate_hz=150.0):
        self.net, self.retina, self.device = net, retina, device
        idx, az, el, _ = retina.to_torch(device)
        self.idx = idx
        # precompute the spatial phase of every driven neuron, once
        self.base_phase = az / spatial_period_deg
        self.max_rate = max_rate_hz
        self.inverts = retina.inverts
        self.rate_buf = torch.zeros(net.n, dtype=torch.float32, device=device)
        self.sp = spatial_period_deg

    def luminance(self, t, tf_hz, direction):
        phase = self.base_phase - direction * tf_hz * t
        return 0.5 + 0.5 * torch.sign(torch.sin(TWO_PI * phase))

    def drive(self, t, tf_hz, direction):
        lum = self.luminance(t, tf_hz, direction)
        if self.inverts:
            lum = 1.0 - lum
        self.rate_buf.zero_()
        self.rate_buf[self.idx] = self.max_rate * lum
        return self.rate_buf


def run_grating(net, rig, duration, tf_hz, direction, monitors,
                dash=None, retina=None, dash_every=25, banner="", gext=None):
    """Step the network under a drifting grating; return spike counts."""
    dt = net.p.dt
    n_steps = int(round(duration / dt))
    counts = torch.zeros(net.n, dtype=torch.int32, device=net.device)
    net.reset()

    # exponential rate estimate for the live display only
    tau = 0.1
    decay = math.exp(-dt / tau)
    filt = torch.zeros(net.n, dtype=torch.float32, device=net.device)

    trace_t, trace_d = [], []
    for step in range(n_steps):
        t = step * dt
        rate = rig.drive(t, tf_hz, direction)
        forced = torch.rand(net.n, generator=net.gen,
                            device=net.device) < (rate * dt).clamp(0, 1)
        s = net.step(g_ext=gext, forced=forced)
        counts += s
        filt.mul_(decay).add_(s.float() * (1 - decay) / tau)

        if dash is not None and step % dash_every == 0:
            l_rate = float(filt[monitors["DNa02_L"]].mean()) if len(monitors["DNa02_L"]) else 0.0
            r_rate = float(filt[monitors["DNa02_R"]].mean()) if len(monitors["DNa02_R"]) else 0.0
            diff = l_rate - r_rate
            pops = {k: (float(filt[v].mean()) if len(v) else 0.0)
                    for k, v in monitors.items()}
            lum = rig.luminance(t, tf_hz, direction)
            per_col = _luminance_by_column(retina, lum, rig.idx)
            trace_t.append(t); trace_d.append(diff)
            alive = dash.update(t, per_col, pops, diff,
                                np.tanh(diff / 10.0), banner)
            if not alive:
                return None

    return counts


def _luminance_by_column(retina, lum_t, idx_t):
    """Scatter per-neuron luminance back onto per-column arrays for drawing."""
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


def rate(counts, idx, duration):
    if counts is None or len(idx) == 0:
        return 0.0
    return float(counts[idx].float().mean()) / duration


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M3 — optomotor")
    ap.add_argument("--live", action="store_true", help="live dashboard")
    ap.add_argument("--sweep", action="store_true",
                    help="temporal frequency tuning curve")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--tf", type=float, default=2.0, help="temporal freq (Hz)")
    ap.add_argument("--period", type=float, default=30.0,
                    help="spatial period (deg)")
    ap.add_argument("--site", default="L1", choices=["L1", "L2", "R1-6"])
    ap.add_argument("--rate", type=float, default=150.0,
                    help="peak photoreceptor drive (Hz)")
    ap.add_argument("--bias", type=float, default=7.5,
                    help="tonic depolarising drive to optic lobe neurons, mV. "
                         "The early visual system is inhibition-dominated "
                         "(L2 E:I=0.11) and computes by DISINHIBITION, which "
                         "needs a baseline to disinhibit from. 0 = off.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(paint("flydoom M3 — optomotor response", "1"))
    print(paint("=" * 76, "90"))

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    retina = Retina.build(g, ann, site=args.site)
    net = LIFNetwork.from_graph(g, device=args.device, seed=0)
    rig = GratingRig(net, retina, args.device, args.period, args.rate)

    print(retina.summary())
    print(f"grating    period {args.period:.0f} deg, {args.tf:.1f} Hz, "
          f"drive {args.rate:.0f} Hz peak")
    print(f"W_SYN      {net.p.w_syn * 1e3:.5f} mV   T_dly "
          f"{net.p.t_dly_effective * 1e3:.1f} ms")

    # T4/T5 are monitored PER SUBTYPE. a/b/c/d are tuned to four different
    # directions, so averaging them cancels direction selectivity by
    # construction -- an easy way to mistake a real signal for none.
    mons = {
        "L1": population_indices(g, ann, "L1"),
        "L2": population_indices(g, ann, "L2"),
        "T4a": population_indices(g, ann, "T4a"),
        "T4b": population_indices(g, ann, "T4b"),
        "T5a": population_indices(g, ann, "T5a"),
        "T5b": population_indices(g, ann, "T5b"),
        "LPLC2": population_indices(g, ann, "LPLC2"),
        "DNa02_L": population_indices(g, ann, "DNa02", side="left"),
        "DNa02_R": population_indices(g, ann, "DNa02", side="right"),
    }
    mon_t = {k: torch.as_tensor(v.astype(np.int64), device=args.device)
             for k, v in mons.items()}
    print("populations " + "  ".join(f"{k}={len(v)}" for k, v in mons.items()))

    gext = None
    if args.bias:
        import polars as pl
        cls = pl.read_csv(config.RAW_DIR / "classification.csv.gz",
                          infer_schema_length=50_000)
        optic = cls.filter(pl.col("super_class") == "optic")["root_id"].to_list()
        oidx = torch.as_tensor(g.index_of(optic).astype(np.int64),
                               device=args.device)
        gext = torch.zeros(net.n, dtype=torch.float32, device=args.device)
        gext[oidx] = args.bias * 1e-3
        print(f"tonic bias  {args.bias:.1f} mV to {len(oidx):,} optic lobe "
              f"neurons (rheobase {net.p.threshold_distance * 1e3:.1f} mV)")

    dash = None
    if args.live:
        from flydoom.viz import LiveDashboard, have_display
        if not have_display():
            print(paint("no display; ignoring --live", "33"))
        else:
            dash = LiveDashboard(retina, list(mons), dt=net.p.dt,
                                 title="flydoom M3 — optomotor")

    c = Checks()
    results = {}

    freqs = [0.5, 1.0, 2.0, 4.0, 8.0] if args.sweep else [args.tf]
    print(f"\n{paint('RESPONSES', '1;36')}")
    print(f"  {'TF(Hz)':>7} {'dir':>5} {'L1':>7} {'T4a':>7} {'T4b':>7}"
          f" {'T5a':>7} {'DNa02_L':>9} {'DNa02_R':>9} {'L-R':>8}")

    for tf in freqs:
        for direction, label in ((+1, "R"), (-1, "L")):
            banner = f"TF {tf:.1f} Hz   drift {'->' if direction > 0 else '<-'}"
            counts = run_grating(net, rig, args.duration, tf, direction,
                                 mon_t, dash, retina, banner=banner, gext=gext)
            if counts is None:
                print(paint("\nwindow closed", "33"))
                return 0
            r = {k: rate(counts, v, args.duration) for k, v in mon_t.items()}
            d = r["DNa02_L"] - r["DNa02_R"]
            results[(tf, direction)] = (r, d)
            print(f"  {tf:7.1f} {label:>5} {r['L1']:7.1f} {r['T4a']:7.2f}"
                  f" {r['T4b']:7.2f} {r['T5a']:7.2f} {r['DNa02_L']:9.2f}"
                  f" {r['DNa02_R']:9.2f} {d:8.2f}")

    # ---------------- acceptance ----------------
    print(f"\n{paint('ACCEPTANCE', '1')}")

    # direction selectivity index per motion-detector subtype
    print(f"\n{paint('DIRECTION SELECTIVITY', '1;36')}")
    print(f"  {'subtype':>8} {'rightward':>10} {'leftward':>10} {'DSI':>8}")
    tf0 = args.tf if not args.sweep else 2.0
    dsis = {}
    if (tf0, +1) in results and (tf0, -1) in results:
        for t in ("T4a", "T4b", "T5a", "T5b"):
            a = results[(tf0, +1)][0][t]
            b = results[(tf0, -1)][0][t]
            dsi = (a - b) / (a + b) if (a + b) > 1e-9 else 0.0
            dsis[t] = dsi
            print(f"  {t:>8} {a:10.2f} {b:10.2f} {dsi:+8.3f}")
    best_dsi = max((abs(v) for v in dsis.values()), default=0.0)

    r_any = any(v[0]["L1"] > 1 for v in results.values())
    c.check(r_any, "visual input reaches the lamina",
            f"L1 {max(v[0]['L1'] for v in results.values()):.1f} Hz")

    t4 = max(v[0]["T4a"] for v in results.values())
    t5 = max(v[0]["T5a"] for v in results.values())
    c.check(t4 > 0.5 or t5 > 0.5, "motion detectors respond",
            f"T4a {t4:.2f} Hz, T5a {t5:.2f} Hz")
    c.check(best_dsi > 0.1, "T4/T5 are DIRECTION SELECTIVE",
            f"best |DSI| {best_dsi:.3f} (need >0.1)")

    dn = max(max(v[0]["DNa02_L"], v[0]["DNa02_R"]) for v in results.values())
    c.check(dn > 0.5, "signal reaches DNa02", f"peak {dn:.2f} Hz")

    if (tf0, +1) in results and (tf0, -1) in results:
        dr = results[(tf0, +1)][1]
        dl = results[(tf0, -1)][1]
        c.check(dr * dl < 0,
                "DNa02 asymmetry REVERSES with drift direction",
                f"rightward {dr:+.2f}, leftward {dl:+.2f} Hz")
        c.check(abs(dr - dl) > 0.5, "the reversal is not just noise",
                f"separation {abs(dr - dl):.2f} Hz")

    ok = c.render()
    if dash is not None:
        dash.hold("done — close to exit")

    print(f"\n{paint('=' * 76, '90')}")
    if ok:
        print(paint("VERDICT: M3 PASS — retina to button, end to end.", "1;32"))
        return 0
    print(paint("VERDICT: M3 FAIL", "1;31"))
    if r_any and best_dsi <= 0.1:
        print(paint("""
DIAGNOSIS: the pipeline works; direction selectivity does not exist.

Signal reaches the lamina, the motion detectors fire, and activity arrives at
the descending neurons. What is absent is any PREFERENCE for one direction --
DSI is ~0.000 in every T4/T5 subtype, which is not noise, it is structural.

T4/T5 build direction selectivity by comparing inputs that arrive with
DIFFERENT temporal filters: Mi9 slow and sign-inverting, Mi1/Tm3 fast, Mi4
slow. Delay one line, compare against an undelayed neighbour, and you get a
correlator.

This model has no temporal heterogeneity to build that from. Every neuron
shares one tau_mem (20 ms) and every synapse one conduction delay (2.0 ms),
because those are global constants in Shiu et al.'s parameterisation. With
uniform delays a Hassenstein-Reichardt correlator cannot be expressed, so no
amount of gain or bias tuning will produce direction selectivity here.

Note WHERE that parameterisation was validated: taste. The SEZ pathway is
feedforward-excitatory and works fine with uniform constants at zero baseline.
The optic lobe is neither -- it is inhibition-dominated (measured on this
graph: L2 E:I = 0.11, Mi1 0.36) and computes by disinhibition, which is why
--bias is needed before anything propagates at all.

To actually pass M3 the model needs per-cell-type time constants, at minimum
distinguishing the fast (Mi1, Tm3) from the slow (Mi9, Mi4, Tm9) medulla
lines. That is a change to the MODEL, not to this experiment, and it is a
defensible one -- but it is a departure from the paper we are replicating and
should be recorded as such.""", "33"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
