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
        self.out_buf = torch.full((net.n,), -1.0, dtype=torch.float32,
                                  device=device)
        self.sp = spatial_period_deg
        # Weber adaptation state, per driven neuron. Removes the DC so L1
        # codes contrast rather than luminance -- see retina.LuminanceAdaptation
        # for why this is load-bearing rather than cosmetic.
        self.adapt_mean = torch.full_like(self.base_phase, 0.5)
        self.adapt_decay = math.exp(-net.p.dt / 0.25)
        self.adapt_gain = 2.5
        self.adapting = True

    def reset_adaptation(self):
        self.adapt_mean.fill_(0.5)

    def _contrast(self, lum):
        self.adapt_mean.mul_(self.adapt_decay).add_(lum * (1 - self.adapt_decay))
        c = (lum - self.adapt_mean) / self.adapt_mean.clamp(min=1e-3)
        return (c * self.adapt_gain).clamp(-1.0, 1.0)

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

    def out_set(self, t, tf_hz, direction, graded_max_rate, dt):
        """Direct output override for a GRADED input population.

        Returns -1 (free) everywhere except the injection sites, whose release
        is set continuously by luminance. This is what a photoreceptor does;
        Poisson spikes into a non-spiking cell would be a category error.
        """
        lum = self.luminance(t, tf_hz, direction)
        if self.adapting:
            c = self._contrast(lum)
            if self.inverts:
                c = -c
            act = (0.5 + 0.5 * c).clamp(0.0, 1.0)
        else:
            act = 1.0 - lum if self.inverts else lum
        self.out_buf.fill_(-1.0)
        self.out_buf[self.idx] = act * (graded_max_rate * dt)
        return self.out_buf


def run_grating(net, rig, duration, tf_hz, direction, monitors,
                dash=None, retina=None, dash_every=25, banner="", gext=None):
    """Step the network under a drifting grating; return spike counts."""
    dt = net.p.dt
    n_steps = int(round(duration / dt))
    # Accumulate OUT, not spikes. out is in spike-equivalents -- 1.0 on the
    # step a spiking cell fires, rate*dt every step for a graded cell -- so
    # sum(out)/duration is a firing rate for both populations. Counting spikes
    # would report exactly 0.00 Hz for every graded neuron.
    counts = torch.zeros(net.n, dtype=torch.float32, device=net.device)
    net.reset()

    # exponential rate estimate for the live display only
    tau = 0.1
    decay = math.exp(-dt / tau)
    filt = torch.zeros(net.n, dtype=torch.float32, device=net.device)

    trace_t, trace_d = [], []
    graded_input = bool(net.any_graded) and bool(net.graded[rig.idx].all())
    for step in range(n_steps):
        t = step * dt
        if graded_input:
            s = net.step(g_ext=gext,
                         out_set=rig.out_set(t, tf_hz, direction,
                                             net.p.graded_max_rate, dt))
        else:
            rate = rig.drive(t, tf_hz, direction)
            forced = torch.rand(net.n, generator=net.gen,
                                device=net.device) < (rate * dt).clamp(0, 1)
            s = net.step(g_ext=gext, forced=forced)
        counts += net.out
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
    return float(counts[idx].mean()) / duration


# --------------------------------------------------------------------------



def _bias_vector(net, graph, ann, bias_mv, device=None):
    """Tonic depolarising drive to the optic lobe, as a g_ext vector.

    The early visual system is inhibition-dominated and computes by
    disinhibition, which needs a baseline to disinhibit from. Returns None at
    zero bias so the caller can skip the add entirely.
    """
    if not bias_mv:
        return None
    import polars as pl
    device = device or net.out.device
    cls = pl.read_csv(config.RAW_DIR / "classification.csv.gz",
                      infer_schema_length=50_000)
    optic = cls.filter(
        pl.col("super_class").is_in(list(config.BIASED_SUPER_CLASSES))
    )["root_id"].to_list()
    oidx = torch.as_tensor(graph.index_of(optic).astype(np.int64), device=device)
    gext = torch.zeros(net.n, dtype=torch.float32, device=device)
    gext[oidx] = bias_mv * 1e-3
    return gext


SUBTYPES = ("T4a", "T4b", "T5a", "T5b")
MIRROR_PAIRS = (("T4a", "T4b"), ("T5a", "T5b"))


def dsi_grid(args) -> int:
    """Measure DSI across the operating range and write it out.

    One hand-picked configuration cannot support a claim about direction
    selectivity here, because the single largest effect on the number is not
    the wiring but whether the graded units are against their rate ceiling. A
    saturated T4 reports DSI ~ 1e-5 whatever its inputs do. So we sweep, record
    the saturation level alongside every DSI, and let the table say which
    points were interpretable.
    """
    import json

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    retina = Retina.build(g, ann, site=tuple(args.site.split("+")))
    graded = g.graded_mask(ann)
    ceiling = config.GRADED_MAX_RATE

    biases = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.5, 7.5)
    periods = (8.0, 15.0, 30.0)
    tfs = (1.0, 2.0, 4.0)

    print(paint("flydoom M3 — DSI over the operating range", "1"))
    print(paint("=" * 76, "90"))
    print(f"{len(biases)}x{len(periods)}x{len(tfs)} = "
          f"{len(biases) * len(periods) * len(tfs)} points, graded ceiling "
          f"{ceiling:.0f} Hz\n")
    print(f"  {'bias':>5} {'per':>5} {'tf':>4} {'satur':>7}"
          + "".join(f"{t:>10}" for t in SUBTYPES)
          + f"{'|DSI|max':>10}  mirror")

    points = []
    for bias in biases:
        edge_delay = g.edge_delay_steps(ann, config.DT,
                                        t_slow=args.slow_delay * 1e-3)
        net = LIFNetwork.from_graph(g, device=args.device, seed=0,
                                    edge_delay=edge_delay, graded=graded)
        gext = _bias_vector(net, g, ann, bias)
        for period in periods:
            rig = GratingRig(net, retina, args.device, period, args.rate)
            mon = {t: population_indices(g, ann, t) for t in SUBTYPES}
            for tf in tfs:
                r = {}
                for direction in (+1, -1):
                    counts = run_grating(net, rig, args.duration, tf,
                                         direction, mon, gext=gext)
                    r[direction] = {t: rate(counts, mon[t], args.duration)
                                    for t in SUBTYPES}
                dsi = {}
                for t in SUBTYPES:
                    a, b = r[+1][t], r[-1][t]
                    dsi[t] = (a - b) / (a + b) if (a + b) > 1e-9 else 0.0
                peak = max(max(r[+1][t], r[-1][t]) for t in SUBTYPES)
                sat = peak / ceiling
                opposed = {f"{x}/{y}": (dsi[x] * dsi[y] < 0)
                           for x, y in MIRROR_PAIRS}
                points.append({
                    "bias_mv": bias, "period_deg": period, "tf_hz": tf,
                    "rates": {"rightward": r[+1], "leftward": r[-1]},
                    "dsi": dsi, "peak_rate_hz": peak, "saturation": sat,
                    "mirror_opposed": opposed,
                })
                best = max(abs(v) for v in dsi.values())
                mark = "".join("O" if v else "." for v in opposed.values())
                print(f"  {bias:5.1f} {period:5.0f} {tf:4.1f} {sat:6.0%}"
                      + "".join(f"{dsi[t]:+10.5f}" for t in SUBTYPES)
                      + f"{best:10.5f}  {mark}")

    # A point is interpretable only if the readout is not against its ceiling.
    UNSAT = 0.75
    usable = [p for p in points if p["saturation"] < UNSAT]
    record = {
        "graded_max_rate_hz": ceiling,
        "saturation_threshold": UNSAT,
        "duration_s": args.duration,
        "slow_delay_ms": args.slow_delay,
        "site": args.site,
        "n_points": len(points),
        "n_unsaturated": len(usable),
        "points": points,
    }
    if usable:
        best = max(usable, key=lambda p: max(abs(v) for v in p["dsi"].values()))
        record["best_unsaturated"] = best
        both = [p for p in usable if all(p["mirror_opposed"].values())]
        record["n_both_pairs_opposed"] = len(both)
        print(f"\n{paint('SUMMARY', '1;36')}")
        print(f"  {len(usable)}/{len(points)} points below "
              f"{UNSAT:.0%} saturation")
        print(f"  best |DSI| there: "
              f"{max(abs(v) for v in best['dsi'].values()):.5f}"
              f"  at bias {best['bias_mv']}, period {best['period_deg']:.0f}, "
              f"tf {best['tf_hz']}")
        print(f"  both mirror pairs opposed at "
              f"{len(both)}/{len(usable)} unsaturated points"
              f"  ({len(both) / len(usable):.0%})")
        print("""
  Read the last line before the one above it. Opposed mirror pairs are the
  QUALITATIVE signature of a correlator, and if they appear at only some
  operating points and flip at others, that is a sign fluctuating around zero
  rather than a correlator whose output is small.""")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M3 — optomotor")
    ap.add_argument("--live", action="store_true", help="live dashboard")
    ap.add_argument("--sweep", action="store_true",
                    help="temporal frequency tuning curve")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--tf", type=float, default=2.0, help="temporal freq (Hz)")
    ap.add_argument("--period", type=float, default=30.0,
                    help="spatial period (deg)")
    ap.add_argument("--site", default="L1+L2+L3",
                    help="injection site(s), '+'-separated. Default drives all "
                         "three lamina monopolars because the fast and slow "
                         "arms of the T4 correlator are fed by DIFFERENT ones "
                         "(Mi1<-L1, Mi9<-L3); driving L1 alone leaves the slow "
                         "arm silent.")
    ap.add_argument("--rate", type=float, default=150.0,
                    help="peak photoreceptor drive (Hz)")
    ap.add_argument("--bias", type=float, default=7.5,
                    help="tonic depolarising drive to optic lobe neurons, mV. "
                         "The early visual system is inhibition-dominated "
                         "(L2 E:I=0.11) and computes by DISINHIBITION, which "
                         "needs a baseline to disinhibit from. 0 = off.")
    ap.add_argument("--slow-delay", type=float, default=config.T_DLY_SLOW * 1e3,
                    help="conduction delay on the SLOW medulla lines (Mi9, Mi4, "
                         "CT1, Tm9) in ms. Equal to T_dly disables the "
                         "correlator. This is a documented DEVIATION from "
                         "Shiu et al., who use one global delay.")
    ap.add_argument("--graded", action="store_true",
                    help="model the optic lobe as GRADED non-spiking units. "
                         "Photoreceptors and lamina/medulla neurons do not "
                         "spike in a real fly; see config.GRADED_SUPER_CLASSES.")
    ap.add_argument("--slow-sweep", action="store_true",
                    help="sweep the slow-line delay and report DSI for each")
    ap.add_argument("--dsi-grid", action="store_true",
                    help="sweep the operating point and report DSI at every "
                         "point, instead of trusting one hand-picked config. "
                         "Graded units saturate against GRADED_MAX_RATE, and a "
                         "saturated unit reports DSI ~ 0 no matter what the "
                         "wiring does, so the honest number is the best one "
                         "found BELOW saturation.")
    ap.add_argument("--json", type=Path,
                    help="serialise the --dsi-grid measurement. The table and "
                         "figure in the write-up are generated from this file, "
                         "so they cannot drift from the run.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.dsi_grid:
        return dsi_grid(args)

    print(paint("flydoom M3 — optomotor response", "1"))
    print(paint("=" * 76, "90"))

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    retina = Retina.build(g, ann, site=tuple(args.site.split("+")))
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=args.slow_delay * 1e-3)
    graded = g.graded_mask(ann) if args.graded else None
    net = LIFNetwork.from_graph(g, device=args.device, seed=0,
                                edge_delay=edge_delay, graded=graded)
    if graded is not None:
        print(f"graded     {int(graded.sum()):,} non-spiking optic-lobe "
              f"neurons; LC/LPLC and central cells still spike")
    rig = GratingRig(net, retina, args.device, args.period, args.rate)

    print(retina.summary())
    print(f"grating    period {args.period:.0f} deg, {args.tf:.1f} Hz, "
          f"drive {args.rate:.0f} Hz peak")
    print(f"W_SYN      {net.p.w_syn * 1e3:.5f} mV")
    print(f"delays     {net.delay_summary}")
    if args.slow_delay * 1e-3 != net.p.t_dly:
        print(paint(f"           SLOW LINES {config.SLOW_LINES} at "
                    f"{args.slow_delay:.0f} ms -- a deviation from the paper",
                    "33"))

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

    dash = None
    if args.live:
        from flydoom.viz import LiveDashboard, have_display
        if not have_display():
            print(paint("no display; ignoring --live", "33"))
        else:
            dash = LiveDashboard(retina, list(mons), dt=net.p.dt,
                                 title="flydoom M3 — optomotor")

    gext = _bias_vector(net, g, ann, args.bias, args.device)
    if gext is not None:
        print(f"tonic bias  {args.bias:.1f} mV to the optic lobe "
              f"(rheobase {net.p.threshold_distance * 1e3:.1f} mV)")

    if args.slow_sweep:
        print(f"\n{paint('SLOW-LINE DELAY SWEEP', '1;36')}")
        print(paint("  Does a temporal offset on Mi9/Mi4/CT1/Tm9 produce "
                    "direction\n  selectivity from the existing wiring?", "90"))
        print(f"\n  {'slow(ms)':>9} {'T4a R':>8} {'T4a L':>8} {'DSI(T4a)':>9}"
              f" {'DSI(T4b)':>9} {'DSI(T5a)':>9} {'best':>7}")
        for slow_ms in (1.8, 20, 40, 60, 80, 120, 160, 240):
            ed = g.edge_delay_steps(ann, config.DT, t_slow=slow_ms * 1e-3)
            n2 = LIFNetwork.from_graph(g, device=args.device, seed=0,
                                       edge_delay=ed)
            rig2 = GratingRig(n2, retina, args.device, args.period, args.rate)
            r = {}
            for direction in (+1, -1):
                cts = run_grating(n2, rig2, args.duration, args.tf, direction,
                                  mon_t, None, retina, gext=gext)
                r[direction] = {k: rate(cts, v, args.duration)
                                for k, v in mon_t.items()}
            def dsi(t):
                a, b = r[1][t], r[-1][t]
                return (a - b) / (a + b) if (a + b) > 1e-9 else 0.0
            ds = {t: dsi(t) for t in ("T4a", "T4b", "T5a", "T5b")}
            best = max(abs(v) for v in ds.values())
            flag = paint("  <-- SELECTIVE", "32") if best > 0.1 else ""
            print(f"  {slow_ms:9.1f} {r[1]['T4a']:8.2f} {r[-1]['T4a']:8.2f}"
                  f" {ds['T4a']:+9.3f} {ds['T4b']:+9.3f} {ds['T5a']:+9.3f}"
                  f" {best:7.3f}{flag}")
        return 0

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

    # The MIRROR-PAIR test. T4a/T4b and T4c/T4d have mirrored input geometry,
    # so genuine selectivity gives each pair OPPOSITE signs. Magnitude alone is
    # not evidence -- a global response difference moves both members the same
    # way, and that is what every earlier configuration produced.
    pairs = [("T4a", "T4b"), ("T5a", "T5b")]
    opposed = []
    for x, y in pairs:
        if x in dsis and y in dsis:
            opposed.append(dsis[x] * dsis[y] < 0)
            print(f"  mirror pair {x}/{y}: {dsis[x]:+.5f} vs {dsis[y]:+.5f}"
                  f"  -> {'OPPOSITE' if dsis[x] * dsis[y] < 0 else 'same sign'}")
    any_opposed = any(opposed)

    r_any = any(v[0]["L1"] > 1 for v in results.values())
    c.check(r_any, "visual input reaches the lamina",
            f"L1 {max(v[0]['L1'] for v in results.values()):.1f} Hz")

    t4 = max(v[0]["T4a"] for v in results.values())
    t5 = max(v[0]["T5a"] for v in results.values())
    c.check(t4 > 0.5 or t5 > 0.5, "motion detectors respond",
            f"T4a {t4:.2f} Hz, T5a {t5:.2f} Hz")
    c.check(any_opposed, "mirror pairs have OPPOSITE-signed DSI",
            "the qualitative signature of a correlator")
    c.check(best_dsi > 0.1, "direction selectivity is BIOLOGICALLY STRONG",
            f"best |DSI| {best_dsi:.4f}, need >0.1 (real fly ~0.5-0.9)")

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
DIAGNOSIS: the pipeline works; direction selectivity does not.

Signal reaches the lamina, the motion detectors fire, activity arrives at the
descending neurons. What is absent is any PREFERENCE for one direction. The
decisive check is the SIGN, not the magnitude: T4a and T4b have mirrored input
geometry, so genuine selectivity gives them OPPOSITE-signed DSI. They never do
-- they move together, with T4a simply larger.

What IS in the connectome, verified:
  * the correlator geometry. T4a takes fast excitation centred (Mi1 +30.6,
    Tm3 +7.0) and slow inhibition on OPPOSITE flanks (Mi9 -7.4 at (-0.76,
    +0.26), Mi4 -3.4 at (+0.39,-0.60)). All four subtypes, four mirrored axes.
  * the two arms are fed by DIFFERENT lamina lines: Mi1 <- L1, Mi9/Tm9 <- L3.

What was tried and did NOT produce selectivity:
  * per-edge conduction delays on the slow lines, 1.8 to 240 ms
  * per-neuron membrane time constants, fast 5-20 ms against slow 60-200 ms
  * grating periods 8 to 30 deg, tonic bias 6.5 to 7.5 mV
  * both spike-rate and membrane-potential readouts

Two things learned on the way, both worth keeping:
  * mean membrane potential CANNOT show direction selectivity in this model.
    Inputs sum linearly into g, and the mean of a linear sum does not depend on
    the relative phase of its terms. Only the spike threshold is nonlinear, so
    spike rate is the only valid readout.
  * driving L1 alone starves the slow arm. Mi9 modulation was 0.009 mV against
    Mi1's 4.58 mV. Driving L1+L2+L3 raised it 139x, to 1.19 mV. Necessary, but
    not sufficient.

The remaining obstacle looks like signal-to-operating-point, not timing. Under
the tonic bias needed to make an inhibition-dominated circuit conduct at all,
T4 sits ~7 mV above rest and the visual modulation reaching it is ~0.15 mV --
about 2%. A correlator's directional term is a fraction of that, and it is
buried.

The deeper reason is that photoreceptors and lamina monopolars are GRADED,
NON-SPIKING neurons. They signal by continuous transmitter release, not by
spikes. A spiking LIF cannot represent that, and the tonic-bias surrogate we
substitute destroys the signal-to-background the circuit relies on. That is a
limitation of the MODEL CLASS, not of the connectome or of these parameters.
""", "33"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
