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


def _hs_indices(graph, ann, side):
    """Horizontal-system cells on one side, resolved by type rather than by a
    registry handle: HS is not a named handle in this project, and the
    optomotor readout in the animal is this population."""
    from flydoom.registry import Handle
    h = Handle(name="HS", group="optic", exact_only=True,
               patterns=("HSE", "HSN", "HSS", "H2"))
    res = ann.resolve(h, side=side)
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



def _lif_params(args):
    """LIFParams carrying whatever mechanisms the CLI switched on."""
    from flydoom.lif import LIFParams
    return LIFParams(stp=bool(getattr(args, "stp", False)),
                     stp_u=getattr(args, "stp_u", config.STP_U),
                     stp_tau_rec=getattr(args, "stp_tau_rec", config.STP_TAU_REC))


def _apply_optic_gain(graph, ann, args) -> None:
    """Scale synapses onto the optic lobe, in place."""
    gain = getattr(args, "optic_gain", 1.0)
    if gain == 1.0:
        return
    from flydoom.gains import optic_gain_multipliers
    m = optic_gain_multipliers(graph, ann, gain)
    graph.signed_syn = (graph.signed_syn * m).astype(np.float32)
    print(paint(f"optic-lobe gain x{gain:g} on {int((m != 1.0).sum()):,} edges "
                f"(replaces tonic bias; W_SYN unchanged elsewhere)", "1;33"))


def _apply_flyvis_gains(graph, ann, args) -> None:
    """Scale edge weights by transplanted per-type gains, in place."""
    path = getattr(args, "flyvis_gains", None)
    if not path:
        return
    from flydoom.gains import edge_multipliers
    mult, rep = edge_multipliers(graph, ann, path)
    graph.signed_syn = (graph.signed_syn * mult).astype(np.float32)
    print(paint("ABLATION: per-type gains transplanted", "1;33"))
    print(f"  source     {rep['source']}")
    print(f"  from       {rep['connectome']}")
    print(f"  covers     {rep['edges_covered']:,}/{rep['edges_total']:,} edges, "
          f"{100 * rep['synapses_covered_frac']:.1f}% of synapses")
    print(f"  multiplier {rep['multiplier_min']:.3f} to "
          f"{rep['multiplier_max']:.2f}, synapse-weighted mean 1.0")


def _bias_vector(net, graph, ann, bias_mv, device=None, spare=()):
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
    if spare:
        # Withhold the bias from named types. Tonic drive carries no stimulus
        # information, so on the cell whose selectivity is being MEASURED it is
        # pure dilution -- see config.OPTIC_GAIN for the 70x figure.
        keep = d = ann.df
        for t in spare:
            ids = d.filter((pl.col("primary_type") == t)
                           | (pl.col("visual_type") == t))["root_id"].to_list()
            if ids:
                gext[torch.as_tensor(graph.index_of(ids).astype(np.int64),
                                     device=device)] = 0.0
    return gext


SUBTYPES = ("T4a", "T4b", "T5a", "T5b")
T4T5 = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
MIRROR_PAIRS = (("T4a", "T4b"), ("T5a", "T5b"))


def _ablate_inputs(g, ann, targets: tuple[str, ...], keep: tuple[str, ...]):
    """Silence every input onto `targets` whose source is not in `keep`.

    This is the in-network counterpart of isolated replay, and it exists
    because replay is not a clean control: applied to a single input, where
    direction selectivity is geometrically impossible, replay still returns
    |DSI| 0.12 instead of 0. Here nothing is replayed. The whole brain runs;
    only the edges landing on the target cells are zeroed, so the surviving
    arms are still driven by the real upstream network with its real dynamics.

    Weights are zeroed rather than removed so the edge list keeps its length
    and order, which keeps edge_delay aligned.
    """
    import polars as _pl
    d = ann.df
    def _ids(names):
        f = d.filter(_pl.col("primary_type").is_in(list(names))
                     | _pl.col("visual_type").is_in(list(names)))
        return set(f["root_id"].unique().to_list())
    pos = {int(r): i for i, r in enumerate(g.root_ids)}
    tgt = np.array(sorted(pos[i] for i in _ids(targets) if i in pos), dtype=np.int64)
    kept = np.array(sorted(pos[i] for i in _ids(keep) if i in pos), dtype=np.int64)
    is_tgt = np.isin(g.post_idx, tgt)
    is_keep = np.isin(g.pre_idx, kept)
    kill = is_tgt & ~is_keep
    n_before = int((g.signed_syn[is_tgt] != 0).sum())
    g.signed_syn = g.signed_syn.copy()
    g.signed_syn[kill] = 0.0
    print(paint(f"ABLATION  onto {targets}: kept {keep}; "
                f"silenced {int(kill.sum()):,} of {n_before:,} input edges; "
                f"{int((is_tgt & is_keep).sum()):,} survive", "1;33"))
    return int(kill.sum())


def _apply_spiking_t4(graded, g, ann, args):
    """Make T4/T5 spiking instead of graded, if asked.

    This used to live inline in one code path only, so --spiking-t4 was
    SILENTLY IGNORED by --dsi-grid and --per-cell: they build their own graded
    mask. Any earlier conclusion about spiking T4 drawn from those paths was
    measuring the graded model twice. It matters because the transfer function
    is not a detail here -- a graded unit's output is a clamped LINEAR ramp in
    voltage, so a phase difference between two arms lands as a small difference
    on a large pedestal, while a threshold turns the same voltage difference
    into a large relative rate difference.
    """
    import polars as _pl
    if graded is None or not getattr(args, "spiking_t4", False):
        return graded
    graded = graded.copy()
    d = ann.df
    n = 0
    for t in T4T5:
        f = d.filter((_pl.col("primary_type") == t) | (_pl.col("visual_type") == t))
        ids = f["root_id"].unique().to_list()
        if ids:
            idx = g.index_of(ids)
            graded[idx] = False
            n += len(idx)
    print(paint(f"T4/T5 made SPIKING ({n:,} cells; threshold nonlinearity "
                f"restored)", "1;33"))
    return graded


def _net_for(g, ann, args, edge_delay, graded):
    """LIFNetwork for `args`, compartmentalised if asked.

    Dendrites are appended after every existing neuron, so population indices,
    retina injection sites and readouts keep their meaning; the edge list keeps
    its order, so edge_delay stays aligned.
    """
    if getattr(args, "keep_inputs", ""):
        _ablate_inputs(g, ann,
                       tuple(t for t in args.ablate_target.split(",") if t),
                       tuple(k for k in args.keep_inputs.split(",") if k))
    if not getattr(args, "compartments", False):
        return LIFNetwork.from_graph(
            g, params=_lif_params(args), device=args.device, seed=0,
            edge_delay=edge_delay, graded=graded)
    from flydoom import compartments as _C
    plan = _C.build(g, ann, g_axial=args.g_axial)
    pre_t, post_t, w_t = g.to_torch(args.device)
    post_t = torch.as_tensor(plan["post_idx"], device=args.device)
    print(paint(f"COMPARTMENTS  {plan['n_cells']:,} T4/T5 split; "
                f"{plan['n_moved']:,} of {plan['n_edges_onto_targets']:,} "
                f"inputs moved distal; g_ax={args.g_axial}", "1;36"))
    return LIFNetwork(plan["n_total"], pre_t, post_t, w_t,
                      _lif_params(args), args.device, 0,
                      edge_delay=edge_delay,
                      graded=_C.extend_graded(graded, plan),
                      axial_partner=plan["axial_partner"],
                      g_axial=plan["g_ax"])


def per_cell_dsi(args) -> int:
    """DSI per individual cell, at the best unsaturated operating point.

    The pooled mean answers "is the population direction selective". This
    answers a different and more permissive question: "is ANY cell". A real
    correlator population is heterogeneous, and experimental work reports the
    cells that pass a selection criterion rather than the average over every
    cell in the field of view.
    """
    import json

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    _apply_flyvis_gains(g, ann, args)
    _apply_optic_gain(g, ann, args)
    if getattr(args, "gain_onto_t4", 1.0) != 1.0:
        from flydoom.gains import gain_onto_types
        gm = gain_onto_types(g, ann, ("T4a","T4b","T4c","T4d",
                                      "T5a","T5b","T5c","T5d"),
                             args.gain_onto_t4)
        g.signed_syn = (g.signed_syn * gm).astype(np.float32)
        print(paint(f"ABLATION: gain x{args.gain_onto_t4:g} onto T4/T5 only "
                    f"({int((gm != 1.0).sum()):,} edges)", "1;33"))
    retina = Retina.build(g, ann, site=tuple(args.site.split("+")))
    graded = _apply_spiking_t4(g.graded_mask(ann), g, ann, args)
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=args.slow_delay * 1e-3)
    net = _net_for(g, ann, args, edge_delay, graded)
    gext = _bias_vector(net, g, ann, args.bias, args.device,
                        spare=T4T5 if getattr(args, 'spare_t4_bias', False) else ())
    rig = GratingRig(net, retina, args.device, args.period, args.rate)
    mon = {t: population_indices(g, ann, t) for t in SUBTYPES}

    print(paint("flydoom M3 — per-cell direction selectivity", "1"))
    print(paint("=" * 76, "90"))
    print(f"bias {args.bias:.1f} mV, period {args.period:.0f} deg, "
          f"tf {args.tf:.1f} Hz, {args.duration:.1f} s per direction\n")

    counts = {}
    for direction in (+1, -1):
        c = run_grating(net, rig, args.duration, args.tf, direction, mon,
                        gext=gext)
        counts[direction] = c.detach().cpu().numpy() / args.duration

    # CONTROLS. With 1,457 cells, some will pass any threshold by chance, and
    # a cell firing 0.4 Hz against 1.2 Hz scores DSI 0.5 on almost no signal.
    # Two checks decide whether a selective minority is real:
    #
    #   1. ACTIVITY. Recompute the fractions over only strongly driven cells.
    #      If selectivity lives entirely in the weakly driven tail it is noise
    #      on a small number, not a computation.
    #   2. MIRROR SIGN. T4a and T4b have mirrored input geometry, so genuinely
    #      selective cells of the two subtypes must prefer OPPOSITE directions.
    #      Per-cell this is a powerful test: with hundreds of selective cells,
    #      a real correlator gives a strong sign bias and noise gives 50/50.
    SEL = 0.5      # the threshold at which experiments SELECT a terminal
    record = {"bias_mv": args.bias, "period_deg": args.period,
              "tf_hz": args.tf, "duration_s": args.duration,
              "selection_threshold": SEL, "cells": {}}
    print(f"  {'type':>6} {'n':>6} {'firing>1Hz':>11} {'|DSI|>0.1':>10} "
          f"{'|DSI|>0.5':>10} {'max|DSI|':>9} {'p95':>8}")
    for t in SUBTYPES:
        idx = mon[t]
        if not len(idx):
            continue
        a, b = counts[+1][idx], counts[-1][idx]
        tot = a + b
        live = tot > 1.0           # cells actually driven by the stimulus
        dsi = np.where(tot > 1e-9, (a - b) / np.maximum(tot, 1e-9), 0.0)
        d_live = np.abs(dsi[live]) if live.any() else np.zeros(1)
        strong = tot > 20.0        # well-driven cells only
        d_strong = np.abs(dsi[strong]) if strong.any() else np.zeros(1)
        # sign bias among cells that are selective at all
        sel = np.abs(dsi) > 0.1
        pos = int((dsi[sel] > 0).sum()); neg = int((dsi[sel] < 0).sum())
        record["cells"][t] = {
            "n_strong": int(strong.sum()),
            "strong_frac_gt_0.5": float((d_strong > SEL).mean()),
            "strong_frac_gt_0.1": float((d_strong > 0.1).mean()),
            "sel_pos": pos, "sel_neg": neg,
            "sel_pos_frac": float(pos / max(pos + neg, 1)),
            "median_rate_hz": float(np.median(tot) / 2.0),
            "n": int(len(idx)), "n_live": int(live.sum()),
            "frac_gt_0.1": float((d_live > 0.1).mean()),
            "frac_gt_0.5": float((d_live > SEL).mean()),
            "max_abs": float(d_live.max()), "p95": float(np.percentile(d_live, 95)),
            "pooled": float((a.mean() - b.mean()) /
                            max(a.mean() + b.mean(), 1e-9)),
        }
        r = record["cells"][t]
        print(f"  {t:>6} {r['n']:6d} {r['n_live']:11d} {r['frac_gt_0.1']:9.1%} "
              f"{r['frac_gt_0.5']:10.1%} {r['max_abs']:9.4f} {r['p95']:8.4f}")

    print(f"\n  CONTROL 1 -- restrict to well-driven cells (>20 Hz total):")
    print(f"  {'type':>6} {'n_strong':>9} {'|DSI|>0.1':>10} {'|DSI|>0.5':>10}")
    for t, r in record["cells"].items():
        print(f"  {t:>6} {r['n_strong']:9d} {r['strong_frac_gt_0.1']:9.1%} "
              f"{r['strong_frac_gt_0.5']:10.1%}")

    print(f"\n  CONTROL 2 -- mirror sign bias among selective cells (|DSI|>0.1):")
    print(f"  {'type':>6} {'prefer +':>9} {'prefer -':>9} {'+ fraction':>11}"
          f"   a real correlator: T4a and T4b OPPOSE")
    for t, r in record["cells"].items():
        print(f"  {t:>6} {r['sel_pos']:9d} {r['sel_neg']:9d} "
              f"{r['sel_pos_frac']:10.1%}")

    print(f"\n  pooled means (what the population number reports):")
    for t, r in record["cells"].items():
        print(f"    {t}: {r['pooled']:+.5f}")
    print(paint("""
  Read frac>0.5 first. If a real minority of cells is selective, the pooled
  mean can sit near zero purely by averaging them against undriven cells, and
  the negative result would be an artefact of the readout rather than of the
  model. If frac>0.5 is ~0 AND max|DSI| is small, pooling is exonerated and the
  failure is in the model.""", "90"))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {args.json}")
    return 0


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
    _apply_flyvis_gains(g, ann, args)
    _apply_optic_gain(g, ann, args)
    if getattr(args, "gain_onto_t4", 1.0) != 1.0:
        from flydoom.gains import gain_onto_types
        gm = gain_onto_types(g, ann, ("T4a","T4b","T4c","T4d",
                                      "T5a","T5b","T5c","T5d"),
                             args.gain_onto_t4)
        g.signed_syn = (g.signed_syn * gm).astype(np.float32)
        print(paint(f"ABLATION: gain x{args.gain_onto_t4:g} onto T4/T5 only "
                    f"({int((gm != 1.0).sum()):,} edges)", "1;33"))
    retina = Retina.build(g, ann, site=tuple(args.site.split("+")))
    graded = _apply_spiking_t4(g.graded_mask(ann), g, ann, args)
    ceiling = config.GRADED_MAX_RATE

    biases = tuple(float(b) for b in args.grid_biases.split(",")) \
        if getattr(args, "grid_biases", "") else \
        (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.5, 7.5)
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
        net = _net_for(g, ann, args, edge_delay, graded)
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
    ap.add_argument("--optic-gain", type=float, default=config.OPTIC_GAIN,
                    help="scale synapses onto the visual populations. This "
                         "REPLACES --bias rather than adding to it: see "
                         "config.OPTIC_GAIN for the measurement showing tonic "
                         "bias costs 70x in DSI while gain does not.")
    ap.add_argument("--grid-biases", default="",
                    help="override the grid's bias axis. The default starts at "
                         "1.0 mV, which turned out to be its own floor: DSI "
                         "rises monotonically as bias falls, so the lowest "
                         "point tested was also the best one.")
    ap.add_argument("--keep-inputs", default="",
                    help="comma-separated presynaptic types to KEEP onto the "
                         "ablation targets; every other input onto them is "
                         "silenced. Run in the live network, unlike isolated "
                         "replay, which manufactures DSI (see m3e).")
    ap.add_argument("--ablate-target", default="T4a,T4b,T5a,T5b",
                    help="cell types whose inputs --keep-inputs filters.")
    ap.add_argument("--compartments", action="store_true",
                    help="split T4/T5 into spiking soma + passive dendrite, "
                         "with off-column inputs re-routed distally. A shunt "
                         "then divides one branch instead of the whole cell, "
                         "which is what null-direction suppression requires.")
    ap.add_argument("--g-axial", type=float, default=1.0,
                    help="axial conductance between the compartments. 0 = "
                         "independent; large = back to the point neuron.")
    ap.add_argument("--spiking-t4", action="store_true",
                    help="make T4/T5 spiking rather than graded. A graded unit "
                         "is rectified-LINEAR in its unsaturated range, and a "
                         "linear unit's mean rate cannot depend on the "
                         "relative phase of its inputs -- so the threshold "
                         "nonlinearity direction selectivity needs is absent "
                         "by construction.")
    ap.add_argument("--spare-t4-bias", action="store_true",
                    help="apply the tonic bias to the optic lobe EXCEPT the "
                         "motion detectors. Tests whether bias is needed "
                         "upstream (to make an inhibition-dominated lamina "
                         "conduct) while being fatal at the output stage, "
                         "where it adds direction-blind drive to the very cell "
                         "whose selectivity is being measured.")
    ap.add_argument("--gain-onto-t4", type=float, default=1.0,
                    help="ABLATION: scale synapses onto T4/T5 only, leaving "
                         "the rest of the optic lobe at its normal operating "
                         "point. Diagnostic for whether a GLOBAL scalar can "
                         "restore selectivity or whether a targeted one is "
                         "required -- the latter would be evidence that "
                         "per-stage gain, not a single number, is what is "
                         "missing.")
    ap.add_argument("--stp", action="store_true",
                    help="short-term synaptic depression, applied uniformly. "
                         "See config.STP: it attenuates tonic high-rate drive "
                         "far more than sparse bursts, which is the shape of "
                         "the measured arm imbalance.")
    ap.add_argument("--stp-u", type=float, default=config.STP_U)
    ap.add_argument("--stp-tau-rec", type=float, default=config.STP_TAU_REC)
    ap.add_argument("--per-cell", action="store_true",
                    help="report the DISTRIBUTION of DSI over individual "
                         "cells, not the population mean. Experimental studies "
                         "SELECT terminals at DSI>0.5 and report those; pooling "
                         "every cell -- including ones barely driven -- dilutes "
                         "any real signal toward zero, so a population mean of "
                         "~0 is consistent with a selective minority.")
    ap.add_argument("--flyvis-gains", type=Path,
                    help="ABLATION: scale each edge by the per-cell-type gain "
                         "fitted by Lappalainen et al. (2024), normalised to "
                         "leave global gain unchanged. Tests whether relative "
                         "per-type gain is the missing quantity. See "
                         "flydoom/gains.py for what this can and cannot show.")
    ap.add_argument("--json", type=Path,
                    help="serialise the --dsi-grid measurement. The table and "
                         "figure in the write-up are generated from this file, "
                         "so they cannot drift from the run.")
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
    args = ap.parse_args()

    if args.per_cell:
        return per_cell_dsi(args)
    if args.dsi_grid:
        return dsi_grid(args)

    print(paint("flydoom M3 — optomotor response", "1"))
    print(paint("=" * 76, "90"))

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    _apply_flyvis_gains(g, ann, args)
    _apply_optic_gain(g, ann, args)
    if getattr(args, "gain_onto_t4", 1.0) != 1.0:
        from flydoom.gains import gain_onto_types
        gm = gain_onto_types(g, ann, ("T4a","T4b","T4c","T4d",
                                      "T5a","T5b","T5c","T5d"),
                             args.gain_onto_t4)
        g.signed_syn = (g.signed_syn * gm).astype(np.float32)
        print(paint(f"ABLATION: gain x{args.gain_onto_t4:g} onto T4/T5 only "
                    f"({int((gm != 1.0).sum()):,} edges)", "1;33"))
    retina = Retina.build(g, ann, site=tuple(args.site.split("+")))
    edge_delay = g.edge_delay_steps(ann, config.DT, t_slow=args.slow_delay * 1e-3)
    graded = g.graded_mask(ann) if args.graded else None
    graded = _apply_spiking_t4(graded, g, ann, args)
    net = _net_for(g, ann, args, edge_delay, graded)
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
        # The horizontal system. In the animal these, not DNa02, are the
        # optomotor output stage: large-field cells whose function is to pool
        # thousands of T4/T5 terminals across the eye. Reading here is an
        # electrode placement, not a change to the model, and it is where the
        # optomotor literature records. DNa02 is one cell per side and carries
        # a standing left-right asymmetry of tens of Hz, which is the wrong
        # instrument for a signal this size.
        "HS_L": _hs_indices(g, ann, "left"),
        "HS_R": _hs_indices(g, ann, "right"),
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

    gext = _bias_vector(net, g, ann, args.bias, args.device,
                        spare=T4T5 if getattr(args, 'spare_t4_bias', False) else ())
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
            n2 = LIFNetwork.from_graph(g, params=_lif_params(args), device=args.device, seed=0,
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

    hs = {k: (v[0]["HS_L"] - v[0]["HS_R"]) for k, v in results.items()}
    if hs:
        print("\n  HORIZONTAL SYSTEM (HS_L - HS_R), the optomotor readout:")
        for k, v in hs.items():
            lbl = f"tf {k[0]:g} {'rightward' if k[1] > 0 else 'leftward'}"
            print(f"    {lbl:>18}  {v:+8.3f} Hz")
        vals = list(hs.values())
        if len(vals) >= 2:
            print(f"    separation  {max(vals) - min(vals):.3f} Hz   "
                  f"signs {'OPPOSE' if max(vals) * min(vals) < 0 else 'agree'}")
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
