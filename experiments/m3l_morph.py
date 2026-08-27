#!/usr/bin/env python3
"""M3l — rebuild the hand-built correlator, then morph it into T4 until it breaks.

TWO JOBS.

First, PROVENANCE. The figure DSI = -0.79 for a hand-built correlator appears
six times in the manuscript and carries its central argument: that the model
class can compute direction, so the network's 0.002 is a failure of the wired
model and not of the method. No script in this repository produces that number.
It is asserted in prose and in code comments only. After a sweep whose reported
runtime was physically impossible went unnoticed for two days, an unreproducible
linchpin is not acceptable. This rebuilds it from scratch.

Second, DIAGNOSIS. Eighteen hypotheses have been eliminated by removing things
from T4 and finding no improvement. That approach cannot succeed if the cause is
a COMBINATION, because removing any single member leaves the rest. So go the
other way: start from a correlator that demonstrably works and add T4's real
properties one at a time until it stops working. The step that breaks it is the
answer, and unlike an ablation it cannot be explained away by what else remains.

THE LADDER. Each rung keeps everything from the rungs above it.

  0 ideal      two point inputs, balanced weights, one delayed
  1 weights    the measured conductance ratio, g_e 0.0018 vs g_i 0.0667
  2 fanin      each arm becomes many cells instead of one point sample
  3 spread     those cells sample a spread of columns, not one phase
  4 graded     inputs pass through the clamped graded transfer function
  5 pedestal   a DC background of the other ~14,600 inputs onto T4

Direction is reversed by negating the spatial phase step, which is what moving
the other way does to a grating. Nothing else changes between the two runs.
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
from flydoom.lif import LIFNetwork, LIFParams  # noqa: E402


def paint(t, c):
    return f"\033[{c}m{t}\033[0m"


def build(n_exc, n_inh, w_exc, w_inh, delay_exc_ms, delay_inh_ms, device):
    """One output cell fed by n_exc excitatory and n_inh inhibitory inputs."""
    n = n_exc + n_inh + 1
    tgt = n - 1
    pre = np.arange(n - 1, dtype=np.int64)
    post = np.full(n - 1, tgt, dtype=np.int64)
    syn = np.concatenate([np.full(n_exc, +w_exc, dtype=np.float32),
                          np.full(n_inh, -w_inh, dtype=np.float32)])
    dt_ms = config.DT * 1e3
    delay = np.concatenate([
        np.full(n_exc, max(1, round(delay_exc_ms / dt_ms))),
        np.full(n_inh, max(1, round(delay_inh_ms / dt_ms)))]).astype(np.int64)
    graded = np.zeros(n, dtype=bool)
    graded[:n - 1] = True                    # inputs are graded, target spikes
    net = LIFNetwork(n,
                     torch.as_tensor(pre, device=device),
                     torch.as_tensor(post, device=device),
                     torch.as_tensor(syn, device=device),
                     LIFParams(), device, 0,
                     edge_delay=delay, graded=graded)
    return net, tgt


def run(net, tgt, n_exc, n_inh, phases_exc, phases_inh, tf, seconds,
        direction, device, clamp_graded, pedestal_hz):
    """Drive the inputs with sinusoids and return the target's mean rate.

    `direction` flips the sign of the spatial phase, which is exactly what
    reversing a grating does and is the only thing that differs between arms.
    """
    dt = config.DT
    steps = int(round(seconds / dt))
    ceiling = net.p.graded_max_rate
    n_in = n_exc + n_inh
    ph = np.concatenate([phases_exc, phases_inh]) * direction
    out_set = torch.full((net.n,), -1.0, dtype=torch.float32, device=device)
    acc, cnt = 0.0, 0
    skip = int(0.5 / dt)
    net.reset()
    for s in range(steps):
        t = s * dt
        val = 0.5 + 0.5 * np.sin(2 * math.pi * tf * t + ph)
        if clamp_graded:
            val = np.clip(val, 0.0, 1.0)
        drive = val * ceiling * dt
        out_set[:n_in] = torch.as_tensor(drive, dtype=torch.float32, device=device)
        if pedestal_hz:
            out_set[:n_in] += pedestal_hz * dt
        net.step(out_set=out_set)
        if s >= skip:
            acc += float(net.out[tgt]) / dt
            cnt += 1
    return acc / max(cnt, 1)


def dsi(net, tgt, **kw):
    r = run(net, tgt, direction=+1, **kw)
    l = run(net, tgt, direction=-1, **kw)
    return (r - l) / max(r + l, 1e-12), r, l


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--phase-step", type=float, default=120.0,
                    help="spatial phase between neighbouring columns, deg. "
                         "5 deg column spacing on a 15 deg grating gives 120.")
    ap.add_argument("--delay-inh", type=float, default=80.0,
                    help="ms on the slow arm; config.T_DLY_SLOW is 80.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
    a = ap.parse_args()

    step = math.radians(a.phase_step)
    # Weights are SYNAPSE COUNTS, the units the graph stores. Calibration:
    # 30 -> 0 Hz, 60 -> 24 Hz, 120 -> 77 Hz with inputs at full tilt. The
    # measured 0.0018/0.0667 quoted elsewhere are REALISED conductances --
    # weight times rate -- and are not interchangeable with these.
    W_BAL = 120.0                    # comfortably above the firing threshold
    # T4a's real totals, measured from the graph: 46.8 excitatory, 41.5
    # inhibitory per cell. Both sit at the ~40-50 threshold for producing any
    # output at all, which is the finding this ladder exists to isolate.
    E_REAL, I_REAL = 46.8, 41.5

    rungs = [
        ("0 ideal",    dict(n_exc=1, n_inh=1, w_exc=W_BAL, w_inh=W_BAL,
                            spread=0, graded=False, pedestal=0.0)),
        ("1 realweight", dict(n_exc=1, n_inh=1, w_exc=E_REAL, w_inh=I_REAL,
                            spread=0, graded=False, pedestal=0.0)),
        ("2 gain x2",  dict(n_exc=1, n_inh=1, w_exc=2 * E_REAL, w_inh=2 * I_REAL,
                            spread=0, graded=False, pedestal=0.0)),
        ("3 fanin",    dict(n_exc=14, n_inh=12, w_exc=2 * E_REAL / 14,
                            w_inh=2 * I_REAL / 12,
                            spread=0, graded=False, pedestal=0.0)),
        ("4 spread",   dict(n_exc=14, n_inh=12, w_exc=2 * E_REAL / 14,
                            w_inh=2 * I_REAL / 12,
                            spread=1, graded=False, pedestal=0.0)),
        ("5 graded",   dict(n_exc=14, n_inh=12, w_exc=2 * E_REAL / 14,
                            w_inh=2 * I_REAL / 12,
                            spread=1, graded=True, pedestal=0.0)),
    ]

    print(paint("M3l — from a working correlator to T4, one property at a time", "1"))
    print(paint("=" * 78, "90"))
    print(f"tf {a.tf} Hz, phase step {a.phase_step:.0f} deg, inhibitory delay "
          f"{a.delay_inh:.0f} ms, {a.seconds:.0f} s per direction\n")
    print(f"  {'rung':<12}{'DSI':>9}{'R_pref':>10}{'R_null':>10}{'|R-L| Hz':>11}  note")
    print("  " + "-" * 74)

    record = {"tf": a.tf, "phase_step_deg": a.phase_step, "rungs": []}
    prev = None
    for name, cfg in rungs:
        rng = np.random.default_rng(0)
        ne, ni = cfg["n_exc"], cfg["n_inh"]
        if cfg["spread"]:
            # arm cells sample a spread of neighbouring columns, as they do in
            # the connectome, instead of all sitting at one point
            pe = rng.normal(0.0, step * 0.5, ne)
            pi = rng.normal(step, step * 0.5, ni)
        else:
            pe = np.zeros(ne)
            pi = np.full(ni, step)
        net, tgt = build(ne, ni, cfg["w_exc"], cfg["w_inh"],
                         2.0, a.delay_inh, a.device)
        d, r, l = dsi(net, tgt, n_exc=ne, n_inh=ni, phases_exc=pe,
                      phases_inh=pi, tf=a.tf, seconds=a.seconds,
                      device=a.device, clamp_graded=cfg["graded"],
                      pedestal_hz=cfg["pedestal"])
        drop = "" if prev is None else f"x{abs(d) / max(abs(prev), 1e-12):.2f}"
        note = "" if prev is None else (paint(f"{drop}  <-- COLLAPSE", "1;31")
                                        if abs(d) < 0.2 * abs(prev) else drop)
        print(f"  {name:<12}{d:>9.4f}{r:>10.2f}{l:>10.2f}{abs(r - l):>11.3f}  {note}")
        record["rungs"].append({"rung": name, "dsi": d, "r_pref": r,
                                "r_null": l, "abs_diff": abs(r - l), **{
                                    k: v for k, v in cfg.items()}})
        prev = d

    print(paint("""
  Rung 0 is the manuscript's hand-built control and must reproduce ~0.79.
  If it does not, the number in the paper is unsupported and the retraction
  built on it has to be revisited. If it does, read down the column for the
  first large drop: that property, added to a working correlator, is what
  destroys direction selectivity -- and unlike an ablation it cannot be
  attributed to whatever else was left connected.""", "90"))
    if a.json:
        import json
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
