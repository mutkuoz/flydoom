#!/usr/bin/env python3
"""M8 — does smell change what the brain tells the body to do?

The first positive non-taste result in this project, and the design is what
makes it trustworthy rather than the effect size.

THE PROBLEM WITH TESTING THIS IN A CLOSED LOOP
----------------------------------------------
M6 taught this the hard way: once two arms behave differently they stop seeing
the same world, and comparing them compares different stimulus distributions
rather than different brains. So this experiment runs OPEN LOOP -- the agent is
held still and Doom is stepped with a null action. Enemies still move, the
scene still evolves, but it evolves IDENTICALLY in both arms because the seed
is fixed and nothing the brain does feeds back.

Smell is then the only difference between the two runs, and `LC4` serves as the
built-in control: it is a visual cell, it should not move, and if it does the
comparison is broken.

WHY THIS PATHWAY AND NOT THE OTHER ONE
---------------------------------------
An earlier attempt routed odour at the pC1 aggression neurons and failed flat
-- 0.00 Hz across a 10x sweep of drive. Measured afterwards, the reason was
structural: DA1's contribution is a rounding error against pC1's +523/-548 of
balanced input, and pC1 is a bistable LATCH that has to be pushed over rather
than nudged.

The lateral horn is the opposite on every count. It receives 24% of the odour
relay's entire output, it is a feedforward innate-valence pathway with no latch,
and it sits ONE HOP from the descending neurons:

    lateral horn -> DNp01    -248 synapses   (inhibitory)
    lateral horn -> BPN       +46 synapses

The lesson worth keeping: a path EXISTING is not a path CARRYING. Check weight,
not hop count.

    python experiments/m8_olfactory_valence.py [--seeds 3] [--shuffled]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.agent import AgentConfig, FlyDoomAgent  # noqa: E402
from flydoom.doom import DoomConfig, DoomSession  # noqa: E402

USE_COLOR = sys.stdout.isatty()
WATCH = ("LHN", "DNp01", "DNa02", "BPN", "MDN", "LC4")


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


def run_arm(smell: bool, seed: int, tics: int, shuffled: bool, device: str):
    """One arm. Held still, so the visual scene is identical across arms."""
    cfg = AgentConfig(
        doom=DoomConfig(window=False, labels=True, seed=seed),
        smell=smell, seed=0, device=device,
    )
    ag = FlyDoomAgent(cfg)
    if shuffled:
        rng = np.random.default_rng(0)
        ag.graph.post_idx = ag.graph.post_idx[rng.permutation(len(ag.graph.post_idx))]
        ag.net = type(ag.net).from_graph(
            ag.graph, device=device, seed=0,
            edge_delay=ag.graph.edge_delay_steps(ag.ann, config.DT,
                                                 t_slow=cfg.slow_delay_s),
            graded=ag.graph.graded_mask(ag.ann),
        )
    pops = {k: torch.as_tensor(ag._idx(k).astype(np.int64), device=device)
            for k in WATCH}
    ag.reset()
    acc = {k: 0.0 for k in pops}
    n = 0
    still = [0.0] * len(DoomSession.BUTTONS)

    for t in range(tics):
        if ag.doom.finished:
            break
        frame = ag.doom.frame()
        if frame is None:
            break
        out_set, _ = ag.vision.drive(frame, ag.net.n,
                                     ag.net.p.graded_max_rate, config.DT)
        if ag.smell is not None:
            ag.smell.on_tic(ag.doom.threats())
        for _ in range(ag.substeps):
            drive = ag.intero.substep()
            forced = None
            if ag.smell is not None:
                drive = drive + ag.smell.substep()
                if ag.smell.active:
                    forced = (torch.rand(ag.net.n, generator=ag.net.gen,
                                         device=device)
                              < (drive * config.DT).clamp(0, 1))
            ag.net.step(out_set=out_set, forced=forced)
            ag.motor.observe(ag.net)
        if t > 40:                       # let the rate filter settle
            n += 1
            f = ag.motor._filt
            for k, v in pops.items():
                acc[k] += float(f[v].mean())
        ag.doom.step(still, 1)
    ag.close()
    return {k: acc[k] / max(n, 1) for k in pops}


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M8 — olfactory valence")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tics", type=int, default=320)
    ap.add_argument("--shuffled", action="store_true",
                    help="degree-preserving shuffle. VALID here, unlike in M6, "
                         "because the stimulus is open loop and identical.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(paint("flydoom M8 — olfactory valence", "1"))
    print(paint("=" * 74, "90"))
    if args.shuffled:
        print(paint("SHUFFLED CONNECTOME — control arm", "1;33"))
    print(f"open loop: agent held still, {args.seeds} seeds x {args.tics} tics\n")

    deltas = {k: [] for k in WATCH}
    for s in range(args.seeds):
        off = run_arm(False, 7 + s, args.tics, args.shuffled, args.device)
        on = run_arm(True, 7 + s, args.tics, args.shuffled, args.device)
        for k in WATCH:
            deltas[k].append(on[k] - off[k])
        print(f"  seed {7+s}:  LHN {off['LHN']:6.2f}->{on['LHN']:6.2f}   "
              f"DNp01 {off['DNp01']:6.1f}->{on['DNp01']:6.1f}")

    print(f"\n{paint('EFFECT OF SMELL', '1;36')}")
    print(f"  {'population':>11} {'mean change':>13} {'sd':>8}  consistent?")
    stats = {}
    for k in WATCH:
        d = np.asarray(deltas[k])
        same = bool(np.all(d > 0)) or bool(np.all(d < 0))
        stats[k] = (float(d.mean()), float(d.std()), same)
        print(f"  {k:>11} {d.mean():+13.2f} {d.std():8.2f}  "
              f"{'yes' if same else 'no'}")

    print(f"\n{paint('ACCEPTANCE', '1')}")
    c = Checks()
    lhn, dnp, lc4 = stats["LHN"], stats["DNp01"], stats["LC4"]

    c.check(abs(lc4[0]) < 1.0,
            "CONTROL: vision is unchanged between arms",
            f"LC4 moved {lc4[0]:+.2f} Hz")
    c.check(lhn[0] > 2.0 and lhn[2],
            "the lateral horn responds to smell",
            f"{lhn[0]:+.2f} Hz, consistent across seeds")
    c.check(abs(dnp[0]) > 2.0 and dnp[2],
            "SMELL REACHES THE DESCENDING NEURONS",
            f"DNp01 {dnp[0]:+.2f} Hz, consistent")
    c.check(dnp[0] < 0,
            "the sign matches the wiring",
            "lateral horn -> DNp01 is -248 synapses, so suppression")

    ok = c.render()
    print(f"\n{paint('=' * 74, '90')}")
    if ok:
        print(paint("VERDICT: M8 PASS", "1;32"))
        print("A sensory channel changes motor output through the frozen")
        print("connectome, via the anatomical route the biology names, with the")
        print("sign the wiring predicts. Nothing was fitted.")
        print(paint("\nTwo caveats that belong next to this result:\n"
                    "  * odour source distances come from Doom's label buffer, so\n"
                    "    we TOLD the fly that enemies exist. This is not evidence\n"
                    "    the connectome detects enemies.\n"
                    "  * 1,227 lateral horn neurons pooled together mixes many\n"
                    "    functional channels; 'the lateral horn responds' is\n"
                    "    coarser than it sounds.", "90"))
        return 0
    print(paint("VERDICT: M8 FAIL", "1;31"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
