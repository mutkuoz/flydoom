#!/usr/bin/env python3
"""M2 — proboscis extension response. The critical validation.

Reproduces the headline result of Shiu et al.: stimulating sugar-sensing
gustatory neurons drives the proboscis motor output.

READ THIS BEFORE INTERPRETING A PASS
------------------------------------
W_SYN is the model's single free parameter, and the paper fitted it *on this
very result* ("We chose W_syn such that activation of sugar GRNs at 100 Hz
resulted in roughly 80% of maximal MN9 firing"). We replicate that fit rather
than copying their number, because they fitted against FlyWire v630 and we are
on v783 with a different synapse pipeline.

So M2 passing does NOT show "the connectome predicts PER". The sugar arm is
true by construction. What M2 genuinely tests is everything else:

    * that a single scalar gain EXISTS which reproduces the behaviour at all
      -- a graph with flipped signs or a broken id map has no such scalar
    * the NEGATIVE CONTROL: bitter must not drive MN9
    * SUPPRESSION: sugar+bitter must be weaker than sugar alone
    * that the dose-response is graded rather than all-or-none

The negative controls are the real content. A too-hot gain fires MN9 on
anything, and only the controls catch it.

    python experiments/m2_per.py [--trials 30] [--quick] [--device cuda]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.cells import AnnotationTable  # noqa: E402
from flydoom.graph import ConnectomeGraph  # noqa: E402
from flydoom.lif import LIFNetwork, poisson_rate_vector  # noqa: E402
from flydoom.registry import by_name  # noqa: E402

# Fitted on this graph; see config.W_SYN_FITTED.

USE_COLOR = sys.stdout.isatty()


def paint(t: str, c: str) -> str:
    return f"\033[{c}m{t}\033[0m" if USE_COLOR else t


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((bool(ok), name, detail))
        return bool(ok)

    def render(self) -> bool:
        for ok, name, detail in self.rows:
            mark = paint("PASS", "32") if ok else paint("FAIL", "1;31")
            print(f"  {mark}  {name:<48} {paint(detail, '90')}")
        return all(ok for ok, _, _ in self.rows)


def respond(
    net: LIFNetwork,
    readout,
    stim: dict[str, tuple],
    duration: float,
    trials: int,
    seed0: int = 0,
) -> tuple[float, float]:
    """Mean +- sd readout rate over `trials` Poisson repetitions.

    stim maps a label to (indices, rate_hz); all entries are driven together.
    """
    n = net.n
    fr = torch.zeros(n, dtype=torch.float32, device=net.device)
    for idx, rate in stim.values():
        fr += poisson_rate_vector(n, idx, rate, device=net.device)

    out = []
    for t in range(trials):
        net.reset()
        net.gen.manual_seed(seed0 + t)
        res = net.run(duration, forced_rate=fr if fr.any() else None)
        out.append(net.rate_of(res, readout))
    sd = statistics.stdev(out) if len(out) > 1 else 0.0
    return statistics.fmean(out), sd


def calibrate_w_syn(
    net, sugar, mn9, duration, trials, target=0.80,
    stim_hz=100.0, sat_hz=200.0, lo=1e-6, hi=3e-2, iters=13,
) -> tuple[float, float]:
    """Replicate the paper's fit: find W_SYN putting sugar@stim_hz at
    `target` of the saturating MN9 response.

    The ratio rises monotonically with gain -- at low gain 100 Hz barely
    registers while 200 Hz does, at high gain both saturate -- so bisection is
    well posed.
    """
    def ratio(w: float) -> tuple[float, float, float]:
        net.p.w_syn = w
        r_stim, _ = respond(net, mn9, {"s": (sugar, stim_hz)}, duration, trials)
        r_sat, _ = respond(net, mn9, {"s": (sugar, sat_hz)}, duration, trials)
        return (r_stim / r_sat if r_sat > 1e-9 else 0.0), r_stim, r_sat

    print(f"  {'W_SYN (mV)':>11}  {'MN9@100Hz':>10}  {'MN9@200Hz':>10}  {'ratio':>7}")
    best = None
    for i in range(iters):
        mid = (lo * hi) ** 0.5           # geometric bisection: gain is scale-free
        r, a, b = ratio(mid)
        print(f"  {mid * 1e3:11.5f}  {a:10.2f}  {b:10.2f}  {r:7.3f}")
        if best is None or abs(r - target) < abs(best[1] - target):
            best = (mid, r)
        if r < target:
            lo = mid
        else:
            hi = mid
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M2 — sugar -> PER")
    ap.add_argument("--trials", type=int, default=30,
                    help="repetitions per condition (paper used 30)")
    ap.add_argument("--calib-trials", type=int, default=3)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true",
                    help="3 trials, skip the dose-response sweep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--w-syn", type=float, default=None,
                    help="skip calibration and use this value (volts)")
    ap.add_argument("--sweep", action="store_true",
                    help="map the W_SYN operating window and exit")
    args = ap.parse_args()
    if args.quick:
        args.trials = 3

    print(paint("flydoom M2 — sugar -> proboscis extension", "1"))
    print(paint("=" * 76, "90"))

    g = ConnectomeGraph.load()
    ann = AnnotationTable.load(config.RAW_DIR)
    net = LIFNetwork.from_graph(g, device=args.device, seed=0)

    def idx(handle):
        res = ann.resolve(by_name(handle))
        if not res.root_ids:
            raise SystemExit(f"handle {handle!r} did not resolve; run M0")
        return g.index_of(res.root_ids)

    sugar, bitter, mn9 = idx("sugar_GRN"), idx("bitter_GRN"), idx("MN9")
    print(f"device     {args.device}")
    print(f"network    {g.n_neurons:,} neurons, {g.n_edges:,} edges")
    print(f"sugar GRNs {len(sugar)}   bitter GRNs {len(bitter)}   "
          f"MN9 pool {len(mn9)}")
    print(f"protocol   {args.duration:.1f} s x {args.trials} trials, "
          f"Poisson drive")

    c = Checks()

    # ---------------- gain sensitivity sweep ----------------
    if args.sweep:
        print(f"\n{paint('W_SYN OPERATING WINDOW', '1;36')}")
        print(paint("  How much of the conclusion is a property of the connectome,"
                    "\n  and how much of one scalar? Anything that survives only a"
                    " narrow\n  band of gain is not a result.", "90"))
        print(f"\n  {'W_SYN(mV)':>10} {'MN9@sugar':>10} {'MN9@bitter':>11}"
              f" {'brain active':>13} {'peak Hz':>8}")
        for mult in (0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 6.0):
            net.p.w_syn = w0 = config.W_SYN_FITTED * mult
            su, _ = respond(net, mn9, {"s": (sugar, 100.0)}, args.duration, 3)
            bi, _ = respond(net, mn9, {"b": (bitter, 100.0)}, args.duration, 3)
            net.reset(); net.gen.manual_seed(0)
            fr = poisson_rate_vector(net.n, sugar, 100.0, device=net.device)
            res = net.run(args.duration, forced_rate=fr)
            frac = float((res["counts"] > 0).float().mean())
            peak = float(res["rates_hz"].max())
            flag = ""
            if su < 1: flag = paint("  SILENT", "31")
            elif frac > 0.25: flag = paint("  SEIZING", "31")
            elif bi > 0.25 * su: flag = paint("  control broken", "31")
            print(f"  {w0 * 1e3:10.5f} {su:10.2f} {bi:11.2f}"
                  f" {100 * frac:12.2f}% {peak:8.1f}{flag}")
        return 0

    # ---------------- calibration ----------------
    if args.w_syn is not None:
        w = args.w_syn
        print(f"\n{paint('CALIBRATION', '1;36')}  skipped, using {w * 1e3:.5f} mV")
        ratio = float("nan")
    else:
        print(f"\n{paint('CALIBRATION  (the paper''s own fit, replicated on v783)', '1;36')}")
        print(paint("  target: sugar@100 Hz = 80% of the saturating MN9 response",
                    "90"))
        w, ratio = calibrate_w_syn(
            net, sugar, mn9, args.duration, args.calib_trials,
            target=config.W_SYN_CALIBRATION["target_fraction_of_max"],
            stim_hz=config.W_SYN_CALIBRATION["stim_rate_hz"],
        )
        print(f"\n  fitted W_SYN = {paint(f'{w * 1e3:.5f} mV', '1')} per synapse"
              f"   (ratio {ratio:.3f})")
        print(f"  the paper's published value is {config.W_SYN_PAPER * 1e3:.5f} mV"
              f" (their v630 fit) -> ours is {w / config.W_SYN_PAPER:.2f}x that")
        peak = w * net.p.peak_psp_fraction()
        print(f"  one synapse moves the membrane {peak * 1e6:.1f} uV;"
              f" ~{net.p.threshold_distance / peak:.0f} coincident to fire")
        c.check(0.70 <= ratio <= 0.90, "calibration converged on the 80% target",
                f"ratio {ratio:.3f}")

    net.p.w_syn = w

    # ---------------- the conditions ----------------
    print(f"\n{paint('CONDITIONS', '1;36')}")
    print(f"  {'condition':<26} {'MN9 rate (Hz)':>16}")

    base, base_sd = respond(net, mn9, {}, args.duration, args.trials)
    print(f"  {'baseline (no stimulus)':<26} {base:10.2f} +- {base_sd:4.2f}")

    sug, sug_sd = respond(net, mn9, {"s": (sugar, 100.0)},
                          args.duration, args.trials)
    print(f"  {'sugar 100 Hz':<26} {sug:10.2f} +- {sug_sd:4.2f}")

    bit, bit_sd = respond(net, mn9, {"b": (bitter, 100.0)},
                          args.duration, args.trials)
    print(f"  {'bitter 100 Hz':<26} {bit:10.2f} +- {bit_sd:4.2f}")

    both, both_sd = respond(net, mn9, {"s": (sugar, 100.0), "b": (bitter, 100.0)},
                            args.duration, args.trials)
    print(f"  {'sugar + bitter':<26} {both:10.2f} +- {both_sd:4.2f}")

    # ---------------- dose-response ----------------
    curve = []
    if not args.quick:
        print(f"\n{paint('DOSE-RESPONSE  (sugar GRN rate -> MN9 rate)', '1;36')}")
        for hz in (0, 10, 25, 50, 100, 150, 200):
            r, sd = respond(net, mn9, {"s": (sugar, float(hz))},
                            args.duration, max(3, args.trials // 5))
            curve.append((hz, r))
            bar = "#" * int(40 * r / max(1e-9, sug * 1.3))
            print(f"  {hz:>4} Hz  {r:7.2f} +- {sd:4.2f}  {bar}")

    # ---------------- acceptance ----------------
    print(f"\n{paint('ACCEPTANCE', '1')}")

    c.check(base < 1.0, "silent with no stimulus",
            f"{base:.2f} Hz baseline")
    c.check(sug > 5.0, "SUGAR DRIVES MN9 (the PER result)",
            f"{sug:.2f} Hz")
    c.check(sug > 10 * max(base, 0.01), "sugar response is far above baseline",
            f"{sug / max(base, 0.01):.0f}x")

    # the controls that actually carry the weight
    c.check(bit < 0.25 * sug, "NEGATIVE CONTROL: bitter does not drive MN9",
            f"bitter {bit:.2f} vs sugar {sug:.2f} Hz")
    c.check(both < sug, "SUPPRESSION: sugar+bitter < sugar alone",
            f"{both:.2f} < {sug:.2f} Hz "
            f"({100 * (1 - both / max(sug, 1e-9)):.0f}% suppressed)")

    if curve:
        rates = [r for _, r in curve]
        monotone = all(b >= a - 0.5 for a, b in zip(rates, rates[1:]))
        c.check(monotone, "dose-response is monotonic", "no inversions")
        c.check(rates[1] < rates[-1] * 0.75,
                "response is graded, not all-or-none",
                f"10 Hz gives {rates[1]:.1f}, 200 Hz gives {rates[-1]:.1f} Hz")

    ok = c.render()
    print(f"\n{paint('=' * 76, '90')}")
    if ok:
        print(paint("VERDICT: M2 PASS", "1;32"))
        print(f"Fitted W_SYN = {w * 1e3:.5f} mV per synapse. Put this in config.py")
        print("as [FITTED] with today's date and the graph it was fitted against.")
        print(paint("Remember what this does and does not show — see the module "
                    "docstring.", "90"))
        return 0
    print(paint("VERDICT: M2 FAIL — stop. Do not touch ViZDoom.", "1;31"))
    print("Order of suspicion (spec 5): GLUT sign, W_SYN magnitude, id_map")
    print("off-by-one, pre/post swap. M1 pins three of those four already, so")
    print("look at gain and at the GRN/MN9 handles first.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
