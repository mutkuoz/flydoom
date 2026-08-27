#!/usr/bin/env python3
"""M1.5 — validate the LIF integrator against closed-form solutions.

This runs on one and two neurons, not the connectome. It answers: does our
discretisation actually solve the equations we think it does?

Crucially it is independent of whether the PARAMETER VALUES are biologically
right. If dt is too coarse, the refractory bookkeeping is off by one, or the
membrane update has the wrong coefficient, this catches it here -- twenty lines
of maths away from the connectome, instead of at M2 where every failure looks
like the same failure.

Four checks:
    1. subthreshold relaxation      v(t) toward V_rest + g
    2. single-spike PSP shape       the two-neuron test (spec's M1.75)
    3. f-I curve                    firing rate vs constant drive
    4. refractory ceiling           rate cannot exceed 1/t_refrac

    python experiments/m15_lif.py [--device cpu] [--plot]
"""

from __future__ import annotations

import argparse
import os
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from flydoom.lif import LIFNetwork, LIFParams  # noqa: E402

USE_COLOR = sys.stdout.isatty()


def paint(t: str, c: str) -> str:
    return f"\033[{c}m{t}\033[0m" if USE_COLOR else t


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((ok, name, detail))
        return ok

    def render(self) -> bool:
        for ok, name, detail in self.rows:
            mark = paint("PASS", "32") if ok else paint("FAIL", "1;31")
            print(f"  {mark}  {name:<46} {paint(detail, '90')}")
        return all(ok for ok, _, _ in self.rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M1.5 — LIF integrator")
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
    args = ap.parse_args()
    dev = args.device
    p = LIFParams()
    c = Checks()

    print(paint("flydoom M1.5 — LIF integrator validation", "1"))
    print(paint("=" * 74, "90"))
    print(f"device     {dev}")
    print(f"dt         {p.dt * 1e3:.2f} ms      tau_mem {p.tau_mem * 1e3:.1f} ms"
          f"     tau_syn {p.tau_syn * 1e3:.1f} ms")
    print(f"V_rest     {p.v_rest * 1e3:.1f} mV   V_thresh {p.v_thresh * 1e3:.1f} mV"
          f"   refrac {p.t_refrac * 1e3:.1f} ms")
    print(f"threshold distance  {p.threshold_distance * 1e3:.2f} mV"
          f"   (rheobase g)")
    print(paint("  all six constants are still [UNVERIFIED] — the checks below "
                "validate the\n  INTEGRATOR, not the values", "90"))

    # ---------------- 1. subthreshold relaxation ----------------
    print(f"\n{paint('1. SUBTHRESHOLD RELAXATION', '1;36')}")
    g_sub = 0.5 * p.threshold_distance          # safely below rheobase
    net = LIFNetwork(1, params=p, device=dev)
    tau_meas = None
    v_inf = p.v_rest + g_sub
    target = p.v_rest + (1 - math.exp(-1.0)) * g_sub   # v at t = tau_mem
    for i in range(int(round(10 * p.tau_mem / p.dt))):
        net.step(g_ext=g_sub)
        if i + 1 == int(round(p.tau_mem / p.dt)):
            v_at_tau = float(net.v[0])
    v_final = float(net.v[0])
    print(f"  v(inf)  predicted {v_inf * 1e3:8.4f} mV   got {v_final * 1e3:8.4f} mV")
    print(f"  v(tau)  predicted {target * 1e3:8.4f} mV   got {v_at_tau * 1e3:8.4f} mV")
    c.check(abs(v_final - v_inf) / abs(g_sub) < 1e-3, "steady state = V_rest + g",
            f"err {abs(v_final - v_inf) * 1e6:.3f} uV after 10 tau")
    c.check(abs(v_at_tau - target) / abs(g_sub) < 0.02,
            "reaches 63% of g in one tau_mem",
            f"err {abs(v_at_tau - target) / abs(g_sub) * 100:.2f}%")
    c.check(not bool(net.spiked[0]), "no spike below rheobase",
            f"g = {g_sub * 1e3:.2f} mV < {p.rheobase() * 1e3:.2f} mV")

    # ---------------- 2. single-spike PSP (two neurons) ----------------
    print(f"\n{paint('2. SINGLE-SPIKE PSP  (neuron 0 -> neuron 1)', '1;36')}")
    syn_count = 10.0
    w = p.w_syn * syn_count
    net = LIFNetwork(
        2,
        pre_idx=torch.tensor([0]), post_idx=torch.tensor([1]),
        signed_syn=torch.tensor([syn_count]),
        params=p, device=dev,
    )
    forced = torch.tensor([True, False], device=dev)
    net.step(forced=forced)                       # neuron 0 fires at t=0
    for _ in range(p.delay_steps - 1):            # T_dly conduction delay
        net.step()
    trace = []
    for _ in range(int(round(0.1 / p.dt))):
        net.step()
        trace.append(float(net.v[1]) - p.v_rest)

    meas_peak = max(trace)
    meas_peak_t = (trace.index(meas_peak) + 1) * p.dt
    pred_peak = w * p.peak_psp_fraction()
    pred_peak_t = p.psp_peak_time()

    print(f"  synapse count {syn_count:.0f}   w = W_SYN * count = {w * 1e3:.4f} mV")
    print(f"  conduction delay {p.t_dly * 1e3:.1f} ms -> {p.delay_steps} steps"
          f" = {p.t_dly_effective * 1e3:.1f} ms (skipped before sampling)")
    print(f"  peak PSP    predicted {pred_peak * 1e3:8.5f} mV"
          f"   got {meas_peak * 1e3:8.5f} mV")
    print(f"  peak time   predicted {pred_peak_t * 1e3:8.2f} ms"
          f"   got {meas_peak_t * 1e3:8.2f} ms")
    print(paint(f"  NOTE: peak PSP is only {p.peak_psp_fraction() * 100:.1f}% of w."
                " W_SYN is not the voltage a", "33"))
    print(paint("  synapse delivers -- tau_syn < tau_mem means v never reaches"
                " V_rest + g.", "33"))
    c.check(abs(meas_peak - pred_peak) / pred_peak < 0.03,
            "PSP peak amplitude matches closed form",
            f"err {abs(meas_peak - pred_peak) / pred_peak * 100:.2f}%")
    c.check(abs(meas_peak_t - pred_peak_t) < 3 * p.dt,
            "PSP peak time matches closed form",
            f"err {abs(meas_peak_t - pred_peak_t) * 1e3:.2f} ms")

    # sample the whole shape, not just the peak
    worst = 0.0
    for i, meas in enumerate(trace):
        t = (i + 1) * p.dt
        pred = p.analytic_psp(w, t)
        if pred > 0.05 * pred_peak:
            worst = max(worst, abs(meas - pred) / pred)
    c.check(worst < 0.05, "PSP full time course matches closed form",
            f"worst {worst * 100:.2f}%")

    # ---------------- 3. f-I curve ----------------
    print(f"\n{paint('3. f-I CURVE', '1;36')}")
    print(f"  {'g (mV)':>9}  {'predicted Hz':>13}  {'measured Hz':>12}  {'err':>7}")
    worst_fi = 0.0
    net = LIFNetwork(1, params=p, device=dev)
    for mult in (1.05, 1.2, 1.5, 2.0, 3.0, 5.0):
        g = mult * p.threshold_distance
        net.reset()
        res = net.run(2.0, g_ext=g)
        meas = float(res["rates_hz"][0])
        pred = p.analytic_rate(g)
        err = abs(meas - pred) / pred if pred else 0.0
        worst_fi = max(worst_fi, err)
        print(f"  {g * 1e3:9.3f}  {pred:13.2f}  {meas:12.2f}  {err * 100:6.2f}%")
    c.check(worst_fi < 0.05, "f-I curve matches closed form",
            f"worst {worst_fi * 100:.2f}% over 1.05-5x rheobase")

    # just below rheobase must be silent
    net.reset()
    res = net.run(1.0, g_ext=0.99 * p.threshold_distance)
    c.check(float(res["rates_hz"][0]) == 0.0, "silent just below rheobase",
            "g = 0.99 x threshold distance")

    # ---------------- 4. refractory ceiling ----------------
    print(f"\n{paint('4. REFRACTORY CEILING', '1;36')}")
    if p.refrac_quantisation_error > 1e-9:
        print(paint(f"  WARNING: dt does not divide t_refrac. "
                    f"{p.t_refrac*1e3:.2f} ms / {p.dt*1e3:.2f} ms = "
                    f"{p.t_refrac/p.dt:.2f} steps", "33"))
        print(paint(f"  -> quantises to {p.refrac_steps} steps = "
                    f"{p.t_refrac_effective*1e3:.2f} ms, "
                    f"{p.refrac_quantisation_error*100:.0f}% short. Raises the "
                    f"rate ceiling", "33"))
        print(paint(f"     from {1/(p.t_refrac+p.dt):.0f} Hz to "
                    f"{p.max_rate:.0f} Hz. Only bites where something saturates.",
                    "33"))
    net.reset()
    res = net.run(1.0, g_ext=1000 * p.threshold_distance)
    meas = float(res["rates_hz"][0])
    ceiling = p.max_rate
    print(f"  huge drive    measured {meas:.1f} Hz   ceiling {ceiling:.1f} Hz"
          f"   (1/(t_refrac_effective+dt))")
    c.check(meas <= ceiling * 1.01, "rate cannot exceed the refractory ceiling",
            f"{meas:.1f} <= {ceiling:.1f} Hz")
    c.check(meas > 0.5 * ceiling, "saturates near the ceiling under huge drive",
            f"{meas / ceiling * 100:.0f}% of ceiling")

    # forced spikes must respect refractoriness too
    net = LIFNetwork(1, params=p, device=dev, seed=0)
    rate = torch.tensor([5000.0], device=dev)
    res = net.run(1.0, forced_rate=rate)
    meas = float(res["rates_hz"][0])
    c.check(meas <= ceiling * 1.01, "forced input also respects refractoriness",
            f"5000 Hz requested -> {meas:.1f} Hz")

    print(f"\n{paint('ACCEPTANCE', '1')}")
    ok = c.render()
    print(f"\n{paint('=' * 74, '90')}")
    if ok:
        print(paint("VERDICT: M1.5 PASS — the integrator solves the equations "
                    "we think it does.", "1;32"))
        print(paint("Remaining risk is the parameter VALUES, not the maths.", "90"))
        return 0
    print(paint("VERDICT: M1.5 FAIL — fix the integrator before M2.", "1;31"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
