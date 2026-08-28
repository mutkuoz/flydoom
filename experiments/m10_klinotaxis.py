#!/usr/bin/env python3
"""M10 — does the brain turn when the smell fades?

WHY THIS IS THE RIGHT TEST.
The two antennae are ~0.3 mm apart and this model gives them IDENTICAL drive
(flydoom/olfaction.py says so in its header, and it is biologically right: a
real fly cannot triangulate across that gap). So odour carries HOW CLOSE and
never WHICH WAY. An animal with no bearing can still find a source, by moving,
comparing the concentration against its own recent past, and turning when it
falls. That is klinotaxis, and it is what E. coli does.

The prediction is specific and falsifiable. Turn magnitude should correlate
NEGATIVELY with dC/dt: smell rising -> keep going; smell falling -> turn and
search. If that coupling is absent, the odour signal cannot steer regardless of
how strong it is -- and it IS strong, reaching the descending neurons at +78 Hz.

This matters beyond olfaction. Comparing a signal against its own past is the
same operation that fails in motion vision. If the coupling is absent here too,
one root cause explains both failures. If it is PRESENT, the two are
independent and the smell story needs a different explanation.

CONTROLS.
  shuffled   degree-preserving shuffle of the same graph. Any coupling that
             survives it comes from the stimulus or the body, not the wiring.
  lag scan   a real controller acts after a delay; reporting only the zero-lag
             correlation would miss it. We scan lags and report the best,
             which is also why the shuffled arm matters -- scanning lags gives
             noise more chances to look like signal.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.agent import AgentConfig, FlyDoomAgent  # noqa: E402
from flydoom.doom import DoomConfig  # noqa: E402


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5:
        return 0.0
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d else 0.0


def episode(seed, tics, shuffled, device, scenario):
    ag = FlyDoomAgent(AgentConfig(
        doom=DoomConfig(scenario=scenario, window=False, seed=seed, labels=True),
        smell=True, shuffle_graph=shuffled, seed=seed, device=device))
    conc, yaw, fwd = [], [], []
    try:
        def on_tic(a, rec):
            # concentration the RECEPTORS report, after adaptation -- this is
            # what the brain actually gets, not the raw distance
            conc.append(float(a.smell.drive["food"]) if a.smell else 0.0)
            # ViZDoom button names, not the internal channel names -- a
            # wrong key here yields a constant and a correlation of exactly
            # 0.0000, which is what the first run of this reported.
            yaw.append(float(rec.action.get("TURN_LEFT_RIGHT_DELTA", 0.0)))
            fwd.append(float(rec.action.get("MOVE_FORWARD_BACKWARD_DELTA", 0.0)))
            return True
        ag.run(tics, on_tic=on_tic)
    finally:
        ag.close()
    return np.asarray(conc), np.asarray(yaw), np.asarray(fwd)


def analyse(conc, yaw, max_lag):
    """Correlate |turn| against dC/dt over a scan of lags."""
    d = np.gradient(conc)
    turn = np.abs(yaw)
    best = (0.0, 0)
    curve = []
    for lag in range(0, max_lag + 1):
        # the turn happens `lag` tics AFTER the concentration change
        a = d[:len(d) - lag] if lag else d
        b = turn[lag:] if lag else turn
        r = pearson(a, b)
        curve.append((lag, r))
        if abs(r) > abs(best[0]):
            best = (r, lag)
    return best, curve, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--tics", type=int, default=500)
    ap.add_argument("--scenario", default="health_gathering_supreme")
    ap.add_argument("--max-lag", type=int, default=12)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--device", default=(os.environ.get("FLYDOOM_DEVICE")
                                         or "cpu"))
    a = ap.parse_args()

    print("\033[1mM10 — does the brain turn when the smell fades?\033[0m")
    print("=" * 74)
    print(f"{a.seeds} seeds x {a.tics} tics, {a.scenario}\n")
    print("  prediction: a klinotaxis controller turns MORE as dC/dt goes")
    print("  negative, so r(dC/dt, |turn|) should be NEGATIVE.\n")

    out = {"seeds": a.seeds, "tics": a.tics, "arms": {}}
    for arm, shuf in (("connectome", False), ("shuffled", True)):
        rs, lags, dyn = [], [], []
        for s in range(a.seeds):
            conc, yaw, fwd = episode(s, a.tics, shuf, a.device, a.scenario)
            if conc.std() < 1e-9:
                print(f"  {arm} seed {s}: odour never varied; skipped")
                continue
            if np.abs(yaw).std() < 1e-12:
                print(f"  {arm} seed {s}: TURN command is constant "
                      f"({yaw[0]:+.4f}); a constant cannot correlate. skipped")
                continue
            (r, lag), curve, d = analyse(conc, yaw, a.max_lag)
            rs.append(r); lags.append(lag)
            dyn.append((float(conc.mean()), float(conc.std()),
                        float(np.abs(d).mean())))
        if not rs:
            print(f"  {arm}: no usable episodes"); continue
        m = float(np.mean(rs)); sd = float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0
        sem = sd / math.sqrt(len(rs)) if len(rs) > 1 else float("inf")
        neg = sum(1 for x in rs if x < 0)
        print(f"  \033[1m{arm}\033[0m  n={len(rs)} episodes")
        print(f"     best r(dC/dt, |turn|) = {m:+.4f} +/- {1.96*sem:.4f}   "
              f"negative in {neg}/{len(rs)}")
        print(f"     best lag {np.mean(lags):.1f} tics    "
              f"odour mean {np.mean([x[0] for x in dyn]):.4f}, "
              f"|dC/dt| {np.mean([x[2] for x in dyn]):.5f}")
        out["arms"][arm] = {"r_mean": m, "ci95": 1.96 * sem, "n": len(rs),
                            "n_negative": neg, "per_episode": rs,
                            "mean_lag": float(np.mean(lags))}
    c, s_ = out["arms"].get("connectome"), out["arms"].get("shuffled")
    if c and s_:
        print(f"\n  connectome minus shuffled: {c['r_mean'] - s_['r_mean']:+.4f}")
        print("  A coupling that the shuffle also shows is not from the wiring.")
    print("""
  Read the sign first, then whether it beats the shuffle. If the connectome
  arm is not reliably negative, the brain does not turn in response to the
  odour falling -- and a bearing-free odour signal it cannot use that way
  cannot steer, however strong it is at the descending neurons.""")
    if a.json:
        import json
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=1))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
