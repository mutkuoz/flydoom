#!/usr/bin/env python3
"""M18 -- does the fly keep a dark bar in front of it?

WHY THIS ONE

Two visual behaviours have been tested here and both came back empty: direction
selectivity (M3) and looming (M4). They share a cause. Each is a computation on
the ORDER in which neighbouring ommatidia fire, and order in time is the one
thing this model reliably loses -- shunting inhibition cancels itself in a point
neuron, and the delay-and-correlate geometry survives only as a whisper.

Fixation does not need order. Working out which side an object is on needs
retinotopy, a spatial fact, and the retinotopy here is verified end to end (L1's
response phase advances at -20.0 deg/deg against an expected 20.0, and survives
to T4a). M7 already found the neural half of this: LC10a's left-right difference
tracks target azimuth. The question left is the behavioural one, which is the
one a fly laboratory would ask first.

It is also the behaviour the corrected behaviour tables hint at without
measuring. In the fly arena the intact model is stuck against walls MORE than
its mirrored control (+0.08, CI excludes zero) and takes more damage, and the
note offers one hypothesis for it: with correct geometry its vision steers it
toward large dark structure. In Doom that is a wall and it is maladaptive. Put
the same drive in a fly's arena and it has a name -- fixation -- and a sign.

THE ARENA (scripts/build_stripe_arena.py)

A 72-sided cylinder, radius 512, uniformly bright, with one dark bar 20 degrees
wide, and the fly at the centre. At constant distance Doom's light diminishing
is constant, so the bar is the only azimuthal feature in the world. Tethered:
the brain's forward and lateral commands are recorded and discarded, and only
its yaw reaches the body, so the bar's retinal position is a pure integral of
what the brain has commanded to steer. That is the torque meter, in software.

THE MEASURABLE

    az(t)      bar azimuth relative to gaze, 0 ahead, positive to the left
    F          mean cos(az): +1 bar always ahead, -1 always behind, 0 no
               preference. Goetz's fixation index, in the form that needs no
               binning.

WHAT WOULD COUNT

  intact    F > 0, bar held ahead
  mirrored  the retinal sampling grid is flipped, so every azimuth the brain
            reads is negated. A genuine fixation drive must turn AWAY by the
            same amount: F < 0, anti-fixation. This is the sharpest control in
            the project, because it predicts a SIGN and not an absence.
  frozen    retina held on one frame: F -> 0.
  blank     the same cylinder with no bar: F -> 0, which is the control for a
            measurement that could manufacture a peak from geometry alone.

THE NULL. Heading here is the cumulative sum of the turns the brain commanded,
so a fly that turns a lot with no regard for the bar still lands somewhere, and
per seed that somewhere is arbitrary. The null is built from the model's OWN
turn sequence: block-shuffle its per-tic heading increments (1 s blocks, which
keeps the autocorrelation) and start from a uniformly random heading. That is a
command-matched random agent, exactly matched, and it needs no second engine.

    python experiments/m18_stripe.py --seeds 20 --jobs 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

BAR_WORLD = (512.0, 0.0)      # the bar's centre, from build_stripe_arena
SPAWN_FREEZE = 12             # tics the engine ignores the player after spawn
BLOCK = 35                    # surrogate block length, one second


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


HALF_PERIOD = 70              # tics the bar is held on one side, 2 s
SETTLE = 15                   # tics dropped after each switch


def run_one(seed: int, tics: int, device: str, *, scenario: str = "stripe_fix",
            mirror: bool = False, blind: bool = False, free: bool = False,
            warmup: int = 150, optic_gain: float = 16.0,
            yaw_source: str = "DNp15", render: str = "fast",
            eye_map: str = "anatomical", hold_deg: float | None = None,
            sweep: float | None = None, head_max: float = 0.0,
            head_gain: float = 0.0) -> dict:
    """One tethered episode. Returns the heading trace and the commands.

    `hold_deg` opens the loop: instead of letting the brain steer, the body is
    driven to put the bar at +hold_deg and -hold_deg on alternating half
    periods, and the brain's yaw command is recorded and discarded. That is
    Goetz's torque meter with the position CLAMPED, and it measures the thing
    a closed loop can only infer -- the turn a given bar position asks for.
    The alternation is there because the motor decoder subtracts a 3 s
    baseline, which would remove any sustained one-sided response; 2 s per side
    sits well inside what survives it.

    `sweep` opens the loop the other way: the body turns at a constant
    `sweep` deg/tic, so the bar walks all the way round the eye and one
    episode yields the whole response curve instead of one point of it.
    Run it at +rate and -rate and average: a response to the rotation
    ITSELF is common to both and cancels, a response to bar POSITION does
    not.
    """
    from flydoom.agent import AgentConfig, FlyDoomAgent
    from flydoom.doom import DoomConfig
    from flydoom.motor import MotorConfig
    from m9_behaviour import RENDER

    agent = FlyDoomAgent(AgentConfig(
        doom=DoomConfig(scenario=scenario, window=False, seed=seed,
                        head_yaw_max=head_max, **RENDER[render]),
        motor=MotorConfig(yaw_source=yaw_source, fixed_turn_sign=True,
                          phasic_mdn=True, forward_gain=0.16,
                          head_gain=head_gain),
        eye_map=eye_map, seed=seed, optic_gain=optic_gain,
        spiking_t4=True, device=device,
    ))
    if mirror:
        agent.vision.mirror()
    agent.reset()

    if blind:
        raw, first = agent.doom.frame, {}

        def frozen():
            if "f" not in first:
                f = raw()
                first["f"] = None if f is None else np.array(f, copy=True)
            return first["f"]
        agent.doom.frame = frozen

    # Every episode starts from the same map thing, so without this every seed
    # would begin at the same heading and a motor system that merely drifts
    # would look like a population of flies agreeing about where to point.
    offset = float(np.random.default_rng(seed).uniform(0.0, 360.0))
    orig_step, tick, yaws = agent.doom.step, {"t": 0}, []

    phases = []

    def step(actions, skip):
        actions = list(actions)
        yaws.append(float(actions[0]))
        if not free:
            actions[1] = actions[2] = 0.0       # tethered: yaw only
        if sweep is not None:
            actions[0] = float(sweep)
        elif hold_deg is None:
            if tick["t"] == SPAWN_FREEZE:
                actions[0] += offset
        else:
            # Positive action turns the body clockwise, which DECREASES Doom's
            # ANGLE; the bar sits at world azimuth 0, so a bar azimuth of `want`
            # is the heading -want, and the correction is the wrapped
            # difference. Applied every tic, so the clamp holds against
            # whatever the brain just asked for.
            sign = 1.0 if (tick["t"] // HALF_PERIOD) % 2 == 0 else -1.0
            want = sign * hold_deg
            phases.append(sign)
            actions[0] = float(wrap180(agent.doom.pose()[2] + want))
        tick["t"] += 1
        return orig_step(actions, skip)
    agent.doom.step = step

    # The command is what the body does; the left-right difference of the
    # steering pair is what the brain SAYS, before the decoder's deadzone and
    # its baseline filter can remove it. A tethered fly's torque is the second.
    yl, yr = f"{yaw_source}_L", f"{yaw_source}_R"
    head, pos, dnp, neck = [], [], [], []
    try:
        for t in range(tics):
            rec = agent.tic(t)
            if rec is None:
                break
            dnp.append(float(rec.rates.get(yl, 0.0) - rec.rates.get(yr, 0.0)))
            x, y, a = agent.doom.pose()
            head.append(a)
            pos.append((x, y))
            neck.append(float(agent.vision.head_deg))
    finally:
        agent.close()

    head = np.asarray(head, float)
    px, py = np.asarray(pos, float).T if pos else (np.zeros(0), np.zeros(0))
    bearing = np.degrees(np.arctan2(BAR_WORLD[1] - py, BAR_WORLD[0] - px))
    # Where the bar is ON THE RETINA. With a neck the eyes lead the body, and
    # a head turned `neck` degrees to the left puts a bar that much further to
    # the right of straight ahead; with the head bolted on, `neck` is zero and
    # this is the body-relative azimuth as before.
    az = wrap180(bearing - head - np.asarray(neck, float))
    return {"seed": seed, "scenario": scenario, "mirror": mirror,
            "blind": blind, "free": free, "warmup": warmup,
            "hold_deg": hold_deg, "sweep": sweep, "head_max": head_max,
            "neck": neck[warmup:],
            "phase": phases[warmup:len(head)],
            "dnp15": dnp[warmup:],
            "az": az[warmup:].tolist(), "head": head[warmup:].tolist(),
            "yaw_cmd": yaws[warmup:len(head)], "n": int(len(az) - warmup)}


# -- statistics ----------------------------------------------------------

def fixation(az_deg: np.ndarray) -> float:
    r = np.radians(np.asarray(az_deg, float))
    return float(np.mean(np.cos(r))) if len(r) else 0.0


def surrogates(head: np.ndarray, n: int, rng) -> np.ndarray:
    """Block-shuffled heading increments from a random start. The turn
    statistics are the model's own; only when each turn happened is destroyed."""
    d = wrap180(np.diff(np.asarray(head, float)))
    blocks = [d[i:i + BLOCK] for i in range(0, len(d), BLOCK)]
    out = np.empty(n)
    for k in range(n):
        order = rng.permutation(len(blocks))
        h = np.concatenate([[rng.uniform(0, 360)],
                            np.concatenate([blocks[i] for i in order])]).cumsum()
        out[k] = fixation(wrap180(-h))
    return out


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else 0.0), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def summarise(shards: list[dict], n_sur: int = 200) -> dict:
    rng = np.random.default_rng(7)
    F, S, front, rear = [], [], [], []
    for sh in shards:
        az = np.asarray(sh["az"], float)
        if len(az) < BLOCK * 2:
            continue
        F.append(fixation(az))
        S.append(float(surrogates(np.asarray(sh["head"], float),
                                  n_sur, rng).mean()))
        front.append(float(np.mean(np.abs(az) < 30.0)))
        rear.append(float(np.mean(np.abs(az) > 150.0)))
    m, c = ci95(F)
    dm, dc = ci95(np.asarray(F) - np.asarray(S))
    return {"n": len(F), "F": m, "F_ci": c, "F_surrogate": float(np.mean(S)),
            "F_minus_surrogate": dm, "F_minus_surrogate_ci": dc,
            "front30": float(np.mean(front)), "rear150": float(np.mean(rear)),
            "per_seed": F}


# -- driver --------------------------------------------------------------

ARMS = {
    "intact":   dict(),
    "mirrored": dict(mirror=True),
    "frozen":   dict(blind=True),
    "blank":    dict(scenario="stripe_blank"),
    # The same fly with a neck: the steering command also drives a head that
    # reaches 20 degrees either side and recentres, so gaze can move without
    # the body. See DoomConfig.head_yaw_max.
    "head":     dict(head_max=20.0, head_gain=1.0),
}


def clamp_response(sh: dict) -> tuple[float, float]:
    """Open loop: commanded yaw with the bar on the left minus the same with
    it on the right, and the same for the steering pair's rate difference.

    Positive yaw turns the body clockwise (to its right), and a positive phase
    holds the bar to the LEFT, so a fly that turns TOWARD the bar returns a
    NEGATIVE first number."""
    ph = np.asarray(sh["phase"], float)
    y = np.asarray(sh["yaw_cmd"], float)[:len(ph)]
    d = np.asarray(sh["dnp15"], float)[:len(ph)]
    # Each switch jumps the bar across the midline, which is a large motion
    # step in one direction, and a motion response would then be counted as a
    # position response. Drop the SETTLE tics after every switch; what is left
    # is the brain looking at a stationary bar.
    keep = np.ones(len(ph), bool)
    switch = np.flatnonzero(np.diff(ph) != 0) + 1
    for k in np.concatenate([[0], switch]):
        keep[k:k + SETTLE] = False
    ph, y, d = ph[keep], y[keep], d[keep]
    if not len(ph) or not (ph > 0).any() or not (ph < 0).any():
        return 0.0, 0.0
    return (float(y[ph > 0].mean() - y[ph < 0].mean()),
            float(d[ph > 0].mean() - d[ph < 0].mean()))


CURVE_BINS = np.arange(-180.0, 181.0, 15.0)


def response_curve(shards):
    """Commanded yaw against where the bar was, averaged over episodes.

    Positive yaw turns the body to its right; positive azimuth puts the bar on
    its left. A fly that steers TOWARD the bar therefore gives a curve that
    falls through zero, and one that steers away a curve that rises.
    """
    centres = (CURVE_BINS[:-1] + CURVE_BINS[1:]) / 2
    acc = np.full((len(shards), len(centres)), np.nan)
    for i, sh in enumerate(shards):
        az = np.asarray(sh["az"], float)
        y = np.asarray(sh["yaw_cmd"], float)[:len(az)]
        idx = np.digitize(az, CURVE_BINS) - 1
        for b in range(len(centres)):
            m = idx == b
            if m.any():
                acc[i, b] = y[m].mean()
    mean = np.nanmean(acc, axis=0)
    n = np.sum(~np.isnan(acc), axis=0)
    sd = (np.nanstd(acc, axis=0, ddof=1) if len(shards) > 1
          else np.zeros(len(centres)))
    return centres, mean, 1.96 * sd / np.sqrt(np.maximum(n, 1))


def restoring_slope(shards, window: float = 90.0) -> float:
    """Slope of that curve across the frontal window, command degrees per
    degree of bar offset. Negative is a fly turning toward the bar."""
    c, m, _ = response_curve(shards)
    k = (np.abs(c) <= window) & ~np.isnan(m)
    if k.sum() < 3:
        return float("nan")
    return float(np.polyfit(c[k], m[k], 1)[0])


def summarise_clamp(shards: list[dict]) -> dict:
    dy = [clamp_response(sh)[0] for sh in shards]
    dd = [clamp_response(sh)[1] for sh in shards]
    m, c = ci95(dy)
    hm, hc = ci95(dd)
    return {"n": len(dy), "d_yaw": m, "d_yaw_ci": c,
            "d_rate": hm, "d_rate_ci": hc, "per_seed": dy}


def _job(a):
    arm, seed, tics, device, out, kw = a
    tag = "" if kw.get("hold_deg") is None else f"hold{kw['hold_deg']:g}_"
    if kw.get("sweep") is not None:
        tag = f"sweep{kw['sweep']:+g}_"
    shard = Path(out) / f"{tag}{arm}_{seed:03d}.json"
    if shard.exists():
        try:
            return arm, seed, json.loads(shard.read_text())
        except Exception:                                   # noqa: BLE001
            pass
    t0 = time.time()
    rec = run_one(seed, tics, device, **kw)
    rec["arm"] = arm
    shard.write_text(json.dumps(rec))
    if kw.get("sweep") is not None:
        score = f"slope={restoring_slope([rec]):+.4f}"
    elif kw.get("hold_deg") is None:
        score = f"F={fixation(np.asarray(rec['az'])):+.3f}"
    else:
        score = f"d_yaw={clamp_response(rec)[0]:+.4f}"
    print(f"  {arm:9s} seed {seed:3d}  {score}  {time.time() - t0:5.1f}s",
          flush=True)
    return arm, seed, rec


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M18 -- bar fixation")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=40)
    ap.add_argument("--tics", type=int, default=900)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=300,
                    help="tics dropped before the analysis window: the brain "
                         "settling, the spawn freeze, and the first approach")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--sweep", type=float, default=None, metavar="DEG_PER_TIC",
                    help="open the loop and turn the body at a constant rate, "
                         "so the bar circles the eye and one episode gives the "
                         "whole response curve. Each seed is run at +rate and "
                         "-rate so the response to the rotation itself cancels.")
    ap.add_argument("--hold", type=float, default=None, metavar="DEG",
                    help="open the loop: clamp the bar at +DEG and -DEG on "
                         "alternate 2 s half periods and measure the turn the "
                         "brain commands, instead of letting it steer")
    ap.add_argument("--free", action="store_true",
                    help="let the fly walk as well as turn")
    ap.add_argument("--out", type=Path,
                    default=Path("paper/data/m18_stripe"))
    ap.add_argument("--device", default=os.environ.get("FLYDOOM_DEVICE", "cpu"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_base + i for i in range(args.seeds)]
    rates = [args.sweep, -args.sweep] if args.sweep else [None]
    jobs = [(arm, s, args.tics, args.device, str(args.out),
             dict(ARMS[arm], free=args.free, warmup=args.warmup,
                  hold_deg=args.hold, sweep=r))
            for arm in args.arms for s in seeds for r in rates]
    print(f"M18 bar fixation: {len(jobs)} episodes, {args.tics} tics, "
          f"jobs {args.jobs}, device {args.device}", flush=True)

    shards: dict[str, list] = {a: [] for a in args.arms}
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for arm, _s, rec in ex.map(_job, jobs):
                shards[arm].append(rec)
    else:
        for j in jobs:
            arm, _s, rec = _job(j)
            shards[arm].append(rec)

    summary = {}
    if args.sweep is not None:
        print(f"\nbar swept round the eye at {args.sweep:g} deg/tic, both ways."
              f" Turning TOWARD the bar is a negative slope.")
        print(f"{'arm':<10}{'n':>3}{'slope':>10}   commanded yaw by bar azimuth")
        centres = (CURVE_BINS[:-1] + CURVE_BINS[1:]) / 2
        print(f"{'':<23}" + "".join(f"{c:+6.0f}" for c in centres[::2]))
        for arm in args.arms:
            rows = shards[arm]
            c, m, _ci = response_curve(rows)
            summary[arm] = {"n": len(rows), "slope": restoring_slope(rows),
                            "azimuth": c.tolist(), "yaw": m.tolist(),
                            "ci": _ci.tolist()}
            print(f"{arm:<10}{len(rows):>3}{summary[arm]['slope']:+10.4f}   "
                  + "".join(f"{v:+6.2f}" for v in m[::2]))
        name = f"summary_sweep{args.sweep:g}.json"
    elif args.hold is not None:
        print(f"\nbar clamped at +-{args.hold:g} deg. Turning toward it is a "
              f"NEGATIVE d_yaw.")
        print(f"{'arm':<10}{'n':>3}  {'d_yaw (deg/tic)':>22}"
              f"  {'d_rate (Hz, L-R)':>22}")
        for arm in args.arms:
            st = summarise_clamp(shards[arm])
            summary[arm] = st
            for k, tag in (("d_yaw", ""), ("d_rate", "")):
                st[k + "_sig"] = abs(st[k]) > st[k + "_ci"]
            print(f"{arm:<10}{st['n']:>3}  {st['d_yaw']:+12.4f} "
                  f"+-{st['d_yaw_ci']:7.4f}{'*' if st['d_yaw_sig'] else ' '}"
                  f"  {st['d_rate']:+12.3f} +-{st['d_rate_ci']:6.3f}"
                  f"{'*' if st['d_rate_sig'] else ' '}")
        name = f"summary_hold{args.hold:g}.json"
    else:
        print(f"\n{'arm':<10}{'n':>3}  {'F':>16}  {'F - surrogate':>17}"
              f"  {'front30':>8}{'rear150':>8}")
        for arm in args.arms:
            st = summarise(shards[arm])
            summary[arm] = st
            star = ("*" if abs(st["F_minus_surrogate"]) > st["F_minus_surrogate_ci"]
                    else " ")
            print(f"{arm:<10}{st['n']:>3}  {st['F']:+8.3f} +-{st['F_ci']:5.3f}  "
                  f"{st['F_minus_surrogate']:+8.3f} "
                  f"+-{st['F_minus_surrogate_ci']:5.3f}"
                  f"{star}  {st['front30']:8.3f}{st['rear150']:8.3f}")
        name = "summary.json"
    (args.out / name).write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
