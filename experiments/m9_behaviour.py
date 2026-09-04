#!/usr/bin/env python3
"""M9 — closed-loop behaviour, measured against baselines.

M5 asked whether the loop is STABLE: does it run, does it chatter, does it
spin. That is a necessary check and it says nothing about whether the agent
does anything. This asks the next question -- is the behaviour distinguishable
from an agent with no brain in it at all?

Four arms, identical scenarios and seeds:

    connectome   the frozen graph
    shuffled     degree-preserving shuffle of the same graph
    random       actions drawn to match the connectome arm's own command
                 distribution, so it moves just as much and just as fast
    still        the null action

`random` is the arm that matters. A brain that produces motion is not thereby
producing BEHAVIOUR: wandering at the same speed in the same map collects
roughly the same medkits by accident. Matching its command statistics to the
connectome's is what makes the comparison about structure rather than energy.

Matching the MEAN and S.D. alone is not enough, and getting it wrong quietly
rigs the comparison. The decoder filters its rates at tau = 80 ms, so the
connectome's commands are smooth; i.i.d. draws with the same mean and s.d.
jitter at tic rate, turn far more erratically, and cover less ground for the
same nominal speed. Beating THAT is not evidence of anything. So the random arm
is an AR(1) process matched on mean, s.d. AND lag-1 autocorrelation, per
channel and per scenario -- same speed, same smoothness, no structure.

Read the closed-loop caveat before using these numbers to attribute anything
to connectivity: the four arms behave differently, so they do not sample the
same stimulus distribution, and `shuffled` here is a BEHAVIOURAL reference
rather than a control on the wiring. Section "shuffle controls are invalid in
closed loop" of the write-up spells this out; open-loop M8 is where the
attribution claim lives.

    python experiments/m9_behaviour.py
    python experiments/m9_behaviour.py --scenarios deathmatch --seeds 5
    python experiments/m9_behaviour.py --json paper/data/m9.json
"""

from __future__ import annotations

import argparse
import os
import datetime as _dt
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flydoom import config  # noqa: E402
from flydoom.agent import AgentConfig, FlyDoomAgent  # noqa: E402
from flydoom.doom import DoomConfig, DoomSession  # noqa: E402
from flydoom.motor import MotorConfig  # noqa: E402

USE_COLOR = sys.stdout.isatty()

SCENARIOS = ("deathmatch", "health_gathering_supreme", "defend_the_center")
ARMS = ("connectome", "shuffled", "random", "still")

# Grid cell size in Doom map units for the coverage metric. 64 is one standard
# floor tile, so "cells visited" reads as "tiles stepped on".
TILE = 64.0


def _git_sha() -> str:
    """Working-tree revision, with a dirty flag. Stamped into every result
    file so a number can always be traced back to the code that made it."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=Path(__file__).parent.parent,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True,
                               cwd=Path(__file__).parent.parent,
                               timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def paint(t: str, c: str) -> str:
    return f"\033[{c}m{t}\033[0m" if USE_COLOR else t


class Episode:
    """Per-tic bookkeeping for one run, reduced to metrics at the end."""

    def __init__(self) -> None:
        self.pos: list[tuple[float, float]] = []
        self.eye_bias: list[float] = []
        self.health: list[float] = []
        self.kills: list[float] = []
        self.healed = 0.0
        self.damage = 0.0
        self.act: dict[str, list[float]] = {}

    def record(self, session, result, action) -> None:
        vzd = session.vzd
        g = session.game
        self.pos.append((g.get_game_variable(vzd.GameVariable.POSITION_X),
                         g.get_game_variable(vzd.GameVariable.POSITION_Y)))
        self.health.append(result["health"])
        self.kills.append(g.get_game_variable(vzd.GameVariable.KILLCOUNT))
        self.healed += result["healed"]
        self.damage += result["damage_taken"]
        for k, v in action.items():
            self.act.setdefault(k, []).append(float(v))

    def record_vision(self, bias: float) -> None:
        """Left-minus-right mean column luminance for this frame."""
        self.eye_bias.append(float(bias))

    def metrics(self) -> dict:
        if len(self.pos) < 2:
            return {"tics": len(self.pos)}
        p = np.asarray(self.pos)
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        tiles = {(int(x // TILE), int(y // TILE)) for x, y in p}
        yaw = np.asarray(self.act.get("TURN_LEFT_RIGHT_DELTA", [0.0]))
        fwd = np.asarray(self.act.get("MOVE_FORWARD_BACKWARD_DELTA", [0.0]))
        lat = np.asarray(self.act.get("MOVE_LEFT_RIGHT_DELTA", [0.0]))
        atk = np.asarray(self.act.get("ATTACK", [0.0]))
        use = np.asarray(self.act.get("USE", [0.0]))
        return {
            "tics": len(self.pos),
            "path": float(step.sum()),
            "net_displacement": float(np.linalg.norm(p[-1] - p[0])),
            "tiles_visited": len(tiles),
            "health_end": float(self.health[-1]),
            "healed": float(self.healed),
            "damage": float(self.damage),
            "kills": float(self.kills[-1]),
            # Exploration efficiency: a spinner racks up path and no tiles.
            "tiles_per_1k_path": float(
                len(tiles) / max(step.sum(), 1.0) * 1000.0),
            "yaw_abs_mean": float(np.abs(yaw).mean()),
            "yaw_clip_frac": float(np.mean(np.abs(yaw) >= 11.99)),
            "fwd_abs_mean": float(np.abs(fwd).mean()),
            "fwd_clip_frac": float(np.mean(np.abs(fwd) >= 21.99)),
            "lat_clip_frac": float(np.mean(np.abs(lat) >= 19.99)),
            "attack_frac": float(atk.mean()),
            "use_frac": float(use.mean()),
            "spin": float(abs(np.mean(np.sign(yaw[12:])))) if len(yaw) > 13
            else 0.0,
            "yaw_chatter": _chatter(yaw),
            "attack_chatter": _chatter(atk),
            **self._wall_metrics(step, fwd),
            **self._vision_metrics(yaw),
        }

    # -- does it avoid walls? --------------------------------------------
    #
    # A real fly does, using optic-flow expansion -- the same computation as
    # direction selectivity (M3) and looming (M4). So this is not a new
    # question so much as the behavioural read-out of those two, and it is
    # worth measuring rather than inferring.
    #
    # "Stuck" = the agent commanded a real walk and did not move. In Doom that
    # means it is pressed against geometry. An agent that sees walls coming
    # steers before this happens; one that does not, scrapes along them.

    STUCK_CMD = 4.0      # map units/tic of commanded walk, above the noise
    STUCK_MOVE = 1.5     # map units/tic actually achieved

    def _wall_metrics(self, step, fwd) -> dict:
        f = np.abs(np.asarray(fwd)[1:])
        m = np.asarray(step)
        if not len(m):
            return {}
        pushing = f >= self.STUCK_CMD
        stuck = pushing & (m < self.STUCK_MOVE)
        n_push = int(pushing.sum())
        # a collision is an ENTRY into the stuck state, not each tic of it
        entries = int(np.sum(stuck[1:] & ~stuck[:-1])) + int(stuck[:1].sum())
        return {
            "stuck_frac": float(stuck.sum() / n_push) if n_push else 0.0,
            "collisions": float(entries),
            "collisions_per_1k_tics": float(entries / max(len(m), 1) * 1000.0),
            # how far it gets between collisions, in tics
            "free_run_tics": float(len(m) / entries) if entries else float(len(m)),
        }

    # -- is the steering coupled to the eyes at all? ----------------------
    #
    # The direct test, and the one that says WHY rather than WHETHER. If the
    # agent steers with its eyes, the left-minus-right luminance falling on
    # the two retinas has to predict the turn command. A real fly's optomotor
    # response makes this correlation large and signed. Near zero means the
    # turn is being produced by something other than the scene.

    def _vision_metrics(self, yaw) -> dict:
        b = np.asarray(self.eye_bias, dtype=float)
        y = np.asarray(yaw, dtype=float)
        k = min(len(b), len(y))
        if k < 30:
            return {}
        b, y = b[:k], y[:k]
        if b.std() < 1e-12 or y.std() < 1e-12:
            return {"vision_steer_r": 0.0}
        out = {"vision_steer_r": float(np.corrcoef(b, y)[0, 1])}
        # the eyes lead the command by the rate-filter delay, so scan the lag
        best, lag = 0.0, 0
        for L in range(0, 13):
            bb, yy = b[:k - L], y[L:]
            if bb.std() < 1e-12 or yy.std() < 1e-12:
                continue
            r = float(np.corrcoef(bb, yy)[0, 1])
            if abs(r) > abs(best):
                best, lag = r, L
        out["vision_steer_r_best"] = best
        out["vision_steer_lag_tics"] = float(lag)
        return out


def _eye_bias(agent) -> float:
    """Mean column luminance, left eye minus right eye, this frame."""
    lum = agent.last_luminance.detach().cpu().numpy()
    out, off = [], 0
    for _, eye in agent.retina.eyes.items():
        k = eye.neuron_idx.size
        out.append(float(lum[off:off + k].mean()) if k else 0.0)
        off += k
    return out[0] - out[1] if len(out) == 2 else 0.0


def _chatter(x) -> float:
    x = np.asarray(x)
    if len(x) < 2:
        return 0.0
    return float(np.mean((x[:-1] > 1e-6) != (x[1:] > 1e-6)))


def run_agent(scenario: str, seed: int, tics: int, shuffled: bool,
              device: str, bias_mv: float = 0.0,
              smell: bool = False,
              tau_baseline: float | None = None,
              optic_gain: float = 1.0,
              spiking_t4: bool = False,
              motor_kw: dict | None = None,
              mirror: bool = False,
              blind: bool = False) -> tuple[dict, dict]:
    """One connectome (or shuffled-connectome) episode.

    Returns (metrics, command distribution) -- the latter feeds the random arm.
    """
    mk = dict(motor_kw or {})
    if tau_baseline is not None:
        mk["tau_baseline"] = tau_baseline
    agent = FlyDoomAgent(AgentConfig(
        doom=DoomConfig(scenario=scenario, window=False, seed=seed,
                        labels=smell),
        motor=MotorConfig(**mk),
        smell=smell,
        shuffle_graph=shuffled,
        seed=seed,
        bias_mv=bias_mv,
        optic_gain=optic_gain,
        spiking_t4=spiking_t4,
        device=device,
    ))
    # ---- controls that should DESTROY a vision-driven effect ----------
    #
    # `mirror` flips the retinal sampling grid horizontally, so every column
    # looks at the mirror image of where it used to. Rigid horizontal motion of
    # the world then sweeps the retina the OTHER way, which reverses the sign
    # of every horizontal optic-flow signal while leaving firing rates,
    # contrast statistics and the wiring untouched. Genuine optomotor steering
    # must change; a collision advantage that survives it is not steering by
    # optic flow.
    #
    # `blind` freezes the first frame, so the retina sees a constant scene for
    # the whole episode. Any behavioural advantage that survives this is not
    # coming from the visual input at all.
    if mirror:
        agent.vision.grid[..., 0] = -agent.vision.grid[..., 0]
    if blind:
        _first = {}

        def _frozen():
            # copy: ViZDoom hands back a view onto its own buffer and
            # overwrites it in place, so keeping the reference would silently
            # give a LIVE frame and quietly destroy the control.
            if "f" not in _first:
                raw = agent.doom.__class__.frame(agent.doom)
                _first["f"] = None if raw is None else np.array(raw, copy=True)
            return _first["f"]
        agent.doom.frame = _frozen

    ep = Episode()
    try:
        def on_tic(ag, rec):
            ep.record(ag.doom, {"health": rec.health, "healed": rec.healed,
                                "damage_taken": rec.damage},
                      rec.action)
            ep.record_vision(_eye_bias(ag))
            return True
        agent.run(tics, on_tic=on_tic)
    finally:
        agent.close()
    m = ep.metrics()
    dist = {k: _summarise(v) for k, v in ep.act.items()}
    return m, dist


def _summarise(v) -> dict:
    """Mean, s.d. and lag-1 autocorrelation of one command channel."""
    x = np.asarray(v, dtype=float)
    mu, sd = float(x.mean()), float(x.std())
    if len(x) > 2 and sd > 1e-9:
        d = x - mu
        rho = float(np.dot(d[:-1], d[1:]) / np.dot(d, d))
        rho = float(np.clip(rho, 0.0, 0.995))
    else:
        rho = 0.0
    return {"mean": mu, "sd": sd, "rho": rho}


def run_scripted(scenario: str, seed: int, tics: int, dist: dict | None,
                 rng: np.random.Generator) -> dict:
    """The `random` and `still` arms. `dist=None` gives the null action.

    Random commands follow the AR(1) process carrying the connectome arm's own
    per-channel mean, s.d. and lag-1 autocorrelation:

        x <- mu + rho * (x - mu) + sqrt(1 - rho^2) * sd * N(0, 1)

    whose stationary distribution is exactly N(mu, sd) and whose lag-1
    correlation is exactly rho. This arm therefore moves as fast as the
    connectome arm and turns as smoothly, and differs from it only in whether
    the commands are about anything.
    """
    session = DoomSession(DoomConfig(scenario=scenario, window=False,
                                     seed=seed))
    session.new_episode()
    ep = Episode()
    clamp = {"TURN_LEFT_RIGHT_DELTA": MotorConfig.yaw_max_deg,
             "MOVE_FORWARD_BACKWARD_DELTA": MotorConfig.forward_max,
             "MOVE_LEFT_RIGHT_DELTA": MotorConfig.lateral_max}
    state: dict[str, float] = {}
    try:
        for _ in range(tics):
            if session.finished:
                break
            action = {}
            for b in DoomSession.BUTTONS:
                if dist is None or b not in dist:
                    action[b] = 0.0
                    continue
                d = dist[b]
                if b not in clamp:
                    # binary buttons: reproduce the duty cycle, not the mean
                    action[b] = float(rng.random() < d["mean"])
                    continue
                rho, sd, mu = d["rho"], d["sd"], d["mean"]
                state[b] = (mu + rho * (state.get(b, mu) - mu)
                            + math.sqrt(max(1.0 - rho * rho, 0.0)) * sd
                            * rng.standard_normal())
                action[b] = float(np.clip(state[b], -clamp[b], clamp[b]))
            result = session.step(
                [action[b] for b in DoomSession.BUTTONS], 1)
            ep.record(session, result, action)
            if result["done"]:
                break
    finally:
        session.close()
    return ep.metrics()


REPORT = [
    ("tiles_visited", "tiles visited", "{:.1f}"),
    ("path", "path (map units)", "{:.0f}"),
    ("tiles_per_1k_path", "tiles / 1k path", "{:.2f}"),
    ("healed", "health picked up", "{:.1f}"),
    ("damage", "damage taken", "{:.1f}"),
    ("health_end", "final health", "{:.1f}"),
    ("kills", "kills", "{:.2f}"),
    ("tics", "tics survived", "{:.0f}"),
    ("collisions_per_1k_tics", "wall hits / 1k tics", "{:.1f}"),
    ("stuck_frac", "stuck when walking", "{:.1%}"),
    ("free_run_tics", "tics between hits", "{:.1f}"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M9 — behaviour")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="First seed. Lets one sweep be sharded across "
                         "processes; the sim is CPU-bound at ~1.3 cores, "
                         "so shards scale near-linearly to nproc.")
    ap.add_argument("--tics", type=int, default=700)
    ap.add_argument("--tau-baseline", type=float, default=None,
                    help="steering baseline time constant, seconds. The "
                         "default 3 s removes the standing L/R asymmetry, but "
                         "an approach to food takes ~1.5 s, so it also eats "
                         "~40%% of the odour distance signal. Raising it keeps "
                         "slow approach signals at the cost of re-admitting "
                         "the reconstruction artefact.")
    ap.add_argument("--smell", action="store_true",
                    help="enable the odour channel. Health pickups are real "
                         "odour sources (ORN_DM1, vinegar), so this is the "
                         "arm to run before claiming the model cannot find "
                         "them. Note what it CANNOT supply: left and right "
                         "get bit-identical drive, so smell carries proximity "
                         "and never bearing.")
    ap.add_argument("--bias", type=float, default=0.0,
                    help="tonic optic-lobe drive, mV. Releases the lobula "
                         "columnar readouts (LPLC2 is silent at 0) but also "
                         "lifts MDN, and BPN-MDN is the walk command -- see "
                         "the sweep in the write-up.")
    ap.add_argument("--optic-gain", type=float, default=1.0,
                    help="synaptic gain onto the visual populations. The "
                         "replacement for --bias; at 1.0 with bias 0 the "
                         "optic lobe is sub-threshold on its own inputs.")
    ap.add_argument("--spiking-t4", action="store_true",
                    help="restore the spike threshold on T4/T5. A graded unit "
                         "is a clamped linear ramp; a correlator needs a "
                         "nonlinearity (10.6x, m3m).")
    ap.add_argument("--mirror", action="store_true",
                    help="CONTROL: flip the retinal sampling grid left-right. "
                         "Reverses the sign of horizontal optic flow without "
                         "changing rates or wiring, so any genuine optomotor "
                         "steering must change.")
    ap.add_argument("--blind", action="store_true",
                    help="CONTROL: freeze the first frame. The retina sees a "
                         "constant scene; anything that survives is not "
                         "visual.")
    ap.add_argument("--yaw-source", default="DNa02",
                    choices=["DNa02", "DNp15"],
                    help="which bilateral descending pair supplies yaw. "
                         "DNa02 is the goal-directed walking steering neuron, "
                         "2.3%% visual input here; DNp15 (DNHS1) is the "
                         "optomotor one the horizontal system drives.")
    ap.add_argument("--yaw-gain", type=float, default=None)
    ap.add_argument("--forward-gain", type=float, default=None)
    ap.add_argument("--lateral-gain", type=float, default=None)
    ap.add_argument("--deadzone-hz", type=float, default=None,
                    help="differential below which the turn command is zero. "
                         "Raising it makes steering SACCADIC -- straight runs "
                         "punctuated by discrete turns -- which is what a "
                         "walking fly does and what an AR(1)-matched random "
                         "arm cannot reproduce. Note the obvious confound: a "
                         "deadzone does that to ANY smooth signal, so the "
                         "shuffled arm is the reference that says whether the "
                         "wiring contributed anything.")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    help="which arms to run. The `shuffled` arm costs as much "
                         "as `connectome` and is a behavioural reference "
                         "rather than an attribution control (see the "
                         "docstring), so a sweep may drop it.")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--device",
                    default=(os.environ.get("FLYDOOM_DEVICE")
                             or ("cuda" if torch.cuda.is_available()
                                 else "cpu")))
    args = ap.parse_args()

    print(paint("flydoom M9 — closed-loop behaviour", "1"))
    print(paint("=" * 76, "90"))
    print(f"{args.seeds} seeds x {args.tics} tics "
          f"({args.tics / 35:.0f} s) x {len(args.scenarios)} scenarios, "
          f"4 arms")

    record = {"tics": args.tics, "seeds": args.seeds,
              "seed_start": args.seed_start, "bias_mv": args.bias,
              "smell": args.smell,
              "tau_baseline": args.tau_baseline,
              "device": args.device,
              "argv": sys.argv[1:],
              "git_sha": _git_sha(),
              "started": _dt.datetime.now().isoformat(timespec="seconds"),
              "runs": []}

    arms = [a for a in ARMS if a in set(args.arms)]
    if "connectome" not in arms:
        arms = ["connectome"] + arms
    motor_kw = {k: v for k, v in (("yaw_source", args.yaw_source),
                                  ("yaw_gain", args.yaw_gain),
                                  ("forward_gain", args.forward_gain),
                                  ("lateral_gain", args.lateral_gain),
                                  ("deadzone_hz", args.deadzone_hz))
                if v is not None}
    record["arms"] = arms
    record["optic_gain"] = args.optic_gain
    record["spiking_t4"] = args.spiking_t4
    record["yaw_source"] = args.yaw_source
    record["motor_kw"] = motor_kw
    record["mirror"] = args.mirror
    record["blind"] = args.blind
    record["env"] = {k: v for k, v in os.environ.items()
                     if k.startswith("FLYDOOM_")}

    for scen in args.scenarios:
        print(f"\n{paint(scen, '1;36')}")
        per_arm: dict[str, list[dict]] = {a: [] for a in arms}
        for seed in range(args.seed_start,
                          args.seed_start + args.seeds):
            m_int, dist = run_agent(scen, seed, args.tics, False, args.device,
                                    args.bias, args.smell, args.tau_baseline,
                                    args.optic_gain, args.spiking_t4,
                                    motor_kw, args.mirror, args.blind)
            per_arm["connectome"].append(m_int)
            if "shuffled" in per_arm:
                m_shuf, _ = run_agent(scen, seed, args.tics, True, args.device,
                                      args.bias, args.smell,
                                      args.tau_baseline, args.optic_gain,
                                      args.spiking_t4, motor_kw,
                                      args.mirror, args.blind)
                per_arm["shuffled"].append(m_shuf)
            rng = np.random.default_rng(seed)
            if "random" in per_arm:
                per_arm["random"].append(
                    run_scripted(scen, seed, args.tics, dist, rng))
            if "still" in per_arm:
                per_arm["still"].append(
                    run_scripted(scen, seed, args.tics, None, rng))
            print(f"  seed {seed}: "
                  f"tiles {m_int.get('tiles_visited', 0):4.0f}  "
                  f"healed {m_int.get('healed', 0):5.0f}  "
                  f"hp {m_int.get('health_end', 0):5.0f}")

        for arm in arms:
            record["runs"].append(
                {"scenario": scen, "arm": arm, "episodes": per_arm[arm]})

        w = max(len(lbl) for _, lbl, _ in REPORT) + 2
        print(f"\n  {'':<{w}}" + "".join(f"{a:>16}" for a in arms))
        for key, label, fmt in REPORT:
            cells = []
            for arm in arms:
                vals = [e.get(key) for e in per_arm[arm] if key in e]
                cells.append(f"{fmt.format(np.mean(vals))}"
                             f" ±{np.std(vals):.0f}" if vals else "-")
            print(f"  {label:<{w}}" + "".join(f"{c:>16}" for c in cells))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1))
        print(f"\n  wrote {args.json}")

    # ---------------- acceptance ----------------
    print(f"\n{paint('ACCEPTANCE', '1')}")
    ok = True
    for scen in args.scenarios:
        rows = {r["arm"]: r["episodes"]
                for r in record["runs"] if r["scenario"] == scen}

        def mean(arm, key):
            v = [e.get(key, 0.0) for e in rows.get(arm, [])]
            return float(np.mean(v)) if v else 0.0

        for key, label in (("yaw_clip_frac", "yaw"),
                           ("fwd_clip_frac", "forward"),
                           ("lat_clip_frac", "lateral")):
            f = mean("connectome", key)
            good = f < 0.25
            ok &= good
            print(f"  {'PASS' if good else paint('FAIL', '1;31')}  "
                  f"{scen}: {label} command is not pinned at its clamp"
                  f"   {f:.0%} of tics")

        spin = mean("connectome", "spin")
        good = spin < 0.98
        ok &= good
        print(f"  {'PASS' if good else paint('FAIL', '1;31')}  "
              f"{scen}: not locked in a constant spin   mean sign {spin:.2f}")

        # The questions this milestone exists to ask.
        for key, label, unit in (
                ("tiles_visited", "explores more than", "tiles"),
                ("tics", "survives longer than", "tics")):
            c, r = mean("connectome", key), mean("random", key)
            beats = c > r * 1.1
            print(f"  {'PASS' if beats else paint('FAIL', '33')}  "
                  f"{scen}: {label} matched random"
                  f"   {c:.0f} vs {r:.0f} {unit}")
        # Wall avoidance: a fly does this with optic flow, so if the visual
        # pathway carried anything the intact arm should scrape less.
        c, r = mean("connectome", "collisions_per_1k_tics"), mean(
            "random", "collisions_per_1k_tics")
        beats = c < r * 0.9
        print(f"  {'PASS' if beats else paint('FAIL', '33')}  "
              f"{scen}: hits walls less than matched random"
              f"   {c:.1f} vs {r:.1f} per 1k tics")

        vr = [abs(e.get("vision_steer_r_best", 0.0)) for e in rows["connectome"]]
        vr = float(np.mean(vr)) if vr else 0.0
        good = vr > 0.2
        print(f"  {'PASS' if good else paint('FAIL', '33')}  "
              f"{scen}: steering is coupled to what the eyes see"
              f"   |r| = {vr:.3f} at best lag")

        # ...and the one that says whether any of it is about the wiring.
        c, sh = mean("connectome", "tiles_visited"), mean("shuffled",
                                                          "tiles_visited")
        print(f"  {'    ' if c > sh else paint('NOTE', '33')}  "
              f"{scen}: intact vs shuffled   {c:.0f} vs {sh:.0f} tiles"
              f"   ({'intact ahead' if c > sh else 'SHUFFLED AHEAD'})")

    print()
    print(paint("VERDICT: " + ("stable" if ok else "UNSTABLE"),
                "1;32" if ok else "1;31"))
    print("""
Read the last two checks per scenario, not the stability ones. Stability only
says the decoder is not clipping or spinning.

Wall avoidance is the behavioural read-out of M3 and M4: a fly steers away from
a looming wall using optic-flow expansion, which is the computation those two
milestones show this model cannot perform. `vision_steer_r` is the same
question one level lower down -- whether the turn command is coupled to the
left-right luminance difference on the retinas at all. A near-zero correlation
there is the mechanism behind a negative wall-avoidance result, and both are
results rather than bugs.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
