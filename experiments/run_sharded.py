#!/usr/bin/env python3
"""Run M9 with the seeds split across processes, then merge and summarise.

WHY THIS EXISTS
---------------
The simulation is dispatch-bound, not compute-bound: one LIF step is 0.31 ms of
GPU work wrapped in ~8.5 ms of Python and ATen dispatch, so a single episode
leaves the GPU 97% idle and uses ~1.3 of 24 cores. Running one episode per
process therefore costs almost nothing extra until the cores run out, and the
whole sweep drops from hours to minutes.

It also fixes the reason a phantom sweep went unnoticed for two days: every
shard writes JSON stamped with argv, git SHA, device and wall-clock times, and
the merged file records how long the run actually took. A result that cannot be
traced back to the code and the clock that produced it is not a result.

    python experiments/run_sharded.py --scenarios health_gathering_supreme \
        --seeds 30 --tics 700 --smell --jobs 14 --json paper/data/m9_n30.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M9 = ROOT / "experiments" / "m9_behaviour.py"
PY = ROOT / ".venv" / "bin" / "python"


def _sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def ci95(vals: list[float]) -> tuple[float, float]:
    """Mean and half-width of the 95% CI. The half-width is the number that
    matters here -- an effect smaller than it cannot be claimed."""
    n = len(vals)
    if n < 2:
        return (float(vals[0]) if vals else 0.0), float("inf")
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["health_gathering_supreme"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--tics", type=int, default=700)
    ap.add_argument("--smell", action="store_true")
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--tau-baseline", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=12,
                    help="concurrent processes. Each takes ~1.3 cores, ~570 MB "
                         "of VRAM and ~1.4 GB of system RAM. RAM is the "
                         "binding constraint and the one easy to miss: "
                         "min(nproc/1.3, vram/0.6GB, budget_GB/1.4). At 24 "
                         "cores / 12 GB VRAM / 18 GB budget that is "
                         "min(18, 20, 12) = 12.")
    ap.add_argument("--seed-base", type=int, default=0,
                    help="First seed. The tuned arm of this study tunes on one "
                         "block of seeds and validates on another, and the "
                         "held-out block has to be one nothing was chosen on.")
    ap.add_argument("--optic-gain", type=float, default=1.0)
    ap.add_argument("--spiking-t4", action="store_true")
    ap.add_argument("--yaw-source", default="DNa02",
                    choices=["DNa02", "DNp15"])
    ap.add_argument("--yaw-gain", type=float, default=None)
    ap.add_argument("--deadzone-hz", type=float, default=None)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    shard_dir = args.json.parent / (args.json.stem + "_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir = shard_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    base = [str(PY), "-u", str(M9),
            "--scenarios", *args.scenarios,
            "--tics", str(args.tics),
            "--seeds", "1",
            "--device", args.device,
            "--bias", str(args.bias)]
    if args.smell:
        base.append("--smell")
    if args.tau_baseline is not None:
        base += ["--tau-baseline", str(args.tau_baseline)]
    if args.optic_gain != 1.0:
        base += ["--optic-gain", str(args.optic_gain)]
    if args.spiking_t4:
        base.append("--spiking-t4")
    if args.yaw_source != "DNa02":
        base += ["--yaw-source", args.yaw_source]
    if args.yaw_gain is not None:
        base += ["--yaw-gain", str(args.yaw_gain)]
    if args.deadzone_hz is not None:
        base += ["--deadzone-hz", str(args.deadzone_hz)]
    if args.mirror:
        base.append("--mirror")
    if args.blind:
        base.append("--blind")
    if args.arms:
        base += ["--arms", *args.arms]

    def _done(path: Path) -> bool:
        """A shard counts as finished only if it parses and carries episodes.
        m9 writes its JSON in one go at the end, but a process killed mid-write
        leaves a truncated file -- which must re-run, not be mistaken for work
        already done."""
        if not path.exists():
            return False
        try:
            d = json.loads(path.read_text())
        except Exception:
            return False
        return bool(d.get("runs")) and all(r.get("episodes") for r in d["runs"])

    shards = [(s, shard_dir / f"seed{s:03d}.json")
              for s in range(args.seed_base, args.seed_base + args.seeds)]
    jobs, resumed = [], []
    for s, out in shards:
        if _done(out):
            resumed.append(s)
            continue
        jobs.append((s, base + ["--seed-start", str(s), "--json", str(out)], out))

    print(f"flydoom M9 sharded: {args.seeds} seeds x {len(args.scenarios)} "
          f"scenarios, {args.jobs} at a time")
    print(f"  git {_sha()}   device {args.device}   tics {args.tics}"
          f"   smell {args.smell}   tau_baseline {args.tau_baseline}")
    print(f"  seeds {args.seed_base}..{args.seed_base + args.seeds - 1}"
          f"   optic_gain {args.optic_gain}   spiking_t4 {args.spiking_t4}")
    _envs = {k: v for k, v in os.environ.items() if k.startswith("FLYDOOM_")}
    print(f"  env {_envs}")
    if resumed:
        print(f"  resuming: {len(resumed)} shards already complete, "
              f"{len(jobs)} to run")
    if not jobs:
        print("  nothing to run; merging existing shards")
    t0 = time.time()

    running: list[tuple[int, subprocess.Popen, Path, object]] = []
    queue = list(jobs)
    done = 0
    failed: list[int] = []
    while queue or running:
        while queue and len(running) < args.jobs:
            s, cmd, out = queue.pop(0)
            fh = open(log_dir / f"seed{s:03d}.log", "w")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            running.append((s, p, out, fh))
        time.sleep(2.0)
        for entry in running[:]:
            s, p, out, fh = entry
            if p.poll() is not None:
                fh.close()
                running.remove(entry)
                done += 1
                ok = p.returncode == 0 and out.exists()
                if not ok:
                    failed.append(s)
                el = time.time() - t0
                print(f"  [{done:3d}/{len(jobs)}] seed {s:3d} "
                      f"{'ok ' if ok else 'FAIL'}  {el/60:5.1f} min elapsed",
                      flush=True)

    wall = time.time() - t0
    if failed:
        print(f"\n  WARNING: {len(failed)} shards failed: {failed}")
        print(f"  logs in {log_dir}")

    # ---- merge -------------------------------------------------------
    merged: dict[tuple[str, str], list[dict]] = {}
    meta = None
    for s, out in shards:
        if not out.exists():
            continue
        d = json.loads(out.read_text())
        meta = meta or d
        for r in d["runs"]:
            merged.setdefault((r["scenario"], r["arm"]), []).extend(r["episodes"])

    record = {
        "tics": args.tics, "seeds": args.seeds, "smell": args.smell,
        "bias_mv": args.bias, "tau_baseline": args.tau_baseline,
        "device": args.device, "git_sha": _sha(),
        "seed_base": args.seed_base, "optic_gain": args.optic_gain,
        "spiking_t4": args.spiking_t4, "yaw_source": args.yaw_source,
        "yaw_gain": args.yaw_gain,
        "mirror": args.mirror, "blind": args.blind,
        "deadzone_hz": args.deadzone_hz,
        "arms": args.arms,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith("FLYDOOM_")},
        "argv": sys.argv[1:],
        "started": dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 1),
        "shards_failed": failed,
        "runs": [{"scenario": k[0], "arm": k[1], "episodes": v}
                 for k, v in sorted(merged.items())],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=1))

    # ---- summary -----------------------------------------------------
    KEYS = [("healed", "health picked up"), ("tiles_visited", "tiles visited"),
            ("collisions_per_1k_tics", "collisions/1k"),
            ("stuck_frac", "stuck frac"), ("tics", "tics survived")]
    ARMS = tuple(args.arms) if args.arms else (
        "connectome", "shuffled", "random", "still")
    print(f"\n  wall {wall/60:.1f} min   ->  {args.json}")
    for scen in args.scenarios:
        print(f"\n{scen}   (n = {args.seeds}, mean +/- 95% CI)")
        print(f"  {'':<20}" + "".join(f"{a:>20}" for a in ARMS))
        for key, label in KEYS:
            cells = []
            for arm in ARMS:
                eps = merged.get((scen, arm), [])
                vals = [e[key] for e in eps if key in e]
                if not vals:
                    cells.append(f"{'-':>20}")
                    continue
                m, h = ci95(vals)
                cells.append(f"{m:>11.1f} +/-{h:>5.1f}")
            print(f"  {label:<20}" + "".join(cells))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
