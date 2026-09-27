#!/usr/bin/env python3
"""Compare configurations on HOW THEY MOVE, not on what they collected.

Collecting medkits is a task the animal has no circuitry for, so scoring a
model of a fly by its Doom score is backwards: a fly that moves like a fly has
improved even if it collects less. These are measures of the movement itself,
paired seed by seed against a baseline, so every comparison is within-level.

Unambiguous, in the sense that a controller which jitters, circles or sits
railed against its own clamp is worse by any standard:

    yaw_chatter        how often the turn command flips sign
    spin               net rotational bias -- is it going in circles
    yaw_clip_frac      fraction of tics railed at the turn clamp
    collisions_per_1k  how often it hits something
    free_run_tics      how far it gets between collisions

Arguable, and marked as such:

    straightness       net displacement over path walked
    tiles_per_1k_path  ground covered per unit walked
    vision_steer_abs_r whether the turn is coupled to the eyes at all

    python scripts/analyse_movement.py [baseline] [config ...]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_behav4 import ci95, load_dir  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "paper" / "data"

METRICS = {          # name -> (+1 larger is better, -1 smaller is better)
    "yaw_chatter": -1, "spin": -1, "yaw_clip_frac": -1,
    "collisions_per_1k_tics": -1, "free_run_tics": +1,
    "straightness": +1, "tiles_per_1k_path": +1, "vision_steer_abs_r": +1,
}


def load(name: str, arm: str = "intact") -> dict:
    d = DATA / name / f"fly_{arm}_shards"
    per = (load_dir(d) or {}) if d.exists() else {}
    for seed in per.values():
        for rec in seed.values():
            if not isinstance(rec, dict):
                continue
            path = rec.get("path", 0.0)
            rec["straightness"] = (rec.get("net_displacement", 0.0) / path
                                   if path else 0.0)
            rec["vision_steer_abs_r"] = abs(rec.get("vision_steer_r", 0.0))
    return per


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 1
    base_name, others = args[0], args[1:]
    base = load(base_name)
    if not base:
        print(f"no shards for {base_name}")
        return 1

    print(f"paired against {base_name}, connectome arm, intact\n")
    head = f"{'metric':<24}"
    for o in others:
        head += f"{o.replace('behav_', ''):>22}"
    print(head)
    print(f"{'':<24}" + "".join(f"{'(better/worse)':>22}" for _ in others))

    for m, sign in METRICS.items():
        row = f"{m:<24}"
        for name in others:
            per = load(name)
            seeds = [s for s in per if s in base
                     and "connectome" in per[s] and "connectome" in base[s]]
            if len(seeds) < 15:
                row += f"{'-':>22}"
                continue
            v = [per[s]["connectome"].get(m, 0.0)
                 - base[s]["connectome"].get(m, 0.0) for s in seeds]
            mm, cc = ci95(v)
            tag = ""
            if abs(mm) > cc:
                tag = "BETTER" if mm * sign > 0 else "WORSE"
            row += f"{mm:+11.4f}+-{cc:.4f}{tag:>7}"[:22].rjust(22)
        print(row)

    print("\nabsolute values, for scale")
    print(f"{'metric':<24}{base_name.replace('behav_', ''):>16}"
          + "".join(f"{o.replace('behav_', ''):>16}" for o in others))
    for m in METRICS:
        row = f"{m:<24}"
        for name in [base_name] + others:
            per = load(name)
            v = [r["connectome"].get(m, 0.0) for r in per.values()
                 if "connectome" in r]
            row += f"{np.mean(v):16.4f}" if v else f"{'-':>16}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
