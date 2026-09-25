#!/usr/bin/env python3
"""Does applying all three improvements actually help?

paper/data/behav_all runs the fly arena with dendritic cables, a neck and
whole-level odour, on THE SAME SEEDS (40-99) as paper/data/behav_fixed. Same
arena, same controls, same everything else, so the difference isolates those
three -- and because the seeds match, every comparison here is PAIRED, which is
worth roughly a doubling of the seeds against comparing two independent groups.

Three questions, in order of how much they are worth:

  1. Paired, per seed: does the all-applied model collect more health and live
     longer than the plain one on the SAME level? This is the direct test and
     the one to quote.
  2. Does each still beat its own command-matched random agent? A model can
     improve on the plain one and still not beat chance, or the reverse.
  3. Do the controls still behave? Mirroring should still remove the
     advantage; if it does not, the extra machinery is driving something that
     is not vision.

    python scripts/analyse_all.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_behav4 import ci95, load_dir  # noqa: E402
from analyse_fixed import METRICS, fmt  # noqa: E402

# HOW IT MOVES, which is the question health cannot answer.
#
# Collecting medkits is a task the animal has no circuitry for, so scoring a
# more faithful model by its Doom score is backwards: a fly that moves like a
# fly has improved even if it collects less. These are the measures of the
# movement itself. +1 means larger is better, -1 means smaller is.
#
# The first four are unambiguous -- a controller that jitters, circles, or sits
# railed against its own clamp is worse by any standard, biological or not.
# The rest are arguable and are marked where they are.
LOCOMOTION = {
    "yaw_chatter": -1,            # turn command flips sign this often
    "spin": -1,                   # net rotational bias: is it circling?
    "yaw_clip_frac": -1,          # fraction of tics railed at the turn clamp
    "fwd_clip_frac": -1,          # ...and at the walk clamp
    "collisions_per_1k_tics": -1,
    "free_run_tics": +1,          # how far it gets between collisions
    "straightness": +1,           # net displacement / path walked
    "tiles_per_1k_path": +1,      # ground covered per unit walked
    "vision_steer_abs_r": +1,     # is the turn coupled to the eyes AT ALL
}

DATA = Path(__file__).resolve().parent.parent / "paper" / "data"
PLAIN, ALL = DATA / "behav_fixed", DATA / "behav_all"
ARMS = ("intact", "mirrored", "frozen")


def load(root: Path, arm: str) -> dict:
    d = root / f"fly_{arm}_shards"
    per = (load_dir(d) or {}) if d.exists() else {}
    for seed in per.values():
        for arm_rec in seed.values():
            if not isinstance(arm_rec, dict):
                continue
            path = arm_rec.get("path", 0.0)
            arm_rec["straightness"] = (arm_rec.get("net_displacement", 0.0)
                                       / path if path else 0.0)
            arm_rec["vision_steer_abs_r"] = abs(arm_rec.get("vision_steer_r",
                                                            0.0))
    return per


def main() -> int:
    plain = {a: load(PLAIN, a) for a in ARMS}
    allon = {a: load(ALL, a) for a in ARMS}
    if not any(allon.values()):
        print(f"no shards in {ALL} yet")
        return 1

    print("1. PAIRED, same seed: all-applied MINUS plain, connectome arm\n")
    print(f"{'arm':<12}{'n':>4}  " + "".join(f"{m:<25}" for m in METRICS))
    for a in ARMS:
        p, q = plain[a], allon[a]
        seeds = [s for s in q if s in p
                 and "connectome" in q[s] and "connectome" in p[s]]
        row = f"{a:<12}{len(seeds):>4}  "
        for m, sign in METRICS.items():
            v = [q[s]["connectome"][m] - p[s]["connectome"][m] for s in seeds]
            row += fmt(*ci95(v), sign) if v else " " * 25
        print(row)

    print("\n2. HOW IT MOVES, paired, same seed: all-applied minus plain\n")
    print(f"{'metric':<24}{'n':>4}  {'difference':>26}   direction")
    for m, sign in LOCOMOTION.items():
        per_a, per_b = allon["intact"], plain["intact"]
        seeds = [s_ for s_ in per_a if s_ in per_b
                 and "connectome" in per_a[s_] and "connectome" in per_b[s_]]
        v = [per_a[s_]["connectome"].get(m, 0.0)
             - per_b[s_]["connectome"].get(m, 0.0) for s_ in seeds]
        if not v:
            continue
        mm, cc = ci95(v)
        tag = ""
        if abs(mm) > cc:
            tag = "BETTER" if mm * sign > 0 else "WORSE"
        arrow = "lower is better" if sign < 0 else "higher is better"
        print(f"{m:<24}{len(v):>4}  {mm:+12.4f} +-{cc:9.4f} {tag:<7} {arrow}")

    print("\n3. each against its OWN command-matched random arm\n")
    print(f"{'condition':<22}{'n':>4}  " + "".join(f"{m:<25}" for m in METRICS))
    for label, src in (("plain", plain), ("all applied", allon)):
        for a in ARMS:
            per = src[a]
            seeds = [s for s in per
                     if "connectome" in per[s] and "random" in per[s]]
            if not seeds:
                continue
            row = f"{label + ', ' + a:<22}{len(seeds):>4}  "
            for m, sign in METRICS.items():
                v = [per[s]["connectome"][m] - per[s]["random"][m]
                     for s in seeds]
                row += fmt(*ci95(v), sign) if v else " " * 25
            print(row)

    print("\n4. all-applied, connectome vs connectome (first minus second)\n")
    for x, y in (("intact", "mirrored"), ("intact", "frozen")):
        px, py = allon[x], allon[y]
        seeds = [s for s in px if s in py
                 and "connectome" in px[s] and "connectome" in py[s]]
        if not seeds:
            continue
        row = f"  {x} - {y:<14}{len(seeds):>4}  "
        for m, sign in METRICS.items():
            v = [px[s]["connectome"][m] - py[s]["connectome"][m] for s in seeds]
            row += fmt(*ci95(v), sign) if v else ""
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
