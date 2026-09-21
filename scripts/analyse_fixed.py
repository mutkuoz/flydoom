#!/usr/bin/env python3
"""The behaviour tables with a forward-walking fly, beside the ones they replace.

Every row is the connectome minus its own command-matched random arm, paired by
seed, mean and 95% CI over held-out seeds 40-99 (* = the CI excludes zero).

    legacy     paper/data/behav_wide: full eye, sky arena, through the
               uncorrected interface -- eye map read with the wrong hexagonal
               convention and unmirrored, reversed turn sign, and MDN's tonic
               107 Hz read as a walk command, so the agent reversed on 97% of
               tics
    corrected  paper/data/behav_fixed: the same brain, seeds and arena through
               the corrected interface, and the same again in the fly arena

The model-alone block reports signed walking as well, which is the check that
the fix did what it was for.

    python scripts/analyse_fixed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_behav4 import ci95, load_dir  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "paper" / "data"
W, F = DATA / "behav_wide", DATA / "behav_fixed"

CONDITIONS = [
    ("legacy, sky: intact", [W / "wide_sky_shards", W / "wide_sky_b2_shards"]),
    ("legacy, sky: mirrored", [W / "wide_sky_mirror_shards",
                               W / "wide_sky_mirror_b2_shards"]),
    ("legacy, sky: frozen", [W / "wide_sky_blind_shards",
                             W / "wide_sky_blind_b2_shards"]),
    ("corrected, sky: intact", [F / "sky_intact_shards"]),
    ("corrected, sky: mirrored", [F / "sky_mirrored_shards"]),
    ("corrected, sky: frozen", [F / "sky_frozen_shards"]),
    ("corrected, fly: intact", [F / "fly_intact_shards"]),
    ("corrected, fly: mirrored", [F / "fly_mirrored_shards"]),
    ("corrected, fly: frozen", [F / "fly_frozen_shards"]),
]

# metric -> +1 if larger is better, -1 if smaller is better
METRICS = {"healed": +1, "tics": +1, "tiles_visited": +1,
           "stuck_frac": -1, "damage": -1}


def fmt(m, c, sign):
    sep = abs(m) > c
    tag = ""
    if sep:
        tag = " better" if m * sign > 0 else " worse"
    return f"{m:+8.2f} +-{c:6.2f}{'*' if sep else ' '}{tag:<7}"


def main() -> int:
    loaded = {}
    for name, dirs in CONDITIONS:
        per = {}
        for d in dirs:
            per.update((d.exists() and load_dir(d)) or {})
        if per:
            loaded[name] = per

    print("connectome minus its command-matched random arm, paired by seed\n")
    print(f"{'condition':<28}{'n':>4}  " + "".join(f"{m:<25}" for m in METRICS))
    for name, per in loaded.items():
        seeds = [s for s in per if "connectome" in per[s] and "random" in per[s]]
        row = f"{name:<28}{len(seeds):>4}  "
        for m, sign in METRICS.items():
            v = [per[s]["connectome"][m] - per[s]["random"][m] for s in seeds]
            row += fmt(*ci95(v), sign) if v else " " * 25
        print(row)

    print("\nconnectome alone, mean over seeds")
    for name, per in loaded.items():
        seeds = [s for s in per if "connectome" in per[s]]
        parts = []
        for m in ("healed", "tics", "tiles_visited", "stuck_frac"):
            mm, cc = ci95([per[s]["connectome"][m] for s in seeds])
            parts.append(f"{m} {mm:7.2f} +-{cc:5.2f}")
        print(f"  {name:<28} " + "   ".join(parts))

    pairs = [("corrected, sky: intact", "corrected, sky: mirrored"),
             ("corrected, sky: intact", "corrected, sky: frozen"),
             ("corrected, sky: intact", "legacy, sky: intact"),
             ("corrected, fly: intact", "corrected, fly: mirrored"),
             ("corrected, fly: intact", "corrected, fly: frozen")]
    print("\nconnectome vs connectome, same seeds (first minus second)")
    for a, b in pairs:
        if a not in loaded or b not in loaded:
            continue
        pa, pb = loaded[a], loaded[b]
        seeds = [s for s in pa if s in pb and "connectome" in pa[s]
                 and "connectome" in pb[s]]
        row = f"  {a} - {b}"
        row = f"{row:<58}{len(seeds):>3}  "
        for m, sign in METRICS.items():
            v = [pa[s]["connectome"][m] - pb[s]["connectome"][m] for s in seeds]
            row += fmt(*ci95(v), sign) if v else ""
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
