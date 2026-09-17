#!/usr/bin/env python3
"""The full eye in the daylight arena, against its controls.

Every row is the connectome minus its own command-matched random arm, paired by
seed, mean and 95% CI over held-out seeds, the same measure as the headline
result in paper/data/behav_best. A row separates when its CI excludes zero.

    intact    the full eye as built
    mirrored  every lens reads its mirror-image direction
    frozen    the eye sees the first frame for the whole episode
    narrow    the old single 110 degree camera, same sky arena

The headline configuration (old arena, single camera) is printed alongside
for reference. It is not paired with the new runs: different arena, and it ran
on CPU where these run on the GPU.

    python scripts/analyse_wide.py [--first-block]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_behav4 import ci95, load_dir  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "paper" / "data"

W, B = DATA / "behav_wide", DATA / "behav_best"
# Each condition may span several seed blocks (seeds 40-69, then 70-99).
CONDITIONS = [
    ("full eye, sky: intact", [W / "wide_sky_shards", W / "wide_sky_b2_shards"]),
    ("full eye, sky: mirrored", [W / "wide_sky_mirror_shards",
                                 W / "wide_sky_mirror_b2_shards"]),
    ("full eye, sky: frozen", [W / "wide_sky_blind_shards",
                               W / "wide_sky_blind_b2_shards"]),
    ("narrow eye, sky: intact", [W / "narrow_sky_shards",
                                 W / "narrow_sky_b2_shards"]),
    ("headline (old arena): intact", [B / "all_applied_shards"]),
    ("headline (old arena): mirrored", [B / "all_applied_mirror_shards"]),
    ("headline (old arena): frozen", [B / "all_applied_blind_shards"]),
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
    first_block = "--first-block" in sys.argv
    loaded = {}
    for name, dirs in CONDITIONS:
        per = {}
        for d in dirs[:1] if first_block else dirs:
            per.update((d.exists() and load_dir(d)) or {})
        if per:
            loaded[name] = per

    print("connectome minus its command-matched random arm, paired by seed\n")
    head = f"{'condition':<32}{'n':>4}  " + "".join(f"{m:<25}" for m in METRICS)
    print(head)
    for name, per in loaded.items():
        seeds = [s for s in per if "connectome" in per[s] and "random" in per[s]]
        row = f"{name:<32}{len(seeds):>4}  "
        for m, sign in METRICS.items():
            v = [per[s]["connectome"][m] - per[s]["random"][m] for s in seeds]
            row += fmt(*ci95(v), sign) if v else " " * 25
        print(row)

    print("\nconnectome alone, mean over seeds")
    for name, per in loaded.items():
        seeds = [s for s in per if "connectome" in per[s]]
        vals = []
        for m in ("healed", "tics", "tiles_visited"):
            mm, cc = ci95([per[s]["connectome"][m] for s in seeds])
            vals.append(f"{m} {mm:7.2f} +-{cc:5.2f}")
        print(f"  {name:<32} " + "   ".join(vals))

    pairs = [("full eye, sky: intact", "narrow eye, sky: intact"),
             ("full eye, sky: intact", "full eye, sky: mirrored"),
             ("full eye, sky: intact", "full eye, sky: frozen")]
    print("\nconnectome vs connectome, same seeds (first minus second)")
    for a, b in pairs:
        if a not in loaded or b not in loaded:
            continue
        pa, pb = loaded[a], loaded[b]
        seeds = [s for s in pa if s in pb and "connectome" in pa[s]
                 and "connectome" in pb[s]]
        row = f"  {a} - {b.split(': ')[-1] if a.split(':')[0] == b.split(':')[0] else b}"
        row = f"{row:<58}{len(seeds):>3}  "
        for m, sign in METRICS.items():
            v = [pa[s]["connectome"][m] - pb[s]["connectome"][m] for s in seeds]
            row += fmt(*ci95(v), sign) if v else ""
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
