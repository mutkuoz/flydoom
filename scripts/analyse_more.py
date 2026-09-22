#!/usr/bin/env python3
"""The fly-arena behaviour tables on 120 seeds instead of 60.

paper/data/behav_fixed holds seeds 40-99; paper/data/behav_more holds 100-159,
run with the same command and the same brain. Pooling them halves the width of
every interval, which is the only thing that will settle the model-versus-model
contrasts: in behav_fixed, fly-arena intact minus mirrored on health is
+3.00 +- 6.18, a CI four times the size it needs to be.

Rows are the connectome minus its own command-matched random arm, paired by
seed (* = the 95% CI excludes zero), then the same contrasts between models.

    python scripts/analyse_more.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_behav4 import ci95, load_dir  # noqa: E402
from analyse_fixed import METRICS, fmt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "paper" / "data"
F, M = DATA / "behav_fixed", DATA / "behav_more"

CONDITIONS = [
    ("fly: intact", [F / "fly_intact_shards", M / "fly_intact_shards"]),
    ("fly: mirrored", [F / "fly_mirrored_shards", M / "fly_mirrored_shards"]),
    ("fly: frozen", [F / "fly_frozen_shards", M / "fly_frozen_shards"]),
]

PAIRS = [("fly: intact", "fly: mirrored"), ("fly: intact", "fly: frozen")]


def main() -> int:
    loaded = {}
    for name, dirs in CONDITIONS:
        per = {}
        for d in dirs:
            if d.exists():
                per.update(load_dir(d) or {})
        if per:
            loaded[name] = per

    if not loaded:
        print("no shards yet")
        return 1

    print("connectome minus its command-matched random arm, paired by seed\n")
    print(f"{'condition':<18}{'n':>4}  " + "".join(f"{m:<25}" for m in METRICS))
    for name, per in loaded.items():
        seeds = [s for s in per if "connectome" in per[s] and "random" in per[s]]
        row = f"{name:<18}{len(seeds):>4}  "
        for m, sign in METRICS.items():
            v = [per[s]["connectome"][m] - per[s]["random"][m] for s in seeds]
            row += fmt(*ci95(v), sign) if v else " " * 25
        print(row)

    print("\nconnectome vs connectome, same seeds (first minus second)")
    for a, b in PAIRS:
        if a not in loaded or b not in loaded:
            continue
        pa, pb = loaded[a], loaded[b]
        seeds = [s for s in pa if s in pb and "connectome" in pa[s]
                 and "connectome" in pb[s]]
        row = f"  {a} - {b}"
        row = f"{row:<38}{len(seeds):>4}  "
        for m, sign in METRICS.items():
            v = [pa[s]["connectome"][m] - pb[s]["connectome"][m] for s in seeds]
            row += fmt(*ci95(v), sign) if v else ""
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
