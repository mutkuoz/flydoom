#!/usr/bin/env python3
"""Sign tests over seeds, with ties excluded, for any pair of conditions.

WHY THIS EXISTS. The seeds here are levels, and levels differ enough that a
few can carry a mean on their own -- so a paired mean should never be quoted
without the per-seed sign beside it. Two runs in this project were saved from
being reported as findings by exactly that check.

AND WHY IT EXCLUDES TIES. Health collected is quantised, so a large minority of
seeds come out exactly equal. Counting those as failures, which a naive
`(v > 0).sum()` does, understates the effect badly: the project's central
result reads p=0.741 that way and p=0.005 done properly. A sign test is over
the seeds that moved.

    python scripts/sign_test.py behav_fixed intact healed
    python scripts/sign_test.py behav_fixed intact healed --vs behav_wind
"""

import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_behav4 import ci95, load_dir  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "paper" / "data"


def sign_p(v):
    pos, neg = int((v > 0).sum()), int((v < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg, int((v == 0).sum())
    p = 2 * min(sum(comb(n, i) for i in range(pos, n + 1)),
                sum(comb(n, i) for i in range(neg, n + 1))) / 2 ** n
    return min(p, 1.0), pos, neg, int((v == 0).sum())


def arm(name, a):
    d = DATA / name / f"fly_{a}_shards"
    return (load_dir(d) or {}) if d.exists() else {}


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    name, a, metric = sys.argv[1:4]
    other = sys.argv[sys.argv.index("--vs") + 1] if "--vs" in sys.argv else None
    p1 = arm(name, a)
    if other:
        p2 = arm(other, a)
        seeds = [s for s in p2 if s in p1 and "connectome" in p1[s]
                 and "connectome" in p2[s]]
        v = np.array([p2[s]["connectome"][metric] - p1[s]["connectome"][metric]
                      for s in seeds])
        label = f"{other} minus {name}, {a}, {metric}"
    else:
        seeds = [s for s in p1 if "connectome" in p1[s] and "random" in p1[s]]
        v = np.array([p1[s]["connectome"][metric] - p1[s]["random"][metric]
                      for s in seeds])
        label = f"{name} {a}: connectome minus its random arm, {metric}"
    if not len(v):
        print("no paired seeds")
        return 1
    m, c = ci95(v)
    p, pos, neg, ties = sign_p(v)
    print(f"{label}\n")
    print(f"  mean {m:+.2f} +- {c:.2f}   median {np.median(v):+.2f}")
    print(f"  {pos} up, {neg} down, {ties} tied of {len(v)} seeds")
    print(f"  sign test over the {pos + neg} that moved: p = {p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
