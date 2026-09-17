#!/usr/bin/env python3
"""Optomotor drum, read per seed against its own no-drum baseline.

m12 reports a linear slope over drum speeds. Two things make that the wrong
summary here. The response is speed-TUNED, strongest at slow drums and gone by
8 deg/tic, as a correlator's velocity tuning predicts, so a straight line
through it averages a peak with nothing. And the modulation at speed 0 is not
zero: episodes that end early leave unequal half-cycles, and yaw drifts, so
every speed carries a seed-specific offset. Subtracting each seed's own
speed-0 value removes that offset, and pairing by seed removes level-to-level
variance.

Sign: the modulation is the commanded yaw during the right-turning half of
the drum minus the left-turning half. The fly's optomotor reflex turns WITH
the world, which moves opposite to the imposed body turn, so the reflex is
NEGATIVE.

    python scripts/analyse_drum.py paper/data/drum_anat/OL_*.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    d = json.loads(Path(path).read_text())
    arm = [k for k, v in d.items() if isinstance(v, dict) and "rows" in v][0]
    m = defaultdict(dict)
    for r in d[arm]["rows"]:
        m[r["seed"]][r["imposed"]] = r["modulation"]
    return d, arm, m


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def main(paths) -> int:
    print("modulation minus the same seed's no-drum value, mean +- 95% CI "
          "(* excludes 0; negative = optomotor)\n")
    for path in paths:
        d, arm, m = load(path)
        speeds = sorted({sp for s in m for sp in m[s] if sp != 0.0})
        tag = (f"{Path(path).stem:<16} eye={d.get('eye_map', '?'):<10} "
               f"turn={'fixed' if d.get('fixed_turn') else 'legacy':<6} "
               f"{'open ' if d.get('open_loop') else 'closed'} {arm:<8}")
        cells = []
        for sp in speeds:
            v = [m[s][sp] - m[s][0.0] for s in m if sp in m[s] and 0.0 in m[s]]
            mm, cc = ci(v)
            cells.append(f"{sp:g}: {mm:+.2f}+-{cc:.2f}{'*' if abs(mm) > cc else ' '}")
        print(tag + "  ".join(cells) + f"   n={len(m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
