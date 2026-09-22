#!/usr/bin/env python3
"""The bar-fixation table, from the shards experiments/m18_stripe.py writes.

Per arm: the fixation index F = mean cos(bar azimuth), its 95% CI over seeds,
the same minus each seed's own block-shuffled null, and where the bar actually
sat. A fly fixates; a mirrored fly, if the drive is genuine, anti-fixates.

    python scripts/analyse_stripe.py [paper/data/m18_stripe]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from m18_stripe import (ARMS, CURVE_BINS, ci95, fixation,  # noqa: E402
                        response_curve, restoring_slope, summarise)

BINS = np.array([-180, -150, -90, -30, 30, 90, 150, 180])
LABEL = ["rear", "L rear", "L side", "AHEAD", "R side", "R rear", "rear"]


def load(d: Path) -> dict[str, list]:
    out: dict[str, list] = {}
    for f in sorted(d.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            r = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        out.setdefault(r.get("arm", f.stem.split("_")[0]), []).append(r)
    return out


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/data/m18_stripe")
    shards = load(d)
    if not shards:
        print(f"no shards in {d}")
        return 1

    swept = {a: [r for r in v if r.get("sweep") is not None]
             for a, v in shards.items()}
    if any(swept.values()):
        centres = (CURVE_BINS[:-1] + CURVE_BINS[1:]) / 2
        print(f"bar swept round the eye, {d}")
        print("commanded yaw (deg/tic) against bar azimuth; positive azimuth "
              "is the bar\non the fly's LEFT and positive yaw turns it RIGHT, "
              "so steering TOWARD\nthe bar is a curve that falls through "
              "zero.\n")
        print(f"{'arm':<10}{'n':>3}{'slope':>9}  "
              + "".join(f"{c:+6.0f}" for c in centres[::2]))
        for arm, rows in swept.items():
            if not rows:
                continue
            c, m, ci = response_curve(rows)
            print(f"{arm:<10}{len(rows):>3}{restoring_slope(rows):+9.4f}  "
                  + "".join(f"{v:+6.2f}" for v in m[::2]))
            print(f"{'':<22}" + "".join(f"{v:6.2f}" for v in ci[::2])
                  + "   (95% CI)")
        print()
        shards = {a: [r for r in v if r.get("sweep") is None]
                  for a, v in shards.items()}
        shards = {a: v for a, v in shards.items() if v}
        if not shards:
            return 0

    print(f"bar fixation, {d}\n")
    print(f"{'arm':<10}{'n':>3}  {'F':>16}  {'F - shuffled null':>21}"
          f"  {'ahead':>7}{'behind':>7}")
    for arm in [a for a in ARMS if a in shards] + \
               [a for a in shards if a not in ARMS]:
        s = summarise(shards[arm])
        star = "*" if abs(s["F_minus_surrogate"]) > s["F_minus_surrogate_ci"] else " "
        print(f"{arm:<10}{s['n']:>3}  {s['F']:+8.3f} +-{s['F_ci']:5.3f}  "
              f"{s['F_minus_surrogate']:+11.3f} +-{s['F_minus_surrogate_ci']:5.3f}"
              f"{star}  {s['front30']:7.3f}{s['rear150']:7.3f}")

    print("\nwhere the bar sat, fraction of time")
    print(f"{'arm':<10}" + "".join(f"{l:>9}" for l in LABEL[1:-1]) + f"{'rear':>9}")
    for arm, rows in shards.items():
        az = np.concatenate([np.asarray(r["az"], float) for r in rows])
        h = np.histogram(az, bins=BINS)[0].astype(float)
        h[0] += h[-1]                       # the two rear bins are one wedge
        h = h[:-1] / max(h[:-1].sum(), 1)
        print(f"{arm:<10}" + "".join(f"{v:9.3f}" for v in h[1:])
              + f"{h[0]:9.3f}")

    print("\nper seed F")
    for arm, rows in shards.items():
        v = sorted(fixation(np.asarray(r["az"], float)) for r in rows)
        m, c = ci95(v)
        print(f"  {arm:<10} {m:+.3f} +-{c:.3f}   "
              + " ".join(f"{x:+.2f}" for x in v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
