#!/usr/bin/env python3
"""Compare the four conductance parameterisations on identical held-out seeds.

Each arm is run through the same protocol on the same environment seeds, so
every comparison here is paired by seed: the seed-to-seed variance in this
environment is far larger than the differences being tested, and an unpaired
test would drown in it.

"Separates from control" means the paired per-seed difference between the
connectome arm and its own command-matched random arm has a 95% CI that
excludes zero. The metric set is listed explicitly below rather than left
implicit, because the count depends on it.
"""
import json
import math
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "paper" / "data" / "behav4"

# Metrics compared against the matched control. Excluded, with reasons:
#   kills, attack_frac, attack_chatter, use_frac -- no weapon in this scenario,
#     these are identically zero and cannot separate;
#   collisions -- raw count, superseded by collisions_per_1k_tics;
#   vision_steer_r -- superseded by vision_steer_r_best;
#   vision_steer_r_best, vision_steer_lag_tics -- defined only for the
#     connectome arm (the control has no retina), so they admit no paired
#     difference and are reported separately below.
# metric -> +1 if larger is better for the agent, -1 if smaller is better,
# 0 if the metric has no better direction (it describes what the agent did,
# not how well). Only the signed ones can support a claim that the model
# "exceeds" a control; the unsigned ones can only show that it differs.
METRICS = {
    "tiles_visited": +1, "tiles_per_1k_path": +1, "path": 0,
    "net_displacement": +1,
    "healed": +1, "health_end": +1, "damage": -1,
    "collisions_per_1k_tics": -1, "free_run_tics": +1, "stuck_frac": -1,
    "spin": -1, "tics": +1,
    "yaw_abs_mean": 0, "yaw_chatter": -1, "yaw_clip_frac": -1,
    "fwd_abs_mean": 0, "fwd_clip_frac": -1, "lat_clip_frac": -1,
}
VISION_ONLY = ["vision_steer_r_best", "vision_steer_lag_tics"]
ARMS = ["arm1_frozen", "arm2_published", "arm3_uniform", "arm4_regional"]


def ci95(v):
    n = len(v)
    if n < 2:
        return (v[0] if v else 0.0), float("inf")
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def load(tag):
    """seed -> arm -> episode metrics.

    Read from the per-seed shards rather than the merged file: the merge
    groups episodes by arm and drops the seed label, and pairing by seed is
    the whole point of the comparison.
    """
    shards = sorted((DATA / f"{tag}_shards").glob("seed*.json"))
    per = {}
    for f in shards:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not d.get("runs"):
            continue
        seed = int(f.stem[4:])
        for r in d["runs"]:
            for e in r.get("episodes", []):
                per.setdefault(seed, {})[r["arm"]] = e
    return per or None


def main():
    loaded = {t: load(t) for t in ARMS}
    missing = [t for t, v in loaded.items() if not v]
    if missing:
        print("missing, not yet run:", ", ".join(missing))
    loaded = {t: v for t, v in loaded.items() if v}
    if not loaded:
        return 1

    common = None
    for v in loaded.values():
        s = {k for k, arms in v.items()
             if "connectome" in arms and "random" in arms}
        common = s if common is None else (common & s)
    common = sorted(common or [])
    print(f"seeds complete in every arm: {len(common)}")
    print(f"metrics tested: {len(METRICS)}\n")

    print(f"{'arm':<16}{'differs':>8}{'better':>11}  {'vision r':>9}")
    print("-" * 46)
    detail = {}
    for tag, per in loaded.items():
        differ, better, rows = 0, 0, []
        for k, direction in METRICS.items():
            diffs = [per[s]["connectome"][k] - per[s]["random"][k]
                     for s in common
                     if k in per[s]["connectome"] and k in per[s]["random"]]
            if len(diffs) < 2:
                continue
            m, h = ci95(diffs)
            ok = abs(m) > h            # CI excludes zero
            good = bool(ok and direction and m * direction > 0)
            differ += ok
            better += good
            rows.append((k, m, h, ok, good, direction))
        vr = [per[s]["connectome"].get("vision_steer_r_best")
              for s in common if per[s].get("connectome")]
        vr = [x for x in vr if x is not None]
        vm = sum(vr) / len(vr) if vr else float("nan")
        n_signed = sum(1 for d in METRICS.values() if d)
        print(f"{tag:<16}{differ:>4}/{len(METRICS):<3}"
              f"{better:>7}/{n_signed:<4}  {vm:>9.3f}")
        detail[tag] = rows

    for tag, rows in detail.items():
        print(f"\n=== {tag} : paired difference vs command-matched control ===")
        for k, m, h, ok, good, direction in rows:
            mark = "+" if good else ("*" if ok else " ")
            arrow = {1: "up", -1: "dn", 0: "--"}[direction]
            print(f"  {mark} {k:<24}{m:>10.2f} +/-{h:<8.2f} ({arrow})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
