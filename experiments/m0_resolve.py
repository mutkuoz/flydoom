#!/usr/bin/env python3
"""M0 — cell resolution report.

Runs before M1. Answers the questions we cannot defer:

  * Do the photoreceptors exist in this volume, or do we inject at the lamina?
  * Do the looming detectors, steering neurons and MN9 resolve cleanly?
  * Are the mushroom body and olfactory populations present, i.e. is the
    odour arm on the table at all?
  * Which handles are absent, and what near-misses exist?

Prints a verdict and exits non-zero if any REQUIRED handle is missing.

    python experiments/m0_resolve.py [--data data/raw] [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flydoom.cells import AnnotationTable, MatchKind, Resolution, Status  # noqa: E402
from flydoom.registry import ALL_HANDLES, GROUP_ORDER  # noqa: E402

USE_COLOR = sys.stdout.isatty()


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


MARK = {
    Status.OK: ("PASS", "32"),
    Status.WEAK_MATCH: ("WEAK", "33"),
    Status.COUNT_OFF: ("COUNT", "33"),
    Status.UNILATERAL: ("1SIDE", "33"),
    Status.MISSING: ("MISS", "31"),
}

GROUP_TITLE = {
    "visual_input": "Visual input — where Doom frames get injected",
    "visual_detector": "Visual feature detectors — probes",
    "gustatory": "Taste — the health / damage channel",
    "olfactory": "Olfactory — the Doom-entity odour channel",
    "mushroom_body": "Mushroom body — learning centre (frozen in v1)",
    "descending": "Descending neurons — the motor readout",
    "halting": "Halting neurons — the stop condition",
    "motor": "Motor output — the M2 readout",
    "aggression": "Aggression state — the ATTACK gate",
}


def fmt_sides(res: Resolution) -> str:
    if not res.side_counts:
        return ""
    order = sorted(res.side_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return " ".join(f"{s}={n}" for s, n in order[:4])


def fmt_expect(res: Resolution) -> str:
    lo, hi = res.handle.expect_min, res.handle.expect_max
    if lo is None and hi is None:
        return ""
    if hi is None:
        return f"expect >={lo}"
    if lo is None:
        return f"expect <={hi}"
    return f"expect {lo}-{hi}"


def report_line(res: Resolution, verbose: bool) -> None:
    label, color = MARK[res.status]
    req = " " if res.handle.required else "?"
    name = f"{res.handle.name:<12}"
    count = f"{res.count:>6}" if res.count else "     -"

    print(f"  {paint(f'{label:<5}', color)} {req} {name} {count}  {fmt_sides(res)}")

    detail: list[str] = []
    if res.match_kind is not MatchKind.EXACT and res.match_kind is not MatchKind.NONE:
        detail.append(
            f"matched by {res.match_kind.value} "
            f"{res.matched_pattern!r} on `{res.matched_column}`"
        )
    if res.status is Status.COUNT_OFF:
        detail.append(f"got {res.count}, {fmt_expect(res)}")
    if res.status is Status.UNILATERAL:
        detail.append("only one hemisphere — expected bilateral")
    if res.matched_types and (verbose or len(res.matched_types) > 1):
        shown = ", ".join(res.matched_types[:8])
        more = f" (+{len(res.matched_types) - 8} more)" if len(res.matched_types) > 8 else ""
        detail.append(f"types: {shown}{more}")
    if res.status is Status.MISSING:
        if res.candidates:
            detail.append("near misses: " + ", ".join(res.candidates))
        else:
            detail.append("no similar type strings found")

    for d in detail:
        print(f"          {paint('·', '90')} {d}")

    if verbose and res.handle.note:
        for line in _wrap(res.handle.note, 76):
            print(f"          {paint(line, '90')}")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def architectural_verdict(results: dict[str, Resolution]) -> list[str]:
    """The decisions this script exists to make."""
    out: list[str] = []

    def ok(name: str) -> bool:
        r = results.get(name)
        return r is not None and r.status is not Status.MISSING

    # The one fork we cannot defer.
    if ok("R1-6"):
        out.append(
            "RETINA: R1-6 resolved -> inject Doom luminance at the "
            "photoreceptors as spec 6.1 describes."
        )
    elif ok("L1") or ok("L2"):
        out.append(
            "RETINA: R1-6 ABSENT, lamina monopolars present -> inject at L1/L2 "
            "instead. Signed contrast (ON/OFF) rather than raw luminance, and "
            "this sidesteps the histamine mis-signing problem entirely. "
            "Update spec 6.1."
        )
    else:
        out.append(
            "RETINA: neither photoreceptors nor lamina resolved. STOP — there is "
            "no visual input path. Check that visual_neuron_types.csv is present "
            "in data/raw."
        )

    if ok("MN9") and ok("sugar_GRN") and ok("bitter_GRN"):
        out.append("M2 IS RUNNABLE: sugar, bitter and MN9 all resolved.")
    else:
        missing = [n for n in ("sugar_GRN", "bitter_GRN", "MN9") if not ok(n)]
        out.append(
            f"M2 IS BLOCKED: {', '.join(missing)} unresolved. Pull root IDs from "
            "the Shiu et al. 2024 supplement before going further — M2 is the "
            "gate everything else depends on."
        )

    if ok("DNa02"):
        out.append("STEERING: DNa02 resolved -> M3 readout available.")
    else:
        out.append("STEERING: DNa02 MISSING -> no yaw signal, M3 cannot run.")

    if ok("LC4") or ok("LPLC2"):
        out.append("LOOMING: escape arm of the engage/flee switch available.")
    else:
        out.append("LOOMING: LC4 and LPLC2 both missing -> M4 and M6 cannot run.")

    mb = [n for n in ("KC", "MBON", "PAM", "PPL1") if ok(n)]
    orn = [n for n in results if n.startswith("ORN_") and ok(n)]
    if len(mb) >= 3:
        out.append(
            f"ODOUR ARM: mushroom body present ({', '.join(mb)}) and "
            f"{len(orn)} ORN channels resolved. Both the innate (lateral horn, "
            "no learning) and plastic (KC->MBON) arms are on the table."
        )
    elif orn:
        out.append(
            f"ODOUR ARM: {len(orn)} ORN channels resolved but the mushroom body "
            "is incomplete — innate/lateral-horn arm only."
        )
    else:
        out.append("ODOUR ARM: no olfactory channels resolved — shelve it.")

    if ok("pC1") or ok("aIPg"):
        out.append("ATTACK GATE: aggression population resolved.")
    else:
        out.append(
            "ATTACK GATE: no aggression population -> use --attack-mode "
            "giantfiber or proboscis."
        )

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M0 — cell resolution report")
    ap.add_argument("--data", default="data/raw", help="directory holding the CSVs")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print the provenance note for every handle")
    args = ap.parse_args()

    print(paint("flydoom M0 — named cell resolution", "1"))
    print(paint("=" * 72, "90"))

    try:
        table = AnnotationTable.load(args.data)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 2

    print(f"sources      {', '.join(table.sources)}")
    print(f"neurons      {table.df.height:,}")
    print(f"type columns {', '.join(table.type_columns)}")
    print(f"vocabulary   {len(table.vocab):,} distinct type strings")

    results: dict[str, Resolution] = {}
    for group in GROUP_ORDER:
        handles = [h for h in ALL_HANDLES if h.group == group]
        if not handles:
            continue
        print(f"\n{paint(GROUP_TITLE.get(group, group), '1;36')}")
        for handle in handles:
            res = table.resolve(handle)
            results[handle.name] = res
            report_line(res, args.verbose)

    # ---- tallies ----
    print(f"\n{paint('=' * 72, '90')}")
    tally = {s: 0 for s in Status}
    for r in results.values():
        tally[r.status] += 1
    print(
        "  ".join(
            paint(f"{MARK[s][0]}={tally[s]}", MARK[s][1])
            for s in Status
            if tally[s]
        )
    )

    print(f"\n{paint('ARCHITECTURAL DECISIONS', '1')}")
    for line in architectural_verdict(results):
        head, _, rest = line.partition(": ")
        print(f"\n  {paint(head, '1;36')}")
        for wrapped in _wrap(rest, 70):
            print(f"    {wrapped}")

    fatal = [r for r in results.values() if r.is_fatal]
    print(f"\n{paint('=' * 72, '90')}")
    if fatal:
        names = ", ".join(r.handle.name for r in fatal)
        print(paint(f"VERDICT: FAIL — required handles unresolved: {names}", "1;31"))
        print("Do not proceed to M1. Resolve these by hand and extend the")
        print("registry patterns, or pull root IDs from the source papers.")
        return 1

    weak = [
        r for r in results.values()
        if r.status in (Status.WEAK_MATCH, Status.COUNT_OFF, Status.UNILATERAL)
        and r.handle.required
    ]
    if weak:
        print(paint("VERDICT: PASS WITH WARNINGS", "1;33"))
        print("Every required handle resolved, but some matched weakly or came")
        print("back the wrong size. Eyeball the `types:` lines above before")
        print("trusting them — a wrong DNa02 produces a plausible-looking agent")
        print("that is measuring nothing.")
        return 0

    print(paint("VERDICT: PASS — every required handle resolved cleanly.", "1;32"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
