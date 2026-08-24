#!/usr/bin/env python3
"""Emit the DSI table straight from the M3 operating-point grid.

Single source of truth, as with gen_table.py: the table and Figure 2c are both
generated from data/m3_dsi.json, so they cannot drift from the run or from each
other.

    python ../experiments/m3_optomotor.py --dsi-grid --json paper/data/m3_dsi.json
    python gen_dsi_table.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUBTYPES = ("T4a", "T4b", "T5a", "T5b")
PAIRS = (("T4a", "T4b"), ("T5a", "T5b"))
# Experimental studies SELECT terminals for analysis at this DSI, so it is the
# threshold at which a real terminal merely qualifies as direction-tuned.
SELECTION_THRESHOLD = 0.5


def main():
    d = json.loads((HERE / "data" / "m3_dsi.json").read_text())
    pts = d["points"]
    unsat = [p for p in pts if p["saturation"] < d["saturation_threshold"]]
    if not unsat:
        raise SystemExit("no unsaturated operating points; nothing to report")

    best = max(unsat, key=lambda p: max(abs(v) for v in p["dsi"].values()))
    peak = max(abs(v) for v in best["dsi"].values())
    both = [p for p in unsat if all(p["mirror_opposed"].values())]
    frac = len(both) / len(unsat)

    rows = []
    for a, b in PAIRS:
        pathway = "ON " if a.startswith("T4") else "OFF"
        opp = best["dsi"][a] * best["dsi"][b] < 0
        rows.append(
            f"{pathway} & \\cell{{{a}}} ${best['dsi'][a]:+.5f}$ & "
            f"\\cell{{{b}}} ${best['dsi'][b]:+.5f}$ & "
            f"{'opposite' if opp else 'same sign'}\\\\"
        )

    # How reliable is the sign? Count agreement per pair across the grid.
    per_pair = {
        f"{a}/{b}": sum(p["mirror_opposed"][f"{a}/{b}"] for p in unsat)
        for a, b in PAIRS
    }
    detail = ", ".join(f"\\cell{{{k}}} {v}/{len(unsat)}"
                       for k, v in per_pair.items())

    tex = f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Direction selectivity, at the strongest operating point that is not
saturated. Graded units are capped at {d['graded_max_rate_hz']:.0f}\\,Hz and a
unit against its ceiling reports $\\mathrm{{DSI}}\\approx0$ whatever its inputs
do, so we sweep bias, spatial period and temporal frequency
({d['n_points']} points), keep the {d['n_unsaturated']} below
{100 * d['saturation_threshold']:.0f}\\% saturation, and report the largest
effect among them --- bias {best['bias_mv']:.1f}\\,mV, period
{best['period_deg']:.0f}$^\\circ$, {best['tf_hz']:.0f}\\,Hz. The magnitude is
${SELECTION_THRESHOLD / peak:.0f}\\times$ below the value at which experimental
studies merely \\emph{{select}} a terminal as direction-tuned. The sign is not
reliable either: both mirror pairs oppose at only {len(both)} of
{len(unsat)} unsaturated points ({100 * frac:.0f}\\%), by pair {detail}. A
correlator's sign should not depend on the operating point, so this is a
quantity fluctuating about zero rather than a small but real selectivity.}}
\\label{{tab:dsi}}
\\begin{{tabular}}{{lrrl}}
\\toprule
Pathway & \\multicolumn{{2}}{{c}}{{DSI}} & Mirror pair\\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}"""

    (HERE / "tab_dsi.tex").write_text(tex + "\n")
    print(tex)
    print(f"\n% best |DSI| {peak:.5f} at bias {best['bias_mv']}, "
          f"period {best['period_deg']}, tf {best['tf_hz']}")
    print(f"% both pairs opposed at {len(both)}/{len(unsat)} "
          f"unsaturated points ({100 * frac:.0f}%)")


if __name__ == "__main__":
    main()
