#!/usr/bin/env python3
"""Emit the M8 LaTeX table straight from the JSON the figure uses.

Single source of truth: table and figure cannot disagree.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROLE = {
    "LHN":   "innate valence",
    "DNp01": "escape (giant fibre)",
    "DNa02": "steering",
    "BPN":   "forward walking",
    "MDN":   "backward walking",
    "LC4":   r"\emph{visual control}",
}


def sign_p(x):
    """Two-sided sign test on the count agreeing with the mean direction."""
    n = len(x)
    k = max(int((x > 0).sum()), int((x < 0).sum()))
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2 * tail / 2 ** n), k, n


def fmt_p(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def main():
    intact = json.loads((HERE / "data" / "m8_intact.json").read_text())
    shuf = json.loads((HERE / "data" / "m8_shuffled.json").read_text())
    pops = list(intact["watch"])

    rows = []
    for p in pops:
        a = np.asarray(intact["deltas"][p])
        b = np.asarray(shuf["deltas"][p])
        pa, ka, n = sign_p(a)
        pb, kb, _ = sign_p(b)
        rows.append(
            f"\\cell{{{p}}} & ${a.mean():+.2f}$ & {a.std():.2f} & {ka}/{n} & "
            f"{fmt_p(pa)} & ${b.mean():+.2f}$ & {b.std():.2f} & {kb}/{n} & "
            f"{ROLE[p]}\\\\"
        )

    n = len(intact["arms"])
    # Derive the shuffled floor from the data rather than restating a range
    # that was true of an earlier run.
    shuf_abs = [abs(float(np.mean(shuf["deltas"][p]))) for p in pops]
    lo, hi = min(shuf_abs), max(shuf_abs)
    tex = f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Effect of the olfactory channel, open loop, {n} seeds $\\times$
{intact['tics']} tics. $\\Delta$ is the mean on-minus-off difference in Hz;
``sign'' counts seeds agreeing with the mean direction, with $p$ from a
two-sided sign test. In the intact connectome every motor and valence
population moves consistently; under a degree-preserving shuffle of the same
graph every effect falls to the ${lo:.2f}$--${hi:.2f}$\\,Hz scale of the
\\cell{{LC4}} control. Sign consistency alone does not separate the arms --- the shuffle
retains some of it --- so the comparison to read is magnitude. \\cell{{LC4}} is a
visual cell whose activity should be unaffected by smell; it carries a small
offset in both arms, at
${abs(np.mean(intact['deltas']['LHN']) / np.mean(intact['deltas']['LC4'])):.0f}\\times$
smaller magnitude than the lateral horn response it bounds.}}
\\label{{tab:m8}}
\\begin{{tabular}}{{l rrcc @{{\\hskip 1.6em}} rrc l}}
\\toprule
& \\multicolumn{{4}}{{c}}{{\\bf Intact}} & \\multicolumn{{3}}{{c}}{{\\bf Degree-preserving shuffle}} & \\\\
\\cmidrule(r){{2-5}}\\cmidrule(lr){{6-8}}
Population & $\\Delta$ (Hz) & s.d. & sign & $p$ & $\\Delta$ (Hz) & s.d. & sign & Role\\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}"""

    (HERE / "tab_m8.tex").write_text(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()
