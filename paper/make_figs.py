#!/usr/bin/env python3
"""Figures for the flydoom preprint. Run after m8_*.json exist."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, RegularPolygon

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
})

INK   = "#1a1a1a"
INTACT = "#1f5c8b"     # deep blue
SHUF   = "#b0b0b0"     # grey
WARM   = "#c1440e"     # rust, for the failure/emphasis
SLOW   = "#7a4fa3"     # violet


# ---------------------------------------------------------------- figure 1
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    YB = 24.0     # box centre line
    H = 11.0

    def box(x, w, label, sub, fc="none", ec=INK, lw=0.9):
        ax.add_patch(FancyBboxPatch((x, YB - H / 2), w, H,
                     boxstyle="round,pad=0.35,rounding_size=1.1",
                     fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, YB + 2.0, label, ha="center", va="center",
                fontsize=8, color=INK, weight="bold")
        ax.text(x + w / 2, YB - 2.4, sub, ha="center", va="center",
                fontsize=6.5, color="#555", linespacing=1.35)

    def arrow(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, YB), (x1, YB), arrowstyle="-|>",
                     mutation_scale=8, lw=1.0, color=INK, shrinkA=0, shrinkB=0))

    spans = [(0.5, 14.5), (19.0, 16.5), (40.0, 19.5), (64.0, 15.0), (83.5, 15.5)]
    box(*spans[0], "Doom frame", "$130^\\circ$ viewport")
    box(*spans[1], "Ommatidia", "796 columns / eye\n$\\pm40^\\circ$ splay")
    box(*spans[2], "Connectome", "139,255 neurons\n2.71M edges  ·  frozen",
        fc="#eef3f7", ec=INTACT, lw=1.3)
    box(*spans[3], "Descending", "8 characterized\nneurons")
    box(*spans[4], "Buttons", "delta + Schmitt")
    for (x0, w0), (x1, _) in zip(spans[:-1], spans[1:]):
        arrow(x0 + w0 + 0.4, x1 - 0.4)

    # closed loop, routed clear beneath the boxes
    yl = 12.0
    ax.plot([91.2, 91.2], [YB - H / 2, yl], color=WARM, lw=1.0)
    ax.plot([91.2, 7.7], [yl, yl], color=WARM, lw=1.0)
    ax.add_patch(FancyArrowPatch((7.7, yl), (7.7, YB - H / 2),
                 arrowstyle="-|>", mutation_scale=8, lw=1.0, color=WARM,
                 shrinkA=0, shrinkB=0))
    ax.text(49.5, 9.4, "closed loop  ·  57 LIF substeps per tic  ·  $1.01\\times$ realtime",
            ha="center", va="top", fontsize=7.2, color=WARM, style="italic")

    # side inputs, well separated
    for x, lab in ((43.0, "taste\nhealth / damage"),
                   (57.0, "smell\n2 labelled lines")):
        ax.add_patch(FancyArrowPatch((x, 34.0), (x, YB + H / 2 + 0.4),
                     arrowstyle="-|>", mutation_scale=7, lw=0.85, color="#555"))
        ax.text(x, 36.2, lab, ha="center", va="bottom", fontsize=6.5,
                color="#555", linespacing=1.35)

    ax.text(0.5, 36.4, "nothing is trained", fontsize=7.6, style="italic",
            color=INTACT, weight="bold", va="bottom")
    fig.savefig(FIGS / "fig1_pipeline.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_mechanism():
    fig = plt.figure(figsize=(7.0, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 0.95], wspace=0.5)

    # -- (a) temporal order is invisible to thr(A+B)
    ax = fig.add_subplot(gs[0])
    t = np.linspace(0, 1, 500)
    bump = lambda c, w=0.05: np.exp(-((t - c) ** 2) / (2 * w ** 2))
    early, late = 0.34, 0.60

    rows = [
        (3.30, bump(early), bump(late), "$A$ then $B$"),
        (1.85, bump(late), bump(early), "$B$ then $A$"),
    ]
    for base, a, b, lab in rows:
        ax.plot(t, base + a, color=INTACT, lw=1.15)
        ax.plot(t, base + b, color=SLOW, lw=1.15, ls="--")
        ax.text(0.015, base + 1.16, lab, fontsize=7.2, color=INK, va="top")

    sum1 = bump(early) + bump(late)
    sum2 = bump(late) + bump(early)
    ax.plot(t, 0.28 + sum1 * 0.52, color=WARM, lw=2.4, alpha=0.85)
    ax.plot(t, 0.28 + sum2 * 0.52, color=INK, lw=0.9, ls=":")
    ax.text(0.015, 1.52, "$A+B$   both cases", fontsize=7.2, color=WARM, va="top")
    ax.axhline(0.86, color="#888", lw=0.7, ls="-.")
    ax.text(0.985, 0.90, "threshold", fontsize=6.4, color="#666",
            ha="right", va="bottom")

    ax.set_xlim(0, 1); ax.set_ylim(0.05, 4.75)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("time", labelpad=1)
    ax.set_title("(a)  $\\mathrm{thr}(A{+}B)$ is order-blind", loc="left", pad=6)

    # -- (b) measured rates of the two correlator arms
    ax = fig.add_subplot(gs[1])
    lo = np.array([2.0, 78.0]); hi = np.array([9.0, 130.0])
    y = np.array([1.0, 0.0])
    ax.barh(y, hi - lo, left=lo, height=0.34,
            color=[INTACT, SLOW], edgecolor=INK, lw=0.7)
    notes = ["fast, excitatory", "slow, inhibitory"]
    for yy, l, h, nt in zip(y, lo, hi, notes):
        ax.text(h + 5, yy + 0.17, f"{l:.0f}–{h:.0f} Hz", va="center",
                fontsize=6.8, color="#333")
        ax.text(h + 5, yy - 0.17, nt, va="center", fontsize=6.3, color="#777")
    ax.set_yticks(y)
    ax.set_yticklabels(["Mi1", "Mi9"])
    ax.set_xlabel("measured firing rate (Hz)")
    ax.set_xlim(0, 232); ax.set_ylim(-0.62, 1.62)
    ax.annotate("", xy=(196, 0.06), xytext=(196, 0.94),
                arrowprops=dict(arrowstyle="<->", color=WARM, lw=0.9))
    ax.text(203, 0.5, "$37\\times$", color=WARM, fontsize=9,
            weight="bold", ha="left", va="center")
    ax.set_title("(b)  the fast arm is starved", loc="left", pad=6)

    # -- (c) DSI achieved vs the experimental selection threshold
    ax = fig.add_subplot(gs[2])
    # Read from the same grid the table is generated from, so the figure and
    # tab_dsi cannot disagree about what was measured.
    grid = json.loads((HERE / "data" / "m3_dsi.json").read_text())
    usable = [p for p in grid["points"]
              if p["saturation"] < grid["saturation_threshold"]]
    best = max(usable, key=lambda p: max(abs(v) for v in p["dsi"].values()))
    names = ["T4a", "T4b", "T5a", "T5b"]
    vals = np.abs([best["dsi"][t] for t in names])
    peak = float(vals.max())
    ax.bar(names, vals, color=INTACT, edgecolor=INK, lw=0.7, width=0.62)
    ax.axhline(peak, color=INK, lw=0.8, ls="--")
    ax.text(3.45, peak * 1.2, f"best unsaturated  {peak:.4f}", fontsize=6.3,
            ha="right", va="bottom", color=INK)
    ax.axhline(0.5, color=WARM, lw=1.3)
    ax.text(3.45, 0.62, "experimental selection\nthreshold  $\\mathrm{DSI}>0.5$",
            fontsize=6.3, ha="right", va="bottom", color=WARM, linespacing=1.3)
    ax.set_yscale("log"); ax.set_ylim(6e-4, 4.0)
    ax.set_ylabel("$|\\mathrm{DSI}|$", labelpad=1)
    ax.set_title("(c)  ${\\sim}40\\times$ short, sign unstable", loc="left",
                 pad=6)

    fig.savefig(FIGS / "fig2_mechanism.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_olfaction():
    intact = json.loads((HERE / "data" / "m8_intact.json").read_text())
    shuf = json.loads((HERE / "data" / "m8_shuffled.json").read_text())

    def arm(rec, pop, key):
        return np.array([a[key][pop] for a in rec["arms"]])

    fig = plt.figure(figsize=(7.0, 2.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4], wspace=0.46)

    # -- (a,b) paired per-seed lines for the lateral horn
    axes = []
    for k, (rec, title, col) in enumerate((
            (intact, "(a)  intact connectome", INTACT),
            (shuf, "(b)  degree-preserving shuffle", SHUF))):
        ax = fig.add_subplot(gs[k]); axes.append(ax)
        off, on = arm(rec, "LHN", "off"), arm(rec, "LHN", "on")
        for o, n in zip(off, on):
            ax.plot([0, 1], [o, n], color=col, lw=0.95, alpha=0.8,
                    marker="o", ms=2.7, mfc=col, mec="none")
        ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1])
        ax.set_xticklabels(["smell off", "smell on"])
        ax.set_title(title, loc="left", pad=6)
        d = on - off
        ax.text(0.5, 0.955, f"$\\Delta={d.mean():+.2f}$ Hz",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.4,
                color=INK if k == 0 else "#666")
        base = (f"dormant at {off.mean():.2f} Hz" if k == 0
                else f"idles at {off.min():.1f}–{off.max():.1f} Hz")
        by = 0.865 if k == 0 else 0.045
        ax.text(0.5, by, base, transform=ax.transAxes, ha="center",
                va="top" if k == 0 else "bottom", fontsize=6.5,
                color="#777", style="italic")
        if k == 0:
            ax.set_ylabel("lateral horn rate (Hz)")
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi * 1.12)

    # -- (c) effect size per population, both arms
    ax = fig.add_subplot(gs[2])
    pops = list(intact["watch"])
    yi = np.arange(len(pops))[::-1]
    di = np.array([np.mean(intact["deltas"][p]) for p in pops])
    si = np.array([np.std(intact["deltas"][p]) for p in pops])
    ds = np.array([np.mean(shuf["deltas"][p]) for p in pops])
    ax.barh(yi + 0.2, di, xerr=si, height=0.37, color=INTACT,
            edgecolor=INK, lw=0.6, label="intact",
            error_kw=dict(lw=0.7, ecolor="#444", capsize=1.6))
    ax.barh(yi - 0.2, ds, height=0.37, color=SHUF, edgecolor=INK, lw=0.6,
            label="shuffled")
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(yi)
    ax.set_yticklabels([f"$\\it{{{p}}}$" if p == "LC4" else p for p in pops])
    ax.set_xlabel("$\\Delta$ rate, smell on $-$ off (Hz)")
    ax.legend(frameon=False, loc="lower left", handlelength=1.1,
              borderaxespad=0.3)
    ax.set_title("(c)  the effect needs the wiring", loc="left", pad=6)
    ax.annotate("visual control", xy=(di[-1] + 0.4, yi[-1] + 0.2),
                xytext=(11.5, yi[-1] + 0.72), fontsize=6.4, color="#666",
                style="italic", ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.55,
                                shrinkA=1, shrinkB=1))

    fig.savefig(FIGS / "fig3_olfaction.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); print("fig1 ok")
    fig_mechanism(); print("fig2 ok")
    if (HERE / "data" / "m8_intact.json").exists():
        fig_olfaction(); print("fig3 ok")
    else:
        print("fig3 skipped - m8 json not ready")
