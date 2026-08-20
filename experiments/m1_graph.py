#!/usr/bin/env python3
"""M1 — build the connectome graph and check it.

Acceptance, from spec 3 but with the SPEC'S NUMBERS CORRECTED against the
actual v783 download:

    n_neurons == 139,255                       (spec was right)
    n_edges   ~= 2,710,038 at syn_count >= 5   (spec said >54,000,000 -- it was
                                                quoting a SYNAPSE total as an
                                                edge count, off by ~20x)
    nt_type subset of the seven known values
    builds in under 60 s, occupies under 1 GB on GPU

Plus two checks the spec asks for by name:

    GLUT must be inhibitory
    photoreceptor output must be overridden to inhibitory (histamine)

    python experiments/m1_graph.py [--princeton] [--no-histamine] [--save]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from flydoom import config, sources  # noqa: E402
from flydoom.graph import ConnectomeGraph  # noqa: E402

USE_COLOR = sys.stdout.isatty()


def paint(t: str, c: str) -> str:
    return f"\033[{c}m{t}\033[0m" if USE_COLOR else t


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((ok, name, detail))
        return ok

    def render(self) -> bool:
        for ok, name, detail in self.rows:
            mark = paint("PASS", "32") if ok else paint("FAIL", "1;31")
            print(f"  {mark}  {name:<44} {paint(detail, '90')}")
        return all(ok for ok, _, _ in self.rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="flydoom M1 — graph build")
    ap.add_argument("--princeton", action="store_true",
                    help="use the Princeton pipeline instead of Buhmann")
    ap.add_argument("--no-histamine", action="store_true",
                    help="skip the photoreceptor sign override (to see the damage)")
    ap.add_argument("--save", action="store_true",
                    help="write the graph to data/processed/")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src = sources.CONNECTIONS_ALT if args.princeton else sources.CONNECTIONS
    print(paint("flydoom M1 — connectome graph", "1"))
    print(paint("=" * 74, "90"))
    print(f"source     {src}")
    print(f"threshold  syn_count >= {config.SYN_THRESHOLD}")
    print(f"histamine  {'DISABLED' if args.no_histamine else 'override active'}\n")

    g = ConnectomeGraph.build(
        connections_file=src,
        apply_histamine_override=not args.no_histamine,
    )
    r = g.report
    assert r is not None

    print(paint("BUILD", "1;36"))
    print(f"  neurons (index space)      {r.n_neurons:>12,}")
    print(f"  edges, raw                 {r.n_edges_raw:>12,}")
    print(f"  edges, syn_count >= {config.SYN_THRESHOLD}      {r.n_edges_thresholded:>12,}")
    print(f"  edges, both ends mapped    {r.n_edges_kept:>12,}"
          f"   (dropped {r.n_edges_dropped_unmapped:,})")
    print(f"  synapses represented       {r.n_synapses:>12,}")
    print(f"  neurons with >=1 edge      {r.n_neurons_connected:>12,}")
    print(f"  excitatory / inhibitory    {r.n_excitatory:>12,} / {r.n_inhibitory:,}"
          f"   ({100 * r.n_inhibitory / g.n_edges:.1f}% inhibitory)")
    print(f"  build time                 {r.build_seconds:>12.1f} s")
    print(f"  host memory                {r.nbytes / 2**20 if hasattr(r,'nbytes') else g.nbytes / 2**20:>12.1f} MB")

    print(f"\n{paint('TRANSMITTERS', '1;36')}")
    for nt, n in sorted(r.nt_counts.items(), key=lambda kv: -kv[1]):
        s = config.SIGN.get(nt, 1)
        tag = paint("excitatory", "33") if s > 0 else paint("inhibitory", "36")
        print(f"  {nt:<6} {n:>10,}  ({100 * n / r.n_edges_thresholded:5.2f}%)  {tag}")

    if not args.no_histamine:
        print(f"\n{paint('HISTAMINE OVERRIDE', '1;36')}")
        print(f"  photoreceptors resolved    {r.n_photoreceptor_neurons:>12,}")
        print(f"  edges re-signed inhibitory {r.n_photoreceptor_edges_flipped:>12,}")
        print(paint("  (predicted ACH/GLUT/GABA; true transmitter is histamine,", "90"))
        print(paint("   which is not one of FlyWire's six predicted classes)", "90"))

    # ---------------- acceptance ----------------
    c = Checks()

    c.check(r.n_neurons == sources.N_NEURONS,
            "n_neurons == 139,255", f"got {r.n_neurons:,}")

    expected = (sources.N_EDGES_THRESHOLDED if not args.princeton else 3_754_052)
    lo, hi = int(expected * 0.98), int(expected * 1.02)
    c.check(lo <= g.n_edges <= hi,
            f"n_edges within 2% of {expected:,}", f"got {g.n_edges:,}")

    c.check(set(r.nt_counts) <= {"ACH", "GABA", "GLUT", "DA", "OCT", "SER", "UNK"},
            "nt_type values all known", ", ".join(sorted(r.nt_counts)))

    c.check(r.n_edges_dropped_unmapped / max(r.n_edges_thresholded, 1) < 0.01,
            "<1% of edges touch an unmapped neuron",
            f"{r.n_edges_dropped_unmapped:,} dropped")

    c.check(config.SIGN["GLUT"] == -1,
            "GLUT is INHIBITORY (fly, not vertebrate)",
            "GluCl chloride channels")

    c.check(np.all(g.signed_syn != 0), "no zero-weight edges")
    c.check(g.pre_idx.dtype == np.int32 and g.post_idx.dtype == np.int32,
            "index arrays are int32", str(g.pre_idx.dtype))
    c.check(g.signed_syn.dtype == np.float32, "weights are float32")
    c.check(int(g.pre_idx.max()) < r.n_neurons and int(g.post_idx.max()) < r.n_neurons,
            "all indices inside the index space",
            f"max {max(int(g.pre_idx.max()), int(g.post_idx.max())):,}")

    c.check(r.build_seconds < 60, "builds in under 60 s",
            f"{r.build_seconds:.1f} s")

    if not args.no_histamine:
        c.check(r.n_photoreceptor_edges_flipped > 0,
                "photoreceptor edges re-signed",
                f"{r.n_photoreceptor_edges_flipped:,} flipped")

    # ---------------- gpu ----------------
    print(f"\n{paint('GPU', '1;36')}")
    try:
        import torch
    except ImportError:
        print(paint("  torch not installed — GPU residency unchecked", "33"))
        print(paint("  uv pip install --python .venv/bin/python torch", "90"))
    else:
        if not torch.cuda.is_available():
            print(paint(f"  CUDA unavailable (torch {torch.__version__}) — "
                        "GPU residency unchecked", "33"))
        else:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            pre, post, w = g.to_torch(args.device)
            used = torch.cuda.memory_allocated() - before
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            print(f"  device                     {name}")
            print(f"  graph resident             {used / 2**20:.1f} MB "
                  f"of {total / 2**30:.1f} GB")
            c.check(used < 2**30, "graph under 1 GB on GPU",
                    f"{used / 2**20:.1f} MB")

            # A real scatter-add, to prove the representation works end to end.
            spiked = torch.zeros(g.n_neurons, dtype=torch.float32, device=args.device)
            spiked[pre[:1000].long()] = 1.0
            cur = torch.zeros(g.n_neurons, dtype=torch.float32, device=args.device)
            cur.index_add_(0, post.long(), w * spiked[pre.long()])
            torch.cuda.synchronize()
            c.check(bool((cur != 0).any()), "scatter-add propagates",
                    f"{int((cur != 0).sum()):,} neurons received current")

    print(f"\n{paint('ACCEPTANCE', '1')}")
    ok = c.render()

    if args.save:
        out = g.save()
        print(f"\nsaved to {out}")

    print(f"\n{paint('=' * 74, '90')}")
    if ok:
        print(paint("VERDICT: M1 PASS", "1;32"))
        return 0
    print(paint("VERDICT: M1 FAIL — do not proceed to M2", "1;31"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
