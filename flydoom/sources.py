"""Ground truth about the v783 files, discovered by inspecting them.

The spec assumed a schema that does not exist. What Codex actually ships:

  classification.csv        root_id, flow, super_class, class, sub_class,
                            hemilineage, side, nerve
                            *** NO cell_type column ***
  consolidated_cell_types   root_id, primary_type, additional_type(s)
                            <- this is where cell types actually live
  visual_neuron_types       root_id, type, family, subsystem, category, side
                            <- optic lobe typing, 95,079 neurons
  labels.csv                root_id, label, user_name, ...  free-text community
                            annotations; the ONLY place BPN and aIPg appear
  connections_princeton     pre, post, neuropil, syn_count, nt_type
                            *** nt_type is 100% NULL -- unusable ***
  connections_princeton_
    no_threshold            same columns, nt_type 100% POPULATED
                            <- use this, apply our own threshold

Measured on the actual download, 2026-08-20.
"""

from __future__ import annotations

# -- file inventory --------------------------------------------------------

CLASSIFICATION = "classification.csv.gz"
CELL_TYPES = "consolidated_cell_types.csv.gz"
VISUAL_TYPES = "visual_neuron_types.csv.gz"
LABELS = "labels.csv.gz"
COLUMN_ASSIGNMENT = "column_assignment.csv.gz"

# Two independent synapse-detection pipelines ship for the same brain. Both
# have nt_type populated; the pre-thresholded `connections_princeton.csv` does
# NOT (100% null) and is unusable, so we always take a no-threshold file and
# apply SYN_THRESHOLD ourselves.
#
#                          rows      synapses     at >=5 syn: rows / synapses
#   buhmann          16,847,997    54,492,922      2,710,038 / 31,578,726
#   princeton        22,285,323    76,944,499      3,754,052 / 47,169,614
#
# Buhmann reproduces the 54,492,922 figure published in Dorkenwald et al. 2024
# and quoted by the spec, so it is the default: it is the graph the papers --
# including Shiu et al., whose PER result M2 reproduces -- describe.
#
# If M2 fails, swapping to Princeton is a cheap and principled diagnostic
# before touching W_SYN. Different pipeline, same brain.
CONNECTIONS = "connections_buhmann_no_threshold.csv.gz"
CONNECTIONS_ALT = "connections_princeton_no_threshold.csv.gz"

SYN_THRESHOLD = 5
"""Minimum synapses per connection, matching Codex's own published threshold."""

# -- measured magnitudes (v783, this download) -----------------------------
# These replace the spec's numbers, which conflated synapses with edges.

N_NEURONS = 139_255
"""Rows in classification.csv. Matches the spec and the published figure."""

# Buhmann (the default), measured on this download.
N_EDGES_UNTHRESHOLDED = 16_847_997
N_SYNAPSES_UNTHRESHOLDED = 54_492_922
"""Matches the published figure exactly."""

N_EDGES_THRESHOLDED = 2_710_038
"""Rows at syn_count >= 5. The spec's `54_000_000 < n_edges` assert is wrong by
~20x -- it was quoting a synapse count as an edge count. 54.5M is the
UNTHRESHOLDED SYNAPSE total, which is N_SYNAPSES_UNTHRESHOLDED above."""

N_SYNAPSES_THRESHOLDED = 31_578_726
N_NEURONS_CONNECTED = 138_533

NT_FRACTIONS = {
    "ACH": 0.5854, "GABA": 0.1982, "GLUT": 0.1811,
    "DA": 0.0190, "SER": 0.0095, "OCT": 0.0068,
}
"""Fraction of untresholded edges by predicted transmitter. Note GLUT is 18%,
not the ~10% commonly assumed -- signing it as excitatory would flip nearly a
fifth of the brain."""

DALES_LAW_HOLDS = True
"""Measured: 0 of 139,003 presynaptic neurons carry more than one nt_type across
their output edges. The Princeton pipeline already applied Dale's law, so the
per-edge and per-neuron signing strategies are identical here and the
'majority vote per neuron' step is a no-op. Kept as an assertion in graph.py
rather than a transformation."""

# -- the histamine override ------------------------------------------------

PHOTORECEPTOR_TYPES = ("R1-6", "R7", "R8")

HISTAMINERGIC_SIGN = -1
"""Photoreceptors release HISTAMINE, which opens chloride channels on their
lamina targets (HisCl1) and therefore INHIBITS them. This is the classic sign
inversion at the first visual synapse: light -> photoreceptor depolarises ->
L1/L2 hyperpolarise.

Histamine is NOT one of FlyWire's six predicted transmitters, so the classifier
had to guess. Measured on R1-6 output edges in this download:

    ACH 83%   GLUT 14%   GABA 3%

ACH maps to EXCITATORY in our table, so ~83% of photoreceptor output would be
signed backwards, inverting contrast for the entire optic lobe. Every
downstream motion and looming computation would be reading a negative image.

graph.py MUST override the predicted transmitter for these types. This is not
a fudge -- it is correcting a known blind spot in the prediction model with
established physiology.
"""
