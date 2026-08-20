"""All tunable parameters, each with a provenance comment.

Provenance tags:
    [MEASURED]   read off this v783 download; see sources.py
    [PUBLISHED]  from a paper, and I checked the paper
    [UNVERIFIED] from the spec's recollection; NOT yet checked against source
    [OURS]       our engineering choice, not a biological claim

Anything tagged [UNVERIFIED] is a live risk. The spec is explicit about this:
"Verify the LIF constants against the paper. They are the difference between a
working sim and a silent one."
"""

from __future__ import annotations

from pathlib import Path

# ==========================================================================
# Paths
# ==========================================================================

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "data" / "raw"
PROCESSED_DIR = REPO / "data" / "processed"

# ==========================================================================
# Graph construction
# ==========================================================================

SYN_THRESHOLD = 5
# [MEASURED] Matches Codex's own published threshold. At >=5 the Buhmann graph
# is 2,710,038 edges / 31,578,726 synapses over 138,533 neurons.

SIGN = {
    "ACH": +1,   # [PUBLISHED] acetylcholine, the fly's main excitatory transmitter
    "GABA": -1,  # [PUBLISHED] standard inhibition, as in vertebrates
    "GLUT": -1,  # [PUBLISHED] *** INHIBITORY IN FLIES *** via GluCl chloride
                 # channels. This is the single most common porting error.
                 # [MEASURED] 18.1% of edges -- getting it wrong flips nearly a
                 # fifth of the brain. Asserted in tests.
    "DA": +1,    # [OURS] dopamine. Really a slow modulator acting over seconds;
    "OCT": +1,   # [OURS] octopamine.  flattening these three to fast excitation
    "SER": +1,   # [OURS] serotonin.   is a real approximation with real cost.
    "UNK": +1,   # [OURS] default to excitatory: 58.5% of the brain is
                 # cholinergic, so it is the least-wrong guess.
}

HISTAMINE_SIGN = -1
# [PUBLISHED] Photoreceptors release histamine, which opens HisCl1 chloride
# channels on lamina targets and INHIBITS them -- the classic sign inversion at
# the first visual synapse.
# [MEASURED] Histamine is not one of FlyWire's six predicted transmitters, so
# R1-6 output edges are predicted ACH 83% / GLUT 14% / GABA 3%. Left alone, 83%
# of photoreceptor output would be signed backwards and the entire optic lobe
# would read a negative image. graph.py overrides these.

PHOTORECEPTOR_HANDLES = ("R1-6", "R7", "R8")
# [MEASURED] All three resolve cleanly in v783; R1-6 is 8,456 neurons with 95%
# carrying output edges.

# ==========================================================================
# LIF dynamics
# ==========================================================================
#
# Source, read directly: Shiu, Sterne, Spiller et al., "A leaky integrate-and-
# fire computational model based on the connectome of the entire adult
# Drosophila brain reveals insights into sensorimotor processing", bioRxiv
# 2023.05.02.539144 (published as Nature 634, 210-219, 2024). Methods section.
#
# ---- THE MODEL FORM IS NOW VERIFIED ----
#
# Two coupled ODEs, with "alpha-synapse dynamics":
#
#     tau_mem * dv/dt = (V_rest - v) + g
#     tau_syn * dg/dt = -g
#
# and on an upstream spike:   g <- g + w,   w = sign * syn_count * W_SYN
#
# g is a VOLTAGE OFFSET in millivolts, not a conductance, despite the name.
# It shifts the potential the neuron decays *toward*; it decays to 0 with
# tau_syn, after which v relaxes back to V_rest. On firing, v resets to V_rest
# and is frozen for the refractory period.
#
# The spec's step equation is therefore CORRECT as written -- it is exactly
# this form with i_syn playing the role of g. An earlier worry in this project
# that TAU_SYN might not exist, and that including it would cost a factor of
# TAU_SYN/TAU_MEM in gain, was wrong: the paper does filter, and the spec
# matches it.
#
# One genuine ambiguity remains: the paper says "alpha-synapse" but describes a
# SINGLE exponential decay (g jumps, then decays with one tau). A true alpha
# synapse is a two-stage filter, t*exp(-t/tau). The prose describes the single
# exponential, so that is what we implement; flagged here in case it matters.
#
# ---- THE VALUES, READ OFF TABLE 1 ----
#
# Recovered from the paper's Table 1 (a bitmap, served as T1.medium.gif).
# Verbatim, with the paper's own attributions:
#
#   V_resting    = -52 * mV      Kakaria and de Bivort, 2017
#   V_reset      = -52 * mV      Kakaria and de Bivort, 2017
#   V_threshold  = -45 * mV      Kakaria and de Bivort, 2017
#   R_mbr        = 10 * Mohm     Kakaria and de Bivort, 2017
#   C_mbr        = .002 * uF     Kakaria and de Bivort, 2017
#   T_mbr        = C_mbr * R_mbr   (RC time constant -> 20 ms)
#   T_refractory = 2.2 * ms      Kakaria and de Bivort 2017; Lazar et al. 2021
#   Tau          = 5 * ms        Synapse decay, Jurgensen et al. 2021
#   T_dly        = 1.8 * ms      Spike-to-membrane delay, Paul et al. 2015
#   W_syn        = .275 * mV     Free parameter
#
# Every value the spec guessed turned out to be right. T_mbr is not stated
# directly -- it is DERIVED as R*C = 10 Mohm * 0.002 uF = 20 ms.
#
# T_dly, however, the spec omitted entirely, and so did our first
# implementation. See below.

V_REST = -52e-3    # V   [PUBLISHED] Table 1
V_THRESH = -45e-3  # V   [PUBLISHED] Table 1
V_RESET = -52e-3   # V   [PUBLISHED] Table 1
TAU_MEM = 20e-3    # s   [PUBLISHED] Table 1, derived: R_mbr * C_mbr
TAU_SYN = 5e-3     # s   [PUBLISHED] Table 1 ("Tau")
T_REFRAC = 2.2e-3  # s   [PUBLISHED] Table 1

R_MBR = 10e6       # ohm [PUBLISHED] Table 1, kept for provenance
C_MBR = 0.002e-6   # F   [PUBLISHED] Table 1, kept for provenance

T_DLY = 1.8e-3     # s   [PUBLISHED] Table 1 -- "Time delay from spike to
                   #     change in membrane potential", Paul et al. 2015.
# *** THE SPEC MISSED THIS ONE. *** Spec 5 has no delay term at all, and its
# step equation propagates the previous step's spikes, which imposes exactly
# one dt = 0.5 ms. The paper uses 1.8 ms, 3.6x longer.
#
# This is not cosmetic in a recurrent network of 139k neurons: conduction delay
# sets the frequency of any oscillation the network can support and changes the
# coincidence window over which the ~267 synapses needed to fire a cell must
# arrive. At dt = 0.5 ms it quantises to 4 steps = 2.0 ms, 11% long.

# ---- W_SYN is not a biological constant. It is fitted. ----
#
# Direct quote: "except for W_syn, the single free parameter of the model...
# We chose W_syn such that activation of sugar GRNs at 100 Hz resulted in
# roughly 80% of maximal MN9 firing."
#
# So the one free parameter of the whole model was tuned ON THE M2 RESULT.
# Two consequences, and both matter:
#
#   1. Copying their number is the wrong move. They fitted it against FlyWire
#      v630 (127,400 neurons); we are on v783 (139,255) with a different
#      synapse pipeline, so their value does not transfer. Replicate the
#      PROCEDURE instead -- see calibrate_w_syn() in lif.py.
#
#   2. M2 is therefore not a clean validation. Sugar -> PER comes out right
#      partly by construction, because the gain was chosen to make it come out
#      right. What M2 genuinely tests is everything ELSE: signs, the id map,
#      pre/post orientation, and whether any single scalar gain exists that
#      produces the correct behaviour. That is still worth having -- a broken
#      graph has no such scalar -- but it is a weaker claim than "the
#      connectome predicts PER", and the writeup should say so.
W_SYN_PAPER = 0.275e-3
# [PUBLISHED] Table 1. Their fit, against FlyWire v630 (127,400 neurons) with
# the Buhmann/Eckstein pipeline of the day. The spec's recollection of this
# number was exactly right -- it is simply not OUR number, because the graph
# underneath it is different.

W_SYN_FITTED = 0.1655e-3
# [FITTED] 2026-08-20, by experiments/m2_per.py against the Buhmann v783 graph.
# Repeat fits land at 0.16470 / 0.16637 mV, so treat this as 0.165 mV +-1%;
# the bisection endpoint is noisier than the value is meaningful.
# at syn_count>=5 (2,710,038 edges) with the LIF constants below. Replicates the
# paper's own protocol: sugar GRNs at 100 Hz Poisson -> 80.9% of the saturating
# MN9 response. The spec's unverified guess of 0.275 mV is 1.65x this; that
# value came from a different snapshot and a different synapse pipeline, which
# is exactly why it does not transfer.
#
# At this gain one synapse moves the membrane 26.2 uV at peak, so ~267
# coincident synapses drive a neuron from rest.
#
# ---- MEASURED OPERATING WINDOW (m2_per.py --sweep, 2026-08-20) ----
#
#   x fitted   W_SYN(mV)   MN9@sugar   MN9@bitter   brain active
#     0.30       0.050         0.00        0.00         0.10%   silent
#     0.70       0.117         0.78        0.00         0.12%   silent
#     0.85       0.141         4.69        0.00         0.14%   weak
#     1.00       0.166        53.69        0.00         0.26%   <- fitted
#     1.20       0.200        72.65        0.00         0.33%   selective
#     1.50       0.250        82.04        0.00         0.40%   selective
#     2.00       0.333        99.01        1.33         0.62%   control leaking
#     3.00       0.499       123.81       29.68         8.71%   CONTROL BROKEN
#     6.00       0.998        81.75       97.86        21.21%   runaway, no
#                                                               selectivity
#
# The QUALITATIVE result -- sugar drives MN9, bitter does not -- holds across
# roughly 0.85x to 1.5x the fitted gain, and the seizure boundary is a clear 3x
# away. That is a wider margin than the steep dose-response first suggested.
#
# The QUANTITATIVE result does not: MN9 goes 4.7 -> 82 Hz across that same band.
# So any NUMBER this project reports is gain-dependent even where the sign of
# the effect is not. The looming crossover angle above all: nothing calibrates
# the VISUAL pathway independently, it simply inherits the gain fitted on taste.
# Publish that number with a sweep beside it or it is a property of this scalar
# rather than of the connectome.

W_SYN = W_SYN_FITTED   # V per synapse

W_SYN_CALIBRATION = dict(
    stim_handle="sugar_GRN",
    stim_rate_hz=100.0,
    readout_handle="MN9",
    target_fraction_of_max=0.80,
    trials=30,
    duration_s=1.0,
)
# [PUBLISHED] The paper's own fitting protocol, to be replicated on v783.
# Stimulation is POISSON, not constant current (spec 5 implies constant).
# Sugar GRNs were driven across 10-200 Hz; "maximal MN9 firing" is the
# saturating end of that sweep.

DT = 0.5e-3        # s   [OURS] our timestep. 57 substeps per Doom tic.
                   #     The paper used Brian2 and does not state its dt here.

NOISE_STD = 0.0    # V per step  [OURS]
# Background drive. The paper has no spontaneous firing, and connectome LIF
# models characteristically show sensory input propagating two or three layers
# and then dying with nothing to push marginal neurons over threshold. Expect
# to need this above zero. It belongs beside W_SYN in the order-of-suspicion
# list, not as an afterthought.

# ==========================================================================
# Where we differ from the paper, deliberately or otherwise
# ==========================================================================

PAPER_SNAPSHOT = "v630"      # 127,400 neurons
OUR_SNAPSHOT = "v783"        # 139,255 neurons -- the spec pins this
# [PUBLISHED] "All 127,400 proofread neurons from Flywire materialization
# version 630 are included in the model." We are two materializations newer,
# which is the spec's explicit choice. Expect quantitative differences.

# Their NT rule, for reference: Eckstein et al. 2020 per-synapse predictions,
# cleft score cutoff 50, take the top prediction per presynaptic site, and if
# >half of a neuron's presynaptic sites are GABA or GLUT the whole neuron is
# inhibitory. We arrive at the same place for free -- measured on our download,
# every presynaptic neuron already carries exactly one nt_type, so the majority
# vote is a no-op. See sources.DALES_LAW_HOLDS.

# ==========================================================================
# Doom I/O  --  not used before M5
# ==========================================================================

DOOM_TICRATE = 35              # [PUBLISHED] Doom runs at 35 tics/s
SUBSTEPS_PER_TIC = round((1.0 / DOOM_TICRATE) / DT)   # 57
