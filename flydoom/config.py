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
# Conductance-based synapses  --  THE LARGEST DEPARTURE FROM THE PAPER
# ==========================================================================
#
# Shiu et al. make inhibition SUBTRACTIVE: a spike adds a signed voltage offset
# to g, and v decays toward V_rest + g. That is fine for feedforward excitatory
# circuits -- M2 (taste) passes cleanly on it.
#
# It cannot compute motion. MEASURED: M3 finds no direction selectivity (T4a
# and T4b never show opposite-signed DSI) and M4 finds no looming selectivity
# (LPLC2 fires 0.42 Hz for expansion AND 0.42 Hz for contraction). Both are
# computations over the TEMPORAL ORDER of neighbouring columns, and with linear
# summation into a single threshold, threshold(A+B) is symmetric in A and B --
# "A then B" is indistinguishable from "B then A". Per-type delays and per-type
# time constants change WHEN terms arrive but not the linearity of their
# combination, which is why neither rescued it.
#
# A correlator needs MULTIPLICATION. Real circuits get it from SHUNTING
# inhibition: GABA and GluCl open chloride channels, raising membrane
# CONDUCTANCE and therefore DIVIDING the effect of excitation.
#
#     tau*dv/dt = (V_rest - v) + g_e*(E_exc - v) + g_i*(E_inh - v) + I_ext
#
# g_e and g_i are dimensionless, in units of the leak conductance. Note what
# this buys: the effective time constant becomes tau_mem/(1+g_e+g_i), so a
# strongly driven neuron is FASTER as well as less sensitive. That is the
# divisive nonlinearity a correlator is built from.
#
# With no synaptic input g_e = g_i = 0 and the equation reduces EXACTLY to the
# subtractive form, so M1.5's closed-form checks stay valid in both modes.

SYNAPSE_MODEL = "conductance"   # "subtractive" (paper) | "conductance" (ours)

E_EXC = 0.0        # V  [PUBLISHED] cation reversal for nicotinic ACh, ~0 mV
E_INH = -70e-3     # V  [PUBLISHED] chloride reversal for GABA-A / GluCl.
                   # CONFIRMED by direct measurement in the exact cell this
                   # project cares about: Groschner et al. 2022 record
                   # E_Glu = -71 mV and E_GABA = -68 mV in ADULT T4 IN VIVO,
                   # and state explicitly that these two are NOT subject to the
                   # space-clamp distortion that corrupts their E_ACh estimate.
                   # Caveat worth stating in any write-up: every published
                   # GABA/Cl reversal in adult Drosophila is whole-cell and so
                   # pipette-imposed. No dialysis-free (gramicidin perforated
                   # patch) E_Cl exists for any adult central neuron.
# Sits 18 mV below V_rest, so inhibition here both hyperpolarises AND shunts.
# Moving E_INH toward V_REST makes it purely divisive; that is the knob to
# sweep if shunting turns out to be the active ingredient.

G_SYN = 0.00278    # dimensionless conductance per synapse  [FITTED]

# ---- per-transmitter conductance: the parameter arm 2 needs ----
#
# G_SYN above is ONE fitted number applied to every synapse regardless of which
# receptor it acts on. That is the assumption this project has now shown to be
# not merely imprecise but self-contradictory: the optic lobe requires a g_i/g_e
# ratio the SEZ taste circuit cannot survive (bitter inverts from suppressing
# the proboscis to driving it at 168 Hz when inhibition is doubled globally).
#
# The connectome already labels every edge with its transmitter, and the model
# already uses PUBLISHED per-class reversal potentials (E_EXC, E_INH). What is
# missing is the conductance, which genuinely differs by receptor:
#
#     ACH  -> nAChR      cation channel
#     GABA -> Rdl        GABA-A, chloride
#     GLUT -> GluCl-alpha  chloride, inhibitory in flies
#
# Setting these from published electrophysiology is INDEPENDENT DATA, not a
# fit: nothing about the values would be chosen to make motion work. That is
# what distinguishes it from tuning.
#
# VALUES ARE NOT YET FILLED IN. Every entry equals G_SYN, so this is currently
# a no-op and the model behaves exactly as before. Populating it requires real
# citations for Drosophila nAChR / Rdl / GluCl-alpha synaptic conductance; they
# are deliberately NOT guessed here, because a fabricated "published" value
# would make the independent arm worse than having no independent arm.
# RATIOS, not absolutes. Absolute unitary conductances divided by assumed
# synapse counts give only a bracket (0.002-0.014, which does contain our
# fitted G_SYN); the RATIO between receptor types is measured far more
# reliably, and the ratio is what matters here -- the correlator's failure is a
# g_i/g_e problem, not a g_tot one.
G_SYN_RATIO = {
    "ACH": 1.0,
    # [PUBLISHED] GABA-A (Rdl) is ~2.5x the cholinergic conductance. Two
    # unrelated preparations agree: Su & O'Dowd 2003 (quantal events, both
    # transmitters in the SAME cells, ratio 2.5) and Groschner et al. 2022
    # (adult T4 IN VIVO, their own fit, 2.6). The agreement across preparations
    # is what makes this usable; neither was measured with our result in view.
    "GABA": 2.5,
    #   Su & O'Dowd 2003, cultured pupal Kenyon cells, quantal events for BOTH
    #     transmitters in the SAME cells: nAChR 22.1 pA / 83.9 mV = 263 pS,
    #     GABA 24.6 pA / 37.0 mV = 665 pS -> 2.5. Driving forces known, so the
    #     ratio survives the pipette-imposed reversals.
    #   Groschner et al. 2022, ADULT T4 IN VIVO, relative conductances already
    #     in units of g_leak: Mi4 2.20, C3 2.98 (GABA) against Mi1 1.30,
    #     Tm3 0.70 (ACh) -> 2.59.
    #   A cultured pupal mushroom-body preparation and an adult in vivo optic
    #   lobe neuron, entirely unrelated, agreeing at 2.5 and 2.6. That
    #   agreement is the argument; neither was measured with our result in
    #   view. Charge-weighted (multiplying by decay tau 1.4 vs 3.7 ms) the
    #   ratio is 6.7, so 2.5 is the conservative choice.
    # [FITTED BY OTHERS, NOT MEASURED] No Drosophila GluCl-alpha conductance
    # measurement exists -- not single-channel, not quantal, not a decay
    # constant. What surfaces in searches must be actively excluded: the 120 pS
    # figure is a CATION channel (Heckmann & Dudel 1995), and the other
    # candidates are nematode AVR-14B or larval-muscle substates. The value
    # below is Groschner et al. 2022's least-squares fit to adult T4 voltage
    # traces (Mi9 gain 0.92 against Mi1 1.30, both already divided by their
    # g_leak = 0.50, i.e. our units). It is an independent group's fit to real
    # data rather than ours, but it IS a fit, and is labelled so.
    "GLUT": 0.92,
    "DA": 1.0,       # metabotropic; magnitude is not the issue, timescale is
    "OCT": 1.0,
    "SER": 1.0,
    "UNK": 1.0,
}
G_SYN_BY_NT = {k: G_SYN * v for k, v in G_SYN_RATIO.items()}

# CAVEAT ON EVERYTHING KEYED BY nt_type, INCLUDING THE RATIOS ABOVE.
# These are PREDICTED transmitters, not measured ones, and the prediction is
# demonstrably unreliable for the minority classes. Measured on this graph:
#
#   46.2% of ORN_DM1's output synapses are predicted SER -- 3,481 of them onto
#   DM1_lPN, the projection neuron carrying that odour to the lateral horn.
#   ORN_DA1 is 19.7% SER. Olfactory receptor neurons are CHOLINERGIC.
#   The largest DA source and the largest SER source in the entire graph are
#   both lLN1_bc, an antennal-lobe local interneuron; lLNs are GABAergic.
#
# Consequences, both load-bearing:
#   * M8's olfactory result survives only because SIGN["SER"] = +1 happens to
#     agree with the acetylcholine those synapses really use. Silencing the
#     "modulatory" edges does not remove modulation, it cuts the olfactory
#     nerve -- which is why M8 degrades into INTERMITTENCY (2 of 4 seeds show
#     nothing) rather than shrinking smoothly.
#   * Applying per-receptor conductances by predicted class inherits this.
#     ACH/GABA/GLUT are the large, better-predicted classes and are where the
#     ratios above act, but the accuracy is not uniform and this is a stated
#     limitation of arm 2 rather than an assumption.

# DA, OCT and SER are METABOTROPIC. They act through second messengers over
# 100 ms to seconds, not as fast ionotropic conductances, and the model treats
# all 23,197 of their edges as fast excitatory synapses. That is wrong in KIND
# rather than in magnitude, and unlike the conductances above it needs no
# literature value to state -- only a timescale. Left as-is for now and
# recorded here so it is not mistaken for a modelling choice that was checked.
METABOTROPIC = ("DA", "OCT", "SER")

# [PUBLISHED, PARTIAL] Onset ~100 ms, decay >= 1 s -- not the 5 ms tau_syn the
# model gives them. Held et al. 2025 measure a 122 ms median latency for
# octopaminergic modulation by in vivo patch clamp; GRAB-OA sensor kinetics
# (tau_off ~1.4 s in HEK, ~5.9 s in vivo) show the second-scale decay is
# biology rather than sensor lag. NO FITTED TIME CONSTANT EXISTS in the
# literature reachable without institutional access; a dozen primary papers
# were read and none reports one. Gervasi & Preat 2010 (PKA-FRET, mushroom
# body) is paywalled and UNREAD -- it is the most likely place such a value
# still hides, so any "no published tau exists" claim must be softened until
# somebody pulls it.
METABOTROPIC_TAU_ONSET_S = 0.122
METABOTROPIC_TAU_DECAY_S = 1.4
# [FITTED] 2026-08-21 by m2_per.py against the Buhmann v783 graph, same
# protocol as W_SYN: sugar GRNs at 100 Hz Poisson -> 79.8% of the saturating
# MN9 response. M2 passes in conductance mode with all eight checks, including
# 99% bitter suppression, so the model change does not cost the positive
# control.
# The conductance-mode analogue of W_SYN, and like it a free parameter fitted
# on M2 rather than assumed. The a-priori estimate matching the subtractive
# model's gain at rest was 0.0032; the fit landed at 0.00278, 13% below.

# ==========================================================================
# Graded (non-spiking) units  --  THE LAST STRUCTURAL CHANGE
# ==========================================================================
#
# Photoreceptors and most lamina/medulla neurons in the fly DO NOT SPIKE. They
# release transmitter continuously in proportion to membrane potential. Shiu
# et al. model every neuron as spiking, which is fine for the SEZ but inverts
# the balance in the optic lobe.
#
# MEASURED, and this is the reason for the change. Weighting T4a's inputs by
# actual firing rate rather than synapse count:
#
#     Mi1  +30.6 syn x  3.5 Hz = +109      <- fast excitatory arm
#     CT1  -10.0 syn x 40.0 Hz = -402      <- slow inhibitory arms
#     Mi9   -7.4 syn x 16.3 Hz = -121
#                    excitation 189 vs inhibition 558
#     measured conductance: g_exc 0.0018 vs g_inh 0.0667  =  37x
#
# The fast arm is starved 37:1. Mi1 sits at 3.5 Hz because L1 (-77.3,
# glutamatergic) suppresses it -- the ON pathway is DISINHIBITORY, and a
# spiking L1 with a high tonic baseline simply holds Mi1 shut. A correlator
# needs its two arms comparable in drive.
#
# A graded L1 instead releases transmitter in proportion to its membrane
# potential, so light MODULATES its release continuously rather than gating a
# spike train. That is what disinhibition actually needs.

GRADED_SUPER_CLASSES = ("optic",)
# Neurons that signal continuously rather than by spikes. `optic` covers the
# lamina and medulla, 77,873 neurons. Lobula projection cells
# (`visual_projection`) and everything central keep spiking, which is right --
# LC/LPLC and descending neurons do fire action potentials.

GRADED_MAX_RATE = 200.0   # Hz  [OURS]
# Graded output is expressed in spike-equivalents so both populations use the
# same synaptic machinery: a graded neuron driven to threshold delivers the
# same conductance per second as a spiking neuron firing at this rate. The
# activation is rectified-linear in (v - V_rest)/(V_thresh - V_rest), saturating
# at 1. That saturation is itself a useful nonlinearity, and in conductance mode
# v is bounded by E_EXC anyway.

# ==========================================================================
# Per-cell-type conduction delays  --  A DELIBERATE DEPARTURE FROM THE PAPER
# ==========================================================================
#
# Shiu et al. give every synapse one global delay (T_dly = 1.8 ms) and every
# neuron one membrane time constant. That is fine for the SEZ taste circuit
# they validated on, which is feedforward-excitatory. It cannot express the
# optic lobe.
#
# MEASURED on this graph, T4a's inputs per cell:
#
#     Mi1   +30.6      Tm3  +7.0        <- fast, excitatory, centre
#     CT1   -10.0      Mi9  -7.4        <- slow, inhibitory, flanks
#     Mi4    -3.4
#
# That is a textbook three-arm motion detector, and the SPATIAL offsets are
# already in the wiring -- T4 takes Mi1 from its own column and Mi9/Mi4/CT1
# from neighbours. What is missing is the TIME. A correlator needs one arm
# delayed relative to the other; with a single global delay both arms arrive
# together and the direction preference cancels exactly, which is the DSI =
# 0.000 that M3 measured.
#
# So we give the known slow lines a longer conduction delay. This is a change
# to the MODEL, not a tuning knob, and it must be reported as a deviation.
# Set T_DLY_SLOW = T_DLY to switch it off and recover the paper's behaviour.

# ==========================================================================
# Optic-lobe gain  --  A REPLACEMENT FOR THE TONIC BIAS, NOT AN ADDITION
# ==========================================================================
#
# RETRACTED FIGURE, KEPT VISIBLE ON PURPOSE. This block used to open with a
# measurement: a T4a cell driven by its own real inputs in ISOLATED REPLAY
# reaching mean |DSI| 0.37, against 0.002 for the same cells in the network.
# That number is an artefact and must not be cited. Applying the identical
# procedure to a SINGLE input -- a geometry in which direction selectivity is
# physically impossible -- returns 0.12 rather than 0, because Mi1 is itself
# weakly selective (|DSI| 0.040) and a threshold unit amplifies the residue.
# Isolated replay is not a clean control for this question, and every claim
# that rested on the 0.37 is withdrawn.
#
# What survives is the bias sweep below, which is a within-network comparison
# and does not use replay:
#
#     route to ~20-50 Hz                 mean |DSI|
#     input gain, no bias                    0.22
#     real weights + bias 2 mV               0.030
#     real weights + bias 4 mV               0.0055
#     real weights + bias 7 mV               0.0031
#
# A 70x loss, monotonic in bias. The cause is arithmetic: a GAIN multiplies,
# so k*(exc - inh) scales the stimulus-driven signal itself, while a BIAS adds
# a constant, leaving the signal the same size on a raised pedestal. Tonic
# drive gets cells firing without carrying any information about the stimulus,
# which is also why LPLC2 reads 76 Hz to a BLANK screen under bias.
#
# So this is a one-for-one swap, not a new degree of freedom. The model already
# had a free scalar for exactly this population (BIAS_MV over
# BIASED_SUPER_CLASSES); this replaces it with a scalar over the same
# population. The count of fitted parameters is unchanged.
#
# It is also the better-motivated of the two. Nothing injects constant current
# into every visual neuron, whereas synaptic strength genuinely differs between
# brain regions -- and W_SYN was fitted on the SEZ taste circuit, so applying
# it unchanged to the optic lobe was always an assumption rather than a
# measurement. Note what this is NOT: a per-cell-type gain. That is the
# quantity this project exists to measure the absence of, and fitting it would
# make this a worse copy of Lappalainen et al. rather than a complement.
#
# M2 (taste) is untouched by construction: the scaling applies only to
# synapses onto the visual populations, so the SEZ circuit W_SYN was fitted on
# runs at exactly the value it was fitted at. That is a hard check, not a hope.

OPTIC_GAIN = 1.0
"""Multiplier on synapses onto the optic lobe. 1.0 = off (reported model)."""

BIASED_SUPER_CLASSES = ("optic", "visual_projection", "visual_centrifugal")
# Which neurons receive the tonic depolarising drive that makes an
# inhibition-dominated circuit conduct at all.
#
# MEASURED, and a trap worth recording: the lobula columnar cells -- LC4,
# LPLC2, LC11, LC6, every output of the visual system -- are classified
# `visual_projection` (7,684 neurons), NOT `optic` (77,873). Biasing only
# `optic` leaves every LC below threshold, so the whole cascade fires happily
# through the medulla and lobula and then reads out exactly 0.00 Hz. That looked
# like "looming detection does not work" when it was actually "the readout layer
# was never switched on".

# ==========================================================================
# Short-term synaptic depression  --  A CANDIDATE FIX FOR THE ARM IMBALANCE
# ==========================================================================
#
# Every failure measured in the visual pathway has one shape: something firing
# tonically at a high rate crushing something that only speaks in brief bursts.
#
#     Mi1  (fast excitatory arm of the T4a correlator)     2-9 Hz
#     Mi9  (slow inhibitory arm)                          78-130 Hz
#     L1   (suppresses Mi1, glutamatergic, -77.3)            103 Hz
#     PVLP011 (holds LPLC2 below rest)                       326 Hz
#
# Real synapses deplete when used hard and recover over hundreds of ms to
# seconds. A synapse driven at 100+ Hz runs down several-fold; one driven at
# 3 Hz stays fully recharged. Depression therefore attenuates exactly the
# tonic, high-rate inhibition that is doing the crushing, while leaving the
# sparse excitatory arm intact -- it is a high-pass filter on synaptic drive,
# and motion detection is inherently about transients.
#
# This is a MODEL-CLASS change of the same kind as conductance-based synapses:
# real biology the reference model omits, applied UNIFORMLY. It is not a fitted
# per-type gain. There are two global constants and they are swept, not tuned.
#
# Tsodyks-Markram, depression only, presynaptic (one resource per neuron):
#
#     released = out * R
#     R <- R + (1 - R) * dt/tau_rec  -  U * R * out
#
# NOTE the confound this creates and the control it needs: depression lowers
# steady-state release overall, which on its own resembles turning W_SYN down.
# To attribute any effect to REDISTRIBUTION rather than to global gain, compare
# against an arm with W_SYN reduced by the same average factor and no
# depression. m3's --stp-control does this.

STP = False
"""Off by default. The reported model is the frozen one."""

STP_U = 0.3
"""Fraction of available resource consumed per unit of release. [OURS]
Depressing central synapses are typically reported in the 0.1-0.5 range;
0.3 sits mid-range and is swept rather than trusted."""

STP_TAU_REC = 0.5
"""Seconds. Recovery of the resource. [PUBLISHED range] Paired-pulse recovery
at central synapses runs from hundreds of ms to several seconds; fly
photoreceptor adaptation has a fast phase near 100 ms. 0.5 s is mid-range."""

FAST_LINES = ("Mi1", "Tm3", "Tm1", "Tm2", "Tm4")
# [PUBLISHED] The fast, transient medulla lines feeding T4 (ON) and T5 (OFF).

SLOW_LINES = ("Mi9", "Mi4", "CT1", "Tm9")
# [PUBLISHED] The slow, sustained lines. Mi9 is glutamatergic (inhibitory in
# fly), Mi4 and CT1 GABAergic; Tm9 is the slow OFF line into T5. Together they
# form the delayed arms of the detector. 370,109 edges, 13.7% of the graph.

TAU_MEM_FAST = 10e-3   # s  [OURS] transient lines: Mi1, Tm3, Tm1, Tm2, Tm4
TAU_MEM_SLOW = 100e-3  # s  [OURS] sustained lines: Mi9, Mi4, CT1, Tm9
# Per-cell-type MEMBRANE time constants, distinct from conduction delay above.
# A delay shifts a waveform; a time constant also SMOOTHS it, and the smoothing
# is what makes Mi9/Mi4 sustained where Mi1/Tm3 are transient. This is the
# mechanism the real T4 correlator is built from, so it is the more faithful
# implementation of "per-type time constants".
#
# CAVEAT worth knowing before reading any result: in a LIF, firing rate falls
# roughly as 1/tau_mem. Giving Mi9 a 5x longer tau also makes it fire ~5x less,
# so this manipulation changes GAIN as well as timing. That confound is real
# and is why the conduction-delay version was tried first.

T_DLY_SLOW = 80e-3   # s   [OURS -- FITTED, see below]
# The delay on SLOW_LINES. Not a published value: it is set by the geometry of
# the stimulus we want the detector tuned to. Neighbouring ommatidia are ~5 deg
# apart, so at a 30 deg spatial period and 2 Hz the pattern crosses one column
# in (5/30)/2 = 83 ms. A correlator is maximally direction-selective when its
# delay matches that crossing time, which is why ~80 ms is the starting point.
# Sweep it with `m3_optomotor.py --slow-sweep`.

# ==========================================================================
# Doom I/O  --  not used before M5
# ==========================================================================

DOOM_TICRATE = 35              # [PUBLISHED] Doom runs at 35 tics/s
SUBSTEPS_PER_TIC = round((1.0 / DOOM_TICRATE) / DT)   # 57
