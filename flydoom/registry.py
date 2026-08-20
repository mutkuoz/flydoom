"""Declarative registry of every named cell population flydoom depends on.

This file is deliberately *data*, not logic. Each entry records what we are
looking for, what we expect to find, and where the claim comes from. The
resolver in `cells.py` does the searching; it never guesses.

On naming: FlyWire's `classification.csv` types olfactory receptor neurons by
their target **glomerulus**, not by the receptor gene they express. So Or42b
neurons appear as DM1, Or56a as DA2, and so on. Every olfactory handle below
therefore carries both names, and the resolver tries both.

Provenance tags in `note` are the reason a handle exists at all. If a handle
fails to resolve, the note tells you whether to look harder or to redesign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Valence = Literal["attractive", "aversive", "social", "neutral", "n/a"]


@dataclass(frozen=True)
class Handle:
    """One named cell population we need to pull out of the connectome."""

    name: str
    group: str
    patterns: tuple[str, ...]
    """Candidate exact / prefix matches, tried in order against type columns."""

    aliases: tuple[str, ...] = ()
    """Case-insensitive substrings to fall back on, and to surface candidates."""

    expect_min: int | None = 1
    expect_max: int | None = None
    """Expected population size across BOTH hemispheres. None = unbounded."""

    required: bool = True
    """If True, an unresolved handle is a hard failure for the run."""

    bilateral: bool = True
    """Whether we expect this to resolve on both sides."""

    exact_only: bool = False
    """Disable prefix/substring fallback. Set this when the handle's own name is
    a proper prefix of a sibling's (ORN_V vs ORN_VA2), where a fallback would
    return a confidently wrong superset."""

    valence: Valence = "n/a"
    """For sensory channels only: innate behavioural sign in a real fly."""

    note: str = ""


# --------------------------------------------------------------------------
# 1. Visual input — where Doom frames get injected
# --------------------------------------------------------------------------

VISUAL_INPUT = [
    Handle(
        "R1-6", "visual_input",
        patterns=("R1-6", "R1", "R2", "R3", "R4", "R5", "R6"),
        aliases=("photoreceptor", "retinula"),
        expect_min=6000, expect_max=11000,
        valence="neutral",
        note="Primary luminance channel. VERIFIED PRESENT in v783: 8,456 "
             "neurons (L=4,425 R=4,031), 95% carry output edges, 1.01M output "
             "synapses, targets are textbook (L2, L1, L3, Lai, T1). The fork is "
             "resolved -- inject here, per spec 6.1. "
             "*** BUT: photoreceptors are HISTAMINERGIC and histamine is not "
             "one of FlyWire's six predicted transmitters, so 83% of these "
             "edges are mispredicted as ACH = excitatory when the true sign is "
             "INHIBITORY. graph.py must override. See sources.HISTAMINERGIC_SIGN.",
    ),
    Handle(
        "R7", "visual_input",
        patterns=("R7", "R7p", "R7y"),
        aliases=("photoreceptor",),
        expect_min=900, expect_max=2000, required=False,
        valence="neutral",
        note="Chromatic. 1,338 in v783. Doom's palette is brown/grey so this "
             "carries little; expected near-dead weight in v1. Also "
             "histaminergic -- same override as R1-6.",
    ),
    Handle(
        "R8", "visual_input",
        patterns=("R8", "R8p", "R8y"),
        aliases=("photoreceptor",),
        expect_min=900, expect_max=2000, required=False,
        valence="neutral",
        note="Chromatic. 1,357 in v783. Same caveats and same override as R7.",
    ),
    Handle(
        "L1", "visual_input",
        patterns=("L1",), aliases=("lamina monopolar", "LMC"),
        expect_min=1200, expect_max=2400, required=False,  # 1,775 in v783
        note="ON-pathway input. FALLBACK injection site if R1-6 is absent — and "
             "arguably better anyway, since L1/L2 split contrast into ON/OFF and "
             "sidestep the histamine problem (photoreceptor output is "
             "histaminergic, and histamine is NOT one of FlyWire's six predicted "
             "transmitters, so every R->L edge would be mis-signed).",
    ),
    Handle(
        "L2", "visual_input",
        patterns=("L2",), aliases=("lamina monopolar", "LMC"),
        expect_min=1200, expect_max=2400, required=False,  # 1,728 in v783
        note="OFF-pathway input. Fallback injection site, see L1.",
    ),
    Handle(
        "L3", "visual_input",
        patterns=("L3",), aliases=("lamina monopolar", "LMC"),
        expect_min=1000, expect_max=2400, required=False,  # 1,477 in v783
        note="Sustained luminance. Secondary fallback injection site.",
    ),
]

# --------------------------------------------------------------------------
# 2. Visual feature detectors — probes, not inputs
# --------------------------------------------------------------------------

VISUAL_DETECTORS = [
    Handle(
        "LC4", "visual_detector",
        patterns=("LC4",), aliases=("lobula columnar",),
        expect_min=40, expect_max=250,
        note="Looming / angular velocity. Feeds the giant fiber. One half of "
             "the escape arm of the engage/flee switch (spec 7).",
    ),
    Handle(
        "LPLC2", "visual_detector",
        patterns=("LPLC2",), aliases=("lobula plate",),
        expect_min=40, expect_max=250,
        note="Looming, angular-size selective. The other escape input to DNp01. "
             "Drives M4.",
    ),
    Handle(
        "LC11", "visual_detector",
        patterns=("LC11",), aliases=("lobula columnar",),
        expect_min=40, expect_max=250,
        note="Small moving object. The approach arm of the engage/flee switch.",
    ),
    Handle(
        "LC10a", "visual_detector",
        patterns=("LC10a", "LC10"), aliases=("lobula columnar",),
        expect_min=40, expect_max=400,
        note="Small object azimuth — supplies fixation error for the aim signal "
             "in --attack-mode fixation. LC10 is split a/b/c/d in the v783 optic "
             "lobe typing; we want the 'a' subtype specifically.",
    ),
    Handle(
        "T4", "visual_detector",
        patterns=("T4", "T4a", "T4b", "T4c", "T4d"),
        expect_min=1000, expect_max=8000, required=False,
        note="ON-motion elementary detector. Not an input or output — used to "
             "sanity-check M3: if the optomotor response is wrong, look here "
             "before blaming the descending neurons.",
    ),
    Handle(
        "T5", "visual_detector",
        patterns=("T5", "T5a", "T5b", "T5c", "T5d"),
        expect_min=1000, expect_max=8000, required=False,
        note="OFF-motion elementary detector. M3 diagnostic, see T4.",
    ),
    Handle(
        "T4a", "visual_detector",
        patterns=("T4a",),
        expect_min=800, expect_max=2500, required=False,
        note="ON-motion detector, direction subtype "
             "'a'. T4/T5 a-d are each tuned to a DIFFERENT direction, so "
             "pooling them cancels direction selectivity by construction -- M3 "
             "must monitor them separately or it will measure zero when there "
             "is a signal.",
    ),
    Handle(
        "T4b", "visual_detector",
        patterns=("T4b",),
        expect_min=800, expect_max=2500, required=False,
        note="ON-motion detector, direction subtype "
             "'b'. T4/T5 a-d are each tuned to a DIFFERENT direction, so "
             "pooling them cancels direction selectivity by construction -- M3 "
             "must monitor them separately or it will measure zero when there "
             "is a signal.",
    ),
    Handle(
        "T4c", "visual_detector",
        patterns=("T4c",),
        expect_min=800, expect_max=2500, required=False,
        note="ON-motion detector, direction subtype "
             "'c'. T4/T5 a-d are each tuned to a DIFFERENT direction, so "
             "pooling them cancels direction selectivity by construction -- M3 "
             "must monitor them separately or it will measure zero when there "
             "is a signal.",
    ),
    Handle(
        "T4d", "visual_detector",
        patterns=("T4d",),
        expect_min=800, expect_max=2500, required=False,
        note="ON-motion detector, direction subtype "
             "'d'. T4/T5 a-d are each tuned to a DIFFERENT direction, so "
             "pooling them cancels direction selectivity by construction -- M3 "
             "must monitor them separately or it will measure zero when there "
             "is a signal.",
    ),
    Handle(
        "T5a", "visual_detector",
        patterns=("T5a",),
        expect_min=800, expect_max=2500, required=False,
        note="OFF-motion detector, direction subtype "
             "'a'. T4/T5 a-d are each tuned to a DIFFERENT direction, so "
             "pooling them cancels direction selectivity by construction -- M3 "
             "must monitor them separately or it will measure zero when there "
             "is a signal.",
    ),
    Handle(
        "T5b", "visual_detector",
        patterns=("T5b",),
        expect_min=800, expect_max=2500, required=False,
        note="OFF-motion detector, direction subtype "
             "'b'. T4/T5 a-d are each tuned to a DIFFERENT direction, so "
             "pooling them cancels direction selectivity by construction -- M3 "
             "must monitor them separately or it will measure zero when there "
             "is a signal.",
    ),
    Handle(
        "T5c", "visual_detector",
        patterns=("T5c",),
        expect_min=800, expect_max=2500, required=False,
        note="OFF-motion detector, direction subtype "
             "'c'. T4/T5 a-d are each tuned to a DIFFERENT direction, so "
             "pooling them cancels direction selectivity by construction -- M3 "
             "must monitor them separately or it will measure zero when there "
             "is a signal.",
    ),
    Handle(
        "T5d", "visual_detector",
        patterns=("T5d",),
        expect_min=800, expect_max=2500, required=False,
        note="OFF-motion detector, direction subtype "
             "'d'. T4/T5 a-d are each tuned to a DIFFERENT direction, so "
             "pooling them cancels direction selectivity by construction -- M3 "
             "must monitor them separately or it will measure zero when there "
             "is a signal.",
    ),
]

# --------------------------------------------------------------------------
# 3. Taste — the health/damage channel (spec 6.3)
# --------------------------------------------------------------------------

GUSTATORY = [
    Handle(
        "sugar_GRN", "gustatory",
        patterns=("sugar/water", "Gr64f", "Gr5a", "GRN_sugar"),
        aliases=("sugar", "sweet", "gr64", "gr5a"),
        expect_min=40, expect_max=300,  # 129 in v783, as sub_class "sugar/water"
        valence="attractive",
        note="Sweet-sensing gustatory receptor neurons. Health/armor pickup "
             "injects here. Also the stimulus for M2, the critical validation. "
             "Cell bodies are in the labellum (outside the brain) but axons "
             "enter the SEZ and are in FAFB. Shiu et al. 2024 enumerate the "
             "root IDs — pull from that supplement if the type string misses.",
    ),
    Handle(
        "bitter_GRN", "gustatory",
        patterns=("bitter", "Gr66a", "Gr33a", "GRN_bitter"),
        aliases=("gr66", "gr33"),
        expect_min=20, expect_max=300,  # 65 in v783, as sub_class "bitter"
        valence="aversive",
        note="Bitter-sensing GRNs. Taking damage injects here. Also the NEGATIVE "
             "control for M2: bitter stimulation must NOT drive MN9, and "
             "sugar+bitter must suppress relative to sugar alone.",
    ),
]

# --------------------------------------------------------------------------
# 4. Olfactory — the Doom-entity -> odour channel
#
# FlyWire names ORNs by target glomerulus, so each handle carries the receptor
# gene AND the glomerulus. The `valence` field records documented innate
# behaviour in a real fly; it is what makes the no-learning arm interesting.
#
# NOTE ON USE: assigning valence-loaded odours to Doom entities (imp -> geosmin)
# hand-codes the threat model and is therefore CIRCULAR as a learning result.
# For the plastic-MB arm, use valence-NEUTRAL patterns and let damage teach the
# association. The loaded ones are for the innate/lateral-horn arm only.
# --------------------------------------------------------------------------

OLFACTORY = [
    Handle(
        "ORN_DM1", "olfactory",
        patterns=("ORN_DM1",), aliases=("or42b", "dm1"),
        expect_min=20, expect_max=400, required=False,
        valence="attractive",
        note="Or42b. Vinegar / ethyl acetate. The strongest single attraction "
             "driver in the fly. Natural mapping: health pickup.",
    ),
    Handle(
        "ORN_DM2", "olfactory",
        patterns=("ORN_DM2",), aliases=("or22a", "dm2"),
        expect_min=20, expect_max=400, required=False,
        valence="attractive",
        note="Or22a. Fruit esters. Secondary attractive channel.",
    ),
    Handle(
        "ORN_VA2", "olfactory",
        patterns=("ORN_VA2",), aliases=("or92a", "va2"),
        expect_min=10, expect_max=300, required=False,
        valence="attractive",
        note="Or92a. Attractive.",
    ),
    Handle(
        "ORN_DA2", "olfactory",
        patterns=("ORN_DA2",), aliases=("or56a", "da2", "geosmin"),
        expect_min=10, expect_max=300, required=False,
        valence="aversive",
        note="Or56a / geosmin. A DEDICATED hardwired aversion channel — a single "
             "receptor labelled line for toxic microbial growth that bypasses "
             "learning entirely. The cleanest innate-avoidance probe available.",
    ),
    Handle(
        "ORN_DM5", "olfactory",
        patterns=("ORN_DM5",), aliases=("or85a", "dm5"),
        expect_min=10, expect_max=300, required=False,
        valence="aversive",
        note="Or85a. Aversive.",
    ),
    Handle(
        "ORN_V", "olfactory",
        patterns=("ORN_V",), aliases=("gr21a", "gr63a", "co2"),
        expect_min=10, expect_max=300, required=False,
        valence="aversive", exact_only=True,
        note="Gr21a/Gr63a. CO2 — the fly stress odour. Hardwired avoidance. "
             "EXACT MATCH ONLY: the glomerulus is literally named 'V', and "
             "'ORN_V' is a proper prefix of ORN_VA2 / ORN_VA1v — any fallback "
             "here returns a confidently wrong superset.",
    ),
    Handle(
        "ORN_DA1", "olfactory",
        patterns=("ORN_DA1",), aliases=("or67d", "cva", "da1"),
        expect_min=10, expect_max=300, required=False,
        valence="social",
        note="Or67d / cVA pheromone. Social + aggression-relevant; feeds the "
             "aIPg / pC1 aggression state. Relevant to --attack-mode fixation, "
             "which otherwise has nothing driving its gate.",
    ),
    Handle(
        "ORN_VA1v", "olfactory",
        patterns=("ORN_VA1v",), aliases=("or47b", "va1v"),
        expect_min=10, expect_max=300, required=False,
        valence="social",
        note="Or47b. Pheromonal, aggression-potentiating.",
    ),
    Handle(
        "ALPN", "olfactory",
        patterns=("ALPN",),
        aliases=("_lPN", "_adPN", "_vPN", "uPN", "mPN"),
        expect_min=100, expect_max=800, required=False,
        note="Antennal lobe projection neurons — the readout from glomeruli to "
             "BOTH the mushroom body calyx (learned valence) and the lateral "
             "horn (innate valence). Everything olfactory passes through here.",
    ),
]

# --------------------------------------------------------------------------
# 5. Mushroom body — the learning centre (frozen in v1, plastic in the v1.5 arm)
# --------------------------------------------------------------------------

MUSHROOM_BODY = [
    Handle(
        "KC", "mushroom_body",
        patterns=("KC",),
        aliases=("kenyon", "KCg", "KCab", "KCapbp"),
        expect_min=3000, expect_max=6000, required=False,
        note="Kenyon cells, all subtypes, both hemispheres (~2,000/side). The "
             "sparse random expansion layer. PRESYNAPTIC side of the only "
             "synapse class the fly actually makes plastic.",
    ),
    Handle(
        "KCg-d", "mushroom_body",
        patterns=("KCg-d",), aliases=("kcg-d", "gamma-d", "ventral accessory"),
        expect_min=100, expect_max=800, required=False,
        note="The VISUAL Kenyon cells — receive medulla input via the ventral "
             "accessory calyx. This is the only route by which what the fly SEES "
             "can reach the learning centre, so it is the purest (and riskiest) "
             "version of the plastic arm: no invented odours needed.",
    ),
    Handle(
        "MBON", "mushroom_body",
        patterns=("MBON",), aliases=("mushroom body output",),
        expect_min=50, expect_max=180, required=False,
        note="~34 per hemisphere. POSTSYNAPTIC side of the plastic synapse. "
             "Their collective vote is the mushroom body's approach/avoid "
             "output. RISK: they reach descending neurons only indirectly, so "
             "their influence may be swamped in a frozen surround.",
    ),
    Handle(
        "PAM", "mushroom_body",
        patterns=("PAM",), aliases=("pam0", "pam1", "dopamin"),
        expect_min=100, expect_max=400, required=False,
        valence="attractive",
        note="Reward dopaminergic neurons. In the plastic arm the health-pickup "
             "signal routes HERE rather than stopping at the sugar GRNs.",
    ),
    Handle(
        "PPL1", "mushroom_body",
        patterns=("PPL1",), aliases=("ppl1", "dopamin"),
        expect_min=10, expect_max=100, required=False,
        valence="aversive",
        note="Punishment dopaminergic neurons. The damage signal routes here. "
             "Together with PAM these carry the teaching signal that v1 "
             "currently injects into a brain with no plasticity to receive it.",
    ),
    Handle(
        "LHN", "mushroom_body",
        patterns=("LH",), aliases=("lateral horn", "LHAV", "LHPV", "LHCENT"),
        expect_min=500, expect_max=4000, required=False,
        note="Lateral horn neurons — the INNATE valence pathway, parallel to the "
             "mushroom body and requiring no plasticity. This is what makes an "
             "olfactory arm possible inside v1's no-learning constraint.",
    ),
]

# --------------------------------------------------------------------------
# 6. Descending neurons — the motor readout (spec 6.2)
# --------------------------------------------------------------------------

DESCENDING = [
    Handle(
        "DNa02", "descending",
        patterns=("DNa02",), aliases=("dna02", "dna2"),
        expect_min=2, expect_max=4,
        note="Steering. THE yaw signal: turn = GAIN * (rate(L) - rate(R)). One "
             "pair. Rayshubskiy et al. A wrong DNa02 produces a plausible-looking "
             "agent that is measuring nothing — do not guess this one.",
    ),
    Handle(
        "DNa01", "descending",
        patterns=("DNa01",), aliases=("dna01", "dna1"),
        expect_min=2, expect_max=4, required=False,
        note="Secondary steering, documented partner of DNa02. RECOMMENDED "
             "REPLACEMENT for the spec's DNg13, whose steering role I could not "
             "substantiate.",
    ),
    Handle(
        "DNg13", "descending",
        patterns=("DNg13",), aliases=("dng13",),
        expect_min=2, required=False,
        note="RESOLVES CLEANLY in v783 (1 pair). But the spec's claim that it "
             "does steering is still UNVERIFIED — I could not confirm that role "
             "in the literature. Existing is not the same as doing what the "
             "spec says. Prefer DNa01 for secondary steering.",
    ),
    Handle(
        "BPN", "descending",
        patterns=("BPN",), aliases=("Bolt Protocerebral", "bolt"),
        expect_min=2, expect_max=120,
        note="Fast straight forward walking (Bidaye et al. 2020). NOT typed in "
             "consolidated_cell_types -- exists ONLY as free-text community "
             "labels ('Type 1 BPN (Bolt Protocerebral Neuron)', types 1-4). "
             "Will resolve as WEAK via labels.csv; verify the population by "
             "hand before trusting MOVE_FORWARD. Also note Bolt neurons are "
             "brain interneurons, NOT descending neurons — fine to read out, "
             "but the spec mis-files them.",
    ),
    Handle(
        "MDN", "descending",
        patterns=("MDN",), aliases=("moonwalker", "mdn"),
        expect_min=2, expect_max=8,
        note="Moonwalker DN — backward walking. Drives MOVE_BACKWARD.",
    ),
    Handle(
        "DNp09", "descending",
        patterns=("DNp09",), aliases=("dnp09",),
        expect_min=2, expect_max=4, required=False,
        note="Forward walking with ipsilateral turn; also reported as freezing "
             "under optogenetic activation. Context-dependent — treat any "
             "readout from it with suspicion.",
    ),
    Handle(
        "DNp01", "descending",
        patterns=("DNp01",), aliases=("giant fiber", "giant fibre", "GF"),
        expect_min=2, expect_max=4,
        note="The giant fiber. Escape. One pair. Drives MOVE_LEFT/RIGHT dodge "
             "and is the M4 readout. Its ELECTRICAL output is invisible to us "
             "(EM sees chemical synapses only) but its cholinergic INPUT from "
             "LC4/LPLC2 is intact, which is all we need.",
    ),
]

HALTING = [
    Handle(
        "BRK", "halting",
        patterns=("BRK", "Brake"), aliases=("brake",),
        expect_min=2, required=False,
        note="PRESENT as free-text labels only: BRK1-BRK6, 6 neurons. "
             "*** BUT READ THE LABEL: 'Ascending projection of prothoracic / "
             "mesothoracic / metathoracic Brake Neuron'. The brake neurons "
             "themselves live in the VNC, which FAFB does not contain. What is "
             "here is their ASCENDING AXON reporting INTO the brain. That makes "
             "BRK an INPUT (proprioceptive 'I am braking'), not a halt command "
             "we can read out. Do NOT wire it to the stop button. ***",
    ),
    Handle(
        "FoG", "halting",
        patterns=("FoG", "Foxglove"), aliases=("foxglove",),
        expect_min=2, required=False,
        note="Foxglove, 2 neurons, free-text labels only. Sterne et al. 2021 "
             "SEZ collection — NOT Sapkal et al. as the spec implies. Genuinely "
             "in the brain (SEZ), so unlike BRK this is usable as a halt "
             "readout. Verify by hand: a 2-neuron population found by substring "
             "is one typo away from being something else.",
    ),
    Handle(
        "BB", "halting",
        patterns=("BB", "Bluebell"), aliases=("bluebell",),
        expect_min=2, required=False,
        note="Bluebell, 2 neurons. Properly typed as additional_types='bluebell' "
             "(not just free text), so this is the most trustworthy of the "
             "three. Labelled 'SEZ stop neuron', Sterne et al. 2021. This is "
             "the halt readout to prefer.",
    ),
]

# --------------------------------------------------------------------------
# 7. Motor output — the M2 readout
# --------------------------------------------------------------------------

MOTOR = [
    Handle(
        "MN9", "motor",
        patterns=("MN9", "proboscis_motor_neuron"), aliases=("mn9", "proboscis"),
        expect_min=2, expect_max=40,
        note="v783 does NOT type MN9 individually -- it exposes a "
             "`proboscis_motor_neuron` sub_class of 24 cells (L=12 R=12), the "
             "whole proboscis motor pool. Using the pool as the M2 readout is "
             "acceptable (MN9 is in it), but the pass criterion should be "
             "'pool activates', not 'MN9 activates'. Narrowing to MN9 proper "
             "needs the Shiu et al. supplement. THE M2 READOUT: sugar GRN stimulation must "
             "drive this, bitter must not. This is the only end-to-end check "
             "that signs, weights, dynamics and the id map are ALL "
             "simultaneously correct. Nothing downstream is diagnosable until "
             "it passes.",
    ),
]

# --------------------------------------------------------------------------
# 8. Aggression state — the ATTACK gate
# --------------------------------------------------------------------------

AGGRESSION = [
    Handle(
        "aIPg", "aggression",
        patterns=("aIPg1", "aIPg3", "aIPg4", "aIPg"), aliases=("aIPg",),
        expect_min=5, expect_max=200, required=False,
        # Not in consolidated_cell_types; present only in labels.csv
        # (aIPg1/aIPg3/aIPg4 appear ~110 times).
        note="Female aggression state-setter. Slow, persistent over tens of "
             "seconds — use as a GATE, never as a per-shot trigger.",
    ),
    Handle(
        "pC1", "aggression",
        patterns=("pC1a", "pC1b", "pC1c", "pC1d", "pC1e", "pC1"), aliases=("pc1",),
        expect_min=4, expect_max=60, required=False,  # 10 in v783 (5/side)
        note="REPLACES the spec's P1. FAFB is a FEMALE brain and P1 neurons are "
             "male-specific, so 'P1' cannot resolve here by construction. pC1d/e "
             "are the female aggression homologues. Without an input driving "
             "these, --attack-mode fixation never opens its gate and never "
             "fires — see ORN_DA1 for a candidate drive.",
    ),
]

# CASE COLLISION, measured in v783: the optic lobe uses distal-medulla types
# Dm1, Dm2, Dm3... while the antennal lobe uses glomeruli DM1, DM2, DM3. Under
# case-insensitive matching these are the same string. Including a bare
# glomerulus name in an ORN handle's patterns pulled 1,275 medulla neurons into
# ORN_DM2 (expected ~50). Always spell ORN handles as ORN_<GLOM>; the bare
# glomerulus name belongs only to the projection-neuron handles (DM1_lPN).
#
# Glomerulus names that are proper prefixes of sibling glomeruli. Any handle
# for one of these MUST set exact_only, or a miss silently returns the siblings.
# Measured from v783: ORN_D, ORN_V, ORN_VA1, ORN_DA4, ORN_DL2, ORN_VC3...
PREFIX_COLLIDING_GLOMERULI = ("D", "V", "VA1", "VC3", "DA4", "DL2", "VM5")


ALL_HANDLES: list[Handle] = [
    *VISUAL_INPUT,
    *VISUAL_DETECTORS,
    *GUSTATORY,
    *OLFACTORY,
    *MUSHROOM_BODY,
    *DESCENDING,
    *HALTING,
    *MOTOR,
    *AGGRESSION,
]

GROUP_ORDER = [
    "visual_input", "visual_detector", "gustatory", "olfactory",
    "mushroom_body", "descending", "halting", "motor", "aggression",
]


def by_name(name: str) -> Handle:
    for h in ALL_HANDLES:
        if h.name == name:
            return h
    raise KeyError(f"no handle named {name!r}")


def required_handles() -> list[Handle]:
    return [h for h in ALL_HANDLES if h.required]
