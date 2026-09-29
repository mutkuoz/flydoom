#!/bin/bash
# THE FLY. One configuration, everything the animal actually has, all at once.
#
# This project accumulated flags because each one had to be isolated to find
# out what it did. That is how you learn; it is not what you ship. A real fly
# does not come with odour off and wind on. It has two eyes with dendritic
# arbors, a neck, two antennae that feel both contact and airflow, and a nose,
# in air that is always moving, all at the same time.
#
# So: everything the animal has, on.
#
#   --wide --eye-map anatomical      the whole visual field, mapped as the
#                                    wiring says the eyes are mapped
#   --fixed-turn --phasic-mdn        steering toward the active side, MDN read
#                                    as bursts -- the interface corrections
#   --spiking-t4 --optic-gain 16     the detectors spike; drive at the level
#                                    every table uses
#   --dendrites 3 --g-axial 8        T4/T5 as three-compartment cables with
#                                    inputs placed by retinotopic offset. A
#                                    neuron is not a point, and this is the
#                                    only change that raised per-cell motion
#                                    selectivity by an order of magnitude
#   --head 20                        a neck. Flies turn their heads
#   --touch                          antennal contact
#   --smell --smell-all              a nose that works through walls, because
#                                    odour does not need line of sight
#   --wind 1.0                       air moves. The antennae feel it, and a
#                                    source is only smelled from downwind
#
# WHAT IS ALREADY KNOWN ABOUT THIS COMBINATION, so it is not oversold. Without
# wind (paper/data/behav_all) it scores +9.60 against its matched random agent,
# better than the plain configuration's +7.53 -- and a MIRRORED retina scores
# +6.17 and a FROZEN one +6.53, neither distinguishable from intact. The
# advantage stops needing eyes, and the decomposition attributes that to the
# dendritic cables (paper/data/behav_cables). Movement is also jitterier.
#
# That is the tension this run exists to state plainly rather than hide: the
# most faithful fly is not the most interpretable experiment. Both facts belong
# in the paper, and the controls are run here for exactly that reason.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_fly_canonical
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --bias 0 --device cuda --jobs 3
        --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch --wide
        --render fast --eye-map anatomical --fixed-turn --phasic-mdn
        --smell --smell-all --dendrites 3 --g-axial 8 --head 20
        --wind 1.0 --wind-dir 0 --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_fly"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json
