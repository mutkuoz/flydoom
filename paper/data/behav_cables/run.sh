#!/bin/bash
# THE CABLES ALONE -- the last arm of the behav_all decomposition.
#
# behav_all applied dendritic cables, a neck and whole-level odour together and
# the vision control collapsed: mirroring stopped abolishing the advantage and
# a frozen retina beat chance outright. Run alone, odour does nothing anywhere
# (+3.80 / +0.40 / -0.07) and the neck does nothing anywhere
# (+2.67 / +3.67 / +0.53). Each change on its own COSTS the advantage without
# producing the broken control; only together does performance return, by a
# route that survives mirroring and freezing.
#
# That is an interaction rather than a sum, and the cables are the one
# component never run on their own. This is that arm: the plain configuration
# plus --dendrites 3 --g-axial 8 and nothing else, on behav_fixed's own seeds
# and controls.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_cables
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --bias 0
        --device cuda --jobs 2 --spiking-t4 --optic-gain 16 --yaw-source DNp15
        --touch --wide --render fast --eye-map anatomical --fixed-turn
        --phasic-mdn --dendrites 3 --g-axial 8
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_fly"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json
