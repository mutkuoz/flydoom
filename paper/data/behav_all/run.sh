#!/bin/bash
# EVERYTHING APPLIED AT ONCE.
#
# The 120-seed fly-arena result (behav_fixed + behav_more) is the best measured
# run, and it is NOT the best available configuration. Three things developed
# since were never in it, each for its own reason:
#
#   --smell-all   whole-level odour. The olfaction in every earlier run was
#                 gated on vision, which hid 82% of the items in the level, and
#                 medkits carried a food_strength of 1 against the 5 they carry
#                 now. Odour does not need line of sight; this is the fix.
#   --dendrites 3 each T4/T5 as a three-compartment cable with every input
#                 placed by its own retinotopic offset (M17). It takes T5
#                 mirror-pair separation from 0.005 to 0.064 open loop, and has
#                 never once been run in closed loop.
#   --head 20     the eyes turn on the neck, up to 20 degrees either side,
#                 driven by the same yaw command, so gaze can move without the
#                 body. Also never run in closed loop.
#
# Same arena, same seeds (40-99), same controls and the same everything else as
# behav_fixed, so the comparison against it isolates exactly these three.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_all
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --smell-all --bias 0
        --device cuda --jobs 3 --spiking-t4 --optic-gain 16 --yaw-source DNp15
        --touch --wide --render fast --eye-map anatomical --fixed-turn
        --phasic-mdn --dendrites 3 --g-axial 8 --head 20
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_fly"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json
