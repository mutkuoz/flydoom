#!/bin/bash
# THE ARENA THAT MEANS TO A FLY WHAT IT LOOKS LIKE.
#
# The corrected tables found the model's vision steers it into walls, and
# excused it as a fly's attraction to dark verticals transferring without the
# wisdom to go with it. That was charitable to the arena: health_gathering_fly
# tiles every wall with a perfect tall dark bar at 0.56 contrast, and dresses
# the food as a pale box. We built the trap.
#
# health_gathering_forage (scripts/build_forage_arena.py) moves the darkness
# off the walls and onto the food. Walls keep their spatial frequency and lose
# their vertical coherence -- phases balanced so contrast down any column is
# flat -- so the motion detectors see what they saw and there is no bar left to
# approach. Food becomes a small dark blob, which is what a fly approaches.
# Poison becomes pale, which it should not.
#
# NOTHING IN THE BRAIN CHANGES, and this is the plain configuration: no
# dendritic cables, no neck, vision-gated odour, exactly behav_fixed's setup on
# exactly its seeds. The only difference from behav_fixed is the arena.
#
# THE PREDICTION. If "it approaches large dark structure" is the right
# description of what its vision does, health should rise, damage should fall
# and time stuck against geometry should fall, with no change to the model. If
# nothing moves, that description was wrong.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_forage
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --bias 0
        --device cuda --jobs 3 --spiking-t4 --optic-gain 16 --yaw-source DNp15
        --touch --wide --render fast --eye-map anatomical --fixed-turn
        --phasic-mdn
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_forage"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json
