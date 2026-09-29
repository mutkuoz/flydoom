#!/bin/bash
# WIND IN THE ARENA THAT MEANS WHAT IT LOOKS LIKE.
#
# Two things in this project have done something. Wind steadies the turning --
# a quarter less circling and a fifth straighter walking, the only addition
# that improves behaviour rather than costing it -- and the forage arena
# removed the tall dark bar the model kept walking into, which turned "it
# approaches dark verticals" into the sharper and truer "it approaches large
# structure of either polarity".
#
# They have never been combined, and there is a reason to expect the
# combination to be more than the sum. Wind's benefit is course stabilisation:
# it steadies HOW the animal turns. In the fly arena that steadiness is spent
# walking more reliably into a wall, because the walls are the strongest thing
# its vision pulls toward. In the forage arena there is no such bar, so a
# steadier course has somewhere better to go.
#
# Isolated: identical to behav_forage -- same arena, same seeds, same
# vision-gated odour, same controls -- with wind and nothing else added. The
# difference is the wind.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_windforage
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --bias 0
        --device cuda --jobs 3 --spiking-t4 --optic-gain 16 --yaw-source DNp15
        --touch --wide --render fast --eye-map anatomical --fixed-turn
        --phasic-mdn --wind 1.0 --wind-dir 0
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_forage"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json
