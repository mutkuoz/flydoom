#!/bin/bash
# THE NECK ALONE — the remaining suspect for the broken control.
#
# behav_all (cables + neck + whole-level odour) lost its vision dependence: a
# mirrored retina stopped abolishing the advantage and a frozen retina beat
# chance outright. behav_smell showed odour alone cannot do that -- it produces
# no advantage to break. That leaves the cables and the neck, and the neck has
# a mechanism: with the head on, freezing the FRAME does not freeze the INPUT,
# because the eyes keep turning across the held picture and the model receives
# self-generated motion over a static scene. That would lift the frozen arm
# specifically, which is the arm that moved.
#
# So: the plain configuration plus --head 20 and nothing else, same 60 seeds,
# same three arms. If the frozen arm beats chance here, the neck is the cause
# and the frozen-frame control is invalid whenever a head can move.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_neck
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --bias 0
        --device cuda --jobs 3 --spiking-t4 --optic-gain 16 --yaw-source DNp15
        --touch --wide --render fast --eye-map anatomical --fixed-turn
        --phasic-mdn --head 20
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_fly"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json
