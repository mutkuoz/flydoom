#!/bin/bash
# ODOUR ALONE — which of the three broke the vision control?
#
# behav_all applied dendritic cables, a neck and whole-level odour together,
# and the result was that mirroring the retina stopped abolishing the
# advantage and a frozen retina beat chance outright. Odour is the obvious
# suspect, because mirroring flips retinal geometry and freezing stops the
# frame, and neither touches smell.
#
# So: the same 60 seeds and the same three arms with ONLY --smell-all added to
# the plain configuration. If mirrored still beats chance here, whole-level
# odour is sufficient to break the control on its own and the cables and the
# neck are not implicated.
#
# One caveat carried over: with the neck OFF, as here, the frozen arm is a
# clean no-vision control again. In behav_all it was not -- the eyes still
# turned across the held frame.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_smell
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --smell-all --bias 0
        --device cuda --jobs 3 --spiking-t4 --optic-gain 16 --yaw-source DNp15
        --touch --wide --render fast --eye-map anatomical --fixed-turn
        --phasic-mdn
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_fly"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json
