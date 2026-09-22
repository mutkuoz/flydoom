#!/bin/bash
# MORE SEEDS, for power.
#
# paper/data/behav_fixed reports the corrected behaviour tables on 60 held-out
# seeds (40-99). The fly-arena advantage there is +7.53 health and +34.0 tics
# with 95% CIs that exclude zero, but no model-versus-model contrast is
# individually significant -- fly-arena intact minus mirrored is +3.00 +- 6.18
# on health, a CI four times the size it would need to be. That is a power
# problem and nothing else will fix it.
#
# So: the same three fly-arena conditions on 60 FURTHER seeds, 100-159, giving
# 120 in total. Same brain, same interface, same controls; only the seeds are
# new, and they were never looked at while any of this was designed.
#
# The sky arena is not repeated: the corrected tables already show no effect
# there, and doubling the seeds on a null is the expensive way to learn nothing.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_more
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 100 --smell --bias 0 --device cuda
        --jobs 3 --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch
        --wide --render fast --eye-map anatomical --fixed-turn --phasic-mdn
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_fly"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json
