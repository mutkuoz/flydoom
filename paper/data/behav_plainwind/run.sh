#!/bin/bash
# PLAIN + WIND. The candidate for the single best configuration.
#
# What the evidence says so far. The plain corrected configuration beats its
# matched random agent decisively -- 28 of 38 non-tied levels, sign test
# p=0.005 -- while its mirrored control is 21 of 42, an exact coin flip. That
# is the cleanest dissociation in the project.
#
# Everything added since has cost something. Whole-level odour takes +7.53 to
# +3.80 because it is a direction-free scalar and strengthening it adds drive
# without information. Dendritic cables keep the score and remove the model's
# dependence on its eyes. The neck lifts the frozen arm by turning the eyes
# across a held frame. All of them together (behav_fly_canonical) fail to beat
# chance at all, p=0.743.
#
# Wind is the exception and the only one: a quarter less circling and a fifth
# straighter walking, 40 of 60 seeds, two independent measures. But it has
# never been run on the PLAIN configuration -- behav_wind carried whole-level
# odour as well, which hurts on its own, so wind's contribution there was
# measured on top of a handicap.
#
# This is plain, exactly as behav_fixed, with wind and nothing else added.
#
# THE PREDICTION, on the record. It should keep plain's advantage (sign test
# near p=0.005, mirrored near chance) AND show wind's stabilisation (spin about
# -0.04). If both hold this is the single configuration to ship. If the
# advantage drops, wind costs performance even without the odour handicap, and
# plain stays the answer.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_plainwind
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --bias 0 --device cuda
        --jobs 3 --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch
        --wide --render fast --eye-map anatomical --fixed-turn --phasic-mdn
        --wind 1.0 --wind-dir 0 --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_fly"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json
