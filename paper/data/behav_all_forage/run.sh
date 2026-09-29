#!/bin/bash
# EVERYTHING APPLIED, IN AN ARENA THE FLY CAN READ.
#
# behav_all found that applying dendritic cables, a neck and whole-level odour
# together destroys the vision control: mirroring the retina stopped abolishing
# the advantage. That was measured on HEALTH, and behav_forage2 has since shown
# health cannot resolve the intact-minus-mirrored contrast at sixty seeds in
# this arena at all -- the claim was a null read off a blind instrument.
#
# Re-scored on collisions, from the data already on disk, the claim survives and
# sharpens (striped arena, seeds 40-99, intact vs mirrored, ties excluded):
#
#   plain                14.12 -> 20.12   45 up / 15 down   p = 1.4e-04
#   neck only            14.59 -> 23.26   48 up / 11 down   p = 1.2e-06
#   odour only           14.34 -> 20.76   44 up / 14 down   p = 1.0e-04
#   cables only          19.12 -> 19.08   29 up / 31 down   p = 0.90
#   everything applied   19.24 -> 18.03   30 up / 28 down   p = 0.90
#
# Neck and odour are the positive controls: they keep the effect at p <= 1e-04,
# so the two nulls are not the metric going blind. And the mechanism is not what
# behav_all described. The cables do not lift the mirrored arm to meet the
# intact one; they drag the INTACT arm down to the mirrored one, 14.12 -> 19.12.
# The model does not stop needing its eyes. It stops using eyes that still work.
#
# THIS RUN asks whether that null is a property of the configuration or of the
# striped arena, which behav_forage2 showed compresses exactly this contrast.
# Everything-applied on the FIXED forage arena, same seeds and controls as
# behav_forage2, so the only difference from behav_all is the room.
#
# PRE-STATED. Two outcomes, both informative:
#   null again (~30/30)  -> the cables break the control wherever it is measured
#   effect returns       -> behav_all's finding was arena-bound, and the
#                           striped arena, not the cables, did the damage
# The test is the ties-excluded sign test on collisions_per_1k_tics, intact
# against mirrored. The frozen arm is not run: with --head 20 a frozen frame is
# not frozen input (the eyes keep turning across it), a control-validity bug
# recorded in behav_all/NOTE.txt, so that arm would not mean anything.
#
# DO NOT REBUILD THE ARENA WAD WHILE THIS RUNS -- see behav_forage2/run_ext.sh.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_all_forage
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --smell-all --bias 0
        --device cuda --jobs 2 --spiking-t4 --optic-gain 16 --yaw-source DNp15
        --touch --wide --render fast --eye-map anatomical --fixed-turn
        --phasic-mdn --dendrites 3 --g-axial 8 --head 20
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_forage"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
