#!/bin/bash
# WIND THAT MEANDERS, because a steady one is invisible to the decoder.
#
# behav_wind gave nothing: +1.40 +- 9.25 health against its matched control.
# The diagnostic says the channel is not the problem. Clamping the antennae and
# reading the steering pair, airflow from the right against airflow from the
# left moves DNp15's left-right differential by about 8 Hz, the same sign in
# 3 of 3 seeds -- far better than contact ever managed (m11: 1 of 3) and far
# larger than the optomotor drum's +1.22 Hz.
#
# The decoder eats it. MotorConfig centres the yaw channel on a 3 s running
# baseline, so a CONSTANT differential is exactly the DC that filter exists to
# remove, and a fly settled on a heading in a steady wind commands nothing
# however clear the signal at the neuron. M12 hit this with the optomotor drum
# and solved it by reversing the drum on a square wave inside the passband.
#
# Real wind meanders, so this is physics rather than a contrivance: the
# direction swings +-35 degrees with a 4 s period, which is 0.25 Hz against the
# decoder's 0.05 Hz corner.
#
# Intact and the no-smell control only. This is a hypothesis test -- does the
# signal reach behaviour once it survives the filter -- and the full table of
# controls is worth running only if it does.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_wind2
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --bias 0 --device cuda --jobs 3
        --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch --wide
        --render fast --eye-map anatomical --fixed-turn --phasic-mdn
        --wind 1.0 --wind-dir 0 --wind-meander 35 --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_fly"
$PY $RS $S $COMMON --smell --smell-all   --json $OUT/fly_intact.json
# the control that tells a food-finder from a compass: wind, no nose at all
$PY $RS $S $COMMON                       --json $OUT/fly_nosmell.json
