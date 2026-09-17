#!/bin/bash
# Full eye (three 170 deg cameras) + daylight sky, all-applied configuration, 30 held-out seeds,
# then its mirrored and blind controls, then the narrow eye in the same arena
# (separates the sky from the wider view).
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_wide
COMMON="--scenarios health_gathering_sky --tics 700 --seeds 30 --seed-base 40 --smell --bias 0 --device cuda --jobs 5 --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch --arms connectome random still"
export FLYDOOM_DEVICE=cuda
.venv/bin/python experiments/run_sharded.py $COMMON --wide --json $OUT/wide_sky.json
.venv/bin/python experiments/run_sharded.py $COMMON --wide --mirror --json $OUT/wide_sky_mirror.json
.venv/bin/python experiments/run_sharded.py $COMMON --wide --blind --json $OUT/wide_sky_blind.json
.venv/bin/python experiments/run_sharded.py $COMMON --json $OUT/narrow_sky.json
