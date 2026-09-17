#!/bin/bash
# Second block of held-out seeds (70-99) for the full-eye comparison. Nothing
# was tuned on the first block; this only adds power. Reported combined.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_wide
COMMON="--scenarios health_gathering_sky --tics 700 --seeds 30 --seed-base 70 --smell --bias 0 --device cuda --jobs 6 --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch --arms connectome random still"
export FLYDOOM_DEVICE=cuda
.venv/bin/python experiments/run_sharded.py $COMMON --wide --json $OUT/wide_sky_b2.json
.venv/bin/python experiments/run_sharded.py $COMMON --wide --mirror --json $OUT/wide_sky_mirror_b2.json
.venv/bin/python experiments/run_sharded.py $COMMON --json $OUT/narrow_sky_b2.json
.venv/bin/python experiments/run_sharded.py $COMMON --wide --blind --json $OUT/wide_sky_blind_b2.json
