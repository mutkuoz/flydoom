#!/bin/bash
# SECOND SEED BLOCK FOR THE FIXED FORAGE ARENA: seeds 100-159.
#
# The first block (seeds 40-99, run.sh) settled the health question -- the
# model does not out-heal its matched random agent in this arena (p = 0.511)
# -- but health is not the benchmark. On movement every metric favoured the
# model and every one landed just short:
#
#   collisions/1k   14.46 vs 19.21   37 up / 22 down   p = 0.067
#   stuck fraction   0.24 vs  0.31   37 up / 23 down   p = 0.092
#   free-run length 83.97 vs 73.46   37 up / 22 down   p = 0.067
#
# Three metrics, one direction, none of them resolved. That is what a second
# block of sixty seeds is for. Pre-stated so it cannot be read as fishing: the
# claim under test is that the model collides less and runs freer than random
# in the fixed forage arena, and the test is the ties-excluded sign test on
# collisions_per_1k_tics pooled over all 120 seeds. The other two are
# correlated with it and are reported, not counted as separate evidence.
#
# Everything except --seed-base is identical to run.sh.
#
# DO NOT REBUILD THE ARENA WAD WHILE THIS RUNS. ViZDoom loads the wad when an
# engine starts; swapping it mid-sweep silently corrupted eleven shards of the
# first block. Recording media against the existing wad is read-only and safe.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_forage2
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 100 --smell --bias 0
        --device cuda --jobs 2 --spiking-t4 --optic-gain 16 --yaw-source DNp15
        --touch --wide --render fast --eye-map anatomical --fixed-turn
        --phasic-mdn
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_forage"
$PY $RS $S $COMMON                --json $OUT/fly_intact_b2.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored_b2.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen_b2.json
