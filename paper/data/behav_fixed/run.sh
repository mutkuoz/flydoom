#!/bin/bash
# The behaviour tables re-run with a forward-walking fly and the fully
# corrected interface: anatomical eye map, correct turn sign, MDN read as
# bursts, antennal contact on forward pushes only. Same brain configuration
# (spiking T4/T5, optic gain 16, DNp15 yaw, touch, smell) and the same 60
# held-out seeds (40-99) as paper/data/behav_wide.
#
# Block 1, the sky arena: differs from behav_wide only by the fixes.
# Block 2, the fly arena.
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_fixed
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --bias 0 --device cuda
        --jobs 6 --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch
        --wide --render fast --eye-map anatomical --fixed-turn --phasic-mdn
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
for arena in sky fly; do
  S="--scenarios health_gathering_$arena"
  $PY $RS $S $COMMON                --json $OUT/${arena}_intact.json
  $PY $RS $S $COMMON --mirror       --json $OUT/${arena}_mirrored.json
  $PY $RS $S $COMMON --blind        --json $OUT/${arena}_frozen.json
done
