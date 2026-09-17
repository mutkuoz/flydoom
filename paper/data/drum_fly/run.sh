#!/bin/bash
# Tethered drum in the fly arena: bold vertical stripes at the spatial period
# the optomotor response prefers, a bright textured floor, flat daylight.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/drum_fly
export FLYDOOM_DEVICE=cuda
COMMON="--imposed 0,1,2,4 --seeds 8 --tics 350 --yaw-source DNp15 --optic-gain 16 --device cuda --open-loop --eye-map anatomical --wide --fixed-turn --scenario health_gathering_fly"
PY=.venv/bin/python
$PY experiments/m12_optomotor_drum.py $COMMON --arm intact   --json $OUT/intact.json   > $OUT/intact.log 2>&1 &
$PY experiments/m12_optomotor_drum.py $COMMON --arm mirrored --json $OUT/mirrored.json > $OUT/mirrored.log 2>&1 &
$PY experiments/m12_optomotor_drum.py $COMMON --arm blind    --json $OUT/blind.json    > $OUT/blind.log 2>&1 &
wait
