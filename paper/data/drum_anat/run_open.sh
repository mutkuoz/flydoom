#!/bin/bash
# Tethered (open-loop) drum: commands recorded, not executed, no walking.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/drum_anat
export FLYDOOM_DEVICE=cuda
COMMON="--imposed 0,1,2,4,8 --seeds 10 --tics 350 --yaw-source DNp15 --optic-gain 16 --device cuda --open-loop"
M12=experiments/m12_optomotor_drum.py
PY=.venv/bin/python
$PY $M12 $COMMON --eye-map anatomical --wide --fixed-turn --arm intact   --json $OUT/OL_B_intact.json   > $OUT/OL_B_intact.log 2>&1 &
$PY $M12 $COMMON --eye-map anatomical --wide --fixed-turn --arm mirrored --json $OUT/OL_B_mirrored.json > $OUT/OL_B_mirrored.log 2>&1 &
$PY $M12 $COMMON --eye-map anatomical --wide --fixed-turn --arm blind    --json $OUT/OL_B_blind.json    > $OUT/OL_B_blind.log 2>&1 &
wait
$PY $M12 $COMMON --eye-map lattice    --wide --fixed-turn --arm intact   --json $OUT/OL_D_intact.json   > $OUT/OL_D_intact.log 2>&1 &
$PY $M12 $COMMON --eye-map lattice    --wide --fixed-turn --arm mirrored --json $OUT/OL_D_mirrored.json > $OUT/OL_D_mirrored.log 2>&1 &
wait
