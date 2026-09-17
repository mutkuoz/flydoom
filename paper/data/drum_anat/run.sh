#!/bin/bash
# Optomotor drum after the two interface fixes (anatomical eye map, turn sign).
# B = both fixes + full eye, with mirrored and frozen controls;
# C = eye fix only (legacy turn sign); D = turn fix only (legacy lattice map).
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/drum_anat
export FLYDOOM_DEVICE=cuda
COMMON="--imposed 0,1,2,4,8 --seeds 6 --tics 350 --yaw-source DNp15 --optic-gain 16 --device cuda"
M12=experiments/m12_optomotor_drum.py
PY=.venv/bin/python
$PY $M12 $COMMON --eye-map anatomical --wide --fixed-turn --arm intact   --json $OUT/B_intact.json   > $OUT/B_intact.log 2>&1 &
$PY $M12 $COMMON --eye-map anatomical --wide --fixed-turn --arm mirrored --json $OUT/B_mirrored.json > $OUT/B_mirrored.log 2>&1 &
$PY $M12 $COMMON --eye-map anatomical --wide --fixed-turn --arm blind    --json $OUT/B_blind.json    > $OUT/B_blind.log 2>&1 &
$PY $M12 $COMMON --eye-map anatomical --wide              --arm intact   --json $OUT/C_intact.json   > $OUT/C_intact.log 2>&1 &
$PY $M12 $COMMON --eye-map lattice    --wide --fixed-turn --arm intact   --json $OUT/D_intact.json   > $OUT/D_intact.log 2>&1 &
wait
