#!/bin/bash
# media/forage_vision.mp4: the vision-dependence result, shown rather than
# tabulated. Same brain, same level, same seed; the only difference is that the
# lower fly's retina is mirrored left-to-right.
#
# What it is showing (paper/data/behav_forage2/NOTE.txt): over 60 levels the
# mirrored retina makes the model collide 43% more often -- 14.46 against
# 20.67 per 1k tics, 42 up / 16 down, p = 0.000862 -- and the same thing holds
# in the striped arena (45/15, p = 0.000135). Pooled: 87/31, p = 2.5e-07.
# In the BROKEN version of this arena, where wall and ground were indistinguish-
# able, the identical mirror cost exactly nothing: 30 up, 30 down, p = 1.000.
#
# SEED 52, NOT 40. Every other clip uses seed 40, which sits at the 35th
# percentile of this effect and would undersell it. Seed 52 is the level whose
# effect is CLOSEST TO THE MEDIAN of the sixty (+5.42 collisions per 1k), which
# is chosen to be typical, not favourable. Seed 40 is still in forage.mp4 and
# shows a weaker version of the same thing.
set -e
cd "$(dirname "$0")/.."
export FLYDOOM_DEVICE=${FLYDOOM_DEVICE:-cuda}
COMMON="--seconds 30 --scenario health_gathering_forage --wide --eye-map anatomical
        --fixed-turn --phasic-mdn --seed 52 --spiking-t4 --optic-gain 16
        --yaw-source DNp15 --touch --no-gif"
.venv/bin/python -u scripts/record_gameplay.py $COMMON \
    --label "intact retina — median level for this effect (seed 52)" \
    --out media/_forage_intact.mp4
.venv/bin/python -u scripts/record_gameplay.py $COMMON --mirror \
    --label "same brain, retina mirrored left-to-right — collides 43% more over 60 levels" \
    --out media/_forage_mirrored.mp4
ffmpeg -loglevel error -y -i media/_forage_intact.mp4 -i media/_forage_mirrored.mp4 \
    -filter_complex vstack=inputs=2 -c:v libopenh264 -b:v 3600k -pix_fmt yuv420p \
    media/forage_vision.mp4
rm -f media/_forage_intact.mp4 media/_forage_mirrored.mp4
echo "wrote media/forage_vision.mp4"
