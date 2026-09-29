#!/bin/bash
# media/forage.mp4: the forage arena AFTER the wall fix.
#
# The first version of this clip was recorded against walls whose contrast
# (0.126) and dominant spatial period matched the ground's (0.122). They
# differed only in mean brightness, which a photoreceptor adapts away, so the
# fly had no way to tell the surface it walked on from the surface it walked
# into -- and you can see it in the old clip, which is what prompted the fix.
# Walls now carry 0.22 contrast against the ground's 0.08 (d' in the drive
# 0.54 -> 1.01). See ../paper/data/behav_forage2/NOTE.txt.
#
# Same brain, same seed and same flags as flydoom.mp4, so the two are
# comparable frame for frame; the only difference is the world.
set -e
cd "$(dirname "$0")/.."
export FLYDOOM_DEVICE=${FLYDOOM_DEVICE:-cuda}
.venv/bin/python -u scripts/record_gameplay.py \
    --seconds 30 --scenario health_gathering_forage --wide --eye-map anatomical \
    --fixed-turn --phasic-mdn --seed 40 --spiking-t4 --optic-gain 16 \
    --yaw-source DNp15 --touch --shadow \
    --label "forage arena, walls fixed: dark food, textured walls, no bar to walk into" \
    --out media/forage.mp4
