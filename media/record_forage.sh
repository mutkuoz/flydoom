#!/bin/bash
# media/forage.mp4: the arena rebuilt to mean to a fly what it looks like --
# walls that carry motion energy without being a tall dark bar, food as a small
# dark blob, poison pale. Same brain and same seed as flydoom.mp4, so the two
# are comparable frame for frame; the only difference is the world.
set -e
cd "$(dirname "$0")/.."
export FLYDOOM_DEVICE=${FLYDOOM_DEVICE:-cuda}
.venv/bin/python -u scripts/record_gameplay.py \
    --seconds 30 --scenario health_gathering_forage --wide --eye-map anatomical \
    --fixed-turn --phasic-mdn --seed 40 --spiking-t4 --optic-gain 16 \
    --yaw-source DNp15 --touch --shadow \
    --label "forage arena: dark food, bright walls, no bar to walk into" \
    --out media/forage.mp4
