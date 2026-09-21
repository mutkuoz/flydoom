#!/bin/bash
# Rebuilds the current clips in media/: the model and its mirrored-retina
# control, on the same level, through the corrected interface. See README.md
# in this directory for what each flag changes and why.
set -e
cd "$(dirname "$0")/.."
export FLYDOOM_DEVICE=${FLYDOOM_DEVICE:-cuda}
COMMON="--seconds 30 --scenario health_gathering_fly --wide --eye-map anatomical
        --fixed-turn --phasic-mdn --seed 40 --spiking-t4 --optic-gain 16
        --yaw-source DNp15 --touch --no-gif"
.venv/bin/python -u scripts/record_gameplay.py $COMMON \
    --label "fly arena, walking forward (MDN read as bursts)" \
    --out media/flydoom.mp4
.venv/bin/python -u scripts/record_gameplay.py $COMMON --mirror \
    --label "same brain, retina mirrored left-to-right (control)" \
    --out media/mirrored.mp4
