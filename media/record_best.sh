#!/bin/bash
# media/best.mp4 and best.gif: the same fly as flydoom.mp4 with the three
# things that were developed after it and never applied together --
# three-compartment dendrites placed by retinotopic offset (M17), a neck that
# turns the eyes up to 20 degrees, and whole-level odour instead of odour gated
# on what is visible. Same arena, same seed, so it is comparable frame for
# frame with flydoom.mp4.
set -e
cd "$(dirname "$0")/.."
export FLYDOOM_DEVICE=${FLYDOOM_DEVICE:-cuda}
.venv/bin/python -u scripts/record_gameplay.py \
    --seconds 30 --scenario health_gathering_fly --wide --eye-map anatomical \
    --fixed-turn --phasic-mdn --seed 40 --spiking-t4 --optic-gain 16 \
    --yaw-source DNp15 --touch --shadow --smell-all --dendrites 3 --g-axial 8 \
    --head 20 \
    --label "everything applied: dendritic cables, a neck, odour through walls" \
    --out media/best.mp4
