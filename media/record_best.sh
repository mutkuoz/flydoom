#!/bin/bash
# media/best.mp4 and best.gif: the fly at its best.
#
# This name used to belong to the everything-applied configuration, on the
# assumption that more biology meant a better model. It does not -- that fly is
# now media/everything.mp4, and it is a cautionary clip: its collision rate is
# identical with its retina mirrored (30 up / 28 down, p = 0.90), meaning it has
# stopped using eyes that still work. Leaving the word "best" on it and
# explaining in a README that best does not mean best was the wrong fix. The
# name belongs to the fly that earned it.
#
# That is the plain configuration -- the one the paper reports, with nothing
# added -- and it is the best-evidenced result in the project:
#
#   collides 43% less than its own mirrored control, 129 up / 48 down over 180
#   levels, p = 9.3e-10, with a negative control (an arena whose walls the fly
#   could not distinguish from its floor) in which the same mirror costs
#   exactly nothing. See paper/data/behav_forage2/NOTE.txt.
#
# ARENA: the fixed forage arena, not the striped one. No tall dark bar to walk
# into, and walls the fly can actually tell from the ground (that took two
# goes; the first build separated them only by mean luminance, which
# photoreceptor adaptation removes).
#
# SEED 96, AND THAT IS A CHOICE THAT NEEDS DECLARING. This is a showcase clip,
# so it is the best of the 120 levels rather than a typical one: health 128
# against a median of 20, collisions 5.72 against a median of 13.42 and against
# the mirrored arm's 20.36, and it survives all 700 tics. One level is an
# illustration, never evidence. For a typical level see media/forage.mp4
# (seed 40); for the effect at its MEDIAN, media/forage_vision.mp4 (seed 52).
set -e
cd "$(dirname "$0")/.."
export FLYDOOM_DEVICE=${FLYDOOM_DEVICE:-cuda}
.venv/bin/python -u scripts/record_gameplay.py \
    --seconds 30 --scenario health_gathering_forage --wide --eye-map anatomical \
    --fixed-turn --phasic-mdn --seed 96 --spiking-t4 --optic-gain 16 \
    --yaw-source DNp15 --touch --shadow \
    --label "the model, best of 120 levels (seed 96) — an illustration, not evidence" \
    --out media/best.mp4
