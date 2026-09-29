#!/bin/bash
# THE FORAGE ARENA, WITH WALLS THE FLY CAN TELL FROM THE GROUND.
#
# behav_forage put the intact model at -5.00 against its matched random agent,
# worse than the striped fly arena's +7.53, and that was read as the arena
# being harder. It was also broken in a way that was not noticed until someone
# looked at the video and said the walls and the floor look the same.
#
# They did. Measured, the two textures had contrast 0.126 and 0.122 and the
# same dominant spatial period -- identical statistics with a brightness offset
# between them. A fly's photoreceptors adapt the mean away, so that offset is
# not a cue: the animal had no way to tell a wall from the ground it was
# walking on. The d-prime of 1.86 quoted when the arena was built measured
# separability of MEAN LUMINANCE, which is precisely the quantity adaptation
# removes.
#
# Fixed by setting the two textures to explicit and different contrasts rather
# than to amplitudes that happened to land in the same place: wall contrast
# 0.22 against ground 0.08, a factor of 2.7, with mean 174 against 106 and six
# times the local detail. Three independent cues, two of which survive
# adaptation.
#
# Same seeds, same controls, same everything else as behav_forage, so the
# difference is the arena fix. If the model does better here, the earlier
# forage result was measuring a broken arena rather than a harder one.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_forage2
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --smell --bias 0
        --device cuda --jobs 3 --spiking-t4 --optic-gain 16 --yaw-source DNp15
        --touch --wide --render fast --eye-map anatomical --fixed-turn
        --phasic-mdn
        --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_forage"
$PY $RS $S $COMMON                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --blind        --json $OUT/fly_frozen.json

# NOTE, 2026-09-29. Eleven shards from the first pass were discarded and re-run.
# While this sweep was going, the arena wad was swapped out for two minutes to
# measure the pre-fix version for a before/after comparison, and ViZDoom loads
# the wad when an engine starts -- so any episode beginning in that window ran
# against the broken arena and would have been indistinguishable in the output.
# Shards touched in that window were deleted by mtime and refilled. Do not
# measure an old build by swapping files under a live sweep; copy the wad
# somewhere else and point a separate scenario at it.
