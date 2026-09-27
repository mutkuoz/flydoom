#!/bin/bash
# WIND: the cue a fly actually steers by when following a smell.
#
# Measured here already: whole-level odour at five times strength bought a
# BLIND model -0.07 against its matched control (paper/data/behav_smell). Odour
# alone does not find food, and the reason is structural -- olfaction.py is a
# scalar with every azimuth discarded, by construction, so that a
# direction-carrying odour input cannot become the confound the study exists to
# avoid. A scalar cannot steer.
#
# A real fly has the same problem. In turbulent air an odour gradient points
# nowhere, so the animal uses odour as a GATE and WIND as the steering cue: it
# casts across the flow and surges upwind on contacting a plume. The direction
# comes from the antennae, which are mechanosensory.
#
# flydoom/wind.py supplies that as physics rather than as a hint:
#   - a source's odour only reaches a fly DOWNWIND of it, so going upwind is
#     worth doing; the bearing this is computed from never reaches the network
#   - the flow direction arrives as a left-right difference at the Johnston's
#     organ afferents the touch channel already drives, because those are the
#     fly's own wind sensor and they are already in the connectome
#
# Baseline is paper/data/behav_smell -- the same whole-level odour, the same 60
# seeds, the same controls, still air -- so the difference is the wind alone.
#
# THE PREDICTION, stated before running. Health should rise only where odour
# AND wind are both present. The no-smell arm is the control that separates the
# two: if wind alone helps, the model has found a compass rather than a food
# source, which is a different and also interesting claim.
#
# Resumable: run_sharded skips every seed whose shard already parses.
cd /home/mutkuoz/Documents/flydoom
OUT=paper/data/behav_wind
export FLYDOOM_DEVICE=cuda
COMMON="--tics 700 --seeds 60 --seed-base 40 --bias 0 --device cuda --jobs 3
        --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch --wide
        --render fast --eye-map anatomical --fixed-turn --phasic-mdn
        --wind 1.0 --wind-dir 0 --arms connectome random still"
RS=experiments/run_sharded.py
PY=.venv/bin/python
S="--scenarios health_gathering_fly"
$PY $RS $S $COMMON --smell --smell-all                --json $OUT/fly_intact.json
$PY $RS $S $COMMON --smell --smell-all --mirror       --json $OUT/fly_mirrored.json
$PY $RS $S $COMMON --smell --smell-all --blind        --json $OUT/fly_frozen.json
# the control that tells a food-finder from a compass: wind, no nose at all
$PY $RS $S $COMMON                                    --json $OUT/fly_nosmell.json
