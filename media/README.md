# Recordings

Every clip is a real run: no cuts, no speed-ups, and the dashboard is drawn from
the same tic the frame came from. All of them were recorded with
`scripts/record_gameplay.py`, whose flags are printed in each clip's banner.

## Current

Both recorded on **the same level** (`health_gathering_sky`, seed 40) through the
corrected interface: the whole visual field from three 170° cameras, the
anatomical eye map, and steering toward the more active side.

| file | what it is |
|---|---|
| `flydoom.mp4` · `flydoom.gif` | the model as it now stands |
| `mirrored.mp4` | the same brain with its retina mirrored left-to-right — the control that removes the behavioural advantage |
| `comparison.mp4` | the two stacked, intact on top |

```bash
.venv/bin/python scripts/record_gameplay.py --seconds 30 \
    --scenario health_gathering_sky --wide --eye-map anatomical --fixed-turn \
    --seed 40 --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch \
    --out media/flydoom.mp4            # add --mirror for the control
```

On this particular level the intact fly loses its first life at 12.1 s and the
mirrored one at 10.3 s. That is one level; the evidence is the 60-level tables
in [`../paper/data/behav_wide/`](../paper/data/behav_wide).

**Reading the panels.** Top: the three Doom cameras — left, ahead, right —
which together cover the fly's whole field of view. Bottom left: what the fly
receives, both eyes, each of the 1,581 lenses drawn at the direction it looks,
so sky sits above the horizon line and the ground below. Then: firing rates of
the nerves that drive the body, the left-minus-right steering signal off
`DNp15`, the two odour channels, and a heading-up map of where it walked.

The eye panel draws **contrast after adaptation** — each lens against its own
running mean — because that is the signal the eye sends on, and raw brightness
is the wrong quantity to look at. Measured in this arena: the sky reads 0.71
and the ground 0.023, so on a brightness scale the entire ventral field
collapses into one flat blob (ground s.d. 0.007). The same lenses carry an
adapted s.d. of 0.281 over the full range, and 77% of all lenses report more
than 10% contrast. What looks like an empty dark half is a full field of
structure once adaptation is applied, which is exactly why a photoreceptor
adapts.

## archive/

Superseded, kept because results elsewhere refer to them. Each was current when
recorded, and each is missing at least one correction found later.

| file | configuration | superseded by |
|---|---|---|
| `2026-08_first_model.mp4` · `.gif` | the first working closed loop: one 130° camera, steering off `DNa02` | everything below |
| `arm1_global` … `arm6_touch.mp4` | the four ways of assigning synaptic strength, then the odour and touch channels, one clip each | the readout result — all six steer from `DNa02`, which draws 2.3% of its input from vision |
| `original.mp4`, `best_eyes.mp4`, `best_mirrored.mp4`, `best_comparison.mp4` | the first configuration to beat its matched control, with its mirrored control | the full visual field |
| `best_full_eye.mp4` | the full field and daylight sky, but the eye map still read with the wrong hexagonal convention and unmirrored between the eyes | `flydoom.mp4` |
