# Recordings

Every clip is a real run: no cuts, no speed-ups, and the dashboard is drawn from
the same tic the frame came from. All of them were recorded with
`scripts/record_gameplay.py`, whose flags are printed in each clip's banner.

## Current

Both recorded on **the same level** (`health_gathering_fly`, seed 40) through the
corrected interface: the whole visual field from three 170° cameras, the
anatomical eye map, steering toward the more active side, and MDN read as
bursts so the fly walks forward (before that it reversed on 97% of tics; see
`MotorConfig.phasic_mdn`). The arena is the
one built for a fly's eye (`scripts/build_fly_arena.py`): walls striped at the
spatial period its motion detectors prefer, bright textured ground, flat
daylight.

| file | what it is |
|---|---|
| `flydoom.mp4` · `flydoom.gif` | the model as it now stands, with stock Doom beside it |
| `mirrored.mp4` | the same brain with its retina mirrored left-to-right — the control that removes the behavioural advantage |
| `comparison.mp4` | the two stacked, intact on top |
| `drum.mp4` | the laboratory arena: one dark bar on a uniformly bright cylinder, the fly tethered at the centre |

```bash
.venv/bin/python scripts/record_gameplay.py --seconds 30 \
    --scenario health_gathering_fly --wide --eye-map anatomical --fixed-turn \
    --phasic-mdn --seed 40 --spiking-t4 --optic-gain 16 --yaw-source DNp15 \
    --touch --out media/flydoom.mp4    # add --mirror for the control
```

or just `media/record.sh`, which rebuilds both.

`drum.mp4` is the arena of M18 (`scripts/build_stripe_arena.py`), and it is
filmed tethered because a walking fly leaves the centre of a cylinder within a
second — `--tether` records the forward and lateral commands and discards them,
which is exactly what the experiment does:

```bash
.venv/bin/python scripts/record_gameplay.py --seconds 25 --scenario stripe_fix \
    --wide --eye-map anatomical --fixed-turn --phasic-mdn --seed 40 \
    --spiking-t4 --optic-gain 16 --yaw-source DNp15 --tether --no-smell \
    --out media/drum.mp4
```

Two of its panels are empty on purpose: there is nothing in the cylinder to
smell, and a tethered fly goes nowhere, so the odour traces and the map stay
blank for the whole clip.

Watch the eye panel rather than the top one: the bar is the single dark column
that slides across it, and the fly parks it somewhere and keeps it there. That
looks like fixation and is not — a fly with a frozen retina does the same, and
so does one in the cylinder with no bar at all. See
[`../paper/data/m18_stripe/NOTE.txt`](../paper/data/m18_stripe/NOTE.txt).

One level is an illustration, not evidence. The evidence is the 60-level tables in
[`../paper/data/behav_wide/`](../paper/data/behav_wide).

**Reading the panels.** Top left: the same room in stock Doom, as a person
would see it — a second engine running the unmodified map from the same seed,
given the identical keypresses every tic. It stays in exact lockstep (position,
heading and health identical over 400 tics), because the fly arena changes only
textures and lighting, never geometry. It is there to show what the arena
changes bought: Doom's own walls carry their detail far below this eye's 5.4°
acceptance angle, and its ceiling is dark where a fly expects sky.

Top right: the world around the fly, 350°, stitched from
the three cameras onto one grid that is linear in azimuth and elevation — the
fly's own coordinates, and the same axis as the eye panel below, so a wall
appears at the same place in both. Laid side by side instead, the three views
overlap by 85° on each seam and show the same wall two and three times at
wildly different distortions, because a 170° rectilinear view squeezes what is
straight ahead into a few percent of its width and magnifies its edges. Bottom left: what the fly
receives, both eyes, each of the 1,581 lenses drawn at the direction it looks,
so sky sits above the horizon line and the ground below; the wall stripes show
as dark bars. Then: firing rates of the nerves that drive the body, the
left-minus-right steering signal off `DNp15`, the two odour channels, and a
heading-up map of where it walked.

**Why brightness, and not what the retina transmits.** The eye panel draws
brightness on a square-root scale. The lamina's actual output — contrast against
each lens's own running mean — is the more faithful quantity, and it is
available with `--eye-panel adapted`, but in motion it is unreadable: about 12%
of lenses sit at full contrast in a walking frame, and the panel shimmers like
noise. Brightness only failed in the old sky arena, whose ground is 30× darker
than its sky and collapsed into one flat blob; in the fly arena the ground is
bright and textured, and brightness reads as a picture of the scene.

## archive/

Superseded, kept because results elsewhere refer to them. Each was current when
recorded, and each is missing at least one correction found later.

| file | configuration | superseded by |
|---|---|---|
| `2026-08_first_model.mp4` · `.gif` | the first working closed loop: one 130° camera, steering off `DNa02` | everything below |
| `arm1_global` … `arm6_touch.mp4` | the four ways of assigning synaptic strength, then the odour and touch channels, one clip each | the readout result — all six steer from `DNa02`, which draws 2.3% of its input from vision |
| `original.mp4`, `best_eyes.mp4`, `best_mirrored.mp4`, `best_comparison.mp4` | the first configuration to beat its matched control, with its mirrored control | the full visual field |
| `best_full_eye.mp4` | the full field and daylight sky, but the eye map still read with the wrong hexagonal convention and unmirrored between the eyes | `flydoom.mp4` |
