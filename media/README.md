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
| `best.mp4` · `best.gif` | the same fly and level with everything applied — dendritic cables, a neck, and odour that passes through walls |

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

## best.mp4 — everything applied

`flydoom.mp4` is the configuration the 120-level tables were measured with.
Three things built after it were never in those tables, each simply because
nothing wired them into the agent:

- **dendritic cables** (`--dendrites 3`): each `T4`/`T5` becomes a
  three-compartment cable with every input placed along it by its own
  retinotopic offset. Open loop this takes `T5` mirror-pair separation from
  0.005 to 0.064 — the largest single gain in the project.
- **a neck** (`--head 20`): the eyes turn up to 20° either side, driven by the
  same yaw command, so gaze can move without the body having to.
- **odour through walls** (`--smell-all`): earlier olfaction was gated on what
  was visible, which hid 82% of the items in the level, and medkits carried a
  fifth of the odour strength they carry now.

`media/record_best.sh` rebuilds it, on the same arena and seed as `flydoom.mp4`
so the two are comparable frame for frame.

**Is it actually better? No — and the file name is a request, not a finding.**
The measurement is in [`../paper/data/behav_all/NOTE.txt`](../paper/data/behav_all/NOTE.txt):
60 seeds, the same ones and the same controls as the plain configuration, so
the comparison is paired and isolates exactly these three changes.

- **The task score does not move.** Health −0.13 ± 6.85 against the plain fly.
- **The movement gets worse.** Turn-command chatter +0.149 ± 0.015, collisions
  +5.11 ± 2.70 per 1k tics, tics between collisions −18.2 ± 15.7, and the
  correlation between what the eyes see and how it steers −0.034 ± 0.027. It
  jitters more, hits things sooner, and steers less by vision.
- **The vision control breaks.** A mirrored retina abolishes the plain fly's
  advantage over chance; here it abolishes nothing, and a fly with its retina
  **frozen on one frame** beats chance outright (+6.53 ± 5.97 health). It
  scores marginally higher and no longer needs its eyes to do it.

So watch this clip as the configuration it is, not as an improvement. The
likely cause is whole-level odour, which reaches every medkit at five times the
old strength with no line of sight, and which neither mirroring nor freezing
touches — `paper/data/behav_smell/` isolates that flag to find out.

**Are these clips current?** `flydoom.mp4`, `mirrored.mp4` and `comparison.mp4`
were recorded at `c2b67f6` and have not been re-rendered since, because nothing
since then changes what the model does — the compartment dendrites, the neck and
the supralinear term are all off by default, and the two synchronisation removals
are bit-identical. That is checked rather than assumed: `scripts/action_trace.py`
hashes every action and every descending rate over 120 tics of the configuration
below, and `c2b67f6` and `HEAD` both give
`3b6d3072d8600c98d8dfb661abfcf5f9aea243a740a372a4f5586d88e17817f2`. Re-run it
after any change that could touch behaviour; if the hash moves, the clips are
stale and `record.sh` rebuilds them.

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
