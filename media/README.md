# Recordings

Every clip is a real run: no cuts, no speed-ups, and the dashboard is drawn from
the same tic the frame came from. All of them were recorded with
`scripts/record_gameplay.py`, whose flags are printed in each clip's banner.

## The configuration that works

`flydoom.mp4` is the model the paper reports, and it is worth naming because
five later additions each made it worse:

```
--wide --eye-map anatomical --fixed-turn --phasic-mdn
--spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch --smell
```

Over 120 held-out levels it beats a command-matched random agent on health
(sign test p=0.010) and survival (p=0.045), and a mirrored retina removes that
(p=0.731). Dendritic cables, a movable head, whole-level odour and airflow are
all real fly biology and all four cost performance, interpretability or both.
The model is not an ablation; it is what survives measurement.

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
| `flydoom.mp4` · `flydoom.gif` | **the model.** The configuration the paper reports, and the one that works |
| `mirrored.mp4` | the same brain with its retina mirrored left-to-right — the control that removes the behavioural advantage |
| `comparison.mp4` | the two stacked, intact on top |
| `drum.mp4` | the laboratory arena: one dark bar on a uniformly bright cylinder, the fly tethered at the centre |
| `best.mp4` · `best.gif` | **the fly at its best.** The same configuration as `flydoom.mp4`, in the rebuilt arena, on the best of 120 levels |
| `everything.mp4` · `everything.gif` | everything applied — dendritic cables, a neck, odour through walls. Kept as a cautionary clip; see below |
| `forage.mp4` · `forage.gif` | the arena rebuilt to mean to a fly what it looks like: dark food, walls with no bar to walk into, **and walls it can tell from the ground** |
| `forage_vision.mp4` | **the vision result.** Intact on top, mirrored retina below, same level. The lower fly collides 43% more |

### `best.mp4` used to be the wrong fly

It held the everything-applied configuration, on the assumption that more
biology meant a better model. It does not: that fly's collision rate is
*identical* with its retina mirrored (30 up / 28 down, `p = 0.90`), so it has
stopped using eyes that still work. Leaving the word "best" on it and explaining
in a README that best does not mean best was the wrong fix — the name belongs to
the fly that earned it, and the old clip is now `everything.mp4`.

`best.mp4` is therefore the plain configuration, filmed in the fixed forage
arena, on **seed 96** — health 128 against a median of 20, collisions 5.72
against a median of 13.42 and against the mirrored arm's 20.36, surviving all
700 tics. That is deliberately the best of the 120 levels rather than a typical
one, because this is a showcase clip and one level is an illustration, never
evidence. For a typical level see `forage.mp4` (seed 40); for the effect at its
*median*, `forage_vision.mp4` (seed 52).

## Why some clips still have striped walls

`flydoom.mp4`, `mirrored.mp4`, `comparison.mp4` and `everything.mp4` are filmed
in `health_gathering_fly`, whose walls are a tall dark bar tiled edge to edge. That
arena is criticised at length below — it is the strongest fixation target a fly
could be shown, and we built it and then called the animal foolish for walking
into it. The clips stay because **that is the arena the 120-level tables were
measured in**, and a clip that does not match the numbers it illustrates is
worse than an ugly one.

The arena to look at is the forage one, and as of 2026-09-29 its walls are
finally readable — the earlier version had walls and ground a fly could not
tell apart, which is why `forage.mp4` was re-recorded. The result that matters
most in the project is in `forage_vision.mp4`, and it is not in a striped room.

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

## everything.mp4 — everything applied, and why it is not the best

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

`media/record_everything.sh` rebuilds it, on the same arena and seed as `flydoom.mp4`
so the two are comparable frame for frame.

**Is it actually better? No, and the question is now settled.**
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

**It plays about as well and it is a worse experiment**, which is a different
and more interesting complaint. Scored by a sign test over the levels that
moved:

  | | intact | mirrored |
  |---|---|---|
  | `flydoom.mp4` (the model) | 28 up / 10 down, **p = 0.005** | 21/21, **p = 1.000** |
  | `everything.mp4` (this clip) | 29 up / 12 down, p = 0.012 | 23/16, p = 0.337 |

The task scores are comparable. What differs is the control: the model's
mirrored arm is an exact coin flip, and this one leans positive. Its advantage
no longer depends on the eyes being right, and the decomposition attributes
that to the dendritic cables
([`../paper/data/behav_cables/NOTE.txt`](../paper/data/behav_cables/NOTE.txt)).

**Re-scored on collisions, and the verdict holds on much better evidence.**
Everything above rests on health, and health turns out to be unable to resolve
the intact-versus-mirrored contrast at 60 seeds at all — on health every
configuration below sits at `p = 0.15`–`0.76`, so "the control breaks" was a
null read off a blind instrument. Scored on collisions per 1k tics instead,
same seeds, ties excluded:

| configuration | intact | mirrored | sign test |
|---|---|---|---|
| plain (`flydoom.mp4`) | 14.12 | 20.12 | 45/15, **p = 1.4e-04** |
| neck only | 14.59 | 23.26 | 48/11, **p = 1.2e-06** |
| odour through walls only | 14.34 | 20.76 | 44/14, **p = 1.0e-04** |
| **cables only** | 19.12 | 19.08 | **29/31, p = 0.90** |
| **everything applied** | 19.24 | 18.03 | **30/28, p = 0.90** |

The neck and the odour rows are the positive controls that make the two nulls
mean something: both *keep* the vision dependence at `p ≤ 1e-04`, so the metric
has not gone blind — the cables are doing something specific.

And the mechanism is not the one described above. The cables do not lift the
mirrored arm up to meet the intact one; they drag the **intact** arm down to the
mirrored one, 14.12 → 19.12, while mirrored barely moves. The model does not
stop needing its eyes. It stops *using* eyes that still work, which is a worse
failure and a more precise one.

Adding the fifth thing — airflow — to this configuration is what collapses it
entirely, to `p = 0.743`
([`../paper/data/behav_fly_canonical/`](../paper/data/behav_fly_canonical)).
That number belongs to that run, not to this clip; an earlier version of this
file attached it here, which was wrong.

This clip was called `best.mp4` until 2026-09-29, and keeping it there was
defended on the grounds that the name had been requested and links should stay
stable. That was the wrong call: a file called "best" that requires a README to
explain it is not the best misleads everyone who does not read the README. The
name now belongs to the plain configuration, which earned it, and this clip is
`everything.mp4` — what it always was.

  | addition | what it costs |
  |---|---|
  | whole-level odour | +7.53 → +3.80: a direction-free scalar, so more is worse |
  | dendritic cables | removes the dependence on vision — mirrored scores the same as intact |
  | a movable head | the frozen-retina control stops being one |
  | airflow | destroys the advantage: 16 seeds up against 18 down |

Details in [`../paper/data/behav_plainwind/NOTE.txt`](../paper/data/behav_plainwind/NOTE.txt),
which answers the question these clips raise: which single configuration is the
best model of a fly.

## forage_vision.mp4 — the strongest result in the project

Two flies, same brain, same level, same seed. The only difference is that the
lower one's retina is mirrored left-to-right. Over 60 levels that mirror makes
it collide **43% more often** — 14.46 against 20.67 per 1k tics, 42 up / 16
down, `p = 0.00086`. That test was then written down and run again on sixty
**fresh** seeds, and it replicates almost exactly: 42 up / 17 down,
`p = 0.0016`. The striped arena gives the same thing independently
(45/15, `p = 0.000135`). Grand pool over 180 levels: **129 up / 48 down,
`p = 9.3e-10`**.

Worth putting beside this project's own record. The *health* advantage in the
striped arena halved between its first and second block of seeds (+7.53 then
+3.80), and we wrote at the time that the first block had evidently been the
favourable half. The collision effect does nothing of the kind — −6.21 then
−5.23, sign counts 42/16 then 42/17. It is the more stable measurement, not
just the more significant one.

The reason this is worth more than the health tables is the control. The
standing objection to any mirroring result is that scrambling a retina is a
lesion, and lesions degrade things generally. But in the **broken** version of
this arena — where wall and ground had the same contrast and the same spatial
period, and differed only in a brightness offset that photoreceptors adapt away
— the identical mirror in the identical brain on the identical task cost
**exactly nothing**: 30 up, 30 down, `p = 1.000`. Remove the information and the
lesion stops mattering. So it is not disruption; it is the loss of a signal that
was being used.

Two checks that had to pass: the command-matched random agent cannot tell the
mirror is there (`p = 0.435` and `p = 0.791`), as it must not, since it never
reads the retina; and `free_run_tics` gives the same 42/16 but correlates with
collisions at `r = -0.871`, so it is the same measurement reported once, not
two results.

At 120 seeds the whole panel separates, including the two measures that were
flat at 60:

| intact vs mirrored, fixed arena, n=120 | | | sign test |
|---|---|---|---|
| collisions per 1k | 14.64 | 20.36 | 84/33, `p = 2.7e-06` |
| free-run tics | 82.90 | 61.76 | 84/33, same split |
| stuck fraction | 0.25 | 0.30 | 75/45, `p = 0.0079` |
| health | 21.80 | 14.00 | 52/31, `p = 0.028` |
| survival (tics) | 445.1 | 420.6 | 53/36, `p = 0.089` |

That health line retires a published limitation. In the striped arena this
contrast was +3.50 at n=120 and we estimated ~210 seeds to resolve it, and said
so rather than running toward the number. Here it is +7.80 at n=120 and already
significant — twice as large, because the striped arena's tall dark bar traps
the intact and the mirrored model about equally and so compresses exactly the
difference the contrast exists to measure. The obstacle was the room and the
measure, not the sample size.

Numbers and the full argument:
[`../paper/data/behav_forage2/NOTE.txt`](../paper/data/behav_forage2/NOTE.txt).

Filmed on **seed 52, not 40** — every other clip uses seed 40, which sits at the
35th percentile of this effect and would undersell it. Seed 52 is the level
closest to the **median** of the first sixty, chosen to be typical rather than
favourable.

## forage.mp4 — an arena that means what it looks like

The corrected tables found the model's vision steers it into walls, and we
excused it as a fly's attraction to dark verticals transferring without the
wisdom to go with it. That was charitable to the arena. Measured,
`health_gathering_fly` tiles every wall with a **perfect tall dark bar** — 64 px
at luminance 205 beside 64 px at 65, uniform down its whole height, 0.56
contrast — which is the strongest fixation target a fly could be shown. And it
dresses the food as a pale box. We built the trap, then called the animal
foolish for walking into it.

`health_gathering_forage` (`scripts/build_forage_arena.py`) moves the darkness
off the walls and onto the food, and nothing in the brain changes:

- **walls** keep their spatial frequency and lose their vertical coherence.
  Band-pass noise, mildly anisotropic, with the column mean subtracted: local
  vertical edges everywhere for the motion detectors, no tall bar anywhere.
  It also puts 66% of its power at the fly's preferred period against a square
  grating's 40%, because a square wave wastes most of its energy on harmonics.
- **food** becomes a small dark blob — the stimulus a fly actually approaches.
- **poison** becomes pale, which it should not.
- **wall luminance** is set at mean 175 against the ground's 145, chosen by
  measuring *through the retina*: separating lenses that look at wall from
  lenses that look at ground gives d′ 1.86 there, against 0.91 at equal means,
  while leaving contrast well above what the optomotor response needs.

> **That last bullet was the bug, and it took two months to notice.** d′ 1.86
> separates *mean luminance*, which is precisely the quantity a photoreceptor
> adapts away — the right statistic computed on the wrong variable. Measured
> properly, the walls carried contrast 0.126 against the ground's 0.122 with the
> same dominant spatial period: to the model there was no boundary between the
> surface it walked on and the surface it walked into. It was caught by someone
> watching the clip and saying the walls and the floor look the same. They did.
>
> Fixed 2026-09-29 by setting the two textures to explicit and *different*
> contrasts — wall 0.22 against ground 0.08 — rather than to amplitudes that
> happened to collide. In the units the network actually receives: forage broken
> d′ 0.54, striped arena 0.59, forage fixed **1.01**. `forage.mp4` has been
> re-recorded against the fixed arena; everything in the paragraph below was
> measured in the broken one.

**What it showed.** The prediction was that moving the darkness onto the food
would raise health and lower collisions. It did not. Health −1.59 ± 5.81,
damage −1.07 ± 5.82, both nothing; collisions actually rose, +2.68 ± 2.39. The
measure that moved was the correlation between what the eyes see and how it
steers, which **reversed**: +0.05 in the fly arena, −0.08 here, a change of
−0.14 ± 0.05.

Positive there means "brighter on the left, turn right" — steering toward the
darker side. In the fly arena the dark things were the walls; here the walls
are the bright things and the sign flipped with them. **In both arenas it
steers into the wall.** So the description was wrong: this is not attraction to
dark verticals, it is attraction to large low-spatial-frequency structure
*regardless of contrast polarity*, which is what reading unsigned contrast
energy would give you. A real fly's fixation is polarity-selective; this is not.

Making the food dark and salient also did not raise health, which says vision
is not what finds the food in either arena. Smell is.

The arena did its job as an instrument even though it did not rescue the
behaviour: it turned a vague story into a falsifiable claim and falsified it.
Numbers: [`../paper/data/behav_forage/`](../paper/data/behav_forage).

**What survived the wall fix, and what did not.** Rebuilding the walls so the
fly can see them recovers the absolute numbers completely — health 18.20 →
21.93, collisions 16.67 → 14.46, stuck 0.311 → 0.242, which is as good as the
striped arena or better. But paired seed by seed the health difference is 23 up
/ 24 down, `p = 1.0000`: the means rose, not any particular level. And the model
still does not out-heal chance here (`p = 0.511`) — because the *random* agent
also improves, 13.53 → 17.87. A forage arena with no tall dark bar is kinder to
a random walker, so the headroom between chance and competence falls from 7.5
health to 4.1 and sixty seeds stops resolving it. The model did not get worse;
the floor came up.

The steering-sign reversal described above is **real but was over-read**. The
claim that "mirroring no longer abolishes, it inverts" was called the strongest
vision-dependence evidence in the project; it is withdrawn. The inversion was an
artifact of an arena where the visual signal was noise — using noise hurts
(intact 18.20, five *below* random's 23.20) and scrambling it helps (mirrored
21.67). Fix the walls and the arm ordering returns to the striped arena's:

| against matched random | intact | frozen | mirrored |
|---|---|---|---|
| fly arena, striped | +7.53 | +4.47 | −0.67 |
| forage, broken walls | −5.00 | +2.67 | **+5.13** ← inverted |
| forage, walls fixed | +4.07 | +3.33 | +1.87 |

The genuine vision-dependence result is the collision one, above — and the
broken arena earns its keep after all, as the negative control that makes it
mean something.

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
