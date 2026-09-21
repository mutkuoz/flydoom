![flydoom](flydoom.jpeg)

# flydoom — a fruit fly's brain plays Doom

Scientists sliced a real fly's brain into seven thousand layers, photographed every layer
with an electron microscope, and traced all **139,255 neurons** and **2.7 million
connections** through the stack. That wiring diagram is public. This project loads it,
simulates every neuron, shows it Doom through the fly's eyes, and reads keypresses out of
the nerves that would normally drive its legs.

**Nothing is trained.** No network is fitted, no reward, no learning of any kind. The
connections are the fly's, exactly as they were measured, and they never change. One
global synaptic gain is calibrated — on a *non-visual* reflex — and that is the only fitted
number in the project.

<p align="center">
  <img src="media/flydoom.gif" width="760" alt="the connectome playing Doom, with its whole visual field">
</p>

**Top:** three Doom cameras — left, ahead, right — covering the fly's whole field of view,
in an arena rebuilt for a fly's eye: striped walls, textured ground, daylight.
**Bottom left:** what the fly actually receives — both eyes, 1,581 lenses, each drawn where
it points. The striped walls show up as dark bars along its horizon.
**Then:** firing rates of the nerves that drive its body, the left-minus-right steering
signal, the two odour channels, and a heading-up map of where it walked.

| | |
|---|---|
| 📄 **The paper** | [`paper/main.pdf`](paper/main.pdf) — 26 pages, every result with its controls |
| 🎬 **Full clip** | [`media/flydoom.mp4`](media/flydoom.mp4) — 30 s of game time, and [every other recording](media/) |
| 🔬 **Deep dive** | [TECHNICAL.md](TECHNICAL.md) · [RESULTS.md](RESULTS.md) — *written in August; where they disagree with the paper, the paper is current* |
| 📊 **Raw data** | [`paper/data/`](paper/data) — every run, with the scripts that made it |

---

## The scorecard

Read this before reading anything else into the video.

| | | |
|---|---|---|
| ✅ | **Taste works.** Sugar makes it extend its tongue (77 Hz); bitter suppresses that by 99% | nobody wrote the suppression — it falls out of the wiring |
| ✅ | **Smell works, and beats a shuffled brain by a thousandfold** | the one result where the real wiring beats a scrambled copy of itself |
| ✅ | **It plays in closed loop** without spinning, stalling or falling over | 60 held-out levels, 700 tics each |
| ✅ | **Vision helps it play — narrowly.** In an arena it can see, it beats a random agent with identical command statistics on health and survival, and mirroring its eyes removes that | only in that arena; freezing its eyes weakens the effect rather than cleanly removing it |
| ⚠️ | **Motion detection is ~2% of a real fly's**, and we can say exactly why | a ceiling of 0.08 where a fly needs ~0.5 — measured, not assumed |
| ❌ | **No recognisable fly reflex.** Spin the world: no counter-turn. Loom something at it: no flinch. No fixation, no odour tracking | the gap between "vision helps" and "behaves like a fly" |

The interesting result was never "can it finish the level". It is *what does a real nervous
system do when you drop it somewhere it never evolved to be* — and, more usefully, **which
missing pieces stop it working**. Most of this README is that list.

---

## How it works

```
   Doom frame ──► 1,581 ommatidial columns ──► 4,541 lamina inputs (L1/L2/L3)
                                                        │
                                          139,255 neurons, 2.7M synapses
                                          57 integration steps per frame
                                                        │
   keypresses ◄── 8 descending nerves ◄────────────────┘
       │
       └─► Doom ─► new frame ─► (loop, 35 times a second)
```

Three ideas carry the whole project:

**A connectome is a circuit diagram with no component values.** Each row of the download
says *neuron A connects to neuron B with N synapses of transmitter T*. What it never says
is how strong that is in millivolts, or how fast. All of that has to be supplied from
elsewhere, and that gap is where nearly every problem in this project lived.

**A neuron is a leaky bucket.** It holds a voltage that drains toward rest; inputs nudge it
up or down; past a threshold it fires a pulse and resets. We run 139,255 of those in
half-millisecond steps — 57 steps per Doom frame, because a fly is faster than the game.

**The body is Doom's.** The dataset is brain-only, so the game's movement code stands in for
legs. Every claim about *timing* is Doom's, not a fly's.

<details>
<summary><b>How the keyboard is wired to the nerves</b></summary>

Everything a fly's brain tells its body goes through about 1,300 nerves. That bottleneck is
narrow enough that labs have worked out what individual ones do — switch this one on in a
live fly and it turns, that one and it walks backwards. We read eight.

| Doom | What we read | What it does in a real fly |
|---|---|---|
| turn left / right | `DNa02` or `DNp15` | Steering. One per side; the difference between them is the turn |
| walk forward | `BPN` | Fast walking |
| walk backward | `MDN` | The "moonwalker" neuron. Reverse gear |
| dodge | `DNp01` | The giant fibre — biggest, fastest cell in the fly. Panic button |
| pick something up | proboscis nerves | The fly's tongue |
| health pack | sugar taste cells | A medkit tastes sweet |
| taking damage | bitter taste cells | Getting hurt tastes foul |

Health packs tasting sweet isn't a joke: those are injections into the fly's real taste
neurons, and its feeding circuit handles item pickup because that is what a feeding circuit
is *for*.
</details>

---

## What we found

### 1. Taste and smell transfer intact

Stimulate the sugar cells and the muscle that extends the tongue fires at **77 Hz**.
Stimulate bitter at the same time and it drops by **99%**. Nobody programmed that
suppression.

Smell is the stronger result, because it has the control that matters. We froze the fly in
place and ran the same level twice with the same seed — identical pictures on screen — and
only switched the nose on:

```
                        nose off    nose on    scrambled brain
"good or bad?" region       0.31      17.63           +0.01
escape neuron             160.02     138.23           +0.41
steering neuron           200.00     216.41           +0.01
a vision cell              96.85      96.94           +0.07   ← control
```

The lateral horn — the fly's hardwired "is this good or bad?" region — wakes from dormant
and reaches the body-driving neurons in a single synapse. The escape neuron is turned
**down**, which the wiring predicted in advance because that connection is inhibitory. A
degree-preserving shuffle of the same brain does **nothing**.

> Permanently attached caveat: the odour strength is computed from enemy distance, which the
> game tells us. This is not evidence the connectome can *detect* enemies. It shows what the
> connectome does with a signal once it has one.

### 2. Motion vision does not transfer, and we can say why

Seeing motion isn't noticing that something changed — it's noticing *which way*, and that
lives entirely in the **order** two neighbouring lenses fire in.

```
moving right:   A fires, then B fires
moving left:    B fires, then A fires
```

Same two events. Only the order differs. The connectome has every piece needed to resolve
it: two input lines, a delay on one, four detector subtypes pointing four ways, mirror
pairs 180° apart. We measured all of them. They are all there.

The detector still reads about **2% of a real fly's** direction selectivity.

<details>
<summary><b>The failure chain, in order (twenty eliminated explanations)</b></summary>

**Build the same correlator out of the same simulated neurons** and it works fine — a
direction selectivity index of 0.79, up to 1.0 with the real fan-in. So the machinery isn't
the problem; how it's embedded is.

**The cell can't fire from its own inputs.** All of T4's excitation sums to 46.8 weight
units against a threshold of 40–50, opposed by 41.5 units of inhibition. Hand the working
correlator T4's real weights and it goes silent — so every early measurement came from a
cell firing on *injected current*, a pedestal carrying no information.

**We modelled the detector as non-spiking.** A non-spiking cell's output is a straight line
in its voltage, and a straight line can't do the comparison. Feed identical recorded signals
into one that spikes: **10.6× more** selectivity.

**The multiplication isn't there.** Telling left from right needs an AND — *was something
there a moment ago* AND *is something here now*. AND is multiplication; addition fires
whenever *either* happens. Neurons multiply one way: some inputs open a drain rather than
pushing, and with the drain open every tap counts for less. **Ours cancels itself.** Opening
the drain also lowers the level, and the lower the level the harder each tap pushes: ÷1.77
against ×1.85, net **1.06**. The cell had been *adding* the whole time.

**Where the fix is applied matters more than how big it is.** Make the drain much bigger and
the cancellation stops — but do it brain-wide and bitter stops suppressing feeding and
starts *driving* it, because **half the brakes in this brain are applied to other brakes**
(557,080 of 1,099,675 inhibitory connections land on inhibitory cells). Do the identical
thing to the visual system only and the feeding circuit comes out byte-identical, because
the tongue motor neuron receives 0.00% of its input from visual cells.

**The ceiling.** Sweeping every parameter we're allowed to touch, selectivity tops out near
**0.08** where a real fly needs ~0.5. Forcing inhibition to be purely divisive abolishes
what little there is — so the residue isn't division at all, it's the subtractive part plus
the firing threshold, faking an AND crudely.
</details>

### 3. Where you record matters more than any parameter

Steering was read from `DNa02`, of which this brain contains exactly one per side, with a
standing 20–45 Hz left-right imbalance (it is one real animal; its halves aren't mirror
images). The motion signal riding on that was ~0.5 Hz. Fly labs don't record there.

| where you listen | motion signal |
|---|---|
| `DNa02` (1 cell/side, **2.3%** visual input) | ~0.5 Hz |
| horizontal system (4 cells/side, pools thousands of detectors) | **17–26 Hz** |
| `DNp15`, the descending neuron the HS cells actually drive | **up to 32 Hz** |

Cut every input to the motion detectors and the HS signal collapses to ~1 Hz, so it is
genuinely motion. And across the whole output stage, **79.8% of the 1,305 descending
neurons never fire at all**.

### 4. Vision helps it play — narrowly, and only in a world it can see

The test: does the fly beat a random agent with *identical command statistics* — same speed,
same smoothness, differing only in whether the commands are about anything? Then break its
vision two different ways (mirror the eyes, freeze them) and see if the advantage goes.

**The current answer**, with every interface bug in section 6 fixed and the fly walking
forward, over 60 held-out levels:

| model minus matched random agent | fly arena: eyes working | mirrored | frozen | sky arena: eyes working |
|---|---|---|---|---|
| health collected | **+7.5** | −0.7 | +4.5 | −4.3 |
| time survived | **+34.0** | −13.1 | +22.6 | −8.2 |
| ground covered | +1.6 | +1.8 | **+3.2** | −0.2 |

Bold = the 95% interval excludes zero. In the **fly arena** — striped walls, textured ground,
a world whose structure the fly can resolve — it collects more health and survives longer
than chance, and mirroring its eyes removes that. In the **sky arena**, whose walls look flat
grey at the fly's resolution, it does no better than chance at all. So: vision helps, but
only when there is something to see, only on health and survival, and freezing the eyes only
weakens the effect rather than cleanly removing it.

Two more things it is honest to say. Head to head, no model-versus-model difference in health
or survival is statistically solid — the effect is in how it beats chance. And with correct
geometry the fly's vision **steers it toward walls**: it's stuck more than the mirrored fly
in both arenas. Real flies are drawn to large dark vertical shapes; in a Doom maze that's a
trap.

<details>
<summary><b>The earlier, stronger result — and why it was partly an artifact</b></summary>

Before the fixes, the same test looked cleaner: **+12.7** health (30 levels, original
arena) and **+8.7** (60 levels, sky arena), with mirroring and freezing both removing it.
That fly was walking backwards on 97% of tics, one eye saw the world reversed, and the turn
sign was flipped. Reversing with a flipped turn sign steers the *direction of travel* toward
the more active side, and it faced away from whatever wall it was pressed against — the
errors partly cancelled into something that looked like skilled play. Numbers:
[`paper/data/behav_wide/`](paper/data/behav_wide) (before) and
[`paper/data/behav_fixed/NOTE.txt`](paper/data/behav_fixed/NOTE.txt) (after).
</details>

What this is **not**: a fly reflex. Spin the world and it doesn't counter-turn; loom
something at it and it doesn't flinch.

### 5. The environment is part of the experiment

Doom is painted in browns and lit for human eyes. A fly's R1-6 photoreceptors are nearly
blind at Doom's red primary, so the scene is **repainted into blue-green** — which is
exactly what a fly-vision lab does when it builds an arena out of green emitters. The
gamma is undone first, because photoreceptors integrate linear light.

Three arena changes came later, and all are environment, not brain:

- **A daylight sky.** The stock arena has a dark stone ceiling, so the upper half of every
  eye looked at near-black. Now it's open sky: top-of-view brightness 10.8 → 196.
- **An arena built for a fly** (`health_gathering_fly`). Doom's wall detail sits near one
  map unit — far finer than the fly's 5.4° resolution — so its eye averaged the walls to
  flat grey; its floor was a dark flat with nothing to track. Now: bold vertical stripes at
  the spatial period the fly's motion detectors prefer (128 map units ≈ 18–37° at wall
  distance), bright textured ground, and flat daylight. Frame-to-frame change on the retina
  — the quantity a motion detector integrates — **nearly doubles** (×1.9 overall, ×3.5 in
  the ventral field, where a walking fly reads its ground flow).
- **The whole visual field.** Each eye sees 170° across and they point outward, together
  covering ~250°, but one Doom camera can't exceed ~170°. So two more Doom instances render
  90° left and right, loading the main game's save every tic — the *same* world, not a
  lookalike. At a 0° offset the side view matches the main game pixel for pixel.

### 6. Three interface bugs, found by looking at the picture

Late in the project we plotted each lens *where it actually looks*, instead of on the
lattice. The field came out a slanted parallelogram — and both eyes slanted the same way,
where a real right eye mirrors the left. Two bugs, both in the interface rather than the
brain:

- **The eye map was scrambled.** The lattice coordinates were read with the wrong
  convention (each eye a 51 × 17 sliver instead of 32 × 27), and neither eye was mirrored.
  We re-derived the orientation from the fly's own wiring: the four detector directions
  agree between T4 and T5 within 1–13°, "up" is confirmed by the dorsal-rim cells, and the
  result matches the published eye map. **One eye had been seeing the world backwards.**
- **Steering was reversed.** The decoder assumed Doom's turn command is positive for *left*;
  it is positive for *right* (+10 lowers the heading by 10° and slides the scene 28 px).
  So the fly turned *away* from the side whose steering neuron fired harder.

Together these are the worst possible pair for the optomotor reflex: with both eyes oriented
alike, spinning the world produces no left-right difference to steer by, while walking
straight produces a fake one — and then the steering is applied backwards.

Both fixes are **opt-in** (`eye_map="anatomical"`, `fixed_turn_sign=True`), so with the
defaults every earlier result reproduces bit for bit — **including everything above**. That
is the honest status of section 4: those numbers are real, and they were measured through an
interface with one eye reversed.

**Did fixing them produce the reflex? No** — and neither did the better arena. Tethered in a spinning drum — commands recorded
but not executed, exactly how this is measured in real flies — the fly's steering is flat at
both the command and the neuron level, and mirroring its eyes doesn't reverse anything. What
the fix *does* do is double the direction signal at the horizontal system (1.6 → 3.4 Hz),
which is still 3 Hz against a standing left-right imbalance of ~20 Hz. The limit is the one
in §2, not the interface: per-cell selectivity capped near 0.08 where a fly needs ~0.5. Full
numbers: [`paper/data/drum_anat/NOTE.txt`](paper/data/drum_anat/NOTE.txt).

In the fly arena the drum finally moves the steering neuron — about 1 Hz, three times the
old arena, against exactly zero with frozen eyes, so it is genuinely visual. But mirroring
the eyes gives the same sign and size, where a real reflex must reverse. The world was worth
fixing; the barrier is the detector, not the world
([`paper/data/drum_fly/NOTE.txt`](paper/data/drum_fly/NOTE.txt)).

**And it had been walking backwards.** Watching it get pinned into wall corners, we logged
what it commands: the walk key was negative on 97% of tics. Walking is read as BPN (forward)
minus MDN (the "moonwalker" neuron, reverse gear), and that was safe when it was designed,
because MDN was silent. Raising the optic gain to switch the visual system on — the change
behind section 4 — also drives MDN from **0 to 107 Hz**, so the fly reversed, blind, into
walls and stayed there. In a real fly MDN fires in bursts to back away from something, so
the fix reads it that way: only MDN's rise above its own running level counts as "back up"
(`phasic_mdn`). The antennae now only register contact when the fly pushes *forward* —
they're on its head.

| three levels | walks backward | stuck | tiles explored | health collected |
|---|---|---|---|---|
| before | 97% | 37–68% | 7–14 | 0–20 |
| after | **0%** | **24–41%** | **15–26** | **16–60** |

It still gets pinned sometimes. That part is the model: it turns at the same rate whether
stuck or free, wall contact barely moves MDN, and touch can't steer because both antennae
reach the same side of the brain. Re-running the behaviour tables with a forward-walking fly
changed the headline — section 4 now shows the corrected result.

---

## Running it

```bash
git clone https://github.com/mutkuoz/flydoom
cd flydoom
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[sim,doom,dev]"
./scripts/fetch_data.sh          # prints how to download the connectome
```

Needs Python 3.11–3.12 and a CUDA GPU with 8 GB or more. Doom itself is covered — ViZDoom
ships the free Freedoom, so no commercial game files are needed. The connectome is a free
download but requires signing in, and is **not** redistributed here.

**Watch it play, live:**

```bash
.venv/bin/python experiments/m5_closed_loop.py --live --tics 600
```

**Record a clip** (headless; writes mp4 + gif):

```bash
.venv/bin/python scripts/record_gameplay.py --seconds 30 \
    --scenario health_gathering_fly --wide --eye-map anatomical --fixed-turn \
    --phasic-mdn --spiking-t4 --optic-gain 16 --yaw-source DNp15 --touch
```

**Speed.** On an RTX 5070 Ti, roughly **0.06 s per Doom tic** with the full eye (three
renders per tic), so a 700-tic episode takes about a minute and the simulation runs at
about half of real time. It is dispatch-bound rather than compute-bound — the GPU sits
mostly idle on one episode — so the way to use the hardware is to run many episodes at
once.

**Run the behavioural battery** across cores, with resumable per-seed checkpoints:

```bash
.venv/bin/python experiments/run_sharded.py --scenarios health_gathering_sky \
    --seeds 30 --tics 700 --jobs 5 --wide --json paper/data/mine/run.json
```

### The experiments

Each prints pass or fail. `m0`–`m9` are the milestones; the rest are diagnostics, most of
them built to kill our own explanations.

| | Test | Result |
|---|---|---|
| M0 | Can we find the neurons we need? | ✅ |
| M1 | Does the wiring diagram load correctly? | ✅ 2,710,038 edges |
| M1.5 | Does the simulator match the textbook maths? | ✅ 0.03% vs closed form |
| M2 | **Sugar makes it stick its tongue out** | ✅ 77 Hz, 99% bitter suppression |
| M3 | Moving stripes make it turn | ⚠️ 2% of a real fly |
| M4 | An approaching object makes it flinch | ❌ looming = receding |
| M5 | It plays Doom without falling over | ✅ |
| M6 | It runs from an enemy, unprompted | ❌ |
| M7 | It can tell which side a target is on | ❌ `LC10a` is silent |
| M8 | **Smell changes what it does** | ✅ beats the shuffled control |
| M9 | Does it play better than a matched random agent? | ✅ with vision, ❌ without |
| M10–M14 | Odour tracking · touch · optomotor drum · gated odour · replay | mostly ❌, all with controls |
| M15–M16 | LIF validation · **where each eye points, from the wiring** | ✅ |

134 automated tests. `m3b`–`m3n` are the motion-vision post-mortem: arm modulation, phase
offset, fan-in, isolation, add-back, saturation, per-subtype geometry, morphology, real
traces, and the shunt cancellation.

### Repo map

```
flydoom/        the model: graph.py, lif.py, retina.py, doom.py, motor.py,
                olfaction.py, interocept.py (taste), mechanosensation.py
experiments/    m0–m16 plus the diagnostics; run_sharded.py for batches
scripts/        recording, arena building, analysis
paper/          the preprint (main.tex/pdf) and every result in paper/data/
tests/          134 tests
```

---

## What we're openly faking

- **No body.** The dataset is brain-only — the fly's spinal-cord equivalent is a separate
  animal in a separate dataset. Doom's movement code stands in for legs.
- **No gap junctions, no neuromodulation.** EM reconstruction captures chemical synapses;
  the giant fibre's fastest output is electrical and therefore invisible here. Dopamine,
  octopamine and serotonin are flattened to fast excitation.
- **One brain, and its halves aren't identical.** One steering neuron sits ~2× above its
  partner. Left uncorrected, the fly spins forever.
- **We told it enemies exist.** The odour channel gets enemy distances from the game.
- **The fly's eye is not a camera.** We match the lens count (~800 per eye), the ~5°
  spacing and blur, the field of view and the eye geometry. We don't have ultraviolet — a
  fly's sky is largely a UV signal — or polarised light, or colour, and Doom draws 35 frames
  a second where a fly resolves flicker several times faster.
- **Angular scale is calibrated, not measured.** Real ommatidial spacing is finest at the
  front; ours is linear across the eye.

The full list, with the measurement behind each one, is in the paper's Limitations section.

---

## Licensing and citation

Code is MIT. **The connectome data is CC BY-NC 4.0 — non-commercial only**, and is not
included here. If you use this, cite the people who made the data:

```bibtex
@article{dorkenwald2024,
  title   = {Neuronal wiring diagram of an adult brain},
  author  = {Dorkenwald, Sven and others},
  journal = {Nature}, volume = {634}, pages = {124--138}, year = {2024}
}
@article{schlegel2024,
  title   = {Whole-brain annotation and multi-connectome cell typing of Drosophila},
  author  = {Schlegel, Philipp and others},
  journal = {Nature}, volume = {634}, pages = {139--152}, year = {2024}
}
@article{shiu2024,
  title   = {A Drosophila computational brain model reveals sensorimotor processing},
  author  = {Shiu, Philip K. and others},
  journal = {Nature}, volume = {634}, pages = {210--219}, year = {2024}
}
@article{zhao2025,
  title   = {Eye structure shapes neuron function in Drosophila motion vision},
  author  = {Zhao, Arthur and others},
  journal = {Nature}, year = {2025}
}
```

Built on [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) (Kempka et al. 2016).

DOOM is a trademark of id Software / ZeniMax. This project is unaffiliated.
