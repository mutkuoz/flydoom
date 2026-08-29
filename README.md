![flydoom](flydoom.jpeg)

# flydoom - A fruit fly's brain plays Doom.

Not a metaphor. Scientists sliced a real fly's brain into seven thousand layers,
photographed every layer with an electron microscope, and traced all 139,255 neurons
through the stack. That wiring diagram is public. This project loads it, simulates every
neuron, shows it Doom through the fly's eyes, and reads keypresses out of the nerves that
would normally go to its legs.

**Nothing is trained.** There is no neural network being fitted, no reward, no learning of
any kind. The connections are the fly's, exactly as they were measured, and they never
change.

> 📘 **[TECHNICAL.md](TECHNICAL.md)** — the detailed version: neuron equations, cell types,
> every parameter and where it came from.
> 📊 **[RESULTS.md](RESULTS.md)** — what the experiments measured.
>
> This page is the plain-language one. No neuroscience assumed.

> **Status: built, and it produced a real result — just not the one we were aiming for.**
> Ten experiments run. The taste circuit works beautifully. The motion-vision circuit
> doesn't, and we can say precisely why. Then smell turned out to work, which nobody
> expected. It plays Doom at full speed.

---

## Watch it

![the connectome playing Doom](media/flydoom.gif)

Thirty seconds, real time, nothing sped up.

**Left** is the frame Doom draws. **Top right** is what the fly actually
receives — 796 hexagonal ommatidial columns per eye, sampled through the real
retinotopy, which is why it looks nothing like the screen. The arena is
repainted into blue-green because *Drosophila* R1-6 photoreceptors are nearly
blind at Doom's red primary, and Doom is painted in browns.

**Bottom right**, left to right: firing rates of the cells that would drive its
legs; the left-minus-right steering signal off `DNa02`; the two odour channels
(`ORN_DM1` vinegar = health pickups, `ORN_DA1` cVA = rivals); and a
heading-up map — the fly sits at the origin facing up, green dots are food it
can smell, red crosses are rivals.

What you are watching it do is **walk and turn**, and that is the whole list.
Read [the honest scorecard](#does-it-actually-do-anything) before reading
anything else into it. It does not aim, it never fires, and it does not seek
out health. It closes distance on enemies and on medkits at the same rate a
*random walker* does — it walks forward in a bounded map and bumps into things.

Full-quality clip: [`media/flydoom.mp4`](media/flydoom.mp4). Record your own:

```bash
.venv/bin/python scripts/record_gameplay.py --seconds 30
```

Runs headless and writes both files. `--scenario` picks the map; the default is
`deathmatch` because it has rooms and corridors to walk through. Avoid
`defend_the_center` — it is a bare circular room and the clip shows nothing.

---

## It is going to be bad at Doom

Saying so up front saves everyone time.

A frozen brain cannot learn a game it never evolved to play. What you get instead is **fly
behaviour inside Doom** — and the interesting question was never "can it finish the level."
It's *what does a real nervous system do when you drop it somewhere it has never been.*

---

## What a connectome is

A brain is neurons wired to each other. A **connectome** is the complete list of that
wiring: which neuron connects to which, and how strongly.

What you actually download is a spreadsheet. Each row says: *neuron A connects to neuron B
with N synapses, using chemical T.* A few million rows of that.

The crucial thing to understand is what this **doesn't** give you. It's a circuit diagram
printed with no component values — every wire is known, but nothing about voltages or
timing. All of that has to be supplied separately, and that gap is where essentially every
problem in this project lived.

A neuron itself is simpler than you'd think. It's a leaky bucket holding a voltage that
constantly drains back toward rest. Inputs nudge it up or down. If it climbs past a
threshold it fires a single all-or-nothing pulse to everything downstream, then resets.
That's it. We run 139,255 of those at once, stepping time forward in half-millisecond
ticks — 57 of them for every frame Doom draws, because a fly is faster than the game.

---

## The idea that was supposed to make this work

Doom's threat model is already built into the fly, and we should get it for free.

The fly's visual system has around 30 hardwired detectors, each firing for exactly one kind
of thing. Two of them matter here:

- **small-object detectors** — a distant enemy is a small moving object. These drive
  *approach*.
- **looming detectors** — a close enemy is an expanding blob filling your view. These drive
  the giant fibre, the fly's panic button, and cause *escape*.

Same demon. Different size on the retina. Opposite behaviour.

So engage-versus-flee shouldn't need any code from us. It should fall out of the wiring —
and the exact size at which the fly flips from "chase it" to "run away" is a real
biological number you could measure against published experiments. That measurement was
meant to be the whole point of the project.

### It doesn't work, and that turned out to be more interesting

**Seeing motion isn't about noticing that something changed. It's about noticing which
way** — and that information lives entirely in the *order* two neighbouring eye-lenses fire
in.

```
moving right:   A fires, then B fires
moving left:    B fires, then A fires
```

Both cases: A fired and B fired. Same events. The only difference is which came first.

Real brains solve this by making one input line slower than the other, then combining them
so that "what was there a moment ago" lines up with "what's here now" for one direction and
not the other. The connectome has all of it: the two input lines, the delay, the spacing,
four detector subtypes pointing four different ways, mirror pairs 180° apart. We measured
every piece. They're all there.

The detector still reads about **2% of a real fly's** direction selectivity.

**An earlier version of this README said that was arithmetic — that `A+B` equals `B+A`, so
a cell that sums and thresholds literally cannot recover order. That was wrong and we
withdrew it.** It describes a unit with no memory. Build the same correlator out of the
same simulated neurons, with one arm delayed, and it reaches a direction selectivity index
of **0.79** — and up to **1.0** once you add the real fan-in and spatial spread. The
machinery works fine. Something about how it's embedded doesn't.

**Two causes we did find**, each isolated by changing one thing and nothing else:

1. **The cell can't fire from its own inputs.** All of T4's excitation adds up to 46.8
   weight units; the threshold for producing *any* output is about 40–50, and it's opposed
   by 41.5 units of inhibition. Hand the toy correlator T4's real weights and it goes
   **silent**. So every measurement this project ever made came from a cell firing on
   *injected current* — a constant that carries no information about what the eye is
   seeing. The signal was a ripple on a meaningless pedestal.

2. **We modelled the detector as non-spiking.** Photoreceptors genuinely are, so it seemed
   safe to model the whole optic lobe that way. But a non-spiking cell's output is a
   straight line in its voltage, and a straight line can't do the comparison — feed the
   *identical* recorded signals into one that spikes and selectivity is **10.6× higher**
   (0.178 vs 0.017; the network reads 0.016–0.020).

Fix both — restore the threshold, drive the cell from its synapses instead of from injected
current — and cells above the experimental cutoff go from **2.8% to 12.3%**.

That still wasn't enough, and finding out why took twenty eliminated explanations. The
answer turned out to be arithmetic.

### The multiplication that wasn't there

To tell left from right you need an **AND**: *was something there a moment ago* **and** *is
something here now*. An AND is multiplication — if either half is zero the answer must be
zero. Addition won't do it; addition fires whenever *either* thing happens, which tells you
something moved but never which way.

Neurons mostly add. There's one way they can multiply: some inputs don't push the cell,
they **open a drain**. Picture the cell as a bathtub — normal inputs are taps, inhibition
is a drain, and with the drain open every tap counts for less. That's division. Divide by
one signal while adding another and you have your multiplication. That is the mechanism the
fly's detector is supposed to use.

**Ours cancels itself.** Opening the drain also lowers the water level — and the lower the
water, the harder the tap pushes. So the drain makes each drop count for less (**÷1.77**)
while making the tap push harder (**×1.85**). Net: **1.06**. A 6% effect where a large one
was needed, pointing the wrong way.

The cell had been *adding* the whole time. Not multiplying badly — not multiplying at all.
And addition cannot tell left from right no matter how perfect the wiring is. That is why
nineteen fixes did nothing: they were all repairing things that weren't broken.

### The fix, and why *where* matters more than *how much*

Make the drain much bigger than the taps and the cancellation stops. Do it **everywhere in
the brain** and motion improves — while bitter stops suppressing the fly's feeding reflex
and starts *driving* it, harder than sugar does. The reason is that **half the brakes in
this brain are applied to other brakes**: 557,080 of 1,099,675 inhibitory connections land
on inhibitory cells, so doubling every brake also doubles the braking *on the brakes*, they
release, and the sign flips.

Do exactly the same thing **to the visual system only** and the feeding circuit comes out
*byte-identical* — same numbers to the last digit, because the tongue motor neuron receives
0.00% of its input from visual cells. Smell survives too.

Same change. Same size. Opposite outcome. The only difference is **where**.

### And then it played better

With that one regional change, on thirty environment seeds it had never been tuned on, the
connectome beat a command-matched random agent on **10 of 18 measures** — the frozen model
beats it on **0 of 18**. It covers more ground, spends less time stuck, collects more
health, and **hits fewer walls**.

Two things we won't oversell. Freeze the retina on a single frame and about **half** the
collision advantage survives — so part of the win is a smoother motor command beating a
random one, not seeing. And the optimum is sharp, which is what over-tuning a single dial
looks like.

The honest headline is not *"a wiring diagram can't produce behaviour."* It's: **one volume
knob for the whole brain is not just imprecise, it's self-contradictory**. The eye needs a
setting the tongue cannot survive, and that is a measurement rather than an opinion.

### We tried it four ways

A wiring diagram tells you which neurons connect. It does **not** tell you how strong each
connection is. Something has to supply those numbers, and we compared four ways of doing it:

| how the strengths were set | applied | motion | plays better than random? | taste | smell |
|---|---|---|---|---|---|
| one number for everything | whole brain | baseline | **0 of 18** | ok | ok |
| real measured values from published biology | whole brain | 3.3× better | — | weaker | **broken** |
| turn all inhibition up 2× | whole brain | 5× better | — | **broken** | — |
| **same 2×, in the visual system only** | **eye only** | **6.7× better** | **10 of 18** | ok | ok |

**Row two is the one that surprised us.** Those are *real numbers*, measured in actual flies
by labs with no connection to this project. GABA synapses really are about 2.5× stronger
than acetylcholine ones. We plugged in the correct values and **smell broke** — an escape
neuron that normally idles got clamped to silence, so there was nothing left for an odour to
push down.

So it was never about finding the right numbers. **No single set of numbers works for the
whole brain, however correct.** You need the right values *and* the right geography, and a
connectome gives you neither.

That is the finding. It's also exactly what the models that *do* get fly vision working are
quietly supplying when they fit 604 separate parameters — we can now say precisely what that
buys and why it's needed.

---

## The thing that did work: smell

Doom draws pictures and nothing else. But a real fly standing in a room with a large animal
in it doesn't just *see* it — it **smells** it. Simulating only vision doesn't make the fly
more honest, it makes it half-blind in a way the real creature never is.

So we built the missing sense. A fly's nose is essentially **50 labelled wires**, each
meaning one specific thing — *another fly is here*, *food*, *danger*. We used two: enemies
go down the first, health packs down the "food" one.

The critical design choice is that the smell carries **no direction at all**. A fly's
antennae sit 0.3 mm apart, far too close to work out where a smell came from — which is why
real flies find things by zigzagging rather than walking straight at them. So both sides get
an identical signal. If smell could tell the fly *where* the enemy was, we'd have handed it
the answer to the exact problem the eyes are supposed to solve.

To test it fairly we **held the fly still** and ran the game twice with the same random
seed — same enemies, same movements, same everything on screen. The only difference was
whether the nose was switched on.

```
                       nose off    nose on     scrambled brain
"good or bad?" region      0.31      17.63          +0.01
escape neuron            160.02     138.23          +0.41
steering                 200.00     216.41          +0.01
a vision cell             96.85      96.94          +0.07   ← control
```

The vision cell barely moving is the check that makes the rest believable — both runs
genuinely saw the same thing.

The fly's hardwired "is this good or bad?" region was sitting dormant and **woke up**, then
reached the neurons that drive the body in a single step. The escape neuron got turned
**down** — and we'd already measured that this particular connection is a negative one, so
the wiring predicted that in advance.

**And the scrambled brain does nothing.** A thousandfold difference. This is the one result
in the project where the real wiring beats a shuffled copy of itself.

One caveat that belongs permanently attached: the smell strength is computed from how far
away enemies are, and the game tells us that. So this is **not** evidence the connectome can
detect enemies — we handed it that. What it shows is what the connectome does with a signal
once it has one.

---

## How it maps to the keyboard

Everything a fly's brain tells its body goes through about 1,300 nerves. Because that
bottleneck is so narrow, scientists have worked out what individual ones do — switch this
one on in a live fly and it turns; that one and it walks backwards. So we just watch about
eight of them.

| Doom | What we read | What it does in a real fly |
|---|---|---|
| turn left / right | `DNa02` | Steering. One per side — the difference between them is the turn. |
| walk forward | `BPN` | Fast walking. |
| walk backward | `MDN` | The "moonwalker" neuron. Reverse gear. |
| dodge | `DNp01` | The giant fibre — biggest, fastest cell in the fly. Panic button. |
| pick something up | proboscis nerves | The fly's tongue. |
| health pack | sugar taste cells | A medkit tastes sweet. |
| taking damage | bitter taste cells | Getting hurt tastes foul. |

Health packs being sugar isn't a joke. Those are genuine injections into the fly's real
taste neurons, and its feeding circuit handles item pickup because that is what a feeding
circuit is *for*.

---

## Running it

```bash
git clone https://github.com/mutkuoz/flydoom
cd flydoom
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[sim,doom,dev]"
./scripts/fetch_data.sh        # prints how to download the connectome
```

Then watch it play:

```bash
.venv/bin/python experiments/m5_closed_loop.py --live --tics 600
```

That opens Doom plus a live dashboard: both eyes drawn as hexagonal grids of lenses shaded
by what each one currently sees, a bar for each cell type, and the steering signal over
time. It does **not** run in real time: one second of game costs about
5 seconds of wall clock. Every Doom tic is 57 integration steps over 139,255
neurons and 2.7 million synapses, and each step costs ~1.4 ms. Spike delivery is a sparse
matrix-vector product; writing it as one (CSR) rather than as
gather-then-scatter made the whole simulator 4.2x faster, with zero spike
disagreement across 139,255 neurons. (An earlier version of this README claimed roughly real
time. That came from counting only the arithmetic and is wrong by 17×.)

Needs Python 3.11–3.12, a CUDA GPU with 8 GB or more, and a connectome download (free, but
requires signing in). Doom itself is covered — ViZDoom ships the free Freedoom, so no
commercial game files are needed.

### The experiments

Each one prints pass or fail.

| | Test | |
|---|---|---|
| M0 | Can we find the neurons we need? | ✅ |
| M1 | Does the wiring diagram load correctly? | ✅ |
| M1.5 | Does the simulator match the textbook maths? | ✅ |
| M2 | **Sugar makes it stick its tongue out** | ✅ |
| M3 | Moving stripes make it turn | ⚠️ 2% of a real fly |
| M4 | An approaching object makes it flinch | ❌ |
| M5 | It plays Doom without falling over | ✅ |
| M6 | It runs from an enemy, unprompted | ❌ |
| M7 | It can tell which side a target is on | ❌ |
| M8 | **Smell changes what it does** | ✅ beats the control |
| M9 | Does it actually play better than random? | ⚠️ frozen: 0/18 · regional fix: **10/18** |

124 automated tests.

**M9 is where the argument lands.** With 30 environment seeds per arm, the frozen model
beats a command-matched random agent on **none** of eighteen behavioural measures. With one
regional parameter changed — inhibition onto the visual system doubled, nothing else — it
beats it on **ten**, on seeds it was never tuned against: more tiles covered, less time
stuck, longer paths, more health, and fewer collisions.

Two caveats we keep attached. An earlier version of this table quoted friendlier numbers
from 5 seeds; the metric has a standard deviation of ~13, so 5 seeds could only resolve
differences of ~20, and two runs of the *same* configuration returned 3.2 and 11.2. Those
were noise and we withdrew them — everything here is n=30. And blinding the fly (freezing
the retina on one frame) leaves about half the collision advantage intact, so a real part
of the win is a smoother motor command rather than vision.

Beyond the milestones there are diagnostics, mostly built to kill our own explanations:
`m3b`–`m3f` (arm modulation, phase offset, fan-in, isolation, add-back), `m3i` (are the
arms saturated?), `m3j`/`m3t` (per-subtype detector geometry), `m3k` (does geometry predict
selectivity, per cell?), `m3l` (build a correlator that works, then add T4's properties one
at a time until it breaks), `m3m` (feed the *real* recorded signals into that correlator).
`run_sharded.py` runs any of them across cores with resumable per-seed checkpoints.

**M2 is the one that matters most.** Stimulate the sugar-tasting neurons and the muscle
that extends the fly's tongue fires at 77 Hz. Stimulate the bitter ones and it drops to
zero. Do both at once and it's 99% suppressed — and nobody programmed that suppression, it
falls out of the wiring. This is a real fly behaviour reproduced from a frozen wiring
diagram, and it's what proves the simulator works when the vision experiments fail.

---

## What we're openly faking

- **No body.** The dataset is brain-only — the fly's equivalent of a spinal cord is a
  separate animal in a separate dataset. Doom's movement code stands in for legs, which
  means any claim about *timing* is Doom's, not a fly's.
- **The fly sees far more than Doom shows.** We push Doom to its widest view, which still
  only covers about 40% of the fly's lenses. The rest are filled with average brightness
  rather than black — a black surround would look like a permanent object sitting there.
- **One brain, and its two halves aren't identical.** It's a real animal, not an idealised
  diagram, so one steering neuron sits about twice as active as its partner. Left
  uncorrected, the fly spins in circles forever.
- **We told it enemies exist.** The smell channel gets enemy distances from the game.

The full list, with the measurements behind each one, is in
[TECHNICAL.md](TECHNICAL.md).

---

## Licensing

Code is MIT.

**The connectome data is CC BY-NC 4.0 — non-commercial only.** It is not included here and
must be downloaded separately. If you use this, cite:

```bibtex
@article{dorkenwald2024,
  title  = {Neuronal wiring diagram of an adult brain},
  author = {Dorkenwald, Sven and others},
  journal = {Nature}, volume = {634}, pages = {124--138}, year = {2024}
}
@article{schlegel2024,
  title  = {Whole-brain annotation and multi-connectome cell typing of Drosophila},
  author = {Schlegel, Philipp and others},
  journal = {Nature}, volume = {634}, pages = {139--152}, year = {2024}
}
@article{shiu2024,
  title  = {A Drosophila computational brain model reveals sensorimotor processing},
  author = {Shiu, Philip K. and others},
  journal = {Nature}, volume = {634}, pages = {210--219}, year = {2024}
}
```

Built on [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) (Kempka et al. 2016).

DOOM is a trademark of id Software / ZeniMax. This project is unaffiliated.
