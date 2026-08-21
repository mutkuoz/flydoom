![flydoom](flydoom.jpeg)

# flydoom

A fruit fly's brain plays Doom.

Not a metaphor. This loads the [FlyWire FAFB v783 connectome](https://codex.flywire.ai) —
the complete synapse-level wiring diagram of an adult *Drosophila melanogaster* brain,
139,255 neurons and 54.5 million synapses reconstructed from electron microscopy — runs
it as a spiking network with **weights taken directly from synapse counts**, feeds it
Doom frames through the fly's photoreceptors, and reads keypresses out of its descending
neurons.

Nothing is trained. There is no policy network, no reward, no gradient. The weights are
the fly's.

> **Status: built, and it produced a negative result.** All six milestones run. The
> taste pathway reproduces published fly behaviour; the visual motion pathway does not,
> and we can say exactly why. It plays Doom at realtime.
> **[RESULTS.md](RESULTS.md) is the writeup** — read that first.

---

## It is going to be bad at Doom

This is the expected result, and saying so up front saves everyone time.

The connectome is fixed. A frozen brain cannot learn a game it never evolved to play, and
the fly's learning centre (the mushroom body) is not in the loop here. What you get instead
is **fly behavior inside Doom**: optomotor wall-following, phototaxis, looming escape.
It will run away from a Cacodemon with real biological fidelity and then walk into a wall.

That is the interesting part. The question is not "can it clear E1M1." It is "what does a
real nervous system do when you drop it somewhere it has never been."

---

## The one thing that was supposed to make it work

Doom's threat model is already implemented in the fly, and we should get it for free.

The optic lobe has ~30 hardwired visual feature detectors. Two of them matter here:

- **`LC11` / `LC10a`** — small moving object detectors. A distant enemy is a small object.
  These drive **approach and fixation**.
- **`LPLC2` / `LC4`** — looming detectors. A close enemy is an expanding blob. These drive
  the giant fiber and **escape**.

Same demon. Different angular size. Opposite behavior.

So engage-vs-flee is not a heuristic anyone wrote — it should fall out of the wiring, and
the crossover angular size is a real biological parameter you could measure against the
looming-escape literature. That measurement was meant to be the output of this project.

**It does not work, and that turned out to be the more interesting result.**

The wiring is all there. `LC4` and `LPLC2` resolve cleanly, they project to the giant
fiber, and the upstream motion detectors have textbook geometry — fast input centred,
slow inputs on opposite flanks, four subtypes for four directions. What is missing is the
computation: this class of model cannot recover the *temporal order* in which neighbouring
ommatidia fire, and both "which way is it moving" and "is it growing or shrinking" are
order problems. `LPLC2` fires identically for an expanding disc and a contracting one.

Six candidate fixes were implemented and tested. Five did nothing. The sixth — modelling
the lamina and medulla as graded, non-spiking cells, which is what they actually are —
produced direction selectivity that is correctly signed and about **2% of a real fly's**.

And the control that stops you fooling yourself: a degree-preserving **shuffled**
connectome, shown the same stimulus, performs the same as the real one. Whatever response
survives is not coming from the wiring diagram.

Full detail, including every departure from the reference model and why: **[RESULTS.md](RESULTS.md)**.

---

## How it maps

Doom's action space is five buttons. The fly has ~1,300 descending neurons — the wires
leaving the brain for the body. A handful of them are individually named and
well-characterised, so the mapping is by hand, not learned:

| Doom | Neuron | What it does in a real fly |
|---|---|---|
| `MOVE_FORWARD` | `BPN` | Bolt protocerebral neuron. Sprinting. Named after Usain Bolt. |
| `MOVE_BACKWARD` | `MDN` | Moonwalker descending neuron. Reverse gear. |
| `TURN_LEFT/RIGHT` | `DNa02` | Steering. One per hemisphere — the L−R difference is yaw. |
| dodge | `DNp01` | The giant fiber. Biggest, fastest neuron in the fly. Panic button. |
| `USE` | proboscis motor output | The fly's tongue. Picks things up. |
| item pickup | `Gr64f` | Sugar taste cells. A health pack tastes sweet. |
| taking damage | `Gr66a` | Bitter taste cells. |
| `ATTACK` | `pC1`/`aIPg` + `LC10a` | Aggression state gates it, object fixation aims it. |

Health packs are sugar. Damage is bitter. This is not a joke — those are current
injections into real gustatory receptor neurons, and the fly's feeding circuit handles
item pickup because that is what it is for.

`P1` is crossed out above for a reason worth knowing: FAFB is a **female** brain and P1 is
male-specific, so it cannot resolve. `pC1`/`aIPg` are the female homologues that actually
exist in this dataset.

---

## Architecture

```
Doom frame (35 tics/s)
      ↓  hex-sampled onto ~800 ommatidial columns per eye
  photoreceptors  R1-6 (luminance) · R7/R8 (colour, mostly wasted — Doom is brown)
      ↓
  optic lobe      77,873 neurons — 56% of the brain. Retinotopic, ~4 layers deep,
      ↓           800 repeating columns = hardware weight sharing
  LC cells        ~30 feature detectors, each pooling the whole visual field
      ↓
  central brain   central complex (ring-attractor compass, path integration)
      ↓           mushroom body (present, not trained)
  descending      ~1,300 wires out
      ↓
Doom buttons
```

57 leaky integrate-and-fire substeps per Doom tic at `dt = 0.5 ms`. One step is a
2.7M-edge scatter-add (54.5M synapses collapse to 2.7M connections at the standard
≥5-synapse threshold) and takes ~300 µs on a consumer GPU. The whole graph is 31 MB of
VRAM — smaller than BERT-base by orders of magnitude.

**Measured: 1.01× realtime.** Compute was never the hard part here; biology is.

---

## Install

```bash
git clone https://github.com/mutkuoz/flydoom
cd flydoom
uv venv --python 3.12 .venv          # 3.13+ has no torch wheels yet
uv pip install --python .venv/bin/python -e ".[sim,doom,dev]"
```

Requires Python 3.11–3.12, PyTorch with CUDA, and a GPU with ≥8 GB. Developed on an
RTX 5070 Ti (12 GB); the graph itself only needs 31 MB, the rest is headroom.

ViZDoom ships Freedoom, so **no commercial WAD is required**.

### Getting the connectome

The connectome CSVs are **not** in this repo — see [licensing](#licensing).

```bash
./scripts/fetch_data.sh   # prints the manual download steps
```

Download from [codex.flywire.ai](https://codex.flywire.ai) → Info → Download Data,
dataset **FAFB**, snapshot **v783**. Do not use a different snapshot; v783 has been
frozen since October 2023 and every number in this repo assumes it.

---

## Milestones

Each is a script under `experiments/` that prints pass/fail. They gate each other.

```bash
.venv/bin/python experiments/m5_closed_loop.py --live --tics 600   # watch it play
```

`--live` opens a dashboard beside the Doom window: both eyes drawn as hexagonal
ommatidial lattices, a bar per monitored cell type, and the steering trace.

| | Test | Status | |
|---|---|---|---|
| M0 | Named cells resolve in v783 | ✅ | 50 handles, every required one found |
| M1 | Graph loads, counts assert clean | ✅ | 2,710,038 edges, 31 MB, 2.1 s |
| M1.5 | LIF integrator vs closed form | ✅ | 0.03% error |
| M2 | **Sugar → proboscis extension** | ✅ | 8/8 checks, 99% bitter suppression |
| M3 | Drifting grating → optomotor turning | ⚠️ | correct sign, 2% of biological strength |
| M4 | Looming disc → giant fiber spike | ❌ | looming ≡ receding |
| M5 | Closed loop, no chatter, no spin | ✅ | 1.01× realtime |
| M6 | Flees an enemy with no flee rule | ❌ | r = +0.00 vs true angular size |

110 unit tests.

**M2 is the one that matters, and it passes.** Injecting current into the sugar-sensing
neurons drives the proboscis motor pool to 77 Hz; bitter drives it to zero; the two
together are 99% suppressed. Nobody fitted that suppression — it falls out of the wiring.
It is the only test that validates neurotransmitter signs, synaptic gain and LIF dynamics
simultaneously, so it is also the positive control that proves the simulator is sound when
M3 and M4 fail.

One caveat stated plainly: `W_syn` is the model's single free parameter and the reference
paper fitted it *on this result*, so we refit it the same way rather than copying. That
makes the sugar arm partly true by construction. What M2 genuinely tests is that *some*
single scalar reproduces the behaviour at all — a graph with flipped signs has no such
scalar — plus the controls nobody fitted for.

---

## Known cheats

Stated plainly rather than buried:

- **No ventral nerve cord.** FAFB is brain-only. Real motor output happens in the VNC,
  which is a different animal in a different dataset. We stop at descending neurons and
  let Doom's kinematics stand in for legs. ([BANC v888](https://blog.flywire.ai/2025/11/03/the-banc-brain-and-nerve-cord/)
  fixes this and is the right dataset the moment this goes embodied.)
- **No gap junctions.** EM reconstruction captures chemical synapses. The giant fiber's
  fastest output is electrical, so the escape reflex is partly invisible to us.
- **No neuromodulation.** Dopamine and octopamine are treated as fast transmitters.
- **Field of view.** The fly sees nearly 360°; Doom is pushed to 130°, the widest that
  still renders sanely. About 40% of ommatidial columns fall inside the viewport; the rest
  are filled with the frame's **mean** luminance, not black — a dark surround would be a
  permanent high-contrast edge and the looming detectors would read it as an object.
- **Angular scale is calibrated, not measured.** The ommatidial lattice is real (796
  columns per eye, hex coordinates straight from `column_assignment.csv`) but its axes are
  not isotropic in visual angle, so the field of view is scaled to the published
  170° × 150°. Any absolute angular claim inherits that.
- **n = 1 brain, and its halves are not mirror images.** `DNa02_R` sits ~2× above
  `DNa02_L`, which is a standing turn command until it is adapted out.
- **`GLUT` is inhibitory.** In flies, glutamate is usually inhibitory via GluCl. This is
  the mistake everyone porting vertebrate intuitions makes, and it is asserted in the tests.
  It is 18% of all edges here, not the ~10% usually assumed.

---

## Deliberate departures from the reference model

Not cheats — considered changes, each with a measured reason and each switchable. The
reference is [Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07763-9), whose
parameters were read off its Table 1 (served as a bitmap, so this took some digging).

| Change | Why |
|---|---|
| **Histamine override** on photoreceptors | Histamine is not one of FlyWire's six predicted transmitters, so 83% of `R1-6` output edges are mispredicted as *excitatory* when the true sign is inhibitory. Left alone, the entire optic lobe reads a negative image. |
| **Conductance-based synapses** | Verified divisive: shunting cuts the input–output *slope* by >40% where subtraction preserves it. M2 still passes, and conduction to the descending neurons went from nothing to `DNa02` 124 Hz. |
| **Per-cell-type conduction delays** | Table 1's single global `T_dly` cannot express a correlator. (Incidentally, the spec omitted `T_dly = 1.8 ms` entirely — every *other* value it guessed was correct.) |
| **Graded, non-spiking optic lobe** | Photoreceptors and lamina monopolars do not fire action potentials. This is the only change that produced sign-correct direction selectivity. |
| **Weber contrast adaptation** | Doom scene brightness varies enormously; this also gives automatic gain control. |
| **L1/L2 transient vs L3 sustained** | Required, not cosmetic. Doom is temporally *static* from a stationary agent (frame-to-frame mean luminance moves by 0.0003), so an all-adapting retina washes out to nothing and the loop can never start. |

`W_syn` was refit rather than copied — **0.165 mV** against the paper's 0.275 mV, because
they used snapshot v630 and a different synapse-detection pipeline.

---

## Licensing

Code is MIT.

**The connectome data is CC BY-NC 4.0 — non-commercial only.** It is not vendored here and
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
