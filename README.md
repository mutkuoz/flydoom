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

> **Status: early.** Nothing works yet. See [SPEC.md](SPEC.md) for the build plan and
> [milestones](#milestones) for where it actually is.

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

## The one thing that makes it work

Doom's threat model is already implemented in the fly, and we get it for free.

The optic lobe has ~30 hardwired visual feature detectors. Two of them matter here:

- **`LC11` / `LC10a`** — small moving object detectors. A distant enemy is a small object.
  These drive **approach and fixation**.
- **`LPLC2` / `LC4`** — looming detectors. A close enemy is an expanding blob. These drive
  the giant fiber and **escape**.

Same demon. Different angular size. Opposite behavior.

So engage-vs-flee is not a heuristic anyone wrote — it falls out of the wiring, and the
crossover angular size is a real biological parameter you can measure and compare against
the looming-escape literature. That measurement is the actual output of this project.

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
| `ATTACK` | `P1`/`aIPg` + `LC10a` | Aggression state gates it, object fixation aims it. |

Health packs are sugar. Damage is bitter. This is not a joke — those are current
injections into real gustatory receptor neurons, and the fly's feeding circuit handles
item pickup because that is what it is for.

---

## Architecture

```
Doom frame (35 tics/s)
      ↓  hex-sampled onto ~800 ommatidial columns per eye
  photoreceptors  R1-6 (luminance) · R7/R8 (colour, mostly wasted — Doom is brown)
      ↓
  optic lobe      ~60% of the entire brain. Retinotopic, ~4 layers deep,
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
54.5M-nonzero scatter-add — tens of microseconds on a GPU. The whole brain is smaller
than BERT-base. Compute is not the hard part here; biology is.

---

## Install

```bash
git clone https://github.com/mutkuoz/flydoom
cd flydoom
pip install -e .
```

Requires Python 3.11+, PyTorch with CUDA, and a GPU with ≥8 GB.

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

| | Test | Status |
|---|---|---|
| M1 | Graph loads, counts assert clean | ☐ |
| M2 | **Sugar → proboscis extension** | ☐ |
| M3 | Drifting grating → optomotor turning | ☐ |
| M4 | Looming disc → giant fiber spike | ☐ |
| M5 | Closed loop, empty room, no chatter | ☐ |
| M6 | Flees an enemy with no flee rule written | ☐ |

**M2 is the one that matters.** Injecting current into the sugar-sensing neurons should
make the proboscis extend — a real fly behavior, reproduced from
[Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07939-3). It is the only test
that validates neurotransmitter signs, synaptic gain, and LIF dynamics simultaneously. If
it fails, nothing downstream means anything.

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
- **Field of view.** The fly sees nearly 360°; Doom shows 90°. Most of the retina is dark.
- **`GLUT` is inhibitory.** In flies, glutamate is usually inhibitory via GluCl. This is
  the mistake everyone porting vertebrate intuitions makes, and it is asserted in the tests.

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
