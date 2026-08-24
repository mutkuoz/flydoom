# flydoom — what was built, what works, what doesn't

A frozen FlyWire FAFB v783 connectome, simulated as a spiking network, wired to
ViZDoom. Nothing is trained. This document is the honest state of it.

---

## Running it

```bash
./scripts/fetch_data.sh                 # manual download steps (needs a login)
.venv/bin/python experiments/m0_resolve.py -v    # do the named cells exist?
.venv/bin/python experiments/m1_graph.py --save  # build the graph
.venv/bin/python experiments/m15_lif.py          # validate the integrator
.venv/bin/python experiments/m2_per.py           # sugar -> proboscis
.venv/bin/python experiments/m3_optomotor.py --graded --live
.venv/bin/python experiments/m4_looming.py --live
.venv/bin/python experiments/m5_closed_loop.py --live --tics 600   # play Doom
.venv/bin/python experiments/m6_enemy.py --live
.venv/bin/python experiments/m7_fixation.py            # is there an aim signal?
.venv/bin/python experiments/m8_olfactory_valence.py   # smell -> motor output
.venv/bin/python experiments/m8_olfactory_valence.py --shuffled   # its control
.venv/bin/python experiments/m9_behaviour.py           # does it DO anything?
.venv/bin/python scripts/record_gameplay.py            # media/flydoom.mp4 + .gif
```

`--live` opens a dashboard: both eyes as hexagonal ommatidial lattices, a bar
per monitored cell type, and the steering trace. M5 and M6 also open the Doom
window itself.

Runs at **1.01x realtime** on an RTX 5070 Ti: 139,255 neurons, 2.7M edges,
57 LIF substeps per Doom tic.

---

## Milestone status

| | | |
|---|---|---|
| M0 | resolver | **PASS** |
| M1 | graph | **PASS** — 2,710,038 edges, 31.6M synapses, 31 MB on GPU |
| M1.5 | LIF integrator | **PASS** — 0.03% against closed form |
| M2 | sugar → PER | **PASS** — 8/8 checks, 99% bitter suppression |
| M3 | optomotor | qualitative pass, **quantitative FAIL** |
| M4 | looming | **FAIL** — looming indistinguishable from receding |
| M5 | closed loop | **PASS** — stable, no chatter, no spin |
| M6 | enemy | **FAIL** — LPLC2 r = +0.000 vs true angular size |
| M7 | object fixation | **FAIL** — LC10a silent; no aim signal anywhere |
| M8 | olfactory valence | **PASS** — and it beats the shuffled control |
| M4b | looming, photoreceptor injection | **FAIL** — identical to lamina; bias releases LPLC2 but it then fires 76 Hz at a blank screen |
| M9 | closed-loop behaviour | **FAIL** — indistinguishable from a command-matched random walker; does not fire, does not gather |

130 unit tests.

---

## The headline result

**The taste pathway works. The visual motion pathway does not, and the reason
is a property of the model rather than of the connectome.**

M2 is the positive control and it passes cleanly: stimulating sugar gustatory
neurons at 100 Hz drives the proboscis motor pool to 77 Hz, bitter drives it to
zero, and sugar+bitter together is 99% suppressed. Nobody fitted the
suppression — it falls out of the wiring.

M3 and M4 fail together, for one shared reason. Direction selectivity (left vs
right) and looming selectivity (expanding vs contracting) are both computations
over the **temporal order** in which neighbouring ommatidial columns activate.
The model detects that neighbours changed; it cannot recover the order.

What is verifiably present and correct:

* the correlator geometry — T4a receives fast excitation centred (Mi1 +30.6,
  Tm3 +7.0) and slow inhibition on opposite flanks (Mi9 −7.4 at (−0.76,+0.26),
  Mi4 −3.4 at (+0.39,−0.60)); four subtypes, four mirrored axes
* the retinotopic travelling wave — L1's response phase advances at −20.0
  deg/deg against an expected 20.0 (360°/18° period), and survives to T4a

Six candidate causes were implemented and tested. Five produced nothing:
per-type conduction delays (1.8–240 ms), per-neuron membrane time constants
(5–200 ms), conductance-based shunting synapses, a forced high-conductance
state (`g_tot` = 105, 100× past the linear regime), and population pooling
(per-cell DSI vs per-cell input offset, r = +0.043 over 874 cells).

The sixth — **graded, non-spiking lamina and medulla** — is the only change
that produced a measurable effect at all, and it does not survive a sweep.

Graded units cap at 200 Hz, and a unit against its ceiling reports DSI ≈ 0
whatever its inputs do, so one configuration proves nothing. Sweeping bias,
spatial period and temporal frequency over 72 points and keeping the 45 below
75% saturation:

```
best |DSI| = 0.0121   (bias 1.0, period 30°, 1 Hz)
            vs a real fly's 0.5–0.9  —  ~40× too weak

both mirror pairs OPPOSITE at 3 of 45 unsaturated points (7%)
            two independent coin flips would give ~25%
```

A correlator's sign does not depend on its operating point. **An earlier
version of this file claimed deterministic mirror-pair opposition from a single
configuration; the sweep does not support it and it is withdrawn.** What is
being measured is a quantity fluctuating about zero.

Best remaining explanation: the fast arm is starved. Weighting T4a's inputs by
measured firing rate rather than synapse count gives excitation 189 against
inhibition 558 — a 3.0× imbalance in drive, which realises as conductances
`g_e` 0.0018 against `g_i` 0.0667, a ratio of 37×. Mi1 fires at 2–9 Hz
because L1 (−77.3, glutamatergic) suppresses it. The ON pathway is
disinhibitory and never rebalances; the OFF pathway, which does not depend on
it, conducts fine (T5a 58 Hz against T4a 2 Hz).

---

## The one thing that beat its control

Everything above is visual, and all of it failed. The exception came from a
different sense entirely.

Doom renders light and nothing else, but a real fly in a room with a large
animal in it genuinely smells that animal. Simulating only vision does not make
the model more honest — it makes it impoverished. So `olfaction.py` stands in
for a sensor the engine lacks, feeding two real labelled lines of the fly's
nose: `ORN_DA1` (a pheromone — "another fly is here") and `ORN_DM1` (vinegar —
food).

It is built deliberately **weak**, because the point is that it must not be
able to substitute for vision:

* **no direction** — a fly's antennae are 0.3 mm apart and cannot triangulate,
  so left and right receive bit-identical drive
* saturating, slow, and patchy, with a plume that lingers after its source goes
  out of sight

Tested open loop, agent held still so the visual scene evolves identically in
both arms, smell the only difference, three seeds:

```
population    smell off   smell on     change    shuffled control
LHN                0.32      14.38     +14.06       +0.01
DNp01            159.5      141.8      -17.67       -0.00
DNa02            199.9      211.6      +11.68       +0.11
LC4               96.9       96.9      -0.00        +0.07   <- control
```

Ten seeds. `LHN` and the walking pair are consistent in 10/10 (p = 0.002);
`DNp01` and `DNa02` in 9/10 (p = 0.021).

The lateral horn — the innate-valence pathway, dormant at 0.32 Hz — wakes to
14.4 Hz and propagates in **one hop** to the descending neurons. `DNp01`, the
escape neuron, is **suppressed**, which is the sign the wiring predicts:
`lateral horn -> DNp01` is −248 synapses, inhibitory. `LC4` moving −0.00 ± 0.02 Hz is
the control confirming both arms saw the same scene.

And the degree-preserving shuffle produces **nothing** — 0.01 against 14.06, a
1563-fold difference, with residuals that flip sign between seeds where the
intact effects are consistent. **This effect is attributable to the wiring.**

A sensory channel changes motor output through the frozen connectome, via the
anatomical route the biology names, with the sign the wiring predicts, and it
beats its control. Nothing was fitted.

Two caveats belong permanently attached. Odour source distances come from
Doom's label buffer, so we *told* the fly enemies exist — this is not evidence
the connectome detects them. And 1,227 lateral horn neurons pooled together
mixes many functional channels.

### Why this path worked when the aggression path did not

Routing the same odour at the `pC1` aggression neurons failed flat — 0.00 Hz
across a 10× sweep of drive. The reason is structural, and it is a lesson worth
keeping: **a path existing is not a path carrying.** `DA1` reaches `pC1` in
three hops touching 8 of its 10 cells, which looked convincing, but its
contribution is a rounding error against pC1's ±523/548 of balanced input — and
`pC1` is a bistable *latch* that must be pushed over, not nudged.

The lateral horn receives **21% of the odour relay's entire output**, has no
latch, and sits one hop from the output. Check weight, not hop count.

---

## The control

A degree-preserving shuffle, run open-loop in M4 so both arms see an identical
stimulus:

```
              LC4   LPLC2   DNp01
INTACT
  looming   36.06    0.00   139.0
  static    73.37    0.02   149.3
  receding  34.58    0.00   149.7
SHUFFLED
  looming    7.04    2.82    88.7
  static    45.15   14.41   207.0
  receding   6.19    2.50    84.0
```

Neither arm distinguishes looming from receding, and both show the same shape:
static ≫ looming ≈ receding, i.e. a response to dark **area**. **The intact
connectome does not outperform a shuffled one.** The measurable looming
response is not attributable to connectivity.

**A shuffle control is invalid in closed loop.** Measured: the intact agent saw
an enemy on 186 of 500 tics and finished at full health; the shuffled one saw
one on 489 of 492 and finished at 16. Different behaviour generates a different
stimulus distribution, so cross-arm correlations describe different worlds.
Controls must hold the stimulus fixed.

---

## Deliberate departures from Shiu et al.

Each is a change to the model, not a tuning knob, and each is switchable.

| Change | Why |
|---|---|
| Histamine override on photoreceptors | Histamine is not one of FlyWire's six predicted transmitters, so 83% of R1-6 output edges are mispredicted as excitatory when the true sign is inhibitory |
| Conductance-based synapses | Verified divisive: shunting cuts the input-output slope >40% where subtraction preserves it. M2 still passes; conduction to descending neurons went from nothing to DNa02 124 Hz |
| Per-type conduction delays | Table 1's single global `T_dly` cannot express a correlator |
| Graded non-spiking optic lobe | Photoreceptors and lamina monopolars do not fire action potentials |
| Weber contrast adaptation | Doom scene brightness varies enormously; also gives automatic gain control |
| L1/L2 transient vs L3 sustained | Doom is temporally static from a stationary agent (frame-to-frame mean luminance changes 0.0003); an all-adapting retina washes out and the loop cannot start |

Parameters were read off the paper's Table 1, which is served as a bitmap.
Every value the spec guessed was correct; the spec omitted `T_dly = 1.8 ms`
entirely. `W_syn` is the model's single free parameter and the paper fitted it
on the PER result, so we refit it the same way rather than copying: **0.165 mV**
against their 0.275 mV, because they used v630 and a different synapse pipeline.

---

## Known limitations

* **No VNC.** FAFB is brain-only, so Doom's movement code substitutes. All
  behavioural *timescales* are Doom's, not a fly's — claims are restricted to
  descending-neuron activity.
* **Angular scale is calibrated, not measured.** The ommatidial lattice is real
  (796 columns/eye with hex coordinates from `column_assignment.csv`) but its
  axes are not isotropic in visual angle, so the field of view is scaled to the
  published 170°×150°. Any absolute angular claim inherits that.
* **n = 1 brain**, and its two halves are not mirror images — DNa02_R sits ~2×
  above DNa02_L, which is a standing turn command until adapted out.
* **M2 is partly true by construction**, since `W_syn` was fitted on it. What
  it genuinely tests is that *some* single scalar gain reproduces the behaviour,
  plus the controls nobody fitted for.
