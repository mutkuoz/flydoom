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

110 unit tests.

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
that produced sign-correct selectivity:

```
T4a +0.00368  vs  T4b −0.00756    OPPOSITE   (ON pathway)
T5a +0.00121  vs  T5b −0.00327    OPPOSITE   (OFF pathway)
```

Both mirror pairs oppose, deterministically across RNG seeds. But best
|DSI| = 0.017 against a real fly's 0.5–0.9 — **30–65× too weak**.

Best remaining explanation: the fast arm is starved. Weighting T4a's inputs by
measured firing rate rather than synapse count gives excitation 189 against
inhibition 558, and the measured conductance ratio is 37×. Mi1 fires at 2–9 Hz
because L1 (−77.3, glutamatergic) suppresses it. The ON pathway is
disinhibitory and never rebalances; the OFF pathway, which does not depend on
it, conducts fine (T5a 58 Hz against T4a 2 Hz).

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
