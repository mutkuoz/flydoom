# flydoom — technical reference

The detailed version. [README.md](README.md) is the plain-language one;
[RESULTS.md](RESULTS.md) is what the experiments actually measured. This
document covers **how the system is built** — the data, the neuron model, the
sensory front end, the motor readout, and every place we deliberately left the
reference model.

---

## 1. The data

FlyWire FAFB, snapshot **v783**, frozen since October 2023. Do not substitute a
different snapshot; every number here assumes it.

| File | What we take from it |
|---|---|
| `classification.csv` | `super_class`, `class`, `sub_class`, `side`. **No `cell_type` column** — that is the trap. |
| `consolidated_cell_types.csv` | `primary_type` — where cell types actually live |
| `visual_neuron_types.csv` | optic lobe typing, 95,079 neurons |
| `labels.csv` | free-text community annotations; the *only* place `BPN` and `aIPg` appear |
| `column_assignment.csv` | ommatidial column + hex coordinates for 45,528 optic lobe neurons |
| `connections_buhmann_no_threshold.csv` | the graph. Both pipelines carry `nt_type`; the *pre-thresholded* `connections_princeton.csv` has it 100% NULL and is unusable. |

### Measured magnitudes

```
neurons in classification.csv         139,255
edges, unthresholded              16,847,997
synapses, unthresholded           54,492,922   ← matches the published figure
edges at syn_count >= 5            2,710,038   ← what we simulate
synapses at that threshold        31,578,726
neurons with at least one edge       134,013   ← at the >=5 threshold
```

The spec's `54_000_000 < n_edges` assertion is wrong by ~20×: 54.5M is the
**synapse** total, not the edge count.

### Transmitter signs

Measured on Buhmann, the pipeline we actually simulate. Shares are of the
edges at `syn_count >= 5`, with the unthresholded share in brackets.

| Label | Sign | Share of simulated edges |
|---|---|---|
| `ACH` | excitatory | 58.2% (53.9%) |
| `GABA` | inhibitory | 23.6% (21.2%) |
| `GLUT` | **inhibitory** | 16.6% (19.5%) |
| `DA` / `SER` / `OCT` | excitatory (flattened) | 1.7% (5.4%) |

`GLUT` is inhibitory in flies via GluCl channels — the classic error when
porting vertebrate intuitions, and a sixth of edges rather than the ~10%
usually assumed. Asserted in the tests.

**Dale's law does not hold in the Buhmann file, and this is a trap.** Transmitter
is predicted per synapse and reported per neuropil, so one neuron can carry
different calls in different neuropils. In the simulated graph, **42,078 of
128,972** presynaptic neurons (32.6%) carry more than one `nt_type` and 26,781
(20.8%) carry both signs; a synapse-weighted majority vote per neuron would flip
**3.1% of edges** and 2.6% of synapses. We sign per edge and say so.

The Princeton release *has* had Dale's law applied upstream — 0 of its 139,003
presynaptic neurons carry more than one `nt_type` — so per-neuron and per-edge
signing coincide there. The two releases are not interchangeable on this point,
and transmitter statistics quoted from one do not describe the other.

### The histamine override

Photoreceptors release **histamine**, which is not one of FlyWire's six
predicted transmitters. Measured on `R1-6` output edges:

```
predicted:  ACH 71.2%   GLUT 20.2%   GABA 5.8%   (Buhmann, >=5 syn)
true sign:  inhibitory (HisCl1 chloride channels)
```

Left uncorrected, the excitatory ~72% of photoreceptor output is signed
backwards and the entire optic lobe reads a negative image. `graph.py` overrides all `R1-6`/`R7`/`R8`
output to inhibitory. 18,437 edges affected, 11,920 of them actually flipped.

---

## 2. The neuron model

### Base equations

From Shiu et al. 2024, Table 1 (published as a bitmap — recovered by fetching
the figure asset directly):

```
V_resting    = -52 mV        V_threshold = -45 mV
V_reset      = -52 mV        T_refractory = 2.2 ms
R_mbr        = 10 MΩ         C_mbr = 0.002 µF   →  T_mbr = 20 ms
Tau (synaptic) = 5 ms        T_dly = 1.8 ms
W_syn        = 0.275 mV      (their single free parameter)
```

Every value the project spec guessed was correct. It omitted `T_dly` entirely.

`dt = 0.5 ms`, giving **57 substeps per Doom tic** (35 tics/s). Integration is
exact for the linear system rather than forward Euler — Brian2, which the
reference used, does the same, and forward Euler was measured 6% low on a single
PSP.

### Conductance-based synapses (departure)

The reference makes inhibition **subtractive**. We made it conductance-based:

```
tau·dv/dt = (V_rest − v) + g_exc·(E_exc − v) + g_inh·(E_inh − v) + I_ext
g_tot = 1 + g_exc + g_inh
v_inf = (V_rest + g_exc·E_exc + g_inh·E_inh + I_ext) / g_tot
```

`E_EXC = 0 mV`, `E_INH = −70 mV`, `G_SYN = 0.00278` (refit on M2). The effective
time constant becomes `tau_mem / g_tot`, so a heavily driven cell is faster as
well as less sensitive.

Verified divisive: shunting cuts the input–output **slope** by >40% where
subtraction preserves it. Both cases are unit-tested.

### Graded units (departure)

Photoreceptors and most lamina/medulla neurons do not fire action potentials.
All 77,873 `super_class == "optic"` neurons are modelled as graded — output is
a rectified-linear function of membrane voltage in spike-equivalents, capped at
`GRADED_MAX_RATE = 200 Hz`. Lobula projection cells and everything central keep
spiking.

Two consequences that bit during development:

* spike-count readouts report **exactly 0.00 Hz** for graded cells — accumulate
  `net.out`, which is in spike-equivalents for both populations
* graded cells **saturate** under tonic bias, returning identical rates for every
  stimulus including blank. Graded needs `bias = 0`; only the all-spiking model
  needed bias.

### Per-edge conduction delays (departure)

A multi-tap ring buffer. Edges are sorted by `(delay, sign)` so each group is a
contiguous slice and the step does one scatter-add per group. Slow medulla lines
(`Mi9`, `Mi4`, `CT1`, `Tm9`) get a longer delay than the default `T_dly`.

### Performance

```
2.7M-edge scatter-add        ~300 µs/step on an RTX 5070 Ti
57 substeps per Doom tic     ~17 ms → 1.01× realtime
graph resident               31 MB
build time                   2.1 s
```

Timing is flat across spike rates — the dense scatter-add touches every edge
each step. Sorting by `pre_idx` and touching only active ranges would be a large
win if it ever mattered; at 1.01× realtime it does not.

---

## 3. Retina and Doom optics

### The lattice

`column_assignment.csv` ships the real retinotopy: **796 ommatidial columns per
eye** with hexagonal axial coordinates. Nearest-neighbour distance is exactly
1.000 axial units, confirming the hex conversion.

**Angular scale is calibrated, not measured.** One eye spans 46.5 × 29.4 lattice
units — aspect 1.58 — while the real eye covers roughly 170° × 150°, aspect 1.13.
The lattice axes are not isotropic in visual angle, so each axis is scaled
independently to land on the published FOV. Any absolute angular claim inherits
this.

### Injection site

Default is `L1 + L2 + L3`, all three lamina monopolars.

`R1-6` exists and is well wired (8,456 neurons, 95% carrying output, 1.01M
output synapses, textbook targets) but has **no column assignment**. Deriving it
from connectivity recovers only 4,147 cells and is badly asymmetric — 451 of 785
left columns against 749 of 796 right, an artefact of uneven optic lobe
proofreading. Since M3 measures a *left–right difference*, that asymmetry sits
directly in the quantity under test.

Driving `L1` alone starves the slow arm of the motion correlator: `Mi1 ← L1` but
`Mi9`/`Tm9 ← L3`. Measured, `Mi9` modulation was 0.009 mV against `Mi1`'s
4.58 mV. Driving all three raised it **139×**.

### Doom viewport

```
horizontal FOV        130°   (widest that renders sanely; +fov at init)
vertical FOV          ~116°  (follows from aspect ratio)
columns inside          ~40%
outside the viewport    filled with the frame's MEAN luminance
eye splay             ±40°
```

Mean-fill rather than black is load-bearing: a dark surround is a permanent
high-contrast edge at a fixed retinotopic position, which the looming detectors
report as an object sitting there forever.

Eye splay is equally load-bearing. Each eye's azimuth is centred on its own gaze
direction, so mapping both to screen centre makes the two eyes see identical
images — and the `DNa02` left-minus-right steering signal becomes zero *by
construction*.

Sampling uses a **Gaussian acceptance function**, implemented as a separable
pre-blur matched to the acceptance angle followed by bilinear sampling at each
column's gaze direction. Mathematically equivalent to convolving 1,581 kernels,
vastly cheaper.

### Adaptation

Weber contrast: a running per-column mean is subtracted, so the retina codes
`(I − ⟨I⟩)/⟨I⟩` rather than absolute intensity. This gives automatic gain
control, which Doom needs.

**`L1`/`L2` are transient; `L3` is sustained.** Measured, Doom's scene changes by
0.0003 in mean brightness frame-to-frame from a stationary agent — it is
temporally static. A fully adapting retina washes out within a second, so the
agent never moves, so nothing changes. Keeping `L3` unadapted lets the loop
start.

---

## 4. Readout

~1,300 descending neurons carry everything the brain sends the body. A handful
are individually characterised, so the mapping is by hand:

| Doom | Neuron | Role in a real fly |
|---|---|---|
| turn | `DNa02` | Steering. One per hemisphere; L−R difference is yaw. |
| forward | `BPN` | Bolt protocerebral neuron. Fast walking. |
| backward | `MDN` | Moonwalker descending neuron. |
| dodge | `DNp01` | The giant fiber. Escape. |
| `USE` | proboscis motor pool | v783 types this as a 24-cell pool, not `MN9` individually. |
| `ATTACK` | `pC1` / `aIPg` | Aggression state gate. |
| health pickup | `Gr64f` (sugar GRNs) | A health pack tastes sweet. |
| damage | `Gr66a` (bitter GRNs) | |

`P1` cannot be used: **FAFB is a female brain** and P1 is male-specific. `pC1`
and `aIPg` are the female homologues. This matters more than a name swap — see
§6.

### Decoding

Steering uses ViZDoom's **delta buttons**, not binary presses: the L−R
difference is graded and the fly's real yaw command is proportional to it.
Binary buttons get a Schmitt trigger (two thresholds) so a rate hovering at the
line cannot rattle the key at 35 Hz.

Rate estimation is an exponential filter over `net.out`. **`out` is `rate × dt`,
so recovering Hz means dividing by `dt`, not by `tau`** — that error scaled every
reported rate by 160×, making a 141 Hz descending neuron read as 0.9 Hz and
leaving the agent permanently inside its deadzone. Every milestone before the
closed loop still passed, because none used the decoder.

Two further corrections the closed loop required:

* **Warmup.** The rate filter needs several `tau` to converge; seeding the yaw
  baseline before it has produces a 12°/tic lurch lasting about a second.
  12 tics (4.3 τ) removes it.
* **Baseline adaptation.** `DNa02_R` sits ~2× above `DNa02_L` permanently — the
  two halves of one real brain are not mirror images. A fixed differential is a
  fixed turn command, so the first working version spun forever. A slow baseline
  (τ = 3 s) removes the DC and keeps transients; it is the motor counterpart of
  the retina's luminance adaptation.

---

## 5. Olfaction

Doom renders light only. `olfaction.py` stands in for a modality the engine
lacks, feeding two real labelled lines:

| Channel | Carries | Means |
|---|---|---|
| `ORN_DA1` | cVA, a fly pheromone | "another fly is here" |
| `ORN_DM1` | vinegar | food — the strongest attractant a fly has |

Built deliberately **weak**, so it cannot substitute for vision:

* **no azimuth** — antennae are ~0.3 mm apart and cannot triangulate; left and
  right receive bit-identical drive. This is the load-bearing test.
* saturating, slow (τ = 0.2 s transport), patchy (3 Hz bursts, 60% duty)
* plume persistence (τ = 1.5 s) so a source stepping behind a pillar still
  smells

Concentration is `Σ 1 / (1 + (r/r_half)²)` with `r_half = 280` map units,
grounded in the measured 48–764 unit range of enemy distances.

ViZDoom's label buffer contains **only on-screen objects** — 421 sightings
tested, none beyond 59° of a 130° viewport — so occlusion is handled by the
renderer and nothing is smelled through a wall.

Off by default: source distances come from the label buffer, so enabling it
tells the agent enemies exist. M6 and M7 are only valid with it off.

---

## 6. Where the visual system breaks

The full argument is in [RESULTS.md](RESULTS.md). The structural summary:

**Direction selectivity and looming selectivity are both computations over
temporal order.** Inputs sum linearly into `g`, and the only nonlinearity is a
threshold applied to that sum — but `threshold(A + B)` is symmetric in A and B,
so "A then B" is indistinguishable from "B then A."

What is verifiably present:

* correlator geometry — `T4a` takes fast excitation centred (`Mi1` +30.6,
  `Tm3` +7.0) and slow inhibition on opposite flanks (`Mi9` −7.4 at
  (−0.76, +0.26), `Mi4` −3.4 at (+0.39, −0.60)); four subtypes, four mirrored axes
* the retinotopic travelling wave — `L1` phase advances at −20.0 deg/deg against
  an expected 20.0, and survives to `T4a`

Six candidate causes tested; only graded units produced any measurable effect,
and a 72-point sweep of the operating point shows it is not selectivity: best
|DSI| = 0.0121 (~40× too weak) and the mirror pairs oppose at 3 of 45
unsaturated points, below the ~25% two coin flips would give. The remaining obstacle is that the fast arm is
starved: weighting `T4a`'s inputs by measured firing rate gives excitation 189
against inhibition 558, a 3.0× imbalance in drive, which shows up as measured
conductances of `g_e` 0.0018 against `g_i` 0.0667 — a ratio of 37×. `Mi1` sits at
2–9 Hz because `L1` (−77.3, glutamatergic) suppresses it. The ON pathway is
disinhibitory and never rebalances; the OFF pathway, which does not depend on the
trick, conducts fine (`T5a` 58 Hz against `T4a` 2 Hz).

### Fixation, and why the female brain matters

`LC10a` is **silent** — 0.00 Hz, E:I 0.54, and only +44/−82 input per cell
against `LC11`'s +477/−456 (46.9 Hz) and `LC4`'s +439/−259 (95.4 Hz). It is held
down by `AOTU042`/`TuTuAa`, which are central rather than retinotopic.

`LC10a`'s state-dependent gain boost is a **male courtship** mechanism, gated by
`P1`. In this female brain that circuit does not exist, and there is no state
signal to substitute: `AOTU042 → LC10a` is **−18** per cell, but **`pC1` and
`AOTU042` are not connected at all** — zero edges either way at any threshold —
so the aggression state cannot reach that suppression to lift it.

Neither `LC11` nor `LC4` carries azimuth either (r = +0.034, +0.070). No aim
signal exists anywhere, so an ATTACK gate would fire blind.

---

## 6b. Input-path bugs found by measurement

Three of these were silently wrong for the whole project and all three corrupt
the visual pathway specifically. None touches the connectome or the neuron
model.

**The field of view was never being set.** `add_game_args("+fov N")` is
silently ignored by ViZDoom — renders at `+fov 90`, `+fov 130` and `+fov 160`
are bit-identical. The game ran at Doom's default **90 deg** while every module
here assumed 130, so every angular claim downstream inherited the wrong number.
`send_game_command("fov N")` does work, and has to be re-sent after each
`new_episode()` because the CVar resets.

**Angle was mapped linearly onto the viewport.** Doom draws a planar
perspective projection, so a ray at azimuth th lands at screen
`tan(th)/tan(FOV/2)`, not at `th/(FOV/2)`. Measured misplacement:

| gaze azimuth | linear | correct | error |
|---|---|---|---|
| 10 deg | 0.154 | 0.082 | **+87%** |
| 30 deg | 0.462 | 0.269 | +71% |
| 50 deg | 0.769 | 0.556 | +38% |
| 65 deg | 1.000 | 1.000 | 0% |

Worst near the centre of gaze, converging only at the edge. Retinotopy is the
substrate every motion computation runs on, and a warp this size makes the
angular spacing between neighbouring columns vary about twofold across the
field — so rigid motion sweeps the retina at a speed that depends on where you
look, which is exactly what a correlator tuned to a *fixed* spacing cannot use.
The acceptance kernel inherited the same error: scaling by `fov/width` rather
than the true centre resolution made it ~1.9x too wide.

**Frames were held across all 57 substeps.** That gives the retina one step
change and 56 substeps of exactly zero temporal derivative, while the delayed
correlator arm is 80 ms — three frames. A correlator was being asked to work on
an impulse train aliased against its own delay line. Luminance now ramps
linearly between consecutive frames across the substeps. A fly in a real room
receives continuous motion; the 35 Hz strobe is an artifact of the engine.

This also fixed the adaptation timescale: `adapt_decay = exp(-dt/TAU_ADAPT)` is
a **per-substep** decay, but `drive()` was called once per tic, so Weber
adaptation advanced one 0.5 ms step per 28.6 ms of game time — an effective tau
of **14 s** instead of 0.25 s.

### What fixing them did not fix

`LPLC2` still reads 0.00 Hz. It is not under-driven; it sits at -52.2 mV,
*below* rest and 7.2 mV from threshold, because `PVLP011` (-45.6 syn/cell)
fires at **371 Hz** — 82% of the model's 455 Hz refractory ceiling — and buries
the excitatory arm (`Tm5f`, +44.7 syn/cell at 36 Hz). Inhibition exceeds
excitation ~10x. This is the `T4a` 37x arm imbalance again, in a second circuit.

Every input-side scale was swept — photoreceptor drive, graded rate ceiling,
Weber contrast gain — and `LPLC2` never leaves 0.00 Hz:

| contrast gain | graded cap | LPLC2 | LC4 | PVLP011 | cells >200 Hz | walk cmd |
|---|---|---|---|---|---|---|
| 2.5 | 200 | 0.0 | 52.7 | 305 | 198 | 22.6 |
| 1.2 | 200 | 0.0 | 53.8 | 305 | 200 | 22.2 |
| 0.6 | 200 | 0.0 | 54.4 | 307 | 205 | 21.7 |
| 2.5 | 120 | 0.0 | 3.8 | 219 | 103 | 20.7 |
| 2.5 | 60 | 0.0 | 0.0 | 113 | 20 | 5.9 |

Expected on inspection, and worth stating as the lesson: **an input scale
multiplies both arms of a ratio and cancels.** No common gain fixes a 10:1
imbalance. Only a gain acting *differentially* on the two arms can — a per-type
gain, which is the quantity the connectome does not specify.

Tonic optic-lobe bias does release `LPLC2`, and shows the same trade rather
than escaping it:

| bias (mV) | LPLC2 | LC4 | LC10a | BPN-MDN (walk) | cells >200 Hz |
|---|---|---|---|---|---|
| 0.0 | 0.0 | 51.5 | 0.0 | +22.3 | 197 |
| 1.0 | 0.2 | 97.6 | 0.0 | +25.5 | 295 |
| 2.0 | 4.0 | 131.9 | 0.1 | **+0.3** | 492 |
| 3.0 | 18.9 | 151.5 | 0.5 | -22.7 | 647 |
| 6.0 | 64.7 | 179.2 | 4.5 | -22.8 | 991 |

There is no setting where the looming pathway conducts and locomotion
survives — past 2 mV the model walks backwards.

### The blank-field control, and why bias readings need one

Raising the tonic optic-lobe bias does make `LPLC2` fire. It does not make it
see. Measured across 8 runs (2 injection sites x 4 bias levels):

| site | bias | looming | receding | static | **blank** | stimulus adds |
|---|---|---|---|---|---|---|
| R1-6 | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 | silent |
| R1-6 | 4.0 | 18.37 | 18.41 | 18.80 | **18.30** | +0.4% |
| R1-6 | 7.5 | 76.22 | 76.24 | 76.34 | **76.17** | +0.07% |
| lamina | 4.0 | 30.62 | 30.54 | 34.89 | **30.52** | +0.3% |
| lamina | 7.5 | 76.35 | 76.46 | 76.61 | **76.35** | **0.0%** |

`LPLC2` fires 76 Hz at an **empty screen**. The bias that releases the cell is
what sets its output, and the stimulus-attributable fraction *falls* as bias
rises. Two regimes exist and neither is looming detection: silent, or firing at
a rate the picture barely perturbs.

Without the blank column, "LPLC2 fires 76 Hz to a looming disc" reads as
success. Always run the blank.

Looming and receding differ by less than the blank offset in every condition,
and the *sign* of the difference flips between them. The only regime with real
stimulus structure is zero-bias lamina, where `LC4` gives static 77.2 against
blank 24.9 — a genuine 3x response, to dark **area**.

### Photoreceptor injection: tested, changes nothing

The default injects at the lamina monopolars, one synapse past the
photoreceptors, which skips neural superposition (six photoreceptors sharing a
visual direction converging on one cartridge) and the histaminergic first
synapse. Injecting at `R1-6` restores both.

It changes nothing. At matched bias the two sites agree to within the
blank-field offset. At zero bias `R1-6` is silent throughout — the
inhibition-dominated optic lobe has no baseline to disinhibit from.

`R1-6` injection also carries its own bias: uneven FAFB proofreading recovers a
column for **451 of 785 left** columns against **749 of 796 right**, so it is
unusable for any left-right measurement. That is why the default is the lamina.

### Doom's own render settings

`vid_gamma`, `vid_contrast`, `vid_brightness`, `vid_saturation` are all **inert**
— ViZDoom exposes the render buffer upstream of display post-processing, so
renders are bit-identical at every value. `r_visibility` does take effect but
shifts frame mean brightness (46.9 to 56.1) while changing the
distance-to-brightness relation by only 3-6% over a 40-tic approach.

Another common scale, another cancellation.

---

## 7. Every deliberate departure

| Change | Reason |
|---|---|
| Histamine override | 83% of photoreceptor edges mispredicted excitatory |
| Conductance synapses | Subtraction cannot express the multiplication a correlator needs |
| Per-type conduction delays | A single global `T_dly` cannot express a correlator either |
| Graded optic lobe | Photoreceptors and lamina cells do not spike |
| Weber contrast adaptation | Doom brightness varies enormously; also gain control |
| `L3` kept sustained | Without it a static scene adapts to nothing |
| `W_syn` refit to 0.165 mV | Paper's 0.275 mV was fitted against v630 and a different synapse pipeline |
| `G_syn` fitted to 0.00278 | The conductance-mode analogue, fitted by the same protocol |
| Olfaction added | Doom renders no smells; the fly has a nose |

Each is switchable, and each carries its measured justification in
`config.py`.

---

## 8. Known limitations

* **No ventral nerve cord.** FAFB is brain-only, so Doom's movement code stands
  in for legs. All behavioural *timescales* are Doom's — claims are restricted to
  descending-neuron activity. ([BANC](https://blog.flywire.ai/2025/11/03/the-banc-brain-and-nerve-cord/)
  is the dataset that fixes this.)
* **No gap junctions.** EM captures chemical synapses; the giant fiber's fastest
  output is electrical and therefore invisible.
* **No neuromodulation.** Dopamine, octopamine and serotonin are flattened to
  fast excitation.
* **n = 1 brain**, and its halves are not mirror images.
* **M2 is partly true by construction.** `W_syn` was fitted on the PER result by
  the original authors and by us. What it genuinely tests is that *some* single
  scalar reproduces the behaviour — a graph with flipped signs has none — plus
  the controls nobody fitted for.
* **A shuffle control is invalid in closed loop.** Measured: the intact agent
  saw an enemy on 186 of 500 tics and finished at full health; the shuffled one
  saw one on 489 of 492 and finished at 16. Different behaviour generates a
  different stimulus distribution. Controls must hold the stimulus fixed.
