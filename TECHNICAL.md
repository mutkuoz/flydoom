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
neurons with at least one edge       138,533
```

The spec's `54_000_000 < n_edges` assertion is wrong by ~20×: 54.5M is the
**synapse** total, not the edge count.

### Transmitter signs

| Label | Sign | Share of edges |
|---|---|---|
| `ACH` | excitatory | 58.5% |
| `GABA` | inhibitory | 19.8% |
| `GLUT` | **inhibitory** | 18.1% |
| `DA` / `SER` / `OCT` | excitatory (flattened) | 3.1% |

`GLUT` is inhibitory in flies via GluCl channels — the classic error when
porting vertebrate intuitions, and 18% of edges rather than the ~10% usually
assumed. Asserted in the tests.

Dale's law already holds in this download: **0 of 139,003** presynaptic neurons
carry more than one `nt_type`, so per-edge and per-neuron signing are identical
and the majority-vote step is a no-op.

### The histamine override

Photoreceptors release **histamine**, which is not one of FlyWire's six
predicted transmitters. Measured on `R1-6` output edges:

```
predicted:  ACH 83%   GLUT 14%   GABA 3%
true sign:  inhibitory (HisCl1 chloride channels)
```

Left uncorrected, 83% of photoreceptor output is signed backwards and the entire
optic lobe reads a negative image. `graph.py` overrides all `R1-6`/`R7`/`R8`
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

Six candidate causes tested; only graded units produced sign-correct selectivity,
at ~2% of biological strength. The remaining obstacle is that the fast arm is
starved: weighting `T4a`'s inputs by measured firing rate gives excitation 189
against inhibition 558, a measured conductance ratio of 37×. `Mi1` sits at
2–9 Hz because `L1` (−77.3, glutamatergic) suppresses it. The ON pathway is
disinhibitory and never rebalances; the OFF pathway, which does not depend on the
trick, conducts fine (`T5a` 58 Hz against `T4a` 2 Hz).

### Fixation, and why the female brain matters

`LC10a` is **silent** — 0.00 Hz, E:I 0.54, and only +44/−82 input per cell
against `LC11`'s +471/−449 (46.9 Hz) and `LC4`'s +407/−227 (95.4 Hz). It is held
down by `AOTU042`/`TuTuAa`, which are central rather than retinotopic.

`LC10a`'s state-dependent gain boost is a **male courtship** mechanism, gated by
`P1`. In this female brain that circuit does not exist, and the only state signal
available makes things worse: `pC1 → AOTU042` is **+215**, and `AOTU042 → LC10a`
is **−18**. Opening the aggression gate suppresses the aim.

Neither `LC11` nor `LC4` carries azimuth either (r = +0.034, +0.070). No aim
signal exists anywhere, so an ATTACK gate would fire blind.

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
