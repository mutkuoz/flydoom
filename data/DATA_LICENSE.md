# Data license

The connectome data this project consumes is **not** distributed here and is
**not** covered by this repository's license.

## FlyWire FAFB (snapshot v783)

Licensed **CC BY-NC 4.0** — Attribution, **NonCommercial**.
<https://creativecommons.org/licenses/by-nc/4.0/>

Non-commercial use only. If your use is commercial, you need separate
permission from the data producers; this repository grants you nothing.

Download it yourself via `scripts/fetch_data.sh` (which documents the manual
steps — the download requires signing in and accepting terms in person).
`data/raw/*.csv` is gitignored precisely so the data never lands in version
control.

## Required citation

> Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain.
> *Nature* **634**, 124–138 (2024).

> Schlegel, P. et al. Whole-brain annotation and multi-connectome cell typing
> of *Drosophila*. *Nature* **634**, 139–152 (2024).

If you use the optic lobe typing (`visual_neuron_types.csv`), also cite:

> Matsliah, A. et al. Neuronal parts list and wiring diagram for a visual
> system. *Nature* **634**, 166–180 (2024).

The simulation parameters and the sugar->PER validation follow:

> Shiu, P.K. et al. A *Drosophila* computational brain model reveals
> sensorimotor processing. *Nature* **634**, 210–219 (2024).
