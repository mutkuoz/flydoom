# Preprint source

Build:

```bash
python gen_table.py     # tab_m8.tex  <- data/m8_*.json
python make_figs.py     # figs/*.pdf  <- data/m8_*.json
latexmk -pdf main.tex
```

`data/m8_*.json` are the serialized runs behind Table 4 and Figure 3, produced by

```bash
python ../experiments/m8_olfactory_valence.py --seeds 10 --tics 320 \
    --json paper/data/m8_intact.json
python ../experiments/m8_olfactory_valence.py --seeds 10 --tics 320 --shuffled \
    --json paper/data/m8_shuffled.json
```

The table and the figure are generated from the same JSON, so they cannot
disagree. The environment is not bit-reproducible between processes; re-running
reproduces every reported effect to within a few percent and reproduces all
sign counts.

No connectome data is stored here — only derived summary statistics.
