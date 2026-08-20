#!/usr/bin/env bash
# FlyWire FAFB v783 — manual download instructions.
#
# This script deliberately does NOT download anything. The FlyWire download
# requires a Google sign-in and in-person acceptance of the CC BY-NC terms.
# Automating around that would violate the terms; scraping Codex or hammering
# the API is likewise out of bounds.
#
# Run it to see the steps and to check what you already have.
set -euo pipefail

RAW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$RAW"

cat <<'INSTRUCTIONS'
FlyWire FAFB v783 — manual download
===================================

  1. Open  https://codex.flywire.ai/?dataset=fafb
  2. Sign in with Google and accept the CC BY-NC 4.0 terms
  3. Go to  Info -> Download Data
  4. Choose snapshot  v783
     ** NOT the live materialization — it drifts, and every number in this
        repo's acceptance tests is pinned to v783. **
  5. Download these files:

       classification.csv         required  root_id, super_class, class,
                                            sub_class, cell_type, side
       connections.csv            required  pre_root_id, post_root_id,
                                            neuropil, syn_count, nt_type
       visual_neuron_types.csv    required  optic lobe typing (LC/LPLC/T4/T5)
       cell_stats.csv             optional  sanity checks only

  6. Put them in  data/raw/

Expected v783 magnitudes (from Codex, 2026-08):

       neurons        139,255
       connections  3,732,460      <- rows in connections.csv
       synapses    ~54,500,000     <- sum of syn_count; NOT the row count

Then:  python experiments/m0_resolve.py
INSTRUCTIONS

echo
echo "Currently in $RAW:"
found=0
for f in classification.csv connections.csv visual_neuron_types.csv cell_stats.csv; do
  if [[ -f "$RAW/$f" ]]; then
    printf '  [x] %-28s %s\n' "$f" "$(du -h "$RAW/$f" | cut -f1)"
    found=$((found + 1))
  else
    printf '  [ ] %-28s missing\n' "$f"
  fi
done
echo
[[ $found -eq 0 ]] && echo "Nothing downloaded yet." || echo "$found file(s) present."
