STRUCTURED PARAMETER SEARCH -- M3k projected DSI under the M2/M8 constraints
===========================================================================
Everything here is resumable. Re-running any driver skips work already in its
checkpoint, so a killed run costs only the points in flight.

FILES
  checkpoint.jsonl    one line per evaluated M3k point, appended + fsynced the
                      instant it lands. 137 points. Key encodes every knob and
                      the stimulus, so it is the resume index too.
  constraints.jsonl   one line per M2 / M8 hard-constraint check, same contract.
  json/<key>.json     the raw m3k --json record for each point (the *_d1 ones
                      also carry per-cell rates, for matched-cell controls).
  clogs/, behave/     constraint logs and per-seed behaviour shards.

DRIVERS  (run from the repo root with .venv/bin/python)
  psearch.py     --points points_*.json --workers N --omp M   -> checkpoint.jsonl
  constraints.py --points cpoints_*.json --workers N           -> constraints.jsonl
  behave.py      --params bparams.json --seed-start 40 --seeds 12 --jobs 6
  report.py      [tag ...]     tabulate checkpoint.jsonl
  curvature.py                 per-axis 20%-width, 3-stimulus mean, controls
  matched.py A.json B.json     matched-cell control between two --dump-cells runs
  banalyse.py                  merge behaviour shards, paired arm comparison
  check_reach.py               structural: what each knob touches, and whether
                               a regional change can reach MN9 (it cannot: 0
                               edges, 0.000000% of MN9 input weight)

CODE CHANGES THIS SEARCH NEEDED (in the worktree, not committed)
  flydoom/config.py  FLYDOOM_GSYN_<NT> env override for G_SYN_RATIO entries.
                     Defaults leave the published values byte-identical.
  flydoom/graph.py   FLYDOOM_NONOPTIC_INH_SCALE: the optic scalar's complement,
                     inhibition onto everything NOT visual. Specificity control.

RESULT IN ONE LINE
  No feasible parameter set beats FLYDOOM_OPTIC_INH_SCALE=2. It already sits
  mid-plateau (20% band 1.67-2.81 on a three-stimulus mean) and it is the best
  point in the whole feasible space measured in Hz rather than as a ratio. The
  higher-scoring alternatives all use FLYDOOM_NT_CONDUCTANCE, which fails M8
  (DNp01 goes silent) even at the published conductances.
