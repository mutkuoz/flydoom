Sweeps run to test whether the accessible parameter space contains a
configuration with substantially higher direction selectivity than the
reported one. It does not; the ceiling is |projected DSI| ~ 0.08.

All runs: m3k_axis_dsi.py --bias 0 --spiking-t4 --min-rate 20
          --period 15 --tf 2 (the search's tuning stimulus), CPU.

ei<X>_oi<Y>.json   inhibitory reversal potential X mV, regional inhibitory
                   scale Y. eidef = the published -70 mV.
nt1_oi<Y>.json     FLYDOOM_OPTIC_NT_CONDUCTANCE=1 (published per-receptor
                   ratios scoped to vision) with regional scale Y.
nt1_g<G>.json      the same at optic gain G, sweeping the operating point.
m8_armv.json       olfactory constraint for the regional per-receptor arm.

Findings:
  * E_inh peaks at the published value. Moving it toward V_rest, which makes
    inhibition purely divisive, destroys selectivity (-0.0716 -> +0.0023 at
    -52 mV). The residual selectivity therefore comes from hyperpolarisation
    acting against the spike threshold, not from shunting division.
  * Published per-receptor ratios applied regionally reach -0.0500 with no
    fitted parameter, against -0.0107 frozen, and pass both constraints.
  * Optic gain raises selectivity monotonically but shallowly, saturating
    near -0.06 at firing rates well above physiological.
  * Best over the whole sweep: -0.0792 (E_inh -75, scale 2), against the
    reported -0.0716. Still roughly 6x below the 0.5 experimental threshold.
