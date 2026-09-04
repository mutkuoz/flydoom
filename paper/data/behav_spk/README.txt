Closed-loop behaviour across the 2x2 of T4/T5 transfer function (graded or
spiking) and regional inhibitory scale (1 or 2). All four cells run through
the identical protocol on the same thirty held-out seeds (40-69), each
against its own command-matched AR(1) control, paired by seed.

  cell              differs  better  vision r
  graded  oinh=1      6/18    2/15    -0.090
  graded  oinh=2      6/18    3/15    +0.161
  spiking oinh=1      4/18    1/15    -0.066
  spiking oinh=2      6/18    3/15    +0.144

No cell separates from its control by more than any other. The spread 4-6 of
18 is what correlated metrics return on thirty noisy episodes.

This was run to test a prediction that failed. Measured at the horizontal
system, the T4/T5-dependent directional signal is +24.1 Hz for spiking at
scale 1 and -1.5 Hz for spiking at scale 2, a difference of an order of
magnitude and of sign. Behaviour does not follow it: those two cells return
4/18 and 6/18.

vision_steer_r tracks the inhibitory scale (about -0.07 at 1 and +0.15 at 2)
in BOTH transfer functions, and does not track the directional signal, which
scale 2 abolishes. Whatever that correlation measures, it is not the optic
flow signal reaching the steering readout.

The instrument, not only the model, is implicated. The optomotor measurement
resolves a signal that varies by an order of magnitude across these
configurations; thirty episodes of this scenario resolve none of it.
