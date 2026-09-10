Two attempts to put gameplay optic flow inside the correlator's temporal band.

Geometry: with 5 degree ommatidial spacing and an 80 ms slow arm, a correlator
responds near ONE column swept per delay, i.e. about 62 deg/s. Gameplay runs at
420 deg/s, which is 6.7 columns per delay. The measured DNp15 tuning agrees:
it peaks at 2-3 Hz drift on a 15 degree grating, 30-45 deg/s, and inverts by
6 Hz.

ATTEMPT 1 -- cap the agent's turn (yaw-max 1.8 deg/tic, 30 seeds)
  all applied, yaw-max 12    11/18 differ   6/15 better   r -0.183
  speed-matched, yaw-max 1.8  3/18          2/15          r -0.006
  speed-matched, mirrored     5/18          3/15          r -0.031
Failed. The loop is self-generating: the agent's own turning produces the
optic flow its vision responds to, so throttling the turn removes the signal
rather than retuning it. Correlation is scale-invariant, so compressing the
command could not have done this on its own.

ATTEMPT 2 -- slow the world instead (FLYDOOM_SUBSTEPS 228, 12 seeds, 400 tics)
  real time, 57 substeps      7/18 differ   5/15 better   r -0.214
  4x dilation, 228 substeps   2/18          1/15          r +0.105
  paired (4x - 1x): vision_steer_r +0.320 +- 0.122 *, healed -17.0 +- 13.5 *
Also failed as a fix, and behaviour is worse.

But the coupling REVERSES rather than vanishing, from -0.214 to +0.105, under
a manipulation that changes only the speed of the world. The DNp15 tuning
curve already showed sign inversions across temporal frequency (negative at 1
and 6 Hz, positive at 2 to 4), so 4x dilation plausibly moves gameplay into an
inverting lobe. A coupling produced by motor statistics has no reason to
reverse when only world speed changes, which makes this the cleanest evidence
so far that vision_steer_r reflects vision.

Conclusion: the temporal hypothesis is refuted as a repair in both
implementations. The best configuration remains all-applied at real time.
