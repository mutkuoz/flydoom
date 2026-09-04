Does the choice of steering readout matter behaviourally?

Three mismatches were found between how the model was characterised and how it
was run in closed loop:

  quantity              measurements used   behaviour used
  T4/T5 transfer        spiking             graded
  optic gain            16                  1
  yaw readout           -                   DNa02

The first two are parameters and were tested in isolation, both null:

  configuration (yaw = DNa02, 30 held-out seeds 40-69)
    graded,  gain 1  (the reported model)   6/18 differ   2/15 better   r -0.090
    spiking, gain 1                         4/18          1/15          r -0.066
    spiking, gain 16                        5/18          1/15          r -0.043

Restoring the spike threshold does nothing. Driving the optic lobe 16x harder,
which is the difference between an effectively silent optic lobe and a fully
active one, does nothing. Note the vision-steering correlation drifts TOWARD
zero as the visual pathway becomes more functional, which is the wrong
direction for a measure of vision reaching the steering command.

The third is not a parameter but a category error. DNa02 is a steering neuron
for goal-directed walking, targeted by central-complex output (PFL3) and two
synapses from the head-direction system; here it draws 97.7% of its input from
the central brain and 2.3% from visual populations. DNp15 (DNHS1) is the
optomotor neuron: HS cells drive it for yaw rotations and its axons reach the
neck motor system. It is the horizontal system's strongest descending target,
32.3% of HS descending output against 7.1% to DNa02. The model has been
feeding optic flow into the navigation pathway.

  spiking, gain 16, 30 seeds, identical but for the readout
    yaw from DNa02 (navigation)   5/18 differ   1/15 better   r -0.043
    yaw from DNp15 (optomotor)    7/18          4/15          r -0.158

  paired, same seeds (DNp15 - DNa02)
    vision_steer_r_best   -0.12 +- 0.10   *
    yaw_abs_mean          -0.43 +- 0.19   *
    tiles/healed/collisions/displacement: no significant change

The coupling between steering and retinal luminance roughly quadruples; the
task metrics do not separate (7/18 against 5/18 is not a result). So the
category error is real and measurable at the level of the sensorimotor
coupling, and does not by itself produce task performance.

CAVEAT THAT MUST TRAVEL WITH THIS. vision_steer_r was withdrawn earlier as
evidence of vision, on the grounds that it tracked the inhibitory scale in
both transfer functions rather than the directional signal that scale
abolishes. Here the inhibitory scale is constant and only the readout varies,
so that specific objection does not apply, but promoting the measure back to
evidence because it now moves favourably would be exactly the error the
withdrawal was meant to prevent. The mirrored-retina control decides it: a
genuine optic-flow coupling must reverse sign when the retina is mirrored.

MIRRORED-RETINA CONTROL (30 seeds, DNp15 readout, otherwise identical)

  DNa02, normal retina     vision r -0.043
  DNp15, normal retina     vision r -0.158
  DNp15, mirrored retina   vision r -0.034
  paired change under mirroring  +0.124 +- 0.102  (CI excludes zero)

Mirroring flips the retinal sampling grid horizontally, reversing every
horizontal optic-flow signal while leaving firing rates, contrast statistics
and the wiring untouched.

What this establishes: the coupling is retinotopy-dependent. Mirroring
abolishes it and returns it to the DNa02 baseline (-0.034 against -0.043), so
the entire advantage of reading the optomotor descending neuron rather than
the navigation one depends on the retina's spatial arrangement being intact.
An artifact of wiring or of motor statistics would survive mirroring, and this
does not.

What it does NOT establish: the coupling does not REVERSE. A clean optic-flow
relationship should change sign when the retina is mirrored, because rightward
world motion then sweeps the retina leftward. It goes to zero instead. So the
measure is vision-dependent without having the sign structure of an optomotor
response, and we report it as the former only.

This is the strongest visual result in the project, and it is a coupling
result rather than a behavioural one: the task metrics do not separate
(7/18 against 5/18).
