Antennal mechanosensation: a sensory channel the model lacked.

Walking into a wall produced no afferent activity anywhere in the brain, so
collisions -- the one behavioural metric that moved under every intervention
tried -- were a consequence the agent could not sense. flydoom/mechanosensation.py
adds antennal contact on the same argument that justifies the olfactory
channel: the engine knows about contact, the animal has a sensor for it, and
FAFB carries 2,674 mechanosensory neurons that otherwise go unused.

Target choice. The obvious one is wrong. 1,113 eye bristles and 304 head
bristles sit one synapse from 109 descending neurons, but those are DNg15,
DNg35, DNg48, DNg84 and DNg85 -- the DNg group is largely grooming, which is
correct biology: deflect a fly's head bristles and it grooms rather than
steering. Flies wall-follow by ANTENNAL contact, and the 486 wind/gravity
afferents are two synapses from DNa02 against roughly six for vision.

Result: the channel works and does not steer.

  baseline DNa02 L-R   -115.19 Hz   (single-brain asymmetry)
  left contact           -3.12 Hz
  right contact          -3.66 Hz
  sign consistency        2/3 seeds

Contact reaches DNa02 and shifts it by about 3 Hz, but the shift does not
reverse with the contacted side, which is the whole test.

The anatomy accounts for it. Summing signed two-hop weight:

                  -> DNa02_L   -> DNa02_R   ratio
  left antenna         691         513       1.35
  right antenna        871         692       1.26

Both antennae project preferentially to the SAME side. A turn-away reflex
needs left to drive one side and right the other; this pathway is not
differentially lateralised at two synapses, so it cannot produce a
side-dependent turn regardless of how the afferents are driven.

Caveat: a signed two-hop weight sum ignores sign interactions and longer
routes, and the real reflex may run through the ventral nerve cord, which
FAFB does not contain.
