#!/usr/bin/env python3
"""Is a recorded clip still what this code does?

media/ holds clips that took minutes to render and are referenced from two
READMEs, so the recurring question is whether a change since has invalidated
them. Re-recording to find out costs more than the answer is worth. This prints
a hash over every action and every descending-neuron rate for the exact
configuration media/record.sh uses, so two checkouts can be compared exactly:

    git worktree add --detach /tmp/ref <commit-the-clip-was-recorded-at>
    ln -s "$PWD/data" /tmp/ref/data          # the parquet is not in git
    (cd /tmp/ref && /path/to/.venv/bin/python /path/to/scripts/action_trace.py)
    .venv/bin/python scripts/action_trace.py

Same hash, same clip. Used to establish that nothing between c2b67f6 and the
dendrite/head/fixation work changed behaviour: both give
3b6d3072d8600c98d8dfb661abfcf5f9aea243a740a372a4f5586d88e17817f2 over 120 tics.

Note it imports from the CURRENT DIRECTORY, not from this file's location, which
is what lets one copy of the script measure another checkout.

    python scripts/action_trace.py [tics]
"""
import hashlib, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from flydoom.agent import AgentConfig, FlyDoomAgent
from flydoom.doom import DoomConfig, DoomSession
from flydoom.motor import MotorConfig
from flydoom.mechanosensation import MechanoConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
agent = FlyDoomAgent(AgentConfig(
    doom=DoomConfig(scenario="health_gathering_fly", window=False, seed=40,
                    labels=True, fov_deg=170.0, width=1280, height=1024,
                    gnomonic=True, pyramid_blur=True, side_views=(90.0, -90.0)),
    motor=MotorConfig(yaw_source="DNp15", fixed_turn_sign=True,
                      phasic_mdn=True, forward_gain=0.16),
    mechano=MechanoConfig(front_only=True),
    eye_map="anatomical", smell=True, seed=40, bias_mv=0.0,
    optic_gain=16.0, spiking_t4=True, touch=True, device="cuda"))
agent.reset()
h = hashlib.sha256()
acts, rates = [], []
for i in range(N):
    r = agent.tic(i)
    if r is None:
        break
    a = [float(r.action.get(b, 0.0)) for b in DoomSession.BUTTONS]
    acts.append(a)
    rates.append([float(r.rates.get(k, 0.0)) for k in sorted(r.rates)])
    h.update(np.asarray(a, np.float64).tobytes())
    h.update(np.asarray(rates[-1], np.float64).tobytes())
agent.close()
acts = np.asarray(acts)
print(f"tics {len(acts)}")
print(f"sha256 {h.hexdigest()}")
print(f"yaw    first5 {np.round(acts[:5,0],6).tolist()}")
print(f"yaw    last5  {np.round(acts[-5:,0],6).tolist()}")
print(f"fwd    mean {acts[:,1].mean():.6f}  yaw mean {acts[:,0].mean():.6f}")
