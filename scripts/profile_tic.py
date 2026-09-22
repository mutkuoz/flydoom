#!/usr/bin/env python3
"""Where a tic goes: five sections, one synchronisation each per tic.

Coarse on purpose. Synchronising around every substep -- 171 syncs a tic --
inflates everything it measures, and reported the per-substep sensory update at
22% of a tic when removing it entirely saves 3%. See paper/data/speed_NOTE.txt
for what the sections cost and which two of them were accidental.

    python scripts/profile_tic.py [tics]
"""
import sys, time, torch
sys.path.insert(0,'.'); sys.path.insert(0,'experiments')
from flydoom.agent import AgentConfig, FlyDoomAgent
from flydoom.doom import DoomConfig, DoomSession
from flydoom.motor import MotorConfig
from flydoom import config
from m9_behaviour import RENDER
N = int(sys.argv[1]) if len(sys.argv)>1 else 60
a = FlyDoomAgent(AgentConfig(
    doom=DoomConfig(scenario="stripe_fix", window=False, seed=1, **RENDER["fast"]),
    motor=MotorConfig(yaw_source="DNp15", fixed_turn_sign=True, phasic_mdn=True,
                      forward_gain=0.16),
    eye_map="anatomical", seed=1, optic_gain=16.0, spiking_t4=True, device="cuda"))
a.reset()
for i in range(5): a.tic(i)
t = dict.fromkeys(("frame","sample","substeps","readout","engine"), 0.0)
sync = torch.cuda.synchronize
for i in range(N):
    sync(); t0=time.perf_counter()
    frame = a.doom.frame()
    sync(); t1=time.perf_counter(); t["frame"]+=t1-t0
    lum_prev, lum_cur = a.vision.begin_tic(frame)
    sync(); t2=time.perf_counter(); t["sample"]+=t2-t1
    for sub in range(a.substeps):
        out_set,_ = a.vision.substep_drive(lum_prev, lum_cur,(sub+1)/a.substeps,
                                a.net.n, a.net.p.graded_max_rate, config.DT)
        a.net.step(g_ext=a.gext, out_set=out_set if a.graded else None)
        a.motor.observe(a.net)
    sync(); t3=time.perf_counter(); t["substeps"]+=t3-t2
    rates = a.motor.sample(); action = a.motor.decode()
    sync(); t4=time.perf_counter(); t["readout"]+=t4-t3
    a.doom.step([action.get(b,0.0) for b in DoomSession.BUTTONS], 1)
    sync(); t["engine"]+=time.perf_counter()-t4
a.close()
tot=sum(t.values())
print(f"{N} tics, {tot/N*1000:.1f} ms/tic\n")
for k,v in sorted(t.items(), key=lambda kv:-kv[1]):
    print(f"  {k:<10}{v/N*1000:8.2f} ms{100*v/tot:7.1f}%")
