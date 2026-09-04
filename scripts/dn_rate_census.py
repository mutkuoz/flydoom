#!/usr/bin/env python3
"""Census of descending-neuron firing rates in the running closed loop.

The output stage was never characterised. Every experiment in this project
examined the sensory side and then read two descending neurons, without
establishing what the descending population as a whole was doing.

MEASUREMENT NOTE, because it is easy to get wrong by a factor of 57: spikes
must be accumulated every SIMULATION SUBSTEP, not once per Doom tic. Sampling
once per tic and dividing by the substep-resolved window undercounts by up to
the substep count and makes the whole population look silent.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import torch

from flydoom import config
from flydoom.agent import FlyDoomAgent, AgentConfig
from flydoom.doom import DoomConfig


def main(tics: int = 120, device: str = "cpu") -> int:
    a = FlyDoomAgent(AgentConfig(device=device,
                                 doom=DoomConfig(labels=True, seed=0,
                                 scenario="health_gathering_supreme")))
    a.reset()
    cls = pl.read_csv(config.RAW_DIR / "classification.csv.gz",
                      infer_schema_length=50_000)
    pos = {int(r): i for i, r in enumerate(a.graph.root_ids)}
    ids = cls.filter(pl.col("super_class") == "descending")["root_id"].to_list()
    DN = torch.as_tensor(np.array(sorted(pos[int(r)] for r in ids
                                         if int(r) in pos)))
    acc = torch.zeros(len(DN))
    steps = [0]
    orig = a.net.step

    def step(*args, **kw):
        r = orig(*args, **kw)
        acc.add_(a.net.spiked[DN].float())
        steps[0] += 1
        return r

    a.net.step = step
    for t in range(tics):
        if a.tic(t) is None:
            break

    hz = (acc / (steps[0] * config.DT)).numpy()
    ceiling = 1.0 / config.T_REFRAC
    print(f"descending neurons {len(DN)}   window {steps[0]*config.DT:.2f} s"
          f"   refractory ceiling {ceiling:.0f} Hz")
    for lo, hi, lab in ((0, 0.1, "silent <0.1 Hz"),
                        (0.1, 50, "1-50 Hz (physiological)"),
                        (50, 200, "50-200 Hz"),
                        (200, 1e9, ">200 Hz")):
        m = (hz >= lo) & (hz < hi)
        print(f"  {lab:<26}{int(m.sum()):5}  ({100*m.mean():.1f}%)")
    print(f"  median {np.median(hz):.1f}   mean {hz.mean():.1f}   "
          f"max {hz.max():.0f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
