#!/usr/bin/env python3
"""Build the daylight-sky arena, flydoom/wads/health_gathering_sky.

health_gathering_supreme with one change a fly would notice first: the dark
stone ceiling is replaced by open sky. The map's ceiling becomes F_SKY1 at its
original height, so the walls keep their proportions, and the sky texture is a
generated daylight gradient with clouds, tileable so there is no seam as the fly
turns through 360 degrees. Top-quarter brightness goes from 10.8 to about 196.

Deterministic: the same seed rebuilds the same bytes.

    python scripts/build_sky_arena.py
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASE = "health_gathering_supreme"
NAME = "health_gathering_sky"

CFG = """\
# Lines starting with # are treated as comments (or with whitespaces+#).
# It doesn't matter if you use capital letters or not.
# It doesn't matter if you use underscore or camel notation for keys, e.g. episode_timeout is the same as episodeTimeout.

doom_scenario_path = health_gathering_sky.wad

# Each step is good for you!
living_reward = 1
# And death is not!
death_penalty = 100

# Rendering options
screen_resolution = RES_320X240
render_hud = false

# Make episodes finish after 2100 actions (tics)
episode_timeout = 2100

# Available buttons
available_buttons =
    {
        TURN_LEFT
        TURN_RIGHT
        MOVE_FORWARD
    }

# Game variables that will be in the state
available_game_variables = { HEALTH }

mode = PLAYER
"""


def sky_png(path: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    W, H = 1024, 200
    y = np.linspace(0, 1, H)[:, None]
    # daylight gradient: deeper blue overhead, paler toward the horizon
    top = np.array([96, 150, 222.])
    hor = np.array([206, 226, 246.])
    base = top * (1 - y[..., None]) + hor * y[..., None]
    base = np.broadcast_to(base, (H, W, 3)).copy()

    # clouds: several octaves of smooth noise, tileable horizontally
    def octave(sc):
        gh, gw = max(2, H // sc), max(2, W // sc)
        g = rng.random((gh, gw + 1))
        g[:, -1] = g[:, 0]
        im = Image.fromarray((g * 255).astype("uint8")).resize(
            (W, H), Image.BICUBIC)
        return np.asarray(im, float) / 255

    n = sum(octave(s) * w for s, w in ((64, .5), (32, .25), (16, .15), (8, .1)))
    n = (n - n.min()) / (n.max() - n.min())
    cover = np.clip((n - 0.45) * 2.6, 0, 1)[..., None]
    cloud = np.array([248, 249, 252.])
    img = base * (1 - cover) + cloud * cover
    Image.fromarray(img.clip(0, 255).astype("uint8")).save(path)


def build_wad(src: Path, png: Path, out: Path) -> list[str]:
    data = src.read_bytes()
    ident, nl, off = struct.unpack("<4sii", data[:12])
    names, blobs = [], []
    for i in range(nl):
        o, s, name = struct.unpack("<ii8s", data[off + 16 * i:off + 16 * i + 16])
        names.append(name)
        blobs.append(data[o:o + s])
    i = [n.rstrip(b"\0") for n in names].index(b"TEXTMAP")
    blobs[i] = blobs[i].decode("latin1").replace(
        'textureceiling = "CEIL4_1";', 'textureceiling = "F_SKY1";'
    ).encode("latin1")
    # ZDoom takes PNG textures from the TX_START..TX_END namespace; a texture
    # named SKY1 there replaces the IWAD's sky for this map.
    pad = lambda s: s.encode().ljust(8, b"\0")  # noqa: E731
    names += [pad("TX_START"), pad("SKY1"), pad("TX_END")]
    blobs += [b"", png.read_bytes(), b""]
    body, dirs, pos = b"", [], 12
    for nm, b in zip(names, blobs):
        dirs.append(struct.pack("<ii8s", pos, len(b), nm))
        body += b
        pos += len(b)
    out.write_bytes(struct.pack("<4sii", ident, len(names), 12 + len(body))
                    + body + b"".join(dirs))
    return [n.rstrip(b"\0").decode() for n in names]


def main() -> int:
    import vizdoom

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "flydoom" / "wads")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src = Path(os.path.dirname(vizdoom.__file__)) / "scenarios" / f"{BASE}.wad"
    png = args.out / "sky_day.png"
    sky_png(png)
    lumps = build_wad(src, png, args.out / f"{NAME}.wad")
    (args.out / f"{NAME}.cfg").write_text(CFG)
    print(f"wrote {args.out / NAME}.wad, lumps {lumps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
