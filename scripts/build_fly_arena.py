#!/usr/bin/env python3
"""Build the fly arena, flydoom/wads/health_gathering_fly.

The sky arena fixed the dorsal field; this fixes the rest. Doom is built for an
eye with about 60 times the fly's angular resolution and a display that assumes
human adaptation, and three of its properties waste the signal the motion
pathway runs on. All three are arena properties, not model properties: this is
the same move a fly-vision laboratory makes when it builds a striped drum.

WALLS carry the horizontal structure a correlator needs, and Doom's carry it at
the wrong scale. STONE2's detail sits near one map unit, which at a typical
wall distance of 200-400 units is well under a degree, so the 5.4 degree
acceptance function averages it into flat grey: measured, the wall band's
retinal contrast comes almost entirely from the silhouette against the sky
rather than from the wall itself. The replacement is a vertical square-wave
grating of period 128 units, which spans 18-37 degrees over that distance
range, inside the 20-30 degree optimum reported for Drosophila, and it is
rendered at full contrast rather than Doom's muted palette.

The FLOOR is where a walking fly gets its translational flow, and Doom's is a
dark liquid flat: raw luminance 0.023 against the sky's 0.71, with a per-lens
standard deviation of 0.007. There is nothing there to track. The replacement
is a bright coarse pattern whose features span roughly 20-40 degrees at walking
distance.

LIGHT is set flat at maximum. Doom shades sectors to fake depth, which is a
cue for an eye that infers depth from brightness; the fly infers it from motion
parallax, and uneven lighting only adds a static luminance gradient across the
retina.

Deterministic: the same seed rebuilds the same bytes.

    python scripts/build_fly_arena.py
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
NAME = "health_gathering_fly"

# Map units. A 128-unit period spans 18 degrees at 400 units and 37 at 200,
# which brackets the reported 20-30 degree optimum for the optomotor response.
WALL_PERIOD = 128
FLOOR_PERIOD = 128

CFG = """\
# Built by scripts/build_fly_arena.py -- see that file for what differs from
# health_gathering_supreme and why.

doom_scenario_path = health_gathering_fly.wad

living_reward = 1
death_penalty = 100

screen_resolution = RES_320X240
render_hud = false

episode_timeout = 2100

available_buttons =
    {
        TURN_LEFT
        TURN_RIGHT
        MOVE_FORWARD
    }

available_game_variables = { HEALTH }

mode = PLAYER
"""


def wall_texture(size: int = 128) -> Image.Image:
    """Vertical square-wave grating, one full period across the texture."""
    x = np.arange(size)
    stripe = np.where((x % size) < size // 2, 205.0, 65.0)
    img = np.repeat(stripe[None, :], size, axis=0)
    # A faint horizontal seam every half height keeps the wall from being a
    # perfectly uniform bar vertically, which would make elevation unreadable.
    img[::size // 2, :] *= 0.88
    rgb = np.stack([img * 0.55, img, img * 0.75], axis=-1)   # green-weighted
    return Image.fromarray(rgb.clip(0, 255).astype("uint8"))


def floor_texture(size: int = 128, seed: int = 11) -> Image.Image:
    """Bright irregular ground, at a scale the fly resolves.

    Irregular rather than a grid, and smooth rather than hard-edged, because a
    regular tile of hard squares lines up with the tiling and renders as radial
    streaks converging on the horizon -- a strong pattern that belongs to the
    tiling rather than to the ground. Features run 32-64 map units, which spans
    20-40 degrees at the distance a walking fly reads ground flow from, and
    under 5 degrees beyond 700 units, where the engine has no mipmaps and the
    lens blur is what removes the aliasing.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size))
    for cells, weight in ((2, 1.0), (4, 0.55), (8, 0.25)):
        g = rng.random((cells + 1, cells + 1))
        g[-1, :] = g[0, :]                    # tileable in both directions
        g[:, -1] = g[:, 0]
        up = np.asarray(Image.fromarray((g * 255).astype("uint8")).resize(
            (size + 1, size + 1), Image.BICUBIC), float)[:size, :size] / 255
        img += weight * up
    img = (img - img.min()) / max(float(np.ptp(img)), 1e-9)
    img = 110 + 105 * img                     # mid-bright, never black
    rgb = np.stack([img * 0.5, img * 0.9, img * 0.65], axis=-1)
    return Image.fromarray(rgb.clip(0, 255).astype("uint8"))


def sky_texture(path: Path) -> None:
    """The daylight sky of the sky arena, rebuilt here so this file stands
    alone."""
    from build_sky_arena import sky_png
    sky_png(path)


def build_wad(src: Path, textures: dict[str, Path], out: Path) -> list[str]:
    data = src.read_bytes()
    ident, nl, off = struct.unpack("<4sii", data[:12])
    names, blobs = [], []
    for i in range(nl):
        o, s, name = struct.unpack("<ii8s", data[off + 16 * i:off + 16 * i + 16])
        names.append(name)
        blobs.append(data[o:o + s])
    i = [n.rstrip(b"\0") for n in names].index(b"TEXTMAP")
    tm = blobs[i].decode("latin1")
    before = (tm.count('"STONE2"'), tm.count('"NUKAGE1"'))
    tm = (tm.replace('textureceiling = "CEIL4_1";', 'textureceiling = "F_SKY1";')
            .replace('texturemiddle = "STONE2";', 'texturemiddle = "FLYWALL";')
            .replace('texturefloor = "NUKAGE1";', 'texturefloor = "FLYFLOOR";'))
    # flat, maximum daylight
    tm = tm.replace("lightlevel = 210;", "lightlevel = 255;")
    blobs[i] = tm.encode("latin1")
    print(f"  TEXTMAP: {before[0]} wall faces, {before[1]} floor -> "
          f"FLYWALL/FLYFLOOR, light 255")

    pad = lambda s: s.encode().ljust(8, b"\0")  # noqa: E731
    names.append(pad("TX_START"))
    blobs.append(b"")
    for lump, path in textures.items():
        names.append(pad(lump))
        blobs.append(path.read_bytes())
    names.append(pad("TX_END"))
    blobs.append(b"")

    body, dirs, pos = b"", [], 12
    for nm, b in zip(names, blobs):
        dirs.append(struct.pack("<ii8s", pos, len(b), nm))
        body += b
        pos += len(b)
    out.write_bytes(struct.pack("<4sii", ident, len(names), 12 + len(body))
                    + body + b"".join(dirs))
    return [n.rstrip(b"\0").decode() for n in names]


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vizdoom

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "flydoom" / "wads")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wall = args.out / "fly_wall.png"
    floor = args.out / "fly_floor.png"
    sky = args.out / "sky_day.png"
    wall_texture().save(wall)
    floor_texture().save(floor)
    if not sky.exists():
        sky_texture(sky)

    src = Path(os.path.dirname(vizdoom.__file__)) / "scenarios" / f"{BASE}.wad"
    lumps = build_wad(src, {"SKY1": sky, "FLYWALL": wall, "FLYFLOOR": floor},
                      args.out / f"{NAME}.wad")
    (args.out / f"{NAME}.cfg").write_text(CFG)
    print(f"wrote {args.out / NAME}.wad, lumps {lumps[-5:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
