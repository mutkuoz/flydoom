#!/usr/bin/env python3
"""Build health_gathering_forage: an arena that means to a fly what it looks like.

THE PROBLEM THIS FIXES

The corrected behaviour tables found that the model's vision steers it toward
large dark vertical structure, which in a Doom maze is a wall, and noted that
real flies are drawn to dark vertical objects so this is biology transferring
without the wisdom to go with it. That reading was charitable to the arena.
Measured, health_gathering_fly's wall texture is 64 px at luminance 205 beside
64 px at 65, uniform down its entire height -- a Michelson contrast of 0.56 in
a PERFECT TALL DARK BAR, tiled across every wall in the level. The arena built
to feed the motion detectors is also the strongest fixation target a fly could
be shown, and the model walks into it. We built the trap and then called the
animal foolish for falling in.

Two properties of a fly's world are missing, and both are arena properties:

  1. A DARK VERTICAL IS NOT A WALL. Out in the world, a small dark object is
     food or a conspecific or somewhere to land, and approaching it pays. The
     thing a fly must not walk into is not especially dark -- it is simply
     where the world stops. Here the relationship is inverted.

  2. FOOD DOES NOT LOOK LIKE FOOD. A medkit is a pale box. Nothing about it
     engages the approach drive the animal actually has, so the only channel
     that can find it is smell.

WHAT THIS ARENA CHANGES, all of it environment and none of it brain:

  WALLS keep their spatial frequency, because the motion detectors need it, and
  lose their vertical coherence. The grating is cut into horizontal bands whose
  phase is randomised, so any small patch still carries a full-contrast edge at
  the same period -- a local motion detector sees exactly what it saw before --
  while a cell that integrates contrast down a column sees the phases cancel.
  There is no tall dark bar left to approach.

  FOOD becomes a small dark high-contrast blob, which is the stimulus a fly
  approaches. The pickup, the odour and the health are untouched; only its
  appearance changes.

  POISON becomes pale, brighter than its background. If the model is steering
  toward dark objects, this is the manipulation that should make it stop
  eating poison -- it took +8.63 more damage than its control in the fly arena.

THE PREDICTION, which is the point of building it. If "it approaches large dark
structure" is the right description, then moving the darkness off the walls and
onto the food should raise health, lower damage, and lower the time spent stuck
against geometry, WITHOUT touching the brain. If nothing moves, that
description was wrong and the wall-hugging has some other cause.

Deterministic: the same seed rebuilds the same bytes.

    python scripts/build_forage_arena.py
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
NAME = "health_gathering_forage"

WALL_PERIOD = 128          # map units, as in the fly arena: 18-37 deg
WALL_PERIOD_PX = 64        # texture pixels per cycle at that period
ANISO = 1.6                # >1 biases toward vertical edges, which yaw needs
BANDWIDTH = 1.2            # cycles; how tightly power sits on the period

# Wall luminance, chosen by what the LENSES receive rather than by what the
# texture file contains. Measured through the real retina, separating lenses
# that look at wall (elevation 3-25 deg) from lenses that look at ground
# (-12 to -45), as d-prime between the two luminance distributions against the
# contrast left in the wall band for the motion detectors:
#
#     mean 145 (as the floor)  d' 0.91   contrast 0.603
#     mean 175                 d' 1.86   contrast 0.381   <- chosen
#     mean 195                 d' 2.66   contrast 0.278
#
# 175 is the knee: it roughly doubles how separable wall is from ground while
# leaving contrast far above anything the optomotor response needs. Pushing to
# 195 costs a quarter of the contrast to buy separability that is already
# sufficient. The period barely moves d-prime at all (1.84 to 1.96 across 40 to
# 96 px), so it stays where the motion detectors want it rather than being
# spent on this.
#
# Note the floor reads DARKER than the wall on the retina even at equal texture
# means (0.433 against 0.260), because the ground is seen at a grazing angle.
# The texture files being equal was never the whole story.
WALL_MEAN = 175.0
WALL_HALF = 70.0


def wall_texture(size: int = 128, seed: int = 7) -> Image.Image:
    """Band-pass noise: local vertical edges everywhere, no tall bar anywhere.

    The wall has to satisfy two things that pull against each other. A yaw
    motion detector needs LOCAL VERTICAL EDGES, because those are what
    horizontal motion sweeps across. The fixation drive that walks this model
    into walls is triggered by GLOBAL VERTICAL COHERENCE. So: edges everywhere,
    bar nowhere.

    Measured against the alternatives, at the same contrast, scoring vertical
    coherence (spread of column means, 0 is no bar), the share of horizontal
    spectral power sitting at the period the motion detectors prefer, and the
    ratio of vertical to horizontal edge energy:

        square bars (the fly arena)   70.0   40.2%   vertical only
        phase-staggered bars           0.0   40.2%   0.78
        pink 1/f noise                 4.0   15.5%   1.00
        band-pass noise               10.8   67.4%   1.02
        THIS: band-pass, aniso 1.6,
              column mean removed      0.0   66.3%   1.24

    Two things earn their place. A square wave spends most of its energy in
    harmonics it does not need -- band-pass noise puts two thirds of its power
    at the preferred period against the grating's two fifths, so it drives the
    detectors HARDER while looking like something a fly might actually see.
    And subtracting the column mean removes exactly the tall-bar component and
    nothing else: local structure is untouched, coherence goes to zero.

    The anisotropy squeezes vertical frequency, which moves energy into
    horizontal frequency, which is vertical EDGES. Pushed the other way it
    starves the yaw detectors -- a first attempt at 2.5 came out at 0.37, all
    horizontal banding, and would have quietly removed the drive this arena
    exists to preserve.
    """
    rng = np.random.default_rng(seed)
    f0 = size / WALL_PERIOD_PX          # cycles per texture at the preferred period
    fy, fx = np.meshgrid(np.fft.fftfreq(size) * size,
                         np.fft.fftfreq(size) * size, indexing="ij")
    ring = np.exp(-((np.hypot(fy * ANISO, fx) - f0) ** 2) / (2 * BANDWIDTH ** 2))
    phase = rng.uniform(0, 2 * np.pi, (size, size))
    img = np.real(np.fft.ifft2(ring * np.exp(1j * phase)))
    img -= img.mean(axis=0, keepdims=True)      # kill the tall-bar component
    img = img / (np.abs(img).max() + 1e-9)
    img = WALL_MEAN + img * WALL_HALF           # brighter than the ground
    rgb = np.stack([img * 0.55, img, img * 0.75], axis=-1)   # green-weighted
    return Image.fromarray(rgb.clip(0, 255).astype("uint8"))


# ---- sprites, in Doom's column-major picture format --------------------

def patch(pix: np.ndarray, transparent: int = 255) -> bytes:
    """numpy array of palette indices -> a Doom patch lump.

    `transparent` marks pixels that are not drawn. Offsets place the sprite so
    it stands on the floor, centred left to right.
    """
    h, w = pix.shape
    head = struct.pack("<4h", w, h, w // 2, h)
    cols, body = [], b""
    for cx in range(w):
        cols.append(len(head) + 4 * w + len(body))
        col = pix[:, cx]
        y = 0
        while y < h:
            while y < h and col[y] == transparent:
                y += 1
            if y >= h:
                break
            y0 = y
            run = []
            while y < h and col[y] != transparent:
                run.append(int(col[y]))
                y += 1
            # post: topdelta, length, pad, pixels, pad
            body += struct.pack("<BBB", y0, len(run), 0)
            body += bytes(run) + b"\x00"
        body += b"\xff"
    return head + struct.pack(f"<{w}I", *cols) + body


def blob(radius: int, core: int, edge: int, seed: int = 3) -> np.ndarray:
    """A round blob of palette indices, `core` at the centre fading to `edge`."""
    n = radius * 2 + 1
    yy, xx = np.mgrid[0:n, 0:n] - radius
    r = np.hypot(xx, yy) / radius
    out = np.full((n, n), 255, dtype=np.uint8)
    inside = r <= 1.0
    # two-tone rather than a gradient: the palette is not a ramp, and a fly's
    # 5.4 degree acceptance angle averages an object this size to one number
    out[inside] = edge
    out[r <= 0.72] = core
    return out


def build_wad(src: Path, textures: dict[str, Path],
              sprites: dict[str, bytes], out: Path) -> list[str]:
    data = src.read_bytes()
    ident, nl, off = struct.unpack("<4sii", data[:12])
    names, blobs = [], []
    for i in range(nl):
        o, s, name = struct.unpack("<ii8s", data[off + 16 * i:off + 16 * i + 16])
        names.append(name)
        blobs.append(data[o:o + s])
    i = [n.rstrip(b"\0") for n in names].index(b"TEXTMAP")
    tm = blobs[i].decode("latin1")
    tm = (tm.replace('textureceiling = "CEIL4_1";', 'textureceiling = "F_SKY1";')
            .replace('texturemiddle = "STONE2";', 'texturemiddle = "FORWALL";')
            .replace('texturefloor = "NUKAGE1";', 'texturefloor = "FORFLOOR";')
            .replace("lightlevel = 210;", "lightlevel = 255;"))
    blobs[i] = tm.encode("latin1")

    pad = lambda s: s.encode().ljust(8, b"\0")            # noqa: E731
    names.append(pad("TX_START")); blobs.append(b"")
    for lump, path in textures.items():
        names.append(pad(lump)); blobs.append(path.read_bytes())
    names.append(pad("TX_END")); blobs.append(b"")
    # Sprites override the IWAD's from inside S_START/S_END.
    names.append(pad("S_START")); blobs.append(b"")
    for lump, raw in sprites.items():
        names.append(pad(lump)); blobs.append(raw)
    names.append(pad("S_END")); blobs.append(b"")

    body, dirs, pos = b"", [], 12
    for nm, b in zip(names, blobs):
        dirs.append(struct.pack("<ii8s", pos, len(b), nm))
        body += b
        pos += len(b)
    out.write_bytes(struct.pack("<4sii", ident, len(names), 12 + len(body))
                    + body + b"".join(dirs))
    return [n.rstrip(b"\0").decode() for n in names]


CFG = """\
# Built by scripts/build_forage_arena.py -- see that file for what differs from
# health_gathering_fly and why.

doom_scenario_path = health_gathering_forage.wad

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

# Palette indices. 8 is near-black (luminance 7), 6 a shade above it (19); 4 is
# pure white and 80 a pale grey, which sit above every background in the level.
FOOD_CORE, FOOD_EDGE = 8, 6
POISON_CORE, POISON_EDGE = 4, 80


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vizdoom

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "flydoom" / "wads")
    ap.add_argument("--food-radius", type=int, default=10)
    ap.add_argument("--poison-radius", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wall = args.out / "forage_wall.png"
    wall_texture().save(wall)
    floor = args.out / "fly_floor.png"          # the bright ground, unchanged
    sky = args.out / "sky_day.png"
    if not floor.exists() or not sky.exists():
        from build_fly_arena import floor_texture, sky_texture
        if not floor.exists():
            floor_texture().save(floor)
        if not sky.exists():
            sky_texture(sky)

    food = patch(blob(args.food_radius, FOOD_CORE, FOOD_EDGE))
    poison = patch(blob(args.poison_radius, POISON_CORE, POISON_EDGE))
    sprites = {"MEDIA0": food}
    for f in "ABCD":                    # the poison's four animation frames
        sprites[f"BON1{f}0"] = poison

    src = Path(os.path.dirname(vizdoom.__file__)) / "scenarios" / f"{BASE}.wad"
    lumps = build_wad(src, {"SKY1": sky, "FORWALL": wall, "FORFLOOR": floor},
                      sprites, args.out / f"{NAME}.wad")
    (args.out / f"{NAME}.cfg").write_text(CFG)

    g = np.asarray(Image.open(wall), float)[..., 1]
    print(f"wrote {args.out / NAME}.wad")
    print(f"  wall: contrast {(g.max()-g.min())/(g.max()+g.min()):.2f}, "
          f"column-mean spread {g.mean(axis=0).std():.1f} "
          f"(the fly arena's is 70; near 0 means no tall bar)")
    print(f"  food:   {2*args.food_radius+1} px blob, palette {FOOD_CORE} (dark)")
    print(f"  poison: {2*args.poison_radius+1} px blob, palette {POISON_CORE} (pale)")
    print(f"  lumps {lumps[-8:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
