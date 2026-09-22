#!/usr/bin/env python3
"""Build the stripe arenas, flydoom/wads/stripe_{fix,blank}.

WHY A CYLINDER WITH ONE BAR ON IT

Everything this project has measured in Doom asks the model to do a job the
animal has no circuitry for, in a room full of structure, and the two visual
computations tested so far -- direction selectivity and looming -- are both
temporal-order computations, which is exactly what the model cannot recover.
Fixation is not. Keeping a dark vertical object in front of you needs only
retinotopy, a spatial fact, and the retinotopy here is verified end to end.
It is also, with the optomotor response, about as canonical as fly behaviour
gets: Goetz's torque meter, Heisenberg and Wolf's arena, Buridan's paradigm.

So this arena is the laboratory's, not Doom's. A 72-sided cylinder, radius 512
map units, is uniformly bright except for one dark bar spanning a settable
angular width, and the fly stands at the centre. The cylinder matters: at
constant distance Doom's light diminishing is constant too, so the wall carries
no brightness gradient and the bar is the ONLY azimuthal feature in the world.
Nothing to collect, no damaging floor, no enemies, and with the fly tethered at
the centre the bar's position on the retina is a pure function of what the
brain has commanded.

    stripe_fix     the bar, default 20 degrees wide
    stripe_blank   the same cylinder with no bar at all, which is the control
                   for an analysis that could manufacture a peak out of a
                   random walk in heading

Deterministic: the same arguments rebuild the same bytes.

    python scripts/build_stripe_arena.py
"""

from __future__ import annotations

import argparse
import math
import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASE = "health_gathering_supreme"          # borrowed for its lump skeleton

SEGMENTS = 72                              # 5 degrees of azimuth each
RADIUS = 512.0
CEILING = 512

CFG = """\
# Built by scripts/build_stripe_arena.py -- see that file for what this is.

doom_scenario_path = {name}.wad

living_reward = 1
death_penalty = 0

screen_resolution = RES_320X240
render_hud = false

episode_timeout = 6000

available_buttons =
    {{
        TURN_LEFT
        TURN_RIGHT
        MOVE_FORWARD
    }}

available_game_variables = {{ HEALTH }}

mode = PLAYER
"""

THING_FLAGS = ("skill1 skill2 skill3 skill4 skill5 skill6 skill7 skill8 "
               "single dm coop class1 class2 class3 class4").split()


def flat(value: float, size: int = 128, tint=(0.62, 1.0, 0.78)) -> Image.Image:
    """A uniform patch. Uniform is the point: any texture detail would be a
    second azimuthal feature competing with the bar."""
    img = np.full((size, size), float(value))
    rgb = np.stack([img * tint[0], img * tint[1], img * tint[2]], axis=-1)
    return Image.fromarray(rgb.clip(0, 255).astype("uint8"))


def textmap(width_deg: float) -> str:
    """One sector, one ring of one-sided walls, one player start at the middle.

    Wound CLOCKWISE, because a linedef's front side lies to the right of
    v1->v2 and that is the inside of the ring only for this winding.
    """
    step = 360.0 / SEGMENTS
    n_bar = int(round(width_deg / step))
    # The bar straddles world azimuth 0, so the heading that fixates it is 0.
    bar = {(-(k + 1)) % SEGMENTS for k in range(n_bar // 2)}
    bar |= {k % SEGMENTS for k in range(n_bar - len(bar))}

    out = ['namespace = "zdoom";', ""]
    flags = "\n".join(f"{f} = true;" for f in THING_FLAGS)
    out += ["thing", "{", "x = 0.000;", "y = 0.000;", "angle = 180;",
            "type = 1;", "id = 1;", flags, "}", ""]
    for i in range(SEGMENTS):
        a = math.radians(-i * step)                 # clockwise
        out += ["vertex", "{", f"x = {RADIUS * math.cos(a):.3f};",
                f"y = {RADIUS * math.sin(a):.3f};", "}", ""]
    for i in range(SEGMENTS):
        out += ["linedef", "{", f"v1 = {i};", f"v2 = {(i + 1) % SEGMENTS};",
                f"sidefront = {i};", "blocking = true;", "}", ""]
    for i in range(SEGMENTS):
        tex = "FIXBAR" if i in bar else "FIXBG"
        out += ["sidedef", "{", "sector = 0;",
                f'texturemiddle = "{tex}";', "}", ""]
    out += ["sector", "{", 'texturefloor = "FIXFLR";',
            'textureceiling = "F_SKY1";', "heightfloor = 0;",
            f"heightceiling = {CEILING};", "lightlevel = 255;", "}", ""]
    return "\n".join(out)


def build_wad(src: Path, tm: str, textures: dict[str, Path], out: Path) -> None:
    """Rewrite the base WAD's map with ours, keeping only the lumps a bare
    room needs. ZNODES goes: it describes geometry that no longer exists, and
    ZDoom builds nodes for itself when the lump is absent. BEHAVIOR and
    SCRIPTS go with it -- their ACS is the medkit spawner."""
    data = src.read_bytes()
    ident, nl, off = struct.unpack("<4sii", data[:12])
    keep = {"DECORATE": None, "MAP01": None, "TEXTMAP": None, "ENDMAP": None}
    for i in range(nl):
        o, s, name = struct.unpack("<ii8s", data[off + 16 * i:off + 16 * i + 16])
        nm = name.rstrip(b"\0").decode()
        if nm in keep:
            keep[nm] = data[o:o + s]
    keep["TEXTMAP"] = tm.encode("latin1")

    names = [k.encode().ljust(8, b"\0") for k in
             ("DECORATE", "MAP01", "TEXTMAP", "ENDMAP")]
    blobs = [keep[k] for k in ("DECORATE", "MAP01", "TEXTMAP", "ENDMAP")]
    pad = lambda s: s.encode().ljust(8, b"\0")            # noqa: E731
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


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vizdoom

    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=20.0,
                    help="angular width of the bar, degrees")
    ap.add_argument("--bright", type=float, default=200.0)
    ap.add_argument("--dark", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=ROOT / "flydoom" / "wads")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tex = {"FIXBG": args.bright, "FIXBAR": args.dark, "FIXFLR": 120.0}
    paths = {}
    for lump, value in tex.items():
        p = args.out / f"{lump.lower()}.png"
        flat(value).save(p)
        paths[lump] = p
    sky = args.out / "sky_day.png"
    if not sky.exists():
        from build_sky_arena import sky_png
        sky_png(sky)
    paths["SKY1"] = sky

    src = Path(os.path.dirname(vizdoom.__file__)) / "scenarios" / f"{BASE}.wad"
    for name, width in (("stripe_fix", args.width), ("stripe_blank", 0.0)):
        build_wad(src, textmap(width), paths, args.out / f"{name}.wad")
        (args.out / f"{name}.cfg").write_text(CFG.format(name=name))
        n = int(round(width / (360.0 / SEGMENTS)))
        print(f"wrote {name}: {SEGMENTS}-gon r={RADIUS:.0f}, "
              f"{n} dark segments = {n * 360.0 / SEGMENTS:.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
