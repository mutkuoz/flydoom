"""ViZDoom binding: frames in, buttons out.

Two jobs, both fiddly enough to be worth isolating from the agent loop.

**Vision.** A Doom frame becomes per-ommatidium luminance. Spec 6.1 asks for a
Gaussian acceptance function rather than nearest-neighbour sampling, and the
efficient way to get one is to pre-blur the frame with a Gaussian matched to
the acceptance angle and then bilinearly sample at each column's gaze
direction. That is mathematically the same as convolving each column's kernel
with the image, and it costs one separable blur instead of 1,581 gathers.

**Timing.** Doom runs at 35 tics/s; the fly resolves flicker to ~150 Hz. The
fly is faster than the game, so each frame is HELD across all 57 LIF substeps.
Blanking between frames would look like the world strobing and the optic lobe
would report violent motion that is not there.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import config
from .retina import DOOM_FOV_DEG, EYE_SPLAY_DEG, ACCEPTANCE_RATIO

# Rec. 601 luma. Doom's palette is overwhelmingly brown and grey, so the
# chromatic channels carry almost nothing -- spec 6.1 predicts R7/R8 are dead
# weight in v1, and this is why we collapse to luminance immediately.
LUMA_HUMAN = np.array([0.299, 0.587, 0.114], dtype=np.float32)
"""Rec.601 photopic luma -- the HUMAN spectral response, weighted to red."""

LUMA_FLY = np.array([0.03, 0.35, 0.62], dtype=np.float32)
"""Approximate R1-6 spectral weighting for Doom's RGB primaries. [OURS]

R1-6 are the photoreceptors this model drives, and they carry Rh1, which is
blue-green sensitive with a second, UV peak via a sensitizing pigment. Doom
renders no UV, so the UV lobe is simply unavailable and what is left is the
blue-green preference; these weights approximate Rh1's relative sensitivity at
Doom's R, G and B primaries and are normalised to sum to one.

The contrast with LUMA_HUMAN is the point: Rec.601 puts 30% of its weight on
red, where R1-6 is nearly blind, and 11% on blue, where it is most sensitive.
Under human luma a red wall and a blue wall of equal fly-brightness look very
different to this retina, and vice versa. Approximate, and flagged as such --
three broad primaries cannot reconstruct a spectrum -- but directionally right
where Rec.601 is directionally wrong.
"""

LUMA = LUMA_FLY

FLYBAND_MIX = np.array([0.0, 0.45, 0.55], dtype=np.float32)
"""Where repainted light is emitted: blue-green, nothing in the red primary.
Split across G and B rather than piled into one so the result survives a
display gamut and stays a plausible surface colour."""


@dataclass
class DoomConfig:
    scenario: str = "defend_the_center"
    width: int = 320
    height: int = 240
    fov_deg: float = DOOM_FOV_DEG
    splay_deg: float = EYE_SPLAY_DEG
    window: bool = True
    frame_skip: int = 1
    """Doom tics advanced per agent decision. 1 = decide every tic."""
    seed: int | None = 0
    living_reward: float | None = None
    render_weapon: bool = False
    """Draw the player's own gun. OFF, and it is not cosmetic.

    The weapon sprite is painted at a FIXED position in the viewport, so it
    never moves when the world moves, and it BOBS while walking. MEASURED: it
    covers 4.0% of the viewport and 137 of 4,541 ommatidial columns (3.0% of
    the lattice), with a temporal s.d. of 20.6 inside its own region during
    walking -- self-generated motion that is uncorrelated with the scene and
    permanently locked to the same retinotopic position.

    That is the failure the mean-luminance surround fix already guards against,
    applied to a second object nobody removed. It is far too small to explain
    the motion-vision result (3% of the retina cannot account for it), but it
    is a real confound for the wide-field looming cells, which pool across the
    whole visual field and which M4 shows are responding to dark AREA.

    An audit of the other non-world elements found nothing else: HUD,
    crosshair, decals, particles, messages and effect sprites are all already
    disabled by the scenario configs. The gun was the only one.

    A fly does not carry a shotgun.
    """

    repaint: str = "flyband"
    """Repaint the rendered world into the band R1-6 can actually see.

    Doom is painted in browns and reds, which is close to the worst possible
    palette for this animal: Rh1 is blue-green sensitive and nearly blind at
    Doom's red primary. MEASURED, weighting Doom's RGB by R1-6 sensitivity
    LOWERS retinal contrast from 0.528 to 0.457, because most of the scene's
    energy sits in a channel the receptor cannot use. The environment, not the
    brain, is what is wrong.

    Changing the environment is not a change to the model, and it is what a
    fly-vision lab does as a matter of course -- arenas are built from green
    and UV emitters precisely because that is the animal's band.

    The transform is licensed by the fact that R1-6 are a SINGLE spectral
    class, so the motion pathway is monochromatic: chromatic detail is not
    something it can use, and preserving hue buys the model nothing while
    costing it most of its contrast. So we take the scene's achromatic
    structure and re-emit it in blue-green, which is equivalent to repainting
    every surface in the arena and much cheaper than shipping a texture pack.

        "none"     leave the frame as rendered
        "flyband"  preserve luminance structure, move it in-band
    """

    fly_spectrum: bool = True
    """Weight Doom's RGB by R1-6 spectral sensitivity instead of human luma.
    See LUMA_FLY. Set False for Rec.601."""

    gnomonic: bool = False
    """Map each lens into the picture with the correct flat-camera formula.

    A flat (rectilinear) camera projects a direction at azimuth a and
    elevation e to x = tan(a), y = tan(e) / cos(a). The original mapping used
    y = tan(e) everywhere, which is right on the vertical midline and wrong
    everywhere else. MEASURED at the default 130 degree view: a lens reads an
    elevation off by a median 1.8 degrees (0.4 lens spacings), 8.0 beyond 40
    degrees of azimuth, 22.7 at worst, and 148 lenses it classed as on-screen
    were in fact off it, reading smeared border pixels. Off by default so
    earlier results reproduce exactly; on for any new work.
    """

    pyramid_blur: bool = False
    """Let each lens's acceptance blur follow its position in the picture.

    One separable blur cannot vary across the image, so it was sized for the
    centre. In a flat projection a pixel covers LESS angle toward the edges,
    by a factor of 1/(1 + r^2) radially, so a centre-sized blur is too narrow
    at the edges and edge lenses sample a pinprick rather than their true
    acceptance angle. At 170 degrees that factor reaches about 130. This builds
    a blur pyramid and gives each lens the level matching its local scale.
    """

    view_bob: float = 0.25
    """Doom's walking head-bob (the `movebob` CVar), set explicitly.

    ViZDoom reads CVars from _vizdoom.ini in the working directory and writes
    them back on exit, so a value set once in any session silently becomes
    every later run's default. That happened: a probe set it to 0 and the ini
    kept it. 0.25 is Doom's default and what every earlier result ran with.
    """

    side_views: tuple = ()
    """Extra cameras, as yaw offsets in degrees (positive = right), each a
    multiple of 90. (90.0, -90.0) gives every lens a camera within 45 degrees
    of its own direction, so each eye's full 170 x 150 field is seen.

    One flat camera cannot see past about 85 degrees, and the eyes point 40
    degrees outward, so without this the outer 40 degrees of each eye -- 18%
    of the lenses -- look at nothing. Doom draws one view per tic, so each
    side view is rendered by a separate engine that is loaded from the main
    game's save every tic and stepped once with the fly's own action turned
    by the offset. It renders the same world, not a lookalike: at offset 0
    the frames match the main game pixel for pixel. Requires gnomonic.
    """

    linearise_gamma: bool = True
    """Undo Doom's sRGB gamma before sampling.

    Photoreceptors respond to photon flux, which is linear in intensity, but
    Doom writes gamma-encoded sRGB. The acceptance function is a spatial
    INTEGRAL over that flux, so the Gaussian pre-blur has to run in linear
    light: blurring gamma-encoded values averages the wrong quantity and
    systematically biases every column towards its darker neighbours, worst at
    exactly the high-contrast edges the motion pathway feeds on.
    """

    interpolate: bool = True
    """Ramp luminance across the substeps between one frame and the next.

    Doom draws at 35 Hz and the simulator runs 57 substeps per frame, so
    holding the frame gives the retina a step change followed by 56 substeps of
    exactly zero temporal derivative. Every computation this model is asked to
    perform on the visual stream -- direction selectivity, looming, wall
    avoidance -- is a computation over temporal ORDER, and the delayed arm of
    the correlator is 80 ms, i.e. three frames. Held frames therefore ask a
    correlator to work on an impulse train aliased against its own delay line.

    A fly in a real room receives continuous motion; the 35 Hz strobe is an
    artifact of the game engine, not of the biology, and it is a confound that
    guarantees failure independently of how the brain is wired. Linear
    interpolation between consecutive frames is the cheapest way to remove it.

    Set False to recover the held-frame behaviour.
    """

    labels: bool = False
    """Enable the object label buffer. Used for MEASUREMENT ONLY -- M6 needs
    ground-truth enemy positions to ask whether the fly responded to them.
    Nothing in the agent's control path may read this; spec 7 forbids a
    hand-coded flee rule and reading enemy positions would be exactly that."""

    @property
    def vfov_deg(self) -> float:
        """Vertical FOV follows from the horizontal one and the aspect ratio."""
        half_h = math.radians(self.fov_deg / 2.0)
        return 2.0 * math.degrees(
            math.atan(math.tan(half_h) * self.height / self.width)
        )


class DoomVision:
    """Doom frame -> per-neuron drive, through the real ommatidial lattice."""

    def __init__(self, retina, cfg: DoomConfig, dt: float, device: str = "cuda"):
        import torch

        self.torch = torch
        self.retina = retina
        self.cfg = cfg
        self.device = device

        # --- where each column looks, in Doom's viewport ---
        # Doom draws a PLANAR PERSPECTIVE projection, so a ray at azimuth
        # `az` lands at screen x proportional to tan(az), not to az. Mapping
        # angle linearly onto the viewport -- which this did until measured --
        # puts every column in the wrong place, and worst near the centre of
        # gaze: a column at 10 deg was sampled 87% too far out, one at 30 deg
        # 71% too far out, converging only at the very edge.
        #
        # That is not a cosmetic error. Retinotopy is the substrate every
        # motion computation here runs on, and a warp of this size means the
        # angular spacing between neighbouring columns varies about twofold
        # across the field. Rigid motion of the world then sweeps the retina at
        # a speed that depends on where you look, which is exactly the input a
        # delay-and-correlate detector tuned to a FIXED spacing cannot use.
        half_h, half_v = cfg.fov_deg / 2.0, cfg.vfov_deg / 2.0
        tan_h = math.tan(math.radians(half_h))
        tan_v = math.tan(math.radians(half_v))
        self.tan_h, self.tan_v = tan_h, tan_v
        self.cameras = (0.0, *[float(o) for o in cfg.side_views])
        if len(self.cameras) > 1:
            if not cfg.gnomonic:
                raise ValueError("side_views needs gnomonic=True")
            if any(o % 90 for o in self.cameras):
                raise ValueError("side_views offsets must be multiples of 90")
        idx = [eye.neuron_idx for eye in retina.eyes.values()
               if eye.neuron_idx.size]
        self.idx = torch.as_tensor(np.concatenate(idx).astype(np.int64),
                                   device=device)
        self.mirrored = False
        self._build_geometry(mirror=False)

        # --- Gaussian acceptance, expressed in pixels ---
        # Under a perspective projection the angular size of a pixel is not
        # uniform: each pixel spans the most angle at the centre and the least
        # at the edge. The flat fov/width figure used before understates the
        # centre by about 1.9x, so the acceptance kernel was roughly twice as
        # wide as intended and blurred away the fine spatial structure the
        # correlator needs.
        # One separable pre-blur cannot vary across the image, so we scale it
        # to the centre, where the resolution actually matters.
        deg_per_px = math.degrees(2.0 * tan_h) / cfg.width
        # interommatidial spacing in degrees, measured off the real lattice
        spacing = self._column_spacing_deg()
        accept_deg = ACCEPTANCE_RATIO * spacing
        # FWHM -> sigma
        self.sigma_px = max(0.6, accept_deg / 2.355 / deg_per_px)
        self.kernel = self._gaussian_kernel(self.sigma_px)

        self.spacing_deg = spacing
        self.accept_deg = accept_deg
        self.deg_per_px = deg_per_px

        # --- adaptation, per driven neuron ---
        from .retina import TAU_ADAPT, CONTRAST_GAIN
        self.adapt_mean = torch.full((self.idx.numel(),), 0.5,
                                     dtype=torch.float32, device=device)
        # MEASURED BUG, fixed here: this decay is per SUBSTEP (dt = 0.5 ms),
        # but drive() used to be called once per Doom tic (28.6 ms), so the
        # Weber adaptation advanced one 0.5 ms step per 28.6 ms of game time
        # and its effective time constant was 57x too long -- 14 s instead of
        # 0.25 s. Stepping it inside the substep loop (see substep_drive) is
        # what makes this constant mean what it says.
        self.adapt_decay = math.exp(-dt / TAU_ADAPT)
        self.dt = dt
        self.lum_prev = None
        self.adapt_gain = CONTRAST_GAIN
        self.luma = LUMA_FLY if cfg.fly_spectrum else LUMA_HUMAN
        self.inverts = retina.inverts
        self.out_buf = None
        # L1/L2 report CHANGE, L3 reports LEVEL. Without the sustained channel
        # a static scene adapts to nothing and the loop cannot start.
        sus = retina.sustained_mask(np.concatenate(idx))
        self.sustained = torch.as_tensor(sus, device=device)
        self.n_sustained = int(sus.sum())

    def _build_geometry(self, mirror: bool) -> None:
        """Where each lens looks: camera, picture coordinates, blur level.

        `mirror` negates every lens azimuth, so each lens reads the direction
        its mirror image would. With one camera that is exactly a flip of the
        picture x coordinate; with side cameras a lens can change camera.
        """
        torch, cfg, device = self.torch, self.cfg, self.device
        tan_h, tan_v = self.tan_h, self.tan_v
        half_h, half_v = cfg.fov_deg / 2.0, cfg.vfov_deg / 2.0
        cams = np.asarray(self.cameras)
        gx, gy, inside, rr, cam = [], [], [], [], []
        for side, eye in self.retina.eyes.items():
            if not eye.neuron_idx.size:
                continue
            gaze = -cfg.splay_deg if side == "left" else cfg.splay_deg
            az = eye.azimuth_deg[eye.neuron_column] + gaze
            el = eye.elevation_deg[eye.neuron_column]
            if mirror:
                az = -az
            # each lens reads the camera nearest its own direction; the main
            # camera wins ties, so a single camera reproduces the old mapping
            off = (az[:, None] - cams[None, :] + 180.0) % 360.0 - 180.0
            k = np.argmin(np.abs(off) + 1e-6 * (cams[None, :] != 0), axis=1)
            # a lone front camera keeps the raw azimuth, so the legacy
            # mapping stays bit-identical
            az_l = off[np.arange(len(az)), k] if np.any(cams != 0) else az
            cam.append(k)
            if cfg.gnomonic:
                front = np.abs(az_l) < 89.9
                azc = np.clip(az_l, -89.9, 89.9)
                u = np.tan(np.radians(azc))
                v = np.tan(np.radians(el)) / np.cos(np.radians(azc))
                gx.append(u / tan_h)
                gy.append(-v / tan_v)                   # screen y grows down
                inside.append(front & (np.abs(u) <= tan_h)
                              & (np.abs(v) <= tan_v))
                rr.append(u * u + v * v)
            else:
                gx.append(np.tan(np.radians(az)) / tan_h)
                gy.append(-np.tan(np.radians(el)) / tan_v)
                inside.append((np.abs(az) <= half_h) & (np.abs(el) <= half_v))
                rr.append(np.zeros_like(az))
        x = np.concatenate(gx).astype(np.float32)
        y = np.concatenate(gy).astype(np.float32)
        cam = np.concatenate(cam).astype(np.int64)
        self.inside = torch.as_tensor(np.concatenate(inside), device=device)
        self.cam = torch.as_tensor(cam, device=device)
        # grid_sample wants [N, H_out, W_out, 2]; one row per camera, holding
        # every lens, so a lens is read from its own camera's row
        xy = np.stack([x, y], axis=-1)[None, None]
        self.grid = torch.as_tensor(
            np.repeat(xy, len(self.cameras), axis=0), dtype=torch.float32,
            device=device,
        ).clamp(-1.0, 1.0)
        self.n_inside = int(self.inside.sum())
        self.n_total = int(self.inside.numel())
        self.per_camera = np.bincount(cam, minlength=len(self.cameras))

        # Per-lens blur scale. With r^2 = u^2 + v^2 in focal units, the
        # rectilinear projection magnifies angle by (1 + r^2) radially and
        # sqrt(1 + r^2) tangentially, so the geometric-mean blur a lens needs
        # grows as (1 + r^2)^(3/4) relative to the centre. Level k of the
        # pyramid carries an effective blur of sigma_px * 2^k.
        self.pyr_levels = 1
        if cfg.pyramid_blur:
            r2 = np.concatenate(rr).astype(np.float64)
            # Lenses outside the view read the mean fill, so they must not set
            # the pyramid depth: a lens at 89.9 deg azimuth would ask for 18.
            r2 = np.where(np.concatenate(inside), r2, 0.0)
            need = np.log2((1.0 + r2) ** 0.75)          # in pyramid levels
            top = max(1, int(np.ceil(need.max())) + 1)
            lo = np.clip(np.floor(need), 0, top - 1).astype(np.int64)
            hi = np.clip(lo + 1, 0, top - 1)
            frac = np.clip(need - lo, 0.0, 1.0).astype(np.float32)
            self.pyr_levels = top
            self.pyr_lo = torch.as_tensor(lo, device=device)
            self.pyr_hi = torch.as_tensor(hi, device=device)
            self.pyr_frac = torch.as_tensor(frac, device=device)

    def mirror(self) -> None:
        """CONTROL: every lens reads its mirror-image direction.

        Reverses the sign of all horizontal optic flow while leaving rates,
        contrast statistics and wiring untouched. With a single camera this is
        the old flip of the sampling grid, bit for bit.
        """
        if len(self.cameras) == 1:
            self.grid[..., 0] = -self.grid[..., 0]
        else:
            self._build_geometry(mirror=not self.mirrored)
        self.mirrored = not self.mirrored

    def _column_spacing_deg(self) -> float:
        eye = next(iter(self.retina.eyes.values()))
        cx, cy = eye.cartesian()
        span_units = max(cx.max() - cx.min(), 1e-9)
        from .retina import EYE_FOV_AZIMUTH_DEG
        return EYE_FOV_AZIMUTH_DEG / span_units

    def _gaussian_kernel(self, sigma: float):
        torch = self.torch
        radius = max(1, int(round(3 * sigma)))
        t = torch.arange(-radius, radius + 1, dtype=torch.float32,
                         device=self.device)
        k = torch.exp(-0.5 * (t / sigma) ** 2)
        return (k / k.sum()).view(1, 1, -1)

    # -- the hot path ----------------------------------------------------

    def _weight(self, img):
        """Linear RGB -> scalar absorbed flux, under the active spectrum."""
        return (img * self.torch.as_tensor(self.luma,
                                           device=self.device)).sum(-1)

    def sample(self, frame: np.ndarray):
        """RGB frame -> per-column luminance in [0, 1], mean-filled outside.

        `frame` is one picture [H, W, 3], or one per camera [K, H, W, 3] in
        the order of `self.cameras` when side views are on.
        """
        torch = self.torch
        import torch.nn.functional as F

        img = torch.as_tensor(frame, device=self.device, dtype=torch.float32)
        if img.ndim == 3 and img.shape[0] == 3:      # CHW
            img = img.permute(1, 2, 0)
        img = img / 255.0
        if self.cfg.linearise_gamma or self.cfg.repaint == "flyband":
            # sRGB EOTF. Cheap, and it has to precede the weighting, the
            # repaint and the blur alike -- see DoomConfig.linearise_gamma.
            img = torch.where(img <= 0.04045, img / 12.92,
                              ((img + 0.055) / 1.055) ** 2.4)
        if self.cfg.repaint == "flyband":
            # Achromatic structure of the scene...
            y = (img * torch.as_tensor(LUMA_HUMAN,
                                       device=self.device)).sum(-1, keepdim=True)
            # ...re-emitted in blue-green, normalised so total absorbed flux is
            # preserved rather than quietly rescaled.
            w = torch.as_tensor(FLYBAND_MIX, device=self.device)
            img = (y * w / float((FLYBAND_MIX * LUMA_FLY).sum())).clamp(0.0, 1.0)
        lum = self._weight(img)
        # [K,1,H,W]: one picture per camera
        lum = lum[None, None] if lum.ndim == 2 else lum[:, None]

        # separable Gaussian = the acceptance function
        k = self.kernel
        pad = k.shape[-1] // 2
        lum = F.conv2d(F.pad(lum, (pad, pad, 0, 0), mode="replicate"),
                       k.view(1, 1, 1, -1))
        lum = F.conv2d(F.pad(lum, (0, 0, pad, pad), mode="replicate"),
                       k.view(1, 1, -1, 1))

        if self.pyr_levels > 1:
            # Level 0 is the centre-sized blur above. Each further level
            # halves resolution after a mild blur, doubling the effective
            # acceptance width. grid_sample takes normalised coordinates, so
            # the same lens grid reads every level.
            levels = [lum]
            cur = lum
            k1 = self._gaussian_kernel(1.0)
            p1 = k1.shape[-1] // 2
            for _ in range(self.pyr_levels - 1):
                cur = F.conv2d(F.pad(cur, (p1, p1, 0, 0), mode="replicate"),
                               k1.view(1, 1, 1, -1))
                cur = F.conv2d(F.pad(cur, (0, 0, p1, p1), mode="replicate"),
                               k1.view(1, 1, -1, 1))
                cur = F.avg_pool2d(cur, 2, ceil_mode=True)
                levels.append(cur)
            ar = torch.arange(self.cam.numel(), device=self.device)
            per = torch.stack([
                F.grid_sample(L, self.grid, mode="bilinear",
                              padding_mode="border", align_corners=False)
                .view(len(levels[0]), -1)[self.cam, ar]
                for L in levels])                      # [levels, lenses]
            a = per[self.pyr_lo, ar]
            b = per[self.pyr_hi, ar]
            sampled = a + (b - a) * self.pyr_frac
        else:
            sampled = F.grid_sample(lum, self.grid, mode="bilinear",
                                    padding_mode="border", align_corners=False)
            if len(self.cameras) == 1:
                sampled = sampled.view(-1)
            else:
                ar = torch.arange(self.cam.numel(), device=self.device)
                sampled = sampled.view(len(self.cameras), -1)[self.cam, ar]

        # Columns outside Doom's viewport see the frame's MEAN, not black. A
        # dark surround is a permanent high-contrast edge at a fixed
        # retinotopic position and the looming detectors read it as an object.
        mean = float(sampled[self.inside].mean()) if self.n_inside else 0.5
        return torch.where(self.inside, sampled, torch.full_like(sampled, mean))

    def reset(self) -> None:
        self.adapt_mean.fill_(0.5)
        self.lum_prev = None

    def begin_tic(self, frame: np.ndarray):
        """Sample a new frame and return (previous, current) column luminance.

        The pair is what substep_drive interpolates between. On the first tic
        of an episode there is no previous frame, so the ramp starts flat.
        """
        lum = self.sample(frame)
        prev = self.lum_prev if self.lum_prev is not None else lum
        self.lum_prev = lum
        return prev, lum

    def _out_set(self, lum, n_neurons: int, graded_max_rate: float,
                 dt: float, adapt: bool = True):
        torch = self.torch
        if adapt:
            self.adapt_mean.mul_(self.adapt_decay).add_(
                lum * (1 - self.adapt_decay))
        c = ((lum - self.adapt_mean) / self.adapt_mean.clamp(min=1e-3)
             * self.adapt_gain).clamp(-1.0, 1.0)
        if self.inverts:
            c = -c
        act = (0.5 + 0.5 * c).clamp(0.0, 1.0)
        # sustained lines bypass adaptation and code absolute level
        level = (1.0 - lum) if self.inverts else lum
        act = torch.where(self.sustained, level.clamp(0.0, 1.0), act)
        if self.out_buf is None or self.out_buf.numel() != n_neurons:
            self.out_buf = torch.full((n_neurons,), -1.0, dtype=torch.float32,
                                      device=self.device)
        self.out_buf.fill_(-1.0)
        self.out_buf[self.idx] = act * (graded_max_rate * dt)
        return self.out_buf

    def substep_drive(self, prev, cur, alpha: float, n_neurons: int,
                      graded_max_rate: float, dt: float):
        """out_set for one substep, `alpha` of the way from prev to cur.

        Adaptation is stepped here rather than once per tic, so TAU_ADAPT is
        finally expressed in the units it is written in.
        """
        lum = prev if alpha <= 0.0 else (
            cur if alpha >= 1.0 else self.torch.lerp(prev, cur, alpha))
        return self._out_set(lum, n_neurons, graded_max_rate, dt), lum

    def drive(self, frame: np.ndarray, n_neurons: int, graded_max_rate: float,
              dt: float):
        """Frame -> out_set, held for the whole tic. The pre-interpolation path.

        Kept because the open-loop experiments hold a frame deliberately: M8
        freezes the agent so both arms see an identical scene, and interpolating
        between two identical frames would only add a redundant adaptation step.
        """
        lum = self.sample(frame)
        self.lum_prev = lum
        return self._out_set(lum, n_neurons, graded_max_rate, dt), lum

    def summary(self) -> str:
        return (
            f"Doom viewport {self.cfg.width}x{self.cfg.height} at "
            f"{self.cfg.fov_deg:.0f}x{self.cfg.vfov_deg:.0f} deg\n"
            f"  {self.deg_per_px:.3f} deg/pixel; ommatidial spacing "
            f"{self.spacing_deg:.2f} deg; acceptance {self.accept_deg:.2f} deg "
            f"(sigma {self.sigma_px:.1f} px)\n"
            f"  {self.n_inside:,} of {self.n_total:,} columns fall inside the "
            f"viewport ({100 * self.n_inside / max(self.n_total, 1):.0f}%); "
            f"the rest see mean luminance\n"
            f"  eyes splayed +/-{self.cfg.splay_deg:.0f} deg, so the two eyes "
            f"see genuinely different views\n"
            f"  {self.n_sustained:,} sustained (L3) and "
            f"{self.idx.numel() - self.n_sustained:,} transient (L1/L2) inputs"
        )


class DoomSession:
    """Thin, explicit wrapper over ViZDoom. No gym, no hidden state."""

    BUTTONS = [
        "TURN_LEFT_RIGHT_DELTA",
        "MOVE_FORWARD_BACKWARD_DELTA",
        "MOVE_LEFT_RIGHT_DELTA",
        "ATTACK",
        "USE",
    ]

    def __init__(self, cfg: DoomConfig) -> None:
        import vizdoom as vzd

        self.vzd = vzd
        self.cfg = cfg
        g = vzd.DoomGame()
        scen_dir = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
        path = os.path.join(scen_dir, f"{cfg.scenario}.cfg")
        # Arena variants built by this project (see flydoom/wads) take
        # precedence, so a modified map is selected by name like any other.
        local = os.path.join(os.path.dirname(__file__), "wads",
                             f"{cfg.scenario}.cfg")
        if os.path.exists(local):
            path = local
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no scenario {cfg.scenario!r} in {scen_dir}; "
                f"try one of {sorted(f[:-4] for f in os.listdir(scen_dir) if f.endswith('.cfg'))[:8]}"
            )
        g.load_config(path)
        g.set_window_visible(cfg.window)
        g.set_render_weapon(cfg.render_weapon)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_screen_resolution(self._resolution(vzd, cfg.width, cfg.height))
        g.set_mode(vzd.Mode.PLAYER)
        g.clear_available_buttons()
        for b in self.BUTTONS:
            g.add_available_button(getattr(vzd.Button, b))
        g.set_available_game_variables([
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.DAMAGECOUNT,
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.ANGLE,
        ])
        if cfg.labels:
            g.set_labels_buffer_enabled(True)
        # A wide FOV is not cosmetic: at Doom's default 90 deg the fly sees a
        # sliver of its own visual field, and every angular claim downstream
        # inherits this number.
        #
        # MEASURED, and this silently invalidated every angular claim until
        # caught: `add_game_args("+fov N")` DOES NOTHING here. Renders at
        # +fov 90, +fov 130 and +fov 160 are bit-identical, so the game was
        # running at Doom's default 90 deg while every module assumed 130. The
        # console command does take effect, and has to be re-sent per episode
        # because a new episode resets the CVar -- see new_episode().
        if cfg.seed is not None:
            g.set_seed(cfg.seed)
        if cfg.living_reward is not None:
            g.set_living_reward(cfg.living_reward)
        g.init()
        self.game = g
        self._apply_fov()
        self._last_health = 100.0
        self._last_damage = 0.0
        self.sides = SideCameras(cfg) if cfg.side_views else None

    @staticmethod
    def _resolution(vzd, w: int, h: int):
        name = f"RES_{w}X{h}"
        if not hasattr(vzd.ScreenResolution, name):
            raise ValueError(
                f"ViZDoom has no resolution {w}x{h}; try 320x240 or 640x480"
            )
        return getattr(vzd.ScreenResolution, name)

    # -- episode ---------------------------------------------------------

    def _apply_fov(self) -> None:
        """Set the horizontal FOV and the head-bob. Must follow every
        new_episode()."""
        self.game.send_game_command(f"fov {self.cfg.fov_deg:.0f}")
        self.game.send_game_command(f"movebob {self.cfg.view_bob:g}")

    def new_episode(self) -> None:
        self.game.new_episode()
        self._apply_fov()
        self._last_health = self.health
        self._last_damage = 0.0
        if self.sides is not None:
            self.sides.new_episode()

    @property
    def finished(self) -> bool:
        return self.game.is_episode_finished()

    @property
    def health(self) -> float:
        return float(self.game.get_game_variable(self.vzd.GameVariable.HEALTH))

    @property
    def kills(self) -> float:
        return float(self.game.get_game_variable(self.vzd.GameVariable.KILLCOUNT))

    def frame(self) -> np.ndarray | None:
        """The picture, [H, W, 3]; with side views, [K, H, W, 3] with the
        main view first and the rest in DoomConfig.side_views order."""
        s = self.game.get_state()
        if s is None:
            return None
        if self.sides is None:
            return s.screen_buffer
        return np.stack([s.screen_buffer, *self.sides.frames(s.screen_buffer)])

    # -- ground truth, for MEASUREMENT ONLY ------------------------------

    ENEMY_HALF_WIDTH = 20.0
    """Doom map units. Monster radii cluster around this; it only sets the
    absolute scale of the angular-size numbers, not their ordering."""

    def pose(self) -> tuple[float, float, float]:
        """Player position and heading: (x, y, angle in degrees)."""
        vz = self.vzd
        return (float(self.game.get_game_variable(vz.GameVariable.POSITION_X)),
                float(self.game.get_game_variable(vz.GameVariable.POSITION_Y)),
                float(self.game.get_game_variable(vz.GameVariable.ANGLE)))

    def threats(self) -> list[dict]:
        """Enemies visible this tic, with true distance and angular size.

        NOT available to the agent. M6 uses this to ask whether the fly
        responded to a threat; letting it reach the control path would be the
        hand-coded flee rule spec 7 forbids.
        """
        s = self.game.get_state()
        if s is None or not getattr(s, "labels", None):
            return []
        vz = self.vzd
        px = self.game.get_game_variable(vz.GameVariable.POSITION_X)
        py = self.game.get_game_variable(vz.GameVariable.POSITION_Y)
        ang = math.radians(self.game.get_game_variable(vz.GameVariable.ANGLE))
        out = []
        for lab in s.labels:
            if lab.object_name in ("DoomPlayer", "BulletPuff", "Blood"):
                continue
            dx = lab.object_position_x - px
            dy = lab.object_position_y - py
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                continue
            # azimuth relative to gaze, positive = to the player's left
            rel = math.degrees(math.atan2(dy, dx)) - math.degrees(ang)
            rel = (rel + 180.0) % 360.0 - 180.0
            out.append({
                "name": lab.object_name,
                "distance": dist,
                "azimuth_deg": rel,
                "half_size_deg": math.degrees(
                    math.atan(self.ENEMY_HALF_WIDTH / dist)
                ),
            })
        return sorted(out, key=lambda t: t["distance"])

    def step(self, action: list[float], tics: int = 1) -> dict:
        """Apply one action and report what changed."""
        if self.sides is not None:
            self.sides.step(self, action, tics)
        else:
            self.game.make_action(action, tics)
        done = self.finished
        health = self._last_health if done else self.health
        delta_health = health - self._last_health
        self._last_health = health
        return {
            "done": done,
            "health": health,
            "damage_taken": max(0.0, -delta_health),
            "healed": max(0.0, delta_health),
        }

    def close(self) -> None:
        if self.sides is not None:
            self.sides.close()
        try:
            self.game.close()
        except Exception:
            pass


class SideCameras:
    """Side views of the main game's world. See DoomConfig.side_views.

    HOW, since Doom draws one view per tic and the camera is the heading:
    every tic the main game writes a save, and each side engine loads it and
    takes one step with the fly's action turned by its offset. What it draws
    is therefore the main world one tic on, seen sideways. Three engine
    behaviours, each measured, shape the bookkeeping:

      * The console `save` lands late, only every other tic, and holds the
        world as it was one tic before the step that wrote it. So the side
        engine replays the fly's last one or two actions before its own.
      * `load()` runs one tic itself, with whatever action was set last.
        Setting the next action first turns that tic into a useful one.
      * Movement commands are floored to whole units, so the quarter turn is
        done as an exact swap of whole numbers, never with trigonometry.
        Turns are exact at any value.

    With all three handled, an offset of 0 reproduces the main game pixel for
    pixel over a whole episode.
    """

    def __init__(self, cfg: DoomConfig) -> None:
        import tempfile
        import uuid
        from dataclasses import replace

        self.offsets = [float(o) for o in cfg.side_views]
        sub = replace(cfg, side_views=(), labels=False, window=False)
        self.cams = [DoomSession(sub) for _ in self.offsets]
        base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        self.stem = os.path.join(base, f"flydoom_{os.getpid()}_"
                                       f"{uuid.uuid4().hex[:8]}")
        B = DoomSession.BUTTONS
        self.iy = B.index("TURN_LEFT_RIGHT_DELTA")
        self.ifw = B.index("MOVE_FORWARD_BACKWARD_DELTA")
        self.ilat = B.index("MOVE_LEFT_RIGHT_DELTA")
        self.iact = [B.index("ATTACK"), B.index("USE")]
        self.episode = 0
        self.loads = self.relabels = self.misses = 0
        self._reset()

    def _reset(self) -> None:
        # A save still queued from the previous episode would land after the
        # reset and carry the old world, so each episode writes its own file.
        self.path = f"{self.stem}_{self.episode}.zds"
        self.hist: dict[int, list] = {}     # time label -> action that made it
        self.saved = None                   # which world state the file holds
        self.sig = None
        self.turned = [False] * len(self.cams)
        self.broken = [False] * len(self.cams)
        self.last = [None] * len(self.cams)

    def new_episode(self) -> None:
        for c in self.cams:
            c.new_episode()
        self._remove(self.path)
        self.episode += 1
        self._reset()

    def _rotate(self, a: list, deg: float) -> list:
        """The same push on the world, from a body turned `deg` to the right."""
        b = list(a)
        f, s = a[self.ifw], a[self.ilat]
        q = int(round(deg / 90.0)) % 4
        # a quarter turn right leaves the old forward pointing left
        b[self.ifw], b[self.ilat] = [(f, s), (s, -f), (-f, -s), (-s, f)][q]
        b[self.iy] = a[self.iy] + deg
        for i in self.iact:      # a shot or a door opened sideways is not ours
            b[i] = 0.0
        return b

    def _load(self, cam, first: list) -> int:
        cam.game.set_action(first)
        cam.game.load(self.path)
        self.loads += 1
        return int(cam.game.get_episode_time()) - 1

    def _render(self, k: int, cam, deg: float, a: list, tics: int,
                now: int) -> None:
        rot = self._rotate(a, deg)

        def plan(f):
            if f is None or f > now or any(t not in self.hist
                                           for t in range(f + 1, now + 1)):
                return None
            return [self.hist[t] for t in range(f + 1, now + 1)]

        replay = plan(self.saved)
        if replay is None:
            if self.broken[k] or self.saved is not None:
                self.misses += 1
                return                  # keep the last good picture
            # Before the first save lands: the side engine plays its own copy
            # of the episode, from the same seed, turned once.
            r = list(rot)
            if self.turned[k]:
                r[self.iy] = a[self.iy]
            cam.game.make_action(r, tics)
            self.turned[k] = True
        else:
            seq = replay + [rot] * tics
            try:
                got = self._load(cam, seq[0])
                if got != self.saved:
                    self.relabels += 1
                    self.saved = got
                    replay = plan(got)
                    if replay is None:
                        self.broken[k] = True
                        self.misses += 1
                        return
                    seq = replay + [rot] * tics
                    self._load(cam, seq[0])
            except Exception:
                self.broken[k] = True
                self.misses += 1
                return
            self.broken[k] = False
            rest = seq[1:]
            n_plain = len(replay) - 1 if replay else 0
            for b in rest[:max(n_plain, 0)]:
                cam.game.make_action(b, 1)
            if len(rest) > max(n_plain, 0):
                cam.game.make_action(rot, len(rest) - max(n_plain, 0))
        f = cam.frame()
        if f is not None:
            self.last[k] = f

    def step(self, main: "DoomSession", action: list, tics: int) -> list:
        a = list(action)
        for i in (self.ifw, self.ilat):
            a[i] = float(math.floor(a[i]))   # what the engine does anyway
        now = int(main.game.get_episode_time())
        for k, (cam, deg) in enumerate(zip(self.cams, self.offsets)):
            self._render(k, cam, deg, a, tics, now)
        main.game.send_game_command(f"save {self.path}")
        main.game.make_action(a, tics)
        after = int(main.game.get_episode_time())
        for t in range(now + 1, after + 1):
            self.hist[t] = a
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return a
        sig = (st.st_mtime_ns, st.st_size, st.st_ino)
        if sig != self.sig:
            self.sig = sig
            self.saved = after - 1
        return a

    def frames(self, like: np.ndarray) -> list:
        return [f if f is not None else np.zeros_like(like) for f in self.last]

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    def close(self) -> None:
        import glob

        for c in self.cams:
            c.close()
        for f in glob.glob(self.stem + "_*.zds"):
            self._remove(f)
