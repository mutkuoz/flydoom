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
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


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
        idx, gx, gy, inside = [], [], [], []
        half_h, half_v = cfg.fov_deg / 2.0, cfg.vfov_deg / 2.0
        for side, eye in retina.eyes.items():
            if not eye.neuron_idx.size:
                continue
            gaze = -cfg.splay_deg if side == "left" else cfg.splay_deg
            az = eye.azimuth_deg[eye.neuron_column] + gaze
            el = eye.elevation_deg[eye.neuron_column]
            idx.append(eye.neuron_idx)
            gx.append(az / half_h)
            gy.append(-el / half_v)        # screen y grows downward
            inside.append((np.abs(az) <= half_h) & (np.abs(el) <= half_v))
        self.idx = torch.as_tensor(np.concatenate(idx).astype(np.int64),
                                   device=device)
        x = np.concatenate(gx).astype(np.float32)
        y = np.concatenate(gy).astype(np.float32)
        self.inside = torch.as_tensor(np.concatenate(inside), device=device)
        # grid_sample wants [N, H_out, W_out, 2]; one row of samples is enough
        self.grid = torch.as_tensor(
            np.stack([x, y], axis=-1)[None, None], dtype=torch.float32,
            device=device,
        ).clamp(-1.0, 1.0)

        # --- Gaussian acceptance, expressed in pixels ---
        deg_per_px = cfg.fov_deg / cfg.width
        # interommatidial spacing in degrees, measured off the real lattice
        spacing = self._column_spacing_deg()
        accept_deg = ACCEPTANCE_RATIO * spacing
        # FWHM -> sigma
        self.sigma_px = max(0.6, accept_deg / 2.355 / deg_per_px)
        self.kernel = self._gaussian_kernel(self.sigma_px)

        self.n_inside = int(self.inside.sum())
        self.n_total = int(self.inside.numel())
        self.spacing_deg = spacing
        self.accept_deg = accept_deg
        self.deg_per_px = deg_per_px

        # --- adaptation, per driven neuron ---
        from .retina import TAU_ADAPT, CONTRAST_GAIN
        self.adapt_mean = torch.full((self.idx.numel(),), 0.5,
                                     dtype=torch.float32, device=device)
        self.adapt_decay = math.exp(-dt / TAU_ADAPT)
        self.adapt_gain = CONTRAST_GAIN
        self.inverts = retina.inverts
        self.out_buf = None
        # L1/L2 report CHANGE, L3 reports LEVEL. Without the sustained channel
        # a static scene adapts to nothing and the loop cannot start.
        sus = retina.sustained_mask(np.concatenate(idx))
        self.sustained = torch.as_tensor(sus, device=device)
        self.n_sustained = int(sus.sum())

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

    def reset(self) -> None:
        self.adapt_mean.fill_(0.5)

    # -- the hot path ----------------------------------------------------

    def sample(self, frame: np.ndarray):
        """RGB frame -> per-column luminance in [0, 1], mean-filled outside."""
        torch = self.torch
        import torch.nn.functional as F

        img = torch.as_tensor(frame, device=self.device, dtype=torch.float32)
        if img.ndim == 3 and img.shape[0] == 3:      # CHW
            img = img.permute(1, 2, 0)
        lum = (img * torch.as_tensor(LUMA, device=self.device)).sum(-1) / 255.0
        lum = lum[None, None]                          # [1,1,H,W]

        # separable Gaussian = the acceptance function
        k = self.kernel
        pad = k.shape[-1] // 2
        lum = F.conv2d(F.pad(lum, (pad, pad, 0, 0), mode="replicate"),
                       k.view(1, 1, 1, -1))
        lum = F.conv2d(F.pad(lum, (0, 0, pad, pad), mode="replicate"),
                       k.view(1, 1, -1, 1))

        sampled = F.grid_sample(lum, self.grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
        sampled = sampled.view(-1)

        # Columns outside Doom's viewport see the frame's MEAN, not black. A
        # dark surround is a permanent high-contrast edge at a fixed
        # retinotopic position and the looming detectors read it as an object.
        mean = float(sampled[self.inside].mean()) if self.n_inside else 0.5
        return torch.where(self.inside, sampled, torch.full_like(sampled, mean))

    def drive(self, frame: np.ndarray, n_neurons: int, graded_max_rate: float,
              dt: float):
        """Frame -> out_set vector for a graded input population."""
        torch = self.torch
        lum = self.sample(frame)
        self.adapt_mean.mul_(self.adapt_decay).add_(lum * (1 - self.adapt_decay))
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
        return self.out_buf, lum

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
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no scenario {cfg.scenario!r} in {scen_dir}; "
                f"try one of {sorted(f[:-4] for f in os.listdir(scen_dir) if f.endswith('.cfg'))[:8]}"
            )
        g.load_config(path)
        g.set_window_visible(cfg.window)
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
        g.add_game_args(f"+fov {cfg.fov_deg:.0f}")
        if cfg.seed is not None:
            g.set_seed(cfg.seed)
        if cfg.living_reward is not None:
            g.set_living_reward(cfg.living_reward)
        g.init()
        self.game = g
        self._last_health = 100.0
        self._last_damage = 0.0

    @staticmethod
    def _resolution(vzd, w: int, h: int):
        name = f"RES_{w}X{h}"
        if not hasattr(vzd.ScreenResolution, name):
            raise ValueError(
                f"ViZDoom has no resolution {w}x{h}; try 320x240 or 640x480"
            )
        return getattr(vzd.ScreenResolution, name)

    # -- episode ---------------------------------------------------------

    def new_episode(self) -> None:
        self.game.new_episode()
        self._last_health = self.health
        self._last_damage = 0.0

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
        s = self.game.get_state()
        return None if s is None else s.screen_buffer

    # -- ground truth, for MEASUREMENT ONLY ------------------------------

    ENEMY_HALF_WIDTH = 20.0
    """Doom map units. Monster radii cluster around this; it only sets the
    absolute scale of the angular-size numbers, not their ordering."""

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
        try:
            self.game.close()
        except Exception:
            pass
