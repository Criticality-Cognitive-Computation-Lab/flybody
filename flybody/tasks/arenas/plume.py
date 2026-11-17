"""Static 2D odor plume generation and arena utilities.

Inspired by NeuroMechFly/FlyGym's olfaction design where olfactory sensors
sample an external odor field instead of relying on MuJoCo-native gas physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from dm_control import composer

from flybody.tasks.arenas.olfaction import PLUME_ODOR_NAMES


@dataclass(frozen=True)
class OdorPlume2DConfig:
    arena_size: tuple[float, float] = (1.2, 1.2)
    grid_shape: tuple[int, int] = (220, 220)  # (ny, nx)
    source_y: float = -0.52
    source_xs: tuple[float, ...] = (-0.40, -0.20, 0.0, 0.20, 0.40)
    peak_intensity: float = 12.0
    base_sigma: float = 0.035
    spread_rate: float = 0.18
    decay_length: float = 0.95
    meander_amplitude: float = 0.08
    meander_wavelength: float = 0.50
    turbulence_scale: float = 0.30
    random_seed: int = 7


def _smooth_noise(field: np.ndarray, steps: int = 10) -> np.ndarray:
    """Cheap periodic smoothing without external deps."""
    smoothed = field.astype(np.float32, copy=True)
    for _ in range(steps):
        smoothed = (
            smoothed
            + np.roll(smoothed, 1, axis=0)
            + np.roll(smoothed, -1, axis=0)
            + np.roll(smoothed, 1, axis=1)
            + np.roll(smoothed, -1, axis=1)
        ) / 5.0
    return smoothed


def generate_static_plume_2d(
    config: OdorPlume2DConfig,
    odor_names: Sequence[str] = PLUME_ODOR_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a simple static 2D odor plume field.

    Returns:
        plume_field: (n_odors, ny, nx)
        x_coords: (nx,)
        y_coords: (ny,)
    """
    odor_names = tuple(odor_names)
    nx = config.grid_shape[1]
    ny = config.grid_shape[0]
    x_coords = np.linspace(-config.arena_size[0], config.arena_size[0], nx, dtype=np.float32)
    y_coords = np.linspace(-config.arena_size[1], config.arena_size[1], ny, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    rng = np.random.RandomState(config.random_seed)

    plume_field = np.zeros((len(odor_names), ny, nx), dtype=np.float32)

    for odor_idx, odor_name in enumerate(odor_names):
        del odor_name  # Name is mainly for indexing / diagnostics.
        source_x = config.source_xs[odor_idx]
        downstream = y_grid - config.source_y
        active_mask = downstream > 0.0

        phase = 0.7 * odor_idx
        centerline_x = source_x + config.meander_amplitude * np.sin(
            2.0 * np.pi * downstream / config.meander_wavelength + phase
        )
        sigma = config.base_sigma + config.spread_rate * np.sqrt(np.maximum(downstream, 0.0))
        crosswind = np.exp(-0.5 * np.square((x_grid - centerline_x) / (sigma + 1e-6)))
        downwind_decay = np.exp(-np.maximum(downstream, 0.0) / config.decay_length)

        raw_noise = rng.normal(size=(ny, nx)).astype(np.float32)
        smooth_noise = _smooth_noise(raw_noise, steps=12)
        smooth_noise -= smooth_noise.min()
        smooth_noise /= smooth_noise.max() + 1e-6
        turbulence = 1.0 + config.turbulence_scale * (2.0 * smooth_noise - 1.0)

        odor_field = config.peak_intensity * crosswind * downwind_decay * turbulence
        odor_field *= active_mask
        plume_field[odor_idx] = np.clip(odor_field, 0.0, None)

    return plume_field.astype(np.float32), x_coords, y_coords


class PlumeNavigationArena2D(composer.Arena):
    """Walking arena with a static 2D multi-odor plume field."""

    def _build(
        self,
        plume_config: OdorPlume2DConfig | None = None,
        odor_names: Sequence[str] = PLUME_ODOR_NAMES,
        name: str = "plume_navigation_arena_2d",
    ):
        super()._build(name=name)
        self._plume_config = plume_config or OdorPlume2DConfig()
        self._odor_names = tuple(odor_names)
        self._plume_field, self._x_coords, self._y_coords = generate_static_plume_2d(
            self._plume_config,
            odor_names=self._odor_names,
        )

        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4],
            diffuse=[0.8, 0.8, 0.8],
            specular=[0.1, 0.1, 0.1],
        )
        self._ground_texture = self._mjcf_root.asset.add(
            "texture",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            type="2d",
            builtin="checker",
            name="groundplane",
            width=300,
            height=300,
        )
        self._ground_material = self._mjcf_root.asset.add(
            "material",
            name="groundplane",
            texrepeat=[4, 4],
            texuniform=False,
            reflectance=0.2,
            texture=self._ground_texture,
        )
        self._ground_geom = self._mjcf_root.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=(self._plume_config.arena_size[0], self._plume_config.arena_size[1], 0.1),
            material=self._ground_material,
        )
        self._mjcf_root.worldbody.add(
            "camera",
            name="top_close",
            mode="fixed",
            pos=(0.0, 0.0, 1.8),
            xyaxes=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )

        marker_rgba = (
            (0.95, 0.25, 0.25, 0.85),
            (0.95, 0.60, 0.20, 0.85),
            (0.85, 0.85, 0.20, 0.85),
            (0.30, 0.80, 0.35, 0.85),
            (0.25, 0.55, 0.95, 0.85),
        )
        for idx, source_x in enumerate(self._plume_config.source_xs[: len(self._odor_names)]):
            body = self._mjcf_root.worldbody.add(
                "body",
                name=f"plume_source_{idx}",
                pos=(source_x, self._plume_config.source_y, 0.02),
            )
            body.add(
                "site",
                name=f"plume_source_marker_{idx}",
                type="sphere",
                size=(0.025,),
                rgba=marker_rgba[idx % len(marker_rgba)],
            )

    @property
    def ground_geoms(self):
        return (self._ground_geom,)

    @property
    def odor_names(self) -> tuple[str, ...]:
        return self._odor_names

    @property
    def odor_dimensions(self) -> int:
        return len(self._odor_names)

    @property
    def plume_field(self) -> np.ndarray:
        return self._plume_field

    @property
    def plume_grid(self) -> tuple[np.ndarray, np.ndarray]:
        return self._x_coords, self._y_coords

    def regenerate(self, random_state):
        del random_state

    def get_olfaction(self, sensor_pos: np.ndarray) -> np.ndarray:
        """Sample the static 2D plume using bilinear interpolation."""
        sensor_pos = np.asarray(sensor_pos, dtype=np.float32)
        x = sensor_pos[:, 0]
        y = sensor_pos[:, 1]

        x_float = np.interp(x, self._x_coords, np.arange(len(self._x_coords), dtype=np.float32))
        y_float = np.interp(y, self._y_coords, np.arange(len(self._y_coords), dtype=np.float32))

        x0 = np.floor(x_float).astype(np.int32)
        y0 = np.floor(y_float).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, len(self._x_coords) - 1)
        y1 = np.clip(y0 + 1, 0, len(self._y_coords) - 1)
        x0 = np.clip(x0, 0, len(self._x_coords) - 1)
        y0 = np.clip(y0, 0, len(self._y_coords) - 1)

        wx = (x_float - x0).astype(np.float32)
        wy = (y_float - y0).astype(np.float32)

        f00 = self._plume_field[:, y0, x0]
        f01 = self._plume_field[:, y1, x0]
        f10 = self._plume_field[:, y0, x1]
        f11 = self._plume_field[:, y1, x1]

        sampled = (
            (1.0 - wx) * (1.0 - wy) * f00
            + (1.0 - wx) * wy * f01
            + wx * (1.0 - wy) * f10
            + wx * wy * f11
        )
        return sampled.astype(np.float32)


class HDF5PlumeDataset2D:
    """Lazy reader for a time-varying 2D plume HDF5 dataset."""

    def __init__(self, path: str | Path, active_odor_name: str = "carbon dioxide"):
        self._path = str(path)
        if active_odor_name not in PLUME_ODOR_NAMES:
            raise ValueError(
                f"active_odor_name must be one of {PLUME_ODOR_NAMES}, got {active_odor_name!r}"
            )
        self._active_odor_name = active_odor_name
        self._active_odor_index = PLUME_ODOR_NAMES.index(active_odor_name)
        self._h5 = h5py.File(self._path, "r")
        self._plume = self._h5["plume"]
        self._inflow_pos = np.asarray(self._h5["inflow_pos"][()], dtype=np.float32)
        self._inflow_radius = float(np.asarray(self._h5["inflow_radius"][()]).reshape(-1)[0])
        self._inflow_scaler = float(np.asarray(self._h5["inflow_scaler"][()]).reshape(-1)[0])

    @property
    def odor_names(self) -> tuple[str, ...]:
        return tuple(PLUME_ODOR_NAMES)

    @property
    def active_odor_name(self) -> str:
        return self._active_odor_name

    @property
    def active_odor_index(self) -> int:
        return self._active_odor_index

    @property
    def n_frames(self) -> int:
        return int(self._plume.shape[0])

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (int(self._plume.shape[1]), int(self._plume.shape[2]))

    @property
    def inflow_pos(self) -> np.ndarray:
        return self._inflow_pos

    @property
    def inflow_radius(self) -> float:
        return self._inflow_radius

    @property
    def inflow_scaler(self) -> float:
        return self._inflow_scaler

    def get_frame(self, frame_idx: int) -> np.ndarray:
        idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        return self._plume[idx].astype(np.float32)

    def get_interpolated_frame(self, frame_time: float) -> np.ndarray:
        if self.n_frames == 1:
            return self.get_frame(0)
        wrapped = float(frame_time) % float(self.n_frames)
        idx0 = int(np.floor(wrapped)) % self.n_frames
        idx1 = (idx0 + 1) % self.n_frames
        w = np.float32(wrapped - np.floor(wrapped))
        if w == 0:
            return self.get_frame(idx0)
        frame0 = self.get_frame(idx0)
        frame1 = self.get_frame(idx1)
        return ((1.0 - w) * frame0 + w * frame1).astype(np.float32)

    def close(self):
        try:
            self._h5.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


class HDF5PlumeArena2D(composer.Arena):
    """Arena backed by a time-varying HDF5 plume dataset."""

    def _build(
        self,
        dataset_path: str | Path,
        arena_size: tuple[float, float] = (1.2, 0.8),
        active_odor_name: str = "carbon dioxide",
        plume_speed: float = 1.0,
        randomize_start: bool = True,
        top_camera_height: float | None = None,
        name: str = "hdf5_plume_arena_2d",
    ):
        super()._build(name=name)
        self._dataset = HDF5PlumeDataset2D(dataset_path, active_odor_name=active_odor_name)
        self._arena_size = tuple(arena_size)
        self._plume_speed = float(plume_speed)
        self._randomize_start = randomize_start
        self._frame_time = 0.0

        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4],
            diffuse=[0.8, 0.8, 0.8],
            specular=[0.1, 0.1, 0.1],
        )
        self._ground_texture = self._mjcf_root.asset.add(
            "texture",
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            type="2d",
            builtin="checker",
            name="groundplane",
            width=300,
            height=300,
        )
        self._ground_material = self._mjcf_root.asset.add(
            "material",
            name="groundplane",
            texrepeat=[4, 4],
            texuniform=False,
            reflectance=0.2,
            texture=self._ground_texture,
        )
        self._ground_geom = self._mjcf_root.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=(self._arena_size[0], self._arena_size[1], 0.1),
            material=self._ground_material,
        )
        if top_camera_height is None:
            scale = self._arena_size[0] / 1.2
            top_camera_height = 1.8 * scale
        self._mjcf_root.worldbody.add(
            "camera",
            name="top_close",
            mode="fixed",
            pos=(0.0, 0.0, top_camera_height),
            xyaxes=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )

        source_world = self._pixel_to_world(
            x_pix=float(self._dataset.inflow_pos[0]),
            y_from_bottom=float(self._dataset.inflow_pos[1]),
        )
        body = self._mjcf_root.worldbody.add(
            "body",
            name="plume_source_0",
            pos=(source_world[0], source_world[1], 0.02),
        )
        body.add(
            "site",
            name="plume_source_marker_0",
            type="sphere",
            size=(0.03,),
            rgba=(0.95, 0.25, 0.25, 0.85),
        )

    @property
    def ground_geoms(self):
        return (self._ground_geom,)

    @property
    def odor_names(self) -> tuple[str, ...]:
        return self._dataset.odor_names

    @property
    def odor_dimensions(self) -> int:
        return len(self.odor_names)

    @property
    def plume_field(self) -> np.ndarray:
        return self._dataset.get_interpolated_frame(self._frame_time)[None, ...]

    @property
    def active_odor_name(self) -> str:
        return self._dataset.active_odor_name

    @property
    def active_odor_index(self) -> int:
        return self._dataset.active_odor_index

    @property
    def current_frame_idx(self) -> int:
        return int(np.floor(self._frame_time)) % self._dataset.n_frames

    @property
    def current_frame_time(self) -> float:
        return self._frame_time

    def regenerate(self, random_state):
        self.reset_olfaction(random_state)

    def reset_olfaction(self, random_state=None):
        if self._randomize_start and random_state is not None:
            self._frame_time = float(random_state.uniform(0, self._dataset.n_frames))
        else:
            self._frame_time = 0.0

    def advance_plume(self):
        self._frame_time = (self._frame_time + self._plume_speed) % float(self._dataset.n_frames)

    def _pixel_to_world(self, x_pix: float, y_from_bottom: float) -> tuple[float, float]:
        height, width = self._dataset.frame_shape
        x_world = (x_pix / max(width - 1, 1) - 0.5) * (2.0 * self._arena_size[0])
        y_world = (y_from_bottom / max(height - 1, 1) - 0.5) * (2.0 * self._arena_size[1])
        return float(x_world), float(y_world)

    def _world_to_float_indices(self, x_world: np.ndarray, y_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = self._dataset.frame_shape
        x_float = ((x_world / (2.0 * self._arena_size[0])) + 0.5) * (width - 1)
        y_from_bottom = ((y_world / (2.0 * self._arena_size[1])) + 0.5) * (height - 1)
        row_float = (height - 1) - y_from_bottom
        return x_float.astype(np.float32), row_float.astype(np.float32)

    def get_olfaction(self, sensor_pos: np.ndarray) -> np.ndarray:
        sensor_pos = np.asarray(sensor_pos, dtype=np.float32)
        frame = self._dataset.get_interpolated_frame(self._frame_time)
        x_float, row_float = self._world_to_float_indices(sensor_pos[:, 0], sensor_pos[:, 1])
        height, width = frame.shape

        x0 = np.floor(x_float).astype(np.int32)
        y0 = np.floor(row_float).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)
        x0 = np.clip(x0, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)

        wx = (x_float - x0).astype(np.float32)
        wy = (row_float - y0).astype(np.float32)

        f00 = frame[y0, x0]
        f01 = frame[y1, x0]
        f10 = frame[y0, x1]
        f11 = frame[y1, x1]
        sampled = (
            (1.0 - wx) * (1.0 - wy) * f00
            + (1.0 - wx) * wy * f01
            + wx * (1.0 - wy) * f10
            + wx * wy * f11
        )
        odor = np.zeros((len(self.odor_names), sensor_pos.shape[0]), dtype=np.float32)
        odor[self.active_odor_index, :] = sampled.astype(np.float32)
        return odor
