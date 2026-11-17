"""Simple static arena with multiple odor sources."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from dm_control import composer


PLUME_ODOR_NAMES = (
    "carbon dioxide",
    "ethyl acetate",
    "putrescine",
    "cadaverine",
    "acetic acid",
)


def inverse_square_diffusion(
    distance: np.ndarray,
    min_distance: float = 0.05,
) -> np.ndarray:
    """Simple bounded inverse-square diffusion profile."""
    clipped_distance = np.maximum(distance, min_distance)
    return 1.0 / np.square(clipped_distance)


class StaticOlfactionArena(composer.Arena):
    """Flat arena with five static odor sources."""

    def _build(
        self,
        size=(2.0, 2.0),
        source_positions: Sequence[Sequence[float]] | None = None,
        peak_odor_intensity: np.ndarray | None = None,
        diffuse_func: Callable[[np.ndarray], np.ndarray] = inverse_square_diffusion,
        name: str = "static_olfaction_arena",
    ):
        super()._build(name=name)

        self._size = tuple(size)
        self._odor_names = tuple(PLUME_ODOR_NAMES)
        self._diffuse_func = diffuse_func

        if source_positions is None:
            source_positions = (
                (-0.45, 0.25, 0.02),
                (0.45, 0.25, 0.02),
                (-0.45, -0.25, 0.02),
                (0.45, -0.25, 0.02),
                (0.00, 0.55, 0.02),
            )
        self._source_positions = np.asarray(source_positions, dtype=np.float32)

        odor_dims = len(self._odor_names)
        if peak_odor_intensity is None:
            peak_odor_intensity = np.eye(odor_dims, dtype=np.float32)
        self._peak_odor_intensity = np.asarray(peak_odor_intensity, dtype=np.float32)
        if self._peak_odor_intensity.shape != (len(self._source_positions), odor_dims):
            raise ValueError(
                "peak_odor_intensity must have shape "
                f"({len(self._source_positions)}, {odor_dims}), "
                f"got {self._peak_odor_intensity.shape}"
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
            size=(self._size[0], self._size[1], 0.1),
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
            (0.95, 0.25, 0.25, 0.75),
            (0.95, 0.60, 0.20, 0.75),
            (0.85, 0.85, 0.20, 0.75),
            (0.30, 0.80, 0.35, 0.75),
            (0.25, 0.55, 0.95, 0.75),
        )
        for idx, source_pos in enumerate(self._source_positions):
            body = self._mjcf_root.worldbody.add(
                "body",
                name=f"odor_source_{idx}",
                pos=source_pos,
            )
            body.add(
                "site",
                name=f"odor_source_marker_{idx}",
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

    def regenerate(self, random_state):
        del random_state

    def get_olfaction(self, sensor_pos: np.ndarray) -> np.ndarray:
        """Return odor intensities at sensor positions.

        Args:
            sensor_pos: Sensor positions in world frame, shape (n_sensors, 3).

        Returns:
            Array of shape (n_odors, n_sensors).
        """
        sensor_pos = np.asarray(sensor_pos, dtype=np.float32)
        deltas = sensor_pos[None, :, :] - self._source_positions[:, None, :]
        distance = np.linalg.norm(deltas, axis=-1)
        source_strength = self._diffuse_func(distance).astype(np.float32)
        odor_sensor = self._peak_odor_intensity.T @ source_strength
        return odor_sensor.astype(np.float32)
