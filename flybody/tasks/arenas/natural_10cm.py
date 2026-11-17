"""Procedural 3-D natural arena for close-up walking videos.

The arena is intentionally built from MuJoCo geometry and inline low-poly
meshes. It does not use a ground image, billboard, sprite, or 2-D texture:
soil, stones, tree trunks, foliage, and grass all have visible 3-D volume.
The fixed layout keeps the 10 cm tracking task and its obstacle avoidance
reproducible across episodes.
"""

from __future__ import annotations

import numpy as np
from dm_control import composer


def _quat_from_z_axis(vector: np.ndarray) -> tuple[float, float, float, float]:
    """Return a MuJoCo quaternion rotating local +Z onto ``vector``."""
    direction = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("A capsule endpoint vector must be non-zero")
    direction /= norm
    z_axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-10:
        return (0.0, 1.0, 0.0, 0.0)
    axis = np.cross(z_axis, direction)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    angle_half = 0.5 * float(np.arccos(dot))
    sine = float(np.sin(angle_half))
    return (
        float(np.cos(angle_half)),
        float(axis[0] * sine),
        float(axis[1] * sine),
        float(axis[2] * sine),
    )


def _add_capsule_between(parent, name, start, end, radius, material):
    """Add a visual capsule whose local z axis joins two 3-D points."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    return parent.add(
        "geom",
        name=name,
        type="capsule",
        size=(float(radius), length * 0.5),
        pos=tuple((start + end) * 0.5),
        quat=_quat_from_z_axis(delta),
        material=material,
        contype=0,
        conaffinity=0,
    )


def _boulder_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Create a closed, faceted boulder mesh in unit dimensions."""
    segments = 8
    rings = ((0.0, 0.62), (0.18, 1.0), (0.67, 0.78), (0.94, 0.30))
    vertices = [[0.0, 0.0, 0.0]]
    for ring_index, (height, radius) in enumerate(rings):
        phase = 0.17 * ring_index
        for segment in range(segments):
            angle = 2.0 * np.pi * segment / segments + phase
            radial = radius * (1.0 + 0.06 * np.sin(3.0 * angle + ring_index))
            vertices.append([
                radial * np.cos(angle),
                radial * np.sin(angle),
                height,
            ])
    top_index = len(vertices)
    vertices.append([0.0, 0.0, 1.06])

    faces = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((0, 1 + nxt, 1 + segment))
    for ring in range(len(rings) - 1):
        first = 1 + ring * segments
        second = first + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            a, b = first + segment, first + nxt
            c, d = second + segment, second + nxt
            faces.extend(((a, b, d), (a, d, c)))
    last = 1 + (len(rings) - 1) * segments
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((last + segment, last + nxt, top_index))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


class NaturalTenCmArena(composer.Arena):
    """A bounded 10 cm x 10 cm, entirely 3-D natural ground patch."""

    half_extent_cm = 5.0

    def _build(self, name: str = "natural_10cm", seed: int = 0):
        super()._build(name=name)
        root = self._mjcf_root
        rng = np.random.RandomState(seed)
        root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4],
            diffuse=[0.8, 0.8, 0.8],
            specular=[0.1, 0.1, 0.1],
        )
        root.worldbody.add(
            "light",
            name="natural_fill_light",
            pos=(0.0, -2.0, 7.0),
            dir=(0.0, 0.15, -1.0),
            directional=True,
            diffuse=(0.55, 0.62, 0.48),
            specular=(0.14, 0.14, 0.11),
        )

        ground_material = root.asset.add(
            "material", name="natural_ground_material", rgba=(0.15, 0.22, 0.08, 1)
        )
        soil_materials = [
            root.asset.add("material", name="natural_soil_dark", rgba=(0.13, 0.18, 0.065, 1)),
            root.asset.add("material", name="natural_soil_mid", rgba=(0.25, 0.22, 0.10, 1)),
            root.asset.add("material", name="natural_soil_warm", rgba=(0.32, 0.25, 0.12, 1)),
        ]
        self._ground_geom = root.worldbody.add(
            "geom", name="natural_ground", type="plane",
            size=(self.half_extent_cm, self.half_extent_cm, 0.1),
            material=ground_material,
        )
        # Embedded soil clumps create 3-D parallax and small highlights.
        for index in range(34):
            x, y = rng.uniform(-4.75, 4.75, size=2)
            sx = float(rng.uniform(0.035, 0.10))
            sy = float(rng.uniform(0.025, 0.075))
            sz = float(rng.uniform(0.008, 0.025))
            root.worldbody.add(
                "geom", name=f"natural_soil_clump_{index}", type="ellipsoid",
                size=(sx, sy, sz), pos=(float(x), float(y), sz * 0.65),
                material=soil_materials[index % len(soil_materials)],
                contype=0, conaffinity=0,
            )

        wall_material = root.asset.add(
            "material", name="natural_wall_material", rgba=(0.10, 0.16, 0.06, 1)
        )
        for wall_name, pos, size in (
            ("north", (0, 5.0, 0.10), (5.0, 0.025, 0.10)),
            ("south", (0, -5.0, 0.10), (5.0, 0.025, 0.10)),
            ("east", (5.0, 0, 0.10), (0.025, 5.0, 0.10)),
            ("west", (-5.0, 0, 0.10), (0.025, 5.0, 0.10)),
        ):
            root.worldbody.add(
                "geom", name=f"natural_boundary_{wall_name}", type="box",
                pos=pos, size=size, material=wall_material,
                contype=1, conaffinity=1,
            )

        trunk_material = root.asset.add(
            "material", name="natural_trunk_material", rgba=(0.25, 0.105, 0.035, 1)
        )
        trunk_light_material = root.asset.add(
            "material", name="natural_trunk_light", rgba=(0.39, 0.18, 0.065, 1)
        )
        leaf_materials = [
            root.asset.add("material", name="natural_leaf_deep", rgba=(0.035, 0.20, 0.025, 1)),
            root.asset.add("material", name="natural_leaf_mid", rgba=(0.08, 0.36, 0.045, 1)),
            root.asset.add("material", name="natural_leaf_light", rgba=(0.18, 0.48, 0.075, 1)),
        ]
        stone_materials = [
            root.asset.add("material", name="natural_stone_grey", rgba=(0.28, 0.29, 0.25, 1)),
            root.asset.add("material", name="natural_stone_warm", rgba=(0.38, 0.32, 0.23, 1)),
            root.asset.add("material", name="natural_stone_dark", rgba=(0.20, 0.22, 0.20, 1)),
        ]
        grass_materials = [
            root.asset.add("material", name="natural_grass_dark", rgba=(0.07, 0.28, 0.025, 1)),
            root.asset.add("material", name="natural_grass_mid", rgba=(0.15, 0.46, 0.045, 1)),
            root.asset.add("material", name="natural_grass_light", rgba=(0.27, 0.58, 0.075, 1)),
        ]

        trees = (
            (-3.9, 3.7, 0.28, 0.65),
            (3.9, 3.6, 0.24, 0.55),
            (3.8, -3.8, 0.22, 0.50),
        )
        self._obstacle_centers_cm = np.asarray(
            [(x, y) for x, y, _, _ in trees], dtype=np.float64
        )
        self._obstacle_radii_cm = np.asarray(
            [canopy_r for _, _, _, canopy_r in trees], dtype=np.float64
        )
        for tree_index, (x, y, trunk_radius, canopy_radius) in enumerate(trees):
            body = root.worldbody.add(
                "body", name=f"natural_tree_{tree_index}", pos=(x, y, 0)
            )
            body.add(
                "geom", name=f"natural_tree_trunk_base_{tree_index}", type="cylinder",
                size=(trunk_radius * 1.12, 0.34), pos=(0, 0, 0.34),
                material=trunk_material, contype=1, conaffinity=1,
            )
            body.add(
                "geom", name=f"natural_tree_trunk_upper_{tree_index}", type="cylinder",
                size=(trunk_radius * 0.78, 0.34), pos=(0, 0, 0.92),
                material=trunk_light_material, contype=1, conaffinity=1,
            )
            branch_specs = (
                ((0.0, 0.0, 0.58), (0.28, 0.03, 0.90)),
                ((0.0, 0.0, 0.72), (-0.22, 0.05, 1.00)),
                ((0.0, 0.0, 0.86), (0.08, -0.25, 1.08)),
            )
            for branch_index, (start, end) in enumerate(branch_specs):
                _add_capsule_between(
                    body, f"natural_tree_branch_{tree_index}_{branch_index}",
                    start, end, trunk_radius * 0.42, trunk_material,
                )
            canopy_specs = (
                ((0.00, 0.00, 1.18), (1.00, 0.82, 0.72), 1),
                ((0.30, 0.00, 1.04), (0.66, 0.55, 0.48), 2),
                ((-0.25, 0.06, 1.04), (0.62, 0.52, 0.46), 0),
                ((0.02, -0.25, 1.08), (0.58, 0.48, 0.42), 2),
                ((0.00, 0.18, 1.30), (0.52, 0.42, 0.38), 1),
            )
            for cluster_index, (position, scale, material_index) in enumerate(canopy_specs):
                body.add(
                    "geom", name=f"natural_tree_leaf_cluster_{tree_index}_{cluster_index}",
                    type="ellipsoid",
                    size=tuple(canopy_radius * value for value in scale),
                    pos=position, material=leaf_materials[material_index],
                    contype=0, conaffinity=0,
                )

        boulder_vertices, boulder_faces = _boulder_mesh()
        rocks = (
            (-3.5, 0.7, 0.17, 0.10),
            (2.9, 1.7, 0.14, 0.08),
            (-2.7, -2.7, 0.12, 0.07),
            (1.0, -3.9, 0.16, 0.09),
            (4.1, 0.2, 0.10, 0.06),
        )
        for rock_index, (x, y, radius, height) in enumerate(rocks):
            mesh = root.asset.add(
                "mesh", name=f"natural_rock_mesh_{rock_index}",
                vertex=boulder_vertices.reshape(-1), face=boulder_faces.reshape(-1),
                scale=(radius, radius * 0.72, height), smoothnormal=False,
            )
            root.worldbody.add(
                "geom", name=f"natural_rock_{rock_index}", type="mesh", mesh=mesh,
                pos=(x, y, 0.0), material=stone_materials[rock_index % len(stone_materials)],
                contype=1, conaffinity=1,
            )

        self._obstacle_centers_cm = np.concatenate(
            [self._obstacle_centers_cm,
             np.asarray([(x, y) for x, y, _, _ in rocks], dtype=np.float64)], axis=0
        )
        self._obstacle_radii_cm = np.concatenate(
            [self._obstacle_radii_cm,
             np.asarray([radius for _, _, radius, _ in rocks], dtype=np.float64)], axis=0
        )

        grass_positions = (
            (-4.3, 1.8), (-4.0, -1.2), (-1.8, 4.1), (0.4, 4.2),
            (2.2, -4.2), (4.2, -1.7), (1.8, 3.2), (-0.7, -4.1),
            (-1.2, 2.8), (2.5, -0.8), (-3.2, -4.0), (3.6, 1.0),
        )
        for tuft_index, (x, y) in enumerate(grass_positions):
            for blade_index in range(5):
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
                height = float(rng.uniform(0.16, 0.30))
                lean = float(rng.uniform(0.025, 0.09))
                start = np.asarray((x + rng.uniform(-0.055, 0.055),
                                    y + rng.uniform(-0.055, 0.055), 0.01))
                end = start + np.asarray((
                    lean * np.cos(angle), lean * np.sin(angle), height,
                ))
                _add_capsule_between(
                    root.worldbody, f"natural_grass_{tuft_index}_{blade_index}",
                    start, end, float(rng.uniform(0.008, 0.015)),
                    grass_materials[blade_index % len(grass_materials)],
                )

    @property
    def ground_geoms(self):
        return (self._ground_geom,)

    @property
    def bounds_cm(self) -> tuple[float, float, float, float]:
        return (-self.half_extent_cm, self.half_extent_cm,
                -self.half_extent_cm, self.half_extent_cm)

    @property
    def obstacle_centers_cm(self) -> np.ndarray:
        return self._obstacle_centers_cm.copy()

    @property
    def obstacle_radii_cm(self) -> np.ndarray:
        return self._obstacle_radii_cm.copy()

    def regenerate(self, random_state):
        del random_state


__all__ = ["NaturalTenCmArena"]
