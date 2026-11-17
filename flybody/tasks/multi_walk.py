"""Multi-fly walking tasks.

This module intentionally does not subclass :mod:`walk_imitation`/
``FruitFlyTask`` because those classes own a single ``self._walker``.  The
classes below keep the single-fly environments unchanged while sharing the same
``FruitFly`` model and walking-time-step conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence, Union

import numpy as np

from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.utils import rewards
from dm_env import specs

from flybody.tasks.walk_root_imitation import RootImitationRewardConfig, _gaussian_factor
from flybody import quaternions
from flybody.tasks.constants import (
    _BODY_PITCH_ANGLE,
    _TERMINAL_ANGVEL,
    _TERMINAL_LINVEL,
    _TERMINAL_QACC,
    _WALK_CONTROL_TIMESTEP,
    _WALK_PHYSICS_TIMESTEP,
)
from flybody.quaternions import get_dquat_local
from flybody.tasks.task_utils import any_substr_in_str


RewardAggregation = Literal["mean", "sum", "product"]
SpawnLayout = Literal["line", "grid"]


@dataclass(frozen=True)
class MultiFlySpawnConfig:
    """Spawn-layout options for a multi-fly walking scene.

    Units follow the rest of the walking environment: centimeters and seconds.
    If ``offsets_cm`` is provided it takes precedence over ``layout`` and
    ``spacing_cm``.  Offsets are added to the root reference trajectory of each
    fly, so flies can track independent root trajectories without starting on
    top of each other.
    """

    num_flies: int = 2
    layout: SpawnLayout = "line"
    spacing_cm: float = 0.45
    origin_cm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    offsets_cm: list[list[float]] = field(default_factory=list)

    def get_offsets_cm(self) -> np.ndarray:
        """Return ``(num_flies, 3)`` spawn offsets in centimeters."""

        if self.num_flies < 1:
            raise ValueError(f"num_flies must be >= 1, got {self.num_flies}")

        origin = np.asarray(self.origin_cm, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError(
                f"origin_cm must contain exactly 3 values, got {self.origin_cm}"
            )

        if self.offsets_cm:
            offsets = np.asarray(self.offsets_cm, dtype=np.float64)
            if offsets.shape != (self.num_flies, 3):
                raise ValueError(
                    "offsets_cm must have shape "
                    f"({self.num_flies}, 3), got {offsets.shape}"
                )
            return offsets + origin

        if self.layout == "line":
            xs = np.arange(self.num_flies, dtype=np.float64)
            xs -= (self.num_flies - 1) / 2
            offsets = np.zeros((self.num_flies, 3), dtype=np.float64)
            offsets[:, 0] = xs * float(self.spacing_cm)
            return offsets + origin

        if self.layout == "grid":
            n_cols = int(np.ceil(np.sqrt(self.num_flies)))
            offsets = np.zeros((self.num_flies, 3), dtype=np.float64)
            for i in range(self.num_flies):
                row = i // n_cols
                col = i % n_cols
                offsets[i, 0] = col * float(self.spacing_cm)
                offsets[i, 1] = row * float(self.spacing_cm)
            offsets[:, 0] -= offsets[:, 0].mean()
            offsets[:, 1] -= offsets[:, 1].mean()
            return offsets + origin

        raise ValueError(f"Unsupported spawn layout: {self.layout!r}")


@dataclass(frozen=True)
class MultiWalkRootImitationTaskConfig:
    """Configuration for root-only multi-fly walking imitation."""

    spawn: MultiFlySpawnConfig = field(default_factory=MultiFlySpawnConfig)
    reward: RootImitationRewardConfig = field(default_factory=RootImitationRewardConfig)
    terminal_com_dist: float = 1.0
    reward_aggregation: RewardAggregation = "mean"
    terminate_on_any_fly: bool = True
    future_steps: int = 16
    initialize_qvel: bool = True
    enable_contact_observables: bool = True
    enable_appendage_observables: bool = True
    add_birdview_camera: bool = True
    birdview_camera_name: str = "birdview"
    birdview_camera_height_cm: float = 5.0
    birdview_camera_fovy: float = 55.0
    trajectory_sites: bool = False
    trajectory_site_stride: int = 10
    trajectory_site_size_cm: float = 0.015
    trajectory_site_alpha: float = 0.75


class MultiWalkRootImitation(composer.Task):
    """Root-only walking imitation with multiple controllable fruit flies.

    The environment action is the concatenation of each fly's single-fly action
    vector in the original ``FruitFly`` action order.  Controls are written via
    ``physics.bind(local_actuators).ctrl`` for each fly, avoiding the global
    ``physics.set_control`` call used by the single-fly walker.
    """

    def __init__(
        self,
        walker: Union["base.Walker", Callable],
        arena: composer.Arena,
        traj_generator,
        task_config: MultiWalkRootImitationTaskConfig | None = None,
        *,
        time_limit: float = 10.0,
        force_actuators: bool = False,
        disable_wings: bool = True,
        joint_filter: float = 0.01,
        adhesion_filter: float = 0.007,
        adhesion_gain: float | None = None,
        claw_friction: float | None = 1.0,
        body_pitch_angle: float = _BODY_PITCH_ANGLE,
        stroke_plane_angle: float = 0.0,
        eye_camera_fovy: float = 150.0,
        eye_camera_size: int = (32, 32),
        observables_options: dict | None = None,
    ):
        """Construct a multi-fly root imitation task.

        Args:
            walker: Walker constructor, normally ``fruitfly.FruitFly``.
            arena: Composer arena.
            traj_generator: Walking trajectory loader.  It is called once per
                fly per episode; lazy HDF5 slicing keeps memory usage bounded.
            task_config: Dataclass containing spawn/reward/task options.
            time_limit: Episode limit in seconds.
            force_actuators: Whether to use force or position actuators.
            disable_wings: Whether to retract and disable wing actuators.
            joint_filter: Timescale of filter for joint actuators. 0 disables.
            adhesion_filter: Timescale of filter for adhesion actuators.
            adhesion_gain: Optional override of adhesion gain.
            claw_friction: Optional claw-friction override.
            body_pitch_angle: Passed to ``FruitFly``.
            stroke_plane_angle: Passed to ``FruitFly``.
            eye_camera_fovy: Passed to ``FruitFly``.
            eye_camera_size: Passed to ``FruitFly``.
            observables_options: Optional observable options propagated to all
                fly observables.
        """

        self._arena = arena
        self._traj_generator = traj_generator
        self._task_config = task_config or MultiWalkRootImitationTaskConfig()
        self._time_limit = float(time_limit)
        self._should_terminate = False
        self._reached_traj_end = False
        self._step_counter = 0
        self._last_action_norms = np.zeros(self._task_config.spawn.num_flies)
        self._pre_step_walker_xy = np.zeros((self._task_config.spawn.num_flies, 2))
        self._last_per_fly_rewards = np.ones(self._task_config.spawn.num_flies)
        self._spawn_offsets = self._task_config.spawn.get_offsets_cm()
        self._ref_qpos = self._make_dummy_ref_qpos()
        self._ref_qvel = [
            np.zeros((self._task_config.future_steps + 1, 6), dtype=np.float64)
            for _ in range(self.num_flies)
        ]
        self._snippet_steps: list[int] = []
        self._last_sample_infos: list[dict] = []
        self._episode_steps = 0
        self._trajectory_site_names: list[list[str]] = []

        # Create controllable flies.  Keep the first model named "walker" so
        # existing FruitFly.initialize_episode hard-coded lookups continue to
        # work without modifying the single-fly environment.
        self._walkers = []
        self._root_joints = []
        self._actuators = []
        self._action_maps = []
        self._local_ctrl_buffers = []
        self._wing_joints = []
        self._wing_springrefs = []

        for fly_idx in range(self.num_flies):
            name = "walker" if fly_idx == 0 else f"walker_{fly_idx}"
            fly = walker(
                name=name,
                use_legs=True,
                use_wings=not disable_wings,
                use_mouth=False,
                use_antennae=False,
                force_actuators=force_actuators,
                joint_filter=joint_filter,
                adhesion_filter=adhesion_filter,
                body_pitch_angle=body_pitch_angle,
                stroke_plane_angle=stroke_plane_angle,
                physics_timestep=_WALK_PHYSICS_TIMESTEP,
                control_timestep=_WALK_CONTROL_TIMESTEP,
                num_user_actions=0,
                eye_camera_fovy=eye_camera_fovy,
                eye_camera_size=eye_camera_size,
            )
            fly.observables.set_options(observables_options)
            if hasattr(arena, "get_olfaction") and hasattr(arena, "odor_names"):
                fly.configure_olfaction(
                    callback=arena.get_olfaction,
                    odor_dimension_names=arena.odor_names,
                )

            spawn_site = self._arena.mjcf_model.worldbody.add(
                "site",
                pos=self._spawn_offsets[fly_idx] + fly.upright_pose.xpos,
            )
            fly.create_root_joints(arena.attach(fly, spawn_site))
            spawn_site.remove()

            root_joint = mjcf.get_frame_freejoint(fly.mjcf_model)
            actuators = tuple(fly.mjcf_model.find_all("actuator"))
            action_map = self._make_action_map(fly)
            wing_joints, wing_springrefs = self._get_wing_joints_and_springrefs(fly)

            if adhesion_gain is not None:
                fly.mjcf_model.find("default", "adhesion_claw").adhesion.gain = (
                    adhesion_gain
                )
            if claw_friction is not None:
                fly.mjcf_model.find("default", "adhesion-collision").geom.friction = (
                    claw_friction,
                )

            self._exclude_wing_leg_collisions(fly)
            self._enable_walking_observables(fly)
            self._add_reference_observables(fly, fly_idx)

            self._walkers.append(fly)
            self._root_joints.append(root_joint)
            self._actuators.append(actuators)
            self._action_maps.append(action_map)
            self._local_ctrl_buffers.append(
                np.zeros(len(actuators), dtype=np.float64)
            )
            self._wing_joints.append(tuple(wing_joints))
            self._wing_springrefs.append(np.asarray(wing_springrefs, dtype=np.float64))

        # Set floor contact params once for the shared arena.
        for geom in self._arena.ground_geoms:
            geom.friction = (0.5,)
            geom.solref = (0.001, 1)
            geom.solimp = (0.95, 0.99, 0.01)

        if self._task_config.add_birdview_camera:
            self._add_birdview_camera()
        if self._task_config.trajectory_sites:
            self._add_trajectory_sites()

        self.set_timesteps(
            physics_timestep=_WALK_PHYSICS_TIMESTEP,
            control_timestep=_WALK_CONTROL_TIMESTEP,
        )

        # Match single-fly mass/inertia compiler settings.
        for fly in self._walkers:
            fly.mjcf_model.compiler.boundmass = 0.0
            fly.mjcf_model.compiler.boundinertia = 0.0

    def _make_dummy_ref_qpos(self) -> list[np.ndarray]:
        """Create placeholder reference arrays for observable-spec creation."""

        refs = []
        for fly_idx in range(self.num_flies):
            qpos = np.zeros(
                (self._task_config.future_steps + 1, 7),
                dtype=np.float64,
            )
            qpos[:, :3] = self._spawn_offsets[fly_idx]
            qpos[:, 3] = 1.0
            refs.append(qpos)
        return refs

    def _add_birdview_camera(self) -> None:
        """Add a fixed top-down camera to the shared arena."""

        name = self._task_config.birdview_camera_name
        if self._arena.mjcf_model.find("camera", name) is not None:
            return

        center_xy = self._spawn_offsets[:, :2].mean(axis=0)
        height = float(self._task_config.birdview_camera_height_cm)
        pos = (float(center_xy[0]), float(center_xy[1]), height)
        # MuJoCo cameras look along their local -Z axis.  With world X/Y as the
        # camera X/Y axes, local +Z is world +Z, so the view direction is down.
        self._arena.mjcf_model.worldbody.add(
            "camera",
            name=name,
            pos=pos,
            xyaxes=(1, 0, 0, 0, 1, 0),
            fovy=float(self._task_config.birdview_camera_fovy),
        )

    @staticmethod
    def _trajectory_color(fly_idx: int) -> tuple[float, float, float]:
        colors = (
            (1.0, 0.12, 0.12),  # red
            (0.12, 1.0, 0.12),  # green
            (0.12, 0.35, 1.0),  # blue
            (1.0, 0.85, 0.12),  # yellow
            (0.85, 0.12, 1.0),  # magenta
            (0.12, 1.0, 1.0),  # cyan
        )
        return colors[int(fly_idx) % len(colors)]

    def _add_trajectory_sites(self) -> None:
        """Add per-fly colored reference trajectory sites to the scene."""

        stride = max(1, int(self._task_config.trajectory_site_stride))
        max_episode_steps = round(self._time_limit / self.control_timestep) + 1
        n_sites = max(1, int(np.ceil(max_episode_steps / stride)) + 1)
        size = float(self._task_config.trajectory_site_size_cm)
        self._trajectory_site_names = []
        for fly_idx in range(self.num_flies):
            color = self._trajectory_color(fly_idx)
            names = []
            for site_idx in range(n_sites):
                name = f"traj_f{fly_idx}_{site_idx}"
                if self._arena.mjcf_model.worldbody.find("site", name) is None:
                    self._arena.mjcf_model.worldbody.add(
                        "site",
                        name=name,
                        pos=(0.0, 0.0, -10.0),
                        size=(size, size, size),
                        rgba=(*color, 0.0),
                        group=1,
                    )
                names.append(name)
            self._trajectory_site_names.append(names)

    def _update_trajectory_sites(self) -> None:
        """Place reference trajectory sites for the current sampled episode."""

        if not self._trajectory_site_names:
            return
        stride = max(1, int(self._task_config.trajectory_site_stride))
        alpha = float(self._task_config.trajectory_site_alpha)
        for fly_idx, names in enumerate(self._trajectory_site_names):
            color = self._trajectory_color(fly_idx)
            ref_qpos = self._ref_qpos[fly_idx]
            for site_idx, name in enumerate(names):
                site = self._arena.mjcf_model.worldbody.find("site", name)
                step = int(site_idx * stride)
                if step <= self._episode_steps and step < ref_qpos.shape[0]:
                    site.pos = ref_qpos[step, :3]
                    site.rgba = (*color, alpha)
                else:
                    site.rgba = (*color, 0.0)

    @property
    def root_entity(self):
        return self._arena

    @property
    def walkers(self):
        return tuple(self._walkers)

    @property
    def walker(self):
        """Return the first fly for code that expects a primary walker."""

        return self._walkers[0]

    @property
    def last_per_fly_rewards(self) -> np.ndarray:
        """Per-fly scalar rewards from the most recent ``get_reward`` call."""

        return self._last_per_fly_rewards.copy()

    @property
    def last_sample_infos(self) -> tuple[dict, ...]:
        """Trajectory/window metadata from the most recent episode reset."""

        return tuple(dict(info) for info in self._last_sample_infos)

    @property
    def num_flies(self) -> int:
        return int(self._task_config.spawn.num_flies)

    @property
    def control_timestep(self) -> float:
        return _WALK_CONTROL_TIMESTEP

    def _enable_walking_observables(self, fly) -> None:
        for sensor in fly.observables.vestibular + fly.observables.proprioception:
            sensor.enabled = True
        if self._task_config.enable_appendage_observables:
            fly.observables.appendages_pos.enabled = True
        if self._task_config.enable_contact_observables:
            fly.observables.force.enabled = True
            fly.observables.touch.enabled = True
        fly.observables.self_contact.enabled = False
        if hasattr(self._arena, "get_olfaction"):
            fly.observables.odor_intensity.enabled = True

    def _add_reference_observables(self, fly, fly_idx: int) -> None:
        fly.observables.add_observable(
            "ref_displacement",
            observable.Generic(self._make_ref_displacement_getter(fly, fly_idx)),
        )
        fly.observables.add_observable(
            "ref_root_quat",
            observable.Generic(self._make_ref_root_quat_getter(fly, fly_idx)),
        )

    def _make_ref_displacement_getter(self, fly, fly_idx: int):
        def get_ref_displacement(physics: "mjcf.Physics"):
            fly_pos, _ = fly.get_pose(physics)
            ref_pos = self._ref_qpos[fly_idx][
                self._step_counter : self._step_counter
                + self._task_config.future_steps
                + 1,
                :3,
            ]
            return fly.transform_vec_to_egocentric_frame(physics, ref_pos - fly_pos)

        return get_ref_displacement

    def _make_ref_root_quat_getter(self, fly, fly_idx: int):
        def get_ref_root_quat(physics: "mjcf.Physics"):
            ref_quat = self._ref_qpos[fly_idx][
                self._step_counter : self._step_counter
                + self._task_config.future_steps
                + 1,
                3:7,
            ]
            _, fly_quat = fly.get_pose(physics)
            return get_dquat_local(fly_quat, ref_quat)

        return get_ref_root_quat

    @staticmethod
    def _make_action_map(fly) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        action_map = []
        for key, action_indices in fly._action_indices.items():
            ctrl_indices = fly._ctrl_indices[key]
            if ctrl_indices and action_indices:
                action_map.append(
                    (
                        np.asarray(ctrl_indices, dtype=np.int64),
                        np.asarray(action_indices, dtype=np.int64),
                    )
                )
        return tuple(action_map)

    @staticmethod
    def _get_wing_joints_and_springrefs(fly) -> tuple[list, list[float]]:
        wing_joints = []
        wing_springrefs = []
        for joint in fly.mjcf_model.find_all("joint"):
            if any_substr_in_str(["yaw", "roll", "pitch"], joint.name):
                springref = joint.springref or joint.dclass.joint.springref or 0.0
                wing_joints.append(joint)
                wing_springrefs.append(springref)
        return wing_joints, wing_springrefs

    @staticmethod
    def _exclude_wing_leg_collisions(fly) -> None:
        contact = fly.mjcf_model.contact
        for body in fly.mjcf_model.find_all("body"):
            if any_substr_in_str(
                ["coxa", "femur", "tibia", "tarsus", "claw"], body.name
            ):
                for wing in ["wing_left", "wing_right"]:
                    contact.add(
                        "exclude",
                        name=f"{body.name}_{wing}",
                        body1=body.name,
                        body2=wing,
                    )

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        if hasattr(self._arena, "regenerate"):
            self._arena.regenerate(random_state)

        # Better visual defaults for CGS units, copied from FruitFlyTask.
        self.root_entity.mjcf_model.visual.map.znear = 0.001
        self.root_entity.mjcf_model.visual.map.zfar = 50.0
        self.root_entity.mjcf_model.visual.map.force = 0.00001
        self.root_entity.mjcf_model.visual.scale.framewidth = 0.06
        self.root_entity.mjcf_model.visual.scale.forcewidth = 0.06
        self.root_entity.mjcf_model.visual.scale.contactwidth = 0.3
        self.root_entity.mjcf_model.visual.scale.contactheight = 0.1
        self.root_entity.mjcf_model.visual.scale.jointwidth = 0.06
        self.root_entity.mjcf_model.statistic.extent = 4.01

        self._sample_reference_trajectories()
        self._update_trajectory_sites()

    def _sample_reference_trajectories(self) -> None:
        self._ref_qpos = []
        self._ref_qvel = []
        self._snippet_steps = []
        self._last_sample_infos = []
        max_episode_steps = round(self._time_limit / self.control_timestep) + 1
        if hasattr(self._traj_generator, "get_multi_trajectory"):
            snippets = self._traj_generator.get_multi_trajectory(
                num_flies=self.num_flies,
                traj_idx=None,
            )
            if len(snippets) != self.num_flies:
                raise ValueError(
                    "Synchronized trajectory loader returned "
                    f"{len(snippets)} snippets for {self.num_flies} flies."
                )
            sample_info = (
                self._traj_generator.last_sample_info
                if hasattr(self._traj_generator, "last_sample_info")
                else {}
            )
            self._last_sample_infos = [
                {**sample_info, "fly_idx": fly_idx}
                for fly_idx in range(self.num_flies)
            ]
        else:
            snippets = []
            for fly_idx in range(self.num_flies):
                try:
                    snippet = self._traj_generator.get_trajectory(traj_idx=None)
                except TypeError:
                    snippet = self._traj_generator.get_trajectory(0)
                snippets.append(snippet)
                sample_info = (
                    self._traj_generator.last_sample_info
                    if hasattr(self._traj_generator, "last_sample_info")
                    else {}
                )
                self._last_sample_infos.append(
                    {**sample_info, "fly_idx": fly_idx}
                )

        for fly_idx, snippet in enumerate(snippets):
            qpos = np.asarray(snippet["qpos"], dtype=np.float64).copy()
            qvel = np.asarray(snippet["qvel"], dtype=np.float64).copy()
            if qpos.shape[1] < 7 or qvel.shape[1] < 6:
                raise ValueError(
                    "Root imitation snippets must contain at least root qpos/qvel "
                    f"(got qpos {qpos.shape}, qvel {qvel.shape})."
                )
            qpos[:, :3] += self._spawn_offsets[fly_idx]
            snippet_steps = qpos.shape[0] - self._task_config.future_steps - 1
            if snippet_steps <= 0:
                raise ValueError(
                    "Trajectory is shorter than future_steps look-ahead: "
                    f"qpos steps={qpos.shape[0]}, "
                    f"future_steps={self._task_config.future_steps}"
                )
            self._ref_qpos.append(qpos)
            self._ref_qvel.append(qvel)
            self._snippet_steps.append(snippet_steps)

        self._episode_steps = min(max_episode_steps, *self._snippet_steps)

    def initialize_episode(
        self,
        physics: "mjcf.Physics",
        random_state: np.random.RandomState,
    ) -> None:
        self._step_counter = 0
        self._should_terminate = False
        self._reached_traj_end = False
        self._last_action_norms[:] = 0.0
        self._pre_step_walker_xy[:] = 0.0
        self._last_per_fly_rewards[:] = 1.0
        if hasattr(self._arena, "reset_olfaction"):
            self._arena.reset_olfaction(random_state)

        for fly_idx, root_joint in enumerate(self._root_joints):
            physics.bind(root_joint).qpos = self._ref_qpos[fly_idx][0, :7]
            if self._task_config.initialize_qvel:
                physics.bind(root_joint).qvel = self._ref_qvel[fly_idx][0, :6]
            if self._wing_joints[fly_idx]:
                physics.bind(self._wing_joints[fly_idx]).qpos = self._wing_springrefs[
                    fly_idx
                ]
            fly_pos, _ = self._walkers[fly_idx].get_pose(physics)
            self._pre_step_walker_xy[fly_idx] = np.asarray(
                fly_pos[:2],
                dtype=np.float64,
            )

    def before_step(
        self,
        physics: "mjcf.Physics",
        action,
        random_state: np.random.RandomState,
    ) -> None:
        del random_state
        self._step_counter += 1
        action = np.asarray(action, dtype=np.float64).copy()
        action[np.isnan(action)] = 0.0

        per_fly_dim = self._single_fly_action_dim()
        if action.shape != (self.num_flies * per_fly_dim,):
            raise ValueError(
                "Expected flattened multi-fly action shape "
                f"{(self.num_flies * per_fly_dim,)}, got {action.shape}"
            )

        for fly_idx, fly in enumerate(self._walkers):
            fly_pos, _ = fly.get_pose(physics)
            self._pre_step_walker_xy[fly_idx] = np.asarray(
                fly_pos[:2],
                dtype=np.float64,
            )
            fly_action = action[fly_idx * per_fly_dim : (fly_idx + 1) * per_fly_dim]
            self._last_action_norms[fly_idx] = float(np.linalg.norm(fly_action))
            self._apply_walker_action(physics, fly, fly_idx, fly_action)

        if hasattr(self._arena, "advance_plume"):
            self._arena.advance_plume()

    def _single_fly_action_dim(self) -> int:
        return int(sum(self._walkers[0]._num_actions.values()))

    def _apply_walker_action(
        self,
        physics: "mjcf.Physics",
        fly,
        fly_idx: int,
        action: np.ndarray,
    ) -> None:
        if not self._actuators[fly_idx]:
            return
        if getattr(fly, "_prev_action", None) is not None:
            fly._prev_action[:] = action

        local_ctrl = self._local_ctrl_buffers[fly_idx]
        local_ctrl.fill(0.0)
        for ctrl_indices, action_indices in self._action_maps[fly_idx]:
            local_ctrl[ctrl_indices] = action[action_indices]

        physics.bind(self._actuators[fly_idx]).ctrl = local_ctrl

    def action_spec(self, physics: "mjcf.Physics"):
        minima = []
        maxima = []
        names = []
        for fly_idx, fly in enumerate(self._walkers):
            local_actuators = self._actuators[fly_idx]
            ctrlrange = physics.bind(local_actuators).ctrlrange
            for ctrl_indices, _ in self._action_maps[fly_idx]:
                minima.extend(ctrlrange[ctrl_indices, 0])
                maxima.extend(ctrlrange[ctrl_indices, 1])
                names.extend(
                    f"{fly.name}/{local_actuators[i].name}" for i in ctrl_indices
                )

        return specs.BoundedArray(
            shape=(len(minima),),
            dtype=float,
            minimum=np.asarray(minima),
            maximum=np.asarray(maxima),
            name="\t".join(names),
        )

    def get_reward(self, physics: "mjcf.Physics") -> float:
        self._should_terminate = self.check_termination(physics)
        per_fly_rewards = np.asarray(
            [
                self._get_fly_reward(physics, fly_idx)
                for fly_idx in range(self.num_flies)
            ],
            dtype=np.float64,
        )
        self._last_per_fly_rewards = per_fly_rewards
        aggregation = self._task_config.reward_aggregation
        if aggregation == "mean":
            return float(np.mean(per_fly_rewards))
        if aggregation == "sum":
            return float(np.sum(per_fly_rewards))
        if aggregation == "product":
            return float(np.prod(per_fly_rewards))
        raise ValueError(f"Unsupported reward aggregation: {aggregation!r}")

    def _get_fly_reward(self, physics: "mjcf.Physics", fly_idx: int) -> float:
        return float(np.prod(self._get_fly_reward_factors(physics, fly_idx)))

    def _get_fly_reward_factors(
        self,
        physics: "mjcf.Physics",
        fly_idx: int,
    ) -> Sequence[float]:
        cfg = self._task_config.reward
        step = int(round(physics.time() / self.control_timestep))
        step = min(step, self._ref_qpos[fly_idx].shape[0] - 1)

        fly = self._walkers[fly_idx]
        walker_pos, walker_quat = fly.get_pose(physics)
        ref_qpos = self._ref_qpos[fly_idx][step]
        ref_qvel = self._ref_qvel[fly_idx][step]
        walker_qvel = physics.bind(self._root_joints[fly_idx]).qvel

        position_error = float(np.linalg.norm(walker_pos - ref_qpos[:3]))
        orientation_error = float(
            quaternions.quat_dist_short_arc(walker_quat, ref_qpos[3:7])
        )
        linear_velocity_error = float(
            np.linalg.norm(walker_qvel[:3] - ref_qvel[:3])
        )
        angular_velocity_error = float(
            np.linalg.norm(walker_qvel[3:6] - ref_qvel[3:6])
        )
        if step > 0:
            ref_delta_xy = ref_qpos[:2] - self._ref_qpos[fly_idx][step - 1, :2]
        else:
            ref_delta_xy = np.zeros(2, dtype=np.float64)
        walker_delta_xy = (
            np.asarray(walker_pos[:2], dtype=np.float64)
            - self._pre_step_walker_xy[fly_idx]
        )
        progress_error = float(np.linalg.norm(walker_delta_xy - ref_delta_xy))

        factors = [
            1.0,  # survival/CPG-style alive factor.
            _gaussian_factor(
                position_error,
                cfg.position_std_cm,
                cfg.weight("position"),
            ),
            _gaussian_factor(
                orientation_error,
                cfg.orientation_std_rad,
                cfg.weight("orientation"),
            ),
            _gaussian_factor(
                linear_velocity_error,
                cfg.linear_velocity_std_cm_s,
                cfg.weight("linear_velocity"),
            ),
            _gaussian_factor(
                angular_velocity_error,
                cfg.angular_velocity_std_rad_s,
                cfg.weight("angular_velocity"),
            ),
            _gaussian_factor(
                progress_error,
                cfg.progress_std_cm,
                cfg.weight("progress"),
            ),
            _gaussian_factor(
                self._last_action_norms[fly_idx],
                cfg.action_l2_std,
                cfg.weight("action"),
            ),
        ]

        if self._wing_joints[fly_idx]:
            qpos_diff = (
                physics.bind(self._wing_joints[fly_idx]).qpos
                - self._wing_springrefs[fly_idx]
            )
            retract_wings = rewards.tolerance(
                qpos_diff,
                bounds=(0, 0),
                sigmoid="linear",
                margin=3.0,
                value_at_margin=0.0,
            )
            factors.extend(np.asarray(retract_wings, dtype=np.float64).tolist())

        return factors

    def check_termination(self, physics: "mjcf.Physics") -> bool:
        step = round(physics.time() / self.control_timestep)
        self._reached_traj_end = step >= self._episode_steps
        if self._reached_traj_end:
            return True

        if np.linalg.norm(physics.data.qacc) > _TERMINAL_QACC:
            return True

        fly_failures = []
        for fly_idx, fly in enumerate(self._walkers):
            linvel = np.linalg.norm(fly.observables.velocimeter(physics))
            angvel = np.linalg.norm(fly.observables.gyro(physics))
            fly_pos, _ = fly.get_pose(physics)
            ref_idx = min(step, self._ref_qpos[fly_idx].shape[0] - 1)
            root_dist = np.linalg.norm(fly_pos - self._ref_qpos[fly_idx][ref_idx, :3])
            fly_failures.append(
                linvel > _TERMINAL_LINVEL
                or angvel > _TERMINAL_ANGVEL
                or root_dist > self._task_config.terminal_com_dist
            )

        if self._task_config.terminate_on_any_fly:
            return any(fly_failures)
        return all(fly_failures)

    def should_terminate_episode(self, physics: "mjcf.Physics") -> bool:
        del physics
        return self._should_terminate

    def get_discount(self, physics: "mjcf.Physics") -> float:
        del physics
        if self._should_terminate and not self._reached_traj_end:
            return 0.0
        return 1.0

    def name(self):
        return "MultiWalkRootImitation"


__all__ = [
    "MultiFlySpawnConfig",
    "MultiWalkRootImitation",
    "MultiWalkRootImitationTaskConfig",
]
