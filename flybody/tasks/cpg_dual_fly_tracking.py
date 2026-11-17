"""Dual-fly random-walk and CPG following task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from dm_control.composer.observation import observable
from dm_env import specs

from .multi_walk import MultiFlySpawnConfig, MultiWalkRootImitation
from .walk_root_imitation import RootImitationRewardConfig


@dataclass(frozen=True)
class CPGDualFlyTrackingTaskConfig:
    """Runtime settings for the two independently CPG-driven flies."""

    spawn: MultiFlySpawnConfig = field(
        default_factory=lambda: MultiFlySpawnConfig(num_flies=2)
    )
    reward: RootImitationRewardConfig = field(default_factory=RootImitationRewardConfig)
    initial_com_distance_cm: float = 1.0
    leader_spawn_xy_range_cm: tuple[float, float] | None = None
    follower_spawn_distance_range_cm: tuple[float, float] | None = None
    follower_spawn_angle_range_rad: tuple[float, float] = (-np.pi, np.pi)
    preferred_follow_distance_cm: float = 0.35
    random_walk_speed: float = 1.0
    turn_gain: float = 0.8
    distance_gain: float = 1.2
    turn_derivative_gain: float = 0.0
    distance_derivative_gain: float = 0.0
    # Explicit tracking experiment setting, shared by collection and policy
    # evaluation. Physical straightness still depends on gait calibration.
    cpg_straight_turn_trim: float = 0.0
    command_update_interval: float = 0.25
    camera_height_cm: float = 5.0
    camera_name: str = "cpg_tracking_camera"
    camera_fovy: float = 60.0
    fly1_camera_name: str = "fly1_third_person_camera"
    fly2_camera_name: str = "fly2_third_person_camera"
    # Close camera: just above and slightly behind the fly, with a short
    # forward look-ahead so the fly's walking direction fills the view.
    third_person_distance_cm: float = 1.35
    third_person_height_cm: float = 0.45
    third_person_look_height_cm: float = 0.05
    third_person_look_ahead_cm: float = 0.35
    third_person_fovy: float = 55.0
    leader_safe_margin_cm: float = 0.9
    leader_obstacle_clearance_cm: float = 0.65
    leader_safety_turn_gain: float = 1.6
    leader_safety_speed_floor: float = 0.25
    leader_prediction_horizon_cm: float = 0.60
    leader_collision_turn_gain: float = 2.4
    future_steps: int = 1
    follower_controller: Literal["cpg", "flybrain_cpg", "external_cpg"] = "cpg"
    natural_scene: bool = False

    @classmethod
    def from_dict(cls, values):
        """Rebuild the composed configuration stored in a policy checkpoint."""
        values = dict(values)
        for name, kind in (("spawn", MultiFlySpawnConfig),
                           ("reward", RootImitationRewardConfig)):
            if isinstance(values.get(name), dict):
                values[name] = kind(**values[name])
        return cls(**values)


class CPGDualFlyTracking(MultiWalkRootImitation):
    """Fly 0 wanders; fly 1 follows and steers toward fly 0."""

    def __init__(self, *args, task_config=None, controllers, **kwargs):
        self._cpg_task_config = task_config or CPGDualFlyTrackingTaskConfig()
        gain_scale = float(kwargs.pop("position_gain_scale", 1.0))
        if gain_scale <= 0:
            raise ValueError("position_gain_scale must be positive")
        if len(controllers) != 2:
            raise ValueError("CPGDualFlyTracking requires two CPG controllers")
        self._controllers = tuple(controllers)
        self._flybrain_follower = hasattr(self._controllers[1], "step_from_physics")
        self._command_steps = max(
            1,
            int(round(self._cpg_task_config.command_update_interval / 0.002)),
        )
        self._leader_turn_target = 0.0
        self._previous_follower_yaw_error = 0.0
        self._previous_follower_distance_error = 0.0
        self._last_expert_parameters = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        self._last_expert_state = np.zeros(5, dtype=np.float32)
        self._camera = None
        self._third_person_cameras = []
        super().__init__(*args, task_config=self._as_multi_config(), **kwargs)
        if gain_scale != 1.0:
            for fly in self._walkers:
                self._scale_leg_position_gains(fly, gain_scale)
        if self.num_flies != 2:
            raise ValueError("CPGDualFlyTracking requires exactly two flies")
        self._camera = self._arena.mjcf_model.worldbody.add(
            "camera",
            name=self._cpg_task_config.camera_name,
            pos=(0.0, -1.0, self._cpg_task_config.camera_height_cm),
            # Low oblique view: horizontal image axis and world-up vertical axis.
            xyaxes=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            fovy=float(self._cpg_task_config.camera_fovy),
        )
        for name in (self._cpg_task_config.fly1_camera_name,
                     self._cpg_task_config.fly2_camera_name):
            self._third_person_cameras.append(
                self._arena.mjcf_model.worldbody.add(
                    "camera",
                    name=name,
                    pos=(0.0, -2.0, self._cpg_task_config.third_person_height_cm),
                    quat=(1.0, 0.0, 0.0, 0.0),
                    fovy=float(self._cpg_task_config.third_person_fovy),
                )
            )
        for fly_idx, fly in enumerate(self._walkers):
            fly.observables.eye_hex_pixels.enabled = fly_idx == 1
            for name in ("left_eye", "right_eye"):
                sensor = getattr(fly.observables, name, None)
                if sensor is not None:
                    sensor.enabled = fly_idx == 1
            fly.observables.add_observable(
                "cpg_joint_targets",
                observable.Generic(lambda physics, i=fly_idx: self._last_targets[i].copy()),
            )
        self._last_targets = [c.neutral_pose.reshape(-1).copy() for c in self._controllers]

    def _as_multi_config(self):
        from .multi_walk import MultiWalkRootImitationTaskConfig

        return MultiWalkRootImitationTaskConfig(
            spawn=self._cpg_task_config.spawn,
            reward=self._cpg_task_config.reward,
            future_steps=self._cpg_task_config.future_steps,
            initialize_qvel=False,
            enable_contact_observables=False,
            enable_appendage_observables=False,
            add_birdview_camera=False,
        )

    @staticmethod
    def _scale_leg_position_gains(fly, scale):
        for actuator in fly.mjcf_model.find_all("actuator"):
            if actuator.tag != "general" or actuator.name is None:
                continue
            if not any(token in actuator.name for token in ("T1_", "T2_", "T3_")):
                continue
            cls = actuator.dclass
            gain = bias = None
            while cls is not None:
                if gain is None and cls.general.gainprm is not None:
                    gain = np.asarray(cls.general.gainprm, dtype=float)
                if bias is None and cls.general.biasprm is not None:
                    bias = np.asarray(cls.general.biasprm, dtype=float)
                if gain is not None and bias is not None:
                    break
                cls = cls.parent
            if gain is not None:
                actuator.gainprm = tuple((gain * scale).tolist())
            if bias is not None:
                actuator.biasprm = tuple((bias * scale).tolist())

    @staticmethod
    def _yaw(quat):
        w, x, y, z = np.asarray(quat, dtype=float)
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    @staticmethod
    def _yaw_quat(yaw):
        return np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for idx, controller in enumerate(self._controllers):
            controller.reset()
            joints = [
                self._walkers[idx].mjcf_model.find("joint", name)
                for name in controller.flybody_joint_names
            ]
            physics.bind(joints).qpos = controller.neutral_pose.reshape(-1)
            physics.bind(joints).qvel = np.zeros(len(joints))

        angle_low, angle_high = self._cpg_task_config.follower_spawn_angle_range_rad
        if angle_high <= angle_low:
            raise ValueError(
                "follower_spawn_angle_range_rad must be increasing"
            )
        angle = float(random_state.uniform(angle_low, angle_high))
        distance_range = self._cpg_task_config.follower_spawn_distance_range_cm
        if distance_range is None:
            distance = float(self._cpg_task_config.initial_com_distance_cm)
        else:
            distance_low, distance_high = distance_range
            if distance_low <= 0 or distance_high <= distance_low:
                raise ValueError(
                    "follower_spawn_distance_range_cm must be positive and increasing"
                )
            distance = float(random_state.uniform(distance_low, distance_high))
        leader_xy_range = self._cpg_task_config.leader_spawn_xy_range_cm
        if leader_xy_range is None:
            leader_xy = np.zeros(2, dtype=np.float64)
        else:
            low, high = (float(leader_xy_range[0]), float(leader_xy_range[1]))
            if high <= low:
                raise ValueError("leader_spawn_xy_range_cm must be increasing")
            leader_xy = np.zeros(2, dtype=np.float64)
            centers = getattr(self._arena, "obstacle_centers_cm", np.zeros((0, 2)))
            radii = getattr(self._arena, "obstacle_radii_cm", np.zeros(0))
            for _ in range(64):
                candidate = random_state.uniform(low, high, size=2)
                if not len(centers) or all(
                    np.linalg.norm(candidate - center) > float(radius) + 1.0
                    for center, radius in zip(centers, radii)
                ):
                    leader_xy = candidate
                    break
        leader = np.asarray([
            leader_xy[0], leader_xy[1], self._walkers[0].upright_pose.xpos[2]
        ])
        follower = leader + distance * np.asarray([np.cos(angle), np.sin(angle), 0.0])
        leader_yaw = float(random_state.uniform(-np.pi, np.pi))
        follower_yaw = float(np.arctan2(leader[1] - follower[1], leader[0] - follower[0]))
        for idx, (pos, yaw) in enumerate(((leader, leader_yaw), (follower, follower_yaw))):
            qpos = physics.bind(self._root_joints[idx]).qpos.copy()
            qpos[:3] = pos
            qpos[3:7] = self._yaw_quat(yaw)
            physics.bind(self._root_joints[idx]).qpos = qpos
        self._leader_turn_target = 0.0
        self._previous_follower_yaw_error = 0.0
        self._previous_follower_distance_error = 0.0
        self._last_expert_parameters[:] = (0.0, 0.0, 1.0, 1.0)
        self._last_expert_state[:] = 0.0
        self._last_targets = [c.neutral_pose.reshape(-1).copy() for c in self._controllers]
        self._update_tracking_camera(physics)
        self._update_third_person_cameras(physics)

    def _update_tracking_camera(self, physics):
        positions = np.asarray([fly.get_pose(physics)[0][:2] for fly in self._walkers])
        center = positions[0] if self._cpg_task_config.natural_scene else positions.mean(axis=0)
        y_offset = 0.0 if self._cpg_task_config.natural_scene else -1.0
        camera_pos = np.asarray([center[0], center[1] + y_offset, self._cpg_task_config.camera_height_cm])
        physics.bind(self._camera).pos = camera_pos

    def _update_third_person_cameras(self, physics):
        """Place one fixed camera behind each fly and aim it at that fly."""
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        for fly, camera in zip(self._walkers, self._third_person_cameras):
            position, quat = fly.get_pose(physics)
            yaw = self._yaw(quat)
            heading = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
            target = np.asarray(position, dtype=np.float64) + np.asarray(
                [0.0, 0.0, self._cpg_task_config.third_person_look_height_cm]
            )
            target[:2] += heading[:2] * self._cpg_task_config.third_person_look_ahead_cm
            camera_position = target - heading * self._cpg_task_config.third_person_distance_cm
            camera_position[2] += self._cpg_task_config.third_person_height_cm
            forward = target - camera_position
            forward /= max(float(np.linalg.norm(forward)), 1e-9)
            # MuJoCo camera local -Z points forward. Build world rotation
            # columns for local X/Y/Z, then convert to MuJoCo wxyz order.
            camera_z = -forward
            camera_x = np.cross(up, camera_z)
            camera_x /= max(float(np.linalg.norm(camera_x)), 1e-9)
            camera_y = np.cross(camera_z, camera_x)
            rotation = np.column_stack((camera_x, camera_y, camera_z))
            from scipy.spatial.transform import Rotation

            xyzw = Rotation.from_matrix(rotation).as_quat()
            physics.bind(camera).pos = camera_position
            physics.bind(camera).quat = np.asarray(
                [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64
            )

    def _leader_command(self, physics, random_state):
        """Return a bounded speed/turn command for the leader."""
        position, quat = self._walkers[0].get_pose(physics)
        position = np.asarray(position[:2], dtype=np.float64)
        yaw = self._yaw(quat)
        cfg = self._cpg_task_config
        turn = float(self._leader_turn_target)
        speed = float(cfg.random_walk_speed)
        safety_vector = np.zeros(2, dtype=np.float64)

        # Turn toward the center before entering the wall safety margin.
        arena = self._arena
        if hasattr(arena, "bounds_cm"):
            x_min, x_max, y_min, y_max = arena.bounds_cm
            distances = np.asarray(
                [position[0] - x_min, x_max - position[0],
                 position[1] - y_min, y_max - position[1]], dtype=np.float64
            )
            nearest = float(distances.min())
            # Repulsive vector from the four walls. This uses a prediction
            # point in front of the fly, so the turn begins before contact.
            probe = position + np.asarray(
                [np.cos(yaw), np.sin(yaw)]
            ) * float(cfg.leader_prediction_horizon_cm)
            wall_clearances = np.asarray(
                [probe[0] - x_min, x_max - probe[0],
                 probe[1] - y_min, y_max - probe[1]], dtype=np.float64
            )
            wall_vectors = np.asarray(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                dtype=np.float64,
            )
            for clearance, vector in zip(wall_clearances, wall_vectors):
                if clearance < cfg.leader_safe_margin_cm:
                    safety_vector += vector * (
                        cfg.leader_safe_margin_cm - clearance
                    ) / max(cfg.leader_safe_margin_cm, 1e-6)
            if nearest < cfg.leader_safe_margin_cm:
                center_vector = -position
                desired = float(np.arctan2(center_vector[1], center_vector[0]))
                error = (desired - yaw + np.pi) % (2.0 * np.pi) - np.pi
                urgency = 1.0 + (cfg.leader_safe_margin_cm - nearest) / max(
                    cfg.leader_safe_margin_cm, 1e-6
                )
                turn += -cfg.leader_safety_turn_gain * urgency * error
                speed *= max(cfg.leader_safety_speed_floor,
                             np.clip(nearest / cfg.leader_safe_margin_cm, 0.0, 1.0))

        # Apply a soft repulsive steering field around tree trunks and rocks.
        centers = getattr(arena, "obstacle_centers_cm", np.zeros((0, 2)))
        radii = getattr(arena, "obstacle_radii_cm", np.zeros(0))
        for center, radius in zip(centers, radii):
            delta = position - np.asarray(center, dtype=np.float64)
            distance = float(np.linalg.norm(delta))
            clearance = float(radius) + cfg.leader_obstacle_clearance_cm
            probe_delta = probe - np.asarray(center, dtype=np.float64) if hasattr(arena, "bounds_cm") else delta
            probe_distance = float(np.linalg.norm(probe_delta))
            if 1e-8 < distance < clearance:
                safety_vector += delta / distance * (clearance - distance) / clearance
            elif 1e-8 < probe_distance < clearance:
                safety_vector += probe_delta / probe_distance * (clearance - probe_distance) / clearance
        if np.linalg.norm(safety_vector) > 1e-8:
            desired = float(np.arctan2(safety_vector[1], safety_vector[0]))
            error = (desired - yaw + np.pi) % (2.0 * np.pi) - np.pi
            turn += -cfg.leader_collision_turn_gain * error
            speed *= cfg.leader_safety_speed_floor
        return float(np.clip(speed, 0.0, 1.0)), float(np.clip(turn, -0.85, 0.85))

    def _commands(self, physics, random_state):
        if self._step_counter % self._command_steps == 0:
            self._leader_turn_target = float(random_state.uniform(-0.18, 0.18))
        leader_pos, leader_quat = self._walkers[0].get_pose(physics)
        follower_pos, follower_quat = self._walkers[1].get_pose(physics)
        delta = np.asarray(leader_pos[:2] - follower_pos[:2], dtype=float)
        distance = float(np.linalg.norm(delta))
        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        yaw_error = (desired_yaw - self._yaw(follower_quat) + np.pi) % (2.0 * np.pi) - np.pi
        # CPG positive turn means right; positive yaw error is a left turn.
        yaw_error_rate = (yaw_error - self._previous_follower_yaw_error) / max(
            self._cpg_task_config.command_update_interval, 1e-6
        )
        distance_error = distance - self._cpg_task_config.preferred_follow_distance_cm
        distance_error_rate = (distance_error - self._previous_follower_distance_error) / max(
            self._cpg_task_config.command_update_interval, 1e-6
        )
        follower_turn = float(np.clip(
            -self._cpg_task_config.turn_gain * yaw_error
            - self._cpg_task_config.turn_derivative_gain * yaw_error_rate,
            -0.35, 0.35,
        ))
        follower_speed = float(np.clip(
            self._cpg_task_config.distance_gain * distance_error
            + self._cpg_task_config.distance_derivative_gain * distance_error_rate,
            0.0, 1.0,
        ))
        self._previous_follower_yaw_error = yaw_error
        self._previous_follower_distance_error = distance_error
        del leader_pos, leader_quat
        leader_speed, leader_turn = self._leader_command(physics, random_state)
        self._controllers[0].set_command(leader_speed, leader_turn)
        if not self._flybrain_follower:
            self._controllers[1].set_command(follower_speed, follower_turn)
        self._last_expert_parameters[:] = (follower_speed, follower_turn, 1.0, 1.0)
        self._last_expert_state[:] = (
            delta[0], delta[1], distance, yaw_error, follower_speed
        )

    def before_step(self, physics, action, random_state):
        self._step_counter += 1
        self._commands(physics, random_state)
        for idx, (fly, controller) in enumerate(zip(self._walkers, self._controllers)):
            if idx == 1 and self._flybrain_follower:
                generated = np.asarray(
                    controller.step_from_physics(
                        physics, fly, other_fly=self._walkers[0]
                    ), dtype=np.float64
                ).reshape(-1)
            elif idx == 1 and self._cpg_task_config.follower_controller == "external_cpg":
                parameters = np.asarray(action, dtype=np.float64).reshape(-1)
                if parameters.shape != (4,):
                    raise ValueError(
                        "external_cpg action must be [speed, turn, amplitude, frequency]"
                    )
                controller.set_command(*parameters)
                generated = np.asarray(controller.step(), dtype=np.float64).reshape(-1)
            else:
                generated = np.asarray(controller.step(), dtype=np.float64).reshape(-1)
            if generated.shape != (48,):
                raise ValueError(f"CPG {idx} must emit 48 joint targets, got {generated.shape}")
            self._last_targets[idx] = generated.copy()
            fly_action = np.zeros(self._single_fly_action_dim(), dtype=np.float64)
            fly_action[np.asarray(fly._action_indices["legs"], dtype=int)] = generated
            self._apply_walker_action(physics, fly, idx, fly_action)
        self._update_tracking_camera(physics)
        self._update_third_person_cameras(physics)

    def action_spec(self, physics):
        del physics
        if self._cpg_task_config.follower_controller == "external_cpg":
            return specs.BoundedArray(
                shape=(4,), dtype=np.float32,
                minimum=np.asarray([0.0, -0.5, 0.7, 0.8], dtype=np.float32),
                maximum=np.asarray([1.0, 0.5, 1.3, 1.2], dtype=np.float32),
                name="follower_cpg_parameters",
            )
        return specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=np.asarray([-1.0], dtype=np.float32),
            maximum=np.asarray([1.0], dtype=np.float32), name="idle",
        )

    def check_termination(self, physics):
        # Root references are dummy placeholders in this autonomous CPG task;
        # terminating on root-reference error would stop on the first step.
        # The Composer time limit remains the episode boundary.
        del physics
        self._reached_traj_end = False
        return False

    @property
    def cpg_controllers(self):
        return self._controllers

    @property
    def expert_parameters(self) -> np.ndarray:
        """Current PD expert target: speed, turn, amplitude, frequency."""
        return self._last_expert_parameters.copy()

    def preview_expert_parameters(self, physics) -> np.ndarray:
        """Compute the PD teacher command without advancing task state."""
        leader_pos, _ = self._walkers[0].get_pose(physics)
        follower_pos, follower_quat = self._walkers[1].get_pose(physics)
        delta = np.asarray(leader_pos[:2] - follower_pos[:2], dtype=float)
        distance = float(np.linalg.norm(delta))
        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        yaw_error = (desired_yaw - self._yaw(follower_quat) + np.pi) % (2.0 * np.pi) - np.pi
        yaw_rate = (yaw_error - self._previous_follower_yaw_error) / max(
            self._cpg_task_config.command_update_interval, 1e-6
        )
        distance_error = distance - self._cpg_task_config.preferred_follow_distance_cm
        distance_rate = (distance_error - self._previous_follower_distance_error) / max(
            self._cpg_task_config.command_update_interval, 1e-6
        )
        turn = float(np.clip(
            -self._cpg_task_config.turn_gain * yaw_error
            - self._cpg_task_config.turn_derivative_gain * yaw_rate,
            -0.35, 0.35,
        ))
        speed = float(np.clip(
            self._cpg_task_config.distance_gain * distance_error
            + self._cpg_task_config.distance_derivative_gain * distance_rate,
            0.0, 1.0,
        ))
        return np.asarray([speed, turn, 1.0, 1.0], dtype=np.float32)

    @property
    def expert_state(self) -> np.ndarray:
        """Current relative state: dx, dy, distance, yaw_error, speed."""
        return self._last_expert_state.copy()

    @property
    def third_person_cameras(self):
        """MJCF camera elements ordered as (Fly 1, Fly 2)."""
        return tuple(self._third_person_cameras)

    @property
    def third_person_camera_names(self):
        return (self._cpg_task_config.fly1_camera_name,
                self._cpg_task_config.fly2_camera_name)


__all__ = ["CPGDualFlyTracking", "CPGDualFlyTrackingTaskConfig"]
