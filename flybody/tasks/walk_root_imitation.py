"""Root-only walking imitation task for MABe22 social-behavior trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Sequence

import numpy as np

from flybody.tasks.walk_imitation import WalkImitation
from flybody import quaternions


@dataclass
class RootImitationRewardWeights:
    """OmegaConf-friendly root-imitation reward weights.

    Keep these as explicit dataclass fields instead of a nested ``dict`` so CLI
    overrides such as ``reward.weights.linear_velocity=0.05`` work reliably with
    structured configs.
    """

    position: float = 1.0
    orientation: float = 1.0
    linear_velocity: float = 0.25
    angular_velocity: float = 0.25
    progress: float = 0.0
    leg_keypoints: float = 0.0
    action: float = 0.0


@dataclass
class RootImitationRewardConfig:
    """Reward scales for root-only FlyBody walking imitation.

    The MABe22 root dataset only contains ``root_qpos`` and ``root_qvel``.  The
    reward therefore tracks root position, orientation and velocity without any
    reference joint/site terms.  All units follow the walking environment:
    centimeters, seconds and radians.
    """

    position_std_cm: float = 0.30
    orientation_std_rad: float = 0.75
    linear_velocity_std_cm_s: float = 2.0
    angular_velocity_std_rad_s: float = 5.0
    progress_std_cm: float = 0.03
    leg_keypoint_std_cm: float = 0.08
    action_l2_std: float = 0.0
    weights: RootImitationRewardWeights | dict[str, float] = field(
        default_factory=RootImitationRewardWeights
    )

    def weight(self, name: str) -> float:
        if isinstance(self.weights, Mapping):
            return float(self.weights.get(name, 0.0))
        return float(getattr(self.weights, name, 0.0))


def _gaussian_factor(error: float, std: float, weight: float) -> float:
    if std <= 0.0 or weight <= 0.0:
        return 1.0
    return float(np.exp(-0.5 * weight * (error / std) ** 2))


class WalkRootImitation(WalkImitation):
    """Walking task that tracks only a reference root trajectory.

    This task reuses ``WalkImitation`` observations and ghost replay, so a policy
    trained for the original walking-imitation task can be warm-started without
    changing actor input/output dimensions.  The reward is explicit root-only
    tracking and does not require FlyBody joint references.
    """

    def __init__(
        self,
        *args,
        reward_config: RootImitationRewardConfig | None = None,
        **kwargs,
    ):
        kwargs["inference_mode"] = False
        super().__init__(*args, **kwargs)
        self._reward_config = reward_config or RootImitationRewardConfig()
        self._last_action_norm = 0.0
        self._pre_step_walker_xy = np.zeros(2, dtype=np.float64)
        self._leg_keypoint_site_names = (
            "claw_T1_right",
            "claw_T2_right",
            "claw_T3_right",
            "claw_T3_left",
            "claw_T2_left",
            "claw_T1_left",
        )
        self._leg_keypoint_sites = [
            self._walker.mjcf_model.find("site", site_name)
            for site_name in self._leg_keypoint_site_names
        ]
        if any(site is None for site in self._leg_keypoint_sites):
            missing = [
                site_name
                for site_name, site in zip(
                    self._leg_keypoint_site_names,
                    self._leg_keypoint_sites,
                )
                if site is None
            ]
            raise ValueError(f"Missing FlyBody leg keypoint sites: {missing}")

    def initialize_episode(
        self,
        physics: "mjcf.Physics",
        random_state: np.random.RandomState,
    ):
        super().initialize_episode(physics, random_state)
        walker_pos, _ = self._walker.get_pose(physics)
        self._pre_step_walker_xy = np.asarray(walker_pos[:2], dtype=np.float64).copy()

    def before_step(
        self,
        physics: "mjcf.Physics",
        action: np.ndarray,
        random_state: np.random.RandomState,
    ):
        walker_pos, _ = self._walker.get_pose(physics)
        self._pre_step_walker_xy = np.asarray(walker_pos[:2], dtype=np.float64).copy()
        self._last_action_norm = float(np.linalg.norm(np.nan_to_num(action)))
        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics: "mjcf.Physics") -> Sequence[float]:
        cfg = self._reward_config
        step = int(round(physics.time() / self.control_timestep))
        step = min(step, self._ref_qpos.shape[0] - 1)

        walker_pos, walker_quat = self._walker.get_pose(physics)
        ref_qpos = self._ref_qpos[step]
        ref_qvel = self._ref_qvel[step]
        walker_qvel = physics.bind(self._root_joint).qvel

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
            ref_delta_xy = ref_qpos[:2] - self._ref_qpos[step - 1, :2]
        else:
            ref_delta_xy = np.zeros(2, dtype=np.float64)
        walker_delta_xy = np.asarray(walker_pos[:2], dtype=np.float64) - self._pre_step_walker_xy
        progress_error = float(np.linalg.norm(walker_delta_xy - ref_delta_xy))
        leg_keypoint_error = self._leg_keypoint_error_cm(
            physics,
            step,
            walker_pos,
            walker_quat,
        )

        factors = [
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
                leg_keypoint_error,
                cfg.leg_keypoint_std_cm,
                cfg.weight("leg_keypoints"),
            ),
            _gaussian_factor(
                self._last_action_norm,
                cfg.action_l2_std,
                cfg.weight("action"),
            ),
        ]
        return factors

    def _leg_keypoint_error_cm(
        self,
        physics: "mjcf.Physics",
        step: int,
        walker_pos: np.ndarray,
        walker_quat: np.ndarray,
    ) -> float:
        """RMS error between model claw sites and social 2-D leg-tip keypoints.

        The reference keypoints are stored as body-egocentric ``xy`` vectors in
        centimeters, ordered like ``self._leg_keypoint_site_names``.  Comparing
        egocentric vectors makes this a posture/foot-placement term rather than
        another global root-position term.
        """

        cfg = self._reward_config
        if cfg.weight("leg_keypoints") <= 0.0:
            return 0.0
        if not hasattr(self, "_snippet") or "leg_tips_ego_xy_cm" not in self._snippet:
            return 0.0
        ref = np.asarray(self._snippet["leg_tips_ego_xy_cm"][step], dtype=np.float64)
        if ref.size == 0:
            return 0.0
        valid = np.isfinite(ref).all(axis=-1)
        if "leg_tip_valid_mask" in self._snippet:
            valid &= np.asarray(self._snippet["leg_tip_valid_mask"][step], dtype=bool)
        if not np.any(valid):
            return 0.0

        site_pos = np.asarray(
            physics.bind(self._leg_keypoint_sites).xpos,
            dtype=np.float64,
        )
        model_ego = quaternions.get_egocentric_vec(
            np.asarray(walker_pos, dtype=np.float64),
            site_pos,
            np.asarray(walker_quat, dtype=np.float64),
        )[:, :2]
        diff = model_ego[valid] - ref[valid]
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


__all__ = [
    "RootImitationRewardConfig",
    "RootImitationRewardWeights",
    "WalkRootImitation",
]
