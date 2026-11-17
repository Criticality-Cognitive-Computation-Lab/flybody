"""Create examples of flight and walking task environments for fruitfly."""

from typing import Callable, Literal, Sequence

import numpy as np

# import sys
from dm_control import composer, mujoco
from dm_control.locomotion.arenas import floors

# sys.path.append('/home/xiaowei/fly_unnamed_torch')
from flybody.fruitfly import fruitfly
from flybody.tasks.arenas.ball import BallFloor
from flybody.tasks.arenas.hills import SineBumps, SineTrench
from flybody.tasks.arenas.natural_10cm import NaturalTenCmArena
from flybody.tasks.arenas.olfaction import StaticOlfactionArena
from flybody.tasks.arenas.plume import (
    HDF5PlumeArena2D,
    OdorPlume2DConfig,
    PlumeNavigationArena2D,
)
from flybody.tasks.constants import _WALK_CONTROL_TIMESTEP
from flybody.tasks.cpg import FlyBodyCPGConfig, FlyBodyCPGController
from flybody.tasks.cpg_dual_fly_tracking import (
    CPGDualFlyTracking,
    CPGDualFlyTrackingTaskConfig,
)
from flybody.tasks.flight_imitation import FlightImitationWBPG
from flybody.tasks.multi_walk import (
    MultiFlySpawnConfig,
    MultiWalkRootImitation,
    MultiWalkRootImitationTaskConfig,
)
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.template_task import TemplateTask
from flybody.tasks.trajectory_loaders import (
    HDF5FlightTrajectoryLoader,
    HDF5SynchronizedMultiFlyTrajectoryLoader,
    HDF5WalkingTrajectoryLoader,
    InferenceFlightTrajectoryLoader,
    InferenceWalkingTrajectoryLoader,
)
from flybody.tasks.vision_flight import VisionFlightImitationWBPG
from flybody.tasks.walk_cpg import WalkCPG
from flybody.tasks.walk_imitation import WalkImitation
from flybody.tasks.walk_on_ball import WalkOnBall
from flybody.tasks.walk_root_imitation import (
    RootImitationRewardConfig,
    WalkRootImitation,
)


def walk_imitation(ref_path: str | None = None,
                   force_actuators: bool = False,
                   disable_wings: bool = True,
                   traj_indices: Sequence[int] | None = None,
                   random_state: np.random.RandomState | None = None,
                   terminal_com_dist: float = 0.3,
                   joint_filter: float = 0.01,
                   inference_mode: bool | None = None):
    """Requires a fruitfly to track a reference walking fly.

    Args:
        ref_path: Path to reference trajectory dataset. If not provided, task
            will run in inference mode with InferenceWalkingTrajectoryLoader,
            without loading actual walking dataset.
        force_actuators: Whether to use force or position actuators.
        disable_wings: Whether to retract and disable wings. This includes
            removing wing DoFs, actuators, and sensors.
        traj_indices: List of trajectory indices to use, e.g. for train/test
            splitting etc. If None, use all available trajectories.
        random_state: Random state for reproducibility.
        terminal_com_dist: Episode will be terminated when distance from model
            CoM to ghost CoM exceeds terminal_com_dist. Can be float('inf').
        joint_filter: Timescale of filter for joint actuators. 0: disabled.
        inference_mode: If True, skip full-body reference reward terms. This is
            useful for root-only reference trajectories, e.g. MABe22
            social-behavior trajectories with no FlyBody joint references. If
            None, defaults to True when ``ref_path`` is not provided and False
            otherwise.

    Returns:
        Environment for walking tracking task.
    """
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = floors.Floor()
    # Initialize a walking trajectory loader.
    if ref_path is not None:
        inference_mode = False if inference_mode is None else inference_mode
        traj_generator = HDF5WalkingTrajectoryLoader(
            path=ref_path, random_state=random_state, traj_indices=traj_indices)
    else:
        inference_mode = True if inference_mode is None else inference_mode
        traj_generator = InferenceWalkingTrajectoryLoader()
    # Build a task that rewards the agent for tracking a walking ghost.
    time_limit = 10.0
    task = WalkImitation(walker=walker,
                         arena=arena,
                         traj_generator=traj_generator,
                         terminal_com_dist=terminal_com_dist,
                         mocap_joint_names=traj_generator.get_joint_names(),
                         mocap_site_names=traj_generator.get_site_names(),
                         inference_mode=inference_mode,
                         force_actuators=force_actuators,
                         disable_wings=disable_wings,
                         joint_filter=joint_filter,
                         future_steps=64,
                         time_limit=time_limit)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def walk_root_imitation(
    ref_path: str,
    force_actuators: bool = False,
    disable_wings: bool = True,
    traj_indices: Sequence[int] | None = None,
    random_state: np.random.RandomState | None = None,
    terminal_com_dist: float = 1.0,
    joint_filter: float = 0.01,
    reward_config: RootImitationRewardConfig | None = None,
    randomize_start_step: bool = True,
    min_active_window_path_cm: float = 0.0,
    active_window_max_tries: int = 64,
    time_limit: float = 10.0,
):
    """Root-only walking imitation environment for MABe22 trajectories.

    The actor observation/action interface is intentionally kept identical to
    ``walk_imitation`` so an existing walking policy can be used as a warm
    start.  The reward only tracks the reference root position/orientation and
    root velocity, so the HDF5 reference may omit joint/site trajectories.

    Args:
        ref_path: Path to root-only HDF5 dataset, e.g.
            ``mabe22_root_train_world/flybody_root_trajectories.hdf5``.
        force_actuators: Whether to use force or position actuators.
        disable_wings: Whether to retract and disable wings.
        traj_indices: Optional trajectory subset.
        random_state: Random state for reproducibility.
        terminal_com_dist: Episode terminates if current root is farther than
            this from the reference root. Use ``float('inf')`` to disable.
        joint_filter: Timescale of filter for joint actuators. 0 disables it.
        reward_config: Optional root-only reward scale/weight config.
        randomize_start_step: If True, sample a random temporal window from
            each long MABe22 trajectory.
        min_active_window_path_cm: If > 0, reject random windows whose xy root
            path over the rollout horizon is shorter than this threshold.
            This can reduce oversampling of nearly-stationary MABe22 windows.
        active_window_max_tries: Maximum rejection-sampling attempts before
            falling back to uniform random sampling.
        time_limit: Episode time limit in seconds.

    Returns:
        Environment for root-only walking tracking.
    """

    walker = fruitfly.FruitFly
    arena = floors.Floor()
    future_steps = 64
    # Keep enough reference look-ahead for the full episode.
    clip_length_steps = (
        round(time_limit / _WALK_CONTROL_TIMESTEP) + future_steps + 2
    )
    active_window_steps = round(time_limit / _WALK_CONTROL_TIMESTEP) + 1
    traj_generator = HDF5WalkingTrajectoryLoader(
        path=ref_path,
        random_state=random_state,
        traj_indices=traj_indices,
        randomize_start_step=randomize_start_step,
        clip_length_steps=clip_length_steps,
        min_active_window_path_cm=min_active_window_path_cm,
        active_window_steps=active_window_steps,
        active_window_max_tries=active_window_max_tries,
    )

    task = WalkRootImitation(
        walker=walker,
        arena=arena,
        traj_generator=traj_generator,
        terminal_com_dist=terminal_com_dist,
        mocap_joint_names=traj_generator.get_joint_names(),
        mocap_site_names=traj_generator.get_site_names(),
        reward_config=reward_config,
        force_actuators=force_actuators,
        disable_wings=disable_wings,
        joint_filter=joint_filter,
        future_steps=future_steps,
        time_limit=time_limit,
    )

    return composer.Environment(
        time_limit=time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def multi_walk_root_imitation(
    ref_path: str | None = None,
    num_flies: int = 2,
    spawn_config: MultiFlySpawnConfig | None = None,
    task_config: MultiWalkRootImitationTaskConfig | None = None,
    force_actuators: bool = False,
    disable_wings: bool = True,
    traj_indices: Sequence[int] | None = None,
    random_state: np.random.RandomState | None = None,
    terminal_com_dist: float = 1.0,
    joint_filter: float = 0.01,
    reward_config: RootImitationRewardConfig | None = None,
    randomize_start_step: bool = True,
    min_active_window_path_cm: float = 0.0,
    active_window_max_tries: int = 64,
    time_limit: float = 10.0,
    future_steps: int = 16,
    reward_aggregation: Literal["mean", "sum", "product"] = "mean",
    terminate_on_any_fly: bool = True,
    enable_contact_observables: bool = True,
    enable_appendage_observables: bool = True,
    add_birdview_camera: bool = True,
    birdview_camera_height_cm: float = 5.0,
    birdview_camera_fovy: float = 55.0,
    trajectory_sites: bool = False,
    trajectory_site_stride: int = 10,
    trajectory_site_size_cm: float = 0.015,
    trajectory_site_alpha: float = 0.75,
    synchronized_multi_fly_ref: bool = False,
    synchronized_center_mode: Literal[
        "first_frame_centroid",
        "first_fly",
        "none",
    ] = "first_frame_centroid",
    synchronized_active_metric: Literal["min", "mean", "max"] = "mean",
    synchronized_fly_slots: Sequence[int] | None = None,
):
    """Multi-fly root-only walking imitation environment.

    This creates several independently controllable ``FruitFly`` walkers in one
    MuJoCo scene.  Each fly receives its own root reference trajectory and the
    flattened action vector is concatenated in fly order:
    ``[fly0_action, fly1_action, ...]``.  No ghost flies are added; optional
    colored reference trajectory sites can be enabled for diagnostic videos.

    Args:
        ref_path: Optional root-trajectory HDF5 path.  If ``None``, use the
            lightweight synthetic walking trajectory loader for smoke tests.
        num_flies: Number of controllable flies. Ignored when ``spawn_config`` or
            ``task_config`` provides a spawn config.
        spawn_config: Optional dataclass controlling spawn layout/offsets.
        task_config: Optional full task dataclass.  Explicit keyword options
            below are folded into a config when this is ``None``.
        force_actuators: Whether to use force or position actuators.
        disable_wings: Whether to retract and disable wings.
        traj_indices: Optional trajectory subset.
        random_state: Random state for reproducibility.
        terminal_com_dist: Per-fly root distance termination threshold.
        joint_filter: Timescale of filter for joint actuators. 0 disables it.
        reward_config: Optional root-only reward config.
        randomize_start_step: If True, sample random HDF5 windows.
        min_active_window_path_cm: Optional active-window rejection threshold.
        active_window_max_tries: Maximum active-window sampling attempts.
        time_limit: Episode time limit in seconds.
        future_steps: Root-reference look-ahead exposed per fly.
        reward_aggregation: How to aggregate per-fly rewards: mean/sum/product.
        terminate_on_any_fly: If True, terminate when any fly fails.
        enable_contact_observables: Enable force/touch observations per fly.
        enable_appendage_observables: Enable appendage-position observations.
        add_birdview_camera: Add a fixed top-down camera named ``birdview``.
        birdview_camera_height_cm: Height of the top-down camera.
        birdview_camera_fovy: Field of view of the top-down camera.
        trajectory_sites: If True, draw colored reference root trajectory
            points for each fly. Red/green/blue correspond to fly0/1/2.
        trajectory_site_stride: Reference-step interval between drawn points.
        trajectory_site_size_cm: Site radius for trajectory points.
        trajectory_site_alpha: Alpha for trajectory points.
        synchronized_multi_fly_ref: If True, ``ref_path`` is interpreted as a
            social-behavior segment HDF5 with synchronized ``root_qpos`` arrays
            shaped ``(T, flies, 7)``.  One segment/window is sampled per
            episode and shared by all flies.
        synchronized_center_mode: Common XY centering mode for synchronized
            segment references.  The default preserves relative positions while
            placing the initial group centroid near the arena origin.
        synchronized_active_metric: Reduction over per-fly reference path
            lengths for active-window rejection sampling.
        synchronized_fly_slots: Optional selected fly slots in each synchronized
            segment.

    Returns:
        Composer environment for multi-fly root walking imitation.
    """

    if task_config is None:
        if spawn_config is None:
            spawn_config = MultiFlySpawnConfig(num_flies=num_flies)
        task_config = MultiWalkRootImitationTaskConfig(
            spawn=spawn_config,
            reward=reward_config or RootImitationRewardConfig(),
            terminal_com_dist=terminal_com_dist,
            reward_aggregation=reward_aggregation,
            terminate_on_any_fly=terminate_on_any_fly,
            future_steps=future_steps,
            enable_contact_observables=enable_contact_observables,
            enable_appendage_observables=enable_appendage_observables,
            add_birdview_camera=add_birdview_camera,
            birdview_camera_height_cm=birdview_camera_height_cm,
            birdview_camera_fovy=birdview_camera_fovy,
            trajectory_sites=trajectory_sites,
            trajectory_site_stride=trajectory_site_stride,
            trajectory_site_size_cm=trajectory_site_size_cm,
            trajectory_site_alpha=trajectory_site_alpha,
        )

    walker = fruitfly.FruitFly
    arena = floors.Floor()

    if ref_path is not None:
        clip_length_steps = (
            round(time_limit / _WALK_CONTROL_TIMESTEP)
            + task_config.future_steps
            + 2
        )
        active_window_steps = round(time_limit / _WALK_CONTROL_TIMESTEP) + 1
        if synchronized_multi_fly_ref:
            traj_generator = HDF5SynchronizedMultiFlyTrajectoryLoader(
                path=ref_path,
                random_state=random_state,
                traj_indices=traj_indices,
                randomize_start_step=randomize_start_step,
                clip_length_steps=clip_length_steps,
                min_active_window_path_cm=min_active_window_path_cm,
                active_window_steps=active_window_steps,
                active_window_max_tries=active_window_max_tries,
                fly_slots=synchronized_fly_slots,
                center_mode=synchronized_center_mode,
                active_metric=synchronized_active_metric,
            )
        else:
            traj_generator = HDF5WalkingTrajectoryLoader(
                path=ref_path,
                random_state=random_state,
                traj_indices=traj_indices,
                randomize_start_step=randomize_start_step,
                clip_length_steps=clip_length_steps,
                min_active_window_path_cm=min_active_window_path_cm,
                active_window_steps=active_window_steps,
                active_window_max_tries=active_window_max_tries,
            )
    else:
        traj_generator = InferenceWalkingTrajectoryLoader()

    task = MultiWalkRootImitation(
        walker=walker,
        arena=arena,
        traj_generator=traj_generator,
        task_config=task_config,
        force_actuators=force_actuators,
        disable_wings=disable_wings,
        joint_filter=joint_filter,
        time_limit=time_limit,
    )

    return composer.Environment(
        time_limit=time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def walk_olfaction(
    random_state: np.random.RandomState | None = None,
    force_actuators: bool = False,
    disable_wings: bool = True,
    joint_filter: float = 0.01,
    terminal_com_dist: float = float('inf'),
):
    """Simple walking environment with static olfactory observations."""
    walker = fruitfly.FruitFly
    arena = StaticOlfactionArena()
    traj_generator = InferenceWalkingTrajectoryLoader()
    time_limit = 10.0
    task = WalkImitation(
        walker=walker,
        arena=arena,
        traj_generator=traj_generator,
        terminal_com_dist=terminal_com_dist,
        mocap_joint_names=traj_generator.get_joint_names(),
        mocap_site_names=traj_generator.get_site_names(),
        inference_mode=True,
        force_actuators=force_actuators,
        disable_wings=disable_wings,
        joint_filter=joint_filter,
        future_steps=64,
        time_limit=time_limit,
    )
    return composer.Environment(
        time_limit=time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def walk_olfaction_plume(
    random_state: np.random.RandomState | None = None,
    force_actuators: bool = False,
    disable_wings: bool = True,
    joint_filter: float = 0.01,
    terminal_com_dist: float = float('inf'),
    plume_config: OdorPlume2DConfig | None = None,
    include_vision: bool = True,
):
    """Walking environment with a static 2D odor plume field.

    If ``include_vision`` is True, the returned observations also include:
    - ``walker/left_eye``
    - ``walker/right_eye``
    - ``walker/eye_hex_pixels``
    """
    walker = fruitfly.FruitFly
    arena = PlumeNavigationArena2D(plume_config=plume_config)
    traj_generator = InferenceWalkingTrajectoryLoader()
    time_limit = 10.0
    task = WalkImitation(
        walker=walker,
        arena=arena,
        traj_generator=traj_generator,
        terminal_com_dist=terminal_com_dist,
        mocap_joint_names=traj_generator.get_joint_names(),
        mocap_site_names=traj_generator.get_site_names(),
        inference_mode=True,
        force_actuators=force_actuators,
        disable_wings=disable_wings,
        joint_filter=joint_filter,
        future_steps=64,
        time_limit=time_limit,
    )
    task._walker.observables.left_eye.enabled = bool(include_vision)
    task._walker.observables.right_eye.enabled = bool(include_vision)
    task._walker.observables.eye_hex_pixels.enabled = bool(include_vision)
    return composer.Environment(
        time_limit=time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def walk_olfaction_plume_dataset(
    plume_dataset_path: str | None = None,
    random_state: np.random.RandomState | None = None,
    force_actuators: bool = False,
    disable_wings: bool = True,
    joint_filter: float = 0.01,
    terminal_com_dist: float = float('inf'),
    active_odor_name: str = "carbon dioxide",
    plume_speed: float = 1.0,
    arena_size: tuple[float, float] = (1.2, 0.8),
    top_camera_height: float | None = None,
    include_vision: bool = True,
):
    """Walking environment backed by an HDF5 odor plume dataset.

    If ``include_vision`` is True, the returned observations also include:
    - ``walker/left_eye``
    - ``walker/right_eye``
    - ``walker/eye_hex_pixels``
    """
    if plume_dataset_path is None:
        raise ValueError("plume_dataset_path must point to an external HDF5 dataset")
    walker = fruitfly.FruitFly
    arena = HDF5PlumeArena2D(
        dataset_path=plume_dataset_path,
        arena_size=arena_size,
        active_odor_name=active_odor_name,
        plume_speed=plume_speed,
        top_camera_height=top_camera_height,
    )
    traj_generator = InferenceWalkingTrajectoryLoader()
    time_limit = 10.0
    task = WalkImitation(
        walker=walker,
        arena=arena,
        traj_generator=traj_generator,
        terminal_com_dist=terminal_com_dist,
        mocap_joint_names=traj_generator.get_joint_names(),
        mocap_site_names=traj_generator.get_site_names(),
        inference_mode=True,
        force_actuators=force_actuators,
        disable_wings=disable_wings,
        joint_filter=joint_filter,
        future_steps=64,
        time_limit=time_limit,
    )
    task._walker.observables.left_eye.enabled = bool(include_vision)
    task._walker.observables.right_eye.enabled = bool(include_vision)
    task._walker.observables.eye_hex_pixels.enabled = bool(include_vision)
    return composer.Environment(
        time_limit=time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def walk_cpg(*, template_path: str,
             trajectory_id: str = "005",
             start_frame: int = 36,
             period_frames: int = 37,
             average_cycles: int = 4,
             random_state: np.random.RandomState | None = None,
             time_limit: float = 10.0,
             cpg_timestep: float = 2e-3,
             physics_timestep: float = 2e-4,
             cpg_frequency: float | None = None,
             cpg_seed: int = 0,
             cpg_amplitude_scale: float = 1.0,
             cpg_initial_phase: float = 0.0,
             cpg_randomize_phase: bool = False,
             cpg_straight_turn_trim: float = 0.05,
             position_gain_scale: float = 2.0,
             force_actuators: bool = False,
             disable_wings: bool = True,
             joint_filter: float = 0.0,
             claw_friction: float | None = 1.0):
    """Create a FlyBody walking environment driven by a native gait template."""
    if force_actuators:
        raise ValueError("FlyBody template CPG requires position actuators")
    controller = FlyBodyCPGController(
        FlyBodyCPGConfig(
            template_path=template_path,
            trajectory_id=trajectory_id,
            start_frame=start_frame,
            period_frames=period_frames,
            average_cycles=average_cycles,
            timestep=cpg_timestep,
            frequency=cpg_frequency,
            amplitude_scale=cpg_amplitude_scale,
            initial_phase=cpg_initial_phase,
            randomize_phase=cpg_randomize_phase,
            straight_turn_trim=cpg_straight_turn_trim,
            seed=cpg_seed,
        )
    )
    task = WalkCPG(
        walker=fruitfly.FruitFly,
        arena=floors.Floor(),
        cpg_controller=controller,
        claw_friction=claw_friction,
        position_gain_scale=position_gain_scale,
        force_actuators=False,
        disable_wings=disable_wings,
        joint_filter=joint_filter,
        time_limit=time_limit,
    )
    task.set_timesteps(
        control_timestep=float(cpg_timestep),
        physics_timestep=float(physics_timestep),
    )
    return composer.Environment(
        time_limit=time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def cpg_dual_fly_tracking(
    *,
    task_config: CPGDualFlyTrackingTaskConfig | None = None,
    follower_controller_instance=None,
    template_path: str,
    trajectory_id: str = "005",
    start_frame: int = 36,
    period_frames: int = 37,
    average_cycles: int = 4,
    random_state: np.random.RandomState | None = None,
    time_limit: float = 30.0,
    cpg_timestep: float = 2e-3,
    physics_timestep: float = 2e-4,
    cpg_frequency: float | None = None,
    cpg_amplitude_scale: float = 1.0,
    position_gain_scale: float = 2.0,
    cpg_seed: int = 0,
    disable_wings: bool = True,
    joint_filter: float = 0.0,
):
    """Create a random-walk leader and a native or injected CPG follower."""
    config = task_config or CPGDualFlyTrackingTaskConfig()
    if config.spawn.num_flies != 2:
        raise ValueError("cpg_dual_fly_tracking requires spawn.num_flies=2")
    if not np.isclose(cpg_timestep, 2e-3):
        raise ValueError("The task command scheduler expects a 2 ms CPG timestep")

    def make_controller(seed):
        return FlyBodyCPGController(
            FlyBodyCPGConfig(
                template_path=template_path,
                trajectory_id=trajectory_id,
                start_frame=start_frame,
                period_frames=period_frames,
                average_cycles=average_cycles,
                timestep=cpg_timestep,
                frequency=cpg_frequency,
                amplitude_scale=cpg_amplitude_scale,
                straight_turn_trim=config.cpg_straight_turn_trim,
                seed=seed,
            )
        )

    if follower_controller_instance is not None:
        follower = follower_controller_instance
    elif config.follower_controller == "cpg":
        follower = make_controller(cpg_seed + 1)
    else:
        raise ValueError(
            "follower_controller_instance is required for neural or external CPG "
            "followers"
        )
    task = CPGDualFlyTracking(
        walker=fruitfly.FruitFly,
        arena=NaturalTenCmArena() if config.natural_scene else floors.Floor(),
        traj_generator=InferenceWalkingTrajectoryLoader(),
        task_config=config,
        controllers=[make_controller(cpg_seed), follower],
        force_actuators=False,
        disable_wings=disable_wings,
        joint_filter=joint_filter,
        position_gain_scale=position_gain_scale,
        time_limit=time_limit,
    )
    if physics_timestep <= 0.0 or cpg_timestep <= 0.0:
        raise ValueError("cpg_timestep and physics_timestep must be positive")
    if not np.isclose(cpg_timestep / physics_timestep,
                      round(cpg_timestep / physics_timestep)):
        raise ValueError("cpg_timestep must be divisible by physics_timestep")
    task.set_timesteps(
        control_timestep=float(cpg_timestep),
        physics_timestep=float(physics_timestep),
    )
    return composer.Environment(
        time_limit=time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def flight_imitation(ref_path: str | None = None,
                     wpg_pattern_path: str | None = None,
                     force_actuators: bool = False,
                     disable_legs: bool = True,
                     traj_indices: Sequence[int] | None = None,
                     randomize_start_step: bool = True,
                     joint_filter: float = 0.,
                     future_steps: int = 5,
                     random_state: np.random.RandomState | None = None,
                     terminal_com_dist: float = 2.0):
    """Requires a fruitfly to track a flying reference.

    Args:
        ref_path: Path to reference trajectory dataset. If None, task will
            run with InferenceFlightTrajectoryLoader, without loading actual
            flight dataset.
        wpg_pattern_path: Path to baseline wing beat pattern for WPG. If None,
            a simple approximate wing pattern will be generated in the
            WingBeatPatternGenerator class (could be used for testing).
        force_actuators: Whether to use force actuators or position actuators
            for everything other than wings. Wings always use force actuators.
        disable_legs: Whether to retract and disable legs. This includes
            removing leg DoFs, actuators, and sensors.
        traj_indices: List of trajectory indices to use, e.g. for train/test
            splitting etc. If None, use all available trajectories.
        randomize_start_step: Whether to select random start point in each
            get_trajectory call in trajectory loader.
        joint_filter: Time constant of joint actuator filter. 0: disabled.
        future_steps: Number of future steps of reference trajectory to provide
            as observables. Zero means only the current step is used.
        random_state: Random state for reproducibility.
        terminal_com_dist: Episode will be terminated when distance from model
            CoM to ghost CoM exceeds terminal_com_dist. Can be float('inf').

    Returns:
        Environment for flight tracking task.
    """
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = floors.Floor()
    # Initialize wing pattern generator and flight trajectory loader.
    wbpg = WingBeatPatternGenerator(base_pattern_path=wpg_pattern_path)
    if ref_path is not None:
        traj_generator = HDF5FlightTrajectoryLoader(
            path=ref_path,
            traj_indices=traj_indices,
            randomize_start_step=randomize_start_step,
            random_state=random_state)
    else:
        traj_generator = InferenceFlightTrajectoryLoader()
    # Build the task.
    time_limit = 0.6
    task = FlightImitationWBPG(walker=walker,                                      # Fruitfly model.
                               arena=arena,                                        # Arena.
                               wbpg=wbpg,                                          # Wing beat pattern generator.
                               traj_generator=traj_generator,                      # Trajectory loader.
                               terminal_com_dist=terminal_com_dist,                # Terminal distance.
                               initialize_qvel=True,                               # Initialize velocity.
                               force_actuators=force_actuators,                    # Force actuators.
                               disable_legs=disable_legs,                          # remove leg DoFs, actuators, and sensors.
                               time_limit=time_limit,                              # Time limit beyond which episode is forced to terminate.
                               joint_filter=joint_filter,                          # Timescale of filter for joint actuators. 0: disabled.
                               future_steps=future_steps)                          # Number of future steps of reference trajectory to provide as observables. Zero means only the current step is used.

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def flight_imitation_sensory(ref_path: str | None = None,
                             wpg_pattern_path: str | None = None,
                             force_actuators: bool = False,
                             disable_legs: bool = True,
                             traj_indices: Sequence[int] | None = None,
                             randomize_start_step: bool = True,
                             joint_filter: float = 0.,
                             future_steps: int = 5,
                             random_state: np.random.RandomState | None = None,
                             terminal_com_dist: float = 2.0):
    """Flight imitation with sensory arena (vision + olfaction enabled)."""
    walker = fruitfly.FruitFly
    arena = StaticOlfactionArena()
    wbpg = WingBeatPatternGenerator(base_pattern_path=wpg_pattern_path)
    if ref_path is not None:
        traj_generator = HDF5FlightTrajectoryLoader(
            path=ref_path,
            traj_indices=traj_indices,
            randomize_start_step=randomize_start_step,
            random_state=random_state)
    else:
        traj_generator = InferenceFlightTrajectoryLoader()
    time_limit = 0.6
    task = FlightImitationWBPG(walker=walker,
                               arena=arena,
                               wbpg=wbpg,
                               traj_generator=traj_generator,
                               terminal_com_dist=terminal_com_dist,
                               initialize_qvel=True,
                               force_actuators=force_actuators,
                               disable_legs=disable_legs,
                               joint_filter=joint_filter,
                               time_limit=time_limit,
                               future_steps=future_steps)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def walk_on_ball(force_actuators: bool = False,
                 disable_wings: bool = True,
                 random_state: np.random.RandomState | None = None):
    """Requires a tethered fruitfly to walk on a floating ball.

    Args:
        force_actuators: Whether to use force or position actuators.
        disable_wings: Whether to retract and disable wings. This includes
            removing wing DoFs, actuators, and sensors.
        random_state: Random state for reproducibility.

    Returns:
        Environment for fly walking on ball.
    """
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = BallFloor(ball_pos=(-0.05, 0, -0.419),
                      ball_radius=0.454,
                      ball_density=0.0025,
                      skybox=False)
    # Build a task that rewards the agent for tracking a walking ghost.
    time_limit = 2.
    task = WalkOnBall(walker=walker,
                      arena=arena,
                      force_actuators=force_actuators,
                      disable_wings=disable_wings,
                      joint_filter=0.01,
                      adhesion_filter=0.007,
                      time_limit=time_limit)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def vision_guided_flight(wpg_pattern_path: str | None = None,
                         bumps_or_trench: str = 'bumps',
                         force_actuators: bool = False,
                         disable_legs: bool = True,
                         random_state: np.random.RandomState | None = None,
                         joint_filter: float = 0.,
                         **kwargs_arena):
    """Vision-guided flight tasks: 'bumps' and 'trench'.

    Args:
        wpg_pattern_path: Path to baseline wing beat pattern for WPG. If None,
            a simple approximate wing pattern will be generated in the
            WingBeatPatternGenerator class (could be used for testing).
        bumps_or_trench: Whether to create 'bumps' or 'trench' vision task.
        force_actuators: Whether to use force actuators or position actuators
            for everything other than wings. Wings always use force actuators.
        disable_legs: Whether to retract and disable legs. This includes
            removing leg DoFs, actuators, and sensors.
        random_state: Random state for reproducibility.
        joint_filter: Timescale of filter for joint actuators. 0: disabled.
        kwargs_arena: kwargs to be passed on to arena.

    Returns:
        Environment for vision-guided flight task.
    """

    if bumps_or_trench == 'bumps':
        arena = SineBumps
    elif bumps_or_trench == 'trench':
        arena = SineTrench
    else:
        raise ValueError("Only 'bumps' and 'trench' terrains are supported.")
    # Build fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = arena(**kwargs_arena)
    # Initialize a wing beat pattern generator.
    wbpg = WingBeatPatternGenerator(base_pattern_path=wpg_pattern_path)
    # Build task.
    time_limit = 0.4
    task = VisionFlightImitationWBPG(walker=walker,
                                     arena=arena,
                                     wbpg=wbpg,
                                     time_limit=time_limit,
                                     force_actuators=force_actuators,
                                     disable_legs=disable_legs,
                                     joint_filter=joint_filter,
                                     floor_contacts=True,
                                     floor_contacts_fatal=True)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def template_task(random_state: np.random.RandomState | None = None,
                  force_actuators: bool = False,
                  disable_wings: bool = True,
                  joint_filter: float = 0.01,
                  adhesion_filter: float = 0.007,
                  time_limit: float = 1.,
                  mjcb_control: Callable | None = None,
                  observables_options: dict | None = None,
                  action_corruptor: Callable | None = None):
    """An empty no-op walking task for testing.

    Args:
        random_state: Random state for reproducibility.
        force_actuators: Whether to use force or position actuators.
        disable_wings: Whether to retract and disable wings. This includes
            removing wing DoFs, actuators, and sensors.
        joint_filter: Timescale of filter for joint actuators. 0: disabled.
        adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
        time_limit: Episode time limit.
        mjcb_control: Optional MuJoCo control callback, a callable with
            arguments (model, data). For more information, see
            https://mujoco.readthedocs.io/en/stable/APIreference/APIglobals.html#mjcb-control
        observables_options (optional): A dict of dicts of configuration options
            keyed on observable names, or a dict of configuration options, which
            will propagate those options to all observables.
        action_corruptor (optional): A callable which takes an action as an
            argument, modifies it, and returns it. An example use case for
            this is to add random noise to the action.

    Returns:
        Template walking environment.
    """
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = floors.Floor()
    # Build a no-op task.
    task = TemplateTask(walker=walker,
                        arena=arena,
                        force_actuators=force_actuators,
                        disable_wings=disable_wings,
                        joint_filter=joint_filter,
                        adhesion_filter=adhesion_filter,
                        observables_options=observables_options,
                        mjcb_control=mjcb_control,
                        action_corruptor=action_corruptor,
                        time_limit=time_limit)
    # Reset control callback, if any.
    mujoco.set_mjcb_control(None)
    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)
