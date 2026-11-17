from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


LEG_MAPPING = {
    "T1_left": "LF",
    "T1_right": "RF",
    "T2_left": "LM",
    "T2_right": "RM",
    "T3_left": "LH",
    "T3_right": "RH",
}
LEG_NAMES = {short: full for full, short in LEG_MAPPING.items()}

DOFS_PER_LEG = [
    "coxa_abduct",
    "coxa_twist",
    "coxa",
    "femur_twist",
    "femur",
    "tibia",
    "tarsus",
    "tarsus2",
]

LEG_ORDER = ["LF", "RF", "LM", "RM", "LH", "RH"]
TRIPOD_GROUP_A = ["LF", "RM", "LH"]
TRIPOD_GROUP_B = ["RF", "LM", "RH"]


@dataclass(frozen=True)
class FlyBodyCPGConfig:
    """CPG settings for a periodic FlyBody joint template."""

    template_path: str
    trajectory_id: str = "005"
    start_frame: int = 36
    period_frames: int = 37
    average_cycles: int = 4
    # The source FlyBody recording is sampled every 2 ms; update the
    # controller at that same cadence and let MuJoCo substep independently.
    timestep: float = 2e-3
    frequency: float | None = None
    convergence: float = 20.0
    amplitude_scale: float = 1.0
    initial_phase: float = 0.0
    randomize_phase: bool = False
    straight_turn_trim: float = 0.05
    seed: int = 0


class FlyBodyCPGController:
    """Drive FlyBody position actuators from a periodic FlyBody qpos template.

    A scalar phase oscillator traverses a synchronized six-leg position cycle
    sampled from a FlyBody recording.  The relative timing between legs is
    already present in that six-leg template, so a second hand-authored tripod
    phase offset is intentionally not applied.  Root pose is never read by the
    controller.  The recording has no adhesion labels, so the controller does
    not invent stance timing from root/site tracks and leaves claw adhesion off.
    """

    legs = LEG_ORDER
    flybody_dofs = DOFS_PER_LEG

    def __init__(self, config: FlyBodyCPGConfig | None = None):
        from pathlib import Path

        import h5py

        self.config = config or FlyBodyCPGConfig()
        if self.config.timestep <= 0 or self.config.convergence <= 0:
            raise ValueError("timestep and convergence must be positive")
        if self.config.period_frames < 4 or self.config.average_cycles < 1:
            raise ValueError("period_frames must be >= 4 and average_cycles >= 1")
        if self.config.amplitude_scale <= 0:
            raise ValueError("amplitude_scale must be positive")
        if not np.isfinite(self.config.initial_phase):
            raise ValueError("initial_phase must be finite")
        if not -1.0 <= self.config.straight_turn_trim <= 1.0:
            raise ValueError("straight_turn_trim must be in [-1, 1]")
        template_path = Path(self.config.template_path).expanduser().resolve()
        if not template_path.is_file():
            raise FileNotFoundError(
                f"Missing FlyBody walking template: {template_path}"
            )
        self.template_path = template_path
        self.timestep = float(self.config.timestep)

        leg_joint_names = [
            f"{dof}_{LEG_NAMES[leg]}"
            for leg in self.legs
            for dof in self.flybody_dofs
        ]
        with h5py.File(template_path, "r") as h5:
            if self.config.trajectory_id not in h5["trajectories"]:
                raise KeyError(
                    f"Trajectory {self.config.trajectory_id!r} not found in "
                    f"{template_path}"
                )
            qpos_names = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in h5["id2name"]["qpos"][:]
            ]
            data_timestep = float(h5["timestep_seconds"][()])
            trajectory = h5["trajectories"][self.config.trajectory_id]
            needed = (
                self.config.start_frame
                + self.config.period_frames * self.config.average_cycles
            )
            if needed > len(trajectory["qpos"]):
                raise ValueError(
                    f"Template window ends at frame {needed}, but trajectory "
                    f"{self.config.trajectory_id} has {len(trajectory['qpos'])} frames"
                )
            joint_indices = [qpos_names.index(name) for name in leg_joint_names]
            cycle_qpos = np.stack([
                trajectory["qpos"][
                    self.config.start_frame + cycle * self.config.period_frames:
                    self.config.start_frame + (cycle + 1) * self.config.period_frames
                ][:, joint_indices].reshape(self.config.period_frames, 6, 8)
                for cycle in range(self.config.average_cycles)
            ])

        if data_timestep <= 0:
            raise ValueError("template timestep must be positive")
        self.template_dt = data_timestep
        self.period_seconds = self.config.period_frames * data_timestep
        self.frequency = (
            float(self.config.frequency)
            if self.config.frequency is not None
            else 1.0 / self.period_seconds
        )
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")

        # Average corresponding phases across adjacent recorded cycles to
        # reduce tracking noise while preserving the FlyBody joint frame.
        self.template_cycle = cycle_qpos.mean(axis=0)
        self.neutral_pose = self.template_cycle.mean(axis=0)
        phase_grid = np.arange(self.config.period_frames) * (
            2.0 * np.pi / self.config.period_frames
        )
        self._phase_grid = phase_grid
        self._joint_spline = CubicSpline(
            np.r_[phase_grid, 2.0 * np.pi],
            np.concatenate((self.template_cycle, self.template_cycle[:1]), axis=0),
            axis=0,
            bc_type="periodic",
        )

        self.all_leg_dofs = [
            f"joint_{leg}{dof}" for leg in self.legs for dof in self.flybody_dofs
        ]
        self.flybody_joint_names = [
            f"{dof}_{LEG_NAMES[leg]}"
            for leg in self.legs
            for dof in self.flybody_dofs
        ]
        self.random_state = np.random.RandomState(self.config.seed)
        self.phase = 0.0
        self.magnitude = 0.0
        self.speed = 1.0
        self.turn = 0.0
        self._runtime_amplitude_scale = 1.0
        self._runtime_frequency_scale = 1.0
        self.adhesion = np.zeros(6, dtype=np.float32)
        self.reset()

    def reset(self):
        self.phase = (
            float(self.random_state.uniform(0.0, 2.0 * np.pi))
            if self.config.randomize_phase
            else float(self.config.initial_phase % (2.0 * np.pi))
        )
        self.magnitude = 0.0
        self._runtime_amplitude_scale = 1.0
        self._runtime_frequency_scale = 1.0
        self.adhesion[:] = 0.0

    def set_command(
        self,
        speed: float = 1.0,
        turn: float = 0.0,
        amplitude_scale: float | None = None,
        frequency_scale: float | None = None,
    ):
        """Set locomotion command and optional runtime CPG parameters.

        ``amplitude_scale`` and ``frequency_scale`` are intentionally runtime
        multipliers.  The frozen dataclass configuration remains the baseline
        gait, while a learned or heuristic readout can modulate the CPG
        without rebuilding its template.
        """
        self.speed = float(np.clip(speed, -1.0, 1.0))
        self.turn = float(np.clip(turn, -1.0, 1.0))
        if amplitude_scale is not None:
            self._runtime_amplitude_scale = float(np.clip(amplitude_scale, 0.1, 2.0))
        if frequency_scale is not None:
            self._runtime_frequency_scale = float(np.clip(frequency_scale, 0.1, 3.0))

    def template_pose(self, phase: float, magnitude: float = 1.0):
        cycle_pose = np.asarray(self._joint_spline(phase % (2.0 * np.pi)))
        scale = float(magnitude) * self.config.amplitude_scale
        return self.neutral_pose + scale * (cycle_pose - self.neutral_pose)

    def command_pose(
        self,
        phase: float,
        magnitude: float = 1.0,
        speed: float = 1.0,
        turn: float = 0.0,
    ):
        """Evaluate the exact position target for a phase and locomotion command."""
        if speed == 0.0:
            return self.neutral_pose.copy()
        effective_turn = np.clip(
            float(turn) + self.config.straight_turn_trim, -1.0, 1.0
        )
        left_gain = np.clip(1.0 + effective_turn, 0.0, 2.0)
        right_gain = np.clip(1.0 - effective_turn, 0.0, 2.0)
        gain = np.array([
            left_gain, right_gain, left_gain, right_gain, left_gain, right_gain
        ])
        pose = self.template_pose(phase, magnitude)
        return (
            self.neutral_pose
            + (pose - self.neutral_pose) * abs(speed) * gain[:, None]
        )

    def step(self):
        self.adhesion[:] = 0.0
        if self.speed == 0.0:
            return self.neutral_pose.reshape(-1).copy()

        self.phase = (
            self.phase
            + 2.0 * np.pi * self.frequency * self._runtime_frequency_scale
            * self.speed * self.timestep
        ) % (2.0 * np.pi)
        self.magnitude += (
            self.config.convergence * (1.0 - self.magnitude) * self.timestep
        )
        self.magnitude = float(np.clip(self.magnitude, 0.0, 1.0))

        pose = self.command_pose(
            self.phase,
            magnitude=self.magnitude,
            speed=self.speed,
            turn=self.turn,
        )
        pose = self.neutral_pose + (
            pose - self.neutral_pose
        ) * self._runtime_amplitude_scale
        return pose.reshape(-1).astype(np.float32)
