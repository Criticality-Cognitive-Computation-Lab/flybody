"""Reference trajectory loaders for fruit fly imitation tasks."""

from typing import Literal, Sequence
from abc import ABC, abstractmethod

import h5py
import numpy as np

from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.constants import _FLY_CONTROL_TIMESTEP, _WALK_CONTROL_TIMESTEP
from flybody.quaternions import get_dquat_local


class HDF5TrajectoryLoader(ABC):
    """Base class for loading and serving trajectories from hdf5 datasets."""

    def __init__(self,
                 path: str,
                 traj_indices: Sequence[int] | None = None,
                 random_state: np.random.RandomState | None = None):
        """Initializes the base trajectory loader.

        Args:
            path: Path to hdf5 dataset file with reference rajectories.
            traj_indices: List of trajectory indices to use, e.g. for train/test
                splitting etc. If None, use all available trajectories.
            random_state: Random state for reproducibility.
        """

        if random_state is None:
            self._random_state = np.random.RandomState(None)
        else:
            self._random_state = random_state

        with h5py.File(path, 'r') as f:
            self._n_traj = len(f['trajectories'])
            self._timestep = f['timestep_seconds'][()]

        if traj_indices is None:
            self._traj_indices = np.arange(self._n_traj)
        else:
            self._traj_indices = traj_indices

    @property
    def timestep(self):
        """Dataset timestep duration, in seconds."""
        return self._timestep

    @property
    def num_trajectories(self):
        """Number of trajectories in dataset."""
        return self._n_traj

    @property
    def traj_indices(self):
        """Indices of trajectories to use for training/testing."""
        return self._traj_indices

    @abstractmethod
    def get_trajectory(self,
                       traj_idx: int | None = None,
                       start_step: int | None = None,
                       end_step: int | None = None):
        """Returns a trajectory."""
        raise NotImplementedError("Subclasses should implement this.")


class HDF5FlightTrajectoryLoader(HDF5TrajectoryLoader):
    """Loads and serves trajectories from hdf5 flight imitation dataset."""

    def __init__(
        self,
        path: str,
        traj_indices: Sequence[int] | None = None,
        randomize_start_step: bool = True,
        random_state: np.random.RandomState | None = None,
    ):
        """Initializes the flight trajectory loader.

        Args:
            path: Path to hdf5 dataset file with reference rajectories.
            traj_indices: List of trajectory indices to use, e.g. for train/test
                splitting etc. If None, use all available trajectories.
            randomize_start_step: Whether to select random start point in each
                get_trajectory call.
            random_state: Random state for reproducibility.
        """
        super().__init__(path, traj_indices, random_state=random_state)

        self._randomize_start_step = randomize_start_step

        self._com_qpos = []
        self._com_qvel = []

        with h5py.File(path, 'r') as f:
            n_zeros = len(str(self._n_traj))
            for idx in range(self._n_traj):
                key = str(idx).zfill(n_zeros)
                self._com_qpos.append(f['trajectories'][key]['com_qpos'][()])
                self._com_qvel.append(f['trajectories'][key]['com_qvel'][()])
                assert self._com_qpos[-1].shape[0] == self._com_qvel[-1].shape[0]

    def trajectory_len(self, traj_idx: int) -> int:
        """Returns length of trajectory with index traj_idx."""
        return len(self._com_qpos[traj_idx])

    def get_trajectory(
            self,
            traj_idx: int | None = None,
            start_step: int | None = None,
            end_step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns a flight trajectory from the dataset.

        Args:
            traj_idx: Index of the desired trajectory. If None, a random
                trajectory out of traj_indices is selected.
            start_step: Start index for the trajectory slice. If None, defaults
                to the beginning. If attribute randomize_start_step is True,
                start_step is ignored.
            end_step: End index for the trajectory slice. If None, defaults to
                the end. If attribute randomize_start_step is True,
                end_step is ignored.

        Returns:
            tuple: Two numpy arrays for com_qpos and com_qvel respectively.
        """
        if traj_idx is None:
            traj_idx = self._random_state.choice(self._traj_indices)

        traj_len = len(self._com_qpos[traj_idx])
        if self._randomize_start_step:
            start_step = self._random_state.randint(traj_len - 50)
            end_step = traj_len
        else:
            start_step = 0 if start_step is None else start_step
            end_step = traj_len if end_step is None else end_step

        com_qpos = self._com_qpos[traj_idx][start_step:end_step].copy()
        com_qvel = self._com_qvel[traj_idx][start_step:end_step]
        com_qpos[:, :2] -= com_qpos[0, :2]

        return com_qpos, com_qvel


class InferenceFlightTrajectoryLoader():
    """Simple drop-in inference-time replacement for flight trajectory loader.

    This trajectory loader can be used for bypassing loading actual flight
    datasets and loading custom trajectories instead, e.g. at inference time.

    A simple synthetic flight trajectory is automatically set upon this class
    initialization.

    To use this class with other custom trajectories, create qpos and qvel for
    your custom trajectory and then set this trajectory for loading in the
    flight task by calling:
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    """

    def __init__(self):
        # Initially, set a simple synthetic trajectory, e.g. for quick testing.
        qpos, qvel = constant_speed_trajectory(
            n_steps=200, speed=20, init_pos=(0, 0, 1),
            body_rot_angle_y=-47.5, control_timestep=_FLY_CONTROL_TIMESTEP)
        self.set_next_trajectory(qpos, qvel)

    def set_next_trajectory(self, com_qpos: np.ndarray, com_qvel: np.ndarray):
        """Set new trajectory to be returned by get_trajectory.

        Args:
            com_qpos: Center-of-mass trajectory, (time, 7).
            com_qvel: Velocity of CoM trajectory, (time, 6).
        """
        self._com_qpos = com_qpos.copy()
        self._com_qpos[:, :2] -= self._com_qpos[0, :2]
        self._com_qvel = com_qvel

    def get_trajectory(self, traj_idx: int):
        del traj_idx  # Unused.
        if not hasattr(self, '_com_qpos'):
            raise AttributeError(
                'Trajectory not set yet. Call set_next_trajectory first.')
        return self._com_qpos, self._com_qvel


class HDF5WalkingTrajectoryLoader(HDF5TrajectoryLoader):
    """Loads and serves trajectories from hdf5 walking imitation dataset."""

    def __init__(
        self,
        path: str,
        traj_indices: Sequence[int] | None = None,
        randomize_start_step: bool = False,
        clip_length_steps: int | None = None,
        min_active_window_path_cm: float = 0.0,
        active_window_steps: int | None = None,
        active_window_max_tries: int = 64,
        random_state: np.random.RandomState | None = None,
    ):
        """Initializes the walking trajectory loader.

        Args:
            path: Path to hdf5 dataset file with reference rajectories.
            traj_indices: List of trajectory indices to use, e.g. for train/test
                splitting etc. If None, use all available trajectories.
            randomize_start_step: If True, each ``get_trajectory`` call samples a
                random start point from the selected trajectory. This is useful
                for long MABe22 root trajectories where an episode should see a
                random temporal window instead of always starting at frame 0.
            clip_length_steps: Optional number of timesteps to return from the
                sampled start point. If None, return until the end of the
                trajectory. When ``randomize_start_step`` is True this value is
                also used to keep enough look-ahead samples for the task.
            min_active_window_path_cm: If > 0 and ``randomize_start_step`` is
                True, rejection-sample trajectory windows until the xy path
                length over ``active_window_steps`` reaches this threshold.
                This is useful for MABe22 root trajectories, where many random
                10s windows are nearly stationary.
            active_window_steps: Number of timesteps used to measure activity.
                If None, use ``clip_length_steps``.  For imitation tasks this
                should usually be the rollout horizon, excluding future
                reference look-ahead.
            active_window_max_tries: Maximum random trajectory/window attempts
                before falling back to uniform sampling.
            random_state: Random state for reproducibility.
        """

        super().__init__(path, traj_indices, random_state=random_state)

        self._randomize_start_step = randomize_start_step
        self._clip_length_steps = clip_length_steps
        self._min_active_window_path_cm = float(min_active_window_path_cm)
        self._active_window_steps = active_window_steps
        self._active_window_max_tries = int(active_window_max_tries)
        self._last_sample_info: dict[str, int | float | bool] = {}
        self._h5 = h5py.File(path, 'r')
        self._traj_lens = self._h5['trajectory_lengths']
        self._n_zeros = int(
            self._h5['trajectories'].attrs.get('key_width', len(str(self._n_traj)))
        )

    def trajectory_len(self, traj_idx: int) -> int:
        """Returns length of trajectory with index traj_idx."""
        return self._traj_lens[traj_idx]

    @property
    def last_sample_info(self) -> dict[str, int | float | bool]:
        """Metadata for the most recent trajectory/window sample."""
        return dict(self._last_sample_info)

    def _trajectory_key(self, traj_idx: int) -> str:
        return str(traj_idx).zfill(self._n_zeros)

    @staticmethod
    def _optional_dataset_slice(
        snippet,
        name: str,
        start_step: int,
        end_step: int,
        trailing_shape: tuple[int, ...],
        dtype: np.dtype | type = np.float32,
    ) -> np.ndarray:
        """Read an optional per-step dataset or return a zero-width fallback.

        The original FlyBody walking dataset stores full-body reference arrays
        (``qpos``, ``qvel``, ``root2site`` and ``joint_quat``).  Root-only
        social-behavior datasets only need ``root_qpos`` and ``root_qvel`` for
        ``WalkRootImitation``.  Returning empty arrays here keeps the legacy
        walking format intact while allowing much smaller root-only HDF5 files.
        """

        if name in snippet:
            return snippet[name][start_step:end_step]
        n_steps = int(end_step) - int(start_step)
        return np.zeros((n_steps, *trailing_shape), dtype=dtype)

    def _window_path_cm(
        self,
        traj_idx: int,
        start_step: int,
        window_steps: int,
    ) -> float:
        """Return xy path length of a root trajectory window in centimeters."""
        if window_steps <= 1:
            return 0.0
        traj_len = int(self._traj_lens[traj_idx])
        end_step = min(traj_len, start_step + window_steps)
        if end_step - start_step <= 1:
            return 0.0
        key = self._trajectory_key(traj_idx)
        xy = self._h5['trajectories'][key]['root_qpos'][start_step:end_step, :2]
        return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())

    def _sample_uniform_window(
        self,
        traj_idx: int,
        requested_len: int | None,
    ) -> tuple[int, int]:
        traj_len = int(self._traj_lens[traj_idx])
        if requested_len is None or requested_len >= traj_len:
            return 0, traj_len
        max_start = traj_len - requested_len
        start_step = int(self._random_state.randint(max_start + 1))
        return start_step, start_step + requested_len

    def _sample_active_window(
        self,
        traj_idx: int | None,
        requested_len: int | None,
    ) -> tuple[int, int, int, float, int] | None:
        """Try to sample an active random window.

        Returns:
            ``(traj_idx, start_step, end_step, path_cm, n_tries)`` or ``None``.
        """
        if (
            not self._randomize_start_step
            or self._min_active_window_path_cm <= 0.0
            or requested_len is None
        ):
            return None

        n_tries = max(1, self._active_window_max_tries)
        selected_indices = [traj_idx] if traj_idx is not None else None
        for attempt in range(1, n_tries + 1):
            if selected_indices is None:
                candidate_idx = int(self._random_state.choice(self._traj_indices))
            else:
                candidate_idx = int(selected_indices[0])
            traj_len = int(self._traj_lens[candidate_idx])
            if requested_len >= traj_len:
                start_step = 0
                end_step = traj_len
            else:
                max_start = traj_len - requested_len
                start_step = int(self._random_state.randint(max_start + 1))
                end_step = start_step + requested_len
            activity_steps = self._active_window_steps or requested_len
            activity_steps = min(activity_steps, end_step - start_step)
            path_cm = self._window_path_cm(
                candidate_idx,
                start_step,
                activity_steps,
            )
            if path_cm >= self._min_active_window_path_cm:
                return candidate_idx, start_step, end_step, path_cm, attempt
        return None

    def get_trajectory(
            self,
            traj_idx: int | None = None,
            start_step: int | None = None,
            end_step: int | None = None) -> dict[str, np.ndarray]:
        """Returns a walking trajectory from the dataset.

        Args:
            traj_idx: Index of the desired trajectory. If None, a random
                trajectory is selected.
            start_step: Start index for the trajectory slice. If None, defaults
                to the beginning.
            end_step: End index for the trajectory slice. If None, defaults to
                the end.

        Returns:
            dict: Dictionary containing qpos, qvel, root2site, and joint_quat
                of the trajectory.
        """
        active_sample = None
        if self._randomize_start_step:
            requested_len = self._clip_length_steps
            if requested_len is None and start_step is not None and end_step is not None:
                requested_len = end_step - start_step
            active_sample = self._sample_active_window(traj_idx, requested_len)
            if active_sample is not None:
                traj_idx, start_step, end_step, activity_path_cm, active_tries = active_sample
            else:
                if traj_idx is None:
                    traj_idx = int(self._random_state.choice(self._traj_indices))
                start_step, end_step = self._sample_uniform_window(traj_idx, requested_len)
                activity_steps = self._active_window_steps or (end_step - start_step)
                activity_steps = min(activity_steps, end_step - start_step)
                activity_path_cm = self._window_path_cm(
                    int(traj_idx),
                    int(start_step),
                    int(activity_steps),
                )
                active_tries = self._active_window_max_tries if (
                    self._min_active_window_path_cm > 0.0
                    and requested_len is not None
                ) else 0
        else:
            if traj_idx is None:
                traj_idx = int(self._random_state.choice(self._traj_indices))
            traj_len = int(self._traj_lens[traj_idx])
            start_step = 0 if start_step is None else start_step
            if end_step is None:
                if self._clip_length_steps is None:
                    end_step = traj_len
                else:
                    end_step = min(traj_len, start_step + self._clip_length_steps)
            else:
                end_step = min(traj_len, end_step)
            activity_steps = self._active_window_steps or (end_step - start_step)
            activity_steps = min(activity_steps, end_step - start_step)
            activity_path_cm = self._window_path_cm(
                int(traj_idx),
                int(start_step),
                int(activity_steps),
            )
            active_tries = 0

        traj_idx = int(traj_idx)
        start_step = int(start_step)
        end_step = int(end_step)
        key = self._trajectory_key(traj_idx)
        snippet = self._h5['trajectories'][key]
        qpos = np.concatenate((
            snippet['root_qpos'][start_step:end_step],
            self._optional_dataset_slice(
                snippet,
                'qpos',
                start_step,
                end_step,
                trailing_shape=(0,),
            ),
        ), axis=1)
        qvel = np.concatenate((
            snippet['root_qvel'][start_step:end_step],
            self._optional_dataset_slice(
                snippet,
                'qvel',
                start_step,
                end_step,
                trailing_shape=(0,),
            ),
        ), axis=1)
        qpos[:, :2] -= qpos[0, :2]

        trajectory = {
            'qpos': qpos,
            'qvel': qvel,
            'root2site': self._optional_dataset_slice(
                snippet,
                'root2site',
                start_step,
                end_step,
                trailing_shape=(0, 3),
            ),
            'joint_quat': self._optional_dataset_slice(
                snippet,
                'joint_quat',
                start_step,
                end_step,
                trailing_shape=(0, 4),
            ),
        }
        for optional_name in (
            'leg_tips_ego_xy_cm',
            'leg_tip_valid_mask',
            'source_xy_mm',
            'source_orientation_cos_sin',
            'frame_numbers',
            'annotations',
        ):
            if optional_name in snippet:
                trajectory[optional_name] = snippet[optional_name][start_step:end_step]
        self._last_sample_info = {
            'traj_idx': traj_idx,
            'start_step': start_step,
            'end_step': end_step,
            'n_steps': end_step - start_step,
            'active_window_path_cm': float(activity_path_cm),
            'min_active_window_path_cm': self._min_active_window_path_cm,
            'active_window_accepted': bool(
                self._min_active_window_path_cm <= 0.0
                or activity_path_cm >= self._min_active_window_path_cm
            ),
            'active_window_tries': int(active_tries),
        }

        return trajectory

    def get_site_names(self):
        """Returns snippet site names."""
        if 'id2name' not in self._h5 or 'sites' not in self._h5['id2name']:
            return []
        return [
            s.decode('utf-8') if isinstance(s, bytes) else str(s)
            for s in self._h5['id2name']['sites']
        ]

    def get_joint_names(self):
        """Returns snippet joint names."""
        if 'id2name' not in self._h5 or 'joints' not in self._h5['id2name']:
            return []
        return [
            s.decode('utf-8') if isinstance(s, bytes) else str(s)
            for s in self._h5['id2name']['joints']
        ]


class HDF5SynchronizedMultiFlyTrajectoryLoader:
    """Loads synchronized multi-fly root trajectories from MABe22 segments.

    The aggression context files store one segment as ``root_qpos`` with shape
    ``(T, num_flies, 7)`` and ``root_qvel`` with shape ``(T, num_flies, 6)``.
    This loader samples one segment/window per episode and returns all selected
    flies from that same temporal window.  A common XY translation is applied to
    every selected fly so their within-segment relative positions are preserved.
    """

    def __init__(
        self,
        path: str,
        traj_indices: Sequence[int] | None = None,
        randomize_start_step: bool = False,
        clip_length_steps: int | None = None,
        min_active_window_path_cm: float = 0.0,
        active_window_steps: int | None = None,
        active_window_max_tries: int = 64,
        random_state: np.random.RandomState | None = None,
        fly_slots: Sequence[int] | None = None,
        center_mode: Literal["first_frame_centroid", "first_fly", "none"] = (
            "first_frame_centroid"
        ),
        active_metric: Literal["min", "mean", "max"] = "mean",
    ):
        if random_state is None:
            self._random_state = np.random.RandomState(None)
        else:
            self._random_state = random_state

        self._h5 = h5py.File(path, "r")
        if "segments" not in self._h5:
            raise KeyError(
                f"{path!r} does not contain a top-level 'segments' group."
            )
        self._segments = self._h5["segments"]
        self._segment_keys = sorted(self._segments.keys())
        self._n_traj = len(self._segment_keys)
        if self._n_traj == 0:
            raise ValueError(f"No segments found in {path!r}.")

        self._timestep = float(
            self._h5.attrs.get(
                "timestep_seconds",
                self._h5.attrs.get("raw_frame_interval_seconds", 0.002),
            )
        )
        if traj_indices is None:
            self._traj_indices = np.arange(self._n_traj)
        else:
            self._traj_indices = np.asarray(traj_indices, dtype=np.int64)

        self._randomize_start_step = bool(randomize_start_step)
        self._clip_length_steps = clip_length_steps
        self._min_active_window_path_cm = float(min_active_window_path_cm)
        self._active_window_steps = active_window_steps
        self._active_window_max_tries = int(active_window_max_tries)
        self._fly_slots = None if fly_slots is None else tuple(int(x) for x in fly_slots)
        self._center_mode = center_mode
        self._active_metric = active_metric
        self._last_sample_info: dict[str, object] = {}

        if self._center_mode not in {"first_frame_centroid", "first_fly", "none"}:
            raise ValueError(f"Unsupported center_mode: {self._center_mode!r}")
        if self._active_metric not in {"min", "mean", "max"}:
            raise ValueError(f"Unsupported active_metric: {self._active_metric!r}")

    @property
    def timestep(self):
        """Dataset timestep duration, in seconds."""
        return self._timestep

    @property
    def num_trajectories(self):
        """Number of synchronized segments."""
        return self._n_traj

    @property
    def traj_indices(self):
        """Indices of synchronized segments available for sampling."""
        return self._traj_indices

    @property
    def last_sample_info(self) -> dict[str, object]:
        """Metadata for the most recent synchronized segment/window sample."""
        return dict(self._last_sample_info)

    def _segment_key(self, traj_idx: int) -> str:
        return self._segment_keys[int(traj_idx)]

    def _segment_len(self, traj_idx: int) -> int:
        key = self._segment_key(traj_idx)
        return int(self._segments[key]["root_qpos"].shape[0])

    def _resolve_fly_slots(self, num_flies: int, segment_key: str) -> tuple[int, ...]:
        available = int(self._segments[segment_key]["root_qpos"].shape[1])
        if self._fly_slots is None:
            slots = tuple(range(int(num_flies)))
        else:
            slots = self._fly_slots[: int(num_flies)]
        if len(slots) != int(num_flies):
            raise ValueError(
                f"Requested {num_flies} flies, but fly_slots={self._fly_slots}"
            )
        if any(slot < 0 or slot >= available for slot in slots):
            raise ValueError(
                f"fly_slots={slots} out of range for segment {segment_key!r} "
                f"with {available} flies."
            )
        return slots

    def _sample_uniform_window(
        self,
        traj_idx: int,
        requested_len: int | None,
    ) -> tuple[int, int]:
        traj_len = self._segment_len(traj_idx)
        if requested_len is None or requested_len >= traj_len:
            return 0, traj_len
        max_start = traj_len - requested_len
        start_step = int(self._random_state.randint(max_start + 1))
        return start_step, start_step + requested_len

    @staticmethod
    def _path_cm(xy: np.ndarray) -> float:
        if xy.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())

    def _window_paths_cm(
        self,
        traj_idx: int,
        slots: Sequence[int],
        start_step: int,
        window_steps: int,
    ) -> list[float]:
        if window_steps <= 1:
            return [0.0 for _ in slots]
        traj_len = self._segment_len(traj_idx)
        end_step = min(traj_len, start_step + window_steps)
        if end_step - start_step <= 1:
            return [0.0 for _ in slots]
        key = self._segment_key(traj_idx)
        root_qpos = self._segments[key]["root_qpos"][
            start_step:end_step,
            list(slots),
            :2,
        ]
        return [self._path_cm(root_qpos[:, i, :]) for i in range(len(slots))]

    def _reduce_paths(self, paths_cm: Sequence[float]) -> float:
        paths = np.asarray(paths_cm, dtype=np.float64)
        if paths.size == 0:
            return 0.0
        if self._active_metric == "min":
            return float(np.min(paths))
        if self._active_metric == "max":
            return float(np.max(paths))
        return float(np.mean(paths))

    def _sample_active_window(
        self,
        traj_idx: int | None,
        requested_len: int | None,
        num_flies: int,
    ) -> tuple[int, int, int, list[float], float, int] | None:
        if (
            not self._randomize_start_step
            or self._min_active_window_path_cm <= 0.0
            or requested_len is None
        ):
            return None

        n_tries = max(1, self._active_window_max_tries)
        selected_indices = [traj_idx] if traj_idx is not None else None
        for attempt in range(1, n_tries + 1):
            if selected_indices is None:
                candidate_idx = int(self._random_state.choice(self._traj_indices))
            else:
                candidate_idx = int(selected_indices[0])
            start_step, end_step = self._sample_uniform_window(
                candidate_idx,
                requested_len,
            )
            key = self._segment_key(candidate_idx)
            slots = self._resolve_fly_slots(num_flies, key)
            activity_steps = self._active_window_steps or (end_step - start_step)
            activity_steps = min(activity_steps, end_step - start_step)
            paths_cm = self._window_paths_cm(
                candidate_idx,
                slots,
                start_step,
                activity_steps,
            )
            activity_value = self._reduce_paths(paths_cm)
            if activity_value >= self._min_active_window_path_cm:
                return (
                    candidate_idx,
                    start_step,
                    end_step,
                    paths_cm,
                    activity_value,
                    attempt,
                )
        return None

    def get_multi_trajectory(
        self,
        num_flies: int,
        traj_idx: int | None = None,
        start_step: int | None = None,
        end_step: int | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Return synchronized snippets for ``num_flies`` from one segment."""
        requested_len = self._clip_length_steps
        if requested_len is None and start_step is not None and end_step is not None:
            requested_len = end_step - start_step

        if self._randomize_start_step:
            active_sample = self._sample_active_window(
                traj_idx,
                requested_len,
                num_flies,
            )
            if active_sample is not None:
                (
                    traj_idx,
                    start_step,
                    end_step,
                    activity_paths_cm,
                    activity_value_cm,
                    active_tries,
                ) = active_sample
            else:
                if traj_idx is None:
                    traj_idx = int(self._random_state.choice(self._traj_indices))
                start_step, end_step = self._sample_uniform_window(
                    traj_idx,
                    requested_len,
                )
                key = self._segment_key(traj_idx)
                slots = self._resolve_fly_slots(num_flies, key)
                activity_steps = self._active_window_steps or (end_step - start_step)
                activity_steps = min(activity_steps, end_step - start_step)
                activity_paths_cm = self._window_paths_cm(
                    traj_idx,
                    slots,
                    start_step,
                    activity_steps,
                )
                activity_value_cm = self._reduce_paths(activity_paths_cm)
                active_tries = self._active_window_max_tries if (
                    self._min_active_window_path_cm > 0.0
                    and requested_len is not None
                ) else 0
        else:
            if traj_idx is None:
                traj_idx = int(self._random_state.choice(self._traj_indices))
            traj_len = self._segment_len(traj_idx)
            start_step = 0 if start_step is None else start_step
            if end_step is None:
                if requested_len is None:
                    end_step = traj_len
                else:
                    end_step = min(traj_len, start_step + requested_len)
            else:
                end_step = min(traj_len, end_step)
            key = self._segment_key(traj_idx)
            slots = self._resolve_fly_slots(num_flies, key)
            activity_steps = self._active_window_steps or (end_step - start_step)
            activity_steps = min(activity_steps, end_step - start_step)
            activity_paths_cm = self._window_paths_cm(
                traj_idx,
                slots,
                start_step,
                activity_steps,
            )
            activity_value_cm = self._reduce_paths(activity_paths_cm)
            active_tries = 0

        traj_idx = int(traj_idx)
        start_step = int(start_step)
        end_step = int(end_step)
        key = self._segment_key(traj_idx)
        slots = self._resolve_fly_slots(num_flies, key)
        segment = self._segments[key]
        root_qpos = np.asarray(
            segment["root_qpos"][start_step:end_step, list(slots), :7],
            dtype=np.float32,
        ).copy()
        root_qvel = np.asarray(
            segment["root_qvel"][start_step:end_step, list(slots), :6],
            dtype=np.float32,
        ).copy()

        if self._center_mode == "first_frame_centroid":
            origin_xy = root_qpos[0, :, :2].mean(axis=0)
        elif self._center_mode == "first_fly":
            origin_xy = root_qpos[0, 0, :2].copy()
        else:
            origin_xy = np.zeros(2, dtype=np.float32)
        root_qpos[:, :, :2] -= origin_xy

        snippets: list[dict[str, np.ndarray]] = []
        n_steps = int(end_step - start_step)
        for fly_i in range(num_flies):
            snippets.append(
                {
                    "qpos": root_qpos[:, fly_i, :],
                    "qvel": root_qvel[:, fly_i, :],
                    "root2site": np.zeros((n_steps, 0, 3), dtype=np.float32),
                    "joint_quat": np.zeros((n_steps, 0, 4), dtype=np.float32),
                }
            )

        valid_fraction = None
        if "valid_mask" in segment:
            valid = np.asarray(
                segment["valid_mask"][start_step:end_step, list(slots)],
                dtype=bool,
            )
            valid_fraction = float(valid.mean()) if valid.size else 1.0

        self._last_sample_info = {
            "traj_idx": traj_idx,
            "segment_key": key,
            "sequence_id": segment.attrs.get("sequence_id", ""),
            "fly_slots": list(slots),
            "fly_ids": (
                segment["fly_ids"][list(slots)].astype(int).tolist()
                if "fly_ids" in segment
                else list(slots)
            ),
            "start_step": start_step,
            "end_step": end_step,
            "n_steps": n_steps,
            "window_start_frame": int(
                segment.attrs.get("window_start_frame", 0) + start_step
            ),
            "window_end_frame_exclusive": int(
                segment.attrs.get("window_start_frame", 0) + end_step
            ),
            "center_mode": self._center_mode,
            "origin_xy_cm": np.asarray(origin_xy, dtype=float).tolist(),
            "activity_paths_cm": [float(x) for x in activity_paths_cm],
            "activity_metric": self._active_metric,
            "activity_metric_path_cm": float(activity_value_cm),
            "min_active_window_path_cm": self._min_active_window_path_cm,
            "active_window_accepted": bool(
                self._min_active_window_path_cm <= 0.0
                or activity_value_cm >= self._min_active_window_path_cm
            ),
            "active_window_tries": int(active_tries),
            "valid_fraction": valid_fraction,
        }

        return snippets


class InferenceWalkingTrajectoryLoader():
    """Simple drop-in inference-time replacement for walking trajectory loader.

    This trajectory loader can be used for bypassing loading actual walking
    datasets and loading custom trajectories instead, e.g. at inference time.

    A simple synthetic walking trajectory is automatically set upon this class
    initialization.

    To use this class with other custom trajectories, create qpos and qvel for
    your custom trajectory and then set this trajectory for loading in the
    walking task by calling:
        env.task._traj_generator.set_next_trajectory(qpos, qvel)
    """

    def __init__(self):
        # Initially, set a simple synthetic trajectory, e.g. for quick testing.
        qpos, qvel = constant_speed_trajectory(
            n_steps=300, speed=2, init_pos=(0, 0, 0.1278),
            control_timestep=_WALK_CONTROL_TIMESTEP)
        self.set_next_trajectory(qpos, qvel)

    def set_next_trajectory(self, qpos: np.ndarray, qvel: np.ndarray):
        """Set new trajectory to be returned by get_trajectory.

        Args:
            qpos: Center-of-mass trajectory, (time, 7).
            qvel: Velocity of CoM trajectory, (time, 6).
        """
        self._snippet = {'qpos': qpos, 'qvel': qvel}

    def get_trajectory(self, traj_idx: int):
        del traj_idx  # Unused.
        if not hasattr(self, '_snippet'):
            raise AttributeError(
                'Trajectory not set yet. Call set_next_trajectory first.')
        return self._snippet

    def get_joint_names(self):
        return []

    def get_site_names(self):
        return []

from flybody.tasks.task_utils import (
    com2root, root2com, add_trajectory_sites,update_trajectory_sites
)
class TrajectoryGenerator():

    def __init__(self, path, num_envs):
        self._loader = HDF5FlightTrajectoryLoader(path)
        self._num_envs = num_envs
        self.expert_traj = [None] * num_envs
        self.expert_traj_len = [0] * num_envs
        self.expert_traj_reach_end = [False] * num_envs
        self.pos_offsets = [np.zeros(3)] * num_envs
        self.step_counter = [0] * num_envs


    def reset(self, envs, list_envs_idx: list[int] = None):
        if list_envs_idx is None:
            list_envs_idx = list(range(self._num_envs))
        # list[env_nums] of fly poses, each is (pos, quat)
        list_init_pose = envs.get_fly_poses()

        for env_idx in list_envs_idx:
            # sample a expert trajectory
            # _ref_qpos is (time, 7), with root com qpos and quat.
            # _ref_qvel is (time, 6).
            _ref_qpos, _ = self._loader.get_trajectory(
                traj_idx=None
            )
            # convert com qpos to root qpos
            ghost_root_pos = com2root(_ref_qpos[:, :3], _ref_qpos[:, 3:])
            ghost_root_quat = _ref_qpos[:, 3:7]
            _ref_qpos = np.concatenate([ghost_root_pos, ghost_root_quat], axis=1)
            #
            fly_start_pos = list_init_pose[env_idx][0]
            expert_start_pos = _ref_qpos[0, :3]
            self.pos_offsets[env_idx] = fly_start_pos - expert_start_pos
            #
            self.expert_traj[env_idx] = _ref_qpos
            self.expert_traj_len[env_idx] = min(_ref_qpos.shape[0], round(0.4 / 2e-4)) - 7
            self.expert_traj_reach_end[env_idx] = False
            #
            self.step_counter[env_idx] = 0


    def get_expert_goal(self, envs):
        #
        current_poses = envs.get_fly_poses()

        world_rel_pos_list = []
        ref_quat_list = []

        for i in range(self._num_envs):

            if self.step_counter[i] == self.expert_traj_len[i]:
                self.expert_traj_reach_end[i] = True
                step = self.expert_traj_len[i]
            else:
                step = self.step_counter[i]

            traj = self.expert_traj[i]

            # get expert displacement
            offset = self.pos_offsets[i]
            ref_slice = traj[step : step + 6, :3]
            rel_pos_world = (ref_slice + offset) - current_poses[i][0]
            world_rel_pos_list.append(rel_pos_world)

            # get expert orientation
            ref_quat_slice = traj[step : step + 6, 3:7]
            ref_quat_list.append(get_dquat_local(current_poses[i][1], ref_quat_slice))

            # update step counter
            self.step_counter[i] += 1

        ref_displacement_list = envs.transform_vec_to_egocentric_frame(world_rel_pos_list)
        ref_displacement_np = np.array(ref_displacement_list)
        ref_quat_np = np.array(ref_quat_list)


        return ref_displacement_np, ref_quat_np
