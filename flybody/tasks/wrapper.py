import dm_env
import numpy as np
import tree
from typing import Callable, Sequence

from acme import specs
from acme import types

from flygym.vision.retina import blow_hw
from flygym.vision import retina


def wrap_all(
    environment: dm_env.Environment,
    wrappers: Sequence[Callable[[dm_env.Environment], dm_env.Environment]],
) -> dm_env.Environment:
  """Given an environment, wrap it in a list of wrappers."""
  for w in wrappers:
    environment = w(environment)

  return environment


class EnvironmentWrapper(dm_env.Environment):
  """Environment that wraps another environment.

  This exposes the wrapped environment with the `.environment` property and also
  defines `__getattr__` so that attributes are invisibly forwarded to the
  wrapped environment (and hence enabling duck-typing).
  """

  _environment: dm_env.Environment

  def __init__(self, environment: dm_env.Environment):
    self._environment = environment

  def __getattr__(self, attr: str):
    # Delegates attribute calls to the wrapped environment.
    return getattr(self._environment, attr)

  # Getting/setting of state is necessary so that getattr doesn't delegate them
  # to the wrapped environment. This makes sure pickling a wrapped environment
  # works as expected.

  def __getstate__(self):
    return self.__dict__

  def __setstate__(self, state):
    self.__dict__.update(state)

  @property
  def environment(self) -> dm_env.Environment:
    return self._environment

  # The following lines are necessary because methods defined in
  # `dm_env.Environment` are not delegated through `__getattr__`, which would
  # only be used to expose methods or properties that are not defined in the
  # base `dm_env.Environment` class.

  def step(self, action) -> dm_env.TimeStep:
    return self._environment.step(action)

  def reset(self) -> dm_env.TimeStep:
    return self._environment.reset()

  def action_spec(self):
    return self._environment.action_spec()

  def discount_spec(self):
    return self._environment.discount_spec()

  def observation_spec(self):
    return self._environment.observation_spec()

  def reward_spec(self):
    return self._environment.reward_spec()

  def close(self):
    return self._environment.close()

#
class SinglePrecisionWrapper(EnvironmentWrapper):
  """Wrapper which converts environments from double- to single-precision."""

  def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    return timestep._replace(
        reward=_convert_value(timestep.reward),
        discount=_convert_value(timestep.discount),
        observation=_convert_value(timestep.observation))

  def step(self, action) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.step(action))

  def reset(self) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.reset())

  def action_spec(self):
    return _convert_spec(self._environment.action_spec())

  def discount_spec(self):
    return _convert_spec(self._environment.discount_spec())

  def observation_spec(self):
    return _convert_spec(self._environment.observation_spec())

  def reward_spec(self):
    return _convert_spec(self._environment.reward_spec())


def _convert_spec(nested_spec: types.NestedSpec) -> types.NestedSpec:
  """Convert a nested spec."""

  def _convert_single_spec(spec: specs.Array):
    """Convert a single spec."""
    if spec.dtype == 'O':
      # Pass StringArray objects through unmodified.
      return spec
    if np.issubdtype(spec.dtype, np.float64):
      dtype = np.float32
    elif np.issubdtype(spec.dtype, np.int64):
      dtype = np.int32
    else:
      dtype = spec.dtype
    return spec.replace(dtype=dtype)

  return tree.map_structure(_convert_single_spec, nested_spec)


def _convert_value(nested_value: types.Nest) -> types.Nest:
  """Convert a nested value given a desired nested spec."""

  def _convert_single_value(value):
    if value is not None:
      value = np.asarray(value)
      if np.issubdtype(value.dtype, np.float64):
        value = np.asarray(value, dtype=np.float32)
      elif np.issubdtype(value.dtype, np.int64):
        value = np.asarray(value, dtype=np.int32)
    return value

  return tree.map_structure(_convert_single_value, nested_value)


#
class CanonicalSpecWrapper(EnvironmentWrapper):
  """Wrapper which converts environments to use canonical action specs.

  This only affects action specs of type `specs.BoundedArray`.

  For bounded action specs, we refer to a canonical action spec as the bounding
  box [-1, 1]^d where d is the dimensionality of the spec. So the shape and
  dtype of the spec is unchanged, while the maximum/minimum values are set
  to +/- 1.
  """

  def __init__(self, environment: dm_env.Environment, clip: bool = False):
    super().__init__(environment)
    self._action_spec = environment.action_spec()
    self._clip = clip

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    scaled_action = _scale_nested_action(action, self._action_spec, self._clip)
    return self._environment.step(scaled_action)

  def action_spec(self):
    return _scale_spec(self._environment.action_spec())


def _scale_spec(nested_spec: types.NestedSpec) -> types.NestedSpec:
  """Converts all bounded specs in nested spec to the canonical scale."""

  def _scale_single_spec(spec: specs.Array) -> specs.Array:
    """Converts a single spec to canonical if bounded."""
    if isinstance(spec, specs.BoundedArray):
      return spec.replace(
          minimum=-np.ones(spec.shape), maximum=np.ones(spec.shape))
    else:
      return spec

  return tree.map_structure(_scale_single_spec, nested_spec)


def _scale_nested_action(
    nested_action: types.NestedArray,
    nested_spec: types.NestedSpec,
    clip: bool,
) -> types.NestedArray:
  """Converts a canonical nested action back to the given nested action spec."""

  def _scale_action(action: np.ndarray, spec: specs.Array):
    """Converts a single canonical action back to the given action spec."""
    if isinstance(spec, specs.BoundedArray):
      # Get scale and offset of output action spec.
      scale = spec.maximum - spec.minimum
      offset = spec.minimum

      # Maybe clip the action.
      if clip:
        action = np.clip(action, -1.0, 1.0)

      # Map action to [0, 1].
      action = 0.5 * (action + 1.0)

      # Map action to [spec.minimum, spec.maximum].
      action *= scale
      action += offset

    return action

  return tree.map_structure(_scale_action, nested_action, nested_spec)


# import ray
# import numpy as np
# from pathlib import Path
# from dm_control import composer
# from dm_env import TimeStep, StepType

# from ..fly_envs import (
#   flight_imitation, walk_imitation, vision_guided_avoidance, vision_guided_flight
# )


# @ray.remote(
#     num_cpus=1,
#     num_gpus=0,
#     runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "osmesa"}}
# )
# class RayEnvActor():
#   def __init__(self, task):
#     if task == 'flight_imitation':
#         self.env_factory = flight_imitation_environment_factory()
#     elif task == 'walking_imitation':
#         self.env_factory = walking_imitation_environment_factory()
#     elif task == 'vision_guided_flight':
#         self.env_factory = vision_guided_flight_environment_factory()
#     elif task == 'vision_guided_avoidance':
#         self.env_factory = vision_guided_avoidance_environment_factory()
#     self.retina = retina.Retina()

#   def reset(self, ):
#     ts = self.env_factory.reset()
#     left_eye = ts.observation['walker/left_eye'].mean(axis=-1)
#     right_eye = ts.observation['walker/right_eye'].mean(axis=-1)
#     left_eye = np.tile(left_eye[:, :, None], reps=(1, 1, 3))
#     right_eye = np.tile(right_eye[:, :, None], reps=(1, 1, 3))
#     left_eye = blow_hw(left_eye, col_repeats=9, row_repeats=8)
#     right_eye = blow_hw(right_eye, col_repeats=9, row_repeats=8)
#     left_fish_img = np.ascontiguousarray(self.retina.correct_fisheye(left_eye))
#     right_fish_img = np.ascontiguousarray(self.retina.correct_fisheye(right_eye))
#     left_eye_hex = self.retina.raw_image_to_hex_pxls(left_fish_img)
#     right_eye_hex = self.retina.raw_image_to_hex_pxls(right_fish_img)
#     hex_pixels = np.stack((left_eye_hex, right_eye_hex), axis=0)
#     ts.observation['walker/eye_hex_pixels'] = hex_pixels.astype('float32')
#     del left_eye, right_eye, left_fish_img, right_fish_img, left_eye_hex, right_eye_hex

#     self.init_pos, self.init_quat = self.get_fly_pose()

#     return ts

#   def step(self, action):
#     ts = self.env_factory.step(action)
#     info = {}
#     #
#     if ts.step_type == StepType.LAST:
#       info["terminal_observation"] = ts.observation
#       reset_ts = self.env_factory.reset()
#       ts = TimeStep(
#         step_type=ts.step_type,
#         reward=ts.reward,
#         discount=ts.discount,
#         observation=reset_ts.observation
#       )
#     #
#     left_eye = ts.observation['walker/left_eye'].mean(axis=-1)
#     right_eye = ts.observation['walker/right_eye'].mean(axis=-1)
#     left_eye = np.tile(left_eye[:, :, None], reps=(1, 1, 3))
#     right_eye = np.tile(right_eye[:, :, None], reps=(1, 1, 3))
#     left_eye = blow_hw(left_eye, col_repeats=9, row_repeats=8)
#     right_eye = blow_hw(right_eye, col_repeats=9, row_repeats=8)
#     #
#     left_fish_img = np.ascontiguousarray(self.retina.correct_fisheye(left_eye))
#     right_fish_img = np.ascontiguousarray(self.retina.correct_fisheye(right_eye))
#     left_eye_hex = self.retina.raw_image_to_hex_pxls(left_fish_img)
#     right_eye_hex = self.retina.raw_image_to_hex_pxls(right_fish_img)
#     hex_pixels = np.stack((left_eye_hex, right_eye_hex), axis=0)
#     ts.observation['walker/eye_hex_pixels'] = hex_pixels.astype('float32')
#     del left_eye, right_eye, left_fish_img, right_fish_img, left_eye_hex, right_eye_hex

#     return ts, info

#   def render(self, camera_ids, **render_kwargs):
#     """Rendering is possible from multiple cameras; in that case,
#     each element in returned `frames` is a list of cameras."""
#     return [self.env_factory.physics.render(camera_id=c, **render_kwargs) for c in camera_ids]

#   def action_spec(self,):
#     return self.env_factory.action_spec()

#   def observation_spec(self, ):
#     return self.env_factory.observation_spec()

#   def get_trench_camera_ids(self,):
#     return self.env_factory.physics.model.name2id('walker/rear', 'camera')

#   def get_fly_pose(self,):
#     # return position and quaternion.
#     pos, quat = self.env_factory._task._walker.get_pose(self.env_factory.physics)
#     # dm_control may return SynchronizingArrayWrapper objects which are not
#     # picklable by Ray. Materialize them into standalone NumPy arrays.
#     return np.asarray(pos).copy(), np.asarray(quat).copy()

#   def transform_vec_to_egocentric_frame(self, xmat):
#     return self.env_factory._task._walker.transform_vec_to_egocentric_frame(
#       self.env_factory.physics, xmat
#     )


# class ParallelEnvManager:
#     def __init__(self, num_envs, task: str):
#         self.num_envs = num_envs
#         self.actors = [RayEnvActor.remote(task=task) for _ in range(num_envs)]

#     def reset(self):
#         futures = [actor.reset.remote() for actor in self.actors]
#         return ray.get(futures)

#     def step(self, actions):
#         futures = [actor.step.remote(a) for actor, a in zip(self.actors, actions)]
#         results = ray.get(futures)
#         #
#         timesteps = [res[0] for res in results]
#         infos = [res[1] for res in results]

#         return timesteps, infos

#     def render(self, camera_ids, **render_kwargs):
#         # [env_id, camera_id, frames_shape]
#         futures = [actor.render.remote(camera_ids, **render_kwargs) for actor in self.actors]
#         return ray.get(futures)

#     def get_trench_camera_ids(self,):
#         # Assuming all envs have the same camera ids.
#         return ray.get(self.actors[0].get_trench_camera_ids.remote())

#     def action_spec(self, ):
#         return ray.get(self.actors[0].action_spec.remote())

#     def observation_spec(self, ):
#         return ray.get(self.actors[0].observation_spec.remote())

#     @staticmethod
#     def rewards(timesteps):
#         return np.array([ts.reward for ts in timesteps])

#     @staticmethod
#     def steps_type(timesteps):
#         return np.array([ts.step_type for ts in timesteps])

#     @staticmethod
#     def observations(timesteps):
#         return [ts.observation for ts in timesteps]

#     @staticmethod
#     def discounts(timesteps):
#         return np.array([ts.discount for ts in timesteps])

#     def get_fly_poses(self,):
#         futures = [actor.get_fly_pose.remote() for actor in self.actors]
#         return ray.get(futures)

#     def transform_vec_to_egocentric_frame(self, xmats):
#         futures = [actor.transform_vec_to_egocentric_frame.remote(xmat) for actor, xmat in zip(self.actors, xmats)]
#         return ray.get(futures)


# def flight_imitation_environment_factory() -> 'composer.Environment':
#     """Creates replicas of environment for the agent."""
#     # base_dir = Path(__file__).resolve().parent
#     ref_flight_path = '/home/xiaowei/fly_unnamed_torch/fly_unnamed/dataset/flight/flight-dataset_saccade-evasion_augmented.hdf5'
#     wpg_pattern_path = '/home/xiaowei/fly_unnamed_torch/fly_unnamed/dataset/flight/wing_pattern_fmech.npy'
#     env = flight_imitation(
#         ref_flight_path,
#         wpg_pattern_path,
#         terminal_com_dist=float('inf'),
#     )
#     env = SinglePrecisionWrapper(env)
#     return CanonicalSpecWrapper(env, clip=True)

# def walking_imitation_environment_factory() -> 'composer.Environment':
#     """Creates replicas of environment for the agent."""
#     # base_dir = Path(__file__).resolve().parent
#     ref_walking_path = '/home/xiaowei/fly_unnamed_torch/fly_unnamed/dataset/walk/datasets_walking-imitation/walking-dataset_female-only_snippets-16252_trk-files-0-9.hdf5'
#     env = walk_imitation(
#         ref_path=ref_walking_path,
#         terminal_com_dist=float('inf'),
#     )
#     env = SinglePrecisionWrapper(env)
#     return CanonicalSpecWrapper(env, clip=True)

# def vision_guided_flight_environment_factory() -> 'composer.Environment':
#   """Creates replicas of environment for the agent."""
#   # base_dir = Path(__file__).resolve().parent
#   # ref_flight_path = '/home/xiaowei/fly_unnamed_torch/fly_unnamed/dataset/flight/flight-dataset_saccade-evasion_augmented.hdf5'
#   wpg_pattern_path = '/home/xiaowei/fly_unnamed_torch/fly_unnamed/dataset/flight/wing_pattern_fmech.npy'
#   env = vision_guided_flight(
#     wpg_pattern_path,
#     bumps_or_trench='trench',
#   )
#   thorax = env.task.root_entity.mjcf_model.find('body', 'walker/thorax')
#   _ = thorax.add(
#     'camera', name='rear', mode='trackcom',
#     pos=(-1.566, 0.037, -0.021),
#     xyaxes=(-0.014, -1, 0, -0.012, 0, 1)
#   )
#   # m = env.physics.model
#   # for i in range(m.ncam):
#   #   cam_name = m.id2name(i, 'camera')
#   #   print(i, cam_name)
#   # print(env.physics.model.name2id('walker/rear', 'camera'))
#   # assert False
#   env = SinglePrecisionWrapper(env)

#   return CanonicalSpecWrapper(env, clip=True)

# def vision_guided_avoidance_environment_factory() -> 'composer.Environment':
#     """Creates replicas of environment for the agent."""
#     # base_dir = Path(__file__).resolve().parent
#     wpg_pattern_path = '/home/xiaowei/fly_unnamed_torch/fly_unnamed/dataset/flight/wing_pattern_fmech.npy'
#     env = vision_guided_avoidance(
#         wpg_pattern_path,
#         obstacle_type='static',
#     )
#     env = SinglePrecisionWrapper(env)
#     return CanonicalSpecWrapper(env, clip=True)
