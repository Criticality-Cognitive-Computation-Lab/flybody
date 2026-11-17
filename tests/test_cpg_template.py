from pathlib import Path

import h5py
import numpy as np
from flybody.fly_envs import cpg_dual_fly_tracking, walk_cpg
from flybody.tasks.cpg import (
    DOFS_PER_LEG,
    LEG_NAMES,
    LEG_ORDER,
    FlyBodyCPGConfig,
    FlyBodyCPGController,
)


def _write_template(path: Path) -> None:
    qpos_names = [
        f"{dof}_{LEG_NAMES[leg]}"
        for leg in LEG_ORDER
        for dof in DOFS_PER_LEG
    ]
    frames = 36 + 4 * 37 + 1
    qpos = np.zeros((frames, len(qpos_names)), dtype=np.float64)
    phase = np.arange(frames, dtype=np.float64)[:, None]
    joint_phase = np.arange(len(qpos_names), dtype=np.float64)[None, :]
    qpos[:] = 0.02 * np.sin(2.0 * np.pi * phase / 37.0 + joint_phase / 10.0)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("timestep_seconds", data=0.002)
        h5.create_dataset("id2name/qpos", data=np.asarray(qpos_names, dtype="S"))
        h5.create_dataset("trajectories/005/qpos", data=qpos)


def test_native_cpg_controller_and_single_fly_factory(tmp_path):
    template_path = tmp_path / "walking.hdf5"
    _write_template(template_path)

    controller = FlyBodyCPGController(
        FlyBodyCPGConfig(template_path=str(template_path))
    )
    controller.set_command(
        speed=0.7,
        turn=-0.2,
        amplitude_scale=1.1,
        frequency_scale=0.8,
    )
    targets = controller.step()
    assert targets.shape == (48,)
    assert np.isfinite(targets).all()
    controller.set_command(speed=0.0)
    np.testing.assert_allclose(controller.step(), controller.neutral_pose.reshape(-1))

    env = walk_cpg(
        template_path=str(template_path),
        time_limit=0.01,
        random_state=np.random.RandomState(0),
    )
    env.reset()
    first = env.step(np.zeros(env.action_spec().shape, dtype=np.float32))
    assert first.observation
    assert env.action_spec().shape == (59,)


def test_dual_factory_accepts_native_controllers(tmp_path):
    template_path = tmp_path / "walking.hdf5"
    _write_template(template_path)

    env = cpg_dual_fly_tracking(
        template_path=str(template_path),
        time_limit=0.01,
        random_state=np.random.RandomState(1),
    )
    env.reset()
    timestep = env.step(np.zeros(env.action_spec().shape, dtype=np.float32))
    assert np.isfinite(timestep.reward)
