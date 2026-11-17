import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


LEG_MAPPING = {
    "T1_left": "LF",
    "T1_right": "RF",
    "T2_left": "LM",
    "T2_right": "RM",
    "T3_left": "LH",
    "T3_right": "RH",
}

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


def calculate_ddt(theta, r, w, phi, nu, R, alpha):
    intrinsic_term = 2 * np.pi * nu
    phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling_term = (r * w * np.sin(phase_diff - phi)).sum(axis=1)
    dtheta_dt = intrinsic_term + coupling_term
    dr_dt = alpha * (R - r)
    return dtheta_dt, dr_dt


class CPGNetwork:
    def __init__(
        self,
        timestep,
        intrinsic_freqs,
        intrinsic_amps,
        coupling_weights,
        phase_biases,
        convergence_coefs,
        init_phases=None,
        init_magnitudes=None,
        seed=0,
    ):
        self.timestep = timestep
        self.num_cpgs = intrinsic_freqs.size
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.coupling_weights = coupling_weights
        self.phase_biases = phase_biases
        self.convergence_coefs = convergence_coefs
        self.random_state = np.random.RandomState(seed)

        self.reset(init_phases, init_magnitudes)

    def step(self):
        dtheta_dt, dr_dt = calculate_ddt(
            theta=self.curr_phases,
            r=self.curr_magnitudes,
            w=self.coupling_weights,
            phi=self.phase_biases,
            nu=self.intrinsic_freqs,
            R=self.intrinsic_amps,
            alpha=self.convergence_coefs,
        )
        self.curr_phases = (self.curr_phases + dtheta_dt * self.timestep) % (2 * np.pi)
        self.curr_magnitudes = self.curr_magnitudes + dr_dt * self.timestep
        self.curr_magnitudes = np.clip(self.curr_magnitudes, 0.0, 1.0)

    def reset(self, init_phases=None, init_magnitudes=None):
        if init_phases is None:
            self.curr_phases = self.random_state.random(self.num_cpgs) * 2 * np.pi
        else:
            self.curr_phases = np.asarray(init_phases, dtype=float)

        if init_magnitudes is None:
            self.curr_magnitudes = np.zeros(self.num_cpgs, dtype=float)
        else:
            self.curr_magnitudes = np.asarray(init_magnitudes, dtype=float)


class CPGController:
    def __init__(self, cpg_network, legs, psi_funcs, psi_rest_phases, dofs_per_leg, all_leg_dofs):
        self.cpg_network = cpg_network
        self.legs = legs
        self.psi_funcs = psi_funcs
        self.psi_rest_phases = psi_rest_phases
        self.dofs_per_leg = dofs_per_leg
        self.all_leg_dofs = all_leg_dofs

    def step(self):
        self.cpg_network.step()
        joints_angles = {}

        for i, leg in enumerate(self.legs):
            psi = self.psi_funcs[leg](self.cpg_network.curr_phases[i])
            adjusted_psi = psi

            for dof, angle in zip(self.dofs_per_leg, adjusted_psi):
                joints_angles[f"joint_{leg}{dof}"] = float(angle)

        action = {"joints": np.array([joints_angles[dof] for dof in self.all_leg_dofs])}
        return action


def load_single_step_data(
    csv_path,
    sim_timestep=0.002,
    leg_mapping=None,
    dofs_per_leg=None,
):
    """Load one walking cycle from CSV and convert to CPG joint-key format."""
    leg_mapping = LEG_MAPPING if leg_mapping is None else leg_mapping
    dofs_per_leg = DOFS_PER_LEG if dofs_per_leg is None else dofs_per_leg

    df = pd.read_csv(csv_path)
    single_steps_data = {"meta": {"timestep": sim_timestep}}

    for original_leg, short_leg in leg_mapping.items():
        for dof_name in dofs_per_leg:
            src_col = f"{dof_name}_{original_leg}"
            dst_key = f"joint_{short_leg}{dof_name}"

            if src_col not in df.columns:
                raise KeyError(f"Missing column in CPG CSV: {src_col}")

            single_steps_data[dst_key] = df[src_col].to_numpy(dtype=float)

    for key, values in single_steps_data.items():
        if not key.startswith("joint_"):
            continue
        if np.isclose(values[0], values[-1], atol=1e-6):
            single_steps_data[key] = values[:-1]

    return single_steps_data


def build_psi_functions(single_steps_data, legs=None, dofs_per_leg=None):
    """Create periodic spline ψ(phase) for each leg from one-cycle joint traces."""
    legs = LEG_ORDER if legs is None else legs
    dofs_per_leg = DOFS_PER_LEG if dofs_per_leg is None else dofs_per_leg

    sample_key = f"joint_{legs[0]}{dofs_per_leg[0]}"
    steps = len(single_steps_data[sample_key])

    # For periodic spline, include endpoint and enforce end equals start.
    phase_grid = np.linspace(0.0, 2.0 * np.pi, steps)
    psi_funcs = {}

    for leg in legs:
        joint_angles = np.array(
            [single_steps_data[f"joint_{leg}{dof}"] for dof in dofs_per_leg],
            dtype=float,
        )
        joint_angles[:, -1] = joint_angles[:, 0]
        psi_funcs[leg] = CubicSpline(
            phase_grid,
            joint_angles,
            axis=1,
            bc_type="periodic",
        )

    return psi_funcs


def build_tripod_phase_biases(legs=None):
    """Build tripod gait phase bias matrix used in the notebook example."""
    legs = LEG_ORDER if legs is None else legs
    phase_biases = np.zeros((len(legs), len(legs)), dtype=float)

    for i, leg_i in enumerate(legs):
        for j, leg_j in enumerate(legs):
            same_group = (
                (leg_i in TRIPOD_GROUP_A and leg_j in TRIPOD_GROUP_A)
                or (leg_i in TRIPOD_GROUP_B and leg_j in TRIPOD_GROUP_B)
            )
            phase_biases[i, j] = 0.0 if same_group else np.pi

    return phase_biases


def make_default_cpg_components(csv_path, sim_timestep=0.002, seed=1):
    """Create notebook-equivalent CPG network/controller and metadata."""
    legs = LEG_ORDER.copy()
    dofs_per_leg = DOFS_PER_LEG.copy()

    single_steps_data = load_single_step_data(
        csv_path=csv_path,
        sim_timestep=sim_timestep,
        leg_mapping=LEG_MAPPING,
        dofs_per_leg=dofs_per_leg,
    )
    psi_funcs = build_psi_functions(single_steps_data, legs=legs, dofs_per_leg=dofs_per_leg)

    sample_key = f"joint_{legs[0]}{dofs_per_leg[0]}"
    preprogrammed_steps_length = len(single_steps_data[sample_key])
    preprogrammed_steps_timestep = float(single_steps_data["meta"]["timestep"])
    cycle_duration = preprogrammed_steps_length * preprogrammed_steps_timestep
    base_freq = 1.0 / cycle_duration

    num_legs = len(legs)
    intrinsic_freqs = np.ones(num_legs, dtype=float) * base_freq
    intrinsic_amps = np.ones(num_legs, dtype=float)
    phase_biases = build_tripod_phase_biases(legs=legs)
    coupling_weights = np.ones((num_legs, num_legs), dtype=float) * 5.0
    np.fill_diagonal(coupling_weights, 0.0)
    convergence_coefs = np.ones(num_legs, dtype=float) * 20.0

    init_phases = np.array(
        [
            0.0,  # LF
            np.pi,  # RF
            np.pi,  # LM
            0.0,  # RM
            0.0,  # LH
            np.pi,  # RH
        ],
        dtype=float,
    )
    init_magnitudes = np.ones(num_legs, dtype=float)

    cpg_network = CPGNetwork(
        timestep=preprogrammed_steps_timestep,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
        init_phases=init_phases.copy(),
        init_magnitudes=init_magnitudes.copy(),
        seed=seed,
    )

    psi_rest_phases = np.zeros(num_legs, dtype=float)
    all_leg_dofs = [f"joint_{leg}{dof}" for leg in legs for dof in dofs_per_leg]

    cpg_controller = CPGController(
        cpg_network=cpg_network,
        legs=legs,
        psi_funcs=psi_funcs,
        psi_rest_phases=psi_rest_phases,
        dofs_per_leg=dofs_per_leg,
        all_leg_dofs=all_leg_dofs,
    )

    return {
        "cpg_network": cpg_network,
        "cpg_controller": cpg_controller,
        "single_steps_data": single_steps_data,
        "legs": legs,
        "dofs_per_leg": dofs_per_leg,
        "all_leg_dofs": all_leg_dofs,
        "base_freq": base_freq,
    }
