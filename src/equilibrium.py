"""Fixed control operating point of the hydraulic DAE: setup of a network case from its INP file and
EPANET initial condition, the equilibrium residual, and its solution. Used by every study script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import least_squares, root

from dae_network import (
    NetworkCase,
    build_network_case,
    pipe_headloss,
    reservoir_heads,
    unpack_state,
)
from fixed_control import (  # noqa: F401  (DEMAND_PRIMARY_AMPLITUDE and HOUR_TO_S are re-exported)
    DEMAND_PRIMARY_AMPLITUDE,
    HOUR_TO_S,
    apply_initial_operating_point_to_case,
    apply_script_pump_settings,
    base_junction_demand,
    calibrate_controlled_valve_minor_loss,
    has_controlled_valves,
    load_fixed_setting_epanet_timeseries,
    load_initial_operating_point,
    load_epanet_initial_condition,
)


@dataclass
class PreparedCase:
    case: NetworkCase
    inp_path: Path
    nominal_epanet: object
    nominal_state: np.ndarray
    nominal_demand_cms: np.ndarray
    use_valve_surrogates: bool



def prepare_fixed_setting_case(inp_path: Path, demand_amplitude: float) -> PreparedCase:
    raw_case = build_network_case(inp_path, approximate_controlled_valves_as_gpv=False)
    initial_operating_point = load_initial_operating_point(raw_case, inp_path)
    apply_initial_operating_point_to_case(raw_case, initial_operating_point)
    apply_script_pump_settings(raw_case)

    use_valve_surrogates = has_controlled_valves(raw_case)
    calibration_grid_s = np.asarray([0.0, HOUR_TO_S], dtype=float)
    fixed_mode_reference = load_fixed_setting_epanet_timeseries(
        raw_case,
        inp_path,
        calibration_grid_s,
        demand_amplitude,
        duration_s=0.0,
        apply_valve_surrogates=False,
    )
    apply_initial_operating_point_to_case(raw_case, fixed_mode_reference)

    surrogate_minor_loss = None
    if use_valve_surrogates:
        surrogate_minor_loss = calibrate_controlled_valve_minor_loss(raw_case, fixed_mode_reference)

    case = build_network_case(
        inp_path,
        approximate_controlled_valves_as_gpv=use_valve_surrogates,
        controlled_valve_minor_loss=surrogate_minor_loss,
    )
    apply_initial_operating_point_to_case(case, initial_operating_point)
    apply_script_pump_settings(case)
    apply_initial_operating_point_to_case(case, fixed_mode_reference)

    if use_valve_surrogates:
        fixed_mode_reference = load_fixed_setting_epanet_timeseries(
            case,
            inp_path,
            calibration_grid_s,
            demand_amplitude,
            duration_s=0.0,
            apply_valve_surrogates=True,
        )
        apply_initial_operating_point_to_case(case, fixed_mode_reference)

    x0 = load_epanet_initial_condition(case, fixed_mode_reference)
    d0 = base_junction_demand(case)
    return PreparedCase(
        case=case,
        inp_path=inp_path,
        nominal_epanet=fixed_mode_reference,
        nominal_state=x0,
        nominal_demand_cms=d0,
        use_valve_surrogates=use_valve_surrogates,
    )


def equilibrium_residual(case: NetworkCase, x: np.ndarray, demand_cms: np.ndarray) -> np.ndarray:
    n_links = len(case.supported_link_ids)
    q = x[:n_links]
    state = unpack_state(case, x)

    head_drop = case.Cj.T @ state["p_j"] + case.Ct.T @ state["p_t"] + case.Cr.T @ state["p_r"]
    pipe_loss = pipe_headloss(case, q)
    link_res = np.zeros(n_links, dtype=float)

    for col, link_id in enumerate(case.supported_link_ids):
        if case.link_base_open[col] < 0.5:
            link_res[col] = q[col]
            continue
        link_type = case.supported_link_types[col]
        if link_type == "PIPE":
            link_res[col] = -case.link_inertance_gain[col] * (head_drop[col] - pipe_loss[col])
        elif link_type == "PUMP":
            from dae_network import pump_link_residual

            law = case.pump_laws[link_id]
            link_res[col] = pump_link_residual(
                case,
                col,
                float(q[col]),
                0.0,
                float(head_drop[col]),
                float(case.link_base_setting[col]),
                law,
            )
        elif link_type == "VALVE":
            from dae_network import valve_link_residual_and_drop

            valve_res, _ = valve_link_residual_and_drop(
                case,
                col,
                q[col],
                0.0,
                head_drop[col],
                case.link_base_setting,
                state,
            )
            link_res[col] = valve_res
        else:
            raise ValueError(f"Unsupported link type {link_type}.")

    res_junctions = -(case.Cj @ q) - demand_cms
    res_tanks = case.Ct @ q
    res_reservoirs = state["p_r"] - reservoir_heads(case, 0.0)
    return np.concatenate([link_res, res_junctions, res_tanks, res_reservoirs])


def solve_dae_equilibrium(case: NetworkCase, demand_cms: np.ndarray, x_guess: np.ndarray) -> tuple[np.ndarray, float]:
    def residual(x: np.ndarray) -> np.ndarray:
        return equilibrium_residual(case, x, demand_cms)

    guess = x_guess.copy()
    n_links = len(case.supported_link_ids)
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    guess[:n_links][case.link_base_open < 0.5] = 0.0
    guess[n_links + n_j + n_t :] = reservoir_heads(case, 0.0)

    sol = root(residual, guess, method="hybr")
    if not sol.success:
        lsq = least_squares(residual, guess, xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10)
        if not lsq.success:
            raise RuntimeError("Failed to solve the nominal DAE equilibrium.")
        x_eq = lsq.x
    else:
        x_eq = sol.x
    return x_eq, float(np.linalg.norm(residual(x_eq), ord=np.inf))
