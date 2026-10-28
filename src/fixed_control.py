#!/usr/bin/env python3
"""
One hour DAE simulation with fixed controls and its comparison with EPANET (Section 4.1 and
Table 1 of the paper).

All pump and valve settings are fixed over the hour, and only the demands vary through a
smooth multiplier.

Run:  python3 src/fixed_control.py --inp src/Networks/Net3.inp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import least_squares, root

from dae_network import (
    HOUR_TO_S,
    BLACK,
    DARKBLUE,
    DARKRED,
    MAINBLUE,
    SOFTBLUE,
    SOFTGRAY,
    EpanetSeries,
    NetworkCase,
    build_network_case,
    dae_supported_flow,
    ensure_epyt,
    epanet_supported_flow,
    flow_factor_to_cms,
    head_factor_to_m,
    junction_error_arrays,
    load_epanet_initial_condition,
    pipe_headloss,
    plot_comparison_figures,
    plot_bundle,
    pump_link_residual,
    reservoir_heads,
    resolve_inp_path,
    state_arrays_in_epanet_order,
    style_axes,
    total_vector_error,
    unpack_state,
    evaluate_pump_head_gain,
    valve_link_residual_and_drop,
    EPANET_INVALID_FLOW_THRESHOLD_M3S,
    EPANET_INVALID_HEAD_THRESHOLD_M,
    CONTROLLED_VALVE_TYPES,
    VALVE_SURROGATE_MIN_MINOR_LOSS,
    ValveLaw,
    plot_error_bundle,
)
import matplotlib.pyplot as plt


# INP file used when --inp is not given (None lists the networks to choose from)
SCRIPT_INP_FILE: Path | None = None  # e.g. "Net3.inp" or "src/Networks/Net3.inp"
# Multiply all initially active pumps by this factor.
SCRIPT_PUMP_SPEED_SCALE: float = 1.0
# Optional per-pump overrides, e.g. {"PU1": 0.9, "PU2": 0.95}.
SCRIPT_PUMP_SPEED_OVERRIDES: dict[str, float] = {}

# Smooth one hour demand profile
DEMAND_PRIMARY_AMPLITUDE = 0.38
DEMAND_SECONDARY_AMPLITUDE = 0.14
DEMAND_PRIMARY_PHASE = -0.30 * np.pi
DEMAND_SECONDARY_PHASE = 0.20 * np.pi
DEMAND_TERTIARY_RATIO = 0.55
DEMAND_TERTIARY_PHASE = 0.68 * np.pi
DEMAND_POSITIVE_PULSE_RATIO = 0.34
DEMAND_POSITIVE_PULSE_CENTER = 0.34
DEMAND_POSITIVE_PULSE_WIDTH = 0.075
DEMAND_NEGATIVE_PULSE_RATIO = 0.28
DEMAND_NEGATIVE_PULSE_CENTER = 0.80
DEMAND_NEGATIVE_PULSE_WIDTH = 0.11
DEMAND_MIN_MULTIPLIER = 0.45
EPANET_ONE_STEP_PATTERN = "__DAE_ONE_STEP__"
VALVE_SURROGATE_FLOW_FLOOR_M3S = 1.0e-5
VALVE_SURROGATE_HEAD_FLOOR_M = 1.0e-3
VALVE_SURROGATE_MAX_MINOR_LOSS = 1.0e6
HEAD_PERCENT_FLOOR_M = 1.0
FLOW_PERCENT_FLOOR_M3S = 1.0e-3
DEMAND_PERCENT_FLOOR_LPS = 1.0


def one_hour_time_grid(samples: int) -> np.ndarray:
    target_intervals = max(int(samples) - 1, 1)
    exact_interval_counts = np.asarray(
        [count for count in range(1, int(HOUR_TO_S) + 1) if int(HOUR_TO_S) % count == 0],
        dtype=int,
    )
    interval_count = int(exact_interval_counts[np.argmin(np.abs(exact_interval_counts - target_intervals))])
    return np.linspace(0.0, HOUR_TO_S, interval_count + 1, dtype=float)


def one_step_pattern_values(t_grid_s: np.ndarray, demand_amplitude: float) -> np.ndarray:
    return np.asarray(
        [
            hour_demand_multiplier(
                t_s,
                demand_amplitude,
                DEMAND_SECONDARY_AMPLITUDE,
                DEMAND_PRIMARY_PHASE,
                DEMAND_SECONDARY_PHASE,
                DEMAND_MIN_MULTIPLIER,
            )
            for t_s in t_grid_s
        ],
        dtype=float,
    )


def as_int_list(values: object) -> list[int]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=int)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return [int(v) for v in arr.ravel().tolist() if int(v) != 0]


def has_controlled_valves(case: NetworkCase) -> bool:
    return any(valve.valve_type.upper() in CONTROLLED_VALVE_TYPES for valve in case.valves)


def load_initial_operating_point(case: NetworkCase, inp_path: Path) -> EpanetSeries:
    ensure_epyt()
    try:
        from epyt import epanet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "EPyT is required. Install it with `python3 -m pip install --target .vendor epyt`."
        ) from exc

    net = epanet(str(inp_path))
    try:
        net.setTimeSimulationDuration(0)
        hts = net.getComputedHydraulicTimeSeries()
        node_ids = list(net.getNodeNameID())
        link_ids = list(net.getLinkNameID())
    finally:
        net.unload()

    q_factor = flow_factor_to_cms(case.options.units)
    h_factor = head_factor_to_m(case.options.units)
    head_m = np.asarray(hts.Head, dtype=float) * h_factor
    flow_m3s = np.asarray(hts.Flow, dtype=float) * q_factor
    head_m[np.abs(head_m) >= EPANET_INVALID_HEAD_THRESHOLD_M] = np.nan
    flow_m3s[np.abs(flow_m3s) >= EPANET_INVALID_FLOW_THRESHOLD_M3S] = np.nan
    return EpanetSeries(
        time_s=np.asarray(hts.Time, dtype=float),
        node_ids=node_ids,
        link_ids=link_ids,
        head_m=head_m,
        flow_m3s=flow_m3s,
        status=np.asarray(hts.Status, dtype=float),
        setting=np.asarray(hts.Setting, dtype=float),
    )


def apply_initial_operating_point_to_case(case: NetworkCase, op_point: EpanetSeries) -> None:
    link_to_idx = {link_id: idx for idx, link_id in enumerate(op_point.link_ids)}
    node_to_idx = {node_id: idx for idx, node_id in enumerate(op_point.node_ids)}

    for col, link_id in enumerate(case.supported_link_ids):
        idx = link_to_idx.get(link_id)
        if idx is None:
            continue
        status0 = float(op_point.status[0, idx])
        case.link_base_open[col] = 1.0 if status0 > 0.0 else 0.0
        if case.supported_link_types[col] == "PUMP":
            setting0 = float(op_point.setting[0, idx])
            if np.isfinite(setting0) and setting0 > 0.0:
                case.link_base_setting[col] = setting0

    for valve in case.valves:
        law = case.valve_laws.get(valve.link_id)
        if law is None or law.mode not in {"fcv", "prv", "psv"}:
            continue
        idx = link_to_idx.get(valve.link_id)
        if idx is None:
            continue
        status0 = float(op_point.status[0, idx])
        if status0 >= 1.5:
            continue
        if status0 <= 0.0:
            col = case.supported_link_ids.index(valve.link_id)
            case.link_base_open[col] = 0.0
            continue

        q0 = float(op_point.flow_m3s[0, idx])
        i_idx = node_to_idx.get(valve.start_node)
        j_idx = node_to_idx.get(valve.end_node)
        if i_idx is not None and j_idx is not None:
            hi = float(op_point.head_m[0, i_idx])
            hj = float(op_point.head_m[0, j_idx])
        else:
            hi = np.nan
            hj = np.nan
        if np.isfinite(hi) and np.isfinite(hj) and abs(q0) > VALVE_SURROGATE_FLOW_FLOOR_M3S:
            k_loss = abs(hi - hj) / max(abs(q0) ** 2, VALVE_SURROGATE_FLOW_FLOOR_M3S**2)
        else:
            if valve.minor_loss > 0.0:
                k_loss = valve.minor_loss / (2.0 * case.gravity * valve.area_m2**2)
            else:
                k_loss = 0.0
        case.valve_laws[valve.link_id] = ValveLaw(
            mode="quadratic_loss",
            valve_type=valve.valve_type.upper(),
            coefficients=(float(k_loss),),
            q_points_m3s=np.zeros(0),
            h_points_m=np.zeros(0),
            start_node=valve.start_node,
            end_node=valve.end_node,
        )


def apply_script_pump_settings(case: NetworkCase) -> None:
    for col, link_id in enumerate(case.supported_link_ids):
        if case.supported_link_types[col] != "PUMP" or case.link_base_open[col] < 0.5:
            continue
        case.link_base_setting[col] *= SCRIPT_PUMP_SPEED_SCALE
        if link_id in SCRIPT_PUMP_SPEED_OVERRIDES:
            case.link_base_setting[col] = float(SCRIPT_PUMP_SPEED_OVERRIDES[link_id])


def apply_epanet_valve_surrogates(net: object, case: NetworkCase) -> None:
    link_to_col = {link_id: col for col, link_id in enumerate(case.supported_link_ids)}
    for valve in case.valves:
        if valve.valve_type.upper() not in CONTROLLED_VALVE_TYPES:
            continue
        law = case.valve_laws.get(valve.link_id)
        if law is None or law.mode != "quadratic_loss":
            continue
        net.setLinkTypeValveTCV(valve.link_id)
        link_index = int(net.getLinkIndex(valve.link_id))
        k_minor = float(law.coefficients[0] * 2.0 * case.gravity * valve.area_m2**2)
        status = 1 if case.link_base_open[link_to_col[valve.link_id]] >= 0.5 else 0
        try:
            net.setLinkInitialSetting(link_index, k_minor)
            net.setLinkSettings(link_index, k_minor)
        except Exception:
            pass
        try:
            net.setLinkInitialStatus(link_index, status)
            net.setLinkStatus(link_index, status)
        except Exception:
            pass


def calibrate_controlled_valve_minor_loss(case: NetworkCase, epanet_series: EpanetSeries) -> dict[str, float]:
    node_to_idx = {node_id: idx for idx, node_id in enumerate(epanet_series.node_ids)}
    link_to_idx = {link_id: idx for idx, link_id in enumerate(epanet_series.link_ids)}
    link_to_col = {link_id: col for col, link_id in enumerate(case.supported_link_ids)}
    surrogate_minor_loss: dict[str, float] = {}
    for valve in case.valves:
        if valve.valve_type.upper() not in CONTROLLED_VALVE_TYPES:
            continue
        default_k = max(valve.minor_loss, VALVE_SURROGATE_MIN_MINOR_LOSS)
        if case.link_base_open[link_to_col[valve.link_id]] < 0.5:
            surrogate_minor_loss[valve.link_id] = default_k
            continue
        link_idx = link_to_idx.get(valve.link_id)
        i_idx = node_to_idx.get(valve.start_node)
        j_idx = node_to_idx.get(valve.end_node)
        if link_idx is None or i_idx is None or j_idx is None:
            surrogate_minor_loss[valve.link_id] = default_k
            continue
        q0 = float(epanet_series.flow_m3s[0, link_idx])
        hi = float(epanet_series.head_m[0, i_idx])
        hj = float(epanet_series.head_m[0, j_idx])
        if not (np.isfinite(q0) and np.isfinite(hi) and np.isfinite(hj)):
            surrogate_minor_loss[valve.link_id] = default_k
            continue
        q_abs = abs(q0)
        head_drop = abs(hi - hj)
        if q_abs <= VALVE_SURROGATE_FLOW_FLOOR_M3S or head_drop <= VALVE_SURROGATE_HEAD_FLOOR_M:
            surrogate_minor_loss[valve.link_id] = default_k
            continue
        k_minor = 2.0 * case.gravity * valve.area_m2**2 * head_drop / max(q_abs**2, VALVE_SURROGATE_FLOW_FLOOR_M3S**2)
        surrogate_minor_loss[valve.link_id] = float(np.clip(k_minor, VALVE_SURROGATE_MIN_MINOR_LOSS, VALVE_SURROGATE_MAX_MINOR_LOSS))
    return surrogate_minor_loss


def configure_fixed_setting_epanet_network(
    net: object,
    case: NetworkCase,
    pattern_values: np.ndarray,
    report_step_s: int,
    duration_s: float = HOUR_TO_S,
    apply_valve_surrogates: bool = True,
) -> None:
    if int(net.getControlCount()) > 0:
        net.deleteControls()
    if int(net.getRuleCount()) > 0:
        net.deleteRules()

    net.setTimeSimulationDuration(int(duration_s))
    net.setTimeHydraulicStep(report_step_s)
    net.setTimeReportingStep(report_step_s)
    net.setTimePatternStep(report_step_s)
    net.setTimePatternStart(0)
    net.setTimeReportingStart(0)
    net.setTimeRuleControlStep(report_step_s)

    pattern_names = list(net.getPatternNameID())
    if EPANET_ONE_STEP_PATTERN in pattern_names:
        pattern_index = int(net.getPatternIndex(EPANET_ONE_STEP_PATTERN))
    else:
        pattern_index = int(net.addPattern(EPANET_ONE_STEP_PATTERN))
    net.setPattern(pattern_index, pattern_values.tolist())
    net.setOptionsDemandPattern(pattern_index)

    base_demands = net.getNodeBaseDemands()
    category_counts = np.asarray(net.getNodeDemandCategoriesNumber(), dtype=int)
    for junction_index in as_int_list(net.getNodeJunctionIndex()):
        n_categories = int(category_counts[junction_index - 1])
        if n_categories <= 0:
            continue
        total_base_demand = 0.0
        for category_index in range(1, n_categories + 1):
            total_base_demand += float(np.asarray(base_demands[category_index], dtype=float)[junction_index - 1])
        net.setNodeBaseDemands(junction_index, 1, total_base_demand)
        net.setNodeDemandPatternIndex(junction_index, 1, pattern_index)
        for category_index in range(2, n_categories + 1):
            net.setNodeBaseDemands(junction_index, category_index, 0.0)
            net.setNodeDemandPatternIndex(junction_index, category_index, pattern_index)

    for pump_index in as_int_list(net.getLinkPumpIndex()):
        net.setLinkPumpPatternIndex(pump_index, 0)

    link_name_to_index = {link_id: idx + 1 for idx, link_id in enumerate(net.getLinkNameID())}
    for col, link_id in enumerate(case.supported_link_ids):
        link_index = link_name_to_index.get(link_id)
        if link_index is None:
            continue
        status = 1 if case.link_base_open[col] >= 0.5 else 0
        if case.supported_link_types[col] == "PUMP":
            try:
                net.setLinkInitialStatus(link_index, status)
                net.setLinkStatus(link_index, status)
            except Exception:
                pass
            setting = float(case.link_base_setting[col])
            try:
                net.setLinkInitialSetting(link_index, setting)
                net.setLinkSettings(link_index, setting)
            except Exception:
                pass
        elif case.supported_link_types[col] == "VALVE":
            try:
                net.setLinkInitialStatus(link_index, status)
                net.setLinkStatus(link_index, status)
            except Exception:
                pass
    if apply_valve_surrogates:
        apply_epanet_valve_surrogates(net, case)


def load_fixed_setting_epanet_timeseries(
    case: NetworkCase,
    inp_path: Path,
    t_grid_s: np.ndarray,
    demand_amplitude: float,
    duration_s: float = HOUR_TO_S,
    apply_valve_surrogates: bool = True,
) -> EpanetSeries:
    ensure_epyt()
    try:
        from epyt import epanet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "EPyT is required. Install it with `python3 -m pip install --target .vendor epyt`."
        ) from exc

    report_step_s = int(round(float(t_grid_s[1] - t_grid_s[0])))
    pattern_values = one_step_pattern_values(t_grid_s, demand_amplitude)

    net = epanet(str(inp_path))
    try:
        configure_fixed_setting_epanet_network(net, case, pattern_values, report_step_s, duration_s, apply_valve_surrogates)
        hts = net.getComputedHydraulicTimeSeries()
        node_ids = list(net.getNodeNameID())
        link_ids = list(net.getLinkNameID())
    finally:
        net.unload()

    q_factor = flow_factor_to_cms(case.options.units)
    h_factor = head_factor_to_m(case.options.units)
    head_m = np.asarray(hts.Head, dtype=float) * h_factor
    flow_m3s = np.asarray(hts.Flow, dtype=float) * q_factor
    head_m[np.abs(head_m) >= EPANET_INVALID_HEAD_THRESHOLD_M] = np.nan
    flow_m3s[np.abs(flow_m3s) >= EPANET_INVALID_FLOW_THRESHOLD_M3S] = np.nan
    return EpanetSeries(
        time_s=np.asarray(hts.Time, dtype=float),
        node_ids=node_ids,
        link_ids=link_ids,
        head_m=head_m,
        flow_m3s=flow_m3s,
        status=np.asarray(hts.Status, dtype=float),
        setting=np.asarray(hts.Setting, dtype=float),
    )


def gaussian_profile(tau: np.ndarray | float, center: float, width: float) -> np.ndarray:
    tau_arr = np.asarray(tau, dtype=float)
    width_eff = max(width, 1.0e-6)
    return np.exp(-0.5 * ((tau_arr - center) / width_eff) ** 2)


def window_demand_multiplier(
    t_s: float,
    duration_s: float,
    primary_amplitude: float,
    secondary_amplitude: float,
    primary_phase: float,
    secondary_phase: float,
    min_multiplier: float,
) -> float:
    tau = np.clip(t_s / max(duration_s, 1.0), 0.0, 1.0)
    tertiary_amplitude = DEMAND_TERTIARY_RATIO * secondary_amplitude
    multiplier = (
        1.0
        + primary_amplitude * np.sin(2.0 * np.pi * tau + primary_phase)
        + secondary_amplitude * np.sin(4.0 * np.pi * tau + secondary_phase)
        + tertiary_amplitude * np.sin(6.0 * np.pi * tau + DEMAND_TERTIARY_PHASE)
        + DEMAND_POSITIVE_PULSE_RATIO * primary_amplitude * gaussian_profile(tau, DEMAND_POSITIVE_PULSE_CENTER, DEMAND_POSITIVE_PULSE_WIDTH)
        - DEMAND_NEGATIVE_PULSE_RATIO * primary_amplitude * gaussian_profile(tau, DEMAND_NEGATIVE_PULSE_CENTER, DEMAND_NEGATIVE_PULSE_WIDTH)
    )
    return float(max(min_multiplier, multiplier))


def hour_demand_multiplier(
    t_s: float,
    primary_amplitude: float,
    secondary_amplitude: float,
    primary_phase: float,
    secondary_phase: float,
    min_multiplier: float,
) -> float:
    return window_demand_multiplier(
        t_s,
        HOUR_TO_S,
        primary_amplitude,
        secondary_amplitude,
        primary_phase,
        secondary_phase,
        min_multiplier,
    )


def base_junction_demand(case: NetworkCase) -> np.ndarray:
    base = np.zeros(len(case.junctions), dtype=float)
    for idx, junction in enumerate(case.junctions):
        base[idx] = sum(term[0] for term in junction.demand_terms)
    return case.options.demand_multiplier * base


def one_step_demand(
    case: NetworkCase,
    base_demand_cms: np.ndarray,
    t_s: float,
    primary_amplitude: float,
    secondary_amplitude: float,
    primary_phase: float,
    secondary_phase: float,
    min_multiplier: float,
) -> np.ndarray:
    return hour_demand_multiplier(
        t_s,
        primary_amplitude,
        secondary_amplitude,
        primary_phase,
        secondary_phase,
        min_multiplier,
    ) * base_demand_cms


def dae_residual_one_step(
    case: NetworkCase,
    t_s: float,
    x: np.ndarray,
    xdot: np.ndarray,
    link_open: np.ndarray,
    link_setting: np.ndarray,
    base_demand_cms: np.ndarray,
    demand_amplitude: float,
) -> np.ndarray:
    n_links = len(case.supported_link_ids)
    q = x[:n_links]
    qdot = xdot[:n_links]
    state = unpack_state(case, x)
    p_tdot = xdot[n_links + len(case.junctions) : n_links + len(case.junctions) + len(case.tanks)]

    head_drop = case.Cj.T @ state["p_j"] + case.Ct.T @ state["p_t"] + case.Cr.T @ state["p_r"]
    pipe_loss = pipe_headloss(case, q)
    link_res = np.zeros(n_links, dtype=float)

    for col, link_id in enumerate(case.supported_link_ids):
        if link_open[col] < 0.5:
            link_res[col] = q[col]
            continue
        link_type = case.supported_link_types[col]
        if link_type == "PIPE":
            link_res[col] = qdot[col] - case.link_inertance_gain[col] * (head_drop[col] - pipe_loss[col])
        elif link_type == "PUMP":
            law = case.pump_laws[link_id]
            link_res[col] = pump_link_residual(
                case,
                col,
                float(q[col]),
                float(qdot[col]),
                float(head_drop[col]),
                float(link_setting[col]),
                law,
            )
        elif link_type == "VALVE":
            valve_res, _ = valve_link_residual_and_drop(case, col, q[col], qdot[col], head_drop[col], link_setting, state)
            link_res[col] = valve_res
        else:
            raise ValueError(f"Unknown link type {link_type}.")

    demand_vec = one_step_demand(
        case,
        base_demand_cms,
        t_s,
        demand_amplitude,
        DEMAND_SECONDARY_AMPLITUDE,
        DEMAND_PRIMARY_PHASE,
        DEMAND_SECONDARY_PHASE,
        DEMAND_MIN_MULTIPLIER,
    )
    res_tanks = case.tank_areas_m2 * p_tdot + case.Ct @ q
    res_junctions = -(case.Cj @ q) - demand_vec
    res_reservoirs = state["p_r"] - reservoir_heads(case, t_s)
    return np.concatenate([link_res, res_junctions, res_tanks, res_reservoirs])


def solve_fixed_step(
    case: NetworkCase,
    x_prev: np.ndarray,
    t_prev_s: float,
    t_next_s: float,
    link_open: np.ndarray,
    link_setting: np.ndarray,
    base_demand_cms: np.ndarray,
    demand_amplitude: float,
) -> tuple[np.ndarray, float]:
    dt = t_next_s - t_prev_s

    def residual(x_next: np.ndarray) -> np.ndarray:
        xdot = (x_next - x_prev) / dt
        return dae_residual_one_step(
            case,
            t_next_s,
            x_next,
            xdot,
            link_open,
            link_setting,
            base_demand_cms,
            demand_amplitude,
        )

    x_guess = x_prev.copy()
    x_guess[: len(case.supported_link_ids)][link_open < 0.5] = 0.0
    n_links = len(case.supported_link_ids)
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    x_guess[n_links + n_j + n_t :] = reservoir_heads(case, t_next_s)

    sol = root(residual, x_guess, method="hybr")
    if not sol.success:
        lsq = least_squares(residual, x_guess, xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10)
        if not lsq.success:
            raise RuntimeError(f"One-step DAE solve failed at t={t_next_s:.2f} s.")
        x_next = lsq.x
    else:
        x_next = sol.x
    return x_next, float(np.linalg.norm(residual(x_next), ord=np.inf))


def simulate_one_hour(
    case: NetworkCase,
    x0: np.ndarray,
    t_grid_s: np.ndarray,
    demand_amplitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = t_grid_s.size
    x_hist = np.zeros((x0.size, samples), dtype=float)
    residual_hist = np.zeros(samples, dtype=float)
    x_hist[:, 0] = x0
    link_open = case.link_base_open.copy()
    link_setting = case.link_base_setting.copy()
    base_demand_cms = base_junction_demand(case)

    for k in range(samples - 1):
        x_next, residual = solve_fixed_step(
            case,
            x_hist[:, k],
            float(t_grid_s[k]),
            float(t_grid_s[k + 1]),
            link_open,
            link_setting,
            base_demand_cms,
            demand_amplitude,
        )
        x_hist[:, k + 1] = x_next
        residual_hist[k + 1] = residual
    return t_grid_s, x_hist, residual_hist


def safe_percent_change(values: np.ndarray, reference: np.ndarray | float, floor: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if ref.ndim < arr.ndim:
        ref = ref.reshape((1,) * (arr.ndim - ref.ndim) + ref.shape)
    ref_full = np.broadcast_to(ref, arr.shape)
    denom = np.broadcast_to(np.maximum(np.abs(ref), floor), arr.shape)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr) & np.isfinite(ref_full)
    out[valid] = 100.0 * (arr[valid] - ref_full[valid]) / denom[valid]
    return out


def main() -> None:
    script_t0 = time.perf_counter()
    parser = argparse.ArgumentParser(description="One-hour fixed-setting DAE simulation with smoothly varying demand.")
    parser.add_argument("--inp", type=Path, default=None, help="Path to the EPANET INP file. If omitted, an interactive picker is shown.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--samples", type=int, default=61, help="Number of time samples over the one-hour horizon.")
    parser.add_argument("--demand-amplitude", type=float, default=DEMAND_PRIMARY_AMPLITUDE, help="Primary amplitude of the smooth one-hour demand multiplier.")
    parser.add_argument("--approximate-controlled-valves", action="store_true", help="Approximate unsupported valve types with GPV-style surrogates.")
    args = parser.parse_args()

    script_inp = Path(SCRIPT_INP_FILE) if SCRIPT_INP_FILE else None
    inp_path = resolve_inp_path(args.inp if args.inp is not None else script_inp)
    raw_case = build_network_case(inp_path, approximate_controlled_valves_as_gpv=False)
    initial_operating_point = load_initial_operating_point(raw_case, inp_path)
    apply_initial_operating_point_to_case(raw_case, initial_operating_point)
    apply_script_pump_settings(raw_case)
    use_valve_surrogates = bool(args.approximate_controlled_valves)
    calibration_grid_s = np.asarray([0.0, HOUR_TO_S], dtype=float)
    fixed_mode_reference = load_fixed_setting_epanet_timeseries(
        raw_case,
        inp_path,
        calibration_grid_s,
        args.demand_amplitude,
        duration_s=0.0,
        apply_valve_surrogates=False,
    )
    apply_initial_operating_point_to_case(raw_case, fixed_mode_reference)

    surrogate_minor_loss = None
    if use_valve_surrogates and has_controlled_valves(raw_case):
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
            args.demand_amplitude,
            duration_s=0.0,
            apply_valve_surrogates=True,
        )
        apply_initial_operating_point_to_case(case, fixed_mode_reference)
    t_grid_s = one_hour_time_grid(max(args.samples, 2))
    epanet_t0 = time.perf_counter()
    epanet_series = load_fixed_setting_epanet_timeseries(
        case,
        inp_path,
        t_grid_s,
        args.demand_amplitude,
        apply_valve_surrogates=use_valve_surrogates,
    )
    epanet_runtime_s = time.perf_counter() - epanet_t0
    x0 = load_epanet_initial_condition(case, epanet_series)
    t_grid_s = epanet_series.time_s.copy()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dae_t0 = time.perf_counter()
    t_grid_s, x_hist, residual_hist = simulate_one_hour(case, x0, t_grid_s, args.demand_amplitude)
    dae_runtime_s = time.perf_counter() - dae_t0
    stem = inp_path.stem.lower()

    dae_head, _ = state_arrays_in_epanet_order(case, x_hist, epanet_series)
    epanet_flow, supported_link_ids = epanet_supported_flow(case, epanet_series)
    dae_flow = dae_supported_flow(case, x_hist, supported_link_ids)
    junction_err, _ = junction_error_arrays(case, dae_head, epanet_series)
    total_err = total_vector_error(case, dae_head, epanet_series, dae_flow, epanet_flow)

    base_demand = base_junction_demand(case)
    total_demand_ls = 1000.0 * np.asarray(
        [
            np.sum(
                one_step_demand(
                    case,
                    base_demand,
                    float(t_s),
                    args.demand_amplitude,
                    DEMAND_SECONDARY_AMPLITUDE,
                    DEMAND_PRIMARY_PHASE,
                    DEMAND_SECONDARY_PHASE,
                    DEMAND_MIN_MULTIPLIER,
                )
            )
            for t_s in t_grid_s
        ],
        dtype=float,
    )

    from paper_output import paper_subdir, save_max_errors
    paper_dir = paper_subdir("fixed_control")
    plot_comparison_figures(
        case,
        epanet_series,
        dae_head,
        dae_flow,
        epanet_flow,
        junction_err,
        supported_link_ids,
        total_err,
        output_dir,
        f"{stem}_one_step",
        total_demand_ls=total_demand_ls,
        one_step=True,
        paper_data_dir=paper_dir,
    )

    flow_err = dae_flow - epanet_flow
    total_l2 = np.sqrt(np.nansum(total_err**2, axis=1))
    max_junction_err = float(np.nanmax(np.abs(junction_err))) if junction_err.size else 0.0
    save_max_errors(paper_dir / f"{stem}_one_step_max_errors.csv", max_junction_err, float(np.nanmax(np.abs(flow_err))))

    print(f"Saved six-panel comparison plot to {output_dir / f'{stem}_one_step_epanet_comparison.pdf'}")
    print(f"Saved state-range plot to {output_dir / f'{stem}_one_step_state_ranges.pdf'}")
    print(f"One-hour samples used: {t_grid_s.size}")
    print(f"EPANET reference runtime: {epanet_runtime_s:.3f} s")
    print(f"DAE solve runtime: {dae_runtime_s:.3f} s")
    print(f"Total script runtime: {time.perf_counter() - script_t0:.3f} s")
    print(f"Max DAE solver residual infinity norm: {np.max(residual_hist):.6e}")
    print(f"Max total ||e(t)||_2: {np.nanmax(total_l2):.6e}")
    print(f"Max flow abs error [m^3/s]: {np.nanmax(np.abs(flow_err)):.6e}")
    print(f"Max junction-head abs error [m]: {max_junction_err:.6e}")


if __name__ == "__main__":
    main()
