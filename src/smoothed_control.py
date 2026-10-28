#!/usr/bin/env python3
"""
Twenty-four hour DAE simulation with smoothed controls and its comparison with EPANET
(Section 4.2 and Table 1 of the paper).

The control schedule is taken from EPANET through EPyT, and each pump or valve switch is
replaced by the quintic transition of Section 2.2.2.

Run:  python3 src/smoothed_control.py --inp src/Networks/Net3.inp
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import least_squares, root

from dae_network import (
    HOUR_TO_S,
    CONTROLLED_VALVE_TYPES,
    EPANET_INVALID_FLOW_THRESHOLD_M3S,
    EPANET_INVALID_HEAD_THRESHOLD_M,
    BLACK,
    DARKBLUE,
    DARKRED,
    MAINBLUE,
    SOFTBLUE,
    SOFTGRAY,
    EpanetSeries,
    NetworkCase,
    ValveLaw,
    build_network_case,
    dae_supported_flow,
    ensure_epyt,
    epanet_supported_flow,
    flow_factor_to_cms,
    head_factor_to_m,
    junction_demands,
    junction_error_arrays,
    load_epanet_initial_condition,
    pipe_headloss,
    plot_comparison_figures,
    pressure_factor_to_head_m,
    reservoir_heads,
    resolve_inp_path,
    state_arrays_in_epanet_order,
    style_axes,
    total_vector_error,
    unpack_state,
    pump_link_residual,
)
import matplotlib.pyplot as plt


# INP file used when --inp is not given (None lists the networks to choose from)
SCRIPT_INP_FILE: Path | None = None  # e.g. "Net3.inp" or "src/Networks/Net3.inp"
SCRIPT_WINDOW_START_H = 0.0
SCRIPT_WINDOW_DURATION_H = 24.0
SCRIPT_EPANET_STEP_S = 300.0
SCRIPT_DAE_SUBSTEPS_PER_EPANET_STEP = 2
SCRIPT_APPROXIMATE_CONTROLLED_VALVES = False

FLOW_PERCENT_FLOOR_M3S = 1.0e-3
HEAD_PERCENT_FLOOR_M = 1.0
VALVE_OPEN_LOSS_MAX = 1.0e6
VALVE_OPEN_LOSS_FLOW_FLOOR_M3S = 1.0e-5
VALVE_OPEN_LOSS_HEAD_FLOOR_M = 1.0e-4
ACTIVE_STATUS_THRESHOLD = 1.5


@dataclass
class SmoothControlReference:
    time_s: np.ndarray
    open_fraction: np.ndarray
    setting_model: np.ndarray
    valve_mode_weights: np.ndarray
    valve_open_loss: np.ndarray


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


def smoothstep5(alpha: float) -> float:
    z = float(np.clip(alpha, 0.0, 1.0))
    return z**3 * (10.0 - 15.0 * z + 6.0 * z**2)


def interpolate_schedule(samples: np.ndarray, time_s: np.ndarray, t_s: float) -> np.ndarray:
    if t_s <= time_s[0]:
        return np.asarray(samples[0], dtype=float).copy()
    if t_s >= time_s[-1]:
        return np.asarray(samples[-1], dtype=float).copy()
    k = int(np.searchsorted(time_s, t_s, side="right") - 1)
    t0 = float(time_s[k])
    t1 = float(time_s[k + 1])
    if t1 <= t0:
        return np.asarray(samples[k + 1], dtype=float).copy()
    beta = smoothstep5((t_s - t0) / (t1 - t0))
    return (1.0 - beta) * np.asarray(samples[k], dtype=float) + beta * np.asarray(samples[k + 1], dtype=float)


def load_epanet_resolved_timeseries(
    case: NetworkCase,
    inp_path: Path,
    duration_s: float,
    hydraulic_step_s: float,
) -> EpanetSeries:
    ensure_epyt()
    try:
        from epyt import epanet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "EPyT is required. Install it with `python3 -m pip install --target .vendor epyt`."
        ) from exc

    step_s = max(int(round(hydraulic_step_s)), 1)
    net = epanet(str(inp_path))
    try:
        net.setTimeSimulationDuration(int(round(duration_s)))
        net.setTimeHydraulicStep(step_s)
        net.setTimeReportingStep(step_s)
        net.setTimeRuleControlStep(step_s)
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


def slice_epanet_series(epanet_series: EpanetSeries, start_s: float, end_s: float) -> EpanetSeries:
    mask = (epanet_series.time_s >= start_s - 1.0e-9) & (epanet_series.time_s <= end_s + 1.0e-9)
    if not np.any(mask):
        raise RuntimeError("No EPANET samples were found in the requested time window.")
    return EpanetSeries(
        time_s=epanet_series.time_s[mask],
        node_ids=epanet_series.node_ids,
        link_ids=epanet_series.link_ids,
        head_m=epanet_series.head_m[mask, :],
        flow_m3s=epanet_series.flow_m3s[mask, :],
        status=epanet_series.status[mask, :],
        setting=epanet_series.setting[mask, :],
    )


def default_setting_model(case: NetworkCase, col: int) -> float:
    link_id = case.supported_link_ids[col]
    link_type = case.supported_link_types[col]
    if link_type == "PUMP":
        return float(case.link_base_setting[col])
    if link_type != "VALVE":
        return 0.0
    valve = case.valves[case.valve_index_by_link[link_id]]
    law = case.valve_laws[link_id]
    kind = valve.valve_type.upper()
    if kind == "TCV":
        return float(law.coefficients[0])
    if kind in {"PBV", "PRV", "PSV", "FCV"} and law.coefficients:
        return float(law.coefficients[0])
    if law.coefficients:
        return float(law.coefficients[0])
    return float(case.link_base_setting[col])


def convert_link_setting_to_model_units(case: NetworkCase, col: int, raw_setting: float) -> float:
    if not np.isfinite(raw_setting):
        return default_setting_model(case, col)
    link_type = case.supported_link_types[col]
    if link_type == "PUMP":
        return float(raw_setting)
    if link_type != "VALVE":
        return default_setting_model(case, col)
    link_id = case.supported_link_ids[col]
    valve = case.valves[case.valve_index_by_link[link_id]]
    kind = valve.valve_type.upper()
    if kind == "FCV":
        return float(raw_setting) * flow_factor_to_cms(case.options.units)
    if kind in {"PBV", "PRV", "PSV"}:
        return float(raw_setting) * pressure_factor_to_head_m(case.options.units)
    if kind == "TCV":
        return float(raw_setting) / max(2.0 * case.gravity * valve.area_m2**2, 1.0e-12)
    return default_setting_model(case, col)


def controlled_valve_open_loss_from_sample(
    case: NetworkCase,
    epanet_series: EpanetSeries,
    valve_link_id: str,
    time_index: int,
) -> float:
    valve = case.valves[case.valve_index_by_link[valve_link_id]]
    node_to_idx = {node_id: idx for idx, node_id in enumerate(epanet_series.node_ids)}
    link_to_idx = {link_id: idx for idx, link_id in enumerate(epanet_series.link_ids)}
    i_idx = node_to_idx.get(valve.start_node)
    j_idx = node_to_idx.get(valve.end_node)
    l_idx = link_to_idx.get(valve.link_id)
    if i_idx is None or j_idx is None or l_idx is None:
        return max(valve.minor_loss / max(2.0 * case.gravity * valve.area_m2**2, 1.0e-12), 0.0)
    q = float(epanet_series.flow_m3s[time_index, l_idx])
    hi = float(epanet_series.head_m[time_index, i_idx])
    hj = float(epanet_series.head_m[time_index, j_idx])
    if not (np.isfinite(q) and np.isfinite(hi) and np.isfinite(hj)):
        return max(valve.minor_loss / max(2.0 * case.gravity * valve.area_m2**2, 1.0e-12), 0.0)
    if abs(q) <= VALVE_OPEN_LOSS_FLOW_FLOOR_M3S or abs(hi - hj) <= VALVE_OPEN_LOSS_HEAD_FLOOR_M:
        return max(valve.minor_loss / max(2.0 * case.gravity * valve.area_m2**2, 1.0e-12), 0.0)
    return float(np.clip(abs(hi - hj) / max(abs(q) ** 2, VALVE_OPEN_LOSS_FLOW_FLOOR_M3S**2), 0.0, VALVE_OPEN_LOSS_MAX))


def valve_status_weights(status_value: float) -> np.ndarray:
    if status_value <= 0.0:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    if status_value >= ACTIVE_STATUS_THRESHOLD:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return np.asarray([0.0, 1.0, 0.0], dtype=float)


def build_smooth_control_reference(case: NetworkCase, epanet_series: EpanetSeries) -> SmoothControlReference:
    link_to_idx = {link_id: idx for idx, link_id in enumerate(epanet_series.link_ids)}
    n_time = epanet_series.time_s.size
    n_links = len(case.supported_link_ids)

    open_fraction = np.ones((n_time, n_links), dtype=float)
    setting_model = np.zeros((n_time, n_links), dtype=float)
    valve_mode_weights = np.zeros((n_time, n_links, 3), dtype=float)
    valve_open_loss = np.zeros((n_time, n_links), dtype=float)

    for col, link_id in enumerate(case.supported_link_ids):
        e_idx = link_to_idx.get(link_id)
        if e_idx is None:
            setting_model[:, col] = default_setting_model(case, col)
            valve_mode_weights[:, col, 2] = 1.0
            continue

        status_series = np.asarray(epanet_series.status[:, e_idx], dtype=float)
        raw_setting_series = np.asarray(epanet_series.setting[:, e_idx], dtype=float)
        open_fraction[:, col] = (status_series > 0.0).astype(float)
        setting_model[:, col] = np.asarray(
            [convert_link_setting_to_model_units(case, col, value) for value in raw_setting_series],
            dtype=float,
        )

        if case.supported_link_types[col] != "VALVE":
            continue

        valve = case.valves[case.valve_index_by_link[link_id]]
        kind = valve.valve_type.upper()
        if kind in CONTROLLED_VALVE_TYPES and case.valve_laws[link_id].mode in {"fcv", "prv", "psv"}:
            valve_mode_weights[:, col, :] = np.asarray([valve_status_weights(v) for v in status_series], dtype=float)
            valve_open_loss[:, col] = np.asarray(
                [controlled_valve_open_loss_from_sample(case, epanet_series, link_id, k) for k in range(n_time)],
                dtype=float,
            )
        else:
            valve_mode_weights[:, col, 1] = open_fraction[:, col]
            valve_mode_weights[:, col, 2] = 1.0 - open_fraction[:, col]
            if case.valve_laws[link_id].coefficients:
                valve_open_loss[:, col] = float(case.valve_laws[link_id].coefficients[0])

    return SmoothControlReference(
        time_s=epanet_series.time_s.copy(),
        open_fraction=open_fraction,
        setting_model=setting_model,
        valve_mode_weights=valve_mode_weights,
        valve_open_loss=valve_open_loss,
    )


def controls_at(control_ref: SmoothControlReference, t_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        interpolate_schedule(control_ref.open_fraction, control_ref.time_s, t_s),
        interpolate_schedule(control_ref.setting_model, control_ref.time_s, t_s),
        interpolate_schedule(control_ref.valve_mode_weights, control_ref.time_s, t_s),
        interpolate_schedule(control_ref.valve_open_loss, control_ref.time_s, t_s),
    )


def refine_time_grid(base_time_s: np.ndarray, substeps_per_interval: int) -> tuple[np.ndarray, np.ndarray]:
    substeps = max(int(substeps_per_interval), 1)
    dense: list[float] = [float(base_time_s[0])]
    sample_indices: list[int] = [0]
    for k in range(base_time_s.size - 1):
        local = np.linspace(float(base_time_s[k]), float(base_time_s[k + 1]), substeps + 1, dtype=float)
        dense.extend(local[1:].tolist())
        sample_indices.append(len(dense) - 1)
    return np.asarray(dense, dtype=float), np.asarray(sample_indices, dtype=int)


def valve_active_residual(case: NetworkCase, col: int, q: float, state: dict[str, np.ndarray], setting_model: np.ndarray) -> float:
    link_id = case.supported_link_ids[col]
    valve = case.valves[case.valve_index_by_link[link_id]]
    kind = valve.valve_type.upper()
    setpoint = float(setting_model[col]) if np.isfinite(setting_model[col]) else default_setting_model(case, col)
    if kind == "FCV":
        return q - setpoint
    if kind == "PRV":
        downstream = valve.end_node
        if downstream in case.junction_index:
            p_down = state["p_j"][case.junction_index[downstream]]
        elif downstream in case.tank_index:
            p_down = state["p_t"][case.tank_index[downstream]]
        else:
            p_down = state["p_r"][case.reservoir_index[downstream]]
        return p_down - case.node_elevation_m.get(downstream, 0.0) - setpoint
    if kind == "PSV":
        upstream = valve.start_node
        if upstream in case.junction_index:
            p_up = state["p_j"][case.junction_index[upstream]]
        elif upstream in case.tank_index:
            p_up = state["p_t"][case.tank_index[upstream]]
        else:
            p_up = state["p_r"][case.reservoir_index[upstream]]
        return p_up - case.node_elevation_m.get(upstream, 0.0) - setpoint
    raise ValueError(f"Unsupported active controlled valve type {kind}.")


def valve_open_residual(case: NetworkCase, col: int, q: float, qdot: float, head_drop: float, setting_model: np.ndarray, valve_open_loss: np.ndarray) -> float:
    link_id = case.supported_link_ids[col]
    valve = case.valves[case.valve_index_by_link[link_id]]
    law = case.valve_laws[link_id]
    kind = valve.valve_type.upper()
    sign = 1.0 if q >= 0.0 else -1.0

    if kind in CONTROLLED_VALVE_TYPES and case.valve_laws[link_id].mode in {"fcv", "prv", "psv"}:
        k_loss = float(valve_open_loss[col])
        signed_loss = k_loss * q * abs(q)
        return qdot - case.link_inertance_gain[col] * (head_drop - signed_loss)

    if law.mode == "curve_loss":
        q_abs = abs(q)
        h_loss = np.interp(q_abs, law.q_points_m3s, law.h_points_m)
        if law.q_points_m3s.size >= 2:
            slope_left = (law.h_points_m[1] - law.h_points_m[0]) / max(law.q_points_m3s[1] - law.q_points_m3s[0], 1.0e-12)
            slope_right = (law.h_points_m[-1] - law.h_points_m[-2]) / max(law.q_points_m3s[-1] - law.q_points_m3s[-2], 1.0e-12)
            if q_abs < law.q_points_m3s[0]:
                h_loss = law.h_points_m[0] + slope_left * (q_abs - law.q_points_m3s[0])
            if q_abs > law.q_points_m3s[-1]:
                h_loss = law.h_points_m[-1] + slope_right * (q_abs - law.q_points_m3s[-1])
        signed_loss = sign * h_loss
        return qdot - case.link_inertance_gain[col] * (head_drop - signed_loss)

    if law.mode == "quadratic_loss" or kind == "TCV":
        if kind == "TCV" and np.isfinite(setting_model[col]):
            k_loss = float(setting_model[col])
        else:
            k_loss = float(law.coefficients[0])
        signed_loss = k_loss * q * abs(q)
        return qdot - case.link_inertance_gain[col] * (head_drop - signed_loss)

    if law.mode == "constant_drop":
        if kind == "PBV" and np.isfinite(setting_model[col]):
            h_set = float(setting_model[col])
        else:
            h_set = float(law.coefficients[0])
        signed_loss = sign * h_set
        return qdot - case.link_inertance_gain[col] * (head_drop - signed_loss)

    raise ValueError(f"Unsupported open-valve law {law.mode}.")


def dae_residual_smooth(
    case: NetworkCase,
    t_s: float,
    x: np.ndarray,
    xdot: np.ndarray,
    open_fraction: np.ndarray,
    setting_model: np.ndarray,
    valve_mode_weights: np.ndarray,
    valve_open_loss: np.ndarray,
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
        closed_res = q[col]
        link_type = case.supported_link_types[col]
        if link_type == "PIPE":
            open_res = qdot[col] - case.link_inertance_gain[col] * (head_drop[col] - pipe_loss[col])
            alpha = float(np.clip(open_fraction[col], 0.0, 1.0))
            link_res[col] = alpha * open_res + (1.0 - alpha) * closed_res
            continue

        if link_type == "PUMP":
            open_res = pump_link_residual(
                case,
                col,
                float(q[col]),
                float(qdot[col]),
                float(head_drop[col]),
                float(setting_model[col]),
                case.pump_laws[link_id],
            )
            alpha = float(np.clip(open_fraction[col], 0.0, 1.0))
            link_res[col] = alpha * open_res + (1.0 - alpha) * closed_res
            continue

        valve = case.valves[case.valve_index_by_link[link_id]]
        kind = valve.valve_type.upper()
        if kind in CONTROLLED_VALVE_TYPES and case.valve_laws[link_id].mode in {"fcv", "prv", "psv"}:
            weights = np.clip(valve_mode_weights[col], 0.0, 1.0)
            weight_sum = float(np.sum(weights))
            if weight_sum > 1.0e-12:
                weights = weights / weight_sum
            else:
                weights = np.asarray([0.0, 0.0, 1.0], dtype=float)
            active_res = valve_active_residual(case, col, float(q[col]), state, setting_model)
            open_res = valve_open_residual(case, col, float(q[col]), float(qdot[col]), float(head_drop[col]), setting_model, valve_open_loss)
            link_res[col] = weights[0] * active_res + weights[1] * open_res + weights[2] * closed_res
        else:
            open_res = valve_open_residual(case, col, float(q[col]), float(qdot[col]), float(head_drop[col]), setting_model, valve_open_loss)
            alpha = float(np.clip(open_fraction[col], 0.0, 1.0))
            link_res[col] = alpha * open_res + (1.0 - alpha) * closed_res

    res_tanks = case.tank_areas_m2 * p_tdot + case.Ct @ q
    res_junctions = -(case.Cj @ q) - junction_demands(case, t_s)
    res_reservoirs = state["p_r"] - reservoir_heads(case, t_s)
    return np.concatenate([link_res, res_junctions, res_tanks, res_reservoirs])


def solve_smooth_step(
    case: NetworkCase,
    x_prev: np.ndarray,
    t_prev_s: float,
    t_next_s: float,
    control_ref: SmoothControlReference,
) -> tuple[np.ndarray, float]:
    dt = t_next_s - t_prev_s
    if dt <= 1.0e-12:
        return x_prev.copy(), 0.0

    open_fraction, setting_model, valve_mode_weights, valve_open_loss = controls_at(control_ref, t_next_s)

    def residual(x_next: np.ndarray) -> np.ndarray:
        xdot = (x_next - x_prev) / dt
        return dae_residual_smooth(
            case,
            t_next_s,
            x_next,
            xdot,
            open_fraction,
            setting_model,
            valve_mode_weights,
            valve_open_loss,
        )

    x_guess = x_prev.copy()
    closed_mask = open_fraction < 1.0e-3
    x_guess[: len(case.supported_link_ids)][closed_mask] = 0.0
    n_links = len(case.supported_link_ids)
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    x_guess[n_links + n_j + n_t :] = reservoir_heads(case, t_next_s)

    sol = root(residual, x_guess, method="hybr")
    if not sol.success:
        lsq = least_squares(residual, x_guess, xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10)
        if not lsq.success:
            raise RuntimeError(f"Smoothed DAE step failed on [{t_prev_s:.2f}, {t_next_s:.2f}] s.")
        x_next = lsq.x
    else:
        x_next = sol.x
    return x_next, float(np.linalg.norm(residual(x_next), ord=np.inf))


def simulate_smoothed(
    case: NetworkCase,
    x0: np.ndarray,
    t_grid_s: np.ndarray,
    control_ref: SmoothControlReference,
) -> tuple[np.ndarray, np.ndarray]:
    x_hist = np.zeros((x0.size, t_grid_s.size), dtype=float)
    residual_hist = np.zeros(t_grid_s.size, dtype=float)
    x_hist[:, 0] = x0
    for k in range(t_grid_s.size - 1):
        x_next, residual = solve_smooth_step(case, x_hist[:, k], float(t_grid_s[k]), float(t_grid_s[k + 1]), control_ref)
        x_hist[:, k + 1] = x_next
        residual_hist[k + 1] = residual
    return x_hist, residual_hist


def apply_initial_modes_from_window(case: NetworkCase, epanet_series: EpanetSeries, control_ref: SmoothControlReference) -> None:
    link_to_idx = {link_id: idx for idx, link_id in enumerate(epanet_series.link_ids)}
    for col, link_id in enumerate(case.supported_link_ids):
        idx = link_to_idx.get(link_id)
        if idx is None:
            continue
        case.link_base_open[col] = 1.0 if float(epanet_series.status[0, idx]) > 0.0 else 0.0
        case.link_base_setting[col] = float(control_ref.setting_model[0, col])


def main() -> None:
    script_t0 = time.perf_counter()
    parser = argparse.ArgumentParser(description="Twenty-four-hour smoothed-control DAE simulation and EPANET comparison.")
    parser.add_argument("--inp", type=Path, default=None, help="Path to the EPANET INP file.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--start-hour", type=float, default=SCRIPT_WINDOW_START_H, help="Clock hour at which the extended-period window begins.")
    parser.add_argument("--duration-hours", type=float, default=SCRIPT_WINDOW_DURATION_H, help="Window duration in hours.")
    parser.add_argument("--epanet-step-s", type=float, default=SCRIPT_EPANET_STEP_S, help="EPANET hydraulic/report step in seconds.")
    parser.add_argument(
        "--dae-substeps",
        type=int,
        default=SCRIPT_DAE_SUBSTEPS_PER_EPANET_STEP,
        help="Number of DAE implicit steps taken inside each EPANET interval.",
    )
    parser.add_argument(
        "--approximate-controlled-valves",
        action="store_true",
        default=SCRIPT_APPROXIMATE_CONTROLLED_VALVES,
        help="Approximate controlled valves as quadratic-loss surrogates before building the case.",
    )
    args = parser.parse_args()

    script_inp = Path(SCRIPT_INP_FILE) if SCRIPT_INP_FILE else None
    inp_path = resolve_inp_path(args.inp if args.inp is not None else script_inp)

    start_s = float(args.start_hour) * HOUR_TO_S
    duration_s = float(args.duration_hours) * HOUR_TO_S
    end_s = start_s + duration_s

    case = build_network_case(inp_path, approximate_controlled_valves_as_gpv=bool(args.approximate_controlled_valves))
    epanet_t0 = time.perf_counter()
    epanet_full = load_epanet_resolved_timeseries(case, inp_path, end_s, args.epanet_step_s)
    epanet_runtime_s = time.perf_counter() - epanet_t0
    epanet_window = slice_epanet_series(epanet_full, start_s, end_s)
    control_ref = build_smooth_control_reference(case, epanet_window)
    apply_initial_modes_from_window(case, epanet_window, control_ref)

    x0 = load_epanet_initial_condition(case, epanet_window)
    dense_time_s, sample_indices = refine_time_grid(epanet_window.time_s, args.dae_substeps)
    dae_t0 = time.perf_counter()
    x_hist_dense, residual_hist = simulate_smoothed(case, x0, dense_time_s, control_ref)
    dae_runtime_s = time.perf_counter() - dae_t0

    x_hist_window = x_hist_dense[:, sample_indices]
    dae_head, _ = state_arrays_in_epanet_order(case, x_hist_window, epanet_window)
    epanet_flow, supported_link_ids = epanet_supported_flow(case, epanet_window)
    dae_flow = dae_supported_flow(case, x_hist_window, supported_link_ids)
    junction_err, _ = junction_error_arrays(case, dae_head, epanet_window)
    total_err = total_vector_error(case, dae_head, epanet_window, dae_flow, epanet_flow)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{inp_path.stem.lower()}_multi_step"

    from paper_output import paper_subdir, save_max_errors
    paper_dir = paper_subdir("smoothed_control")
    plot_comparison_figures(
        case,
        epanet_window,
        dae_head,
        dae_flow,
        epanet_flow,
        junction_err,
        supported_link_ids,
        total_err,
        output_dir,
        stem,
        paper_data_dir=paper_dir,
    )

    total_l2 = np.sqrt(np.nansum(total_err**2, axis=1))
    flow_err = dae_flow - epanet_flow
    max_junction_err = float(np.nanmax(np.abs(junction_err))) if junction_err.size else 0.0
    save_max_errors(paper_dir / f"{stem}_max_errors.csv", max_junction_err, float(np.nanmax(np.abs(flow_err))))

    print(f"Saved six-panel comparison plot to {output_dir / f'{stem}_epanet_comparison.pdf'}")
    print(f"Saved state-range plot to {output_dir / f'{stem}_state_ranges.pdf'}")
    print(f"Window: {args.start_hour:.2f} h to {args.start_hour + args.duration_hours:.2f} h")
    print(f"EPANET window samples: {epanet_window.time_s.size}")
    print(f"Dense DAE samples: {dense_time_s.size}")
    print(f"EPANET reference runtime: {epanet_runtime_s:.3f} s")
    print(f"DAE solve runtime: {dae_runtime_s:.3f} s")
    print(f"Total script runtime: {time.perf_counter() - script_t0:.3f} s")
    print(f"Max DAE solver residual infinity norm: {np.max(residual_hist):.6e}")
    print(f"Max total ||e(t)||_2: {np.nanmax(total_l2):.6e}")
    print(f"Max flow abs error [m^3/s]: {np.nanmax(np.abs(flow_err)):.6e}")
    print(f"Max junction-head abs error [m]: {max_junction_err:.6e}")


if __name__ == "__main__":
    main()
