#!/usr/bin/env python3
"""Stability and controllability margins of Net3 under the aggressive demand scenario (Section 4.5
and Figure 6 of the paper).

The fixed control DAE equilibrium is solved at every demand sample, the graph DAE is linearized
at each equilibrium, and the script records

* descriptor stability margin from finite generalized eigenvalues of (A_h,E_h)
* finite-mode PBH controllability margin for pump-speed inputs
* finite-horizon reachability gain from pump-speed perturbations

The script writes the NPZ cache results/operating_point_margins/{net}_hourly_operating_margins.npz,
which make_figures.py plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, root

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dae_network import HOUR_TO_S, junction_demands  # noqa: E402
from paper_output import paper_subdir, save_npz_and_csv  # noqa: E402
from fixed_control import DEMAND_PRIMARY_AMPLITUDE  # noqa: E402
from equilibrium import equilibrium_residual, prepare_fixed_setting_case  # noqa: E402
from linear_dae import graph_linear_dae, pbh_margin, stability_margin  # noqa: E402


NETWORK_SPECS = [
    {"key": "net3", "label": "Net3", "file": "Net3.inp", "demand_min_ls": 200.0, "demand_max_ls": 800.0},
]
STEM_TEMPLATE = "{key}_hourly_operating_margins"
DEMAND_MIN_LS = 200.0
DEMAND_MAX_LS = 800.0
DEMAND_MIN_FACTOR = 0.30
DEMAND_MAX_FACTOR = 1.20
SAMPLE_STEP_H = 0.5
AUTHORITY_STEP_S = 300.0
AUTHORITY_HORIZON_H = 2.0
FLOW_SCALE_M3S = 1.0
HEAD_SCALE_M = 10.0


def aggressive_total_demand_ls(hours: np.ndarray, demand_min_ls: float, demand_max_ls: float) -> np.ndarray:
    """Smooth stress-test demand with daily and subdaily oscillations."""
    h = np.asarray(hours, dtype=float)
    phase = h / 24.0
    raw = (
        0.54
        + 0.33 * np.sin(2.0 * np.pi * (phase - 0.20))
        + 0.15 * np.sin(4.0 * np.pi * (phase + 0.08))
        + 0.08 * np.sin(10.0 * np.pi * (phase - 0.06))
    )
    raw += 0.18 * np.exp(-0.5 * ((h - 19.0) / 2.0) ** 2)
    raw -= 0.16 * np.exp(-0.5 * ((h - 3.5) / 1.7) ** 2)
    raw = (raw - np.min(raw)) / max(float(np.ptp(raw)), 1.0e-12)
    return float(demand_min_ls) + (float(demand_max_ls) - float(demand_min_ls)) * raw


def rescaled_junction_demands(case, hour: float, target_total_ls: float) -> np.ndarray:
    """Preserve the EPANET spatial demand split while imposing total demand."""
    base = junction_demands(case, float(hour * HOUR_TO_S))
    base_total = float(np.sum(np.maximum(base, 0.0)))
    target_total = 1.0e-3 * float(target_total_ls)
    if base_total <= 1.0e-12:
        positive = np.maximum(base, 0.0)
        if not np.any(positive > 0.0):
            return np.full_like(base, target_total / max(base.size, 1))
        return target_total * positive / float(np.sum(positive))
    return base * (target_total / base_total)


def nominal_total_demand_ls(case) -> float:
    base = junction_demands(case, 0.0)
    return 1000.0 * float(np.sum(np.maximum(base, 0.0)))


def demand_range_for_case(case, spec: dict) -> tuple[float, float]:
    if spec["demand_min_ls"] is not None and spec["demand_max_ls"] is not None:
        return float(spec["demand_min_ls"]), float(spec["demand_max_ls"])
    nominal = max(nominal_total_demand_ls(case), 1.0)
    return DEMAND_MIN_FACTOR * nominal, DEMAND_MAX_FACTOR * nominal


def finite_horizon_pump_authority(data) -> tuple[float, float, float]:
    """Return normalized reachability gain, average gain, and unit response energy.

    The implicit Euler map keeps the descriptor constraints in the discrete
    response:

        (E - h A) dx[k+1] = E dx[k] + h B du[k].

    The singular values of the resulting finite-horizon reachability matrix
    quantify pump-speed authority around the current fixed operating point.
    """
    n_steps = max(1, int(round(AUTHORITY_HORIZON_H * HOUR_TO_S / AUTHORITY_STEP_S)))
    if data.B.size == 0 or not np.any(np.abs(data.B) > 0.0):
        return 0.0, 0.0, float("inf")

    M = data.E - AUTHORITY_STEP_S * data.A
    phi = np.linalg.solve(M, data.E)
    gamma = np.linalg.solve(M, AUTHORITY_STEP_S * data.B)

    blocks = []
    block = gamma.copy()
    for _ in range(n_steps):
        blocks.append(block)
        block = phi @ block
    reachability = np.hstack(blocks)

    n_links = len(data.active_link_ids)
    n_heads = reachability.shape[0] - n_links
    state_scale = np.concatenate(
        [
            np.full(n_links, FLOW_SCALE_M3S, dtype=float),
            np.full(max(n_heads, 0), HEAD_SCALE_M, dtype=float),
        ]
    )
    reachability_scaled = reachability / state_scale[:, None]
    singular_values = np.linalg.svd(reachability_scaled, compute_uv=False)
    gain = float(singular_values[0]) if singular_values.size else 0.0
    average_gain = float(np.linalg.norm(reachability_scaled, ord="fro") / np.sqrt(reachability_scaled.shape[1]))
    unit_response_energy = float(1.0 / max(gain * gain, 1.0e-30))
    return gain, average_gain, unit_response_energy


def snapshot_residual(case, x: np.ndarray, demand_cms: np.ndarray, tank_heads_m: np.ndarray) -> np.ndarray:
    """DAE consistency residual with tank levels fixed at reference heads."""
    full = equilibrium_residual(case, x, demand_cms)
    n_links = len(case.supported_link_ids)
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    tank_slice = slice(n_links + n_j, n_links + n_j + n_t)
    return np.concatenate(
        [
            full[: n_links + n_j],
            x[tank_slice] - tank_heads_m,
            full[n_links + n_j + n_t :],
        ]
    )


def solve_consistent_snapshot(case, demand_cms: np.ndarray, x_guess: np.ndarray, tank_heads_m: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve algebraic consistency equations for a frozen operating snapshot."""
    guess = x_guess.copy()
    n_links = len(case.supported_link_ids)
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    guess[:n_links][case.link_base_open < 0.5] = 0.0
    guess[n_links + n_j : n_links + n_j + n_t] = tank_heads_m

    def residual(x: np.ndarray) -> np.ndarray:
        return snapshot_residual(case, x, demand_cms, tank_heads_m)

    candidates: list[tuple[np.ndarray, float]] = []
    sol = root(residual, guess, method="hybr")
    if sol.success:
        r = float(np.linalg.norm(residual(sol.x), ord=np.inf))
        candidates.append((sol.x, r))

    lsq_start = sol.x if sol.success else guess
    lsq = least_squares(residual, lsq_start, xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11, max_nfev=4000)
    r_lsq = float(np.linalg.norm(residual(lsq.x), ord=np.inf))
    candidates.append((lsq.x, r_lsq))

    x_best, r_best = min(candidates, key=lambda item: item[1])
    if r_best > 1.0e-6:
        raise RuntimeError(f"Failed to solve a consistent operating snapshot; residual={r_best:.3e}")
    return x_best, r_best


def compute_hourly_margins(spec: dict) -> dict[str, np.ndarray]:
    network_file = ROOT / "Networks" / str(spec["file"])
    prepared = prepare_fixed_setting_case(network_file, DEMAND_PRIMARY_AMPLITUDE)
    case = prepared.case
    hours = np.arange(0.0, 24.0 + 0.5 * SAMPLE_STEP_H, SAMPLE_STEP_H, dtype=float)
    demand_min_ls, demand_max_ls = demand_range_for_case(case, spec)
    demand_targets = aggressive_total_demand_ls(hours, demand_min_ls, demand_max_ls)

    x_guess = prepared.nominal_state.copy()
    n_links = len(case.supported_link_ids)
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    reference_tank_heads = prepared.nominal_state[n_links + n_j : n_links + n_j + n_t].copy()
    total_demand_ls = []
    stability = []
    pbh = []
    authority_gain = []
    authority_average_gain = []
    authority_unit_energy = []
    kappa_min = []
    kappa_max = []
    consistency_residual_inf = []
    finite_modes = []

    for hour, target_ls in zip(hours, demand_targets):
        demand = rescaled_junction_demands(case, float(hour), float(target_ls))
        x_eq, residual = solve_consistent_snapshot(case, demand, x_guess, reference_tank_heads)
        x_guess = x_eq
        data = graph_linear_dae(case, x_eq)
        gain, avg_gain, unit_energy = finite_horizon_pump_authority(data)

        total_demand_ls.append(1000.0 * float(np.sum(np.maximum(demand, 0.0))))
        stability.append(stability_margin(data.E, data.A))
        pbh.append(pbh_margin(data.E, data.A, data.B))
        authority_gain.append(gain)
        authority_average_gain.append(avg_gain)
        authority_unit_energy.append(unit_energy)
        kappa_min.append(float(np.min(data.K)))
        kappa_max.append(float(np.max(data.K)))
        consistency_residual_inf.append(float(residual))
        finite_modes.append(int(data.A.shape[0] - len(case.junctions)))

    return {
        "time_h": hours,
        "total_demand_ls": np.asarray(total_demand_ls, dtype=float),
        "target_total_demand_ls": np.asarray(demand_targets, dtype=float),
        "stability_margin": np.asarray(stability, dtype=float),
        "pbh_margin": np.asarray(pbh, dtype=float),
        "authority_gain": np.asarray(authority_gain, dtype=float),
        "authority_average_gain": np.asarray(authority_average_gain, dtype=float),
        "authority_unit_energy": np.asarray(authority_unit_energy, dtype=float),
        "kappa_min": np.asarray(kappa_min, dtype=float),
        "kappa_max": np.asarray(kappa_max, dtype=float),
        "consistency_residual_inf": np.asarray(consistency_residual_inf, dtype=float),
        "equilibrium_residual_inf": np.asarray(consistency_residual_inf, dtype=float),
        "finite_mode_count": np.asarray(finite_modes, dtype=int),
        "authority_step_s": np.asarray(AUTHORITY_STEP_S),
        "authority_horizon_h": np.asarray(AUTHORITY_HORIZON_H),
        "demand_min_ls": np.asarray(demand_min_ls),
        "demand_max_ls": np.asarray(demand_max_ls),
        "network": np.asarray(str(spec["label"])),
        "network_key": np.asarray(str(spec["key"])),
    }


def main() -> None:
    for spec in NETWORK_SPECS:
        arrays = compute_hourly_margins(spec)
        stem = STEM_TEMPLATE.format(key=spec["key"])
        path = paper_subdir("operating_point_margins") / stem
        save_npz_and_csv(path, **arrays)
        print(f"Saved hourly operating-point margin cache to {path.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
