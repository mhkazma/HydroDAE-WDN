#!/usr/bin/env python3
"""
Graph linearized DAE and Algorithms 1 and 2 of the paper (Section 4.4), at the fixed control
operating points of Net3 and Anytown.

1. Algorithm 1: all pipe roughness values are perturbed together, and the first-order
   prediction of the linear DAE matrix is compared with the recomputed matrix.
2. Algorithm 2: each pipe roughness is perturbed separately, and the pipes are ranked by the
   change of the linear DAE matrix and of its stability and PBH margins.

Outputs are written to src/output, and the plotted arrays to results/linear_dae.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, norm, svdvals

from dae_network import (
    BLACK,
    DARKBLUE,
    DARKRED,
    MAINBLUE,
    SOFTBLUE,
    NetworkCase,
    evaluate_pump_head_gain,
    pipe_headloss,
    style_axes,
)
from fixed_control import DEMAND_PRIMARY_AMPLITUDE
from equilibrium import (
    prepare_fixed_setting_case,
    solve_dae_equilibrium,
)
from plot_style import GRIDGRAY


NETWORK_FILE = "Net3.inp"
CENTRAL_STEP = 1.0e-4
ROUGHNESS_DELTAS = np.asarray([0.0025, 0.005, 0.01, 0.02, 0.05, 0.10], dtype=float)
ONE_AT_A_TIME_DELTA = 0.05
TOP_COUNT = 6


@dataclass
class LinearDAEData:
    E: np.ndarray
    A: np.ndarray
    B: np.ndarray
    K: np.ndarray
    active_link_ids: list[str]
    active_link_types: list[str]


@dataclass
class ProcedureOneRow:
    delta: float
    true_shift_norm: float
    predicted_shift_norm: float
    residual_norm: float
    relative_residual: float


@dataclass
class ProcedureTwoRow:
    pipe_id: str
    start_node: str
    end_node: str
    flow_m3s: float
    sensitivity_norm: float
    alpha_shift: float
    pbh_shift: float


def scaled_case(case: NetworkCase, scale: float, pipe_id: str | None = None) -> NetworkCase:
    out = copy.deepcopy(case)
    for pipe in out.pipes:
        if pipe_id is None or pipe.link_id == pipe_id:
            pipe.roughness_raw *= scale
    return out


def link_eta(case: NetworkCase, q: np.ndarray) -> np.ndarray:
    eta = np.zeros_like(q, dtype=float)
    pipe_loss = pipe_headloss(case, q)
    for col, link_id in enumerate(case.supported_link_ids):
        link_type = case.supported_link_types[col]
        if link_type == "PIPE":
            eta[col] = pipe_loss[col]
        elif link_type == "PUMP":
            law = case.pump_laws[link_id]
            speed = float(case.link_base_setting[col])
            eta[col] = -float(evaluate_pump_head_gain(np.asarray([q[col]]), np.asarray([speed]), law)[0])
        elif link_type == "VALVE":
            # The Net3 study has no valves.  This fallback treats an open
            # valve as a local quadratic-loss link if the script is reused.
            k_loss = float(case.link_minor_resistance[col])
            eta[col] = k_loss * q[col] * abs(q[col])
        else:
            raise ValueError(f"Unsupported link type {link_type}.")
    return eta


def incremental_slopes(case: NetworkCase, q: np.ndarray, active_cols: np.ndarray) -> np.ndarray:
    slopes = np.zeros(int(np.count_nonzero(active_cols)), dtype=float)
    active_indices = np.flatnonzero(active_cols)
    for out_idx, col in enumerate(active_indices):
        step = max(1.0e-7, 1.0e-4 * max(abs(float(q[col])), 1.0e-3))
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[col] += step
        q_minus[col] -= step
        derivative = (link_eta(case, q_plus)[col] - link_eta(case, q_minus)[col]) / (2.0 * step)
        slopes[out_idx] = max(abs(float(derivative)), 1.0e-10)
    return slopes


def pump_speed_derivative(case: NetworkCase, col: int, q_value: float) -> float:
    link_id = case.supported_link_ids[col]
    law = case.pump_laws[link_id]
    speed = max(float(case.link_base_setting[col]), 1.0e-4)
    step = max(1.0e-5, 1.0e-4 * speed)
    gain_plus = evaluate_pump_head_gain(np.asarray([q_value]), np.asarray([speed + step]), law)[0]
    gain_minus = evaluate_pump_head_gain(np.asarray([q_value]), np.asarray([max(speed - step, 1.0e-6)]), law)[0]
    return float((gain_plus - gain_minus) / (2.0 * step))


def graph_linear_dae(case: NetworkCase, x_eq: np.ndarray) -> LinearDAEData:
    n_links = len(case.supported_link_ids)
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    q = x_eq[:n_links]
    active_cols = case.link_base_open > 0.5

    Cj = case.Cj[:, active_cols]
    Ct = case.Ct[:, active_cols]
    gamma = case.link_inertance_gain[active_cols]
    q_active = q[active_cols]
    kappa = incremental_slopes(case, q, active_cols)

    n_a = q_active.size
    Lambda = np.diag(1.0 / np.maximum(gamma, 1.0e-12))
    tank_area = np.diag(case.tank_areas_m2) if n_t else np.zeros((0, 0))

    E = np.block(
        [
            [Lambda, np.zeros((n_a, n_j)), np.zeros((n_a, n_t))],
            [np.zeros((n_j, n_a)), np.zeros((n_j, n_j)), np.zeros((n_j, n_t))],
            [np.zeros((n_t, n_a)), np.zeros((n_t, n_j)), tank_area],
        ]
    )
    A = np.block(
        [
            [-np.diag(kappa), Cj.T, Ct.T],
            [-Cj, np.zeros((n_j, n_j)), np.zeros((n_j, n_t))],
            [-Ct, np.zeros((n_t, n_j)), np.zeros((n_t, n_t))],
        ]
    )

    pump_cols = [idx for idx, col in enumerate(np.flatnonzero(active_cols)) if case.supported_link_types[col] == "PUMP"]
    B = np.zeros((A.shape[0], max(len(pump_cols), 1)), dtype=float)
    if pump_cols:
        for input_idx, active_idx in enumerate(pump_cols):
            col = np.flatnonzero(active_cols)[active_idx]
            B[active_idx, input_idx] = pump_speed_derivative(case, col, float(q[col]))

    active_link_ids = [case.supported_link_ids[col] for col in np.flatnonzero(active_cols)]
    active_link_types = [case.supported_link_types[col] for col in np.flatnonzero(active_cols)]
    return LinearDAEData(E=E, A=A, B=B, K=kappa, active_link_ids=active_link_ids, active_link_types=active_link_types)


def finite_eigenvalues(E: np.ndarray, A: np.ndarray) -> np.ndarray:
    vals = eig(A, E, right=False)
    vals = vals[np.isfinite(vals)]
    vals = vals[np.abs(vals) < 1.0e8]
    return vals


def stability_margin(E: np.ndarray, A: np.ndarray) -> float:
    vals = finite_eigenvalues(E, A)
    if vals.size == 0:
        return float("nan")
    return float(-np.max(np.real(vals)))


def pbh_margin(E: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    vals = finite_eigenvalues(E, A)
    if vals.size == 0:
        return float("nan")
    margins = []
    for lam in vals:
        test_matrix = np.hstack([lam * E - A, B])
        margins.append(float(svdvals(test_matrix)[-1]))
    return float(np.min(margins))


def global_matrix_at_scale(base_case: NetworkCase, x_guess: np.ndarray, demand: np.ndarray, scale: float) -> tuple[NetworkCase, np.ndarray, LinearDAEData, float]:
    case_s = scaled_case(base_case, scale)
    x_s, residual = solve_dae_equilibrium(case_s, demand, x_guess)
    return case_s, x_s, graph_linear_dae(case_s, x_s), residual


def procedure_one(base_case: NetworkCase, x_eq: np.ndarray, demand: np.ndarray, base_data: LinearDAEData) -> list[ProcedureOneRow]:
    _, _, data_plus, _ = global_matrix_at_scale(base_case, x_eq, demand, 1.0 + CENTRAL_STEP)
    _, _, data_minus, _ = global_matrix_at_scale(base_case, x_eq, demand, 1.0 - CENTRAL_STEP)
    dA_dtheta = (data_plus.A - data_minus.A) / (2.0 * CENTRAL_STEP)

    rows: list[ProcedureOneRow] = []
    for delta in ROUGHNESS_DELTAS:
        _, _, data_delta, _ = global_matrix_at_scale(base_case, x_eq, demand, 1.0 + float(delta))
        true_shift = data_delta.A - base_data.A
        prediction = dA_dtheta * float(delta)
        residual = true_shift - prediction
        true_norm = float(norm(true_shift, 2))
        pred_norm = float(norm(prediction, 2))
        res_norm = float(norm(residual, 2))
        rows.append(
            ProcedureOneRow(
                delta=float(delta),
                true_shift_norm=true_norm,
                predicted_shift_norm=pred_norm,
                residual_norm=res_norm,
                relative_residual=res_norm / max(true_norm, 1.0e-14),
            )
        )
    return rows


def procedure_two(base_case: NetworkCase, x_eq: np.ndarray, base_data: LinearDAEData) -> list[ProcedureTwoRow]:
    alpha0 = stability_margin(base_data.E, base_data.A)
    pbh0 = pbh_margin(base_data.E, base_data.A, base_data.B)

    active_flow = {link_id: x_eq[idx] for idx, link_id in enumerate(base_case.supported_link_ids)}
    rows: list[ProcedureTwoRow] = []
    for pipe in base_case.pipes:
        if pipe.link_id not in base_data.active_link_ids:
            continue
        plus_case = scaled_case(base_case, 1.0 + CENTRAL_STEP, pipe.link_id)
        minus_case = scaled_case(base_case, 1.0 - CENTRAL_STEP, pipe.link_id)
        plus_data = graph_linear_dae(plus_case, x_eq)
        minus_data = graph_linear_dae(minus_case, x_eq)
        dA = (plus_data.A - minus_data.A) / (2.0 * CENTRAL_STEP)

        pert_case = scaled_case(base_case, 1.0 + ONE_AT_A_TIME_DELTA, pipe.link_id)
        pert_data = graph_linear_dae(pert_case, x_eq)
        alpha_i = stability_margin(pert_data.E, pert_data.A)
        pbh_i = pbh_margin(pert_data.E, pert_data.A, pert_data.B)

        rows.append(
            ProcedureTwoRow(
                pipe_id=pipe.link_id,
                start_node=pipe.start_node,
                end_node=pipe.end_node,
                flow_m3s=float(active_flow.get(pipe.link_id, 0.0)),
                sensitivity_norm=float(norm(dA, 2)),
                alpha_shift=float(abs(alpha_i - alpha0)) if np.isfinite(alpha0) and np.isfinite(alpha_i) else float("nan"),
                pbh_shift=float(abs(pbh_i - pbh0)) if np.isfinite(pbh0) and np.isfinite(pbh_i) else float("nan"),
            )
        )
    rows.sort(key=lambda r: r.sensitivity_norm, reverse=True)
    return rows


def write_tables(
    network_name: str,
    eq_residual: float,
    base_data: LinearDAEData,
    proc1: list[ProcedureOneRow],
    proc2: list[ProcedureTwoRow],
    output_dir: Path,
) -> None:
    alpha0 = stability_margin(base_data.E, base_data.A)
    pbh0 = pbh_margin(base_data.E, base_data.A, base_data.B)
    lambda_min = float(np.min(base_data.K))
    lambda_max = float(np.max(base_data.K))

    csv1 = output_dir / f"procedure_one_linearization_screen_{network_name}.csv"
    csv1.write_text(
        "\n".join(
            ["network,delta,true_shift_norm,predicted_shift_norm,residual_norm,relative_residual"]
            + [
                f"{network_name},{row.delta:.8e},{row.true_shift_norm:.8e},{row.predicted_shift_norm:.8e},{row.residual_norm:.8e},{row.relative_residual:.8e}"
                for row in proc1
            ]
        )
    )

    csv2 = output_dir / f"procedure_two_parameter_ranking_{network_name}.csv"
    csv2.write_text(
        "\n".join(
            ["network,pipe_id,start_node,end_node,flow_m3s,sensitivity_norm,alpha_shift_5pct,pbh_shift_5pct"]
            + [
                f"{network_name},{row.pipe_id},{row.start_node},{row.end_node},{row.flow_m3s:.8e},{row.sensitivity_norm:.8e},{row.alpha_shift:.8e},{row.pbh_shift:.8e}"
                for row in proc2
            ]
        )
    )

    lines = [
        "Procedure theorem case study",
        f"Network: {network_name}",
        f"Equilibrium residual infinity norm: {eq_residual:.6e}",
        f"Active links in graph DAE: {len(base_data.active_link_ids)}",
        f"Finite-eigenvalue stability margin alpha: {alpha0:.6e}",
        f"Descriptor PBH margin: {pbh0:.6e}",
        f"Incremental slope range: [{lambda_min:.6e}, {lambda_max:.6e}]",
        "",
        "Algorithm 1: global roughness perturbation",
    ]
    for row in proc1:
        lines.append(
            f"  delta={row.delta:.4f}: true={row.true_shift_norm:.6e}, "
            f"prediction={row.predicted_shift_norm:.6e}, residual={row.residual_norm:.6e}, "
            f"rel={row.relative_residual:.6e}"
        )
    lines.extend(["", "Algorithm 2: top pipe directions by local A_h sensitivity"])
    for row in proc2[:TOP_COUNT]:
        lines.append(
            f"  {row.pipe_id} ({row.start_node}->{row.end_node}): "
            f"|q|={abs(row.flow_m3s):.6e}, gA={row.sensitivity_norm:.6e}, "
            f"alpha shift={row.alpha_shift:.6e}, PBH shift={row.pbh_shift:.6e}"
        )
    (output_dir / f"linear_dae_summary_{network_name}.txt").write_text("\n".join(lines))



def plot_results_combined(
    runs: list[dict],
    output_path: Path,
) -> None:
    """Overlay Algorithm 1 residual curves and grouped Algorithm 2 bars across networks.

    Each run is a dict with keys ``label``, ``color``, ``proc1``, ``proc2``.
    """
    rc = {
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 15,
        "legend.fontsize": 13,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
        fig.subplots_adjust(left=0.09, right=0.995, bottom=0.27, top=0.84, wspace=0.42)

        ax = axes[0]
        markers = ["o", "s", "^", "D"]
        for k, run in enumerate(runs):
            proc1 = run["proc1"]
            delta = np.asarray([row.delta for row in proc1])
            residual = np.asarray([row.residual_norm for row in proc1])
            ax.loglog(
                delta,
                residual,
                marker=markers[k % len(markers)],
                color=run["color"],
                linewidth=1.35,
            )
            true_shift = np.asarray([row.true_shift_norm for row in proc1])
            ax.loglog(
                delta,
                true_shift,
                marker=markers[k % len(markers)],
                color=run["color"],
                linewidth=1.0,
                linestyle=":",
                alpha=0.7,
            )
        first = runs[0]["proc1"]
        delta0 = np.asarray([row.delta for row in first])
        ax.set_xlabel(r"Roughness perturbation $\delta$")
        ax.set_ylabel(r"Matrix-shift norm $\|\Delta A_h\|_2$")
        ax.set_title("Algorithm 1")
        style_axes(ax)
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([], [], color=BLACK, marker="o", linestyle="-", linewidth=1.2, label="Residual"),
            Line2D([], [], color=BLACK, linestyle=":", linewidth=1.0, label="True shift"),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=10)

        ax = axes[1]
        n_runs = len(runs)
        x = np.arange(TOP_COUNT)
        bar_w = 0.8 / n_runs
        all_labels = []
        for k, run in enumerate(runs):
            top = run["proc2"][:TOP_COUNT]
            values = np.asarray([row.sensitivity_norm for row in top])
            ax.bar(
                x + (k - (n_runs - 1) / 2.0) * bar_w,
                values,
                width=bar_w,
                color=run["color"],
                edgecolor=BLACK,
                linewidth=0.5,
                label=run["label"],
            )
            all_labels.append([row.pipe_id for row in top])
        # Use composite tick labels: "Net3-rank/anytown-rank" with both pipe ids stacked.
        tick_labels = []
        for i in range(TOP_COUNT):
            ids = "/".join(all_labels[k][i] for k in range(n_runs))
            tick_labels.append(ids)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=9)
        ax.set_xlabel("Top-6 pipe IDs (Net3 / anytown)")
        ax.set_ylabel(r"$\|\partial A_h/\partial \theta_i\|_2$")
        ax.set_title("Algorithm 2")
        ax.legend(frameon=False, loc="upper right", fontsize=11)
        style_axes(ax)

        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def _run_for_inp(inp_path: Path) -> dict:
    prepared = prepare_fixed_setting_case(inp_path, DEMAND_PRIMARY_AMPLITUDE)
    x_eq, eq_residual = solve_dae_equilibrium(
        prepared.case,
        prepared.nominal_demand_cms,
        prepared.nominal_state,
    )
    base_data = graph_linear_dae(prepared.case, x_eq)
    proc1 = procedure_one(prepared.case, x_eq, prepared.nominal_demand_cms, base_data)
    proc2 = procedure_two(prepared.case, x_eq, base_data)
    return {
        "stem": inp_path.stem,
        "case": prepared.case,
        "x_eq": x_eq,
        "eq_residual": eq_residual,
        "base_data": base_data,
        "proc1": proc1,
        "proc2": proc2,
    }



def main() -> None:
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = _run_for_inp(ROOT / "Networks" / NETWORK_FILE)
    overlay_inp = ROOT / "Networks" / "anytown.inp"
    overlay = _run_for_inp(overlay_inp) if overlay_inp.exists() else None

    figure_path = output_dir / "linear_dae.pdf"
    runs = [{"label": primary['stem'], "color": MAINBLUE, "proc1": primary["proc1"], "proc2": primary["proc2"]}]
    if overlay is not None:
        runs.append({"label": overlay['stem'], "color": DARKRED, "proc1": overlay["proc1"], "proc2": overlay["proc2"]})
    plot_results_combined(runs, figure_path)
    write_tables(primary["stem"], primary["eq_residual"], primary["base_data"], primary["proc1"], primary["proc2"], output_dir)
    if overlay is not None:
        write_tables(overlay["stem"], overlay["eq_residual"], overlay["base_data"], overlay["proc1"], overlay["proc2"], output_dir)

    from paper_output import paper_subdir
    paper_dir = paper_subdir("linear_dae")
    (paper_dir / figure_path.name).write_bytes(figure_path.read_bytes())
    for run in runs:
        sl = run["label"].lower()
        np.savez_compressed(
            paper_dir / f"linear_dae_{sl}.npz",
            delta=np.asarray([r.delta for r in run["proc1"]]),
            residual_norm=np.asarray([r.residual_norm for r in run["proc1"]]),
            true_shift_norm=np.asarray([r.true_shift_norm for r in run["proc1"]]),
            predicted_shift_norm=np.asarray([r.predicted_shift_norm for r in run["proc1"]]),
            relative_residual=np.asarray([r.relative_residual for r in run["proc1"]]),
            top_pipe_ids=np.array([r.pipe_id for r in run["proc2"][:TOP_COUNT]]),
            top_sensitivity_norm=np.asarray([r.sensitivity_norm for r in run["proc2"][:TOP_COUNT]]),
            top_flow_m3s=np.asarray([r.flow_m3s for r in run["proc2"][:TOP_COUNT]]),
        )

    for run in runs:
        stem = run["label"]
        pr1 = run["proc1"]
        pr2 = run["proc2"]
        print(f"Procedure study network: {stem}")
        print(f"  Algorithm 1 largest relative residual: {max(row.relative_residual for row in pr1):.3e}")
        print("  Algorithm 2 top pipe sensitivities:")
        for row in pr2[:TOP_COUNT]:
            print(
                f"    {row.pipe_id}: sensitivity={row.sensitivity_norm:.3e}, "
                f"|q|={abs(row.flow_m3s):.3e} m^3/s"
            )
    print(f"Saved combined figure to {figure_path}")


if __name__ == "__main__":
    main()
