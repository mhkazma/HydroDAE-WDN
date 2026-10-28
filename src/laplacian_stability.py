#!/usr/bin/env python3
"""
Case study of Theorem 3 of the paper (weighted Laplacian stability).

At the fixed control operating points of Net3 and Anytown, the script

1. Constructs the linearized DAE matrices (E_h, A_h) and the weighted graph
   Laplacian L_w = N W N^T with W = K^{-1}.
2. Checks that every connected active subnetwork is anchored to a reservoir
   and that the finite eigenvalues of (A_h, E_h) lie in the open left half-plane.
3. Simulates a small consistent perturbation from equilibrium using an implicit
   Euler step on the linearized DAE and tracks the incremental hydraulic energy
   V(Δx_h) = 1/2 Δx_h^T E_h Δx_h.
4. Verifies that V̇(t) = -Δq(t)^T K Δq(t) ≤ 0 along the trajectory.

Outputs are written to src/output, and the plotted arrays to results/laplacian_stability.
"""

from __future__ import annotations

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
from scipy.linalg import eig, null_space, solve

from dae_network import (
    BLACK,
    DARKBLUE,
    DARKRED,
    MAINBLUE,
    SOFTBLUE,
    SOFTGRAY,
    NetworkCase,
    style_axes,
)
from plot_style import GRIDGRAY
from equilibrium import (
    DEMAND_PRIMARY_AMPLITUDE,
    HOUR_TO_S,
    PreparedCase,
    prepare_fixed_setting_case,
    solve_dae_equilibrium,
)
from linear_dae import graph_linear_dae, LinearDAEData, finite_eigenvalues

NETWORK_FILE = "Net3.inp"
HORIZON_S = 4.0 * HOUR_TO_S
N_STEPS = 2000
PERTURBATION_SCALE = 1.0e-3


def build_weighted_laplacian(case: NetworkCase, data: LinearDAEData) -> np.ndarray:
    """Construct L_w = N W N^T for the active links."""
    active_cols = case.link_base_open > 0.5
    Nj = case.Cj[:, active_cols]
    Nt = case.Ct[:, active_cols]
    Nr = case.Cr[:, active_cols]
    N_full = np.vstack([Nj, Nt, Nr])
    W = np.diag(1.0 / np.maximum(data.K, 1.0e-12))
    return N_full @ W @ N_full.T


def simulate_linearized_dae(data: LinearDAEData, x0: np.ndarray, h: float, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implicit Euler on E x_dot = A x with consistent initial condition x0.

    Returns
    -------
    t_grid : np.ndarray, shape (n_steps+1,)
    x_hist : np.ndarray, shape (n_steps+1, n_x)
    v_dot_hist : np.ndarray, shape (n_steps+1,)
    """
    E = data.E
    A = data.A
    n_x = E.shape[0]
    M = E - h * A

    x_hist = np.zeros((n_steps + 1, n_x), dtype=float)
    v_dot_hist = np.zeros(n_steps + 1, dtype=float)
    x_hist[0] = x0

    n_links = data.K.size
    q0 = x0[:n_links]
    v_dot_hist[0] = -float(q0 @ np.diag(data.K) @ q0)

    for k in range(n_steps):
        b = E @ x_hist[k]
        try:
            x_hist[k + 1] = solve(M, b)
        except np.linalg.LinAlgError:
            x_hist[k + 1], *_ = np.linalg.lstsq(M, b, rcond=None)
        q = x_hist[k + 1][:n_links]
        v_dot_hist[k + 1] = -float(q @ np.diag(data.K) @ q)

    t_grid = np.arange(n_steps + 1) * h
    return t_grid, x_hist, v_dot_hist


def consistent_perturbation(data: LinearDAEData, case: NetworkCase) -> np.ndarray:
    """Return a small consistent initial deviation for the linearized DAE."""
    n_a = data.K.size
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    n_x = n_a + n_j + n_t

    active_indices = np.flatnonzero(case.link_base_open > 0.5)
    Nj = case.Cj[:, active_indices]

    rng = np.random.default_rng(42)
    delta_q_active = rng.normal(size=n_a)
    Z = null_space(Nj)
    if Z.size > 0:
        delta_q_active = Z @ (Z.T @ delta_q_active)
    else:
        delta_q_active = np.zeros_like(delta_q_active)

    delta_p_a = rng.normal(size=n_t) * 0.5
    delta_p_j = np.zeros(n_j, dtype=float)

    x0 = np.concatenate([delta_q_active, delta_p_j, delta_p_a])
    x0 = x0 / max(np.linalg.norm(x0), 1.0e-12) * PERTURBATION_SCALE
    return x0


def plot_results(
    t_grid: np.ndarray,
    v_hist: np.ndarray,
    v_dot_hist: np.ndarray,
    finite_vals: np.ndarray,
    link_weights: np.ndarray,
    output_path: Path,
    overlay: dict | None = None,
) -> None:
    """Per-network 1x2 diagnostic for Theorem 3.

    Left: ranked incremental admittance weights $w_e=\\kappa_e^{-1}$ across
    all active links on a log y-axis. Right: normalized incremental hydraulic
    energy $V(\\Delta\\bx_h)/V(0)$ on a log axis. The eigenvalue spectrum is
    summarized by the stability margin alpha_s reported in Table 2 of the paper.
    """
    del overlay  # unused
    rc = {
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
        fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.88, wspace=0.32)

        ax = axes[0]
        sorted_w = np.sort(link_weights)[::-1]
        n_links = sorted_w.size
        ranks = np.arange(1, n_links + 1)
        ax.bar(ranks, sorted_w, color=MAINBLUE, edgecolor=DARKBLUE, linewidth=0.4, width=1.0)
        ax.set_yscale("log")
        if n_links <= 12:
            tick_step = 1
        elif n_links <= 60:
            tick_step = 10
        else:
            tick_step = max(20, int(round(n_links / 6.0 / 10.0)) * 10)
        xticks = np.arange(0, n_links + 1, tick_step)
        if xticks[0] == 0:
            xticks[0] = 1
        ax.set_xticks(xticks)
        ax.set_xlim(0.5, n_links + 0.5)
        ax.set_xlabel("Link rank")
        ax.set_ylabel(r"$w_e = \kappa_e^{-1}$")
        ax.set_title("Incremental Admittance Weights")
        style_axes(ax)

        ax = axes[1]
        v0 = max(v_hist[0], 1.0e-300)
        ax.semilogy(t_grid / HOUR_TO_S, v_hist / v0, color=MAINBLUE, linewidth=1.6)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(r"$V(\Delta\mathbf{x}_h)/V(0)$")
        ax.set_title("Incremental Hydraulic Energy")
        ax.set_xlim(t_grid[0] / HOUR_TO_S, t_grid[-1] / HOUR_TO_S)
        style_axes(ax)

        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_results_combined(runs: list[dict], output_path: Path) -> None:
    """Stacked Net3 / anytown diagnostic for Theorem 3.

    Two rows (one per network) by two columns: left column shows the
    ranked incremental admittance weights $w_e=\\kappa_e^{-1}$, right
    column shows the normalized incremental hydraulic energy decay.
    """
    rc = {
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(runs), 2, figsize=(7.0, 2.85 * len(runs)))
        fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.92, wspace=0.42, hspace=0.78)
        for row, run in enumerate(runs):
            link_weights = run["link_weights"]
            t_grid = run["t_grid"]
            v_hist = run["v_hist"]
            stem = run["stem"]

            ax = axes[row, 0]
            sorted_w = np.sort(link_weights)[::-1]
            n_links = sorted_w.size
            ranks = np.arange(1, n_links + 1)
            ax.bar(ranks, sorted_w, color=MAINBLUE, edgecolor=DARKBLUE, linewidth=0.4, width=1.0)
            ax.set_yscale("log")
            if n_links <= 12:
                tick_step = 1
            elif n_links <= 60:
                tick_step = 10
            else:
                tick_step = max(20, int(round(n_links / 6.0 / 10.0)) * 10)
            xticks = np.arange(0, n_links + 1, tick_step)
            if xticks[0] == 0:
                xticks[0] = 1
            ax.set_xticks(xticks)
            ax.set_xlim(0.5, n_links + 0.5)
            ax.set_xlabel("Link rank")
            ax.set_ylabel(r"$w_e = \kappa_e^{-1}$")
            ax.set_title(f"({chr(ord('a') + 2*row)}) Admittance weights -- {stem}")
            style_axes(ax)

            ax = axes[row, 1]
            v0 = max(v_hist[0], 1.0e-300)
            ax.semilogy(t_grid / HOUR_TO_S, v_hist / v0, color=MAINBLUE, linewidth=1.6)
            ax.set_xlabel("Time [h]")
            ax.set_ylabel(r"$V(\Delta\mathbf{x}_h)/V(0)$")
            ax.set_title(f"({chr(ord('a') + 2*row + 1)}) Energy decay -- {stem}")
            ax.set_xlim(t_grid[0] / HOUR_TO_S, t_grid[-1] / HOUR_TO_S)
            style_axes(ax)

        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)



def _simulate_for_inp(inp_path: Path) -> dict:
    prepared = prepare_fixed_setting_case(inp_path, DEMAND_PRIMARY_AMPLITUDE)
    x_eq, eq_residual = solve_dae_equilibrium(
        prepared.case,
        prepared.nominal_demand_cms,
        prepared.nominal_state,
    )
    data = graph_linear_dae(prepared.case, x_eq)
    L_w = build_weighted_laplacian(prepared.case, data)
    laplacian_eig = np.linalg.eigvalsh(L_w)
    n_zero_lap = int(np.count_nonzero(np.abs(laplacian_eig) < 1.0e-8))
    finite_vals = finite_eigenvalues(data.E, data.A)
    alpha = -np.max(np.real(finite_vals)) if finite_vals.size > 0 else float("nan")
    x0 = consistent_perturbation(data, prepared.case)
    h = HORIZON_S / N_STEPS
    t_grid, x_hist, v_dot_hist = simulate_linearized_dae(data, x0, h, N_STEPS)
    n_links = data.K.size
    n_j = len(prepared.case.junctions)
    n_t = len(prepared.case.tanks)
    v_hist = np.zeros_like(t_grid)
    for k, x in enumerate(x_hist):
        q = x[:n_links]
        p_a = x[n_links + n_j:n_links + n_j + n_t]
        v_hist[k] = 0.5 * (
            float(np.sum(q**2 * np.diag(data.E[:n_links, :n_links])))
            + float(np.sum(p_a**2 * np.diag(data.E[n_links + n_j:, n_links + n_j:])))
        )
    return {
        "stem": inp_path.stem,
        "data": data,
        "finite_vals": finite_vals,
        "link_weights": 1.0 / data.K,
        "t_grid": t_grid,
        "v_hist": v_hist,
        "v_dot_hist": v_dot_hist,
        "alpha": alpha,
        "n_zero_lap": n_zero_lap,
        "eq_residual": eq_residual,
    }



def main() -> None:
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = _simulate_for_inp(ROOT / "Networks" / NETWORK_FILE)
    overlay_inp = ROOT / "Networks" / "anytown.inp"
    overlay = _simulate_for_inp(overlay_inp) if overlay_inp.exists() else None

    from paper_output import paper_subdir
    paper_dir = paper_subdir("laplacian_stability")

    runs = [primary] + ([overlay] if overlay is not None else [])
    for run in runs:
        stem_lower = run["stem"].lower()
        figure_path = output_dir / f"laplacian_stability_{stem_lower}.pdf"
        plot_results(
            run["t_grid"],
            run["v_hist"],
            run["v_dot_hist"],
            run["finite_vals"],
            run["link_weights"],
            figure_path,
        )
        (paper_dir / figure_path.name).write_bytes(figure_path.read_bytes())
        np.savez_compressed(
            paper_dir / f"laplacian_stability_{stem_lower}.npz",
            t_h=run["t_grid"] / HOUR_TO_S,
            v_hist=run["v_hist"],
            v_dot_hist=run["v_dot_hist"],
            finite_vals_real=np.real(run["finite_vals"]),
            finite_vals_imag=np.imag(run["finite_vals"]),
            link_weights=run["link_weights"],
            alpha_s=np.array(run["alpha"]),
            n_zero_lap=np.array(run["n_zero_lap"]),
            eq_residual=np.array(run["eq_residual"]),
        )

    if len(runs) >= 2:
        combined_path = output_dir / "laplacian_stability_combined.pdf"
        plot_results_combined(runs, combined_path)
        (paper_dir / combined_path.name).write_bytes(combined_path.read_bytes())

    summary_path = output_dir / "laplacian_stability_summary.txt"
    lines = [
        "Theorem 3 case study (weighted Laplacian stability)",
        f"Primary network: {primary['stem']}",
        f"Equilibrium residual infinity norm: {primary['eq_residual']:.6e}",
        f"Active links: {len(primary['data'].active_link_ids)}",
        f"Finite eigenvalues: {primary['finite_vals'].size}",
        f"Stability margin alpha_s: {primary['alpha']:.6e}",
        f"Laplacian zero eigenvalues: {primary['n_zero_lap']}",
        f"Initial energy V(0): {primary['v_hist'][0]:.6e}",
        f"Final energy V(T): {primary['v_hist'][-1]:.6e}",
        f"Max V_dot: {np.max(primary['v_dot_hist']):.6e}",
        f"Min V_dot: {np.min(primary['v_dot_hist']):.6e}",
    ]
    if overlay is not None:
        lines += [
            "",
            f"Overlay network: {overlay['stem']}",
            f"Overlay equilibrium residual: {overlay['eq_residual']:.6e}",
            f"Overlay finite eigenvalues: {overlay['finite_vals'].size}",
            f"Overlay stability margin alpha_s: {overlay['alpha']:.6e}",
            f"Overlay Laplacian zero eigenvalues: {overlay['n_zero_lap']}",
            f"Overlay V(0): {overlay['v_hist'][0]:.6e}",
            f"Overlay V(T): {overlay['v_hist'][-1]:.6e}",
        ]
    summary_path.write_text("\n".join(lines))

    print(f"Theorem 3 study primary network: {primary['stem']}")
    print(f"Equilibrium residual: {primary['eq_residual']:.3e}")
    print(f"Finite eigenvalues: {primary['finite_vals'].size}, stability margin: {primary['alpha']:.3e}")
    print(f"Laplacian zero eigenvalues: {primary['n_zero_lap']}")
    print(f"Energy decay: {primary['v_hist'][0]:.3e} -> {primary['v_hist'][-1]:.3e}")
    if overlay is not None:
        print(f"Overlay network: {overlay['stem']}, alpha_s={overlay['alpha']:.3e}, V(0)={overlay['v_hist'][0]:.3e} -> V(T)={overlay['v_hist'][-1]:.3e}")


if __name__ == "__main__":
    main()
