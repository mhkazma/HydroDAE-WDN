#!/usr/bin/env python3
"""Controllability margin sigma_c, eigenvector conditioning kappa(V), and the Algorithm 2
remaining margins (Sections 4.4 and 4.5 of the paper).

Theorem 5 defines two margins of the reduced linearized DAE: the stability margin alpha_s and
the controllability margin sigma_c = sigma_min(K_N), the smallest singular value of the
controllability matrix over a horizon N. This script computes

  * sigma_c on the consistent differential states. The reduced state dz = {dq, dp_A} of
    Theorem 5 evolves on the subspace N_J dq = 0 (linearized junction balance). K_N is the
    finite horizon reachability matrix of the implicit Euler discretization of the linear DAE
    (step tau, horizon H, as in the pump allocation study), its differential rows are scaled by
    1 m3/s for flows and 10 m for heads, and projected onto an orthonormal basis of that
    subspace before the smallest singular value is taken. Without the projection the
    smallest singular value is zero by construction, because the junction constraints make
    the differential state space larger than the set of reachable consistent states.
  * kappa(V) = cond(V_f), the condition number of the matrix of right eigenvectors of the
    finite modes of the pencil (A_h, E_h), which is the constant in (25a).
  * Algorithm 2 per pipe: g_A,i = ||dA_h/dtheta_i||_2, g_K,i = ||dK_N/dtheta_i||_2, and the
    guaranteed remaining margins alpha_hat_i = alpha_s - kappa(V) g_A,i |dtheta_i| and
    sigma_hat_i = sigma_c - g_K,i |dtheta_i| at the one at a time roughness change of Text S9.
  * sigma_c and kappa(V) along the Net3 demand variation of Section 4.5 (same samples as
    operating_point_margins.py), shown in Figure 6 next to the PBH margin.

Outputs: results/controllability_margins/{net3,anytown}_margins.npz,
         results/controllability_margins/net3_demand_sigma_c.npz,
         src/output/controllability_margins_summary.txt.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eig, null_space

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from fixed_control import DEMAND_PRIMARY_AMPLITUDE  # noqa: E402
from equilibrium import prepare_fixed_setting_case, solve_dae_equilibrium  # noqa: E402
from linear_dae import graph_linear_dae, stability_margin, pbh_margin, scaled_case, LinearDAEData  # noqa: E402
from paper_output import paper_subdir, save_npz_and_csv  # noqa: E402

STEP_S = 60.0                 # tau, as in the pump allocation study (Section 4.6)
HORIZON_H = 2.0               # H
FLOW_SCALE_M3S = 1.0
HEAD_SCALE_M = 10.0
CENTRAL_STEP = 1.0e-4         # finite difference step for the roughness sensitivities (Text S9: h_theta)
ONE_AT_A_TIME_DELTA = 0.05    # single pipe roughness change (Text S9)
TOP = 6
NETWORKS = ["Net3.inp", "anytown.inp"]


def consistent_basis(case, data: LinearDAEData) -> np.ndarray:
    """Orthonormal basis of the consistent differential states {dq, dp_A} with N_J dq = 0."""
    n_a = len(data.active_link_ids); n_j = len(case.junctions); n_t = len(case.tanks)
    N_J = -data.A[n_a:n_a + n_j, :n_a]
    Qq = null_space(N_J)
    Q = np.zeros((n_a + n_t, Qq.shape[1] + n_t))
    Q[:n_a, :Qq.shape[1]] = Qq
    Q[n_a:, Qq.shape[1]:] = np.eye(n_t)
    return Q


def reachability_consistent(case, data: LinearDAEData, Q: np.ndarray) -> np.ndarray:
    """K_N in the coordinates of the consistent differential subspace."""
    n_a = len(data.active_link_ids); n_j = len(case.junctions); n_t = len(case.tanks)
    n_steps = int(round(HORIZON_H * 3600.0 / STEP_S))
    M = data.E - STEP_S * data.A
    phi = np.linalg.solve(M, data.E)
    gamma = np.linalg.solve(M, STEP_S * data.B)
    blocks, block = [], gamma.copy()
    for _ in range(n_steps):
        blocks.append(block); block = phi @ block
    R = np.hstack(blocks)
    rows = np.r_[0:n_a, n_a + n_j:n_a + n_j + n_t]
    scale = np.concatenate([np.full(n_a, FLOW_SCALE_M3S), np.full(n_t, HEAD_SCALE_M)])
    return Q.T @ (R[rows, :] / scale[:, None])


def sigma_c(case, data, Q) -> float:
    Rc = reachability_consistent(case, data, Q)
    return float(np.linalg.svd(Rc, compute_uv=False)[-1])


def kappa_V(data: LinearDAEData) -> tuple[float, int]:
    vals, vecs = eig(data.A, data.E, right=True)
    keep = np.isfinite(vals) & (np.abs(vals) < 1.0e8)
    V = vecs[:, keep]
    s = np.linalg.svd(V, compute_uv=False)
    return float(s[0] / s[-1]), int(keep.sum())


def fixed_point_study(inp_name: str) -> dict:
    inp = ROOT / "Networks" / inp_name
    prepared = prepare_fixed_setting_case(inp, DEMAND_PRIMARY_AMPLITUDE)
    case = prepared.case
    x_eq, res = solve_dae_equilibrium(case, prepared.nominal_demand_cms, prepared.nominal_state)
    data = graph_linear_dae(case, x_eq)
    Q = consistent_basis(case, data)
    alpha = stability_margin(data.E, data.A)
    pbh = pbh_margin(data.E, data.A, data.B)
    sig = sigma_c(case, data, Q)
    kap, n_f = kappa_V(data)
    R0 = reachability_consistent(case, data, Q)
    sig_max = float(np.linalg.svd(R0, compute_uv=False)[0])   # largest singular value of K_N
    total_demand_ls = 1000.0 * float(np.sum(np.maximum(prepared.nominal_demand_cms, 0.0)))

    pipe_ids, gA, gK, flows = [], [], [], []
    for pipe in case.pipes:
        if pipe.link_id not in data.active_link_ids:
            continue
        dp = graph_linear_dae(scaled_case(case, 1.0 + CENTRAL_STEP, pipe.link_id), x_eq)
        dm = graph_linear_dae(scaled_case(case, 1.0 - CENTRAL_STEP, pipe.link_id), x_eq)
        gA.append(float(np.linalg.norm((dp.A - dm.A) / (2 * CENTRAL_STEP), 2)))
        Rp = reachability_consistent(case, dp, Q); Rm = reachability_consistent(case, dm, Q)
        gK.append(float(np.linalg.norm((Rp - Rm) / (2 * CENTRAL_STEP), 2)))
        pipe_ids.append(pipe.link_id)
        flows.append(float(x_eq[case.supported_link_ids.index(pipe.link_id)]))
    gA = np.asarray(gA); gK = np.asarray(gK)
    alpha_hat = alpha - kap * gA * ONE_AT_A_TIME_DELTA
    sigma_hat = sig - gK * ONE_AT_A_TIME_DELTA
    # exact margins after the one at a time change, for comparison with the guaranteed ones
    alpha_exact, sigma_exact = [], []
    for pid in pipe_ids:
        d = graph_linear_dae(scaled_case(case, 1.0 + ONE_AT_A_TIME_DELTA, pid), x_eq)
        alpha_exact.append(stability_margin(d.E, d.A)); sigma_exact.append(sigma_c(case, d, Q))
    order_A = np.argsort(-gA); order_K = np.argsort(-gK)
    return dict(
        stem=inp.stem, total_demand_ls=total_demand_ls, eq_residual=res, alpha_s=alpha, pbh=pbh, sigma_c=sig, sigma_max=sig_max,
        kappa_V=kap, n_finite_modes=n_f, n_consistent=Q.shape[1], n_columns=R0.shape[1],
        pipe_ids=np.asarray(pipe_ids), flow_m3s=np.asarray(flows), gA=gA, gK=gK,
        alpha_hat=alpha_hat, sigma_hat=sigma_hat,
        alpha_exact=np.asarray(alpha_exact), sigma_exact=np.asarray(sigma_exact),
        top_A_ids=np.asarray(pipe_ids)[order_A[:TOP]], top_K_ids=np.asarray(pipe_ids)[order_K[:TOP]],
    )


def demand_variation_net3() -> dict:
    import operating_point_margins as opm  # noqa: E402
    spec = [s for s in opm.NETWORK_SPECS if s["key"] == "net3"][0]
    prepared = prepare_fixed_setting_case(ROOT / "Networks" / spec["file"], DEMAND_PRIMARY_AMPLITUDE)
    case = prepared.case
    hours = np.arange(0.0, 24.0 + 0.5 * opm.SAMPLE_STEP_H, opm.SAMPLE_STEP_H)
    dmin, dmax = opm.demand_range_for_case(case, spec)
    targets = opm.aggressive_total_demand_ls(hours, dmin, dmax)
    n_links, n_j, n_t = len(case.supported_link_ids), len(case.junctions), len(case.tanks)
    tank_ref = prepared.nominal_state[n_links + n_j:n_links + n_j + n_t].copy()
    x_guess = prepared.nominal_state.copy()
    sig, kap, dem, alpha, pbh = [], [], [], [], []
    Q = None
    for hour, target in zip(hours, targets):
        demand = opm.rescaled_junction_demands(case, float(hour), float(target))
        x_eq, _ = opm.solve_consistent_snapshot(case, demand, x_guess, tank_ref)
        x_guess = x_eq
        data = graph_linear_dae(case, x_eq)
        if Q is None:
            Q = consistent_basis(case, data)
        sig.append(sigma_c(case, data, Q)); kap.append(kappa_V(data)[0])
        alpha.append(stability_margin(data.E, data.A)); pbh.append(pbh_margin(data.E, data.A, data.B))
        dem.append(1000.0 * float(np.sum(np.maximum(demand, 0.0))))
    return dict(time_h=hours, total_demand_ls=np.asarray(dem), sigma_c=np.asarray(sig), kappa_V=np.asarray(kap),
                stability_margin=np.asarray(alpha), pbh_margin=np.asarray(pbh))


def main() -> None:
    t0 = time.perf_counter()
    pdir = paper_subdir("controllability_margins")
    lines = [f"Controllability margins: tau = {STEP_S:.0f} s, H = {HORIZON_H:.0f} h, scaling {FLOW_SCALE_M3S} m3/s and {HEAD_SCALE_M} m, one at a time roughness change {ONE_AT_A_TIME_DELTA:.0%}"]
    for name in NETWORKS:
        r = fixed_point_study(name)
        save_npz_and_csv(pdir / f"{r['stem'].lower()}_margins", **{k: v for k, v in r.items()})
        iA = [list(r["pipe_ids"]).index(p) for p in r["top_A_ids"]]
        iK = [list(r["pipe_ids"]).index(p) for p in r["top_K_ids"]]
        lines += [
            f"\n{r['stem']}: nominal total demand {r['total_demand_ls']:.1f} L/s, equilibrium residual {r['eq_residual']:.2e}",
            f"  finite modes {r['n_finite_modes']}, consistent differential states {r['n_consistent']}, K_N columns {r['n_columns']}",
            f"  alpha_s = {r['alpha_s']:.3e} 1/s, PBH = {r['pbh']:.3e}, sigma_c = {r['sigma_c']:.3e} (largest {r['sigma_max']:.3g}), kappa(V) = {r['kappa_V']:.3e}",
            f"  pipes with largest g_A (||dA_h/dtheta_i||): " + ", ".join(f"{p} ({r['gA'][i]:.3g})" for p, i in zip(r['top_A_ids'], iA)),
            f"  pipes with largest g_K (||dK_N/dtheta_i||): " + ", ".join(f"{p} ({r['gK'][i]:.3g})" for p, i in zip(r['top_K_ids'], iK)),
            f"  guaranteed remaining alpha_hat at {ONE_AT_A_TIME_DELTA:.0%}: min {r['alpha_hat'].min():.3e} (pipe {r['pipe_ids'][np.argmin(r['alpha_hat'])]}), exact alpha after change: min {r['alpha_exact'].min():.3e}, max {r['alpha_exact'].max():.3e}",
            f"  guaranteed remaining sigma_hat at {ONE_AT_A_TIME_DELTA:.0%}: min {r['sigma_hat'].min():.3e} (pipe {r['pipe_ids'][np.argmin(r['sigma_hat'])]}), exact sigma_c after change: min {r['sigma_exact'].min():.3e}, max {r['sigma_exact'].max():.3e}",
            f"  pipes with the smallest guaranteed alpha_hat: " + ", ".join(str(p) for p in r['pipe_ids'][np.argsort(r['alpha_hat'])[:TOP]]),
            f"  pipes with the smallest guaranteed sigma_hat: " + ", ".join(str(p) for p in r['pipe_ids'][np.argsort(r['sigma_hat'])[:TOP]]),
        ]
    sw = demand_variation_net3()
    save_npz_and_csv(pdir / "net3_demand_sigma_c", **sw)
    lines += [f"\nNet3 demand variation: sigma_c from {sw['sigma_c'].min():.3e} to {sw['sigma_c'].max():.3e}; kappa(V) from {sw['kappa_V'].min():.3e} to {sw['kappa_V'].max():.3e}; alpha_s {sw['stability_margin'].min():.3e} to {sw['stability_margin'].max():.3e}; demand {sw['total_demand_ls'].min():.0f} to {sw['total_demand_ls'].max():.0f} L/s",
              f"  Pearson correlation of sigma_c with demand: {np.corrcoef(sw['total_demand_ls'], sw['sigma_c'])[0,1]:.3f}"]
    out = ROOT / "output"; out.mkdir(exist_ok=True)
    (out / "controllability_margins_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"runtime {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
