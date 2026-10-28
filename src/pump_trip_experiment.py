#!/usr/bin/env python3
"""Fast pump trip on Net3: continuous time DAE versus EPANET (Section 4.2 of the paper).

The Lake pump (pump 10) of Net3 is tripped within a few seconds while demands are constant,
which is faster than the inertial time constants tau_e = 1/(gamma_e kappa_e) of the links.
EPANET moves to the new equilibrium inside one hydraulic step (300 s). The DAE resolves the transient, and the
recovery rate of the flows and junction heads is compared with what the linearized DAE
predicts at the post trip operating point:
  * the inertial time constants tau_e of the links whose flow changes the most,
  * the slowest inertial eigenvalue of the finite spectrum of (E_h, A_h),
  * the stability margin alpha_s before and after the trip (the tank modes),
  * the finite horizon pump influence G_H before and after the trip (pump reach).

The study uses a copy of Net3.inp with all [CONTROLS] removed, pump 10 initially open, and one time
control that closes pump 10 at TRIP_CLOCK. The window starts at WINDOW_START_S (one
pattern hour) so demands are constant over the window. The DAE uses the smoothed control
of Section 2.2.2 with a short smoothing width TRIP_SMOOTHING_S, a fine step after the trip,
and the EPANET state at the window start as initial condition.

Outputs: src/output/net3_pump_trip_summary.txt and results/pump_trip/net3_pump_trip.npz
(the arrays plotted in row (d) of Figure 4 by src/make_figures.py).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from dae_network import (  # noqa: E402
    HOUR_TO_S, build_network_case, junction_demands, load_epanet_initial_condition, unpack_state,
)
from smoothed_control import (  # noqa: E402
    SmoothControlReference, build_smooth_control_reference, load_epanet_resolved_timeseries,
    slice_epanet_series, simulate_smoothed, default_setting_model,
)
from equilibrium import equilibrium_residual  # noqa: E402
from linear_dae import graph_linear_dae, finite_eigenvalues, stability_margin  # noqa: E402
from paper_output import paper_subdir, save_npz_and_csv  # noqa: E402

INP_NAME = "Net3.inp"
TRIPPED_PUMP = "10"              # Lake pump; its discharge trunk is pipe 101 (14 200 ft, 18 in)
TRIP_CLOCK = "1:10"              # EPANET time control, hours:minutes after the simulation start
WINDOW_START_S = 1.0 * HOUR_TO_S # one pattern hour, so the demand is constant on the window
WINDOW_END_S = 2.0 * HOUR_TO_S
TRIP_TIME_S = 70.0 * 60.0        # 1:10
TRIP_SMOOTHING_S = 10.0          # smoothing width tau_s of the trip in the DAE
EPANET_STEP_S = 300.0
DAE_STEP_BEFORE_S = 60.0
DAE_STEP_FINE_S = 2.0            # for FINE_WINDOW_S after the trip
FINE_WINDOW_S = 900.0
DAE_STEP_AFTER_S = 10.0
SLOW_MODE_THRESHOLD = 1.0e-4     # |Re lambda| below this is a tank (storage) mode
TOP_LINKS = 8


def tripped_inp(src: Path, dst_dir: Path) -> Path:
    """Copy of the INP without [CONTROLS], pump open at start, one trip control."""
    text = src.read_text()
    lines = text.splitlines()
    out, section = [], None
    for line in lines:
        s = line.strip()
        if s.startswith("["):
            section = s.upper()
            if section == "[CONTROLS]":
                out.append("[CONTROLS]")
                out.append(f"LINK {TRIPPED_PUMP} CLOSED AT TIME {TRIP_CLOCK}")
                out.append("")
                continue
        if section == "[CONTROLS]" and not s.startswith("["):
            continue
        if section == "[STATUS]" and s.split() and s.split()[0] == TRIPPED_PUMP:
            out.append(f" {TRIPPED_PUMP}\tOpen")
            continue
        out.append(line)
    dst = dst_dir / f"{src.stem}_trip_{TRIPPED_PUMP}.inp"
    dst.write_text("\n".join(out) + "\n")
    return dst


def trip_control_reference(case, epanet_window) -> SmoothControlReference:
    """Constant controls from the window start, except the tripped pump which goes from open
    to closed over [TRIP_TIME_S, TRIP_TIME_S + TRIP_SMOOTHING_S] with the quintic smoothstep."""
    base = build_smooth_control_reference(case, epanet_window)
    col = case.supported_link_ids.index(TRIPPED_PUMP)
    t = np.asarray([epanet_window.time_s[0], TRIP_TIME_S, TRIP_TIME_S + TRIP_SMOOTHING_S, epanet_window.time_s[-1]])
    def rep(arr):
        a0 = arr[0]
        return np.stack([a0, a0, a0, a0])
    open_fraction = rep(base.open_fraction)
    open_fraction[:, col] = [1.0, 1.0, 0.0, 0.0]
    return SmoothControlReference(
        time_s=t,
        open_fraction=open_fraction,
        setting_model=rep(base.setting_model),
        valve_mode_weights=rep(base.valve_mode_weights),
        valve_open_loss=rep(base.valve_open_loss),
    )


def dae_time_grid(epanet_times: np.ndarray) -> np.ndarray:
    t0, t1 = float(epanet_times[0]), float(epanet_times[-1])
    g = [np.arange(t0, TRIP_TIME_S, DAE_STEP_BEFORE_S),
         np.arange(TRIP_TIME_S, TRIP_TIME_S + FINE_WINDOW_S, DAE_STEP_FINE_S),
         np.arange(TRIP_TIME_S + FINE_WINDOW_S, t1 + 1e-9, DAE_STEP_AFTER_S),
         epanet_times]
    return np.unique(np.round(np.concatenate(g), 6))


def post_trip_snapshot(case, x_pre: np.ndarray, t_s: float):
    """Equilibrium of the post trip DAE with the tank heads frozen at their trip values."""
    from operating_point_margins import solve_consistent_snapshot  # noqa: E402
    n_links, n_j, n_t = len(case.supported_link_ids), len(case.junctions), len(case.tanks)
    tank_heads = x_pre[n_links + n_j: n_links + n_j + n_t].copy()
    demand = junction_demands(case, t_s)
    return solve_consistent_snapshot(case, demand, x_pre, tank_heads)


def pump_influence(data) -> float:
    from operating_point_margins import finite_horizon_pump_authority  # noqa: E402
    return finite_horizon_pump_authority(data)[0]


def main() -> None:
    t_start = time.perf_counter()
    src = ROOT / "Networks" / INP_NAME
    tmp = Path(tempfile.mkdtemp(prefix="pump_trip_"))
    inp = tripped_inp(src, tmp)
    case = build_network_case(inp)
    col = case.supported_link_ids.index(TRIPPED_PUMP)
    n_links, n_j, n_t = len(case.supported_link_ids), len(case.junctions), len(case.tanks)

    # EPANET with the trip
    epanet_full = load_epanet_resolved_timeseries(case, inp, WINDOW_END_S, EPANET_STEP_S)
    epanet = slice_epanet_series(epanet_full, WINDOW_START_S, WINDOW_END_S)
    control_ref = trip_control_reference(case, epanet)
    case.link_base_open[:] = control_ref.open_fraction[0]
    case.link_base_setting[:] = control_ref.setting_model[0]

    # DAE with the smoothed trip
    x0 = load_epanet_initial_condition(case, epanet)
    t_dae = dae_time_grid(epanet.time_s)
    t_sim = time.perf_counter()
    x_hist, residual_hist = simulate_smoothed(case, x0, t_dae, control_ref)
    dae_runtime = time.perf_counter() - t_sim

    # series for the figure
    link_idx = {lid: i for i, lid in enumerate(epanet.link_ids)}
    node_idx = {nid: i for i, nid in enumerate(epanet.node_ids)}
    i_pre = int(np.searchsorted(t_dae, TRIP_TIME_S)) - 1
    i_post = int(np.searchsorted(t_dae, TRIP_TIME_S + TRIP_SMOOTHING_S, side="right"))
    q_pre = x_hist[:n_links, i_pre]
    q_end = x_hist[:n_links, -1]
    dq = np.abs(q_end - q_pre)
    # pipes that still carry flow after the trip; the pump discharge pipe becomes a dead end
    # (zero flow, hence a near zero slope kappa_e) and is reported separately
    pipe_cols = [c for c in range(n_links) if case.supported_link_types[c] == "PIPE"]
    dead_cols = [c for c in pipe_cols if abs(q_end[c]) < 1e-4 and dq[c] > 1e-3]
    live_cols = [c for c in pipe_cols if c not in dead_cols]
    top_cols = sorted(live_cols, key=lambda c: -dq[c])[:TOP_LINKS]
    trunk_col = top_cols[0]
    trunk_id = case.supported_link_ids[trunk_col]
    # downstream junction of the trunk: the end node with the larger head change
    state_pre = unpack_state(case, x_hist[:, i_pre]); state_end = unpack_state(case, x_hist[:, -1])
    dpj = np.abs(state_end["p_j"] - state_pre["p_j"])
    dead_nodes = {case.supported_link_start[c] for c in dead_cols} | {case.supported_link_end[c] for c in dead_cols}
    j_candidates = [j for j in range(n_j) if case.junctions[j].node_id not in dead_nodes]
    j_col = max(j_candidates, key=lambda j: dpj[j]); j_id = case.junctions[j_col].node_id
    dead_ids = [case.supported_link_ids[c] for c in dead_cols]

    dae_pump_flow = x_hist[col]
    dae_trunk_flow = x_hist[trunk_col]
    dae_j_head = x_hist[n_links + j_col]
    epn_pump_flow = epanet.flow_m3s[:, link_idx[TRIPPED_PUMP]]
    epn_trunk_flow = epanet.flow_m3s[:, link_idx[trunk_id]]
    epn_j_head = epanet.head_m[:, node_idx[j_id]]

    # post trip operating point and linearization
    case_post = build_network_case(inp)
    case_post.link_base_open[:] = control_ref.open_fraction[-1]
    case_post.link_base_setting[:] = control_ref.setting_model[-1]
    x_eq_post, res_post = post_trip_snapshot(case_post, x_hist[:, i_pre], TRIP_TIME_S)
    data_post = graph_linear_dae(case_post, x_eq_post)
    case_pre = build_network_case(inp)
    case_pre.link_base_open[:] = control_ref.open_fraction[0]
    case_pre.link_base_setting[:] = control_ref.setting_model[0]
    x_eq_pre, res_pre = post_trip_snapshot(case_pre, x_hist[:, i_pre], TRIP_TIME_S)
    data_pre = graph_linear_dae(case_pre, x_eq_pre)

    active_post = np.flatnonzero(case_post.link_base_open > 0.5)
    gamma_post = case_post.link_inertance_gain[active_post]
    tau_post = 1.0 / np.maximum(gamma_post * data_post.K, 1e-300)           # per active link
    tau_by_col = {int(c): float(tau_post[k]) for k, c in enumerate(active_post)}
    top_ids = [case.supported_link_ids[c] for c in top_cols]
    top_tau = np.asarray([tau_by_col.get(c, np.nan) for c in top_cols])
    top_dq = dq[top_cols]

    vals_post = finite_eigenvalues(data_post.E, data_post.A)
    re_post = -np.real(vals_post)
    inertial = re_post[re_post > SLOW_MODE_THRESHOLD]
    slow = re_post[re_post <= SLOW_MODE_THRESHOLD]
    t_slowest_inertial = float(1.0 / inertial.min()) if inertial.size else float("nan")
    t_fastest_inertial = float(1.0 / inertial.max()) if inertial.size else float("nan")
    alpha_pre = stability_margin(data_pre.E, data_pre.A)
    alpha_post = stability_margin(data_post.E, data_post.A)
    gain_pre = pump_influence(data_pre)
    gain_post = pump_influence(data_post)

    # observed recovery: deviation of the DAE flows and junction heads from the quasi steady state
    # that a static solve returns with the same tank levels and demands (what EPANET computes at
    # every step). This removes the slow tank drift and leaves the inertial transient only.
    from operating_point_margins import solve_consistent_snapshot  # noqa: E402
    sel = np.r_[0:n_links, n_links:n_links + n_j]
    scale = np.concatenate([np.full(n_links, 1.0), np.full(n_j, 10.0)])   # 1 m3/s and 10 m, as in Section 4.5
    t_rel = t_dae - (TRIP_TIME_S + TRIP_SMOOTHING_S)
    r = np.full(t_dae.size, np.nan)
    dev_qs_links = np.full((n_links, t_dae.size), np.nan)   # |q_e(t) - quasi steady q_e(t)| per link
    qs_guess = x_eq_post.copy()
    demand = junction_demands(case, TRIP_TIME_S)
    for k in range(t_dae.size):
        if t_rel[k] < -120.0 or t_rel[k] > FINE_WINDOW_S:
            continue
        c = case_post if t_dae[k] >= TRIP_TIME_S + TRIP_SMOOTHING_S else case_pre
        tank_heads = x_hist[n_links + n_j: n_links + n_j + n_t, k]
        try:
            x_qs, _ = solve_consistent_snapshot(c, demand, qs_guess, tank_heads)
            qs_guess = x_qs
        except RuntimeError:
            continue
        r[k] = np.linalg.norm((x_hist[sel, k] - x_qs[sel]) / scale)
        dev_qs_links[:, k] = np.abs(x_hist[:n_links, k] - x_qs[:n_links])
    r_post = r / r[i_post]
    head_min = float(np.min(dae_j_head[i_pre:])); head_min_t = float(t_dae[i_pre + int(np.argmin(dae_j_head[i_pre:]))] - TRIP_TIME_S)
    def crossing(level):
        k = np.flatnonzero((r_post <= level) & (t_rel >= 0))
        return float(t_rel[k[0]]) if k.size else float("nan")
    t_efold, t_5pct = crossing(np.exp(-1.0)), crossing(0.05)
    # fitted decay rate (log linear fit between 80 and 5 percent of the initial deviation)
    mask = (t_rel >= 0) & (t_rel <= (t_5pct if np.isfinite(t_5pct) else FINE_WINDOW_S)) & (r_post > 0.05) & (r_post < 0.8)
    fit_rate = float(-np.polyfit(t_rel[mask], np.log(r_post[mask]), 1)[0]) if mask.sum() > 3 else float("nan")

    # EPANET at the same instants for the error after the trip
    epn_q = np.full((epanet.time_s.size, n_links), np.nan)
    for c, lid in enumerate(case.supported_link_ids):
        if lid in link_idx:
            epn_q[:, c] = epanet.flow_m3s[:, link_idx[lid]]
    k_dae = np.searchsorted(t_dae, epanet.time_s)
    flow_err = np.nanmax(np.abs(x_hist[:n_links, k_dae].T - epn_q), axis=1)

    # per pipe observed settling: time after the trip at which |q_e(t) - q_e(end of fine window)| falls to
    # 1/e and to 5 percent of its value right after the trip
    i_fin = int(np.searchsorted(t_dae, TRIP_TIME_S + FINE_WINDOW_S)) - 1
    live_t_efold, live_t_5pct = [], []
    for c in live_cols:
        dev = dev_qs_links[c, :]; d0 = dev[i_post]   # inertial part only: deviation from the quasi steady flow
        if d0 < 1e-6:
            live_t_efold.append(np.nan); live_t_5pct.append(np.nan); continue
        def cross(level):
            k = np.flatnonzero((dev <= level * d0) & (t_rel >= 0) & np.isfinite(dev))
            return float(t_rel[k[0]]) if k.size else np.nan
        live_t_efold.append(cross(np.exp(-1.0))); live_t_5pct.append(cross(0.05))
    pdir = paper_subdir("pump_trip")
    save_npz_and_csv(
        pdir / "net3_pump_trip",
        time_s=t_dae, trip_time_s=TRIP_TIME_S, smoothing_s=TRIP_SMOOTHING_S,
        tripped_pump=TRIPPED_PUMP, trunk_link=trunk_id, junction=j_id, dead_end_links=np.asarray(dead_ids),
        live_link_ids=np.asarray([case.supported_link_ids[c] for c in live_cols]),
        live_dq_m3s=np.asarray([dq[c] for c in live_cols]),
        live_tau_s=np.asarray([tau_by_col.get(c, np.nan) for c in live_cols]),
        live_q_pre_m3s=np.asarray([q_pre[c] for c in live_cols]),
        live_t_efold_s=np.asarray(live_t_efold), live_t_5pct_s=np.asarray(live_t_5pct),
        dead_end_tau_s=np.asarray([tau_by_col.get(c, np.nan) for c in dead_cols]),
        dae_pump_flow=dae_pump_flow, dae_trunk_flow=dae_trunk_flow, dae_junction_head=dae_j_head,
        epanet_time_s=epanet.time_s, epanet_pump_flow=epn_pump_flow, epanet_trunk_flow=epn_trunk_flow,
        epanet_junction_head=epn_j_head, epanet_max_flow_error=flow_err,
        recovery_norm=r_post, t_rel_s=t_rel, t_efold_s=t_efold, t_5pct_s=t_5pct, fit_rate=fit_rate,
        junction_head_min_m=head_min, junction_head_min_time_s=head_min_t,
        top_link_ids=np.asarray(top_ids), top_link_tau_s=top_tau, top_link_dq_m3s=top_dq,
        tau_slowest_inertial_s=t_slowest_inertial, tau_fastest_inertial_s=t_fastest_inertial,
        n_inertial_modes=inertial.size, n_slow_modes=slow.size,
        alpha_pre=alpha_pre, alpha_post=alpha_post, gain_pre=gain_pre, gain_post=gain_post,
        snapshot_residual_pre=res_pre, snapshot_residual_post=res_post,
        max_dae_residual=float(np.max(residual_hist)), dae_runtime_s=dae_runtime,
    )
    lines = [
        f"Net3 pump trip experiment: pump {TRIPPED_PUMP} closed at {TRIP_CLOCK} (t = {TRIP_TIME_S:.0f} s), smoothing width {TRIP_SMOOTHING_S:.0f} s",
        f"window {WINDOW_START_S/3600:.2f} to {WINDOW_END_S/3600:.2f} h, EPANET step {EPANET_STEP_S:.0f} s, DAE steps {DAE_STEP_BEFORE_S:.0f}/{DAE_STEP_FINE_S:.0f}/{DAE_STEP_AFTER_S:.0f} s, {t_dae.size} DAE samples, DAE runtime {dae_runtime:.1f} s",
        f"max DAE solver residual {np.max(residual_hist):.2e}; snapshot residuals pre {res_pre:.2e}, post {res_post:.2e}",
        f"pre trip pump flow {q_pre[col]*1e3:.1f} L/s; dead end pipes after the trip: {dead_ids} (tau_e {[round(tau_by_col.get(c, float('nan'))) for c in dead_cols]} s, zero flow so kappa_e is near zero)",
        f"most affected live pipe {trunk_id}: {q_pre[trunk_col]*1e3:.1f} -> {q_end[trunk_col]*1e3:.1f} L/s",
        f"junction {j_id} head: {state_pre['p_j'][j_col]:.2f} -> {state_end['p_j'][j_col]:.2f} m, transient minimum {head_min:.2f} m at {head_min_t:.0f} s after the trip",
        f"observed recovery of flows and junction heads toward the quasi steady state: e-fold {t_efold:.0f} s, 5 percent {t_5pct:.0f} s, fitted rate {fit_rate:.3e} 1/s (1/rate = {1/fit_rate if fit_rate>0 else float('nan'):.0f} s)",
        f"linearized post trip DAE: {inertial.size} inertial modes with time constants {t_fastest_inertial:.1f} to {t_slowest_inertial:.0f} s; {slow.size} storage modes (|Re| <= {SLOW_MODE_THRESHOLD:g})",
        f"stability margin alpha_s: pre {alpha_pre:.3e}, post {alpha_post:.3e} 1/s",
        f"finite horizon pump influence G_H: pre {gain_pre:.3f}, post {gain_post:.3f}",
        f"EPANET max flow error per sample after trip [m3/s]: {np.array2string(flow_err[np.searchsorted(epanet.time_s, TRIP_TIME_S)-1:np.searchsorted(epanet.time_s, TRIP_TIME_S)+4], precision=4)}",
        "links with the largest flow change (id, |dq| L/s, tau_e s):",
    ] + [f"  {lid:>6s}  {d*1e3:8.2f}  {tau:10.1f}" for lid, d, tau in zip(top_ids, top_dq, top_tau)]
    out = ROOT / "output"; out.mkdir(exist_ok=True)
    (out / "net3_pump_trip_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {pdir / 'net3_pump_trip.npz'}; total runtime {time.perf_counter() - t_start:.1f} s")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
