#!/usr/bin/env python3
"""Redraw the case study figures of the paper from the cached results in ``results/``.

No EPANET or DAE simulation is rerun. Outputs in ``figures/``:
  epanet_comparison_suite.pdf   DAE versus EPANET (Net3 1 h fixed control, Net3 and BELL_CL1 24 h smoothed
                                control, Net3 pump trip)
  stability_and_procedures.pdf  energy decay, conductance weights, Algorithms 1 and 2 (Net3, Anytown)
  net3_hourly_study.pdf         margins of Net3 under the aggressive demand scenario
  relevance_case_studies.pdf    pump allocation on Net3 and MOD

Run:  python3 src/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "results"
FIG_DIR = REPO / "figures"
sys.path.insert(0, str(HERE))
from _replot_common import (  # noqa: E402
    BLACK, DARKBLUE, DARKRED, MAINBLUE, SOFTBLUE, SOFTGRAY, GRIDGRAY, plt, style_axes,
)
# Imported here on purpose: plot_inp_networks updates the global rcParams on import, and that
# update must happen before any rc_context below, otherwise it would shrink the fonts of one figure.
from plot_inp_networks import (  # noqa: E402
    draw_network, load_network, overview_positions, original_positions, squarify_positions,
)

NETWORKS_DIR = HERE / "Networks"

# Fonts are set for figures drawn at 7.2 in and shown at 0.65\textwidth, or 11 in
# shown at \textwidth (AGU text width 32.25 pc = 5.37 in), which gives about 8 pt.
RC = {
    "axes.labelsize": 16, "axes.titlesize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "lines.linewidth": 2.4,
}


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
# Figure 4: DAE versus EPANET (rows a to c) and the Net3 pump trip (row d)
# --------------------------------------------------------------------------- #
def epanet_row(axes, d, letter: str, row_label: str, legends: bool) -> None:
    t = d["time_h"]
    node = str(d["rep_node_id"]); link = str(d["rep_link_id"])

    ax = axes[0]
    ln1 = ax.plot(t, d["head_epn_rep"], color=BLACK, linewidth=2.5, linestyle="--", label="EPANET")
    ln2 = ax.plot(t, d["head_dae_rep"], color=MAINBLUE, linewidth=2.5, label="DAE")
    ax.set_xlabel("Time [h]", labelpad=2)
    ax.set_ylabel(f"Junc. {node} head [m]", color=MAINBLUE)
    ax.tick_params(axis="y", labelcolor=MAINBLUE)
    ax.set_xlim(t[0], t[-1])
    style_axes(ax)
    ax2 = ax.twinx()
    ax2.plot(t, d["flow_epn_rep"], color=BLACK, linewidth=2.5, linestyle=":")
    ax2.plot(t, d["flow_dae_rep"], color=DARKRED, linewidth=2.5)
    ax2.set_ylabel(f"Link {link} flow [m$^3$/s]", color=DARKRED)
    ax2.tick_params(axis="y", labelcolor=DARKRED)
    ax.set_title("Trajectories", pad=5)
    if legends:
        ax.legend(handles=ln1 + ln2, loc="center left", frameon=False, fontsize=12)

    ax = axes[1]
    ax.fill_between(t, np.maximum(d["head_relerr_q10"], 1e-12), np.maximum(d["head_relerr_q90"], 1e-12),
                    color=MAINBLUE, alpha=0.20, linewidth=0)
    ax.plot(t, d["head_relerr_med"], color=MAINBLUE, linewidth=2.5, label="Heads")
    ax.fill_between(t, np.maximum(d["flow_relerr_q10"], 1e-12), np.maximum(d["flow_relerr_q90"], 1e-12),
                    color=DARKRED, alpha=0.18, linewidth=0)
    ax.plot(t, d["flow_relerr_med"], color=DARKRED, linewidth=2.5, linestyle="--", label="Flows")
    ax.set_xlabel("Time [h]", labelpad=2)
    ax.set_ylabel("Relative $|e|$")
    ax.set_xlim(t[0], t[-1])
    ax.set_yscale("log")
    ax.set_title("Error bands", pad=5)
    if legends:
        ax.legend(frameon=False, loc="lower right", fontsize=12)
    style_axes(ax)

    ax = axes[2]
    ax.plot(t, d["total_l2_norm"], color=MAINBLUE, linewidth=2, label=r"$\Vert e\Vert_{2}/n_x$")
    ax.plot(t, d["total_linf_norm"], color=DARKRED, linewidth=2.5, linestyle="--", label=r"$\Vert e\Vert_{\infty}/n_x$")
    ax.set_xlabel("Time [h]", labelpad=2)
    ax.set_ylabel("Normalized error")
    ax.set_xlim(t[0], t[-1])
    ax.set_title("Total error", pad=5)
    if legends:
        ax.legend(frameon=False, loc="upper left", fontsize=12)
    style_axes(ax)

    ax = axes[3]
    ax.plot(t, d["total_demand_ls"], color=DARKRED, linewidth=2)
    ax.fill_between(t, 0.0, d["total_demand_ls"], color=SOFTBLUE, alpha=0.18)
    ax.set_xlabel("Time [h]", labelpad=2)
    ax.set_ylabel("Demand [L/s]")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_title("Total network demand", pad=5)
    style_axes(ax)


def pump_trip_row(axes, d) -> None:
    """Fourth row: fast trip of a Net3 pump, DAE (fine step) versus EPANET (300 s step)."""
    t0 = float(d["trip_time_s"]); tau_s = float(d["smoothing_s"])
    tm = (d["time_s"] - t0) / 60.0; te = (d["epanet_time_s"] - t0) / 60.0
    win = (-5.0, 20.0)
    pump = str(d["tripped_pump"]); trunk = str(d["trunk_link"]); junc = str(d["junction"])

    ax = axes[0]
    ax.plot(tm, 1e3 * d["dae_pump_flow"], color=MAINBLUE, linewidth=2.5, label=f"Pump {pump}, DAE")
    ax.step(te, 1e3 * d["epanet_pump_flow"], where="post", color=BLACK, linewidth=2.0, linestyle="--", label=f"Pump {pump}, EPANET")
    ax.plot(tm, 1e3 * d["dae_trunk_flow"], color=DARKRED, linewidth=2.5, label=f"Pipe {trunk}, DAE")
    ax.step(te, 1e3 * d["epanet_trunk_flow"], where="post", color=BLACK, linewidth=2.0, linestyle=":", label=f"Pipe {trunk}, EPANET")
    ax.set_xlim(*win); ax.set_xlabel("Time after trip [min]", labelpad=2); ax.set_ylabel("Flow [L/s]")
    ax.set_title("Flows", pad=5); ax.legend(frameon=False, loc="upper right", fontsize=10.5, handlelength=1.8); style_axes(ax)

    ax = axes[1]
    ax.plot(tm, d["dae_junction_head"], color=MAINBLUE, linewidth=2.5, label="DAE")
    ax.step(te, d["epanet_junction_head"], where="post", color=BLACK, linewidth=2.0, linestyle="--", label="EPANET")
    ax.set_xlim(*win); ax.set_xlabel("Time after trip [min]", labelpad=2); ax.set_ylabel(f"Junc. {junc} head [m]")
    ax.set_title("Junction head", pad=5); ax.legend(frameon=False, loc="lower right", fontsize=12); style_axes(ax)

    ax = axes[2]
    tr = d["t_rel_s"]; r = d["recovery_norm"]; m = (tr >= 0) & (tr <= 600.0)
    lo = float(d["tau_fastest_inertial_s"]); hi = float(d["tau_slowest_inertial_s"]); tfit = 1.0 / float(d["fit_rate"])
    # Shaded band, curve, and arrow are described in the figure caption (no legend, to keep the row compact).
    ax.axvspan(lo, hi, color=SOFTBLUE, alpha=0.9, lw=0)
    ax.semilogy(tr[m], np.maximum(r[m], 1e-6), color=MAINBLUE, linewidth=2.5)
    ax.annotate("", xy=(300.0, 12.0), xytext=(0.0, 12.0), arrowprops=dict(arrowstyle="<->", color=DARKRED, lw=1.4))
    ax.text(150.0, 17.0, "EPANET step", ha="center", va="bottom", fontsize=11, color=DARKRED)
    ax.annotate(f"decay time {tfit:.0f} s", xy=(tfit, np.exp(-1.0)), xytext=(250, 0.4), fontsize=11, color=BLACK,
                arrowprops=dict(arrowstyle="-", color=BLACK, lw=0.7))
    ax.set_xlim(0.0, 600.0); ax.set_ylim(1e-3, 80.0)
    ax.set_xlabel("Time after trip [s]", labelpad=2); ax.set_ylabel("Normalized deviation")
    ax.set_title("Recovery", pad=5); style_axes(ax)

    ax = axes[3]
    ids = [str(v) for v in d["live_link_ids"]]; dq = 1e3 * np.asarray(d["live_dq_m3s"], dtype=float)
    te = np.asarray(d["live_t_efold_s"], dtype=float)
    ok = np.isfinite(te) & (dq > 5.0)
    ax.axhspan(lo, hi, color=SOFTBLUE, alpha=0.9, lw=0)
    ax.scatter(dq[ok], te[ok], s=16, color=SOFTGRAY, edgecolor="none", zorder=3)
    top = [str(v) for v in d["top_link_ids"]]
    sel = np.array([i for i, lid in enumerate(ids) if lid in top and ok[i]])
    ax.scatter(dq[sel], te[sel], s=36, color=DARKRED, edgecolor=BLACK, linewidth=0.4, zorder=4)
    ax.axhline(300.0, color=DARKRED, linestyle="--", linewidth=1.8)
    ax.text(0.97, 300.0 * 1.25, "EPANET step", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=11, color=DARKRED)
    ax.set_yscale("log"); ax.set_ylim(5.0, 1500.0)
    ax.set_xlabel("Flow change [L/s]", labelpad=2); ax.set_ylabel("Settling time [s]")
    ax.set_title("Pipe settling", pad=5); style_axes(ax)


def fig_epanet_suite() -> None:
    rows = [
        (OUT / "fixed_control" / "net3_one_step_epanet_comparison.npz", "a", "Net3, 1 h fixed control", True),
        (OUT / "smoothed_control" / "net3_multi_step_epanet_comparison.npz", "b", "Net3, 24 h smoothed control", False),
        (OUT / "smoothed_control" / "bell_cl1_multi_step_epanet_comparison.npz", "c", "BELL CL1, 24 h smoothed control", False),
    ]
    trip = np.load(OUT / "pump_trip" / "net3_pump_trip.npz", allow_pickle=True)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(4, 4, figsize=(15.0, 12.6))
        fig.subplots_adjust(left=0.06, right=0.965, bottom=0.05, top=0.96, wspace=0.75, hspace=0.95)
        heads = []
        for (path, letter, label, legends), ax_row in zip(rows, axes[:3]):
            epanet_row(ax_row, np.load(path, allow_pickle=True), letter, label, legends)
            heads.append((ax_row, f"({letter}) {label}"))
        pump_trip_row(axes[3], trip)
        heads.append((axes[3], f"(d) Net3, trip of pump {str(trip['tripped_pump'])} with a {float(trip['smoothing_s']):.0f} s smoothing width"))
        for ax_row, text in heads:
            top = max(a.get_position().y1 for a in ax_row)
            fig.text(0.5, top + 0.04, text, ha="center", va="bottom", fontsize=19, color=DARKBLUE, fontweight="semibold")
        save(fig, "epanet_comparison_suite.pdf")


# --------------------------------------------------------------------------- #
# Figure 5: energy decay (Theorem 3) and Algorithms 1 and 2
# --------------------------------------------------------------------------- #
def admittance_panel(ax, w, title, bar_width):
    sw = np.sort(w)[::-1]; n = sw.size; ranks = np.arange(1, n + 1)
    ax.bar(ranks, sw, color=MAINBLUE, edgecolor=DARKBLUE, linewidth=0.4, width=bar_width)
    ax.set_yscale("log")
    step = 1 if n <= 12 else 10 if n <= 60 else max(20, int(round(n / 6.0 / 10.0)) * 10)
    xt = np.arange(0, n + 1, step); xt[0] = 1
    ax.set_xticks(xt); ax.set_xlim(0.5, n + 0.5)
    ax.set_xlabel("Link rank"); ax.set_ylabel(r"$w_e = \kappa_e^{-1}$")
    ax.set_title(title, pad=5); style_axes(ax)


def energy_panel(ax, t, v, title):
    ax.semilogy(t, v / max(float(v[0]), 1e-300), color=MAINBLUE, linewidth=2.5)
    ax.set_xlabel("Time [h]"); ax.set_ylabel(r"$V(\Delta\mathbf{x}_h)/V(0)$")
    ax.set_title(title, pad=5); ax.set_xlim(t[0], t[-1]); style_axes(ax)


def fig_stability_and_procedures() -> None:
    t5 = {s: np.load(OUT / "laplacian_stability" / f"laplacian_stability_{s.lower()}.npz", allow_pickle=True) for s in ("Net3", "Anytown")}
    t6 = {s: np.load(OUT / "linear_dae" / f"linear_dae_{s.lower()}.npz", allow_pickle=True) for s in ("Net3", "Anytown")}
    colors = {"Net3": MAINBLUE, "Anytown": DARKRED}; markers = {"Net3": "o", "Anytown": "s"}
    with plt.rc_context(RC):
        # Two rows by three columns (2026-10-21): row 1 Net3 weights, Net3 energy, Algorithm 1;
        # row 2 Anytown weights, Anytown energy, Algorithm 2.
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.6))
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.94, wspace=0.42, hspace=0.62)
        admittance_panel(axes[0, 0], t5["Net3"]["link_weights"], "(a) Net3", 8.5)
        energy_panel(axes[0, 1], t5["Net3"]["t_h"], t5["Net3"]["v_hist"], "(b) Net3")
        admittance_panel(axes[1, 0], t5["Anytown"]["link_weights"], "(d) Anytown", 3.0)
        energy_panel(axes[1, 1], t5["Anytown"]["t_h"], t5["Anytown"]["v_hist"], "(e) Anytown")

        ax = axes[0, 2]
        for s, d in t6.items():
            ax.loglog(d["delta"], d["residual_norm"], marker=markers[s], color=colors[s], linewidth=2)
            ax.loglog(d["delta"], d["true_shift_norm"], marker=markers[s], color=colors[s], linewidth=1.0, linestyle=":", alpha=0.7)
        ax.set_xlabel("Roughness perturbation"); ax.set_ylabel(r"$\|\Delta \mathbf{A}_h\|_2$")
        ax.set_title("(c) Algorithm 1", pad=5); style_axes(ax)
        ax.set_ylim(top=ax.get_ylim()[1] * 400.0)
        ax.legend(handles=[
            Line2D([], [], color=BLACK, marker="o", linestyle="-", linewidth=2.0, label="Residual"),
            Line2D([], [], color=BLACK, linestyle=":", linewidth=2.0, label="True shift"),
            Line2D([], [], color=MAINBLUE, linewidth=2.0, label="Net3"),
            Line2D([], [], color=DARKRED, linewidth=2.0, label="Anytown"),
        ], frameon=False, loc="upper left", fontsize=11, handlelength=1.8, ncol=2, columnspacing=0.8)

        ax = axes[1, 2]
        K = 6; x = np.arange(K); bw = 0.4
        for k, (s, d) in enumerate(t6.items()):
            ax.bar(x + (k - 0.5) * bw, d["top_sensitivity_norm"][:K], width=bw, color=colors[s],
                   edgecolor=BLACK, linewidth=0.5, label=s)
        ids = [[str(v) for v in d["top_pipe_ids"][:K]] for d in t6.values()]
        ax.set_xticks(x); ax.set_xticklabels([f"{a}/{b}" for a, b in zip(*ids)], rotation=25, ha="right", fontsize=10)
        ax.set_xlabel(f"Top {K} pipe IDs"); ax.set_ylabel(r"$\|\partial \mathbf{A}_h/\partial \theta_i\|_2$")
        ax.set_title("(f) Algorithm 2", pad=5)
        ax.legend(frameon=False, loc="upper right", fontsize=12, handlelength=1.8); style_axes(ax)
        save(fig, "stability_and_procedures.pdf")


# --------------------------------------------------------------------------- #
# Figure 6: margins of Net3 under the aggressive demand scenario
# --------------------------------------------------------------------------- #
def fig_net3_hourly_study() -> None:
    d = np.load(OUT / "operating_point_margins" / "net3_hourly_operating_margins.npz", allow_pickle=True)
    t = d["time_h"]; demand = d["total_demand_ls"]
    with plt.rc_context(RC):
        from matplotlib.ticker import ScalarFormatter
        def power_on_top(ax):
            fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(fmt); ax.yaxis.get_offset_text().set_fontsize(12)
        # Two rows by three columns (2026-10-21): demand, stability margin, PBH margin; slopes, influence, influence vs demand
        fig = plt.figure(figsize=(11.0, 5.6))
        gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.15))
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.94, wspace=0.42, hspace=0.62)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(t, demand, color=DARKRED, linewidth=2.2)
        ax.fill_between(t, 0.0, demand, color=SOFTBLUE, alpha=0.22, linewidth=0)
        ax.set_xlabel("Time [h]"); ax.set_ylabel("Demand [L/s]"); ax.set_title("(a) Demand scenario", pad=5)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(bottom=0.0); style_axes(ax)

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(t, d["stability_margin"], color=MAINBLUE, marker="o", markersize=3.6)
        ax.set_xlabel("Time [h]"); ax.set_ylabel(r"$\alpha_s$ [1/s]"); ax.set_title("(b) Stability margin", pad=5)
        ax.set_xlim(t[0], t[-1]); power_on_top(ax); style_axes(ax)

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(t, d["pbh_margin"], color=DARKBLUE, marker="s", markersize=3.6, label=r"$\sigma_{\mathrm{PBH}}$")
        ax.set_xlabel("Time [h]"); ax.set_ylabel(r"$\sigma_{\mathrm{PBH}}$", color=DARKBLUE); ax.tick_params(axis="y", labelcolor=DARKBLUE)
        ax.set_title("(c) Controllability margins", pad=5)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(bottom=0.0); power_on_top(ax); style_axes(ax)
        sw_path = OUT / "controllability_margins" / "net3_demand_sigma_c.npz"
        if sw_path.exists():
            sw = np.load(sw_path, allow_pickle=True)
            ax2 = ax.twinx()
            ax2.semilogy(sw["time_h"], sw["sigma_c"], color=DARKRED, marker="o", markersize=3.2, linestyle="--")
            ax2.set_ylabel(r"$\sigma_c$", color=DARKRED); ax2.tick_params(axis="y", labelcolor=DARKRED)

        ax = fig.add_subplot(gs[1, 0])
        kmin = np.maximum(d["kappa_min"], 1e-16); kmax = np.maximum(d["kappa_max"], 1e-16)
        ax.semilogy(t, kmax, color=DARKRED, label=r"$\max\,\kappa_e$")
        ax.semilogy(t, kmin, color=MAINBLUE, linestyle="--", label=r"$\min\,\kappa_e$")
        ax.fill_between(t, kmin, kmax, color=SOFTBLUE, alpha=0.20)
        ax.set_xlabel("Time [h]"); ax.set_ylabel(r"Slope $\kappa_e$"); ax.set_title("(d) Link slopes", pad=5)
        ax.set_xlim(t[0], t[-1]); ax.legend(frameon=False, loc="center right"); style_axes(ax)

        gain = d["authority_gain"]; avg = d["authority_average_gain"]
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(t, gain, color=MAINBLUE, marker="o", markersize=3.6, label=r"$G_H$")
        ax.plot(t, avg, color=DARKRED, marker="s", markersize=3.2, linestyle="--", label="Average")
        ax.set_xlabel("Time [h]"); ax.set_ylabel("Normalized gain"); ax.set_title("(e) Pump influence", pad=5)
        ax.set_xlim(t[0], t[-1]); ax.legend(frameon=False, loc="center right", handlelength=1.6); style_axes(ax)

        ax = fig.add_subplot(gs[1, 2])
        pts = ax.scatter(demand, gain, c=t, cmap="viridis", s=42, edgecolor=BLACK, linewidth=0.35, zorder=3)
        order = np.argsort(demand); trend = np.poly1d(np.polyfit(demand, gain, deg=1))
        ax.plot(demand[order], trend(demand[order]), color=DARKRED, linewidth=2.0, label="Trend")
        ax.set_xlabel("Demand [L/s]"); ax.set_ylabel(r"$G_H$"); ax.set_title("(f) Influence vs demand", pad=5)
        ax.legend(frameon=False, loc="upper right", handlelength=1.8)
        cbar = fig.colorbar(pts, ax=ax, pad=0.02, fraction=0.05); cbar.set_label("Time [h]")
        style_axes(ax)
        save(fig, "net3_hourly_study.pdf")


# --------------------------------------------------------------------------- #
# Figure 7: pump allocation on Net3 and MOD
# --------------------------------------------------------------------------- #
def overview_network(ax, stem: str):
    """Draw the network exactly as in the all-networks overview figure and return (wn, positions)."""
    wn = load_network(NETWORKS_DIR / f"{stem}.inp")
    if stem in {"Net1", "anytown"}:
        pos = squarify_positions(original_positions(wn))
    else:
        pos = overview_positions(wn)
    draw_network(ax, wn, annotate=False, legend=False, compact=True, overview=True, positions=pos)
    return wn, pos


def link_segment(wn, pos, link_id: str):
    link = wn.get_link(link_id)
    return np.asarray(pos[link.start_node_name]), np.asarray(pos[link.end_node_name])


def highlight_links(ax, wn, pos, ids, color, width=3.6, zorder=9):
    for lid in ids:
        if lid not in wn.link_name_list:
            continue
        p0, p1 = link_segment(wn, pos, lid)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=width, solid_capstyle="round", zorder=zorder)


K_SHOW = 6   # added pumps shown in the study (the data files hold up to 8)


def own_set_panel(ax, d8, title, i0, xlabel=r"Added pumps"):
    """Cumulative gain of each selection (pump_allocation_net3.npz relative to the existing pumps,
    pump_allocation_mod_anchor.npz relative to pipe 335), drawn from two added pumps on. i0 is the index of the
    two pump point in the stored curves (1 on Net3, 0 on MOD, where pipe 335 is the first pump)."""
    ks = np.arange(2, K_SHOW + 1); r = len(ks); b = float(d8["base_logdet"])
    rel = lambda k: np.asarray(d8[k], dtype=float)[i0:i0 + r] - b
    # the band is the 10 to 90 percentile range of the 500 randomly drawn sets
    ax.fill_between(ks, rel("curve_random_lo"), rel("curve_random_hi"), color=SOFTBLUE, alpha=0.85, lw=0, label="Randomly drawn sets")
    ax.plot(ks, rel("curve_random_mean"), color=SOFTGRAY, ls=":", label="Randomly drawn sets, mean")
    ax.plot(ks, rel("curve_betweenness"), color=SOFTGRAY, ls="--", marker="^", markersize=3.6, label="Betweenness rank")
    ax.plot(ks, rel("curve_greedy"), color=DARKRED, marker="o", markersize=3.6, label="Greedy")
    ax.set_xlabel(xlabel); ax.set_ylabel("Controllability gain")
    ax.set_xticks(ks); ax.set_xlim(ks[0] - 0.3, ks[-1] + 0.3); ax.set_title(title, pad=5)
    style_axes(ax)
    return ax.get_legend_handles_labels()


def map_panel(ax, fig, stem, d8, title, legend_extra):
    """Network with the selected links in red; the pipe identifiers are listed in selection order in a
    vertical box to the right of the network instead of being written on the map."""
    wn, pos = overview_network(ax, stem)
    # keep the axes box at the full cell height so the panel title aligns with the other panels;
    # the equal aspect is kept by expanding the data limits instead of shrinking the box
    ax.set_aspect("equal", adjustable="datalim")
    ids = [str(v) for v in d8["greedy_ids"]][:K_SHOW]
    highlight_links(ax, wn, pos, ids, DARKRED, width=3.0)
    ax.set_title(title, pad=5)
    handles = [Line2D([], [], color=DARKRED, lw=2.4, label="Selected link")] + legend_extra + [
        Line2D([], [], color=BLACK, marker="D", linestyle="None", markersize=6, label="Reservoir"),
        Line2D([], [], color=BLACK, marker="s", markerfacecolor=SOFTBLUE, linestyle="None", markersize=7, label="Tank")]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=15, handlelength=1.6, columnspacing=0.8)
    listing = "Selection order\n" + "\n".join(f"{k}: pipe {lid}" for k, lid in enumerate(ids, start=1))
    xs = [float(p[0]) for p in pos.values()]; ys = [float(p[1]) for p in pos.values()]
    x_right = max(xs) + 0.08 * max(max(xs) - min(xs), max(ys) - min(ys)); y_mid = 0.5 * (max(ys) + min(ys))
    ax.text(x_right, y_mid, listing, fontsize=14.5, color=DARKRED, ha="left", va="center", linespacing=1.35, clip_on=False)


def fig_relevance() -> None:
    """Pump allocation on Net3 (existing pumps kept) and on MOD, the largest benchmark (no existing pump)."""
    d_net3 = np.load(OUT / "pump_allocation" / "pump_allocation_net3.npz", allow_pickle=True)
    d_mod = np.load(OUT / "pump_allocation" / "pump_allocation_mod.npz", allow_pickle=True)
    # MOD gains are measured relative to the best single location (pipe 335), which every selection keeps
    d_mod_anchor = np.load(OUT / "pump_allocation" / "pump_allocation_mod_anchor.npz", allow_pickle=True)
    rc = dict(RC); rc.update({"legend.fontsize": 11, "axes.labelsize": 19, "axes.titlesize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17})
    with plt.rc_context(rc):
        # One row of four panels at full text width, same size and scale as a row of the EPANET comparison suite.
        fig = plt.figure(figsize=(15.0, 5.2))
        # two groups (Net3, MOD) with a small gap inside each group and a wide gap between groups;
        # the bottom margin holds the shared legend of panels (a) and (c)
        outer = fig.add_gridspec(1, 2, wspace=0.30, left=0.05, right=0.985, bottom=0.33, top=0.92)
        g1 = outer[0].subgridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.12)
        g2 = outer[1].subgridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.12)
        handles, labels = own_set_panel(fig.add_subplot(g1[0, 0]), d_net3, "(a) Net3", i0=1)
        pump_handle = [Line2D([], [], color=DARKBLUE, marker="o", markerfacecolor="white", markeredgecolor=DARKBLUE, markeredgewidth=1.3, markersize=7, lw=1.8, label="Existing pump")]
        map_panel(fig.add_subplot(g1[0, 1]), fig, "Net3", d_net3, "(b) Net3, selected links", pump_handle)
        own_set_panel(fig.add_subplot(g2[0, 0]), d_mod_anchor, "(c) MOD", i0=0)
        map_panel(fig.add_subplot(g2[0, 1]), fig, "MOD", d_mod, "(d) MOD, selected links", [])
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=18, handlelength=2.0, columnspacing=1.8, handletextpad=0.6)
        save(fig, "relevance_case_studies.pdf")


def main() -> None:
    print("Figures 4 to 7 of the paper:")
    fig_epanet_suite()
    fig_stability_and_procedures()
    fig_net3_hourly_study()
    fig_relevance()


if __name__ == "__main__":
    main()
