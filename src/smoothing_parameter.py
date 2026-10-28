#!/usr/bin/env python3
"""Figure of the quintic transition for several smoothing widths tau_s (left) and of the control rate
du/dt (right), Figure 1 of the paper. The curves follow from closed-form expressions, so no
simulation is needed. Writes figures/smoothing_parameter.pdf.

Run:  python3 src/smoothing_parameter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from _replot_common import (  # noqa: E402
    BLACK, DARKBLUE, GRIDGRAY, MAINBLUE, SOFTBLUE, SOFTGRAY,
    plt, style_axes,
)


def quintic(alpha: np.ndarray) -> np.ndarray:
    return 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5


def quintic_deriv(alpha: np.ndarray) -> np.ndarray:
    return 30 * alpha**2 - 60 * alpha**3 + 30 * alpha**4


def generate_figure() -> Path:
    tk = 0.0
    u_lo, u_hi = 0.0, 1.0

    tau_values = [0.5, 0.2, 0.1, 0.05, 0.02]
    colors = [SOFTGRAY, SOFTBLUE, MAINBLUE, DARKBLUE, BLACK]
    linestyles = ["--", "-.", "-", "-", "-"]

    rc = {
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 2.4))
        fig.subplots_adjust(
            left=0.05,
            right=0.99,
            bottom=0.20,
            top=0.82,
            wspace=0.16,
        )

        # Use actual time t on both panels.
        xlim = (-0.07, 0.20)
        xticks = np.arange(-0.05, 0.201, 0.05)

        # left: control trajectories
        ax = axes[0]

        t_full = np.linspace(-0.3, 0.8, 800)
        u_hard = np.where(t_full < tk, u_lo, u_hi)

        ax.plot(
            t_full,
            u_hard,
            color="#c03030",
            linestyle="--",
            linewidth=1.6,
            label="step",
            zorder=5,
        )

        for tau_s, color, ls in zip(tau_values, colors, linestyles):
            tend = tk + tau_s

            u_smooth = np.full_like(t_full, u_lo)
            mask = (t_full >= tk) & (t_full <= tend)

            alpha = (t_full[mask] - tk) / tau_s
            u_smooth[mask] = u_lo + (u_hi - u_lo) * quintic(alpha)
            u_smooth[t_full > tend] = u_hi

            ax.plot(
                t_full,
                u_smooth,
                color=color,
                linestyle=ls,
                linewidth=4,
                label=rf"$\tau_s={tau_s:.2f}$",
                zorder=3,
            )

        ax.axvline(
            tk,
            color=GRIDGRAY,
            linewidth=0.2,
            linestyle=":",
            zorder=1,
        )

        ax.set_ylabel(r"$u(t)$")
        ax.set_xlabel(r"Time $t$")
        ax.set_title("Quintic smoothstep for varying smoothing widths")

        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(-0.05, 1.15)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.025, 1.05),
            frameon=False,
            fontsize=15,
            handlelength=2.0,
            labelspacing=0.1,
            borderpad=0.35,
            ncol=1,
        )

        style_axes(ax)
        ax.grid(True, color=GRIDGRAY, linewidth=0.2)

        # right: control rate
        ax = axes[1]

        t_rate = np.linspace(xlim[0], xlim[1], 900)

        for tau_s, color, ls in zip(tau_values, colors, linestyles):
            tend = tk + tau_s

            du_dt = np.zeros_like(t_rate)

            mask = (t_rate >= tk) & (t_rate <= tend)
            alpha = (t_rate[mask] - tk) / tau_s

            du_dt[mask] = (u_hi - u_lo) / tau_s * quintic_deriv(alpha)

            ax.plot(
                t_rate,
                du_dt,
                color=color,
                linestyle=ls,
                linewidth=4,
                label=rf"$\tau_s={tau_s:.2f}$",
                zorder=3,
            )

        ax.axvline(
            tk,
            color=GRIDGRAY,
            linewidth=0.6,
            linestyle=":",
            zorder=1,
        )

        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\dfrac{du}{dt}$")
        ax.set_title("Control rate during the smoothing window")

        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1, 1.05),
            frameon=False,
            fontsize=16,
            handlelength=2.0,
            labelspacing=0.1,
            borderpad=0.35,
        )

        style_axes(ax)
        ax.grid(True, color=GRIDGRAY, linewidth=0.2)

        out = REPO / "figures" / "smoothing_parameter.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved figure to {out}")
    return out


if __name__ == "__main__":
    generate_figure()
