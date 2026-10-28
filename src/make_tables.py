#!/usr/bin/env python3
"""Collect the values of Tables 1 and 2 of the paper from the cached results in ``results/``.

No simulation is rerun. Outputs in ``results/tables/``:
  table1_epanet_dae_errors.csv  maximum junction head and link flow errors of the DAE versus EPANET
  table2_margins.csv            stability, linearization, and controllability quantities (Net3, Anytown)

Run:  python3 src/make_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = RES / "tables"

TABLE1_NETWORKS = [("Threenodes", "threenodes"), ("Net1", "net1"), ("Net3", "net3"),
                   ("Anytown", "anytown"), ("MOD", "mod"), ("BELL_CL1", "bell_cl1")]
TABLE2_NETWORKS = [("Net3", "net3"), ("Anytown", "anytown")]


def read_max_errors(path: Path) -> tuple[float, float]:
    if not path.exists():
        return float("nan"), float("nan")
    head, flow = np.loadtxt(path, delimiter=",", skiprows=1)
    return float(head), float(flow)


def table1() -> None:
    rows = ["network,fixed_1h_head_m,fixed_1h_flow_m3s,smoothed_24h_head_m,smoothed_24h_flow_m3s"]
    print("Table 1: maximum errors of the DAE versus EPANET")
    print(f"  {'network':<11} {'1 h head':>10} {'1 h flow':>10} {'24 h head':>10} {'24 h flow':>10}")
    for name, key in TABLE1_NETWORKS:
        h1, q1 = read_max_errors(RES / "fixed_control" / f"{key}_one_step_max_errors.csv")
        h24, q24 = read_max_errors(RES / "smoothed_control" / f"{key}_multi_step_max_errors.csv")
        rows.append(f"{name},{h1:.3e},{q1:.3e},{h24:.3e},{q24:.3e}")
        print(f"  {name:<11} {h1:10.2e} {q1:10.2e} {h24:10.2e} {q24:10.2e}")
    (OUT / "table1_epanet_dae_errors.csv").write_text("\n".join(rows) + "\n")


def table2() -> None:
    cols = {}
    for name, key in TABLE2_NETWORKS:
        m = np.load(RES / "controllability_margins" / f"{key}_margins.npz", allow_pickle=True)
        w = np.load(RES / "laplacian_stability" / f"laplacian_stability_{key}.npz")["link_weights"]
        lin = np.load(RES / "linear_dae" / f"linear_dae_{key}.npz", allow_pickle=True)
        kappa_e = 1.0 / np.asarray(w, dtype=float)          # link weights are w_e = 1 / kappa_e
        cols[name] = {
            "alpha_s_per_s": f"{float(m['alpha_s']):.3e}",
            "kappa_V": f"{float(m['kappa_V']):.3g}",
            "kappa_e_min_s_per_m2": f"{kappa_e.min():.3e}",
            "kappa_e_max_s_per_m2": f"{kappa_e.max():.3e}",
            "largest_relative_residual_10pct": f"{float(lin['relative_residual'][np.argmax(lin['delta'])]):.3g}",
            "top_pipes_gA": " ".join(str(p) for p in m["top_A_ids"][:3]),
            "pbh_margin": f"{float(m['pbh']):.3e}",
            "sigma_c": f"{float(m['sigma_c']):.3e}",
            "sigma_max_K_N": f"{float(m['sigma_max']):.3g}" if "sigma_max" in m.files else "nan",
            "top_pipes_gK": " ".join(str(p) for p in m["top_K_ids"][:3]),
            "equilibrium_residual_inf": f"{float(m['eq_residual']):.3e}",
        }
    names = [n for n, _ in TABLE2_NETWORKS]
    rows = ["quantity," + ",".join(names)]
    print("Table 2: stability, linearization, and controllability quantities")
    for q in cols[names[0]]:
        rows.append(q + "," + ",".join(cols[n][q] for n in names))
        print(f"  {q:<34} " + "  ".join(f"{cols[n][q]:>14}" for n in names))
    (OUT / "table2_margins.csv").write_text("\n".join(rows) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table1()
    table2()
    print(f"Saved tables to {OUT}")


if __name__ == "__main__":
    main()
