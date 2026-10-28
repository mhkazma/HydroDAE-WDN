"""Pump allocation on the linearized hydraulic DAE (Section 4.6 of the paper).

Every active link is a candidate location for an added pump, modeled as a head ``u_e`` [m] on that
link. The existing pumps, acting through their speed, are kept in every candidate set. The linearized
DAE ``E dx = A x + B u`` from ``linear_dae.graph_linear_dae`` is discretized with an implicit Euler step
of 60 s over a two hour horizon, and the finite-horizon controllability Gramian

    W_N(S) = sum_{k=0}^{N_H-1} Phi^k Gamma_S Gamma_S^T (Phi^k)^T

is built from the columns of the selected links. The greedy algorithm adds, one link at a time, the
link that most increases ``log det(W_N(S) + eps I)``. It is compared with the betweenness rank (edge
betweenness centrality of the network layout) and with randomly drawn sets of each size.

Outputs: results/pump_allocation/pump_allocation_{net3,mod}.npz and src/output/pump_allocation_{net3,mod}_summary.txt.
The gains on MOD relative to its best single location are computed by pump_allocation_mod_anchor.py.

Run:  python3 src/pump_allocation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import networkx as nx
import numpy as np
from numpy.linalg import slogdet

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper_output import paper_subdir  # noqa: E402
from fixed_control import DEMAND_PRIMARY_AMPLITUDE  # noqa: E402
from equilibrium import prepare_fixed_setting_case, solve_dae_equilibrium  # noqa: E402
from linear_dae import graph_linear_dae, LinearDAEData  # noqa: E402

# Same horizon and state scaling as operating_point_margins.py
STEP_S = 60.0               # 120 input samples over two hours
HORIZON_H = 2.0
FLOW_SCALE_M3S = 1.0
HEAD_SCALE_M = 10.0
LOGDET_REG = 1.0e-6          # epsilon in log det(W_N + epsilon I)
MAX_ACTUATORS = 8
RANDOM_DRAWS = 500
RANDOM_SEED = 0
NETWORKS = ["Net3.inp", "MOD.inp"]


@dataclass
class DiscreteModel:
    phi: np.ndarray            # n x n, scaled
    gamma_links: np.ndarray    # n x m, one column per candidate link (unit head actuation)
    gamma_pumps: np.ndarray    # n x n_pumps, existing pump speed columns (may be empty)
    link_ids: list[str]
    link_types: list[str]
    n_steps: int


def discretize(data: LinearDAEData) -> DiscreteModel:
    n = data.A.shape[0]
    m = len(data.active_link_ids)
    M = data.E - STEP_S * data.A
    phi = np.linalg.solve(M, data.E)
    # unit head actuation on each active link enters the link momentum row
    B_links = np.zeros((n, m))
    B_links[:m, :m] = np.eye(m)
    gamma_links = np.linalg.solve(M, STEP_S * B_links)
    gamma_pumps = np.linalg.solve(M, STEP_S * data.B) if data.B.size and np.any(data.B != 0.0) else np.zeros((n, 0))
    # state scaling (flows in m^3/s, heads in m)
    scale = np.concatenate([np.full(m, FLOW_SCALE_M3S), np.full(n - m, HEAD_SCALE_M)])
    S_inv = np.diag(1.0 / scale)
    S = np.diag(scale)
    return DiscreteModel(
        phi=S_inv @ phi @ S,
        gamma_links=S_inv @ gamma_links,
        gamma_pumps=S_inv @ gamma_pumps,
        link_ids=list(data.active_link_ids),
        link_types=list(data.active_link_types),
        n_steps=int(round(HORIZON_H * 3600.0 / STEP_S)),
    )


def reachability_blocks(model: DiscreteModel, gamma: np.ndarray) -> np.ndarray:
    """Return R of shape (n, n_steps, m) with R[:, k, j] = Phi^k Gamma[:, j]."""
    n, m = gamma.shape
    R = np.zeros((n, model.n_steps, m))
    block = gamma.copy()
    for k in range(model.n_steps):
        R[:, k, :] = block
        block = model.phi @ block
    return R


def gramian(R_links: np.ndarray, R_fixed: np.ndarray, selected: list[int]) -> np.ndarray:
    """Finite-horizon controllability Gramian of the fixed actuators plus the selected links."""
    n = R_links.shape[0]
    W = np.zeros((n, n))
    if R_fixed.shape[2]:
        F = R_fixed.reshape(n, -1)
        W += F @ F.T
    if selected:
        Rs = R_links[:, :, selected].reshape(n, -1)
        W += Rs @ Rs.T
    return W


def logdet_metric(W: np.ndarray) -> float:
    sign, val = slogdet(W + LOGDET_REG * np.eye(W.shape[0]))
    return float(val) if sign > 0 else -np.inf


def greedy_actuator_selection(R_links: np.ndarray, R_fixed: np.ndarray, r: int) -> tuple[list[int], list[float]]:
    """Greedy algorithm: add the link that most increases log det of the controllability Gramian, r times."""
    m = R_links.shape[2]
    selected: list[int] = []
    values: list[float] = []
    remaining = set(range(m))
    for _ in range(r):
        best_v, best_val = None, -np.inf
        for v in remaining:
            val = logdet_metric(gramian(R_links, R_fixed, selected + [v]))
            if val > best_val:
                best_val, best_v = val, v
        selected.append(best_v)
        remaining.remove(best_v)
        values.append(best_val)
    return selected, values


def ranked_curve(R_links: np.ndarray, R_fixed: np.ndarray, order: list[int], r: int) -> list[float]:
    return [logdet_metric(gramian(R_links, R_fixed, order[:k])) for k in range(1, r + 1)]


def edge_betweenness_by_link(case) -> dict[str, float]:
    graph = nx.Graph()
    for pipe in case.pipes:
        graph.add_edge(pipe.start_node, pipe.end_node)
    eb = nx.edge_betweenness_centrality(graph)
    out: dict[str, float] = {}
    for pipe in case.pipes:
        key = (pipe.start_node, pipe.end_node)
        out[pipe.link_id] = eb.get(key, eb.get((key[1], key[0]), 0.0))
    return out


def run_network(inp_name: str) -> dict:
    inp_path = ROOT / "Networks" / inp_name
    prepared = prepare_fixed_setting_case(inp_path, DEMAND_PRIMARY_AMPLITUDE)
    x_eq, residual = solve_dae_equilibrium(prepared.case, prepared.nominal_demand_cms, prepared.nominal_state)
    data = graph_linear_dae(prepared.case, x_eq)
    model = discretize(data)
    R_links = reachability_blocks(model, model.gamma_links)
    R_fixed = reachability_blocks(model, model.gamma_pumps)
    m = len(model.link_ids)
    r = min(MAX_ACTUATORS, m)

    greedy_idx, greedy_vals = greedy_actuator_selection(R_links, R_fixed, r)

    q_eq = x_eq[: len(prepared.case.supported_link_ids)]
    flow_by_link = {lid: abs(float(q_eq[prepared.case.supported_link_ids.index(lid)])) for lid in model.link_ids}
    eb = edge_betweenness_by_link(prepared.case)
    betw_by_link = {lid: eb.get(lid, 0.0) for lid in model.link_ids}
    betw_order = sorted(range(m), key=lambda j: -betw_by_link[model.link_ids[j]])

    curves = {
        "greedy": greedy_vals,
        "betweenness": ranked_curve(R_links, R_fixed, betw_order, r),
    }
    rng = np.random.default_rng(RANDOM_SEED)
    rand = np.zeros((RANDOM_DRAWS, r))
    for d in range(RANDOM_DRAWS):
        perm = list(rng.permutation(m))
        rand[d] = ranked_curve(R_links, R_fixed, perm, r)
    curves["random_mean"] = rand.mean(axis=0).tolist()
    curves["random_lo"] = np.percentile(rand, 10, axis=0).tolist()
    curves["random_hi"] = np.percentile(rand, 90, axis=0).tolist()
    curves["random_min"] = rand.min(axis=0).tolist()
    curves["random_max"] = rand.max(axis=0).tolist()

    return {
        "stem": inp_path.stem,
        "residual": residual,
        "n_states": data.A.shape[0],
        "n_candidates": m,
        "n_steps": model.n_steps,
        "existing_pumps": [lid for lid, t in zip(model.link_ids, model.link_types) if t == "PUMP"],
        "greedy_ids": [model.link_ids[j] for j in greedy_idx],
        "greedy_types": [model.link_types[j] for j in greedy_idx],
        "greedy_flow": [flow_by_link[model.link_ids[j]] for j in greedy_idx],
        "betw_rank_ids": [model.link_ids[j] for j in betw_order[:r]],
        "curves": curves,
        "base_logdet": logdet_metric(gramian(R_links, R_fixed, [])),
    }


def write_summary(res: dict, path: Path) -> None:
    c = res["curves"]
    b = res["base_logdet"]
    lines = [
        f"Pump allocation on {res['stem']}",
        f"Equilibrium residual infinity norm: {res['residual']:.3e}",
        f"States: {res['n_states']}, candidate links: {res['n_candidates']}, horizon steps: {res['n_steps']} (step {STEP_S:.0f} s, horizon {HORIZON_H:.1f} h)",
        f"Existing pumps kept in every set: {', '.join(res['existing_pumps']) if res['existing_pumps'] else 'none'}",
        f"log det with existing pumps only: {b:.3f}",
        "",
        "Greedy order (link id, type, |q| at equilibrium in L/s):",
    ]
    for k, (lid, typ, q) in enumerate(zip(res["greedy_ids"], res["greedy_types"], res["greedy_flow"])):
        lines.append(f"  {k + 1}: {lid} ({typ}), |q| = {1e3 * q:.1f}")
    lines += [
        "",
        f"Gain in log det over the existing pumps, k = 1..{len(c['greedy'])} ({RANDOM_DRAWS} randomly drawn sets):",
        f"  greedy:            {' '.join(f'{v - b:.1f}' for v in c['greedy'])}",
        f"  betweenness rank:  {' '.join(f'{v - b:.1f}' for v in c['betweenness'])}",
        f"  random mean:       {' '.join(f'{v - b:.1f}' for v in c['random_mean'])}",
        "",
        f"Top links by betweenness: {', '.join(res['betw_rank_ids'])}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = paper_subdir("pump_allocation")
    for inp in NETWORKS:
        res = run_network(inp)
        write_summary(res, output_dir / f"pump_allocation_{res['stem'].lower()}_summary.txt")
        np.savez_compressed(
            paper_dir / f"pump_allocation_{res['stem'].lower()}.npz",
            greedy_ids=np.array(res["greedy_ids"]),
            greedy_types=np.array(res["greedy_types"]),
            greedy_flow=np.asarray(res["greedy_flow"]),
            base_logdet=res["base_logdet"],
            n_states=res["n_states"],
            n_candidates=res["n_candidates"],
            random_draws=RANDOM_DRAWS,
            **{f"curve_{k}": np.asarray(v) for k, v in res["curves"].items()},
        )
        print(f"{res['stem']}: {res['n_states']} states, {res['n_candidates']} candidates, greedy -> {res['greedy_ids']}")


if __name__ == "__main__":
    main()
