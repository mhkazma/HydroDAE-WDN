#!/usr/bin/env python3
"""Pump allocation on MOD relative to its best single location (Section 4.6 of the paper).

MOD has no pump or valve. The best single location for an added pump is pipe 335 (first choice of the
greedy algorithm). To compare the selections on the same footing as on Net3, where every selection adds
pumps to the existing ones, this script places a pump on pipe 335 first and treats it as existing. Each
selection then adds its own links, and the gain is measured relative to {335}:

    gain(S) = log det(W_N({335} u S) + eps I) - log det(W_N({335}) + eps I).

Selections (same definitions as pump_allocation.py, with pipe 335 removed from the
candidates): greedy algorithm, betweenness rank, and 500 randomly drawn sets of each size.

Output: results/pump_allocation/pump_allocation_mod_anchor.npz and src/output/pump_allocation_mod_anchor.txt.

Run:  python3 src/pump_allocation_mod_anchor.py
"""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pump_allocation as t8  # noqa: E402
from paper_output import paper_subdir  # noqa: E402
from fixed_control import DEMAND_PRIMARY_AMPLITUDE  # noqa: E402
from equilibrium import prepare_fixed_setting_case, solve_dae_equilibrium  # noqa: E402
from linear_dae import graph_linear_dae  # noqa: E402

INP = "MOD.inp"
ANCHOR = "335"
ADDED = 7
DRAWS = 500
SEED = 0


def main() -> None:
    t0 = time.perf_counter()
    prepared = prepare_fixed_setting_case(ROOT / "Networks" / INP, DEMAND_PRIMARY_AMPLITUDE)
    case = prepared.case
    x_eq, _ = solve_dae_equilibrium(case, prepared.nominal_demand_cms, prepared.nominal_state)
    data = graph_linear_dae(case, x_eq)
    model = t8.discretize(data)
    R_all = t8.reachability_blocks(model, model.gamma_links)
    ia = model.link_ids.index(ANCHOR)
    cand = [j for j in range(len(model.link_ids)) if j != ia]
    ids = [model.link_ids[j] for j in cand]
    R_links = R_all[:, :, cand]                 # candidates, pipe 335 removed
    R_fixed = R_all[:, :, [ia]]                 # pipe 335 acts as the existing actuator
    m = len(cand)

    base = t8.logdet_metric(t8.gramian(R_links, R_fixed, []))
    greedy_idx, greedy_vals = t8.greedy_actuator_selection(R_links, R_fixed, ADDED)

    q_eq = x_eq[: len(case.supported_link_ids)]
    flow = {lid: abs(float(q_eq[case.supported_link_ids.index(lid)])) for lid in ids}
    eb = t8.edge_betweenness_by_link(case)
    betw = {lid: eb.get(lid, 0.0) for lid in ids}

    def order_by(score):
        return sorted(range(m), key=lambda j: -score[ids[j]])

    curves = {
        "greedy": greedy_vals,
        "betweenness": t8.ranked_curve(R_links, R_fixed, order_by(betw), ADDED),
    }
    rng = np.random.default_rng(SEED)
    rand = np.array([t8.ranked_curve(R_links, R_fixed, list(rng.permutation(m)), ADDED) for _ in range(DRAWS)])
    curves["random_mean"] = rand.mean(axis=0)
    curves["random_min"] = rand.min(axis=0)
    curves["random_max"] = rand.max(axis=0)
    curves["random_lo"] = np.percentile(rand, 10, axis=0)
    curves["random_hi"] = np.percentile(rand, 90, axis=0)

    out = {f"curve_{k}": np.asarray(v, dtype=float) for k, v in curves.items()}
    out.update(base_logdet=base, anchor=ANCHOR, n_candidates=m, random_draws=DRAWS,
               greedy_ids=np.array([ids[j] for j in greedy_idx]),
               greedy_flow=np.array([flow[ids[j]] for j in greedy_idx]),
               betw_rank_ids=np.array([ids[j] for j in order_by(betw)[:ADDED]]),
               n_states=data.A.shape[0])
    path = paper_subdir("pump_allocation") / "pump_allocation_mod_anchor.npz"
    np.savez_compressed(path, **out)

    g = np.asarray(greedy_vals) - base
    lines = [f"MOD pump allocation relative to the pump on pipe {ANCHOR} ({m} candidates, {DRAWS} randomly drawn sets)",
             f"greedy added links: {', '.join(out['greedy_ids'])}",
             f"betweenness rank: {', '.join(out['betw_rank_ids'])}",
             "gain relative to {335}, k = 1..%d added links:" % ADDED]
    for k in ["greedy", "betweenness", "random_mean", "random_max", "random_min"]:
        lines.append(f"  {k:16s} " + " ".join(f"{v - base:6.1f}" for v in out[f'curve_{k}']))
    lines.append("greedy marginal gains: " + " ".join(f"{v:.1f}" for v in np.diff(np.concatenate([[0.0], g]))))
    gap = 100 * (1 - (out["curve_random_mean"] - base) / g)
    lines.append("random mean below greedy (%): " + " ".join(f"{v:.0f}" for v in gap))
    lines.append(f"runtime {time.perf_counter() - t0:.0f} s")
    (ROOT / "output" / "pump_allocation_mod_anchor.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
