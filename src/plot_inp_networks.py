#!/usr/bin/env python3
"""
Schematic overview of the benchmark networks (Figure 2 of the paper). Writes
figures/all_networks_overview.pdf.

The goal here is readability rather than geographic fidelity: the figures are
not drawn to scale. Instead, the layout preserves rough orientation from the
INP coordinates while compressing oversized gaps, stretching neighborhoods
around pumps/valves/tanks, and keeping labels sparse enough to read.

Run:  python3 src/plot_inp_networks.py
"""

from __future__ import annotations

import math
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
NETWORKS_DIR = ROOT / "Networks"
OUTPUT_DIR = REPO_ROOT / "figures"
TMP_DIR = ROOT / "output" / "_tmp"
INP_FILES: list[str] | None = None
OVERVIEW_EXCLUDED_STEMS = {"BWSN_Network_1_original", "Threenodes"}
# Networks shown in Figure 2 of the paper
OVERVIEW_INCLUDED_STEMS = ["Net1", "Net3", "anytown", "MOD", "BELL_CL1"]

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import wntr

from plot_style import BLACK, DARKBLUE, DARKRED, MAINBLUE, SOFTBLUE, SOFTGRAY


SEED = 17


def discover_inp_files() -> list[Path]:
    if INP_FILES:
        return [Path(name) if Path(name).is_absolute() else NETWORKS_DIR / name for name in INP_FILES]
    return sorted(NETWORKS_DIR.glob("*.inp")) + sorted(NETWORKS_DIR.glob("*.INP"))


def overview_inp_files(inp_paths: list[Path]) -> list[Path]:
    by_stem = {p.stem: p for p in inp_paths}
    selected = [by_stem[s] for s in OVERVIEW_INCLUDED_STEMS if s in by_stem]
    if selected:
        return selected
    return [inp_path for inp_path in inp_paths if inp_path.stem not in OVERVIEW_EXCLUDED_STEMS]


def sanitize_inp_for_wntr(inp_path: Path) -> Path:
    text = inp_path.read_text()
    sanitized = re.sub(
        r"(^\s*Statistic\s+)(Average)(\s*$)",
        r"\1AVERAGED\3",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    sanitized = re.sub(
        r"(^\s*Quality\s+)Chemical\s+TIME(\s*$)",
        r"\1NONE\2",
        sanitized,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if sanitized == text:
        return inp_path
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_DIR / f"{inp_path.stem}_wntr_sanitized.inp"
    tmp_path.write_text(sanitized)
    return tmp_path


def load_network(inp_path: Path) -> wntr.network.WaterNetworkModel:
    plot_path = sanitize_inp_for_wntr(inp_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Not all curves were used", category=UserWarning)
        return wntr.network.WaterNetworkModel(str(plot_path))


def original_positions(wn: wntr.network.WaterNetworkModel) -> dict[str, np.ndarray]:
    coords: dict[str, np.ndarray] = {}
    missing = False
    for node_id in wn.node_name_list:
        xy = getattr(wn.get_node(node_id), "coordinates", None)
        if xy is None:
            missing = True
            break
        coords[node_id] = np.asarray(xy, dtype=float)
    if not missing:
        return coords

    graph = wn.to_graph().to_undirected()
    layout = nx.spring_layout(graph, seed=SEED)
    return {node_id: np.asarray(xy, dtype=float) for node_id, xy in layout.items()}


def normalize_positions(pos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arr = np.vstack([pos[node_id] for node_id in pos])
    center = np.mean(arr, axis=0)
    arr = arr - center
    scale = max(float(np.max(np.ptp(arr, axis=0))), 1.0e-12)
    arr = arr / scale
    return {node_id: arr[idx] for idx, node_id in enumerate(pos)}


def squarify_positions(pos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Scale x and y independently so the bounding box becomes a square.
    This removes geographic elongation and gives each network a compact,
    schematic appearance suitable for multi-panel overviews."""
    arr = np.vstack([pos[node_id] for node_id in pos])
    center = np.mean(arr, axis=0)
    arr = arr - center
    ranges = np.ptp(arr, axis=0)
    ranges = np.maximum(ranges, 1.0e-12)
    arr[:, 0] = arr[:, 0] / (0.5 * ranges[0])
    arr[:, 1] = arr[:, 1] / (0.5 * ranges[1])
    return {node_id: arr[idx] for idx, node_id in enumerate(pos)}


def overview_positions(wn: wntr.network.WaterNetworkModel) -> dict[str, np.ndarray]:
    """Topology-only layout for overview figures.
    Uses Kamada--Kawai layout (pure graph distances, no geography) followed by
    gentle special-neighborhood expansion and normalization.  This produces
    compact, symmetric, schematic networks that fill each subplot naturally."""
    graph = layout_graph(wn)
    pos = nx.kamada_kawai_layout(graph, weight="weight")
    pos = {node_id: np.asarray(xy, dtype=float) for node_id, xy in pos.items()}
    pos = normalize_positions(pos)
    # Gentle special-neighborhood expansion so pumps/valves don't overlap
    pos = expand_special_neighborhoods(wn, pos, graph)
    pos = normalize_positions(pos)
    # Very gentle compression to equalize scales without squashing loops
    n = max(len(graph.nodes()), 2)
    exponent = 0.92 if n <= 20 else 0.88 if n <= 90 else 0.90
    pos = nonlinear_compress(pos, exponent)
    pos = normalize_positions(pos)
    return pos


def dict_to_array(pos: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    return np.vstack([pos[node_id] for node_id in order])


def array_to_dict(arr: np.ndarray, order: list[str]) -> dict[str, np.ndarray]:
    return {node_id: arr[idx] for idx, node_id in enumerate(order)}


def align_to_target(pos: dict[str, np.ndarray], target: dict[str, np.ndarray], order: list[str]) -> dict[str, np.ndarray]:
    x = dict_to_array(pos, order)
    y = dict_to_array(target, order)
    x = x - np.mean(x, axis=0)
    y = y - np.mean(y, axis=0)
    h = x.T @ y
    u, _, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    x_aligned = x @ r
    x_scale = max(float(np.max(np.ptp(x_aligned, axis=0))), 1.0e-12)
    y_scale = max(float(np.max(np.ptp(y, axis=0))), 1.0e-12)
    x_aligned *= y_scale / x_scale
    return array_to_dict(x_aligned, order)


def nonlinear_compress(pos: dict[str, np.ndarray], exponent: float) -> dict[str, np.ndarray]:
    order = list(pos)
    arr = dict_to_array(pos, order)
    arr = np.sign(arr) * np.abs(arr) ** exponent
    return normalize_positions(array_to_dict(arr, order))


def network_metadata(wn: wntr.network.WaterNetworkModel) -> dict[str, set[str]]:
    return {
        "junctions": set(wn.junction_name_list),
        "tanks": set(wn.tank_name_list),
        "reservoirs": set(wn.reservoir_name_list),
        "pipes": set(wn.pipe_name_list),
        "pumps": set(wn.pump_name_list),
        "valves": set(wn.valve_name_list),
    }


def special_nodes(wn: wntr.network.WaterNetworkModel) -> set[str]:
    meta = network_metadata(wn)
    special = set(meta["tanks"]) | set(meta["reservoirs"])
    for link_id in wn.pump_name_list + wn.valve_name_list:
        link = wn.get_link(link_id)
        special.add(link.start_node_name)
        special.add(link.end_node_name)
    return special


def layout_graph(wn: wntr.network.WaterNetworkModel) -> nx.Graph:
    meta = network_metadata(wn)
    special = special_nodes(wn)
    graph = nx.Graph()
    graph.add_nodes_from(wn.node_name_list)
    for link_id in wn.link_name_list:
        link = wn.get_link(link_id)
        u = link.start_node_name
        v = link.end_node_name
        weight = 1.0
        if link_id in meta["pumps"] or link_id in meta["valves"]:
            weight = 0.18
        elif u in special or v in special:
            weight = 0.40
        if graph.has_edge(u, v):
            graph[u][v]["weight"] = min(graph[u][v]["weight"], weight)
            graph[u][v]["multiplicity"] += 1
        else:
            graph.add_edge(u, v, weight=weight, multiplicity=1)
    return graph


def edge_length_stats(pos: dict[str, np.ndarray], graph: nx.Graph) -> tuple[float, float]:
    lengths = []
    for u, v in graph.edges():
        lengths.append(float(np.linalg.norm(pos[v] - pos[u])))
    if not lengths:
        return 0.2, 0.2
    return float(np.median(lengths)), float(np.max(lengths))


def unit_vector(vec: np.ndarray, fallback_angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12:
        return np.array([math.cos(fallback_angle), math.sin(fallback_angle)])
    return vec / norm


def stretch_special_edges(
    wn: wntr.network.WaterNetworkModel,
    pos: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    graph: nx.Graph,
) -> dict[str, np.ndarray]:
    pos = {k: v.copy() for k, v in pos.items()}
    base, _ = edge_length_stats(pos, graph)
    meta = network_metadata(wn)
    special = special_nodes(wn)

    def push_pair(u: str, v: str, desired: float) -> None:
        current = pos[v] - pos[u]
        orig = target[v] - target[u]
        direction = unit_vector(orig if np.linalg.norm(orig) > 1.0e-12 else current, fallback_angle=0.35)
        dist = float(np.linalg.norm(current))
        if dist >= desired:
            return
        gap = desired - dist
        if u in special and v in special:
            pos[u] -= 0.5 * gap * direction
            pos[v] += 0.5 * gap * direction
        elif u in special:
            pos[v] += 0.70 * gap * direction
            pos[u] -= 0.30 * gap * direction
        elif v in special:
            pos[u] -= 0.70 * gap * direction
            pos[v] += 0.30 * gap * direction
        else:
            pos[u] -= 0.50 * gap * direction
            pos[v] += 0.50 * gap * direction

    for link_id in wn.pump_name_list + wn.valve_name_list:
        link = wn.get_link(link_id)
        push_pair(link.start_node_name, link.end_node_name, 1.65 * base)

    for node_id in wn.tank_name_list + wn.reservoir_name_list:
        neighbors = list(graph.neighbors(node_id))
        if len(neighbors) == 1:
            push_pair(neighbors[0], node_id, 1.85 * base)

    return normalize_positions(pos)


def expand_special_neighborhoods(wn: wntr.network.WaterNetworkModel, pos: dict[str, np.ndarray], graph: nx.Graph) -> dict[str, np.ndarray]:
    pos = {k: v.copy() for k, v in pos.items()}
    base, _ = edge_length_stats(pos, graph)
    special = special_nodes(wn)
    for _ in range(4):
        for s in sorted(special):  # sorted, so the layout does not depend on the string hash seed
            lengths = nx.single_source_shortest_path_length(graph, s, cutoff=2)
            for node_id, hop in lengths.items():
                if node_id == s:
                    continue
                desired = base * (1.55 if hop == 1 else 1.10)
                vec = pos[node_id] - pos[s]
                direction = unit_vector(vec, fallback_angle=0.4 + 0.3 * hop)
                dist = float(np.linalg.norm(vec))
                if dist >= desired:
                    continue
                gap = desired - dist
                if node_id in special:
                    pos[node_id] += 0.20 * gap * direction
                    pos[s] -= 0.20 * gap * direction
                else:
                    pos[node_id] += 0.35 * gap * direction
        pos = normalize_positions(pos)
    return pos


def peripheral_storage_placement(
    wn: wntr.network.WaterNetworkModel,
    pos: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    graph: nx.Graph,
) -> dict[str, np.ndarray]:
    pos = {k: v.copy() for k, v in pos.items()}
    base, _ = edge_length_stats(pos, graph)
    for node_id in wn.tank_name_list + wn.reservoir_name_list:
        neighbors = list(graph.neighbors(node_id))
        if len(neighbors) != 1:
            continue
        nbr = neighbors[0]
        preferred = target[node_id] - target[nbr]
        direction = unit_vector(preferred if np.linalg.norm(preferred) > 1.0e-12 else pos[node_id] - pos[nbr], fallback_angle=0.7)
        pos[node_id] = pos[nbr] + 1.85 * base * direction
    return normalize_positions(pos)


def schematic_positions(wn: wntr.network.WaterNetworkModel, squarify: bool = False) -> dict[str, np.ndarray]:
    target = normalize_positions(original_positions(wn))
    graph = layout_graph(wn)
    order = list(graph.nodes())
    pos0 = {node_id: target[node_id] for node_id in order}
    n = max(len(order), 2)
    if n <= 12:
        k = 1.45 / math.sqrt(n)
        iterations = 250
    elif n <= 90:
        k = 1.80 / math.sqrt(n)
        iterations = 420
    else:
        k = 2.55 / math.sqrt(n)
        iterations = 720
    spring = nx.spring_layout(graph, pos=pos0, seed=SEED, k=k, iterations=iterations, weight="weight", scale=1.0)
    pos = {node_id: np.asarray(xy, dtype=float) for node_id, xy in spring.items()}
    pos = align_to_target(pos, target, order)
    pos = stretch_special_edges(wn, pos, target, graph)
    pos = expand_special_neighborhoods(wn, pos, graph)
    pos = peripheral_storage_placement(wn, pos, target, graph)
    exponent = 0.93 if n <= 20 else 0.89 if n <= 90 else 0.90
    pos = nonlinear_compress(pos, exponent)
    pos = align_to_target(pos, target, order)
    pos = peripheral_storage_placement(wn, pos, target, graph)
    pos = normalize_positions(pos)
    if squarify:
        pos = squarify_positions(pos)
    return pos


def pair_offsets(wn: wntr.network.WaterNetworkModel, scale: float) -> dict[str, float]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for link_id in wn.link_name_list:
        link = wn.get_link(link_id)
        pair = tuple(sorted((link.start_node_name, link.end_node_name)))
        groups[pair].append(link_id)

    offsets: dict[str, float] = {}
    spacing = 0.020 * max(scale, 1.0)
    for link_ids in groups.values():
        if len(link_ids) == 1:
            offsets[link_ids[0]] = 0.0
            continue
        ordered = sorted(link_ids)
        center = 0.5 * (len(ordered) - 1)
        for idx, link_id in enumerate(ordered):
            offsets[link_id] = (idx - center) * spacing
    return offsets


def shifted_segment(p0: np.ndarray, p1: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 1.0e-12 or abs(offset) <= 1.0e-12:
        return p0, p1
    perp = np.array([-vec[1], vec[0]], dtype=float) / length
    shift = offset * perp
    return p0 + shift, p1 + shift


def label_candidates(wn: wntr.network.WaterNetworkModel, graph: nx.Graph) -> list[str]:
    if len(wn.node_name_list) <= 8:
        return list(wn.node_name_list)
    special = list(wn.reservoir_name_list) + list(wn.tank_name_list)
    if len(wn.node_name_list) <= 26:
        extras = []
        for node_id in wn.junction_name_list:
            degree = graph.degree(node_id)
            if degree != 2 or node_id in special_nodes(wn):
                extras.append(node_id)
        ordered = special + [node_id for node_id in extras if node_id not in special]
        return ordered[:14]
    return special


def node_label_position(
    node_id: str,
    wn: wntr.network.WaterNetworkModel,
    graph: nx.Graph,
    positions: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    point = positions[node_id]
    meta = network_metadata(wn)
    if node_id in meta["tanks"] or node_id in meta["reservoirs"]:
        neighbors = list(graph.neighbors(node_id))
        if neighbors:
            nbr = neighbors[0]
            outward = point - positions[nbr]
            direction = unit_vector(outward, fallback_angle=0.3)
            tangent_pref = target[node_id] - target[nbr]
            tangent = np.array([-direction[1], direction[0]], dtype=float)
            if np.dot(tangent, tangent_pref) < 0.0:
                tangent *= -1.0
            return point + 0.070 * scale * direction + 0.020 * scale * tangent
    preferred = target[node_id]
    direction = preferred if np.linalg.norm(preferred) > 1.0e-12 else point - center
    direction = unit_vector(direction, fallback_angle=0.3)
    return point + 0.045 * scale * direction


def edge_label_position(p0: np.ndarray, p1: np.ndarray, bias: float, scale: float) -> np.ndarray:
    mid = 0.5 * (p0 + p1)
    vec = p1 - p0
    perp = unit_vector(np.array([-vec[1], vec[0]], dtype=float), fallback_angle=0.6)
    return mid + bias * 0.055 * scale * perp


def annotate_network(
    ax: plt.Axes,
    wn: wntr.network.WaterNetworkModel,
    positions: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    center: np.ndarray,
    scale: float,
    offsets: dict[str, float],
    show_link_labels: bool,
) -> None:
    graph = layout_graph(wn)
    node_ids = label_candidates(wn, graph)
    for node_id in node_ids:
        point = node_label_position(node_id, wn, graph, positions, target, center, scale)
        ax.text(
            point[0],
            point[1],
            str(node_id),
            fontsize=8.0,
            color=BLACK,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.88),
            zorder=10,
        )

    if not show_link_labels:
        return

    for idx, link_id in enumerate(list(wn.pump_name_list) + list(wn.valve_name_list)):
        link = wn.get_link(link_id)
        p0, p1 = shifted_segment(
            positions[link.start_node_name],
            positions[link.end_node_name],
            offsets.get(link_id, 0.0),
        )
        label_pt = edge_label_position(p0, p1, bias=1.0 if idx % 2 == 0 else -1.0, scale=scale)
        if link_id in wn.valve_name_list:
            valve_type = getattr(link, "valve_type", "VALVE")
            label = f"{valve_type} {link_id}"
            color = DARKRED
        else:
            label = f"Pump {link_id}"
            color = DARKBLUE
        ax.text(
            label_pt[0],
            label_pt[1],
            label,
            fontsize=7.6,
            color=color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.90),
            zorder=11,
        )


def draw_network(ax: plt.Axes, wn: wntr.network.WaterNetworkModel, annotate: bool, legend: bool, compact: bool = False, overview: bool = False, positions: dict[str, np.ndarray] | None = None) -> None:
    if positions is None:
        if overview:
            positions = overview_positions(wn)
        else:
            positions = schematic_positions(wn)
    target = normalize_positions(original_positions(wn))
    coords = np.vstack([positions[node_id] for node_id in wn.node_name_list])
    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)
    xspan = max(float(xmax - xmin), 1.0e-12)
    yspan = max(float(ymax - ymin), 1.0e-12)
    scale = max(xspan, yspan)
    center = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
    offsets = pair_offsets(wn, scale)
    meta = network_metadata(wn)

    small = len(wn.node_name_list) <= 20
    medium = len(wn.node_name_list) <= 120
    pipe_width = 1.15 if small else 0.95 if medium else 0.75
    node_size = 26 if small else 16 if medium else 9
    tank_size = node_size * 2.8
    reservoir_size = node_size * 2.4
    pump_size = 58 if small else 46 if medium else 34
    valve_size = 44 if small else 36 if medium else 28
    if overview:
        # Larger markers and thicker lines for the overview figure
        pipe_width *= 1.8
        node_size *= 2.0
        tank_size *= 1.8
        reservoir_size *= 1.8
        pump_size *= 2.0
        valve_size *= 2.0
    elif compact:
        pipe_width *= 0.96
        node_size *= 0.92
        tank_size *= 1.10
        reservoir_size *= 1.10
        pump_size *= 1.22
        valve_size *= 1.22

    for link_id in wn.link_name_list:
        link = wn.get_link(link_id)
        p0, p1 = shifted_segment(
            positions[link.start_node_name],
            positions[link.end_node_name],
            offsets.get(link_id, 0.0),
        )
        if link_id in meta["pipes"]:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=BLACK, alpha=0.72, linewidth=pipe_width, zorder=1)
        elif link_id in meta["pumps"]:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=DARKBLUE, linewidth=1.75 if small else 1.45, zorder=3)
            mid = 0.5 * (p0 + p1)
            ax.scatter(mid[0], mid[1], s=pump_size, c="white", edgecolors=DARKBLUE, linewidths=1.3, zorder=6)
        elif link_id in meta["valves"]:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=DARKRED, linewidth=1.60 if small else 1.30, zorder=4)
            mid = 0.5 * (p0 + p1)
            ax.scatter(mid[0], mid[1], s=valve_size, c="white", edgecolors=DARKRED, marker="s", linewidths=1.2, zorder=7)

    if wn.junction_name_list:
        junction_xy = np.vstack([positions[node_id] for node_id in wn.junction_name_list])
        ax.scatter(junction_xy[:, 0], junction_xy[:, 1], s=node_size, c="white", edgecolors=BLACK, linewidths=0.65, zorder=5)
    if wn.tank_name_list:
        tank_xy = np.vstack([positions[node_id] for node_id in wn.tank_name_list])
        ax.scatter(tank_xy[:, 0], tank_xy[:, 1], s=tank_size, c=SOFTBLUE, edgecolors=BLACK, marker="s", linewidths=0.90, zorder=8)
    if wn.reservoir_name_list:
        reservoir_xy = np.vstack([positions[node_id] for node_id in wn.reservoir_name_list])
        ax.scatter(reservoir_xy[:, 0], reservoir_xy[:, 1], s=reservoir_size, c=BLACK, edgecolors=BLACK, marker="D", linewidths=0.90, zorder=8)

    if annotate:
        show_link_labels = len(wn.node_name_list) <= 70
        annotate_network(ax, wn, positions, target, center, scale, offsets, show_link_labels=show_link_labels)

    if overview:
        margin_x = 0.10 * xspan
        margin_y = 0.10 * yspan
    elif compact:
        margin_x = 0.020 * xspan
        margin_y = 0.028 * yspan
    else:
        margin_x = 0.10 * xspan
        margin_y = 0.12 * yspan
    ax.set_xlim(xmin - margin_x, xmax + margin_x)
    ax.set_ylim(ymin - margin_y, ymax + margin_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if legend:
        handles = network_legend_handles()
        leg = ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            ncol=2,
            frameon=True,
            fontsize=10.2,
            borderpad=0.45,
            labelspacing=0.45,
            handlelength=1.6,
            columnspacing=1.15,
            prop={"family": "serif", "size": 10.2},
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.92)
        leg.get_frame().set_edgecolor("#d0d0d0")


def title_for_network(inp_path: Path, wn: wntr.network.WaterNetworkModel) -> str:
    aliases = {
        "ctown": "C-Town",
        "BWSN_Network_1": "BWSN Network 1",
        "BWSN_Network_1_original": "BWSN Network 1 Original",
        "Threenodes": "Three-Nodes",
    }
    return aliases.get(inp_path.stem, inp_path.stem.replace("_", " "))


def network_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=BLACK, alpha=0.62, linewidth=2.1, label="Pipe"),
        Line2D(
            [0],
            [0],
            color=DARKBLUE,
            linewidth=2.4,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=DARKBLUE,
            markersize=11.5,
            label="Pump",
        ),
        Line2D(
            [0],
            [0],
            color=DARKRED,
            linewidth=2.2,
            marker="s",
            markerfacecolor="white",
            markeredgecolor=DARKRED,
            markersize=11.0,
            label="Valve",
        ),
        Line2D(
            [0],
            [0],
            color="white",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=BLACK,
            markersize=10.4,
            linewidth=0.0,
            label="Junction",
        ),
        Line2D(
            [0],
            [0],
            color=SOFTBLUE,
            marker="s",
            markerfacecolor=SOFTBLUE,
            markeredgecolor=BLACK,
            markersize=11.8,
            linewidth=0.0,
            label="Tank",
        ),
        Line2D(
            [0],
            [0],
            color=BLACK,
            marker="D",
            markerfacecolor=BLACK,
            markeredgecolor=BLACK,
            markersize=11.2,
            linewidth=0.0,
            label="Reservoir",
        ),
    ]


def save_overview_figure(inp_paths: list[Path]) -> Path:
    overview_paths = overview_inp_files(inp_paths)
    if not overview_paths:
        raise ValueError("No INP files available for the overview figure after filtering.")
    n = len(overview_paths)
    ncols = n
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.6))
    fig.subplots_adjust(left=0.005, right=0.998, bottom=0.30, top=0.84, wspace=0.04)
    axes_flat = np.atleast_1d(axes).ravel()
    # Colors used in the legend handles, mapped to each tuple entry so the
    # numbers under each panel match the components in the drawing:
    # (n_J=junction, n_T=tank, n_R=reservoir, n_E=pipe, n_P=pump, n_V=valve).
    TANK_TEXT = "#2A5F9E"
    tuple_colors = [BLACK, TANK_TEXT, BLACK, BLACK, DARKBLUE, DARKRED]
    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
    # Networks whose INP coordinates already give a clean, non-overlapping
    # schematic. For these the original geographic layout is kept so that
    # parallel pipes do not collapse onto each other; the rest go through
    # the topology-only Kamada-Kawai overview layout.
    OVERVIEW_USE_ORIGINAL = {"Net1", "anytown"}
    for ax, inp_path in zip(axes_flat, overview_paths):
        wn = load_network(inp_path)
        if inp_path.stem in OVERVIEW_USE_ORIGINAL:
            pos_override = squarify_positions(original_positions(wn))
        else:
            pos_override = None
        draw_network(ax, wn, annotate=False, legend=False, compact=True, overview=True, positions=pos_override)
        ax.set_title(
            title_for_network(inp_path, wn),
            fontsize=19,
            pad=6,
            color=DARKBLUE,
            fontfamily="serif",
            fontweight="semibold",
        )
        counts = (
            len(wn.junction_name_list),
            len(wn.tank_name_list),
            len(wn.reservoir_name_list),
            len(wn.pipe_name_list),
            len(wn.pump_name_list),
            len(wn.valve_name_list),
        )
        text_kw = dict(textprops=dict(fontfamily="serif", fontsize=16.5))
        children = [TextArea("{", textprops=dict(fontfamily="serif", fontsize=16.5, color=BLACK))]
        for k, val in enumerate(counts):
            children.append(TextArea(str(val), textprops=dict(fontfamily="serif", fontsize=16.5,
                                                              color=tuple_colors[k], fontweight="bold")))
            if k < len(counts) - 1:
                children.append(TextArea(",", textprops=dict(fontfamily="serif", fontsize=16.5, color=BLACK)))
        children.append(TextArea("}", textprops=dict(fontfamily="serif", fontsize=16.5, color=BLACK)))
        box = HPacker(children=children, align="center", pad=0, sep=1.5)
        anchored = AnchoredOffsetbox(loc="upper center", child=box, pad=0.0,
                                     borderpad=0.2, frameon=False,
                                     bbox_to_anchor=(0.5, -0.03),
                                     bbox_transform=ax.transAxes)
        ax.add_artist(anchored)
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.legend(
        handles=network_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
        handlelength=2.6,
        columnspacing=2.0,
        fontsize=16,
        prop={"family": "serif", "size": 16},
    )
    pdf_path = OUTPUT_DIR / "all_networks_overview.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main() -> None:
    inp_paths = discover_inp_files()
    if not inp_paths:
        raise FileNotFoundError(f"No INP files found in {NETWORKS_DIR}.")
    overview_path = save_overview_figure(inp_paths)
    print(f"Saved overview PDF to {overview_path}")


if __name__ == "__main__":
    main()
