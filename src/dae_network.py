#!/usr/bin/env python3
"""
Continuous-time hydraulic DAE of a water network built from an EPANET INP file, and its
comparison with EPANET.

The state is

    x = [q, p_J, p_T, p_R]

where q are the flows in pipes, pumps, and valves, and p_J, p_T, and p_R are the junction,
tank, and reservoir heads. The module parses the INP file, builds the DAE without eliminating
the heads, integrates it, and compares the trajectories with EPANET through EPyT.
EPANET simple controls (status and setting changes at clock times, times since start, and
node levels) are supported, while EPANET rule-based controls are not.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from scipy.optimize import least_squares, root


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
VENDOR_DIR = REPO_ROOT / ".vendor"
NETWORKS_DIR = ROOT / "Networks"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# INP file used when --inp is not given
SCRIPT_INP_FILE: str | None = None  # e.g. "Net3.inp" or "src/Networks/Net3.inp"

from plot_style import BLACK, DARKBLUE, DARKRED, MAINBLUE, SOFTBLUE, SOFTGRAY, style_axes
import matplotlib.pyplot as plt


FT_TO_M = 0.3048
IN_TO_M = 0.0254
MM_TO_M = 1.0e-3
PSI_TO_M = 6894.757293168 / (1000.0 * 9.80665)
HOUR_TO_S = 3600.0
DAY_TO_S = 24.0 * HOUR_TO_S
EVENT_TOL_S = 1.0e-8
MAX_EVENT_DEPTH = 16
PLOT_DETAIL_LIMIT = 18
# When False, the error bands are drawn for |error|, on the same scale as the max-abs envelope
USE_SIGNED_ERROR_BAND = False
EPANET_INVALID_HEAD_THRESHOLD_M = 1.0e6
EPANET_INVALID_FLOW_THRESHOLD_M3S = 1.0e3
CONTROLLED_VALVE_TYPES = {"FCV", "PRV", "PSV"}
VALVE_SURROGATE_MIN_MINOR_LOSS = 1.0

US_FLOW_UNITS_TO_CMS = {
    "CFS": 0.028316846592,
    "GPM": 6.30901964e-5,
    "MGD": 0.0438126363889,
    "IMGD": 0.0526167824074,
    "AFD": 0.0142764101852,
}

SI_FLOW_UNITS_TO_CMS = {
    "LPS": 1.0e-3,
    "LPM": 1.0e-3 / 60.0,
    "MLD": 1.0e3 / 86400.0,
    "CMH": 1.0 / 3600.0,
    "CMD": 1.0 / 86400.0,
}


@dataclass
class Junction:
    node_id: str
    elevation_m: float
    demand_terms: list[tuple[float, str | None]] = field(default_factory=list)


@dataclass
class Reservoir:
    node_id: str
    base_head_m: float
    pattern_id: str | None


@dataclass
class Tank:
    node_id: str
    elevation_m: float
    init_level_m: float
    min_level_m: float
    max_level_m: float
    diameter_m: float

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m**2 / 4.0

    @property
    def initial_head_m(self) -> float:
        return self.elevation_m + self.init_level_m


@dataclass
class Pipe:
    link_id: str
    start_node: str
    end_node: str
    length_m: float
    diameter_m: float
    roughness_raw: float
    minor_loss: float
    status: str

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m**2 / 4.0


@dataclass
class Pump:
    link_id: str
    start_node: str
    end_node: str
    curve_id: str
    speed: float
    status: str


@dataclass
class Valve:
    link_id: str
    start_node: str
    end_node: str
    diameter_m: float
    valve_type: str
    setting_raw: str
    minor_loss: float
    status: str

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m**2 / 4.0


@dataclass
class PumpLaw:
    mode: str
    coefficients: tuple[float, ...]
    q_points_m3s: np.ndarray
    h_points_m: np.ndarray


@dataclass
class ValveLaw:
    mode: str
    valve_type: str
    coefficients: tuple[float, ...]
    q_points_m3s: np.ndarray
    h_points_m: np.ndarray
    start_node: str
    end_node: str


@dataclass
class ControlRule:
    rule_type: str
    link_id: str
    setting_kind: str
    setting_value: float | None
    node_id: str | None
    trigger_value_m: float | None
    trigger_time_s: float | None
    raw_text: str


@dataclass
class HydraulicOptions:
    units: str
    headloss: str
    default_pattern_id: str | None
    demand_multiplier: float
    viscosity_rel: float


@dataclass
class TimeOptions:
    duration_s: float
    hydraulic_timestep_s: float
    pattern_timestep_s: float
    report_timestep_s: float
    pattern_start_s: float
    start_clocktime_s: float


@dataclass
class EpanetSeries:
    time_s: np.ndarray
    node_ids: list[str]
    link_ids: list[str]
    head_m: np.ndarray
    flow_m3s: np.ndarray
    status: np.ndarray
    setting: np.ndarray


@dataclass
class NetworkCase:
    junctions: list[Junction]
    reservoirs: list[Reservoir]
    tanks: list[Tank]
    pipes: list[Pipe]
    pumps: list[Pump]
    valves: list[Valve]
    patterns: dict[str, np.ndarray]
    curves: dict[str, np.ndarray]
    options: HydraulicOptions
    time: TimeOptions
    controls: list[ControlRule]
    gravity: float
    kinematic_viscosity_m2s: float
    absolute_roughness_m: float
    pump_inertance_scale: float
    valve_inertance_scale: float
    supported_link_ids: list[str]
    supported_link_types: list[str]
    supported_link_start: list[str]
    supported_link_end: list[str]
    pipe_index_by_link: dict[str, int]
    pump_index_by_link: dict[str, int]
    valve_index_by_link: dict[str, int]
    junction_index: dict[str, int]
    tank_index: dict[str, int]
    reservoir_index: dict[str, int]
    node_elevation_m: dict[str, float]
    Cj: np.ndarray
    Ct: np.ndarray
    Cr: np.ndarray
    tank_areas_m2: np.ndarray
    link_inertance_gain: np.ndarray
    link_minor_resistance: np.ndarray
    link_base_open: np.ndarray
    link_base_setting: np.ndarray
    pump_laws: dict[str, PumpLaw]
    valve_laws: dict[str, ValveLaw]


def ensure_epyt() -> None:
    if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
        sys.path.append(str(VENDOR_DIR))


def parse_time_to_seconds(value: str) -> float:
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            return 3600.0 * int(parts[0]) + 60.0 * int(parts[1])
        if len(parts) == 3:
            return 3600.0 * int(parts[0]) + 60.0 * int(parts[1]) + float(parts[2])
            raise ValueError(f"Unsupported EPANET time token: {value}")
        raise ValueError(f"Unsupported EPANET time token: {value}")
    return 3600.0 * float(text)


def parse_clocktime_tokens(tokens: Sequence[str]) -> float:
    joined = " ".join(tokens).strip().lower()
    if not joined:
        raise ValueError("Empty clock time.")
    parts = joined.split()
    meridiem = None
    if parts[-1] in {"am", "pm"}:
        meridiem = parts[-1]
        time_part = " ".join(parts[:-1]).strip()
    else:
        time_part = joined
    if ":" in time_part:
        time_fields = time_part.split(":")
        if len(time_fields) == 2:
            hh_str, mm_str = time_fields
            ss_str = "0"
        elif len(time_fields) == 3:
            hh_str, mm_str, ss_str = time_fields
        else:
            raise ValueError(f"Unsupported EPANET clock token: {time_part}")
        hour = int(float(hh_str))
        minute = int(float(mm_str))
        second = int(float(ss_str))
    else:
        raw = float(time_part)
        if meridiem is None and raw > 24.0:
            return raw
        hour = int(raw)
        minute = int(round((raw - hour) * 60.0))
        second = 0
    if meridiem is not None:
        if hour == 12:
            hour = 0
        if meridiem == "pm":
            hour += 12
    return float(hour * 3600 + minute * 60 + second)


def parse_inp_sections(path: Path) -> Dict[str, list[str]]:
    sections: Dict[str, list[str]] = {}
    current = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]").upper()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def parse_patterns(lines: Sequence[str]) -> dict[str, np.ndarray]:
    patterns: dict[str, list[float]] = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        patterns.setdefault(parts[0], []).extend(float(v) for v in parts[1:])
    return {k: np.asarray(v, dtype=float) for k, v in patterns.items()}


def parse_curves(lines: Sequence[str]) -> dict[str, np.ndarray]:
    curves: dict[str, list[tuple[float, float]]] = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        curves.setdefault(parts[0], []).append((float(parts[1]), float(parts[2])))
    return {k: np.asarray(v, dtype=float) for k, v in curves.items()}


def parse_status(lines: Sequence[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            status[parts[0]] = parts[1].upper()
    return status


def flow_factor_to_cms(units: str) -> float:
    units_up = units.upper()
    if units_up in US_FLOW_UNITS_TO_CMS:
        return US_FLOW_UNITS_TO_CMS[units_up]
    if units_up in SI_FLOW_UNITS_TO_CMS:
        return SI_FLOW_UNITS_TO_CMS[units_up]
    raise ValueError(f"Unsupported EPANET flow units: {units}")


def is_us_units(units: str) -> bool:
    return units.upper() in US_FLOW_UNITS_TO_CMS


def head_factor_to_m(units: str) -> float:
    return FT_TO_M if is_us_units(units) else 1.0


def length_factor_to_m(units: str) -> float:
    return FT_TO_M if is_us_units(units) else 1.0


def diameter_factor_to_m(units: str) -> float:
    return IN_TO_M if is_us_units(units) else MM_TO_M


def roughness_dw_factor_to_m(units: str) -> float:
    return FT_TO_M * 1.0e-3 if is_us_units(units) else MM_TO_M


def pressure_factor_to_head_m(units: str) -> float:
    return PSI_TO_M if is_us_units(units) else 1.0


def parse_options(lines: Sequence[str]) -> HydraulicOptions:
    opts = {
        "UNITS": "GPM",
        "HEADLOSS": "H-W",
        "PATTERN": None,
        "DEMAND MULTIPLIER": "1.0",
        "VISCOSITY": "1.0",
    }
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        key = " ".join(parts[:-1]).upper()
        if key in opts:
            opts[key] = parts[-1]
    return HydraulicOptions(
        units=str(opts["UNITS"]),
        headloss=str(opts["HEADLOSS"]).upper(),
        default_pattern_id=None if opts["PATTERN"] in {None, ""} else str(opts["PATTERN"]),
        demand_multiplier=float(opts["DEMAND MULTIPLIER"]),
        viscosity_rel=float(opts["VISCOSITY"]),
    )


def parse_times(lines: Sequence[str]) -> TimeOptions:
    values = {
        "DURATION": 24.0 * HOUR_TO_S,
        "HYDRAULIC TIMESTEP": HOUR_TO_S,
        "PATTERN TIMESTEP": HOUR_TO_S,
        "REPORT TIMESTEP": HOUR_TO_S,
        "PATTERN START": 0.0,
        "START CLOCKTIME": 0.0,
    }
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        up_line = line.upper()
        if up_line.startswith("START CLOCKTIME"):
            values["START CLOCKTIME"] = parse_clocktime_tokens(parts[2:])
            continue
        key = " ".join(parts[:-1]).upper()
        if key in values:
            values[key] = parse_time_to_seconds(parts[-1])
    return TimeOptions(
        duration_s=float(values["DURATION"]),
        hydraulic_timestep_s=float(values["HYDRAULIC TIMESTEP"]),
        pattern_timestep_s=float(values["PATTERN TIMESTEP"]),
        report_timestep_s=float(values["REPORT TIMESTEP"]),
        pattern_start_s=float(values["PATTERN START"]),
        start_clocktime_s=float(values["START CLOCKTIME"]),
    )


def parse_control_setting(token: str) -> tuple[str, float | None]:
    token_up = token.upper()
    if token_up in {"OPEN", "CLOSED"}:
        return token_up, None
    return "NUMERIC", float(token)


def convert_control_threshold_to_m(
    node_id: str,
    raw_value: float,
    options: HydraulicOptions,
    tank_index: dict[str, int],
    junction_index: dict[str, int],
    reservoir_index: dict[str, int],
) -> float:
    if node_id in tank_index:
        return raw_value * head_factor_to_m(options.units)
    if node_id in junction_index:
        return raw_value * pressure_factor_to_head_m(options.units)
    if node_id in reservoir_index:
        return raw_value * head_factor_to_m(options.units)
    return raw_value


def parse_controls(
    lines: Sequence[str],
    options: HydraulicOptions,
    tank_index: dict[str, int],
    junction_index: dict[str, int],
    reservoir_index: dict[str, int],
) -> list[ControlRule]:
    controls: list[ControlRule] = []
    for line in lines:
        parts = line.split()
        up = [p.upper() for p in parts]
        if len(parts) < 4 or up[0] != "LINK":
            continue
        link_id = parts[1]
        setting_kind, setting_value = parse_control_setting(parts[2])
        if up[3] == "AT" and len(parts) >= 6:
            if up[4] == "CLOCKTIME":
                controls.append(
                    ControlRule(
                        rule_type="TIMEOFDAY",
                        link_id=link_id,
                        setting_kind=setting_kind,
                        setting_value=setting_value,
                        node_id=None,
                        trigger_value_m=None,
                        trigger_time_s=parse_clocktime_tokens(parts[5:]),
                        raw_text=line,
                    )
                )
            elif up[4] == "TIME":
                controls.append(
                    ControlRule(
                        rule_type="TIMER",
                        link_id=link_id,
                        setting_kind=setting_kind,
                        setting_value=setting_value,
                        node_id=None,
                        trigger_value_m=None,
                        trigger_time_s=parse_clocktime_tokens(parts[5:]),
                        raw_text=line,
                    )
                )
        elif up[3] == "IF" and len(parts) >= 8 and up[4] == "NODE":
            node_id = parts[5]
            relation = up[6]
            threshold_m = convert_control_threshold_to_m(
                node_id,
                float(parts[7]),
                options,
                tank_index,
                junction_index,
                reservoir_index,
            )
            controls.append(
                ControlRule(
                    rule_type="LOWLEVEL" if relation == "BELOW" else "HIGHLEVEL",
                    link_id=link_id,
                    setting_kind=setting_kind,
                    setting_value=setting_value,
                    node_id=node_id,
                    trigger_value_m=threshold_m,
                    trigger_time_s=None,
                    raw_text=line,
                )
            )
    return controls


def fit_abg_pump_law(q: np.ndarray, h: np.ndarray) -> PumpLaw:
    def residual(theta: np.ndarray) -> np.ndarray:
        A, B, C = theta
        return A - B * np.maximum(q, 0.0) ** C - h

    guess = np.array([max(float(h[0]), 1.0e-6), 1.0e4, 2.0], dtype=float)
    fit = least_squares(
        residual,
        x0=guess,
        bounds=([1.0e-6, 1.0e-12, 0.5], [1.0e6, 1.0e12, 5.0]),
    )
    return PumpLaw(
        mode="abg",
        coefficients=(float(fit.x[0]), float(fit.x[1]), float(fit.x[2])),
        q_points_m3s=q,
        h_points_m=h,
    )


def build_pump_laws(curves: dict[str, np.ndarray], pumps: Sequence[Pump], units: str) -> dict[str, PumpLaw]:
    q_factor = flow_factor_to_cms(units)
    h_factor = head_factor_to_m(units)
    laws: dict[str, PumpLaw] = {}
    for pump in pumps:
        if pump.curve_id not in curves:
            raise KeyError(f"Pump curve {pump.curve_id} not found for pump {pump.link_id}.")
        raw = np.asarray(curves[pump.curve_id], dtype=float)
        q = raw[:, 0] * q_factor
        h = raw[:, 1] * h_factor
        order = np.argsort(q)
        q = q[order]
        h = h[order]
        if q.size == 1:
            qd = float(q[0])
            hd = float(h[0])
            q = np.asarray([0.0, qd, 2.0 * qd], dtype=float)
            h = np.asarray([1.3333333333 * hd, hd, 0.0], dtype=float)
        if q.size == 3:
            law = fit_abg_pump_law(q, h)
        else:
            law = PumpLaw(mode="piecewise", coefficients=tuple(), q_points_m3s=q, h_points_m=h)
        laws[pump.link_id] = law
    return laws


def build_valve_laws(
    curves: dict[str, np.ndarray],
    valves: Sequence[Valve],
    options: HydraulicOptions,
    gravity: float,
    approximate_controlled_valves_as_gpv: bool,
    controlled_valve_minor_loss: dict[str, float] | None = None,
) -> dict[str, ValveLaw]:
    q_factor = flow_factor_to_cms(options.units)
    h_factor = head_factor_to_m(options.units)
    p_factor = pressure_factor_to_head_m(options.units)
    laws: dict[str, ValveLaw] = {}
    for valve in valves:
        kind = valve.valve_type.upper()
        coeffs: tuple[float, ...]
        if approximate_controlled_valves_as_gpv and kind in CONTROLLED_VALVE_TYPES:
            warnings.warn(
                f"Approximating valve {valve.link_id} of type {kind} as a quadratic-loss GPV surrogate.",
                stacklevel=2,
            )
            k_minor = max(
                controlled_valve_minor_loss.get(valve.link_id, valve.minor_loss) if controlled_valve_minor_loss else valve.minor_loss,
                VALVE_SURROGATE_MIN_MINOR_LOSS,
            )
            coeffs = (k_minor / (2.0 * gravity * valve.area_m2**2),)
            laws[valve.link_id] = ValveLaw("quadratic_loss", kind, coeffs, np.zeros(0), np.zeros(0), valve.start_node, valve.end_node)
            continue
        if kind == "GPV":
            if valve.setting_raw not in curves:
                raise KeyError(f"GPV curve {valve.setting_raw} not found for valve {valve.link_id}.")
            raw = np.asarray(curves[valve.setting_raw], dtype=float)
            q = np.abs(raw[:, 0] * q_factor)
            h = np.abs(raw[:, 1] * h_factor)
            order = np.argsort(q)
            laws[valve.link_id] = ValveLaw(
                mode="curve_loss",
                valve_type=kind,
                coefficients=tuple(),
                q_points_m3s=q[order],
                h_points_m=h[order],
                start_node=valve.start_node,
                end_node=valve.end_node,
            )
            continue
        if kind == "TCV":
            k_minor = float(valve.setting_raw)
            coeffs = (k_minor / (2.0 * gravity * valve.area_m2**2),)
            laws[valve.link_id] = ValveLaw(kind, kind, coeffs, np.zeros(0), np.zeros(0), valve.start_node, valve.end_node)
            continue
        if kind == "PBV":
            coeffs = (float(valve.setting_raw) * p_factor,)
            laws[valve.link_id] = ValveLaw("constant_drop", kind, coeffs, np.zeros(0), np.zeros(0), valve.start_node, valve.end_node)
            continue
        if kind == "FCV":
            coeffs = (float(valve.setting_raw) * q_factor,)
            laws[valve.link_id] = ValveLaw("fcv", kind, coeffs, np.zeros(0), np.zeros(0), valve.start_node, valve.end_node)
            continue
        if kind == "PRV":
            coeffs = (float(valve.setting_raw) * p_factor,)
            laws[valve.link_id] = ValveLaw("prv", kind, coeffs, np.zeros(0), np.zeros(0), valve.start_node, valve.end_node)
            continue
        if kind == "PSV":
            coeffs = (float(valve.setting_raw) * p_factor,)
            laws[valve.link_id] = ValveLaw("psv", kind, coeffs, np.zeros(0), np.zeros(0), valve.start_node, valve.end_node)
            continue
        if approximate_controlled_valves_as_gpv:
            warnings.warn(
                f"Approximating valve {valve.link_id} of type {kind} as a quadratic-loss GPV surrogate.",
                stacklevel=2,
            )
            coeffs = (valve.minor_loss / (2.0 * gravity * valve.area_m2**2),)
            laws[valve.link_id] = ValveLaw("quadratic_loss", kind, coeffs, np.zeros(0), np.zeros(0), valve.start_node, valve.end_node)
            continue
        raise NotImplementedError(
            f"Valve type {kind} is not yet fully supported. "
            f"Use --approximate-controlled-valves to replace it with a GPV-style surrogate."
        )
    return laws


def build_network_case(
    inp_path: Path,
    gravity: float = 9.80665,
    kinematic_viscosity_water_m2s: float = 1.007e-6,
    absolute_roughness_m: float = 2.591e-4,
    pump_inertance_scale: float = 1.0,
    valve_inertance_scale: float = 1.0,
    approximate_controlled_valves_as_gpv: bool = False,
    controlled_valve_minor_loss: dict[str, float] | None = None,
) -> NetworkCase:
    sections = parse_inp_sections(inp_path)
    options = parse_options(sections.get("OPTIONS", []))
    time = parse_times(sections.get("TIMES", []))
    patterns = parse_patterns(sections.get("PATTERNS", []))
    curves = parse_curves(sections.get("CURVES", []))
    status_overrides = parse_status(sections.get("STATUS", []))

    q_factor = flow_factor_to_cms(options.units)
    head_factor = head_factor_to_m(options.units)
    length_factor = length_factor_to_m(options.units)
    diameter_factor = diameter_factor_to_m(options.units)

    junctions: list[Junction] = []
    for line in sections.get("JUNCTIONS", []):
        parts = line.split()
        node_id = parts[0]
        elev_m = float(parts[1]) * head_factor
        base_demand = (float(parts[2]) * q_factor) if len(parts) >= 3 else 0.0
        pattern_id = parts[3] if len(parts) >= 4 else options.default_pattern_id
        terms: list[tuple[float, str | None]] = []
        if abs(base_demand) > 0.0 or pattern_id is not None:
            terms.append((base_demand, pattern_id))
        junctions.append(Junction(node_id=node_id, elevation_m=elev_m, demand_terms=terms))

    junction_lookup = {j.node_id: j for j in junctions}
    for line in sections.get("DEMANDS", []):
        parts = line.split()
        if len(parts) < 2:
            continue
        node_id = parts[0]
        demand_cms = float(parts[1]) * q_factor
        pattern_id = parts[2] if len(parts) >= 3 else options.default_pattern_id
        if node_id in junction_lookup:
            junction_lookup[node_id].demand_terms.append((demand_cms, pattern_id))

    reservoirs: list[Reservoir] = []
    for line in sections.get("RESERVOIRS", []):
        parts = line.split()
        reservoirs.append(
            Reservoir(
                node_id=parts[0],
                base_head_m=float(parts[1]) * head_factor,
                pattern_id=parts[2] if len(parts) >= 3 else None,
            )
        )

    tanks: list[Tank] = []
    for line in sections.get("TANKS", []):
        parts = line.split()
        tanks.append(
            Tank(
                node_id=parts[0],
                elevation_m=float(parts[1]) * head_factor,
                init_level_m=float(parts[2]) * head_factor,
                min_level_m=float(parts[3]) * head_factor,
                max_level_m=float(parts[4]) * head_factor,
                diameter_m=float(parts[5]) * length_factor,
            )
        )

    pipes: list[Pipe] = []
    for line in sections.get("PIPES", []):
        parts = line.split()
        link_id = parts[0]
        status = parts[7].upper() if len(parts) >= 8 else "OPEN"
        status = status_overrides.get(link_id, status).upper()
        pipes.append(
            Pipe(
                link_id=link_id,
                start_node=parts[1],
                end_node=parts[2],
                length_m=float(parts[3]) * length_factor,
                diameter_m=float(parts[4]) * diameter_factor,
                roughness_raw=float(parts[5]),
                minor_loss=float(parts[6]) if len(parts) >= 7 else 0.0,
                status=status,
            )
        )

    pumps: list[Pump] = []
    for line in sections.get("PUMPS", []):
        parts = line.split()
        up = [p.upper() for p in parts]
        if "HEAD" not in up:
            raise NotImplementedError("Only HEAD-curve pumps are supported.")
        speed = 1.0
        if "SPEED" in up:
            speed = float(parts[up.index("SPEED") + 1])
        status = "OPEN"
        if parts[0] in status_overrides and status_overrides[parts[0]].upper() == "CLOSED":
            status = "CLOSED"
        pumps.append(
            Pump(
                link_id=parts[0],
                start_node=parts[1],
                end_node=parts[2],
                curve_id=parts[up.index("HEAD") + 1],
                speed=speed,
                status=status,
            )
        )

    valves: list[Valve] = []
    for line in sections.get("VALVES", []):
        parts = line.split()
        link_id = parts[0]
        status = "OPEN"
        override = status_overrides.get(link_id, "OPEN").upper()
        if override in {"OPEN", "CLOSED"}:
            status = override
        setting_raw = parts[5]
        if override not in {"OPEN", "CLOSED"}:
            setting_raw = override
        valves.append(
            Valve(
                link_id=link_id,
                start_node=parts[1],
                end_node=parts[2],
                diameter_m=float(parts[3]) * diameter_factor,
                valve_type=parts[4].upper(),
                setting_raw=setting_raw,
                minor_loss=float(parts[6]) if len(parts) >= 7 else 0.0,
                status=status,
            )
        )

    junction_index = {node.node_id: i for i, node in enumerate(junctions)}
    tank_index = {node.node_id: i for i, node in enumerate(tanks)}
    reservoir_index = {node.node_id: i for i, node in enumerate(reservoirs)}
    controls = parse_controls(
        sections.get("CONTROLS", []),
        options,
        tank_index,
        junction_index,
        reservoir_index,
    )
    if sections.get("RULES"):
        warnings.warn(
            "EPANET rule-based controls are not yet implemented in the DAE. "
            "Comparisons may degrade for rule-driven networks.",
            stacklevel=2,
        )

    supported_link_ids: list[str] = []
    supported_link_types: list[str] = []
    supported_link_start: list[str] = []
    supported_link_end: list[str] = []
    pipe_index_by_link: dict[str, int] = {}
    pump_index_by_link: dict[str, int] = {}
    valve_index_by_link: dict[str, int] = {}

    for idx, pipe in enumerate(pipes):
        pipe_index_by_link[pipe.link_id] = idx
        supported_link_ids.append(pipe.link_id)
        supported_link_types.append("PIPE")
        supported_link_start.append(pipe.start_node)
        supported_link_end.append(pipe.end_node)

    for idx, pump in enumerate(pumps):
        pump_index_by_link[pump.link_id] = idx
        supported_link_ids.append(pump.link_id)
        supported_link_types.append("PUMP")
        supported_link_start.append(pump.start_node)
        supported_link_end.append(pump.end_node)

    for idx, valve in enumerate(valves):
        valve_index_by_link[valve.link_id] = idx
        supported_link_ids.append(valve.link_id)
        supported_link_types.append("VALVE")
        supported_link_start.append(valve.start_node)
        supported_link_end.append(valve.end_node)

    n_links = len(supported_link_ids)
    Cj = np.zeros((len(junctions), n_links), dtype=float)
    Ct = np.zeros((len(tanks), n_links), dtype=float)
    Cr = np.zeros((len(reservoirs), n_links), dtype=float)

    def add_incidence(node_id: str, col: int, sign: float) -> None:
        if node_id in junction_index:
            Cj[junction_index[node_id], col] += sign
        elif node_id in tank_index:
            Ct[tank_index[node_id], col] += sign
        elif node_id in reservoir_index:
            Cr[reservoir_index[node_id], col] += sign
        else:
            raise KeyError(f"Unknown node {node_id} connected to a supported link.")

    for col, (node_i, node_j) in enumerate(zip(supported_link_start, supported_link_end)):
        add_incidence(node_i, col, +1.0)
        add_incidence(node_j, col, -1.0)

    tank_areas_m2 = np.asarray([tank.area_m2 for tank in tanks], dtype=float)
    node_elevation_m = {j.node_id: j.elevation_m for j in junctions}
    node_elevation_m.update({t.node_id: t.elevation_m for t in tanks})
    node_elevation_m.update({r.node_id: 0.0 for r in reservoirs})

    link_inertance_gain = np.zeros(n_links, dtype=float)
    link_minor_resistance = np.zeros(n_links, dtype=float)
    link_base_open = np.ones(n_links, dtype=float)
    link_base_setting = np.ones(n_links, dtype=float)

    pipe_gains = []
    for col, link_id in enumerate(supported_link_ids):
        if link_id in pipe_index_by_link:
            pipe = pipes[pipe_index_by_link[link_id]]
            gain = gravity * pipe.area_m2 / max(pipe.length_m, 1.0e-9)
            pipe_gains.append(gain)
            link_inertance_gain[col] = gain
            link_minor_resistance[col] = pipe.minor_loss / (2.0 * gravity * pipe.area_m2**2)
            link_base_open[col] = 0.0 if pipe.status == "CLOSED" else 1.0
        elif link_id in pump_index_by_link:
            pump = pumps[pump_index_by_link[link_id]]
            link_base_open[col] = 0.0 if pump.status == "CLOSED" else 1.0
            link_base_setting[col] = pump.speed
        elif link_id in valve_index_by_link:
            valve = valves[valve_index_by_link[link_id]]
            link_base_open[col] = 0.0 if valve.status == "CLOSED" else 1.0
            link_minor_resistance[col] = valve.minor_loss / (2.0 * gravity * valve.area_m2**2)

    gain_ref = float(np.median(pipe_gains)) if pipe_gains else 1.0
    for col, link_type in enumerate(supported_link_types):
        if link_type == "PUMP":
            link_inertance_gain[col] = pump_inertance_scale * gain_ref
        elif link_type == "VALVE":
            link_inertance_gain[col] = valve_inertance_scale * gain_ref

    pump_laws = build_pump_laws(curves, pumps, options.units)
    valve_laws = build_valve_laws(curves, valves, options, gravity, approximate_controlled_valves_as_gpv, controlled_valve_minor_loss)

    return NetworkCase(
        junctions=junctions,
        reservoirs=reservoirs,
        tanks=tanks,
        pipes=pipes,
        pumps=pumps,
        valves=valves,
        patterns=patterns,
        curves=curves,
        options=options,
        time=time,
        controls=controls,
        gravity=gravity,
        kinematic_viscosity_m2s=kinematic_viscosity_water_m2s * options.viscosity_rel,
        absolute_roughness_m=absolute_roughness_m,
        pump_inertance_scale=pump_inertance_scale,
        valve_inertance_scale=valve_inertance_scale,
        supported_link_ids=supported_link_ids,
        supported_link_types=supported_link_types,
        supported_link_start=supported_link_start,
        supported_link_end=supported_link_end,
        pipe_index_by_link=pipe_index_by_link,
        pump_index_by_link=pump_index_by_link,
        valve_index_by_link=valve_index_by_link,
        junction_index=junction_index,
        tank_index=tank_index,
        reservoir_index=reservoir_index,
        node_elevation_m=node_elevation_m,
        Cj=Cj,
        Ct=Ct,
        Cr=Cr,
        tank_areas_m2=tank_areas_m2,
        link_inertance_gain=link_inertance_gain,
        link_minor_resistance=link_minor_resistance,
        link_base_open=link_base_open,
        link_base_setting=link_base_setting,
        pump_laws=pump_laws,
        valve_laws=valve_laws,
    )


def evaluate_pump_head_gain(q_m3s: np.ndarray, speed: np.ndarray, law: PumpLaw) -> np.ndarray:
    speed_eff = np.maximum(speed, 1.0e-6)
    q_nom = np.maximum(q_m3s / speed_eff, 0.0)
    if law.mode == "abg":
        A, B, C = law.coefficients
        h_nom = A - B * q_nom**C
    elif law.mode == "piecewise":
        q_pts = law.q_points_m3s
        h_pts = law.h_points_m
        h_nom = np.interp(q_nom, q_pts, h_pts)
        left_mask = q_nom < q_pts[0]
        right_mask = q_nom > q_pts[-1]
        if q_pts.size >= 2:
            slope_left = (h_pts[1] - h_pts[0]) / max(q_pts[1] - q_pts[0], 1.0e-12)
            slope_right = (h_pts[-1] - h_pts[-2]) / max(q_pts[-1] - q_pts[-2], 1.0e-12)
            h_nom[left_mask] = h_pts[0] + slope_left * (q_nom[left_mask] - q_pts[0])
            h_nom[right_mask] = h_pts[-1] + slope_right * (q_nom[right_mask] - q_pts[-1])
    else:
        raise ValueError(f"Unsupported pump law mode {law.mode}.")
    return speed_eff**2 * h_nom


def pump_link_residual(
    case: NetworkCase,
    col: int,
    q_value: float,
    qdot_value: float,
    head_drop_value: float,
    setting_value: float,
    law: PumpLaw,
) -> float:
    gain = evaluate_pump_head_gain(
        np.asarray([q_value], dtype=float),
        np.asarray([setting_value], dtype=float),
        law,
    )[0]
    net_drive = head_drop_value + gain
    if q_value <= 0.0 and net_drive <= 0.0:
        return q_value
    return qdot_value - case.link_inertance_gain[col] * net_drive


def bellos_friction_factor(q_m3s: np.ndarray, diameter_m: np.ndarray, nu_m2s: float, eps_m: np.ndarray) -> np.ndarray:
    area = math.pi * diameter_m**2 / 4.0
    re = np.maximum(np.abs(q_m3s) * diameter_m / (np.maximum(area, 1.0e-12) * nu_m2s), 1.0)
    a = (1.0 + (re / 2712.0) ** 8.4) ** (-1.0)
    b = (1.0 + (re * eps_m / 150.0 / np.maximum(diameter_m, 1.0e-12)) ** 1.8) ** (-1.0)
    f1 = (64.0 / re) ** a
    f2 = (0.75 * np.log(np.maximum(re / 5.37, 1.0000001))) ** (2.0 * (a - 1.0) * b)
    f3 = (0.88 * np.log(np.maximum(6.82 * diameter_m / np.maximum(eps_m, 1.0e-12), 1.0000001))) ** (
        2.0 * (a - 1.0) * (1.0 - b)
    )
    return f1 * f2 * f3


def pipe_headloss(case: NetworkCase, q_links: np.ndarray) -> np.ndarray:
    loss = np.zeros_like(q_links)
    headloss_model = case.options.headloss
    for col, link_id in enumerate(case.supported_link_ids):
        if link_id not in case.pipe_index_by_link:
            continue
        pipe = case.pipes[case.pipe_index_by_link[link_id]]
        q = q_links[col]
        if headloss_model in {"H-W", "HW"}:
            coeff = 10.67 * pipe.length_m / (max(pipe.roughness_raw, 1.0e-12) ** 1.852 * max(pipe.diameter_m, 1.0e-12) ** 4.8704)
            friction = coeff * abs(q) ** 0.852 * q
        elif headloss_model in {"D-W", "DW"}:
            eps_m = np.asarray([pipe.roughness_raw * roughness_dw_factor_to_m(case.options.units)], dtype=float)
            f = bellos_friction_factor(
                np.asarray([q], dtype=float),
                np.asarray([pipe.diameter_m], dtype=float),
                case.kinematic_viscosity_m2s,
                eps_m,
            )[0]
            coeff = 8.0 * f * pipe.length_m / (case.gravity * math.pi**2 * pipe.diameter_m**5)
            friction = coeff * q * abs(q)
        elif headloss_model in {"C-M", "CM"}:
            coeff = 10.29 * pipe.length_m * max(pipe.roughness_raw, 1.0e-12) ** 2 / max(pipe.diameter_m, 1.0e-12) ** 5.3333333333
            friction = coeff * abs(q) * q
        else:
            raise ValueError(f"Unsupported headloss model {case.options.headloss}.")
        minor = case.link_minor_resistance[col] * q * abs(q)
        loss[col] = friction + minor
    return loss


def demand_pattern_value(case: NetworkCase, pattern_id: str | None, t_s: float) -> float:
    if pattern_id is None or pattern_id not in case.patterns:
        return 1.0
    values = case.patterns[pattern_id]
    if values.size == 0:
        return 1.0
    tau = t_s - case.time.pattern_start_s
    index = int(math.floor(tau / case.time.pattern_timestep_s)) % values.size
    return float(values[index])


def junction_demands(case: NetworkCase, t_s: float) -> np.ndarray:
    d = np.zeros(len(case.junctions), dtype=float)
    for idx, junction in enumerate(case.junctions):
        total = 0.0
        for base_cms, pattern_id in junction.demand_terms:
            total += base_cms * demand_pattern_value(case, pattern_id, t_s)
        d[idx] = case.options.demand_multiplier * total
    return d


def reservoir_heads(case: NetworkCase, t_s: float) -> np.ndarray:
    h = np.zeros(len(case.reservoirs), dtype=float)
    for idx, reservoir in enumerate(case.reservoirs):
        h[idx] = reservoir.base_head_m * demand_pattern_value(case, reservoir.pattern_id, t_s)
    return h


def unpack_state(case: NetworkCase, x: np.ndarray) -> dict[str, np.ndarray]:
    n_links = len(case.supported_link_ids)
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    n_r = len(case.reservoirs)
    return {
        "q": x[:n_links],
        "p_j": x[n_links : n_links + n_j],
        "p_t": x[n_links + n_j : n_links + n_j + n_t],
        "p_r": x[n_links + n_j + n_t : n_links + n_j + n_t + n_r],
    }


def control_measure(case: NetworkCase, state: dict[str, np.ndarray], node_id: str) -> float:
    if node_id in case.tank_index:
        idx = case.tank_index[node_id]
        return float(state["p_t"][idx] - case.node_elevation_m[node_id])
    if node_id in case.junction_index:
        idx = case.junction_index[node_id]
        return float(state["p_j"][idx] - case.node_elevation_m[node_id])
    if node_id in case.reservoir_index:
        idx = case.reservoir_index[node_id]
        return float(state["p_r"][idx])
    raise KeyError(f"Unknown control node {node_id}.")


def apply_setting_to_link(
    case: NetworkCase,
    link_id: str,
    setting_kind: str,
    setting_value: float | None,
    open_mask: np.ndarray,
    link_setting: np.ndarray,
) -> None:
    col = case.supported_link_ids.index(link_id)
    if setting_kind == "OPEN":
        open_mask[col] = 1.0
        if link_id in case.pump_index_by_link:
            link_setting[col] = case.pumps[case.pump_index_by_link[link_id]].speed
    elif setting_kind == "CLOSED":
        open_mask[col] = 0.0
        if link_id in case.pump_index_by_link:
            link_setting[col] = 0.0
    elif setting_kind == "NUMERIC":
        open_mask[col] = 1.0
        if setting_value is not None:
            link_setting[col] = setting_value


def control_triggers_now(rule: ControlRule, t_prev_s: float, state_prev: dict[str, np.ndarray], case: NetworkCase) -> bool:
    tol = 1.0e-9
    if rule.rule_type == "TIMEOFDAY":
        clocktime = (case.time.start_clocktime_s + t_prev_s) % DAY_TO_S
        return abs(clocktime - float(rule.trigger_time_s)) <= tol
    if rule.rule_type == "TIMER":
        return abs(t_prev_s - float(rule.trigger_time_s)) <= tol
    if rule.rule_type == "LOWLEVEL":
        return control_measure(case, state_prev, str(rule.node_id)) <= float(rule.trigger_value_m) + 1.0e-10
    if rule.rule_type == "HIGHLEVEL":
        return control_measure(case, state_prev, str(rule.node_id)) >= float(rule.trigger_value_m) - 1.0e-10
    return False


def runtime_link_controls(case: NetworkCase, t_prev_s: float, x_prev: np.ndarray, open_prev: np.ndarray, setting_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    open_mask = open_prev.copy()
    link_setting = setting_prev.copy()
    state_prev = unpack_state(case, x_prev)
    for rule in case.controls:
        if rule.link_id not in case.supported_link_ids:
            continue
        if control_triggers_now(rule, t_prev_s, state_prev, case):
            apply_setting_to_link(case, rule.link_id, rule.setting_kind, rule.setting_value, open_mask, link_setting)
    return open_mask, link_setting


def rule_would_change_state(case: NetworkCase, rule: ControlRule, open_mask: np.ndarray, link_setting: np.ndarray) -> bool:
    if rule.link_id not in case.supported_link_ids:
        return False
    col = case.supported_link_ids.index(rule.link_id)
    if rule.setting_kind == "OPEN":
        if open_mask[col] < 0.5:
            return True
        if rule.link_id in case.pump_index_by_link:
            default_speed = case.pumps[case.pump_index_by_link[rule.link_id]].speed
            return abs(link_setting[col] - default_speed) > 1.0e-12
        return False
    if rule.setting_kind == "CLOSED":
        return open_mask[col] >= 0.5 or abs(link_setting[col]) > 1.0e-12
    if rule.setting_kind == "NUMERIC":
        return open_mask[col] < 0.5 or abs(link_setting[col] - float(rule.setting_value)) > 1.0e-12
    return False


def time_rule_occurrences(rule: ControlRule, case: NetworkCase, t_prev_s: float, t_next_s: float) -> list[float]:
    occurrences: list[float] = []
    if rule.rule_type == "TIMER":
        t_event = float(rule.trigger_time_s)
        if t_prev_s + EVENT_TOL_S < t_event <= t_next_s + EVENT_TOL_S:
            occurrences.append(t_event)
        return occurrences
    if rule.rule_type != "TIMEOFDAY":
        return occurrences
    base = float(rule.trigger_time_s) - case.time.start_clocktime_s
    n = math.floor((t_prev_s - base) / DAY_TO_S) + 1
    t_event = base + n * DAY_TO_S
    while t_event <= t_next_s + EVENT_TOL_S:
        if t_event > t_prev_s + EVENT_TOL_S:
            occurrences.append(t_event)
        n += 1
        t_event = base + n * DAY_TO_S
    return occurrences


def threshold_event_time(
    rule: ControlRule,
    case: NetworkCase,
    x_prev: np.ndarray,
    x_trial: np.ndarray,
    t_prev_s: float,
    t_next_s: float,
) -> float | None:
    if rule.rule_type not in {"LOWLEVEL", "HIGHLEVEL"}:
        return None
    state_prev = unpack_state(case, x_prev)
    state_trial = unpack_state(case, x_trial)
    value_prev = control_measure(case, state_prev, str(rule.node_id))
    value_trial = control_measure(case, state_trial, str(rule.node_id))
    threshold = float(rule.trigger_value_m)
    crossed = False
    if rule.rule_type == "LOWLEVEL":
        crossed = value_prev > threshold and value_trial <= threshold
    elif rule.rule_type == "HIGHLEVEL":
        crossed = value_prev < threshold and value_trial >= threshold
    if not crossed:
        return None
    denom = value_trial - value_prev
    if abs(denom) <= 1.0e-14:
        alpha = 1.0
    else:
        alpha = float(np.clip((threshold - value_prev) / denom, 0.0, 1.0))
    return t_prev_s + alpha * (t_next_s - t_prev_s)


def first_control_event(
    case: NetworkCase,
    t_prev_s: float,
    x_prev: np.ndarray,
    t_next_s: float,
    x_trial: np.ndarray,
    open_mask: np.ndarray,
    link_setting: np.ndarray,
) -> tuple[float, list[ControlRule]] | None:
    candidates: list[tuple[float, ControlRule]] = []
    for rule in case.controls:
        if not rule_would_change_state(case, rule, open_mask, link_setting):
            continue
        if rule.rule_type in {"TIMER", "TIMEOFDAY"}:
            for t_event in time_rule_occurrences(rule, case, t_prev_s, t_next_s):
                candidates.append((t_event, rule))
        else:
            t_event = threshold_event_time(rule, case, x_prev, x_trial, t_prev_s, t_next_s)
            if t_event is not None:
                candidates.append((t_event, rule))
    if not candidates:
        return None
    t_first = min(t for t, _ in candidates)
    tol = max(EVENT_TOL_S, 1.0e-8 * max(1.0, abs(t_first)))
    rules = [rule for t, rule in candidates if abs(t - t_first) <= tol]
    return t_first, rules


def solve_fixed_control_step(
    case: NetworkCase,
    x_prev: np.ndarray,
    t_prev_s: float,
    t_next_s: float,
    link_open: np.ndarray,
    link_setting: np.ndarray,
) -> tuple[np.ndarray, float]:
    dt = t_next_s - t_prev_s
    if dt <= EVENT_TOL_S:
        return x_prev.copy(), 0.0

    def step_residual(x_next: np.ndarray) -> np.ndarray:
        xdot = (x_next - x_prev) / dt
        return dae_residual(case, t_next_s, x_next, xdot, link_open, link_setting)

    x_guess = x_prev.copy()
    x_guess[: len(case.supported_link_ids)][link_open < 0.5] = 0.0
    n_links = len(case.supported_link_ids)
    n_j = len(case.junctions)
    n_t = len(case.tanks)
    x_guess[n_links + n_j + n_t :] = reservoir_heads(case, t_next_s)

    sol = root(step_residual, x_guess, method="hybr")
    if not sol.success:
        lsq = least_squares(step_residual, x_guess, xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10)
        if not lsq.success:
            raise RuntimeError(
                f"Implicit DAE step failed on [{t_prev_s:.3f}, {t_next_s:.3f}] s: "
                f"{sol.message}; least-squares fallback also failed."
            )
        x_next = lsq.x
    else:
        x_next = sol.x
    residual = float(np.linalg.norm(step_residual(x_next), ord=np.inf))
    return x_next, residual


def advance_with_events(
    case: NetworkCase,
    x_prev: np.ndarray,
    t_prev_s: float,
    t_next_s: float,
    link_open: np.ndarray,
    link_setting: np.ndarray,
    depth: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if depth > MAX_EVENT_DEPTH or t_next_s - t_prev_s <= EVENT_TOL_S:
        x_end, residual = solve_fixed_control_step(case, x_prev, t_prev_s, t_next_s, link_open, link_setting)
        open_end, setting_end = runtime_link_controls(case, t_next_s, x_end, link_open, link_setting)
        return x_end, open_end, setting_end, residual

    x_trial, residual = solve_fixed_control_step(case, x_prev, t_prev_s, t_next_s, link_open, link_setting)
    event = first_control_event(case, t_prev_s, x_prev, t_next_s, x_trial, link_open, link_setting)
    if event is None:
        open_end, setting_end = runtime_link_controls(case, t_next_s, x_trial, link_open, link_setting)
        return x_trial, open_end, setting_end, residual

    t_event, rules = event
    if t_event >= t_next_s - EVENT_TOL_S:
        open_end, setting_end = runtime_link_controls(case, t_next_s, x_trial, link_open, link_setting)
        return x_trial, open_end, setting_end, residual

    if t_event <= t_prev_s + EVENT_TOL_S:
        open_event = link_open.copy()
        setting_event = link_setting.copy()
        for rule in rules:
            apply_setting_to_link(case, rule.link_id, rule.setting_kind, rule.setting_value, open_event, setting_event)
        open_event, setting_event = runtime_link_controls(case, t_prev_s, x_prev, open_event, setting_event)
        return advance_with_events(case, x_prev, t_prev_s, t_next_s, open_event, setting_event, depth + 1)

    x_event, residual_event = solve_fixed_control_step(case, x_prev, t_prev_s, t_event, link_open, link_setting)
    open_event = link_open.copy()
    setting_event = link_setting.copy()
    for rule in rules:
        apply_setting_to_link(case, rule.link_id, rule.setting_kind, rule.setting_value, open_event, setting_event)
    open_event, setting_event = runtime_link_controls(case, t_event, x_event, open_event, setting_event)
    x_end, open_end, setting_end, residual_rest = advance_with_events(
        case,
        x_event,
        t_event,
        t_next_s,
        open_event,
        setting_event,
        depth + 1,
    )
    return x_end, open_end, setting_end, max(residual, residual_event, residual_rest)


def valve_link_residual_and_drop(
    case: NetworkCase,
    col: int,
    q: float,
    qdot: float,
    head_drop: float,
    link_setting: np.ndarray,
    state: dict[str, np.ndarray],
) -> tuple[float, float]:
    link_id = case.supported_link_ids[col]
    valve = case.valves[case.valve_index_by_link[link_id]]
    law = case.valve_laws[link_id]
    sign = 1.0 if q >= 0.0 else -1.0
    if law.mode == "curve_loss":
        q_abs = abs(q)
        h_loss = np.interp(q_abs, law.q_points_m3s, law.h_points_m)
        if law.q_points_m3s.size >= 2:
            slope_left = (law.h_points_m[1] - law.h_points_m[0]) / max(law.q_points_m3s[1] - law.q_points_m3s[0], 1.0e-12)
            slope_right = (law.h_points_m[-1] - law.h_points_m[-2]) / max(law.q_points_m3s[-1] - law.q_points_m3s[-2], 1.0e-12)
            if q_abs < law.q_points_m3s[0]:
                h_loss = law.h_points_m[0] + slope_left * (q_abs - law.q_points_m3s[0])
            if q_abs > law.q_points_m3s[-1]:
                h_loss = law.h_points_m[-1] + slope_right * (q_abs - law.q_points_m3s[-1])
        signed_loss = sign * h_loss
        residual = qdot - case.link_inertance_gain[col] * (head_drop - signed_loss)
        return residual, signed_loss
    if law.mode == "quadratic_loss" or law.mode == "TCV":
        k_loss = law.coefficients[0]
        signed_loss = k_loss * q * abs(q)
        residual = qdot - case.link_inertance_gain[col] * (head_drop - signed_loss)
        return residual, signed_loss
    if law.mode == "constant_drop":
        signed_loss = sign * law.coefficients[0]
        residual = qdot - case.link_inertance_gain[col] * (head_drop - signed_loss)
        return residual, signed_loss
    if law.mode == "fcv":
        return q - law.coefficients[0], math.nan
    if law.mode == "prv":
        downstream = law.end_node
        if downstream in case.junction_index:
            p_down = state["p_j"][case.junction_index[downstream]]
        elif downstream in case.tank_index:
            p_down = state["p_t"][case.tank_index[downstream]]
        else:
            p_down = state["p_r"][case.reservoir_index[downstream]]
        return p_down - case.node_elevation_m.get(downstream, 0.0) - law.coefficients[0], math.nan
    if law.mode == "psv":
        upstream = law.start_node
        if upstream in case.junction_index:
            p_up = state["p_j"][case.junction_index[upstream]]
        elif upstream in case.tank_index:
            p_up = state["p_t"][case.tank_index[upstream]]
        else:
            p_up = state["p_r"][case.reservoir_index[upstream]]
        return p_up - case.node_elevation_m.get(upstream, 0.0) - law.coefficients[0], math.nan
    raise ValueError(f"Unsupported valve law {law.mode}.")


def dae_residual(
    case: NetworkCase,
    t_s: float,
    x: np.ndarray,
    xdot: np.ndarray,
    link_open: np.ndarray,
    link_setting: np.ndarray,
) -> np.ndarray:
    n_links = len(case.supported_link_ids)
    q = x[:n_links]
    qdot = xdot[:n_links]
    state = unpack_state(case, x)
    p_tdot = xdot[n_links + len(case.junctions) : n_links + len(case.junctions) + len(case.tanks)]

    head_drop = case.Cj.T @ state["p_j"] + case.Ct.T @ state["p_t"] + case.Cr.T @ state["p_r"]
    pipe_loss = pipe_headloss(case, q)
    link_res = np.zeros(n_links, dtype=float)

    for col, link_id in enumerate(case.supported_link_ids):
        if link_open[col] < 0.5:
            link_res[col] = q[col]
            continue
        link_type = case.supported_link_types[col]
        if link_type == "PIPE":
            link_res[col] = qdot[col] - case.link_inertance_gain[col] * (head_drop[col] - pipe_loss[col])
        elif link_type == "PUMP":
            law = case.pump_laws[link_id]
            link_res[col] = pump_link_residual(
                case,
                col,
                float(q[col]),
                float(qdot[col]),
                float(head_drop[col]),
                float(link_setting[col]),
                law,
            )
        elif link_type == "VALVE":
            valve_res, _ = valve_link_residual_and_drop(case, col, q[col], qdot[col], head_drop[col], link_setting, state)
            link_res[col] = valve_res
        else:
            raise ValueError(f"Unknown supported link type {link_type}.")

    res_tanks = case.tank_areas_m2 * p_tdot + case.Ct @ q
    res_junctions = -(case.Cj @ q) - junction_demands(case, t_s)
    res_reservoirs = state["p_r"] - reservoir_heads(case, t_s)
    return np.concatenate([link_res, res_junctions, res_tanks, res_reservoirs])


def load_epanet_timeseries(case: NetworkCase, inp_path: Path) -> EpanetSeries:
    ensure_epyt()
    try:
        from epyt import epanet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "EPyT is required. Install it with `python3 -m pip install --target .vendor epyt`."
        ) from exc

    net = epanet(str(inp_path))
    try:
        hts = net.getComputedHydraulicTimeSeries()
        node_ids = list(net.getNodeNameID())
        link_ids = list(net.getLinkNameID())
    finally:
        net.unload()

    q_factor = flow_factor_to_cms(case.options.units)
    h_factor = head_factor_to_m(case.options.units)
    head_m = np.asarray(hts.Head, dtype=float) * h_factor
    flow_m3s = np.asarray(hts.Flow, dtype=float) * q_factor
    head_m[np.abs(head_m) >= EPANET_INVALID_HEAD_THRESHOLD_M] = np.nan
    flow_m3s[np.abs(flow_m3s) >= EPANET_INVALID_FLOW_THRESHOLD_M3S] = np.nan
    return EpanetSeries(
        time_s=np.asarray(hts.Time, dtype=float),
        node_ids=node_ids,
        link_ids=link_ids,
        head_m=head_m,
        flow_m3s=flow_m3s,
        status=np.asarray(hts.Status, dtype=float),
        setting=np.asarray(hts.Setting, dtype=float),
    )


def load_epanet_initial_condition(case: NetworkCase, epanet_series: EpanetSeries) -> np.ndarray:
    node_head = {node_id: epanet_series.head_m[0, idx] for idx, node_id in enumerate(epanet_series.node_ids)}
    link_flow = {link_id: epanet_series.flow_m3s[0, idx] for idx, link_id in enumerate(epanet_series.link_ids)}
    q0 = np.asarray([link_flow.get(link_id, 0.0) for link_id in case.supported_link_ids], dtype=float)
    p_j0 = np.asarray([node_head[node.node_id] for node in case.junctions], dtype=float)
    p_t0 = np.asarray([node_head[node.node_id] for node in case.tanks], dtype=float)
    p_r0 = np.asarray([node_head[node.node_id] for node in case.reservoirs], dtype=float)
    return np.concatenate([q0, p_j0, p_t0, p_r0])


def simulate(case: NetworkCase, x0: np.ndarray, t_grid_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_hist = np.zeros((x0.size, t_grid_s.size), dtype=float)
    residual_hist = np.zeros(t_grid_s.size, dtype=float)
    open_hist = np.zeros((len(case.supported_link_ids), t_grid_s.size), dtype=float)
    setting_hist = np.zeros((len(case.supported_link_ids), t_grid_s.size), dtype=float)

    x_hist[:, 0] = x0
    open0, setting0 = runtime_link_controls(case, float(t_grid_s[0]), x0, case.link_base_open.copy(), case.link_base_setting.copy())
    open_hist[:, 0] = open0
    setting_hist[:, 0] = setting0

    for k in range(t_grid_s.size - 1):
        t_prev = float(t_grid_s[k])
        t_next = float(t_grid_s[k + 1])
        x_prev = x_hist[:, k]
        link_open = open_hist[:, k].copy()
        link_setting = setting_hist[:, k].copy()
        x_next, next_open, next_setting, residual = advance_with_events(
            case,
            x_prev,
            t_prev,
            t_next,
            link_open,
            link_setting,
        )
        x_hist[:, k + 1] = x_next
        open_hist[:, k + 1] = next_open
        setting_hist[:, k + 1] = next_setting
        residual_hist[k + 1] = residual

    return x_hist, residual_hist, open_hist, setting_hist


def state_arrays_in_epanet_order(case: NetworkCase, x_hist: np.ndarray, epanet_series: EpanetSeries) -> tuple[np.ndarray, np.ndarray]:
    state = unpack_state(case, x_hist)
    node_head = {}
    for idx, node in enumerate(case.junctions):
        node_head[node.node_id] = state["p_j"][idx]
    for idx, node in enumerate(case.tanks):
        node_head[node.node_id] = state["p_t"][idx]
    for idx, node in enumerate(case.reservoirs):
        node_head[node.node_id] = state["p_r"][idx]
    link_flow = {link_id: state["q"][idx] for idx, link_id in enumerate(case.supported_link_ids)}
    dae_head = np.column_stack([node_head[node_id] for node_id in epanet_series.node_ids])
    dae_flow = np.column_stack([link_flow[link_id] for link_id in epanet_series.link_ids if link_id in link_flow])
    return dae_head, dae_flow


def epanet_supported_flow(case: NetworkCase, epanet_series: EpanetSeries) -> tuple[np.ndarray, list[str]]:
    cols = []
    ids = []
    for idx, link_id in enumerate(epanet_series.link_ids):
        if link_id in case.supported_link_ids:
            cols.append(epanet_series.flow_m3s[:, idx])
            ids.append(link_id)
    return np.column_stack(cols), ids


def dae_supported_flow(case: NetworkCase, x_hist: np.ndarray, supported_link_ids: list[str]) -> np.ndarray:
    state = unpack_state(case, x_hist)
    link_flow = {link_id: state["q"][idx] for idx, link_id in enumerate(case.supported_link_ids)}
    return np.column_stack([link_flow[link_id] for link_id in supported_link_ids])


def junction_error_arrays(case: NetworkCase, dae_head: np.ndarray, epanet_series: EpanetSeries) -> tuple[np.ndarray, list[str]]:
    cols = []
    ids = []
    node_to_idx = {node_id: i for i, node_id in enumerate(epanet_series.node_ids)}
    for node in case.junctions:
        idx = node_to_idx[node.node_id]
        ep_vals = epanet_series.head_m[:, idx]
        cols.append(np.where(np.isfinite(ep_vals), dae_head[:, idx] - ep_vals, np.nan))
        ids.append(node.node_id)
    return np.column_stack(cols) if cols else np.zeros((dae_head.shape[0], 0)), ids


def total_vector_error(case: NetworkCase, dae_head: np.ndarray, epanet_series: EpanetSeries, dae_flow: np.ndarray, epanet_flow: np.ndarray) -> np.ndarray:
    head_err = np.where(np.isfinite(epanet_series.head_m), dae_head - epanet_series.head_m, np.nan)
    flow_err = np.where(np.isfinite(epanet_flow), dae_flow - epanet_flow, np.nan)
    return np.hstack([head_err, flow_err])


def plot_bundle(
    ax: plt.Axes,
    time_h: np.ndarray,
    values: np.ndarray,
    color: str,
    label: str,
) -> None:
    if values.size == 0:
        return
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    mean = np.mean(values, axis=0)
    if values.shape[0] == 1:
        ax.plot(time_h, mean, color=color, linewidth=1.45, label=label)
        return
    ax.fill_between(time_h, lower, upper, color=color, alpha=0.14, label=f"{label} range")
    ax.plot(time_h, mean, color=color, linewidth=1.55, label=f"{label} mean")


def plot_error_bundle(
    ax: plt.Axes,
    time_h: np.ndarray,
    values: np.ndarray,
    color: str,
    ylabel: str,
    title: str,
    detail_limit: int = PLOT_DETAIL_LIMIT,
    show_legend: bool = True,
) -> None:
    if values.size == 0:
        ax.set_title(title)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        return
    plot_values = values if USE_SIGNED_ERROR_BAND else np.abs(values)
    if values.shape[1] <= detail_limit:
        for j in range(values.shape[1]):
            ax.plot(time_h, plot_values[:, j], color=SOFTGRAY, linewidth=0.7, alpha=0.8)
    else:
        q10, q90 = np.nanquantile(plot_values, [0.1, 0.9], axis=1)
        ax.fill_between(time_h, q10, q90, color=SOFTBLUE, alpha=0.35)
    ax.plot(time_h, np.nanmax(np.abs(values), axis=1), color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(time_h[0], time_h[-1])
    if not USE_SIGNED_ERROR_BAND:
        ax.set_ylim(bottom=0.0)
    style_axes(ax)
    if show_legend:
        ax.legend(frameon=False, loc="best")


def plot_range_bundle(
    ax: plt.Axes,
    time_h: np.ndarray,
    values: np.ndarray,
    color: str,
    ylabel: str,
    title: str,
) -> None:
    if values.size == 0:
        ax.set_title(title)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        return
    finite_values = np.where(np.isfinite(values), values, np.nan)
    lower = np.nanmin(finite_values, axis=1)
    upper = np.nanmax(finite_values, axis=1)
    ax.fill_between(time_h, lower, upper, color=SOFTBLUE, alpha=0.30, label="min-max range")
    ax.plot(time_h, lower, color=color, linewidth=1.0, alpha=0.85, label="minimum")
    ax.plot(time_h, upper, color=color, linewidth=1.25, alpha=0.95, label="maximum")
    ax.set_title(title)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(time_h[0], time_h[-1])
    style_axes(ax)
    ax.legend(frameon=False, loc="best")


def plot_comparison_figures(
    case: NetworkCase,
    epanet_series: EpanetSeries,
    dae_head: np.ndarray,
    dae_flow: np.ndarray,
    epanet_flow: np.ndarray,
    junction_err: np.ndarray,
    supported_link_ids: list[str],
    total_err: np.ndarray,
    output_dir: Path,
    stem: str,
    total_demand_ls: np.ndarray | None = None,
    one_step: bool = False,
    paper_data_dir: Path | None = None,
) -> None:
    """Two-by-two DAE--EPANET comparison figure.

    Panels: (a) time-series overlay of one representative junction head and one
    representative supported link flow; (b) CDF of normalized absolute errors
    at the final timestep over heads and flows; (c) total state error
    (l2/n_x and l_inf/n_x); (d) total network demand.

    If ``paper_data_dir`` is provided, the underlying arrays are also written
    there as a compressed NPZ file alongside the saved PDF.
    """
    time_h = epanet_series.time_s / HOUR_TO_S
    n_x = len(case.supported_link_ids) + len(case.junctions) + len(case.tanks) + len(case.reservoirs)
    total_l2 = np.sqrt(np.nansum(total_err**2, axis=1)) / n_x
    total_linf = np.nanmax(np.abs(total_err), axis=1) / n_x
    flow_err = dae_flow - epanet_flow
    if total_demand_ls is None:
        # total consumption: sum of the positive junction demands (negative demands are inflows)
        total_demand_ls = 1000.0 * np.asarray(
            [np.sum(np.clip(junction_demands(case, float(t_s)), 0.0, None)) for t_s in epanet_series.time_s],
            dtype=float,
        )

    # Pick a representative junction (largest EPANET head range) and link
    # (largest |q| at the final timestep).
    junction_node_ids = [n.node_id for n in case.junctions]
    node_id_to_col = {nid: i for i, nid in enumerate(epanet_series.node_ids)}
    j_cols = [node_id_to_col[nid] for nid in junction_node_ids]
    if j_cols:
        ranges = np.nanmax(epanet_series.head_m[:, j_cols], axis=0) - np.nanmin(epanet_series.head_m[:, j_cols], axis=0)
        j_pick = int(j_cols[int(np.nanargmax(ranges))])
    else:
        j_pick = 0
    rep_node_id = epanet_series.node_ids[j_pick]
    if dae_flow.shape[1] > 0:
        last_q = np.nan_to_num(np.abs(epanet_flow[-1, :]))
        l_pick = int(np.argmax(last_q))
        rep_link_id = supported_link_ids[l_pick]
    else:
        l_pick = 0
        rep_link_id = ""

    head_dae_rep = dae_head[:, j_pick]
    head_epn_rep = epanet_series.head_m[:, j_pick]
    flow_dae_rep = dae_flow[:, l_pick] if dae_flow.shape[1] > 0 else np.full_like(time_h, np.nan)
    flow_epn_rep = epanet_flow[:, l_pick] if epanet_flow.shape[1] > 0 else np.full_like(time_h, np.nan)

    # Per-component relative error envelope over time.
    # Head errors are normalized by the operating head range of each junction,
    # and flow errors by the |q| range of each link over the EPANET window.
    eps_h = 1e-6
    head_scale = np.maximum(
        np.nanmax(np.abs(epanet_series.head_m[:, j_cols]), axis=0) if j_cols else np.array([]),
        eps_h,
    )
    if junction_err.size and head_scale.size:
        head_rel_t = np.abs(junction_err) / head_scale[None, :]
    else:
        head_rel_t = np.zeros_like(time_h)[:, None]
    flow_scale = np.maximum(
        np.nanmax(np.abs(epanet_flow), axis=0) if epanet_flow.size else np.array([]),
        eps_h,
    )
    if flow_err.size and flow_scale.size:
        flow_rel_t = np.abs(flow_err) / flow_scale[None, :]
    else:
        flow_rel_t = np.zeros_like(time_h)[:, None]
    head_med = np.nanmedian(head_rel_t, axis=1)
    head_q10 = np.nanpercentile(head_rel_t, 10, axis=1)
    head_q90 = np.nanpercentile(head_rel_t, 90, axis=1)
    head_max = np.nanmax(head_rel_t, axis=1)
    flow_med = np.nanmedian(flow_rel_t, axis=1)
    flow_q10 = np.nanpercentile(flow_rel_t, 10, axis=1)
    flow_q90 = np.nanpercentile(flow_rel_t, 90, axis=1)
    flow_max = np.nanmax(flow_rel_t, axis=1)

    rc = {
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
        fig.subplots_adjust(left=0.10, right=0.93, bottom=0.09, top=0.94, wspace=0.78, hspace=0.55)

        # (a) Time-series overlay: head (left axis) + flow (right axis).
        ax = axes[0, 0]
        ln1 = ax.plot(time_h, head_epn_rep, color=BLACK, linewidth=1.0, linestyle="--", label="EPANET")
        ln2 = ax.plot(time_h, head_dae_rep, color=MAINBLUE, linewidth=1.5, label="DAE")
        ax.set_xlabel("Time [h]", labelpad=2)
        ax.set_ylabel(f"Junc. {rep_node_id} head [m]", color=MAINBLUE)
        ax.tick_params(axis="y", labelcolor=MAINBLUE)
        ax.set_xlim(time_h[0], time_h[-1])
        style_axes(ax)
        ax2 = ax.twinx()
        ax2.plot(time_h, flow_epn_rep, color=BLACK, linewidth=1.0, linestyle=":")
        ax2.plot(time_h, flow_dae_rep, color=DARKRED, linewidth=1.3)
        ax2.set_ylabel(f"Link {rep_link_id} flow [m$^3$/s]", color=DARKRED)
        ax2.tick_params(axis="y", labelcolor=DARKRED)
        ax.set_title(f"(a) Trajectory Overlay (junc. {rep_node_id}, link {rep_link_id})", pad=5)
        ax.legend(handles=ln1 + ln2, loc="best", frameon=False)

        # (b) Per-component relative-error envelope (median + 10-90% band).
        ax = axes[0, 1]
        head_q10 = np.nanpercentile(head_rel_t, 10, axis=1)
        flow_q10 = np.nanpercentile(flow_rel_t, 10, axis=1)
        ax.fill_between(time_h, np.maximum(head_q10, 1e-12), np.maximum(head_q90, 1e-12),
                        color=MAINBLUE, alpha=0.20, linewidth=0)
        ax.plot(time_h, head_med, color=MAINBLUE, linewidth=1.5, label="Heads")
        ax.fill_between(time_h, np.maximum(flow_q10, 1e-12), np.maximum(flow_q90, 1e-12),
                        color=DARKRED, alpha=0.18, linewidth=0)
        ax.plot(time_h, flow_med, color=DARKRED, linewidth=1.3, linestyle="--", label="Flows")
        ax.set_xlabel("Time [h]", labelpad=2)
        ax.set_ylabel("Relative $|e|$ per component")
        ax.set_xlim(time_h[0], time_h[-1])
        ax.set_yscale("log")
        ax.set_title("(b) Component-wise Error", pad=5)
        ax.legend(frameon=False, loc="best", fontsize=10)
        style_axes(ax)

        # (c) Total state error.
        ax = axes[1, 0]
        ax.plot(time_h, total_l2, color=MAINBLUE, linewidth=1.5, label=r"$\Vert e\Vert_{2}/n_x$")
        ax.plot(time_h, total_linf, color=DARKRED, linewidth=1.2, linestyle="--", label=r"$\Vert e\Vert_{\infty}/n_x$")
        ax.set_xlabel("Time [h]", labelpad=2)
        ax.set_ylabel("Normalized error")
        ax.set_xlim(time_h[0], time_h[-1])
        ax.set_title("(c) Total State Error", pad=5)
        ax.legend(frameon=False, loc="upper left")
        style_axes(ax)

        # (d) Total demand.
        ax = axes[1, 1]
        ax.plot(time_h, total_demand_ls, color=DARKRED, linewidth=1.45)
        ax.fill_between(time_h, 0.0, total_demand_ls, color=SOFTBLUE, alpha=0.18)
        ax.set_xlabel("Time [h]", labelpad=2)
        ax.set_ylabel("Demand [L/s]")
        ax.set_xlim(time_h[0], time_h[-1])
        ax.set_ylim(bottom=0.0)
        ax.set_title("(d) Total Network Demand", pad=5)
        style_axes(ax)

        out_pdf = output_dir / f"{stem}_epanet_comparison.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

    if paper_data_dir is not None:
        paper_data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paper_data_dir / f"{stem}_epanet_comparison.npz",
            time_h=time_h,
            head_dae_rep=head_dae_rep,
            head_epn_rep=head_epn_rep,
            flow_dae_rep=flow_dae_rep,
            flow_epn_rep=flow_epn_rep,
            rep_node_id=np.array(rep_node_id),
            rep_link_id=np.array(rep_link_id),
            head_relerr_med=head_med,
            head_relerr_q10=head_q10,
            head_relerr_q90=head_q90,
            head_relerr_max=head_max,
            flow_relerr_med=flow_med,
            flow_relerr_q10=flow_q10,
            flow_relerr_q90=flow_q90,
            flow_relerr_max=flow_max,
            total_l2_norm=total_l2,
            total_linf_norm=total_linf,
            total_demand_ls=total_demand_ls,
            n_x=np.array(n_x),
        )

        fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.65))
        fig.subplots_adjust(left=0.095, right=0.99, bottom=0.22, top=0.88, wspace=0.34)

        ax = axes[0]
        plot_range_bundle(ax, time_h, dae_head, MAINBLUE, "Head [m]", "All Network Heads")

        ax = axes[1]
        plot_range_bundle(ax, time_h, dae_flow, DARKBLUE, "Flow [m$^3$/s]", "All Link Flows")

        fig.savefig(output_dir / f"{stem}_state_ranges.pdf", bbox_inches="tight")
        plt.close(fig)


def save_dae_csv(case: NetworkCase, t_grid_s: np.ndarray, x_hist: np.ndarray, output_dir: Path, stem: str) -> None:
    state = unpack_state(case, x_hist)
    header = ["time_s", "time_h"]
    cols = [t_grid_s, t_grid_s / HOUR_TO_S]
    for idx, node in enumerate(case.junctions):
        header.append(f"head_junction_{node.node_id}_m")
        cols.append(state["p_j"][idx])
    for idx, node in enumerate(case.tanks):
        header.append(f"head_tank_{node.node_id}_m")
        cols.append(state["p_t"][idx])
    for idx, node in enumerate(case.reservoirs):
        header.append(f"head_reservoir_{node.node_id}_m")
        cols.append(state["p_r"][idx])
    for idx, link_id in enumerate(case.supported_link_ids):
        header.append(f"flow_link_{link_id}_m3s")
        cols.append(state["q"][idx])
    np.savetxt(
        output_dir / f"{stem}_dae_results.csv",
        np.column_stack(cols),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.10f",
    )


def write_summary(
    case: NetworkCase,
    epanet_series: EpanetSeries,
    dae_head: np.ndarray,
    dae_flow: np.ndarray,
    epanet_flow: np.ndarray,
    junction_err: np.ndarray,
    total_err: np.ndarray,
    residual_hist: np.ndarray,
    open_hist: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    flow_err = dae_flow - epanet_flow
    n_x = len(case.supported_link_ids) + len(case.junctions) + len(case.tanks) + len(case.reservoirs)
    total_l2 = np.linalg.norm(total_err, axis=1) / n_x
    total_linf = np.max(np.abs(total_err), axis=1) / n_x

    status_mismatch = 0
    link_id_to_epanet = {link_id: idx for idx, link_id in enumerate(epanet_series.link_ids)}
    for j, link_id in enumerate(case.supported_link_ids):
        if link_id in link_id_to_epanet:
            ep_status = (epanet_series.status[:, link_id_to_epanet[link_id]] > 0.0).astype(float)
            dae_status = open_hist[j, :]
            status_mismatch += int(np.count_nonzero(ep_status != dae_status))

    lines = [
        f"DAE vs EPANET summary for {stem}",
        "",
        f"State dimension n_x: {n_x}",
        f"Nodes: {len(case.junctions)} junctions, {len(case.tanks)} tanks, {len(case.reservoirs)} reservoirs",
        f"Links: {len(case.supported_link_ids)} supported links "
        f"({sum(t == 'PIPE' for t in case.supported_link_types)} pipes, "
        f"{sum(t == 'PUMP' for t in case.supported_link_types)} pumps, "
        f"{sum(t == 'VALVE' for t in case.supported_link_types)} valves)",
        f"Controls parsed: {len(case.controls)}",
        f"Headloss model: {case.options.headloss}",
        f"Flow units: {case.options.units}",
        f"EPANET time points: {epanet_series.time_s.size}",
        f"Max DAE solver residual infinity norm: {np.max(residual_hist):.6e}",
        "",
        "Total vector error (normalized by n_x):",
        f"  max ||e(t)||_2 / n_x = {np.max(total_l2):.6e}",
        f"  mean ||e(t)||_2 / n_x = {np.mean(total_l2):.6e}",
        f"  max ||e(t)||_inf / n_x = {np.max(total_linf):.6e}",
        "",
        "Flow error over all supported links:",
        f"  max abs error = {np.max(np.abs(flow_err)):.6e}",
        f"  mean abs error = {np.mean(np.abs(flow_err)):.6e}",
        f"  rms error = {np.sqrt(np.mean(flow_err**2)):.6e}",
        "",
        "Junction-head error:",
        f"  max abs error = {np.max(np.abs(junction_err)) if junction_err.size else 0.0:.6e}",
        f"  mean abs error = {np.mean(np.abs(junction_err)) if junction_err.size else 0.0:.6e}",
        f"  rms error = {np.sqrt(np.mean(junction_err**2)) if junction_err.size else 0.0:.6e}",
        "",
        f"Control/status mismatches against EPANET status samples: {status_mismatch}",
    ]
    (output_dir / f"{stem}_comparison_summary.txt").write_text("\n".join(lines))


def select_inp_interactively(default_dir: Path) -> Path:
    candidates = sorted(default_dir.glob("*.inp")) + sorted(default_dir.glob("*.INP"))
    if not candidates:
        raise FileNotFoundError(f"No INP files found in {default_dir}.")
    print("Available INP files:")
    for idx, path in enumerate(candidates, start=1):
        print(f"  {idx}. {path.name}")
    while True:
        choice = input("Pick an INP file by number: ").strip()
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        print("Please enter a valid number.")


def resolve_inp_path(inp_arg: Path | None) -> Path:
    if inp_arg is not None:
        if inp_arg.exists():
            return inp_arg
        candidate = ROOT / inp_arg
        if candidate.exists():
            return candidate
        candidate = NETWORKS_DIR / inp_arg
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find INP file: {inp_arg}")
    return select_inp_interactively(NETWORKS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="General full-network DAE simulator and EPANET comparison tool.")
    parser.add_argument("--inp", type=Path, default=None, help="Path to the EPANET INP file. If omitted, an interactive picker is shown.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--pump-inertance-scale", type=float, default=1.0)
    parser.add_argument("--valve-inertance-scale", type=float, default=1.0)
    parser.add_argument("--approximate-controlled-valves", action="store_true", help="Approximate unsupported valve types with GPV-style quadratic-loss surrogates.")
    args = parser.parse_args()

    script_inp = Path(SCRIPT_INP_FILE) if SCRIPT_INP_FILE else None
    inp_path = resolve_inp_path(args.inp if args.inp is not None else script_inp)
    case = build_network_case(
        inp_path,
        pump_inertance_scale=args.pump_inertance_scale,
        valve_inertance_scale=args.valve_inertance_scale,
        approximate_controlled_valves_as_gpv=args.approximate_controlled_valves,
    )
    epanet_series = load_epanet_timeseries(case, inp_path)
    x0 = load_epanet_initial_condition(case, epanet_series)
    t_grid_s = epanet_series.time_s.copy()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x_hist, residual_hist, open_hist, setting_hist = simulate(case, x0, t_grid_s)
    stem = inp_path.stem.lower()

    save_dae_csv(case, t_grid_s, x_hist, output_dir, stem)

    dae_head, _ = state_arrays_in_epanet_order(case, x_hist, epanet_series)
    epanet_flow, supported_link_ids = epanet_supported_flow(case, epanet_series)
    dae_flow = dae_supported_flow(case, x_hist, supported_link_ids)
    junction_err, _ = junction_error_arrays(case, dae_head, epanet_series)
    total_err = total_vector_error(case, dae_head, epanet_series, dae_flow, epanet_flow)

    plot_comparison_figures(
        case,
        epanet_series,
        dae_head,
        dae_flow,
        epanet_flow,
        junction_err,
        supported_link_ids,
        total_err,
        output_dir,
        stem,
    )

    write_summary(case, epanet_series, dae_head, dae_flow, epanet_flow, junction_err, total_err, residual_hist, open_hist, output_dir, stem)

    print(f"Saved DAE CSV to {output_dir / f'{stem}_dae_results.csv'}")
    print(f"Saved six-panel comparison figure to {output_dir / f'{stem}_epanet_comparison.pdf'}")
    print(f"Saved state-range figure to {output_dir / f'{stem}_state_ranges.pdf'}")
    print(f"Saved summary to {output_dir / f'{stem}_comparison_summary.txt'}")


if __name__ == "__main__":
    main()
