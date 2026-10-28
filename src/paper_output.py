"""Shared helpers for writing the cached plot data into ``results/``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
OUTPUT_PAPER = REPO_ROOT / "results"


def paper_subdir(name: str) -> Path:
    p = OUTPUT_PAPER / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_npz_and_csv(path_no_ext: Path, **arrays) -> None:
    """Write a compressed NPZ plus a flat CSV header listing array shapes.

    The CSV is for human inspection; the NPZ is the canonical replay file.
    """
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path_no_ext.with_suffix(".npz"), **arrays)
    lines = ["# array, shape, dtype"]
    for k, v in arrays.items():
        a = np.asarray(v)
        lines.append(f"{k}, {tuple(a.shape)}, {a.dtype}")
    path_no_ext.with_suffix(".manifest.txt").write_text("\n".join(lines) + "\n")


def save_max_errors(path: Path, max_head_err_m: float, max_flow_err_m3s: float) -> None:
    """Write the maximum junction head and link flow errors of a DAE versus EPANET run (Table 1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"max_junction_head_error_m,max_link_flow_error_m3s\n{max_head_err_m:.6e},{max_flow_err_m3s:.6e}\n")
