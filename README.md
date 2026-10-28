# HydroDAE-WDN

**Companion code and data for:**

> **Stability and Controllability Margins of Water Network Hydraulics**
>
> A. F. Taha and M. H. Kazma
>
> Submitted to *Water Resources Research*, 2026 (preprint: <https://arxiv.org/abs/2605.01707>)

---

## Overview

This repository contains the Python code and pre-computed results for reproducing the figures and tables of the paper. The code writes the hydraulics of a water distribution network, read from an EPANET input file, as a differential algebraic equation (DAE) model with pipe inertia and tank storage. It compares the DAE with EPANET, computes the stability and controllability margins of the linearized DAE, and selects where to add pumps with a greedy algorithm on the controllability Gramian.

Six networks are included:

| Network | Junctions | Tanks | Reservoirs | Pipes | Pumps | Valves |
|---------|-----------|-------|------------|-------|-------|--------|
| **Threenodes** | 1 | 1 | 1 | 1 | 1 | 0 |
| **Net1** | 9 | 1 | 1 | 12 | 1 | 0 |
| **Net3** | 92 | 3 | 2 | 117 | 2 | 0 |
| **Anytown** | 19 | 0 | 3 | 40 | 1 | 0 |
| **MOD** | 268 | 0 | 4 | 317 | 0 | 0 |
| **BELL_CL1** | 121 | 2 | 0 | 168 | 0 | 1 |

## Repository Structure

```
HydroDAE-WDN/
├── src/
│   ├── dae_network.py                 # Hydraulic DAE built from an EPANET INP file
│   ├── equilibrium.py                 # Operating point of the DAE with fixed controls
│   ├── linear_dae.py                  # Linearized DAE, stability and PBH margins, Algorithms 1 and 2
│   │
│   ├── fixed_control.py               # 1 h DAE with fixed controls versus EPANET (Section 4.1)
│   ├── smoothed_control.py            # 24 h DAE with smoothed controls versus EPANET (Section 4.2)
│   ├── pump_trip_experiment.py        # Pump trip on Net3 (Section 4.2)
│   ├── laplacian_stability.py         # Weighted Laplacian stability and energy decay (Section 4.3)
│   ├── controllability_margins.py     # Controllability margin and pipe sensitivities (Sections 4.4, 4.5)
│   ├── operating_point_margins.py     # Margins of Net3 under the demand scenario (Section 4.5)
│   ├── pump_allocation.py             # Greedy pump allocation on Net3 and MOD (Section 4.6)
│   ├── pump_allocation_mod_anchor.py  # MOD gains relative to pipe 335 (Section 4.6)
│   │
│   ├── smoothing_parameter.py         # Figure 1
│   ├── plot_inp_networks.py           # Figure 2
│   ├── make_figures.py                # Figures 4 to 7 from results/
│   ├── make_tables.py                 # Tables 1 and 2 from results/
│   │
│   └── Networks/                      # INP files of the six networks
│
├── results/                           # Pre-computed output, one folder per study
│   └── tables/                        # Values of Tables 1 and 2 (.csv)
└── figures/                           # Figures written by the scripts
```

## Requirements

- **Python 3.9** or later
- The packages in `requirements.txt` (NumPy, SciPy, Matplotlib, NetworkX, WNTR, and EPyT, which includes the EPANET library)

```bash
pip install -r requirements.txt
```

## Quick Start

### Redraw the figures from the pre-computed results

The `results/` directory contains the arrays plotted in the paper. From the repository root,

```bash
python3 src/make_figures.py
```

writes Figures 4 to 7 to `figures/` in a few seconds without running any simulation, and

```bash
python3 src/make_tables.py
```

writes the values of Tables 1 and 2 to `results/tables/`. Any result file can also be loaded directly:

```python
import numpy as np
d = np.load("results/pump_allocation/pump_allocation_net3.npz")
print(d["greedy_ids"])   # links selected by the greedy algorithm
```

### Rerun the studies

Each script writes its output to `results/` and a text summary to `src/output/`. Run them from the repository root, then run `make_figures.py` and `make_tables.py` again.

| Paper | Command |
|-------|---------|
| Figure 1 | `python3 src/smoothing_parameter.py` |
| Figure 2 | `python3 src/plot_inp_networks.py` |
| Table 1 | `python3 src/fixed_control.py --inp src/Networks/Net3.inp` and `python3 src/smoothed_control.py --inp src/Networks/Net3.inp`, for each of the six networks |
| Figure 4 | the two scripts above on Net3 and BELL_CL1, and `python3 src/pump_trip_experiment.py` |
| Figure 5 | `python3 src/laplacian_stability.py` and `python3 src/linear_dae.py` |
| Table 2 | `python3 src/linear_dae.py` and `python3 src/controllability_margins.py` |
| Figure 6 | `python3 src/operating_point_margins.py` and `python3 src/controllability_margins.py` |
| Figure 7 | `python3 src/pump_allocation.py` and `python3 src/pump_allocation_mod_anchor.py` |

Figure 3 is a diagram drawn in LaTeX.

## Citation

If you use this code, please cite the paper and the code.

```bibtex
@article{taha2026stability,
  title   = {Stability and Controllability Margins of Water Network Hydraulics},
  author  = {Taha, A. F. and Kazma, M. H.},
  journal = {Water Resources Research},
  year    = {2026},
  note    = {Under review, arXiv:2605.01707}
}

@misc{kazma2026hydrodae,
  title     = {{HydroDAE-WDN}: Stability and controllability margins of water network hydraulics (Version 1.0)},
  author    = {Kazma, M. H. and Taha, A. F.},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23734032}
}
```

## Data Acknowledgment

Net1 and Net3 are example networks of EPANET (<https://www.epa.gov/water-research/epanet>). Anytown, MOD, and BELL_CL1 are from the KIOS Research EPANET Benchmarks collection (<https://github.com/KIOS-Research/EPANET-Benchmarks>).

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE) for details.
