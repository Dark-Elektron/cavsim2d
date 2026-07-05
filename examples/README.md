# cavsim2d examples

Runnable, headless example scripts that exercise each part of cavsim2d and save
their results and plots to disk — the same workflows as the notebooks, but as
plain scripts you can run from a terminal.

## Running

From this `examples/` directory, in the environment where NGSolve is installed:

```bash
python 01_eigenmode.py          # a single example
python run_all.py               # every example, in order
python run_all.py 1 2 5         # only examples 1, 2 and 5
```

The scripts add the repo root to `sys.path`, so they work whether or not
cavsim2d is `pip install`-ed.

## Where results go

Everything is written under `SIM_ROOT` (set in [`_common.py`](_common.py),
default `C:\Users\Soske\Documents\git_projects\cavsim2d_simulations`):

```
<SIM_ROOT>/examples/<example>/
    cavities/<name>/        geometry, meshes, eigenmode/wakefield results, qois.json
    plots/                  saved PNG figures
```

Edit `SIM_ROOT` in `_common.py` to point somewhere else.

## The examples

| # | Script | What it demonstrates |
|---|--------|----------------------|
| 1 | `01_eigenmode.py` | Fundamental (monopole) eigenmode on 1- and 2-cell cavities; QOIs, geometry / field / axis-field / dispersion / comparison plots. |
| 2 | `02_eigenmode_mpole.py` | Higher-order-mode passbands: monopole + dipole + quadrupole on one mesh via the `'polarisation'` config; per-polarisation QOIs and dipole field envelopes. |
| 3 | `03_tune.py` | Tune the equator radius `Req` so the fundamental hits a target frequency; before/after geometry. |
| 4 | `04_wakefield.py` | ABCI wakefield/impedance analysis; longitudinal & transverse impedance and wake potentials. **Windows + bundled `ABCI.exe` only** (self-skips otherwise). |
| 5 | `05_eigenmode_uq.py` | Uncertainty quantification: propagate 5% Gaussian geometry errors through the eigenmode solve (Stroud-3 quadrature); mean/std of the figures of merit. |
| 6 | `06_optimisation.py` | Small multi-objective shape optimisation (minimise `Epk/Eacc` and `Bpk/Eacc` at a fixed frequency); Pareto front, history and convergence plots. |

## Requirements

- NGSolve + gmsh (all examples).
- The bundled `cavsim2d/solvers/ABCI/ABCI.exe` for example 4 (Windows, or via
  `wine` elsewhere).
