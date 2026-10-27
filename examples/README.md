# cavsim2d examples

Runnable, headless example scripts that exercise each part of cavsim2d and save
their results and plots to disk: the same workflows as the documentation's
example notebooks, as plain scripts you can run from a terminal.

## Running

From this `examples/` directory, in an environment with cavsim2d installed:

```bash
python 01_eigenmode.py          # a single example
python run_all.py               # every example, in order
python run_all.py 1 2 5         # only examples 1, 2 and 5
```

The scripts add the repository root to `sys.path`, so they also work from a
`git clone` that is not `pip install`-ed.

## Where results go

Everything is written under `examples/output/`, or under the folder named by the
`CAVSIM2D_EXAMPLES_DIR` environment variable if you set it:

```text
<output>/<example>/
    cavities/<name>/        geometry, meshes, eigenmode/wakefield results, qois.json
    plots/                  saved PNG figures
```

## The examples

| # | Script | What it demonstrates |
|---|--------|----------------------|
| 1 | `01_eigenmode.py` | Monopole eigenmode of a 1- and a 2-cell cavity: figures of merit, geometry, field, on-axis field, dispersion and comparison plots. |
| 2 | `02_eigenmode_mpole.py` | Monopole, dipole and quadrupole modes on one mesh via `'polarisation'`; per-polarisation results and dipole field envelopes. |
| 3 | `03_tune.py` | Tune the equator radius `Req` so the fundamental hits a target frequency, solve the tuned cavity, and compare the geometry before and after. |
| 4 | `04_wakefield.py` | ABCI wakefield: longitudinal and transverse impedance, wake potential and loss-factor spectrum. Needs ABCI, which cavsim2d does not ship: download it from [KEK](https://abci.kek.jp/abci.htm) and set `CAVSIM2D_ABCI` to the executable or its folder. The script skips itself when ABCI is not found. |
| 5 | `05_eigenmode_uq.py` | Uncertainty quantification: Gaussian tolerances on the cell's ellipse semi-axes, propagated by Stroud-3 quadrature to the figures of merit. |
| 6 | `06_optimisation.py` | A small multi-objective optimisation: every candidate is tuned to the target frequency, then Epk/Eacc and Bpk/Eacc are minimised; Pareto front, history and convergence plots. |
