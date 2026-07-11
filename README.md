![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-beta-orange)

# cavsim2d

`cavsim2d` is a Python toolkit for designing and analysing 2D axisymmetric RF
structures. It provides, through one small object-oriented API:

- **Eigenmode analysis** — fundamental (monopole) *and* higher-order-mode
  (dipole, quadrupole, sextupole, …) passbands, with the usual figures of merit
  (frequency, R/Q, G, Q, Epk/Eacc, Bpk/Eacc, field flatness, cell-to-cell
  coupling).
- **Frequency tuning** to a target frequency.
- **Wakefield / impedance analysis** via the ABCI solver.
- **Uncertainty quantification (UQ)** on any of the above.
- **Multi-objective shape optimisation.**
- **Plotting and comparison** helpers for all of the above.

The eigenmode solver is built on [NGSolve](https://ngsolve.org); wakefield
analysis uses the free [ABCI](https://abci.kek.jp/abci.htm) code.

## Cavity types and what is supported

`EllipticalCavity` is the fully-supported, exercised path. Other geometry
classes exist at varying maturity:

| Cavity type | Eigenmode | Tuning | Wakefield | UQ / Optimisation |
|---|---|---|---|---|
| `EllipticalCavity` | ✅ | ✅ | ✅ | ✅ |
| `EllipticalCavityFlatTop` | ✅ | ✅ | ✅ | ✅ |
| `Pillbox` | ✅ | ✅ | ✅ | ✅ |
| `SplineCavity` | ✅ | ✅² | ✅ | ✅² |
| `RFGun` | ✅¹ | ✅ | ✅ | ✅ |

✅ supported and exercised · ¹ ² see notes.

Every type builds its mesh from the same `Profile` blueprint, implements a single
`rebuild()` hook, and declares its own tunable parameters via `tune_variables()`.
Adding a geometry therefore needs no edit to any central list:

```python
tesla.tune_variables()     # {'A', 'B', 'a', 'b', 'Ri', 'L', 'Req'}
gun.tune_variables()       # {'y1', 'R2', 'T2', 'L3', ..., 'R6', ..., 'x'}
spline.tune_variables()    # {'p0_z', 'p0_r', ..., 'p5_z', 'p5_r'}
```

Any of those names can be a tune variable (`tune_config['cell_type']`) or a UQ /
optimisation variable. A geometry imported from a mesh or CAD file has no
parameters, and says so rather than reporting an unknown variable name. Unless
noted, examples below use `EllipticalCavity`.

¹ RF-gun eigenmode gives correct frequencies and fields; the length-normalised
QOIs (`Eacc`, `Epk/Eacc`) default to the on-axis field extent as the active
length (override with `eigenmode_config['normalization_length']`, mm). Its
`Epk/Eacc` does not converge under mesh refinement — the cathode corner is a
field singularity, so treat that number as mesh-dependent.

² A `SplineCavity` is parameterised by control points rather than by scalars, so
its handles are the point *coordinates*: `'p3_r'` moves the radial coordinate of
`p3` and leaves `p3_z` alone.

---

## Installation

Clone the repository and install it (editable install recommended while the API
is stabilising):

```bash
git clone https://github.com/Dark-Elektron/cavsim2d.git
cd cavsim2d
pip install -e .
```

This installs the core dependencies (NGSolve, gmsh, numpy, scipy, pandas,
matplotlib, …). For the example notebooks and the Jupyter banner, also install
the optional extra:

```bash
pip install -e ".[jupyter]"
```

Requirements: **Python ≥ 3.10**. NGSolve provides the finite-element backend and
is easiest to obtain in a conda environment (`conda install -c ngsolve ngsolve`)
if the pip wheel is not available for your platform.

### Third-party code — ABCI (wakefield only)

Wakefield analysis uses the ABCI electromagnetic time-domain solver. A Windows
build of `ABCI.exe` ships in `cavsim2d/solvers/ABCI/`. If you need to (re)install
it, download the latest release from [ABCI](https://abci.kek.jp/abci.htm) and
place the executable at:

```
cavsim2d/solvers/ABCI/ABCI.exe
```

`ABCI.exe` is a Windows binary. On Windows it runs directly; on Linux/macOS it is
launched through [`wine`](https://www.winehq.org) if that is installed. If the
executable or `wine` is missing, wakefield runs fail early with a clear message —
eigenmode, tuning, UQ and optimisation do not need ABCI.

> [!TIP]
> `pip install pprintpp` gives nicer nested-dict printing for the quantities of
> interest, but is not required.

---

## Quickstart

Everything is driven by two classes: **`Cavities`**, a container that owns a
project folder, and **`Cavity`** (here `EllipticalCavity`), a single cavity.
Results are written under the project folder and cached on the objects.

```python
import pprint
pp = pprint.PrettyPrinter(indent=4)
from cavsim2d.cavity import Cavities, EllipticalCavity

# A project folder — all results and geometry are written here.
cavs = Cavities(r'/path/to/my_project')

# A 1-cell TESLA mid-cell:  A, B, a, b, Ri, L, Req  [mm]
midcell = [42, 42, 12, 19, 35, 57.7, 103.353]
tesla = EllipticalCavity(1, midcell, midcell, midcell, beampipe='none')
cavs.add_cavity([tesla], names=['TESLA'])

# Fundamental-mode eigenmode analysis.
cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
pp.pprint(cavs.eigenmode_qois)
```

For the 1-cell TESLA cavity this reports `freq ≈ 1300 MHz`, `R/Q ≈ 113 Ω`,
`G ≈ 271 Ω`, `Epk/Eacc ≈ 2.04`, `Bpk/Eacc ≈ 4.16 mT/(MV/m)`.

> [!TIP]
> Prefer runnable scripts to prose? The [`examples/`](examples/) directory has
> one self-contained script per feature (`01_eigenmode.py` … `06_optimisation.py`
> and `run_all.py`) that writes results and plots to disk. They are the quickest
> way to see each workflow end to end.

### Project folder layout

```
my_project/
└── cavities/
    └── TESLA/
        ├── geometry/            geodata.geo, snapshot
        ├── eigenmode/
        │   ├── monopole/        qois.json, qois_all_modes.json, fields, mesh
        │   ├── dipole/          (if requested)
        │   └── quadrupole/      (if requested)
        ├── tuned/               tuned cavity + tune_info/
        ├── wakefield/
        │   ├── longitudinal/    ABCI output
        │   └── transversal/
        └── uq/                  UQ nodes and statistics
```

---

## Eigenmode analysis

`run_eigenmode` accepts a configuration dictionary; every key is optional:

| key | meaning | default |
|-----|---------|---------|
| `processes` | parallel worker processes | 1 |
| `rerun` | recompute even if results exist | True |
| `boundary_conditions` | end-wall BCs, e.g. `'mm'` (magnetic-magnetic) | `'mm'` |
| `polarisation` | azimuthal order(s) to solve — see below | `'monopole'` |
| `n_modes` | physical modes per m-pole solve | `n_cells + 2` |
| `mesh_config` | `{'h': ..., 'p': ...}` mesh size / order | h=20, p=3 |
| `conductivity` | wall conductivity [S/m] for Q/Ploss (normal conductor) | copper, 5.96e7 |
| `surface_resistance` | fixed surface resistance [Ω] (e.g. SRF) — overrides `conductivity` | — |
| `uq_config` | enable uncertainty quantification (see below) | — |

> [!TIP]
> Q, Rsh and wall power depend on the surface resistance. By default the walls
> are copper; pass `conductivity` for another normal conductor, or
> `surface_resistance` (a fixed Rs, e.g. `1e-8` Ω) to model an SRF cavity. The
> geometric factor `G` is material-independent.

Access results either on the container or per cavity:

```python
cavs.eigenmode_qois            # {name: {qoi: value}} for the fundamental mode
tesla.get_eigenmode_qois()     # populate tesla.eigenmode_qois, tesla.freq, ...
tesla.eigenmode.qois           # fundamental-mode QOIs (solver-object API)
tesla.eigenmode.modes          # list of all computed modes
```

Compare several cavities:

```python
cavs.add_cavity([EllipticalCavity(1,
        [53.58, 36.58, 8.08, 9.84, 35, 57.7, 98.27],  # a re-entrant mid-cell
        [53.58, 36.58, 8.08, 9.84, 35, 57.7, 98.27],
        [53.58, 36.58, 8.08, 9.84, 35, 57.7, 98.27], beampipe='none')], ['reentrant'])
cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
cavs.plot_compare_fm_bar()
```

### Visualising the mesh and fields

Meshes and fields belong to a **`Cavity`** (not the container), so index or name
the cavity:

```python
cavs['TESLA'].plot_mesh()
cavs['TESLA'].plot_fields(mode=1, which='E')     # |E| of the fundamental
cavs['TESLA'].plot_fields(mode=1, which='H')     # |H|
cavs['TESLA'].plot_axis_field()                  # on-axis Ez and field flatness
```

All `plot*` methods return a `matplotlib` axes/figure so you can restyle or save
them, and share one warm-cast, publication-oriented palette and font (STIX;
`cavsim2d.utils.style`, `apply_style()` to make it your session default). For
multi-cell cavities, `plot_dispersion()` draws the Brillouin diagram: it overlays
every computed polarisation, colour-coded, and shows all modes (grouped into
passbands of `n_cells`). Options:

- `pol=` — a name ('dipole'), a list, or `None` for all computed polarisations;
- `bands=` — 1-based passband selection; a flat `[1, 2]` for all polarisations or
  a nested `[[1, 2], [1]]` for one selection per polarisation;
- `break_axis=` (default `True`) — splits the y-axis where passbands are far
  apart so the empty space between them is not wasted; the split points are found
  automatically from the frequency grouping. Pass `breaks=[(1350, 1600), (1900,
  2350)]` to place them by hand (each is an approximate `(low, high)` gap).
- `light_line=` (default `True`) — overlays the speed-of-light line `f = c·μ/(2π·d)`.
  Because the diagram is a reduced (folded) zone, it appears as a triangle wave
  crossing every band; the accelerating π-mode sits on it. Skipped for geometries
  with no defined cell period.

---

## Higher-order-mode (m-pole) eigenmodes

Dipole, quadrupole, and higher azimuthal modes ($m \ge 1$) are solved on the same
2D meridian mesh via an $H(\mathrm{curl}) \times H^1$ product-space formulation.
Select them with the `polarisation` key (names or azimuthal mode numbers, a
single value or a list):

```python
cavs.run_eigenmode({
    'processes': 1,
    'rerun': True,
    'boundary_conditions': 'mm',
    'polarisation': ['monopole', 'dipole', 'quadrupole'],  # or [0, 1, 2]
    'n_modes': 4,
})

# Per-polarisation results:
tesla.eigenmode.mpole_qois('dipole')          # all dipole modes, by index
tesla.eigenmode.mpole_modes('quadrupole')     # list of EigenmodeResult
tesla.plot_fields(mode=0, which='E', pol='dipole')      # in-plane |E| envelope
tesla.plot_fields(mode=0, which='Ephi', pol='dipole')   # azimuthal envelope
```

Each polarisation is written to its own folder (`eigenmode/monopole/`,
`eigenmode/dipole/`, …). Reruns are per polarisation: rerunning the dipole never
overwrites monopole results, and `rerun=False` only solves the polarisations
whose results are missing.

For $m \ge 1$ the accelerating quantities are the **transverse** analogues: since
$E_z \sim r^m$ vanishes on axis, the longitudinal voltage is evaluated off-axis
and converted to a transverse kick voltage via Panofsky–Wenzel. The QOI keys
`Vacc`/`Eacc`/`R/Q` hold these transverse values, and `R/Q_t [Ohm/m^(2(m-1))]`
is normalised so it is independent of the evaluation radius (equal to `R/Q` for
the dipole).

**Conventions to note.** The headline `qois.json` for an m-pole solve is the
**lowest** mode (index 0), whereas the monopole headline is the accelerating
π-mode (index `n_cells`). Use `mpole_qois(pol)` for the full set. The `kcc [%]`
cell-to-cell coupling for $m \ge 1$ reuses the monopole passband formula
(relative to mode index 1) and is of limited physical meaning for a deflecting
passband. Plot the passband with `cav.plot_dispersion(pol='dipole')`.

M-pole QOIs are not yet available as UQ or optimisation objectives (those still
operate on the monopole figures of merit).

---

## Cavity tuning

Tuning drives a chosen geometry parameter until the fundamental hits a target
frequency. Give a target `freqs` and a `cell_type` mapping (cell → parameter to
vary); a nested `eigenmode_config` controls the solves.

```python
cavs = Cavities(r'/path/to/my_project')
midcell = [42, 42, 12, 19, 35, 57.7, 100]        # deliberately wrong Req
tesla = EllipticalCavity(1, midcell, midcell, midcell, beampipe='none')
cavs.add_cavity([tesla], ['TESLA'])

cavs.run_tune({
    'freqs': 1300,                               # MHz
    'cell_type': {'mid-cell': 'Req'},            # vary Req of the mid-cell
    'processes': 1,
    'rerun': True,
    'eigenmode_config': {'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'},
})

pp.pprint(tesla.tune.qois)      # tuned parameters and achieved frequency
tuned = tesla.tuned             # the tuned Cavity (its own folder + results)
```

`Req` converges to ≈ 103.35 mm and the frequency to 1300 MHz. The tuned cavity is
a full `Cavity` living under `<cavity>/tuned/`, so you can run eigenmode/wakefield
on it directly (`tesla.tuned.run_eigenmode(...)`).

---

## Wakefield analysis

```python
cavs = Cavities(r'/path/to/my_project')
midcell   = [42, 42, 12, 19, 35, 57.7, 103.353]
endcell_l = [40.34, 40.34, 10, 13.5, 39, 55.716, 103.353]
endcell_r = [42, 42, 9, 12.8, 39, 56.815, 103.353]
tesla = EllipticalCavity(9, midcell, endcell_l, endcell_r, beampipe='both')
cavs.add_cavity([tesla], ['TESLA'])

cavs.run_wakefield({
    'processes': 1,
    'rerun': True,
    'MROT': 'both',        # 'monopole'/'longitudinal' (0), 'dipole'/'transverse' (1), 'both' (2)
    'wakelength': 50,      # metres
    'bunch_length': 25,    # mm
})
```

> [!NOTE]
> In **wakefield** configs the beam mode is `MROT` (`'polarisation'` is accepted
> as a deprecated alias). This is *not* the same as `polarisation` in an
> **eigenmode** config, which selects the azimuthal mode order.

ABCI requires a beam pipe at **each** end of the structure, of at least five mesh
lengths. Any end that does not already have one is given a pipe of **three times
the device's axial length**; ends that already have one are left alone. Override
with `wakefield_config['beampipe_length']` (metres). This means geometries built
with `beampipe='none'` still run — the pipes exist only in the ABCI deck, not in
the cavity you defined.

Plot the impedance spectra and wake potentials:

```python
ax = tesla.plot('zl')            # longitudinal impedance |Z_L|
ax = tesla.plot('zt', ax)        # transverse impedance |Z_T| on the same axes
tesla.plot('wpl'); tesla.plot('wpt')             # wake potentials

# Or via the solver-object API, which returns DataFrames:
tesla.wakefield.wake_z           # f [MHz], |Z|, Re(Z), Im(Z), s [m], W
tesla.wakefield.plot_z(quantity='impedance')
tesla.wakefield.plot_t(quantity='wake')

# The main run always reports the loss/kick factors:
tesla.wakefield.qois             # {'|k_loss| [V/pC]', 'k_FM [V/pC]',
                                 #  'k_loss_HOM [V/pC]', '|k_kick| [V/pC/m]'}
```

Any wakefield solver plugs in behind `cav.wakefield` — ABCI is the default
backend, selected by `wakefield_config['solver']` (`'abci'`). The results above
are read through a normalised schema, so nothing above changes when the solver
does.

### Loss/kick factors and HOM power at operating points

The main run's loss/kick factors are always in `cav.wakefield.qois`. To also get
higher-order-mode *power* at machine operating points, pass an `operating_points`
dictionary (this re-runs per bunch length and needs R/Q from a prior eigenmode
run); those results are reported separately in `cav.wakefield.qois_op`:

```python
op_points = {
    "Z": {
        "freq [MHz]": 400.79, "E [GeV]": 45.6, "I0 [mA]": 1280, "V [GV]": 0.12,
        "Eacc [MV/m]": 5.72, "nu_s []": 0.0370, "alpha_p [1e-5]": 2.85,
        "tau_z [ms]": 354.91, "tau_xy [ms]": 709.82, "f_rev [kHz]": 3.07,
        "beta_xy [m]": 56, "N_c []": 56, "T [K]": 4.5,
        "sigma_SR [mm]": 4.32, "sigma_BS [mm]": 15.2, "Nb [1e11]": 2.76,
    }
}
cavs.run_wakefield({
    'processes': 1, 'rerun': True,
    'bunch_length': 25, 'wakelength': 50,
    'operating_points': op_points,
})
pp.pprint(cavs.wakefield_qois)
cavs.plot_compare_hom_bar('Z_SR_4.32mm')
```

---

## Objective names

Monopole and m-pole results share QOI key names — `freq [MHz]`, `R/Q [Ohm]`,
`G [Ohm]` and 15 others appear in every polarisation's `qois.json`. So every UQ
and optimisation objective must say which polarisation it means:

```text
monopole:R/Q [Ohm]              # the polarisation's primary mode of interest
dipole:R/Q_t [Ohm/m^(2(m-1))]   # a dipole-only QOI
dipole:2:freq [MHz]             # dipole, mode 2 (1-based)
1:freq [MHz]                    # azimuthal number instead of a name
```

The separator is `:`; no QOI name contains one, and only the first two fields
are split, so the QOI itself may contain anything. A bare `'freq [MHz]'` is
**rejected** — it used to silently mean *the monopole's*, which becomes a wrong
answer the moment a second polarisation is solved. An objective naming a QOI or
mode that does not exist raises a `ValueError` listing what is available;
previously it was dropped without a word and the statistics were quietly
computed over whichever objectives happened to match.

Naming a polarisation in an objective is enough — the UQ run adds it to
`eigenmode_config['polarisation']` automatically. Wakefield objectives (`ZL`,
`ZT`) carry no polarisation and are written as before.

### Which mode is reported: `mode_of_interest`

`eigenmode_config['mode_of_interest']` selects the modes whose QOIs are reported.
It is **1-based** (mode 1 is the lowest of the passband) and may be an int, a list
of **any length**, or a dict keyed by polarisation — each polarisation may carry a
different number of modes. Order is preserved and duplicates collapse. The
**first** mode listed is the primary one and lands in `qois.json`; all of them are
written to `qois_moi.json` keyed by 1-based mode, and each is addressable as an
objective (`monopole:4:freq [MHz]`).

```python
'mode_of_interest': 9                                        # every polarisation
'mode_of_interest': [1, 2, 3]                                # three, for every one
'mode_of_interest': {'monopole': [1, 2, 3, 4], 'dipole': 1}  # different counts
```

Defaults: **monopole → `n_cells`**, the π-mode — the operating mode of an
accelerating structure. A 1-cell cavity, gun or pillbox therefore gets mode 1,
so the convention degrades gracefully to non-accelerator geometries. **m-pole →
mode 1**, the lowest of the deflecting passband (an m-pole spectrum has no
spurious DC mode to skip).

## Uncertainty quantification

Any analysis can carry uncertainty quantification by adding a `uq_config`. It
propagates geometry uncertainties through the solve and reports the mean and
standard deviation of each objective. Result keys are the canonical objective
names, e.g. `uq.json['dipole:freq [MHz]']['expe'][0]`.

```python
cavs.run_eigenmode({
    'processes': 1,
    'rerun': True,
    'boundary_conditions': 'mm',
    'uq_config': {
        'variables': ['A', 'B', 'a', 'b'],           # varied parameters
        # Objectives must name a polarisation (see "Objective names" below)
        'objectives': ["monopole:freq [MHz]", "monopole:R/Q [Ohm]", "monopole:Epk/Eacc []",
                       "monopole:Bpk/Eacc [mT/MV/m]", "monopole:G [Ohm]"],
        'delta': [0.05, 0.05, 0.05, 0.05],           # 5% std on each
        'processes': 1,
        'distribution': 'gaussian',
        'method': ['Quadrature', 'Stroud3'],
        'cell_type': 'mid-cell',
        'cell_complexity': 'simplecell',
    },
})
pp.pprint(cavs.uq_fm_results)
cavs.plot_compare_fm_bar(uq=True)     # bar chart with error bars
```

> [!IMPORTANT]
> The key is `cell_complexity` (underscore). If a config contains an unrecognised
> key, `cavsim2d` emits a `UserWarning` with a "did you mean …?" suggestion —
> useful for catching typos that would otherwise be silently ignored.

---

## Optimisation

Multi-objective shape optimisation uses a genetic algorithm. Each candidate is
tuned to the target frequency (via a nested `tune_config`) before its objectives
are evaluated.

```python
cavs = Cavities(r'/path/to/my_project')

optimisation_config = {
    'initial_points': 6,
    'no_of_generation': 2,
    'method': {'LHS': {'seed': 5}},
    'bounds': {                       # search space; fix a variable with equal bounds
        'A': [38, 46], 'B': [38, 46], 'a': [10, 15], 'b': [15, 22],
        'Ri': [33, 37], 'L': [57.7, 57.7], 'Req': [100, 106],
    },
    'objectives': [
        ['min', 'monopole:Epk/Eacc []'],
        ['min', 'monopole:Bpk/Eacc [mT/MV/m]'],
        # ['max', 'monopole:R/Q [Ohm]'],
        # ['min', 'dipole:R/Q_t [Ohm/m^(2(m-1))]'],   # deflecting-mode impedance
        # ['min', 'ZL', [1, 2, 5]],   # impedance peak in a frequency window (needs wakefield)
    ],
    'tune_config': {
        'freqs': 1300,
        'cell_type': {'mid-cell': 'Req'},
        'processes': 1,
        'eigenmode_config': {'n_cells': 1, 'processes': 1, 'boundary_conditions': 'mm'},
    },
    # GA controls — these are integer offspring counts, not fractions:
    'mutation_factor': 4,
    'crossover_factor': 4,
    'elites_for_crossover': 2,
    'chaos_factor': 2,
    'weights': [1, 1],
}
cavs.run_optimisation(optimisation_config)

opt = cavs.optimisation
opt.history            # every evaluated candidate (DataFrame)
opt.pareto             # Pareto-optimal candidates
opt.plot_pareto(kind='scatter')
opt.plot_convergence()
```

Supported objective quantities: `freq [MHz]`, `Epk/Eacc []`,
`Bpk/Eacc [mT/MV/m]`, `R/Q [Ohm]`, `G [Ohm]`, `Q []`, and the wakefield impedance
peaks `ZL`, `ZT` (with a `[low, ..., high]` frequency window as the third
element). Each objective is `['min'|'max'|'equal', name]` (plus a target value
for `'equal'`, or the window for `ZL`/`ZT`).

> [!NOTE]
> `mutation_factor`, `crossover_factor` and `chaos_factor` are **counts** of
> offspring / random individuals per generation (integers), not probabilities.

---

## Configuration dictionaries

Every analysis is driven by a config dict, and they nest consistently. UQ slots
into any solver config via `uq_config`; optimisation wraps a `tune_config` (which
itself wraps an `eigenmode_config`), and optionally a `wakefield_config`:

```
optimisation_config
├── tune_config
│   └── eigenmode_config
│       └── uq_config
├── wakefield_config
│   └── uq_config
├── bounds, objectives, initial_points, no_of_generation, ...
└── mutation_factor, crossover_factor, elites_for_crossover, chaos_factor
```

Unrecognised keys are reported as warnings (with suggestions), so a mistyped key
never silently changes behaviour. For the full accepted-key reference, see
`cavsim2d/utils/config_validation.py`, or `help(cavs.run_eigenmode)` etc.

---

## Parallelisation

Set `processes` in any config to run cavities in parallel. When UQ is enabled, a
second `processes` inside `uq_config` parallelises the quadrature/sample points.
The default is a single process (which also gives the cleanest tracebacks and
Jupyter output).

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

The solver tests self-skip where NGSolve/gmsh are unavailable, and the wakefield
tests skip off Windows (they need the bundled `ABCI.exe`), so the suite stays
green in minimal environments.

---

## License

MIT — see [LICENSE](LICENSE).
