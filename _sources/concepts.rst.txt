Concepts
========

This page collects the cross-cutting ideas that the analysis guides assume: how a
project is laid out on disk, how the configuration dictionaries work, and how the
quantities of interest (QOIs) are defined — including the azimuthal-mode
conventions.

Objects
-------

Two objects carry the whole workflow:

- :class:`~cavsim2d.study.Study` — a container for one or more cavities and the
  home of multi-cavity actions (``run_tune``, ``run_eigenmode``, ``run_wakefield``,
  ``run_optimisation``) and the comparison plots.
- :class:`~cavsim2d.models.base.Cavity` and its subclasses
  (:class:`~cavsim2d.models.elliptical.EllipticalCavity`,
  :class:`~cavsim2d.models.pillbox.Pillbox`,
  :class:`~cavsim2d.models.rfgun.RFGun`,
  :class:`~cavsim2d.models.spline.SplineCavity`) — one simulatable structure.

Each analysis is reached through a **namespace** attached to the cavity (and, for
comparisons, to the study):

.. code-block:: python

    cav.eigenmode     # EigenmodeSolver   -> cav.eigenmode.run(...), .qois, .plot_*
    cav.wakefield     # WakefieldSolver
    cav.tune          # TuneSolver
    cav.multipacting  # MultipactingSolver
    study.optimisation  # OptimisationSolver

Project folder layout
---------------------

A study is rooted at a project directory (``Study(project_dir)`` or, for a single
standalone cavity, ``cav.set_workspace(dir)``). Every cavity gets its own folder,
and every analysis writes into a subfolder of it::

    <project>/
      <cavity_name>/
        geometry/        # geodata.geo, the meridian contour + a JSON snapshot
        eigenmode/
          monopole/      # qois.json, fields, Ez_0_abs.csv, mesh, ...
          dipole/        # one folder per solved polarisation (m >= 1)
        wakefield/        # ABCI decks + parsed impedance / wake / QOIs
        multipacting/    # tracked-particle results, field
        uq/              # nodes.csv, table.csv, uq.json, sobol.json
        tuned/           # a *separate* cavity that hit the tune target
          geometry/  eigenmode/  tune_info/
      optimisation/       # history.csv, pareto_front.csv, candidates/

``cav.tuned`` loads the cavity under ``tuned/`` (its ``tune_info/tuned_parameters.json``
records the complete, self-consistent parameter set the tuner converged on).

Persistence and reloading
-------------------------

Results are written to disk and read back lazily: ``cav.eigenmode.qois``,
``cav.wakefield.qois`` and ``cav.tuned`` all load from the folders above the first
time you touch them, so within a session a completed analysis is never re-run.

An entire project reloads in one call. Each cavity also saves a small
``geometry/model.json`` describing its type and parameters, so::

    study = Study.load('my_project')     # reconstruct every cavity + its cached results

returns a Study with every cavity reconstructed (the right model type, pointing at
its existing folder). All cached results are then available with **no
re-simulation**, and because a loaded Study behaves like any other, two studies —
from different sessions, or a loaded one against a fresh one — compare directly
(``study.eigenmode.qois_df``, ``study.eigenmode.plot_compare()``). Custom
in-notebook models whose module cannot be imported by name are passed explicitly:
``Study.load(dir, models={'MyCavity': MyCavity})``.

Parameter sweeps
----------------

``study.sweep(template, {var: values}, mode='tensor'|'hadamard')`` builds one
cavity per parameter combination (``'tensor'`` = the full grid; ``'hadamard'`` =
element-wise, requiring equal-length value lists), each in its own folder. It
returns a Study, so any analysis runs on the whole family at once; ``sw.results()``
joins the swept values with the QOIs into a single comparison table, and
``sw.sweep_table`` records which cavity is which. See the
:doc:`examples/studies/parameter_sweep` example.

Configuration dictionaries
--------------------------

Every ``run_*`` call takes one dictionary. Missing keys fall back to that
analysis' documented defaults, and the **merged** dict (defaults + your overrides)
is saved to ``config.json`` beside the results — so a saved run always records
every setting it used, not just the keys you passed. Config keys double as keyword
arguments. Truthy values (never mere presence) switch features on.

Eigenmode (``cav.eigenmode.run`` / ``study.run_eigenmode``):

.. code-block:: python

    {
      'polarisation': 'monopole',   # 'monopole'|'dipole'|... , an m, or a list
      'n_modes': None,              # None -> n_cells + 2
      'mode_of_interest': None,     # 1-based; None -> the accelerating pi-mode
      'boundary_conditions': 'mm',  # magnetic/electric on the two end planes
      'conductivity': 5.96e7,       # [S/m]; or set 'surface_resistance' for SRF
      'mesh_config': {'h': 20, 'p': 3, 'adaptive': None},  # h in mm, p order >= 2
      'uq_config': None,            # attach to propagate geometry uncertainty
    }

Setting ``mesh_config['adaptive']`` refines the eigenmode mesh where the solver's own
error estimate is largest. It is a *mode* of the eigenmode solve — the refined mesh is
the eigenmode result, so ``show_fields``/``show_mesh``/``multipacting`` use it directly
and ``cav.eigenmode.plot_convergence()`` shows the error-vs-DOF history. Because the
wakefield below runs on the **ABCI** backend (its own deck, meshed from
``cav.profile()``), adaptivity applies to eigenmode/multipacting only.

Wakefield (``cav.wakefield.run`` / ``study.run_wakefield``) — ABCI backend:

.. code-block:: python

    {
      'MROT': 2,               # 0 longitudinal, 1 transverse, 2 both
      'wakelength': 50,        # [m]
      'bunch_length': 25,      # [mm]
      'beampipe_length': None, # [m]; None -> 3x the axial length
      'save_fields': None,     # True/{'nshot': N} -> E-field-line snapshots
      'operating_points': None,
      'uq_config': None,
    }

Tune (``cav.tune.run`` / ``study.run_tune``):

.. code-block:: python

    {
      'freqs': 1300.0,                       # target [MHz] (list for many cavities)
      'cell_type': {'mid-cell': 'Req'},      # {cell block: handle}; schedule several
      'eigenmode_config': {...},             # the solve used at each tuning step
    }

See :doc:`tuning` for the cell-block keys and the multicell mid/end recipe.

UQ (``uq_config``, nested inside an eigenmode/wakefield/tune config):

.. code-block:: python

    {
      'variables': ['Req', 'Ri'],   # geometry variables to perturb
      'delta': [0.05, 0.05],        # relative (or absolute) spread per variable
      'method': ['Quadrature', 'Stroud3'],  # or a Gaussian/LHS Monte-Carlo design
      'cell_complexity': 'simplecell',       # or 'multicell' (per-half-cell perturb)
    }

Optimisation (``study.run_optimisation``) — see :doc:`optimisation` for the full
list of GA controls, ``bounds``, ``objectives`` and the nested ``tune_config``.

Quantities of interest and mode conventions
-------------------------------------------

Solved fields are reduced to scalar QOIs (written to ``qois.json``). The azimuthal
order ``m`` sets both *which* mode is reported and *how* the QOIs are defined.

- **Monopole (m = 0)** — the accelerating family. The reported mode defaults to the
  **pi-mode** (index ``n_cells`` of the fundamental passband). QOIs include
  ``freq [MHz]``, ``R/Q [Ohm]``, ``G [Ohm]``, ``Q []``, ``Epk/Eacc []``,
  ``Bpk/Eacc [mT/MV/m]``, and — for multicell cavities — the cell-to-cell coupling
  ``kcc [%]`` and field flatness ``ff [%]``.
- **m >= 1 (dipole, quadrupole, …)** — the deflecting / higher-order-mode families.
  The reported mode defaults to mode 1 (the lowest of the deflecting passband); the
  accelerating voltage is evaluated off-axis because ``E_z ~ r^m`` vanishes on axis.

**R/Q convention.** cavsim2d reports the *linac* (accelerator) definition
:math:`R/Q = V^2/(\omega U)`. This is **twice** the *circuit* definition
:math:`V^2/(2\omega U)` used in some references (e.g. the TESLA report's 518 Ohm
per cell is the circuit value; cavsim2d reports ~1036 Ohm for the same cell).

**Impedance.** The eigenmode impedance spectrum sums the contribution of **every**
solved m-pole mode: the longitudinal spectrum is the monopole modes, and the
transverse spectrum spans every ``m >= 1`` multipole. Solve the polarisations you
want represented before reading the impedance.

.. tip::

   Because the merged config is saved next to every result, the fastest way to see
   exactly what a run used is to open ``<cavity>/<analysis>/config.json``.
