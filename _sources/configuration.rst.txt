Configuration reference
=======================

Every analysis is driven by **one dictionary**. This page lists all of them and
all of their keys. See :doc:`concepts` for the conventions they share:

- **Complete + saved.** Missing keys fall back to the analysis' defaults, and the
  *merged* dict (defaults + your overrides) is written to ``config.json`` beside
  the results, so a saved run records every setting it used.
- **Keys double as kwargs.** ``cav.eigenmode.run(mesh_config={'h': 10})`` is the
  same as passing it in the dict; an explicit kwarg wins.
- **Truthy, not present.** Features switch on for a *truthy* value (an empty
  ``uq_config={}`` does **not** trigger UQ).

Cross-cutting keys
------------------

These appear in several configs:

``processes``
   *(int, default 1)* Worker processes for parallel evaluation (UQ nodes,
   optimisation candidates, multi-cavity studies).
``rerun``
   *(bool, default True)* Re-run even if cached results exist; ``False`` reuses them.
``uq_config``
   *(dict, default None)* Nest this inside an eigenmode / wakefield / tune config to
   turn a single run into an uncertainty sweep — see :ref:`config-uq`.

Eigenmode — ``cav.eigenmode.run`` / ``study.run_eigenmode``
-----------------------------------------------------------

``polarisation``
   *(str | int | list, default* ``'monopole'`` *)* The azimuthal order(s) to solve:
   a name (``'monopole'``, ``'dipole'``, ``'quadrupole'``, …), the integer ``m``, or
   a list. Every ``m >= 1`` is written to ``eigenmode/<pol name>/``.
``mesh_config``
   *(dict, default* ``{'h': 20, 'p': 3, 'adaptive': None}`` *)* ``h`` maximum element
   size in **mm**; ``p`` polynomial order (**>= 2**); ``adaptive`` opt-in error-driven
   h-refinement.
``n_modes`` / ``nmodes``
   *(int, default None)* Number of eigenmodes to solve. ``None`` → ``n_cells + 2``.
``mode_of_interest``
   *(int | list | dict, default None)* **1-based** index of the mode(s) whose QOIs are
   reported. ``None`` → the accelerating pi-mode (monopole) / the lowest deflecting
   mode (``m >= 1``).
``boundary_conditions``
   *(str, default* ``'mm'`` *)* The electric/magnetic condition on the two end planes.
``conductivity``
   *(float, default 5.96e7)* Wall conductivity [S/m] for the loss-based Q / G.
``surface_resistance``
   *(float, default None)* Fixed surface resistance [Ohm] override (e.g. for SRF),
   used instead of ``conductivity``.
``f_shift``
   *(float, default 0)* Spectral shift for the eigensolver.
``normalization_length``
   *(float, default None)* Active length for the accelerating-voltage normalisation;
   elliptical cavities take it from ``L_m`` automatically.

Wakefield — ``cav.wakefield.run`` / ``study.run_wakefield``
-----------------------------------------------------------

``MROT``
   *(int, default 2)* ``0`` longitudinal, ``1`` transverse, ``2`` both.
``wakelength``
   *(float, default 50)* Wake length [m].
``bunch_length``
   *(float, default 25)* RMS bunch length [mm].
``beampipe_length``
   *(float, default None)* Beam-pipe length [m]; ``None`` → 3× the axial length.
``save_fields``
   *(bool | dict, default None)* ``True`` (or ``{'nshot': N}``) saves electric
   field-line snapshots for the field animation.
``operating_points``
   *(dict, default None)* Machine operating points (currents, bunch populations,
   bunch lengths) to fold into loss/kick-factor and HOM-power tables.
``solver``
   *(str, default* ``'abci'`` *)* The wakefield backend.
``MT``, ``NFS``, ``DDR_SIG``, ``DDZ_SIG``
   ABCI meshing / sampling controls (mesh density, number of frequency samples,
   radial and longitudinal mesh ratios). Defaults ``10``, ``10000``, ``0.1``, ``0.1``.

Tuning — ``cav.tune.run`` / ``study.run_tune``
----------------------------------------------

``freqs``
   *(float | list)* Target frequency [MHz] (a list gives a per-cavity target).
``cell_type``
   *(dict)* ``{cell block: handle}``, e.g. ``{'mid-cell': 'Req'}``. Schedule several
   blocks (``{'mid-cell': 'Req', 'end-cell': 'L'}``) to tune them in turn — see
   :doc:`tuning`.
``eigenmode_config``
   *(dict)* The eigenmode settings used for the solve at **each** tuning step.
``tol``
   *(float, default 1e-4)* Secant-solver tolerance.
``maxiter``
   *(int, default 10)* Maximum secant iterations.

.. _config-multipacting:

Multipacting — ``cav.multipacting``
-----------------------------------

Emission sites are set with ``cav.multipacting.set_emission_points(xrange, ...)``
(or ``xrange`` / ``n_points`` in the config); the rest are ``run()`` keys:

``mode``
   *(int, default 0)* Which computed mode to track (0 = the first).
``polarisation`` / ``n_modes``
   *(default None)* Solve multipacting's **own** field for this polarisation / this
   many modes (``None`` → reuse the monopole eigenmode field).
``epks``
   *(array, default None)* Peak-field levels [V/m] to sweep (``None`` → 0–80 MV/m, 192 steps).
``phis``
   *(array, default None)* RF launch phases (``None`` → 72 phases over ``[0, 2*pi]``).
``v_init``
   *(float, default 2)* Secondary-emission energy [eV].
``t_max``
   *(float, default 1e-7)* Tracking duration [s].
``loss_model``
   *(str, default* ``'field'`` *)* How escaped particles are handled.
``pec_maxh``
   *(float, default None)* If set [mm], multipacting builds its **own** finer field
   on a mesh this size (with ``mesh_config``) instead of reusing the eigenmode one.
``n_points``
   *(int, default None)* Extra emission sites interpolated along the wall in-band.
``proc_count``
   *(int, default None)* Worker count (``None`` → auto).
``progress``
   *(bool, default True)* Live progress bar over the field-level sweep.

.. _config-uq:

Uncertainty quantification — nested ``uq_config``
-------------------------------------------------

Nest inside an eigenmode / wakefield / tune config to sweep the uncertain geometry.

``variables``
   *(list of str)* Geometry variables to perturb, in that model's own names
   (``['Req', 'Ri']`` for elliptical, ``['p2_r']`` for spline, …).
``delta``
   *(list of float)* Perturbation magnitude per variable. **By default this is an
   additive perturbation in millimetres** (``perturbation_mode`` defaults to
   ``['add', delta]``), *not* a fraction — ``delta=0.3`` is ±0.3 mm. A perturbation
   far below the solver's numerical resolution leaves the result unchanged.
``perturbation_mode``
   *(list, default* ``['add', delta]`` *)* ``['add', d]`` → ``x + d`` (absolute, mm);
   ``['mul', d]`` → ``x * (1 + d)`` (relative fraction).
``method``
   *(list)* The design: ``['Quadrature', 'Stroud3']`` (default) or ``'Stroud5'``,
   a Gaussian Monte-Carlo ``['normal', N]``, or Latin-hypercube ``['lhs', N]``.
``distribution``
   *(str)* ``'gaussian'`` (``delta`` = std) or ``'uniform'`` (``delta`` = half-width).
``cell_complexity``
   *(str, default* ``'simplecell'`` *)* ``'simplecell'`` perturbs the mid/end-cell
   groups; ``'multicell'`` makes every half-cell an independent random variable
   (honouring continuity). Available for eigenmode and wakefield.
``independent_half_cells``
   *(bool, default False)* Multicell only: perturb every half-cell independently then
   weld (average) the shared seams — the WEPB015 before/after-continuity workflow.
``cell_type``
   *(str, default* ``'mid-cell'`` *)* Simplecell only: which cell block to perturb.
``objectives``
   *(list)* The QOIs to gather: ``'pol:qoi'`` for eigenmode (``'monopole:R/Q [Ohm]'``),
   or ``['min', 'ZL', [lo, hi]]`` / ``['min', 'ZT', ...]`` for wakefield impedance
   (the interval is a **flat** ``[lo, hi, …]`` GHz list defining consecutive windows).
``tune_config``
   *(dict, default None)* Re-tune each perturbed variant to a target frequency before
   measuring, so only the *shape* varies at fixed frequency (the paper workflow).

Optimisation — ``study.run_optimisation``
------------------------------------------

``bounds``
   *(dict)* ``{variable: [lo, hi]}`` in that model's own variable names (mm). A fixed
   variable takes equal bounds.
``objectives``
   *(list of lists)* ``[direction, metric, ...]`` — direction ``'min'`` / ``'max'`` /
   ``'equal'``; metric a polarisation-qualified QOI (``'monopole:Epk/Eacc []'``) or a
   wakefield ``'ZL'`` / ``'ZT'`` with a frequency-interval third element.
``tune_config``
   *(dict)* Applied to every candidate so each is on-frequency before its objectives
   are read.
``initial_points``
   *(int)* Initial candidates before the GA evolves.
``no_of_generation``
   *(int)* Number of GA generations.
``method``
   *(dict, default LHS)* Initial-point sampler, e.g. ``{'LHS': {'seed': 5}}``.
``crossover_factor`` / ``mutation_factor`` / ``chaos_factor``
   *(int)* How many offspring come from crossover, mutation and random injection each
   generation.
``elites_for_crossover``
   *(int)* Top candidates retained in the parent pool.
``weights``
   *(list)* Per-objective weights.
``seed``
   *(int)* Makes the run reproducible.
