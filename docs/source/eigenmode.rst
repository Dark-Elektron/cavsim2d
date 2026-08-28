Eigenmode Analysis
==================

The eigenmode analysis module computes the electromagnetic resonant modes of an RF cavity using a unified product-space finite-element solver built on `NGSolve <https://ngsolve.org>`_. A single formulation covers monopole (:math:`m=0`), dipole (:math:`m=1`), and arbitrary higher-order multipole modes; see :doc:`theory` for the mathematical derivation.

Interface
*********
To run the eigenmode analysis on all cavities loaded in your project:

.. code-block:: python

    cavs.run_eigenmode(eigenmode_config=None)

Or for a single cavity object:

.. code-block:: python

    tesla.run_eigenmode(eigenmode_config=None)

When called without arguments, default settings are used (monopole polarisation, polynomial order :math:`p=3`, initial mesh size 20 mm, number of modes = ``n_cells + 2``).

Configuration Dictionary
************************
You can pass an optional dictionary to control the solver, meshing, and physical
boundary options. Every key can equally be passed as a keyword argument —
``cav.eigenmode.run(mesh_config={'h': 10})`` — and kwargs override the dictionary.
The config is merged over a complete set of defaults, and the **merged** dict is
what runs and what ``eigenmode/config.json`` records: a saved config always
contains every setting the run used.

.. code-block:: python

    eigenmode_config = {
        'polarisation': 'monopole',
        'n_modes': 10,
        'mesh_config': {
            'h': 15,
            'p': 3,
            'adaptive': {
                'tol': 1e-12,
                'max_refinements': 8,
                'max_ndof': 100000
            }
        },
        'conductivity': 5.96e7,
        'surface_resistance': None,
        'mode_of_interest': 9
    }

Settings description:

``polarisation``
   *(int, str, or list, default:* ``0`` *)*
   Specifies the azimuthal mode number(s) to solve. Accepts an integer (``0`` for monopole, ``1`` for dipole, ``2`` for quadrupole) or a descriptive name (``'monopole'``, ``'dipole'``, ``'quadrupole'``, etc.). Pass a list to solve several polarisations in one call.

``mesh_config``
   *(dict)* Controls the finite-element discretisation.

   - ``h`` *(float, default: 20)* — Maximum mesh element size (in mm).
   - ``p`` *(int, default: 3)* — Polynomial order of the finite-element basis functions.
   - ``adaptive`` *(bool or dict, default: None)* — Enables adaptive mesh refinement (AMR) driven by Zienkiewicz--Zhu error indicators (see :ref:`theory/eigenmode:Zienkiewicz--Zhu (ZZ) Error Estimator`). Set to ``True`` for default AMR settings, or pass a dictionary with:

     - ``tol`` — target error tolerance,
     - ``max_refinements`` — maximum refinement iterations,
     - ``max_ndof`` — upper bound on the number of degrees of freedom.

``n_modes`` / ``nmodes``
   *(int, default:* ``n_cells + 2`` *)* Number of eigenmodes to compute.

``conductivity``
   *(float, default:* ``5.96e7`` *S/m)* Electrical conductivity of normal-conducting walls. Used to compute wall power losses :math:`P_{\text{loss}}` and quality factor :math:`Q`.

``surface_resistance``
   *(float, default: None, in Ohm)* Explicit surface resistance. Useful for superconducting cavities (SRF) where the BCS surface resistance replaces the normal-conducting calculation.

``mode_of_interest``
   *(int, list, or dict)* 1-based index of the physical mode(s) whose results are reported. For monopole modes, this defaults to the accelerating :math:`\pi`-mode (index equals ``n_cells``).

``materials``
   *(dict, default: None)* Overrides the properties of dielectric regions declared
   on the cavity with :meth:`~cavsim2d.models.base.Cavity.add_dielectric`, e.g.
   ``{'quartz': {'eps_r': 3.8, 'tan_delta': 1e-4}}`` (shorthands:
   ``{'quartz': 3.8}``, or a complex ``{'quartz': 3.8 - 3.8e-4j}``). Overrides
   *merge* onto the declared properties, so ``{'quartz': {'tan_delta': 0.05}}`` is
   a loss sweep that leaves ``eps_r`` alone. Naming a region the cavity does not
   have raises, so a typo cannot silently fall back to vacuum. See
   :ref:`eigenmode:Dielectric regions`.

``loss_model``
   *(str, default: None)* How a complex permittivity reaches ``Q``:
   ``'lossless'`` (real eigenproblem, dielectric loss added perturbatively),
   ``'lossy'`` (full complex eigenproblem), or ``'auto'`` / ``None`` — the default —
   which inspects the loss tangents and switches to ``'lossy'`` above
   :math:`\tan\delta = 10^{-2}`. Asking for ``'lossless'`` above that threshold is
   honoured, with a warning. See :ref:`eigenmode:Dielectric loss`.

``arnoldi_vectors``
   *(int, default: 6)* Krylov vectors per shift in the lossy eigensolve. Raise it
   if a lossy run warns that a mode did not converge.

Quantities of Interest (QOIs)
*****************************
Once computed, the figures of merit are written to ``eigenmode/<polarisation_name>/qois.json`` and stored on the cavity objects. The table below summarises the output keys:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Unit
     - Description
   * - ``freq [MHz]``
     - MHz
     - Resonant mode frequency.
   * - ``U [J]``
     - J
     - Stored electromagnetic energy in the cavity volume.
   * - ``Vacc [MV]``
     - MV
     - Accelerating voltage integrated along the :math:`z`-axis.
   * - ``Eacc [MV/m]``
     - MV/m
     - Accelerating electric field gradient.
   * - ``R/Q [Ohm]``
     - :math:`\Omega`
     - Shunt impedance over quality factor (longitudinal or transverse).
   * - ``Ploss [W]``
     - W
     - Wall power dissipation from surface resistance.
   * - ``Q []``
     - —
     - Cavity quality factor.
   * - ``G [Ohm]``
     - :math:`\Omega`
     - Geometry factor :math:`G = Q R_s`.
   * - ``Epk/Eacc []``
     - —
     - Peak electric field normalised by the accelerating gradient.
   * - ``Bpk/Eacc [mT/MV/m]``
     - mT/(MV/m)
     - Peak magnetic field normalised by the accelerating gradient.

Dielectric regions
******************

By default a cavity is vacuum throughout. :meth:`~cavsim2d.models.base.Cavity.add_dielectric`
fills part of it with a dielectric — a ceramic window, a beam-pipe liner, an
absorber ring:

.. code-block:: python

    cav = Pillbox(1, [20, 37.5, 2.5, 0, 5], beampipe='both')

    # a 0.5 mm-thick quartz tube lining the 5 mm aperture, full length
    cav.add_dielectric('quartz', 3.8, z=(-1e4, 1e4), r=(2.0, 2.5), maxh=0.15)

    cav.eigenmode.run(mesh_config={'h': 3, 'p': 3})

The region is an axisymmetric **rectangular ring** in the (z, r) meridian plane —
an annular cylinder in 3D. All lengths are in **millimetres**, like every other
cavity dimension, and the region is *clipped* to the cavity, so a generous span
such as ``z=(-1e4, 1e4)`` simply means "the full length".

``maxh`` sets the local element size **inside** the region. A thin shell needs
it: a 0.5 mm tube wall in a cavity meshed at the default ``h = 20`` would
otherwise be crossed by a single element.

What changes in the solve
-------------------------

Weighting the mass form by :math:`\varepsilon_r` solves
:math:`\nabla\times\nabla\times \mathbf{E} = \lambda\, \varepsilon_r \mathbf{E}`,
so the eigenvalue is still :math:`\lambda = k_0^2 = (\omega/c_0)^2` and the
frequency conversion is unchanged. The stored energy carries
:math:`\varepsilon_r` **inside** the integral, so ``R/Q``, ``Q``, ``G``, ``Rsh``
and ``GR/Q`` all follow.

No special interface treatment is needed: ``HCurl`` enforces tangential-:math:`E`
continuity and :math:`u_\varphi = r E_\varphi` is tangential to any r-z
interface — exactly the physical dielectric interface conditions. Normal
:math:`D` continuity is natural.

One extra QOI appears per region, ``Epk_<material> [MV/m]``: :math:`E` is
discontinuous across the interface, so the peak field *inside* the dielectric —
the one that matters for breakdown — is not visible in the wall-sampled ``Epk``.

.. _eigenmode-dielectric-loss:

Dielectric loss
---------------

A region can be lossy. With the :math:`e^{+j\omega t}` convention the
permittivity is :math:`\varepsilon_r = \varepsilon_r' - j\varepsilon_r''` and
:math:`\tan\delta = \varepsilon_r''/\varepsilon_r'`:

.. code-block:: python

    cav.add_dielectric('alumina', 9.8, tan_delta=1e-4,
                       z=(-1e4, 1e4), r=(2.0, 2.5), maxh=0.15)

There are two ways to get :math:`Q` out of that, and they cost very different
amounts.

**Perturbation** (the default up to :math:`\tan\delta = 10^{-2}`). The cavity is
solved as if it were lossless and the loss is integrated over the resulting field:

.. math::

   Q_{\text{diel}} = \frac{\int \varepsilon_r'\,|\mathbf{E}|^2\, r\,dr\,dz}
                          {\int \varepsilon_r''\,|\mathbf{E}|^2\, r\,dr\,dz},
   \qquad
   \frac{1}{Q} = \frac{1}{Q_{\text{wall}}} + \frac{1}{Q_{\text{diel}}}

This costs nothing beyond the lossless solve. It is accurate to
:math:`O(\tan^2\delta)`, and the reported frequency and field distribution are the
lossless ones — so it is the right tool for low-loss dielectrics (sapphire,
alumina, quartz windows) and says nothing about the frequency pull a lossy
material causes.

**Full complex eigenproblem** (the default above :math:`\tan\delta = 10^{-2}`).
The resonant frequency itself is solved for as a complex number,
:math:`\omega = \omega_r + \mathrm{i}\alpha`, and

.. math::

   Q_{\text{diel}} = \frac{\omega_r}{2\alpha}

so :math:`Q`, the frequency **and the mode shape** are all the lossy ones. This is
what a lossy ferrite or an absorber at :math:`\tan\delta \sim 0.1`–:math:`1`
needs, and it costs roughly twice a lossless solve. See
:ref:`theory/eigenmode:Lossy Dielectrics` for the formulation.

Which one ran is never left to inference — a run with a lossy material reports:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``Q model``
     - ``'lossless'``, ``'perturbation'`` or ``'lossy'`` — how ``Q_diel`` was obtained
   * - ``Q []``
     - total :math:`Q`, wall and dielectric combined
   * - ``Q_wall []``
     - wall loss alone (what ``Q []`` was before this feature)
   * - ``Q_diel []``
     - dielectric loss alone
   * - ``Pdiel [W]``
     - dielectric power dissipation at the solved field amplitude
   * - ``tan_delta []``
     - the largest loss tangent in the run

``G [Ohm]`` stays :math:`Q_{\text{wall}} R_s`: the geometry factor is a property
of the wall and the mode shape, so folding dielectric loss into it would make a
material-independent quantity depend on the filling. A run with no dielectric
loss adds none of these keys.

``tan_delta`` is a UQ variable like any geometric one (``'alumina:tan_delta'``),
and changing it invalidates cached results.

Limits
------

- :math:`\mu_r` is *rejected*, not ignored: magnetic materials change the
  stiffness form, not just the mass form.
- A **conductive** loss, :math:`\varepsilon_r'' = \sigma/(\omega\varepsilon_0)`, is
  not modelled, because it makes the permittivity depend on the answer; pass a loss
  tangent evaluated at the frequency of interest instead.
- Wall loss is always treated perturbatively, on both paths.
- **Eigenmode only.** ``wakefield`` and ``multipacting`` raise on a cavity with
  dielectric regions rather than returning a vacuum answer for it.
- **Native geometry only.** Cavities meshed from an imported ``.geo`` file cannot
  carry regions (gmsh writes a single physical surface and the STEP round-trip
  drops surface names), so this raises too. Every built-in model has a native
  ``profile()``.

Accessing Results
*****************
To read the output results directly in Python:

.. code-block:: python

    # Primary mode of interest QOIs (all cavities)
    print(cavs.eigenmode_qois)

    # Access local result objects
    q = tesla.eigenmode.qois
    print(f"Resonant frequency: {q['freq [MHz]']} MHz")
    print(f"R/Q: {q['R/Q [Ohm]']} Ohm")

Results as a table
******************
The dictionaries above are convenient for lookups but awkward to filter and plot
across polarisations. ``eigenmode.qois_df`` returns **every** mode of **every** solved
polarisation as a single :class:`pandas.DataFrame` — the same layout the mesh-convergence
study uses, so it slices the same way:

.. code-block:: python

    df = tesla.eigenmode.qois_df

Each row holds all the QOIs as columns, plus:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - ``m``
     - Azimuthal mode number (0 monopole, 1 dipole, 2 quadrupole, …)
   * - ``polarisation``
     - Its name
   * - ``mode``
     - Mode index within that polarisation
   * - ``mode_index``
     - ``'<m>-<mode>'`` — e.g. ``'0-0'`` monopole fundamental, ``'1-0'`` dipole fundamental

.. code-block:: python

    # the dipole modes, strongest coupling first
    df[df.polarisation == 'dipole'].nlargest(5, 'R/Q [Ohm]')

    # the accelerating (TM) monopole modes: TE modes have no E_z, so R/Q = 0
    mono = df[df.m == 0]
    mono[mono['R/Q [Ohm]'] > 1e-6]

See the worked examples: :doc:`examples/eigenmode/elliptical_tesla` (TESLA cell,
modes, impedance), :doc:`examples/eigenmode/pillbox` (analytic verification) and
:doc:`examples/eigenmode/dielectric_quartz_tube` (a dielectric-loaded cavity) and
:doc:`examples/eigenmode/dielectric_loss` (lossless vs perturbative vs fully lossy,
swept over :math:`\tan\delta`).

Visualisation
*************
Meshes and field profiles are properties of the individual ``Cavity`` objects:

.. code-block:: python

    # Plot the finite element mesh
    tesla.show_mesh()          # interactive NGSolve/webgui view

    # Plot the electric (E) and magnetic (H) field magnitudes for the fundamental mode
    tesla.show_fields(mode=1, which='E')
    tesla.show_fields(mode=1, which='H')
