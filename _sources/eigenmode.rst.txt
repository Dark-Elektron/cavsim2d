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
modes, impedance) and :doc:`examples/eigenmode/pillbox` (analytic verification).

Visualisation
*************
Meshes and field profiles are properties of the individual ``Cavity`` objects:

.. code-block:: python

    # Plot the finite element mesh
    tesla.show_mesh()          # interactive NGSolve/webgui view

    # Plot the electric (E) and magnetic (H) field magnitudes for the fundamental mode
    tesla.show_fields(mode=1, which='E')
    tesla.show_fields(mode=1, which='H')
