Geometry and parameterisation
=============================

Every cavity is an **axisymmetric** structure, so its geometry is fully described
by the **meridian** — the 2D outline of the wall in the ``(z, r)`` half-plane
(``r >= 0``). cavsim2d builds that outline once, from a small set of named
parameters, and the *same* outline drives every analysis (eigenmode, wakefield,
multipacting). There is no separate geometry per solver — see
:ref:`geometry-one-for-all` below.

The unified geometry model
--------------------------

A model turns its parameters into a :class:`~cavsim2d.geometry.Profile`: a chain
of boundary segments (lines and exact ellipse / circle / spline arcs) walked from
the axis, around the wall, and back. Each segment carries a **boundary-condition
label**:

- ``PEC`` — a perfect *electric* conductor: the cavity **wall**.
- ``PMC`` — a perfect *magnetic* conductor: an **aperture / symmetry plane**
  (the beam-pipe mouth, or a cell mid-plane in a reduced tuning geometry).
- the **axis** ``r = 0`` is the axisymmetry line and is handled implicitly.

These labels are the only "physics" the geometry carries; the solver reads them to
set boundary conditions. ``cav.plot('geometry')`` draws the meridian (the upper
half only — the analysed domain).

.. _geometry-one-for-all:

One geometry, every analysis
----------------------------

The single meridian serves all three analyses, and the differences between them
live entirely in the **configuration dictionaries** (see
:doc:`configuration`), never in the geometry:

- **Eigenmode** meshes the profile directly (native ``netgen.occ``) or via a
  ``.geo`` file; mesh resolution is ``eigenmode_config['mesh_config']`` (``h``, ``p``).
- **Wakefield** writes an ABCI deck from the *same* profile. ABCI requires a beam
  pipe at each end, so the writer adds one automatically if the geometry has none
  — you do **not** author a wakefield-specific shape. Bunch, wake length and
  rotation are ``wakefield_config`` keys.
- **Multipacting** reuses the eigenmode field and mesh: it reads the ``PEC``
  wall vertices as emission sites and tracks electrons in the solved field. It
  needs no geometry of its own; its resolution can be refined with
  ``multipacting_config['pec_maxh']`` / ``mesh_config``.

So a new geometry only has to produce a correct labelled meridian *once*; it then
works in every analysis (see :doc:`extending`).

Built-in parameterisations
--------------------------

All lengths are in **millimetres**.

Elliptical cavity (:class:`~cavsim2d.models.elliptical.EllipticalCavity`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The workhorse SRF shape. A **half-cell** (one quarter of a full cell's meridian)
is seven parameters ``[A, B, a, b, Ri, L, Req]``:

- ``A``, ``B`` — the *z* and *r* semi-axes of the **equator** ellipse (outer wall).
- ``a``, ``b`` — the *z* and *r* semi-axes of the **iris** ellipse (near the aperture).
- ``Ri``      — the iris / aperture (beam-pipe) radius.
- ``L``       — the half-cell length (iris plane to equator plane); a full cell is ``2 L``.
- ``Req``     — the equator radius.

The wall inclination ``alpha`` is *derived* from these (an optional 8th slot is
accepted and ignored on input). A full cell mirrors the half-cell about the
equator plane and the axis; a multi-cell cavity chains ``n_cells`` of them::

    EllipticalCavity(n_cells, mid_cell, end_cell_left, end_cell_right, beampipe='both')

``mid_cell`` sets the interior cells; ``end_cell_left`` / ``end_cell_right`` set
the two ends (each defaults to ``mid_cell``). See :ref:`geometry-multicell`.

Flat-top elliptical (:class:`~cavsim2d.models.elliptical_flattop.EllipticalCavityFlatTop`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An elliptical cell with a straight **flat section at the equator**, taking the
same per-cell parameterisation as the elliptical cavity. Used where a flattened
equator is wanted (e.g. some low-beta or crab geometries).

Pillbox (:class:`~cavsim2d.models.pillbox.Pillbox`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A right-cylinder cavity, five parameters ``[L, Req, Ri, S, L_bp]``:

- ``L``    — cell (barrel) length along the axis.
- ``Req``  — cavity (barrel) radius.
- ``Ri``   — iris / aperture (beam-pipe) radius.
- ``S``    — inter-cell drift length (the straight gap between adjacent cells at ``Ri``; 0 to butt cells together).
- ``L_bp`` — beam-pipe length added at each end selected by ``beampipe``.

::

    Pillbox(n_cells, [L, Req, Ri, S, L_bp], beampipe='both')

Spline cavity (:class:`~cavsim2d.models.spline.SplineCavity`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A free-form wall defined by six Bézier **control points** ``p0 .. p5``, each a
``[z, r]`` coordinate pair::

    SplineCavity({'geometry': {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
                               'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}})

Each coordinate is individually addressable as a tune / UQ variable named
``p<i>_z`` and ``p<i>_r`` (e.g. ``'p2_r'``, ``'p3_z'``). Placing a control point
at the pipe radius gives a *C1*-continuous (smooth, non-sharp) iris.

RF gun (:class:`~cavsim2d.models.rfgun.RFGun`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A photoinjector / VHF-gun profile built from a dictionary of named segment
lengths, radii and angles (``y1, R2, T2, L3, R4, L5, R6, L7, R8, T9, R10, T10,
L11, R12, L13, R14, x``)::

    RFGun({'geometry': {...}}, beampipe='none')

Every entry is a scalar tune / UQ variable. A cathode beam pipe can be added with
``beampipe`` for wakefield studies.

Circular waveguide (``CircularWaveguide``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A plain cylinder ``CircularWaveguide(R, L)`` (radius, length) — a minimal
reference geometry, useful for validating mode frequencies against analytics.

.. _geometry-multicell:

The multicell parameterisation
------------------------------

Only the elliptical families decompose into cells; the canonical multicell
representation is the **half-cell array**, ``cav.half_cells()``: a
``(2 * n_cells, 7)`` array whose row ``2k`` is the forward half of cell ``k+1``
and row ``2k+1`` its backward half.

Cells share geometry at their seams, and continuity is enforced:

- ``Req`` is shared by a cell's two halves (its equator) — ``n_cells`` values.
- ``Ri`` is shared across an iris plane — ``n_cells + 1`` values (two apertures
  and ``n_cells - 1`` internal irises).
- ``A, B, a, b, L`` are free per half-cell — ``2 * n_cells`` values each.

Two ways to name multicell parameters:

- **By cell block** (the default). Because ``uses_cell_suffixes`` is set for
  elliptical cavities, a bare tune variable like ``'Req'`` is resolved through the
  cell type: ``Req_m`` (mid-cells), ``Req_el`` (end-cell-left), ``Req_er``
  (end-cell-right). This is what a ``tune_config['cell_type']`` mapping uses — see
  :doc:`tuning`.
- **Per half-cell** (independent). ``cav.set_half_cells(array)`` installs an
  explicit, independently-varying half-cell array (honouring the continuity
  constraints above). This is what multicell **UQ** perturbs — every free entry
  becomes its own random variable — see :doc:`uq`.

Because a multicell UQ variant is expressed purely through ``half_cells()``, the
native ``profile()`` renders it directly, so it flows into eigenmode *and*
wakefield without any special writer.
