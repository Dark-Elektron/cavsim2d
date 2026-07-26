Writing a new cavity model
==========================

Adding a geometry is deliberately small: implement a handful of methods on a
:class:`~cavsim2d.models.base.Cavity` subclass and it plugs into **every**
analysis — eigenmode, tuning, wakefield, multipacting, UQ and optimisation — with
no changes anywhere else. This page walks through the contract and ends with a
complete, runnable example.

The idea is the one from :doc:`geometry`: you describe the meridian **once** (a
labelled outline), and that single description drives every solver. The
per-analysis knobs (mesh size, bunch length, launch phases, …) all live in the
:doc:`configuration` dictionaries, so a new geometry never has to know which
analysis will consume it.

What you must provide
---------------------

Subclass :class:`~cavsim2d.models.base.Cavity` and supply:

1. **``self.parameters``** — a dict of the geometry's *scalar* parameters
   (millimetres). This **is** the parameterisation: every scalar entry
   automatically becomes a valid tune / UQ / optimisation variable. You get
   ``tune_variables()``, ``get_tune_value()`` and ``set_tune_value()`` for free
   from the base class, which just read and write this dict.

2. **``self.shape``** — a dict. The framework stores the target frequency in it
   during tuning (``self.shape['FREQ'] = ...``), so it must exist (any dict will
   do; conventionally ``{'IC': [...params...], 'BP': beampipe}``).

3. **``profile()``** — return a :class:`~cavsim2d.geometry.Profile`: the meridian
   as a chain of boundary segments in **metres**, each carrying a boundary label
   (``'PEC'`` wall, ``'PMC'`` aperture/symmetry plane), closed along the axis
   (``'AXI'``). Read ``self.parameters`` here — the tuner mutates it in place, so
   the profile must reflect the *live* values. Return ``None`` for a degenerate
   parameter set (the solver then reports the failure instead of meshing garbage).

4. **``create(n_cells, beampipe, mode)``** — provision the cavity's folders and
   point ``self.geo_filepath``. For a native (``profile()``-based) geometry, set
   ``self.geo_filepath = None`` and call ``self._write_geometry_snapshot()``; the
   solver meshes ``profile()`` directly (exact arcs, no gmsh round-trip). Only
   implement a ``.geo`` writer here if your geometry has no ``profile()``.

5. **``rebuild(parameters, beampipe=None)``** — return a *fresh, bare* instance of
   your model built from a parameter dict (same keys as ``self.parameters``). This
   is the one hook the generic machinery stands on: tuning, UQ and optimisation all
   reconstruct candidates through it, so implementing ``rebuild`` gives you all
   three for free.

The Profile API
---------------

A :class:`~cavsim2d.geometry.Profile` is built by walking the wall from the axis:

.. code-block:: python

    from cavsim2d.geometry import Profile
    p = Profile('my-cavity')
    p.start(z0, 0.0)                    # begin on the axis (r = 0)
    p.line_to(z, r, 'PMC')             # straight segment to (z, r) with a label
    p.arc_to(z, r, through, 'PEC')     # arc through an intermediate point
    p.ellipse_arc_to(...)              # exact ellipse arc (see the elliptical model)
    p.spline_to(...)                   # Bézier spline (see the spline model)
    p.close('AXI')                     # close back along the axis

Boundary labels are the only physics the outline carries:

- ``'PEC'`` — the metal **wall**;
- ``'PMC'`` — an **aperture / symmetry** plane (beam-pipe mouth, cell mid-plane);
- ``'AXI'`` — the axis, supplied by ``close('AXI')``.

One geometry, every analysis
----------------------------

You do **not** write anything analysis-specific in the geometry:

- **Eigenmode** meshes ``profile()`` and reads the ``PEC`` / ``PMC`` labels as
  boundary conditions.
- **Wakefield** writes an ABCI deck from the same ``profile()``; a beam pipe is
  added automatically at each end if the outline has none.
- **Multipacting** reuses the eigenmode field and mesh, reading the ``PEC``
  vertices as electron-emission sites.

Resolution and physics settings for each of these are passed in the
:doc:`configuration` dictionaries, not baked into the geometry.

Optional extras
---------------

- **Per-cell decomposition.** Set ``uses_cell_suffixes = True`` and implement
  ``half_cells()`` / ``set_half_cells()`` only if your geometry has independently
  varying cells (as the elliptical family does); this unlocks multicell tuning and
  multicell UQ. Most geometries do not need it.
- **Non-scalar parameters.** If a parameter is a coordinate pair rather than a
  scalar (as the spline's control points are), override ``expand_variable`` /
  ``get_tune_value`` / ``set_tune_value`` to address the sub-fields (e.g.
  ``'p2_r'``) — see :class:`~cavsim2d.models.spline.SplineCavity`.

A complete example
------------------

A single-cell cavity whose wall tapers straight from the aperture up to a central
equator and back — three parameters, all straight segments:

.. code-block:: python

    import os
    from cavsim2d.models.base import Cavity
    from cavsim2d.geometry import Profile


    class ConeCavity(Cavity):
        """A single-cell cone cavity. dims = [L, Req, Ri] in mm."""

        def __init__(self, dims, name='cone'):
            super().__init__(n_cells=1, beampipe='none', name=name)
            L, Req, Ri = dims
            self.n_cells = 1
            self.kind = 'cone'
            self.beampipe = 'none'
            self.parameters = {'L': float(L), 'Req': float(Req), 'Ri': float(Ri)}
            self.shape = {'IC': [L, Req, Ri], 'BP': 'none'}

        def profile(self):
            try:
                L, Req, Ri = (self.parameters[k] * 1e-3 for k in ('L', 'Req', 'Ri'))
            except (KeyError, TypeError, ValueError):
                return None
            p = Profile('cone')
            p.start(-L / 2, 0.0)
            p.line_to(-L / 2, Ri, 'PMC')     # left aperture plane
            p.line_to(0.0, Req, 'PEC')       # taper up to the equator
            p.line_to(L / 2, Ri, 'PEC')      # taper down to the right aperture
            p.line_to(L / 2, 0.0, 'PMC')     # right aperture plane
            p.close('AXI')                   # close along the axis
            return p

        def create(self, n_cells=None, beampipe=None, mode=None):
            if self.projectDir:
                self.self_dir = os.path.join(self.projectDir, self.name)
                os.makedirs(os.path.join(self.self_dir, 'geometry'), exist_ok=True)
                self.uq_dir = os.path.join(self.self_dir, 'uq')
                self.geo_filepath = None       # native-only: the solver meshes profile()
                self._write_geometry_snapshot()

        def rebuild(self, parameters, beampipe=None):
            return ConeCavity([parameters['L'], parameters['Req'], parameters['Ri']],
                              name=self.name)

That is the whole model. It now behaves like any built-in cavity:

.. code-block:: python

    from cavsim2d import Study

    cav = ConeCavity([100, 100, 20])          # L = Req = 100 mm, Ri = 20 mm
    study = Study('project_dir')
    study.add_cavity([cav], ['cone'])

    study.run_eigenmode({'polarisation': 'monopole', 'mesh_config': {'h': 8, 'p': 2}})
    print(cav.eigenmode.qois['freq [MHz]'])

    # tuning, UQ and optimisation all work through rebuild() + self.parameters:
    study.run_tune({'freqs': 1500.0, 'cell_type': {'mid-cell': 'Req'}})
    print(cav.tuned.eigenmode.qois['freq [MHz]'])   # ~1500 MHz

Because ``profile()`` labels its wall ``PEC`` and its apertures ``PMC``, the same
``ConeCavity`` also runs through the wakefield and multipacting analyses unchanged.
