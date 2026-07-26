API Reference
=============

The public API is what ``from cavsim2d import ...`` exposes: the study container,
the cavity models, and the analysis solver namespaces. Internal modules are
documented in their own docstrings but are not part of the supported surface.

Study
-----

.. autoclass:: cavsim2d.study.Study
   :members:
   :undoc-members:
   :show-inheritance:

Cavity models
-------------

All models inherit the run / tune / plot machinery from :class:`Cavity` and add
their own geometry parameterisation and tuning handles.

.. autoclass:: cavsim2d.models.base.Cavity
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: cavsim2d.models.elliptical.EllipticalCavity
   :members:
   :show-inheritance:

.. autoclass:: cavsim2d.models.elliptical_flattop.EllipticalCavityFlatTop
   :members:
   :show-inheritance:

.. autoclass:: cavsim2d.models.pillbox.Pillbox
   :members:
   :show-inheritance:

.. autoclass:: cavsim2d.models.rfgun.RFGun
   :members:
   :show-inheritance:

.. autoclass:: cavsim2d.models.spline.SplineCavity
   :members:
   :show-inheritance:

Analysis namespaces
-------------------

Reached as ``cav.eigenmode``, ``cav.wakefield``, ``cav.tune``,
``cav.multipacting`` and ``study.optimisation``.

.. autoclass:: cavsim2d.solvers.solver_objects.EigenmodeSolver
   :members:
   :show-inheritance:

.. autoclass:: cavsim2d.solvers.solver_objects.WakefieldSolver
   :members:
   :show-inheritance:

.. autoclass:: cavsim2d.solvers.solver_objects.TuneSolver
   :members:
   :show-inheritance:

.. autoclass:: cavsim2d.solvers.solver_objects.MultipactingSolver
   :members:
   :show-inheritance:

.. autoclass:: cavsim2d.solvers.solver_objects.OptimisationSolver
   :members:
   :show-inheritance:

Result objects
--------------

.. autoclass:: cavsim2d.solvers.eigenmode_result.EigenmodeResult
   :members:
   :show-inheritance:
