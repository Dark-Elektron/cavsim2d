Advanced: uncertainty quantification everywhere
===============================================

cavsim2d was written so that **uncertainty quantification plugs into every
analysis**. A run is deterministic — one geometry, one answer — until you add a
``uq_config``. With it, the analysis is evaluated over a small Stroud-3
quadrature of the uncertain geometry and returns a **distribution** (mean and
standard deviation) instead of a single number. The same pattern turns:

- eigenmode into **eigenmode + UQ** (a spread on the figures of merit),
- tuning into **robust tuning** (the tuned design's spread),
- wakefield into **wakefield + UQ** (a spread on the ``ZL``/``ZT`` impedance),
- optimisation into **robust optimisation** (rank candidates by a mean + k·std
  objective, not just the nominal value).

*(Only multipacting does not yet take a* ``uq_config`` *.)*

These examples deliberately span several cavity types — **spline**, **RF gun**,
**flat-top**, **elliptical** — a good stress test of the model-agnostic
machinery.

.. toctree::
   :maxdepth: 1

   eigenmode_uq
   robust_tuning
   wakefield_uq
   robust_optimisation
