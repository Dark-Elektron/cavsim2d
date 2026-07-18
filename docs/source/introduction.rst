Introduction
************

This section introduces the numerical methods and solvers used in ``cavsim2d``
for conducting analysis on accelerating RF cavities. The toolkit provides capabilities
for eigenmode analysis, frequency tuning, wakefield/impedance analysis, shape
optimisation, uncertainty quantification (UQ), and post-processing of results.

Each module performs a different operation:

* Eigenmode analysis - NGSolve :cite:p:`NGSolve`
* Wakefield analysis - ABCI :cite:p:`ABCI`
* Optimisation - Python (Genetic Algorithm)
* Uncertainty quantification - Python (Quadrature/QMC via SALib/Dakota)
* Post-processing - Python (Matplotlib/Pandas)

Eigenmode Analysis
==================

Eigenmode analysis is performed using the NGSolve electromagnetic finite-element code :cite:p:`NGSolve`.
Unlike legacy workflows that require executing external binary utilities, ``cavsim2d`` wraps
NGSolve directly as a Python library. This enables end-to-end simulation pipelines—including
meshing, solving, and post-processing—to run entirely within Python in-memory.

The solver computes the fields, frequencies, and relevant RF figures of merit (stored energy,
accelerating/kick voltages, R/Q, surface power losses, geometric factor G, and field flatness)
for both standard monopole modes (:math:`m=0`) and higher-order multipole modes (:math:`m \ge 1`).

Wakefield Analysis
==================

Wakefield analysis is performed using the ABCI (Azimuthal Beam Cavity Interaction) code written by
Yang Ho Chin :cite:p:`ABCI`. It solves the Maxwell equations directly in the time domain
when a bunched beam goes through an axisymmetric structure on or off axis. It can be found
on the official `ABCI website <https://abci.kek.jp/abci.htm>`_.

Optimisation
============

Optimisation is carried out using self-written Python codes that call the previously mentioned analysis
codes. Currently, multi-objective Genetic Algorithm (GA) optimisations are supported to find Pareto-optimal
cavity geometries.

Uncertainty Quantification
==========================

Uncertainty quantification (UQ) propagates geometric tolerancing and alignment uncertainties
through the solvers to calculate the statistical mean and variance of the cavity objectives.
The underlying mathematics of the UQ methods can be found in the theory section.