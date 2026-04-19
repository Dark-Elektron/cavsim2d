"""
cavsim2d - 2D axisymmetric RF cavity simulation toolkit.

Main entry points:
    Cavities           - Container for multiple Cavity objects
    EllipticalCavity   - Standard elliptical cavity
    EllipticalCavityFlatTop - Flat-top elliptical cavity
    SplineCavity       - Spline-based cavity
    RFGun              - RF gun cavity
    Pillbox            - Pillbox cavity

Solver objects (attached as lazy properties on Cavity/Cavities):
    cav.tune           - TuneSolver
    cav.eigenmode      - EigenmodeSolver
    cav.wakefield      - WakefieldSolver
    cavs.optimisation  - OptimisationSolver
"""
from cavsim2d.cavity import (
    Cavities,
    Cavity,
    EllipticalCavity,
    EllipticalCavityFlatTop,
    SplineCavity,
    RFGun,
    Pillbox,
    Dakota,
    OperationPoints,
    show_welcome,
)

from cavsim2d.solvers import (
    TuneSolver,
    EigenmodeSolver,
    WakefieldSolver,
    OptimisationSolver,
    EigenmodeResult,
)

__all__ = [
    'Cavities',
    'Cavity',
    'EllipticalCavity',
    'EllipticalCavityFlatTop',
    'SplineCavity',
    'RFGun',
    'Pillbox',
    'Dakota',
    'OperationPoints',
    'show_welcome',
    'TuneSolver',
    'EigenmodeSolver',
    'WakefieldSolver',
    'OptimisationSolver',
    'EigenmodeResult',
]
