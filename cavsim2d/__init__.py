"""
cavsim2d - 2D axisymmetric RF cavity simulation toolkit.

Main entry points:
    Study              - Container for multiple Cavity objects
    EllipticalCavity   - Standard elliptical cavity
    EllipticalCavityFlatTop - Flat-top elliptical cavity
    SplineCavity       - Spline-based cavity
    RFGun              - RF gun cavity
    Pillbox            - Pillbox cavity

Solver objects (attached as lazy properties on Cavity/Study):
    cav.tune           - TuneSolver
    cav.eigenmode      - EigenmodeSolver
    cav.wakefield      - WakefieldSolver
    cavs.optimisation  - OptimisationSolver
"""
from cavsim2d.models import (
    Cavity,
    EllipticalCavity,
    EllipticalCavityFlatTop,
    SplineCavity,
    RFGun,
    Pillbox,
    CircularWaveguide,
)
from cavsim2d.study import Study
from cavsim2d.analysis.uq.dakota import Dakota
from cavsim2d.data_module.operating_points import OperationPoints
from cavsim2d.utils.welcome import show_welcome

from cavsim2d.solvers import (
    TuneSolver,
    EigenmodeSolver,
    WakefieldSolver,
    OptimisationSolver,
    EigenmodeResult,
)

from cavsim2d.utils.style import apply_style, house_style, WARM

__all__ = [
    'Study',
    'Cavity',
    'EllipticalCavity',
    'EllipticalCavityFlatTop',
    'SplineCavity',
    'RFGun',
    'Pillbox',
    'Dakota',
    'OperationPoints',
    'show_welcome',
    'CircularWaveguide',
    'TuneSolver',
    'EigenmodeSolver',
    'WakefieldSolver',
    'OptimisationSolver',
    'EigenmodeResult',
    'apply_style',
    'house_style',
    'WARM',
]
