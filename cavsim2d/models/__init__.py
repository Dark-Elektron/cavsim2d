"""Cavity models — the simulatable axisymmetric structures.

Each model owns its geometry (a :class:`~cavsim2d.geometry.Profile`, or a gmsh
``.geo`` writer) and inherits the run/tune/plot machinery from :class:`Cavity`.

This package holds *only* models. The container (:class:`~cavsim2d.cavity.Cavities`),
the Dakota driver and the operating-point tables live in :mod:`cavsim2d.cavity`.

Models are re-exported from :mod:`cavsim2d.cavity` too, so
``from cavsim2d.cavity import EllipticalCavity`` keeps working.
"""
from cavsim2d.models.base import Cavity
from cavsim2d.models.elliptical import EllipticalCavity
from cavsim2d.models.elliptical_flattop import EllipticalCavityFlatTop
from cavsim2d.models.spline import SplineCavity
from cavsim2d.models.rfgun import RFGun
from cavsim2d.models.pillbox import Pillbox
from cavsim2d.models.circular_waveguide import CircularWaveguide

__all__ = [
    'Cavity',
    'EllipticalCavity',
    'EllipticalCavityFlatTop',
    'SplineCavity',
    'RFGun',
    'Pillbox',
    'CircularWaveguide',
]
