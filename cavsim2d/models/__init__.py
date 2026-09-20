"""Cavity models — the simulatable axisymmetric structures.

Each model owns its geometry (a :class:`~cavsim2d.geometry.Profile`, or a gmsh
``.geo`` writer) and inherits the run/tune/plot machinery from :class:`Cavity`.

This package holds *only* models. The multi-device manager
(:class:`~cavsim2d.study.Study`) lives in :mod:`cavsim2d.study`, the uncertainty
quantification and sensitivity analysis in :mod:`cavsim2d.analysis.uq`, and the
operating-point tables in :mod:`cavsim2d.data_module`.

The public entry point is the top-level package, so
``from cavsim2d import EllipticalCavity, Study`` is the supported import.
"""
from cavsim2d.models.base import Cavity
from cavsim2d.models.elliptical import EllipticalCavity
from cavsim2d.models.elliptical_flattop import EllipticalCavityFlatTop
from cavsim2d.models.spline import SplineCavity
from cavsim2d.models.rfgun import RFGun
from cavsim2d.models.pillbox import Pillbox
from cavsim2d.models.beampipe import Beampipe, CircularWaveguide
from cavsim2d.models.bellows import Bellows
from cavsim2d.models.bla import BLA
from cavsim2d.models.taper import Taper
from cavsim2d.models.assembly import Assembly

__all__ = [
    'Cavity',
    'EllipticalCavity',
    'EllipticalCavityFlatTop',
    'SplineCavity',
    'RFGun',
    'Pillbox',
    'Beampipe',
    'Bellows',
    'BLA',
    'Taper',
    'Assembly',
    'CircularWaveguide',
]
