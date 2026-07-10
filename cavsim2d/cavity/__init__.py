"""Cavity package — the public façade, plus the pieces that are not models.

The cavity *models* themselves (``Cavity`` and its subclasses) now live in
:mod:`cavsim2d.models`. What remains here is everything that is not a model:

- :class:`Cavities` — the container that runs a study over several cavities
- :class:`Dakota` — the Dakota optimisation driver
- :class:`OperationPoints` — operating-point tables

Every model is re-exported below, so ``from cavsim2d.cavity import EllipticalCavity``
continues to work unchanged.
"""
from cavsim2d.models import (Cavity, EllipticalCavity, EllipticalCavityFlatTop,
                             SplineCavity, RFGun, Pillbox, CircularWaveguide)
from cavsim2d.cavity.cavities import Cavities
from cavsim2d.cavity.dakota import Dakota
from cavsim2d.cavity.operating_points import OperationPoints
from cavsim2d.cavity.welcome import show_welcome

__all__ = [
    'Cavity',
    'Cavities',
    'EllipticalCavity',
    'SplineCavity',
    'EllipticalCavityFlatTop',
    'RFGun',
    'Pillbox',
    'Dakota',
    'OperationPoints',
    'show_welcome',
    'CircularWaveguide',
]
