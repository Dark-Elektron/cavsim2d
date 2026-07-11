"""Backward-compatibility façade — prefer ``from cavsim2d import ...``.

Everything now lives at its natural home: the *models* (``Cavity`` and subclasses)
in :mod:`cavsim2d.models`, the multi-device manager :class:`Study` in
:mod:`cavsim2d.study` (``Cavities`` is a deprecated alias), operating-point tables
in :mod:`cavsim2d.data_module`, the Dakota driver in :mod:`cavsim2d.analysis.uq`,
the banner in :mod:`cavsim2d.utils`. This module just re-exports them so existing
``from cavsim2d.cavity import EllipticalCavity`` / ``Cavities`` imports keep working.
"""
from cavsim2d.models import (Cavity, EllipticalCavity, EllipticalCavityFlatTop,
                             SplineCavity, RFGun, Pillbox, CircularWaveguide)
from cavsim2d.study import Study, Cavities
from cavsim2d.analysis.uq.dakota import Dakota
from cavsim2d.data_module.operating_points import OperationPoints
from cavsim2d.utils.welcome import show_welcome

__all__ = [
    'Cavity',
    'Study',
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
