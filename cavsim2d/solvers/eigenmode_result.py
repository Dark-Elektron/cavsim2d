"""Unified eigenmode result container."""
import json
import os

import numpy as np

# Azimuthal mode number <-> polarisation folder name. Results of an m-pole
# eigenmode solve are stored in ``<cavity>/eigenmode/<pol_name(m)>/``
# (the monopole keeps its flat ``<cavity>/eigenmode/`` layout).
MPOLE_NAMES = {
    0: 'monopole',
    1: 'dipole',
    2: 'quadrupole',
    3: 'sextupole',
    4: 'octupole',
    5: 'decapole',
    6: 'dodecapole',
    7: 'tetradecapole',
    8: 'hexadecapole',
    9: 'octadecapole',
    10: 'icosapole',
    11: 'docosapole',
    12: 'tetracosapole',
    13: 'hexacosapole',
    14: 'octacosapole',
    15: 'triacontapole'
}
_MPOLE_M = {v: k for k, v in MPOLE_NAMES.items()}


def pol_name(m):
    """Folder name for azimuthal mode number *m* (1 -> 'dipole', ...)."""
    return MPOLE_NAMES.get(int(m), f'{2 * int(m)}-pole')


def pol_number(pol):
    """Azimuthal mode number from an int or name ('dipole' -> 1, ...)."""
    if isinstance(pol, str):
        key = pol.strip().lower()
        if key in _MPOLE_M:
            return _MPOLE_M[key]
        raise ValueError(f"Unknown polarisation '{pol}'. "
                         f"Use an integer m or one of {list(_MPOLE_M)}.")
    return int(pol)


def monopole_dir(eigenmode_folder):
    """Monopole results folder for a given ``eigenmode`` folder.

    Current layout is ``<eigenmode>/monopole/``; results produced during the
    flat-layout refactor era live directly in ``<eigenmode>/``, so fall back
    to that when no monopole subfolder exists."""
    sub = os.path.join(str(eigenmode_folder), 'monopole')
    return sub if os.path.isdir(sub) else str(eigenmode_folder)


class EigenmodeResult:
    """A single eigenmode with unified azimuthal mode number.

    Attributes
    ----------
    index : int
        Mode index within the solve.
    frequency : float
        Resonant frequency in MHz.
    m : int
        Azimuthal mode number (0 = monopole, 1 = dipole, ...).
    qois : dict
        Quantities of interest for this mode.
    """

    def __init__(self, index, frequency, m, qois, field_data=None):
        self.index = index
        self.frequency = frequency
        self.m = m
        self.qois = qois
        self._field_data = field_data

    def __repr__(self):
        return f"EigenmodeResult(index={self.index}, f={self.frequency:.2f} MHz, m={self.m})"

    def to_dict(self):
        return {
            'index': self.index,
            'frequency': self.frequency,
            'm': self.m,
            'qois': self.qois,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            index=d['index'],
            frequency=d['frequency'],
            m=d.get('m', 0),
            qois=d.get('qois', {}),
        )
