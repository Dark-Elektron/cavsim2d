"""Unified eigenmode result container."""
import json
import numpy as np


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
