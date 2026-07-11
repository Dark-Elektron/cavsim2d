"""Solver-agnostic wakefield backend protocol.

A backend runs a wakefield / impedance solve and reads the result into one
normalised schema, so callers (``cav.wakefield``, ``plot('zl')``, the ZL/ZT
objectives) never touch a solver-specific reader. ABCI is one backend; the
planned in-house NGSolve time-domain solver will be another — swapping them
changes only ``wakefield_config['solver']``.
"""
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Which normalised QOI keys belong to each polarisation folder. Longitudinal
# quantities are per-charge (V/pC); transverse ones per-charge-per-metre.
POL_QOI_KEYS = {
    'longitudinal': ('|k_loss| [V/pC]', 'k_FM [V/pC]', 'k_loss_HOM [V/pC]'),
    'transversal': ('|k_kick| [V/pC/m]',),
}


@dataclass
class WakefieldResult:
    """Normalised wakefield result, independent of the solver that produced it.

    ``wake_z`` / ``wake_t`` are DataFrames with columns ``f [MHz]``, ``|Z|``,
    ``Re(Z)``, ``Im(Z)`` (units ``[Ohm]`` longitudinal, ``[Ohm/m]`` transverse)
    and ``s [m]``, ``W`` (``[V/pC]`` / ``[V/pC/m]``). The impedance columns share
    the frequency grid; the wake columns share the ``s`` grid; the two grids are
    padded to equal length with NaN. ``qois`` is a flat dict of scalars
    (``|k_loss| [V/pC]``, ``k_FM [V/pC]``, ``k_loss_HOM [V/pC]``,
    ``|k_kick| [V/pC/m]``). Missing polarisations give empty frames / absent keys.
    """
    wake_z: pd.DataFrame = field(default_factory=pd.DataFrame)
    wake_t: pd.DataFrame = field(default_factory=pd.DataFrame)
    qois: dict = field(default_factory=dict)


class WakefieldBackend(ABC):
    """Run a wakefield solve and read it into a :class:`WakefieldResult`."""

    #: registry key; also the human name used in messages
    name = 'base'

    @abstractmethod
    def run(self, cav, config, subdir=''):
        """Execute the solve for *cav*. ``subdir`` (default ``''`` = the main
        run) selects a sub-run under ``<cav>/wakefield/`` — the operating-point /
        bunch-length sweep runs each combination in its own subdir."""

    @abstractmethod
    def read_dir(self, folder):
        """Read the raw output under *folder* into a :class:`WakefieldResult`."""

    def read(self, cav):
        """Read this cavity's main wakefield run."""
        return self.read_dir(str(self.wakefield_dir(cav)))

    # -- shared -------------------------------------------------------------

    @staticmethod
    def wakefield_dir(cav):
        return Path(cav.self_dir) / 'wakefield'

    def write_qois(self, cav, result=None):
        """Persist the normalised scalars to ``wakefield/<pol>/qois.json`` so
        ``cav.wakefield.qois`` can be read back without re-solving. Each
        polarisation folder gets only the keys that belong to it (see
        :data:`POL_QOI_KEYS`) and only when that folder exists (i.e. that
        polarisation was actually run."""
        if result is None:
            result = self.read(cav)
        base = self.wakefield_dir(cav)
        for pol, keys in POL_QOI_KEYS.items():
            pol_dir = base / pol
            if not pol_dir.is_dir():
                continue
            subset = {k: result.qois[k] for k in keys if k in result.qois}
            if subset:
                with open(pol_dir / 'qois.json', 'w') as f:
                    json.dump(subset, f, indent=2)

    @classmethod
    def read_qois(cls, cav):
        """Merge the persisted ``wakefield/<pol>/qois.json`` scalars into one
        dict (solver-agnostic; used by ``cav.wakefield.qois``)."""
        base = Path(cav.self_dir) / 'wakefield'
        merged = {}
        for pol in POL_QOI_KEYS:
            path = base / pol / 'qois.json'
            if path.exists():
                with open(path) as f:
                    merged.update(json.load(f))
        return merged
