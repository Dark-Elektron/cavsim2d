"""Secondary electron yield (SEY) table for multipacting.

Ported from PyMultipact's ``domain.py``. The SEY delta(E) curve decides how many
secondary electrons an impact of energy E releases; multipacting grows when
delta > 1 over a resonant band. The table is read from a two-column file
(impact energy [eV], yield). A bundled copper-like default ships with cavsim2d.
"""
import os

import numpy as np
import pandas as pd

# Bundled default SEY table (copper-like), resolved relative to this package so
# it works from any working directory once cavsim2d is installed.
DEFAULT_SEY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'data_module', 'sey', 'sey')


class _TableInterp:
    """Picklable linear table interpolation.

    Multiprocessing passes the SEY object through process arguments, so a
    lambda/closure would break the parallel path. Linear interpolation matches
    how MultiPac treats secy files; a CubicSpline oscillated wildly across the
    huge gap up to the 1e12 eV sentinel row and poisoned the recorded yields
    (which are diagnostics only -- they never feed back into the dynamics)."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def __call__(self, xq):
        return np.interp(xq, self.x, self.y)


class SEY:
    """Secondary emission yield curve delta(E) loaded from a two-column file."""

    def __init__(self, sey_filepath=None):
        if sey_filepath is None:
            sey_filepath = DEFAULT_SEY
        self.filepath = str(sey_filepath)
        self.data = pd.read_csv(self.filepath, sep=r'\s+', engine='python',
                                header=None, names=["E", "sey"])
        self.Emax = max(self.data['E'])
        self.Emin = min(self.data['E'])
        self.sey = _TableInterp(self.data['E'], self.data['sey'])
