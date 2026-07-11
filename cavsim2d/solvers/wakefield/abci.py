"""ABCI wakefield backend — wraps the bundled ABCI solver and its ``.top`` reader
behind the solver-agnostic :class:`WakefieldBackend` protocol."""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cavsim2d.data_module.abci_data import ABCIData
from cavsim2d.solvers.ABCI.abci import ABCI
from cavsim2d.solvers.wakefield.base import WakefieldBackend, WakefieldResult

# Per-polarisation ABCI curve titles and result units.
_CURVES = {
    0: {'mag': 'Longitudinal Impedance Magnitude',
        're': 'Real Part of Longitudinal Impedance',
        'im': 'Imaginary Part of Longitudinal Impedance',
        'z': 'Ohm', 'w': 'V/pC', 'loss': 'Longitudinal'},
    1: {'mag': 'Transversal Impedance Magnitude',
        're': 'Real Part of Transverse Impedance',
        'im': 'Imaginary Part of Transverse Impedance',
        'z': 'Ohm/m', 'w': 'V/pC/m', 'loss': 'Transverse'},
}


class ABCIWakefield(WakefieldBackend):
    """Run ABCI and read its output into the normalised schema."""

    name = 'abci'

    def __init__(self):
        self._abci = ABCI()

    def run(self, cav, config, subdir=''):
        # geo_to_abc writes the ABCI deck from the cavity Profile (seam 2);
        # ABCI.solve runs the bundled executable and raises if it aborted.
        # subdir routes an operating-point sub-run to <cav>/wakefield/<subdir>/.
        if subdir:
            deck_folder = os.path.join(cav.self_dir, 'wakefield', subdir)
            cav.geo_to_abc(config, deck_folder)
            self._abci.solve(cav, config, os.path.join(subdir, 'wakefield'))
        else:
            cav.geo_to_abc(config)
            self._abci.solve(cav, config)

    def read_dir(self, folder):
        folder = str(folder)
        return WakefieldResult(wake_z=self._frame(folder, 0),
                               wake_t=self._frame(folder, 1),
                               qois=self._qois(folder))

    # -- reading ------------------------------------------------------------

    @staticmethod
    def _frame(folder, mrot):
        """One polarisation's impedance spectrum + wake potential as a frame."""
        c = _CURVES[mrot]
        try:
            d = ABCIData(folder, '', mrot)
            f, zmag, _ = d.get_data(c['mag'])
            _, zre, _ = d.get_data(c['re'])
            _, zim, _ = d.get_data(c['im'])
            s, w, _ = d.get_data('Wake Potentials')
        except (FileNotFoundError, KeyError, IndexError):
            return pd.DataFrame()
        n = max(len(f), len(s))

        def pad(a):
            a = list(a)
            return a + [np.nan] * (n - len(a))

        zu, wu = c['z'], c['w']
        return pd.DataFrame({
            'f [MHz]': pad(np.asarray(f) * 1e3),        # ABCI f is GHz
            f'|Z| [{zu}]': pad(zmag),
            f'Re(Z) [{zu}]': pad(zre),
            f'Im(Z) [{zu}]': pad(zim),
            's [m]': pad(s),
            f'W [{wu}]': pad(w),
        })

    @staticmethod
    def _qois(folder):
        """Normalised loss / kick factors.

        ``|k_loss|`` / ``|k_kick|`` are ``abs(loss_factor)`` — the exact
        quantities the operating-point loop used. ``k_FM`` (fundamental-mode
        loss) and ``k_loss_HOM`` mirror the objective's ``calc_k_loss`` call
        sequence."""
        q = {}
        if (Path(folder) / 'longitudinal' / 'cavity.top').exists():
            try:
                d = ABCIData(folder, '', 0)
                k_loss = abs(d.loss_factor.get('Longitudinal', math.nan))
                q['|k_loss| [V/pC]'] = k_loss
                try:
                    d.get_data('Real Part of Longitudinal Impedance')
                    # ABCI titles this 'upto' (no space); the space form silently
                    # fails to parse and leaves k_FM missing.
                    d.get_data('Loss Factor Spectrum Integrated upto F')
                    k_fm = float(d.y_peaks[0])
                    q['k_FM [V/pC]'] = k_fm
                    q['k_loss_HOM [V/pC]'] = k_loss - k_fm
                except (KeyError, IndexError):
                    pass
            except (FileNotFoundError, KeyError, IndexError):
                pass
        if (Path(folder) / 'transversal' / 'cavity.top').exists():
            try:
                d = ABCIData(folder, '', 1)
                q['|k_kick| [V/pC/m]'] = abs(d.loss_factor.get('Transverse', math.nan))
            except (FileNotFoundError, KeyError, IndexError):
                pass
        return q
