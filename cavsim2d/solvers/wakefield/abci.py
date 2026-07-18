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
# ABCI quotes its impedance in **kOhm** (kOhm/m transverse), so that is what the
# numbers in these frames are. They used to be labelled [Ohm], which made a
# wakefield spectrum sit a factor of 1000 below an eigenmode-reconstructed one
# plotted beside it. Only the label was wrong — the values are ABCI's, untouched.
# Use WakefieldSolver.impedance(unit=...) to get them in Ohm/MOhm instead.
_CURVES = {
    0: {'mag': 'Longitudinal Impedance Magnitude',
        're': 'Real Part of Longitudinal Impedance',
        'im': 'Imaginary Part of Longitudinal Impedance',
        'z': 'kOhm', 'w': 'V/pC', 'loss': 'Longitudinal',
        'kname': 'k_loss(f)', 'ku': 'V/pC'},
    1: {'mag': 'Transversal Impedance Magnitude',
        're': 'Real Part of Transverse Impedance',
        'im': 'Imaginary Part of Transverse Impedance',
        'z': 'kOhm/m', 'w': 'V/pC/m', 'loss': 'Transverse',
        'kname': 'k_kick(f)', 'ku': 'V/pC/m'},
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
        """One polarisation's impedance spectrum + wake potential + cumulative
        loss/kick-factor spectrum as a frame."""
        c = _CURVES[mrot]
        try:
            d = ABCIData(folder, '', mrot)
            f, zmag, _ = d.get_data(c['mag'])
            _, zre, _ = d.get_data(c['re'])
            _, zim, _ = d.get_data(c['im'])
            s, w, _ = d.get_data('Wake Potentials')
        except (FileNotFoundError, KeyError, IndexError):
            return pd.DataFrame()

        # Cumulative loss/kick factor k(F) vs frequency — ABCI's 'Loss Factor
        # Spectrum Integrated upto F' (kick factor for the transverse run). Its
        # own frequency grid, so it is padded like the wake columns; absent on
        # some runs, in which case those columns are simply omitted.
        fk, kcum = None, None
        try:
            fk, kcum, _ = d.get_data('Loss Factor Spectrum Integrated upto F')
        except (KeyError, IndexError):
            pass

        lengths = [len(f), len(s)] + ([len(fk)] if fk is not None else [])
        n = max(lengths)

        def pad(a):
            a = list(a)
            return a + [np.nan] * (n - len(a))

        zu, wu = c['z'], c['w']
        cols = {
            'f [MHz]': pad(np.asarray(f) * 1e3),        # ABCI f is GHz
            f'|Z| [{zu}]': pad(zmag),
            f'Re(Z) [{zu}]': pad(zre),
            f'Im(Z) [{zu}]': pad(zim),
            's [m]': pad(s),
            f'W [{wu}]': pad(w),
        }
        if fk is not None:
            cols['fk [MHz]'] = pad(np.asarray(fk) * 1e3)
            cols[f"{c['kname']} [{c['ku']}]"] = pad(kcum)
        return pd.DataFrame(cols)

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
