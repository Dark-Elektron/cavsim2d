"""Helpers the solver namespaces share."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cavsim2d.constants import BOUNDARY_CONDITIONS_DICT
from cavsim2d.analysis.impedance import prefix_factor


def _require_vacuum(cavity, analysis):
    """Raise if *cavity* has dielectric regions.

    Only the eigenmode solver models materials. The others would happily mesh the
    cavity and return a perfectly plausible **vacuum** answer for a dielectric-loaded
    structure, so this fails loudly instead.
    """
    diel = list(getattr(cavity, 'dielectrics', ()) or ())
    if diel:
        names = [d['material'] for d in diel]
        raise NotImplementedError(
            f"cavity {cavity.name!r} has dielectric region(s) {names}, which the "
            f"{analysis} solver does not model — it would silently solve the vacuum "
            f"problem and report it as a result. Dielectrics are supported by the "
            f"eigenmode solver only. Use cav.clear_dielectrics() to run {analysis} on "
            "the vacuum geometry.")


def _solved_with_open_boundary(cfg, modes=None):
    """True if an eigenmode run used an OPEN (PML) end, so its Q is a real radiation Q.

    Two sources of evidence, because neither alone is reliable:

    - the saved ``boundary_conditions`` (digit 2 = open/PML, 4 = waveguide port), and
    - ``Q_rad []`` or ``Q_ext []`` in the results, which only an open (PML) or a port
      solve produces.

    The results are checked first and are the stronger evidence: they are what the solver
    actually did, whereas ``config.json`` is written by ``EigenmodeSolver.run`` and not by
    every path that can produce results (``Study.run_eigenmode`` does not write one), so a
    perfectly good open solve can arrive with an empty config.
    """
    for col in ('Q_rad []', 'Q_ext []'):
        if modes is not None and col in getattr(modes, 'columns', ()):
            if modes[col].notna().any():
                return True
    bc = (cfg or {}).get('boundary_conditions')
    if bc is None:
        return False
    if isinstance(bc, str):
        bc = BOUNDARY_CONDITIONS_DICT.get(bc.strip().lower(), 33)
    try:
        digits = f'{int(bc):02d}'
        return '2' in digits or '4' in digits
    except (TypeError, ValueError):
        return False


#: The caveat on every impedance reconstructed from eigenmodes. See
#: EigenmodeSolver._warn_mode_filtering for where the numbers behind it come from.
_MODE_FILTER_CAVEAT = (
    "The filtering is not perfect: genuine cavity modes may have been discarded (a Q "
    "that has not converged with pipe length, or a mode the second solve missed) and "
    "some beam-pipe or PML modes may still be included. Treat |Z| above the beam-pipe "
    "cutoff as indicative and check it against a wakefield solve.")


def _z_axis_label(unit, transverse=False):
    """Mathtext y-label for an impedance axis, e.g. r'$|Z_\\parallel|$ [k$\\Omega$]'."""
    _, prefix = prefix_factor(unit)
    if transverse:
        return rf'$|Z_\perp|$ [{prefix}$\Omega$/m]'
    return rf'$|Z_\parallel|$ [{prefix}$\Omega$]'


def _maybe_show(show):
    """Display the current figure when *show* is True.

    Every ``plot_*`` method defaults to ``show=True`` so a plot appears without
    a manual ``plt.show()`` (the common interactive case, and notebooks with the
    non-interactive ``inline`` backend only render on show). Pass ``show=False``
    to keep the axes live for overlaying/compositing or to reuse the returned
    ``ax``; the study-level overlays pass it through for exactly that reason."""
    if show:
        plt.show()


def _has_mesh(folder):
    """A saved eigenmode mesh exists in *folder* — the round-tripping ``.vol``
    (current format) or the legacy ``mesh.pkl``."""
    return any((Path(folder) / f).exists() for f in ('mesh.vol', 'mesh.pkl'))


def _has_field(folder):
    """A saved eigenmode field exists in *folder* — the ``field_meta.json`` +
    projected vectors (current format) or the legacy ``gfu_EH.pkl`` pickle."""
    return any((Path(folder) / f).exists() for f in ('field_meta.json', 'gfu_EH.pkl'))


def _reference_series(reference):
    """Normalise a plot ``reference`` into a list of ``(label, mean, std|None)``.

    Accepts a ``pandas.DataFrame`` (each column a mean-only series), or a mapping
    of ``label -> {'mean'/'avg'/'expe': [...], 'std'/'sd'/'stdDev': [...]}`` or
    ``label -> [values]`` (mean only). Used by the reference-overlay plots."""
    if reference is None:
        return []
    if isinstance(reference, pd.DataFrame):
        return [(str(col), reference[col].to_numpy(), None)
                for col in reference.columns]
    if isinstance(reference, dict):
        mean_keys = ('mean', 'avg', 'average', 'expe')
        std_keys = ('std', 'sd', 'stddev', 'stdev')
        out = []
        for label, val in reference.items():
            if isinstance(val, dict):
                lower = {k.lower(): v for k, v in val.items()}
                mean = next((lower[k] for k in mean_keys if k in lower), None)
                std = next((lower[k] for k in std_keys if k in lower), None)
                if mean is None:
                    continue
                std = np.asarray(std, float) if std is not None else None
                out.append((str(label), np.asarray(mean, float), std))
            else:
                out.append((str(label), np.asarray(val, float), None))
        return out
    raise TypeError("reference must be a DataFrame or a mapping of label -> series.")


def _gaussian_sum(means, stds, grid):
    """Sum of area-normalised Gaussians (one per mode) on *grid* — the
    'probability of resonance' density for a spectrum given per-mode mean/std."""
    y = np.zeros_like(grid, dtype=float)
    for mu, sd in zip(np.asarray(means, float), np.asarray(stds, float)):
        sd = max(float(sd), 1e-12)
        y += np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))
    return y


def _accelerating_mode_row(qois_df, ref_freq=None):
    """The accelerating (pi) monopole mode's QoI row, chosen robustly.

    The pi-mode carries the largest on-axis field, so it is the maximum-R/Q
    ``m == 0`` mode -- but a naive ``idxmax`` can be captured by a spurious
    near-zero-frequency eigenmode (the solver occasionally returns one) or, in a
    weakly-coupled multicell whose fundamental passband is tightly packed, by a
    neighbouring band. Restrict to a physical window around ``ref_freq`` (the tune
    target, when known) so neither a ~0 MHz mode nor a higher passband can win,
    then take the max-R/Q mode. Without ``ref_freq``, just drop the sub-1-MHz
    spurious modes before the max-R/Q pick.
    """
    d = qois_df.query('m == 0')
    if ref_freq:
        rf = float(ref_freq)
        band = d[(d['freq [MHz]'] > 0.5 * rf) & (d['freq [MHz]'] < 1.5 * rf)]
        if not band.empty:
            d = band
    phys = d[d['freq [MHz]'] > 1.0]
    if not phys.empty:
        d = phys
    return d.loc[d['R/Q [Ohm]'].idxmax()]
