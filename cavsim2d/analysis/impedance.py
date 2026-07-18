"""Impedance reconstructed from eigenmode results (equivalent-circuit model).

Each resonant mode is treated as a parallel RLC resonator, so a handful of
eigenmode quantities — the resonant frequency, the R/Q and the Q — reproduce the
impedance spectrum the beam sees around those resonances::

    Z(f) = sum_i  R_i / (1 + i Q_i (f/f_i - f_i/f))          [Ohm]

with the shunt impedance ``R = 1/2 Q (R/Q)``.

This is a *reconstruction*, not a wakefield simulation: it contains exactly the
modes that were solved for and nothing else — no broadband/resistive-wall
contribution, and nothing above the highest computed mode. Within that range it
is cheap, has no wake-length truncation, and lets an eigenmode result be compared
directly against a beam spectrum or against a wakefield solve (the frames use the
same columns as ``cav.wakefield.wake_z`` / ``wake_t``).

The cavity is axisymmetric, so the two transverse planes are degenerate — a
single transverse R/Q describes both, and the x/y average collapses to it.
"""
import re

import numpy as np
import pandas as pd

C0 = 299792458.0        # speed of light [m/s]

# Impedance is quoted in kOhm as often as Ohm, so the unit is an argument rather
# than a convention to memorise: '' = Ohm, 'k' = kOhm, 'M' = MOhm, 'G' = GOhm
# (per metre for the transverse impedance).
SI_PREFIXES = {'': 1.0, 'k': 1e3, 'M': 1e6, 'G': 1e9}

# The unit the wakefield backends store their impedance in (ABCI's native kOhm).
# The eigenmode reconstruction is computed in SI Ohm and converted on the way out,
# so both end up in the same default unit.
NATIVE_Z_UNIT = 'k'


def prefix_factor(unit):
    """``(divisor, prefix)`` for an SI prefix. ``'k'`` -> ``(1e3, 'k')``."""
    u = '' if unit in (None, 'Ohm', 'ohm') else str(unit)
    if u not in SI_PREFIXES:
        raise ValueError(
            f"unknown unit prefix {unit!r}; use one of {sorted(SI_PREFIXES)} "
            f"— '' for Ohm, 'k' for kOhm, 'M' for MOhm.")
    return SI_PREFIXES[u], u


def impedance_unit(unit, transverse=False):
    """Column/axis unit string, e.g. ``'kOhm'`` or ``'kOhm/m'``."""
    _, u = prefix_factor(unit)
    return f'{u}Ohm/m' if transverse else f'{u}Ohm'


def frame_unit(df, default=None):
    """The SI prefix a frame *declares* in its ``|Z|`` column.

    ``'|Z| [kOhm]'`` -> ``'k'``, ``'|Z| [Ohm/m]'`` -> ``''``. Read the unit from
    the data rather than assuming a backend's convention: a backend that reports
    in Ohm and one that reports in kOhm then both convert correctly.
    """
    if df is None or df.empty:
        return NATIVE_Z_UNIT if default is None else default
    col = next((c for c in df.columns if c.startswith('|Z|')), None)
    if col is None:
        return NATIVE_Z_UNIT if default is None else default
    match = re.search(r'\[([kMG]?)Ohm', col)
    return match.group(1) if match else (NATIVE_Z_UNIT if default is None else default)


def convert_impedance_frame(df, from_unit, to_unit, transverse=False):
    """Rescale and relabel the ``|Z|``/``Re(Z)``/``Im(Z)`` columns of a frame.

    Columns are matched on their stem, so this works whatever prefix the source
    frame declares. Other columns (``f [MHz]``, ``s [m]``, ``W``) pass through.
    """
    if df is None or df.empty:
        return df
    f_from, _ = prefix_factor(from_unit)
    f_to, _ = prefix_factor(to_unit)
    if f_from == f_to:
        return df
    ratio = f_from / f_to
    label = impedance_unit(to_unit, transverse)

    out = df.copy()
    renames = {}
    for stem in ('|Z|', 'Re(Z)', 'Im(Z)'):
        col = next((c for c in out.columns if c.startswith(stem)), None)
        if col is None:
            continue
        out[col] = out[col] * ratio
        renames[col] = f'{stem} [{label}]'
    return out.rename(columns=renames)


def impedance_frame(f_mhz, z, unit='k', transverse=False):
    """Standard impedance frame from a complex spectrum *z* in **Ohm** (SI)."""
    factor, _ = prefix_factor(unit)
    label = impedance_unit(unit, transverse)
    return pd.DataFrame({
        'f [MHz]': f_mhz,
        f'|Z| [{label}]': np.abs(z) / factor,
        f'Re(Z) [{label}]': z.real / factor,
        f'Im(Z) [{label}]': z.imag / factor,
    })


def reconstruct_impedance(freqs, r_over_q, q_factors, f_span, transverse=False):
    """Sum-of-resonators impedance spectrum. All frequencies in **Hz**.

    Parameters
    ----------
    freqs : array-like
        Resonant frequency of each mode [Hz].
    r_over_q : array-like
        R/Q of each mode [Ohm]. Longitudinal for ``transverse=False``; for
        ``transverse=True`` this is the transverse (Panofsky-Wenzel) R/Q, also
        in Ohm — the ``omega/c`` factor below turns it into Ohm/m.
    q_factors : array-like
        Quality factor of each mode. Pass the *loaded* Q if that is what the
        beam sees; the eigenmode solver reports the unloaded Q0.
    f_span : array-like
        Frequencies to evaluate the spectrum at [Hz]. ``f = 0`` is allowed and
        is evaluated by its limit.
    transverse : bool
        Transverse (dipole) impedance [Ohm/m] instead of longitudinal [Ohm].

    Returns
    -------
    ndarray of complex
        The impedance at each frequency in *f_span*: Ohm (longitudinal) or
        Ohm/m (transverse).
    """
    f0 = np.asarray(freqs, dtype=float)
    roq = np.asarray(r_over_q, dtype=float)
    q = np.asarray(q_factors, dtype=float)
    f = np.asarray(f_span, dtype=float)

    if not (f0.shape == roq.shape == q.shape):
        raise ValueError(f"freqs, r_over_q and q_factors must have the same length; "
                         f"got {f0.shape}, {roq.shape}, {q.shape}.")
    if np.any(f0 <= 0):
        raise ValueError("mode frequencies must be positive.")

    # Shunt impedance of each resonator. The transverse case picks up omega_0/c,
    # which is what converts a transverse R/Q [Ohm] into a shunt impedance [Ohm/m].
    r_shunt = 0.5 * q * roq
    if transverse:
        r_shunt = r_shunt * (2 * np.pi * f0 / C0)

    z = np.zeros(f.shape, dtype=complex)
    live = f > 0                      # f = 0 is singular in the resonator term
    f_live = f[live]

    for fi, ri, qi in zip(f0, r_shunt, q):
        # Breit-Wigner resonator: 1 + i Q (f/f0 - f0/f)
        denom = 1 + 1j * qi * (f_live / fi - fi / f_live)
        term = ri / denom
        if transverse:
            term = term * (fi / f_live)
        z[live] += term

    if not live.all():
        # DC limit. denom -> -i Q f0/f as f -> 0, so the longitudinal term tends
        # to 0 while the transverse term tends to R/(-iQ) = i R/Q (finite).
        z[~live] = np.sum(1j * r_shunt / q) if transverse else 0.0

    return z
