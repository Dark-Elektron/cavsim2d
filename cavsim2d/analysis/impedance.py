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


def reconstruct_impedance_qnm(freqs, r_over_q_complex, q_factors, f_span,
                              transverse=False):
    """Pole-expansion impedance from **complex** mode residues. Frequencies in Hz.

    :func:`reconstruct_impedance` gives every mode a positive-real shunt impedance,
    which is exact for an isolated resonance and wrong for overlapping ones: it makes
    neighbouring poles add in phase. Above a beam-pipe cutoff, where radiating modes
    have Q of order 10 and linewidths wider than the mode spacing, that coherent
    addition overestimates ``|Z|`` by roughly the number of modes overlapping at each
    frequency.

    .. warning::
       **Experimental, and currently not validated on radiating modes.** The residue
       *magnitudes* are exact — ``|(R/Q)~|`` reproduces ``R/Q`` to machine precision —
       but their *phases* do not yet come out physical: on a PML solve of a TESLA
       3-cell the resulting spectrum violates passivity (``Re Z < 0``) over most of
       the band above cutoff, and is further from a wakefield reference than the plain
       RLC sum. Correct QNM normalisation of a PML eigenvector needs more than the
       unconjugated volume integral used here. Treat the output as a diagnostic, not
       as an impedance. Above the cutoff the reliable route today is to filter the
       mode set with :func:`pml_stable_modes` and sum it with
       :func:`reconstruct_impedance`.

    This builds the same spectrum from the quasi-normal-mode residues instead. With
    ``w~ = w(1 - i/2Q)`` the decaying complex eigenfrequency and ``a~ = w~ (R/Q)~ / 2``
    the wake amplitude, the wake ``W(t) = Re[a~ exp(-i w~ t)]`` transforms to::

        Z(w) = (i/2) sum_n [ a~_n / (w - w~_n)  +  conj(a~_n) / (w + conj(w~_n)) ]

    The conjugate-pole partner is what enforces ``Z(-w) = conj(Z(w))``, i.e. a real
    wake. For a real ``(R/Q)~`` and large Q this reduces term by term to the RLC form,
    peaking at ``1/2 Q (R/Q)``.

    Parameters
    ----------
    freqs : array-like
        Real part of each mode frequency [Hz].
    r_over_q_complex : array-like of complex
        ``(R/Q)~ = V~^2 / (w~ U~)`` per mode [Ohm] — the solver's
        ``'Re(R/Q) [Ohm]'`` + 1j*``'Im(R/Q) [Ohm]'``.
    q_factors : array-like
        Quality factor of each mode.
    f_span : array-like
        Frequencies to evaluate at [Hz]. ``f = 0`` is evaluated by its limit.
    transverse : bool
        Transverse impedance [Ohm/m], using the same ``w_0/c`` and ``w_0/w``
        convention as :func:`reconstruct_impedance`.

    Returns
    -------
    ndarray of complex
    """
    f0 = np.asarray(freqs, dtype=float)
    roq = np.asarray(r_over_q_complex, dtype=complex)
    q = np.asarray(q_factors, dtype=float)
    f = np.asarray(f_span, dtype=float)

    if not (f0.shape == roq.shape == q.shape):
        raise ValueError(f"freqs, r_over_q_complex and q_factors must have the same "
                         f"length; got {f0.shape}, {roq.shape}, {q.shape}.")
    if np.any(f0 <= 0):
        raise ValueError("mode frequencies must be positive.")

    w0 = 2 * np.pi * f0
    w = 2 * np.pi * f
    z = np.zeros(f.shape, dtype=complex)

    for w0i, roqi, qi in zip(w0, roq, q):
        w_c = w0i * (1 - 0.5j / max(abs(qi), 1e-30))     # decaying pole
        a_c = w_c * roqi / 2                             # wake amplitude
        term = 0.5j * (a_c / (w - w_c) + np.conj(a_c) / (w + np.conj(w_c)))
        if transverse:
            # Same convention as the RLC path: R/Q_t [Ohm] -> Ohm/m via w_0/c, and
            # the extra w_0/w that makes the transverse resonance peak at w_0.
            with np.errstate(divide='ignore', invalid='ignore'):
                term = term * (w0i / C0) * np.where(w > 0, w0i / np.where(w > 0, w, 1), 0.0)
        z += term

    return z


def pml_stable_modes(modes, other, rtol_f=1e-3, rtol_q=0.15):
    """Boolean mask over *modes*: which ones survive a change of PML settings.

    An open (PML) solve returns two kinds of eigenpair above the beam-pipe cutoff,
    and they look alike in the results table:

    - genuine resonance poles of the open cavity, fixed by the geometry. Lengthening
      the absorbing layer or changing its stretch moves them by ~1e-6;
    - modes of the finite PML *block* itself. These shift by whole percent, appear and
      disappear between runs, and — because the layer damps them — carry a plausible
      low Q. They can also carry a LARGE R/Q, so they dominate a reconstructed
      impedance while being an artifact of the truncation.

    ``'Q balance []'`` does not separate them: a PML-block mode decays into the layer
    and radiates through the mouth at the same rate, so its balance sits at 1.00 like
    everything else. The only reliable discriminator is re-solving with a different
    layer and keeping what does not move, which is what this does.

    Parameters
    ----------
    modes, other : pandas.DataFrame
        Two ``cav.eigenmode.qois_df`` frames for the same cavity, solved with
        different ``pml_length`` (or ``pml_alpha``, or mesh).
    rtol_f, rtol_q : float
        A mode in *modes* is kept when *other* holds a mode within ``rtol_f``
        in relative frequency and ``rtol_q`` in relative Q.

    Returns
    -------
    ndarray of bool
        Aligned with ``modes.index``.

    Notes
    -----
    **Compare only where both runs reach.** Each solve returns its lowest ``n_modes``,
    and two runs need not stop at the same frequency. A mode above *other*'s highest has
    no partner to be matched against and comes back False -- unverified, not disproved.
    Restrict the judgement to
    ``min(modes['freq [MHz]'].max(), other['freq [MHz]'].max())`` before quoting a
    fraction, or raise ``n_modes`` until both runs cover the band of interest.

    Examples
    --------
    >>> keep = pml_stable_modes(cav_a.eigenmode.qois_df, cav_b.eigenmode.qois_df)
    >>> trusted = cav_a.eigenmode.qois_df[keep]
    """
    f_a = np.asarray(modes['freq [MHz]'], dtype=float)
    q_a = np.asarray(modes['Q []'], dtype=float)
    f_b = np.asarray(other['freq [MHz]'], dtype=float)
    q_b = np.asarray(other['Q []'], dtype=float)
    if f_b.size == 0:
        return np.zeros(f_a.shape, dtype=bool)

    keep = np.zeros(f_a.shape, dtype=bool)
    for i, (f, q) in enumerate(zip(f_a, q_a)):
        j = int(np.argmin(np.abs(f_b - f)))
        near_f = abs(f_b[j] - f) <= rtol_f * abs(f)
        near_q = abs(q_b[j] - q) <= rtol_q * max(abs(q), 1e-30)
        keep[i] = bool(near_f and near_q)
    return keep
