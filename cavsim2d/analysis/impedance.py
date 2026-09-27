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

from cavsim2d.constants import c0

C0 = c0                 # speed of light [m/s]

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


def classify_modes(modes, other, rtol_f=2e-4, linewidth_frac=0.02, rtol_q=0.25,
                   pair_frac=0.1, runner_up=3.0):
    """Cavity mode or artefact, for each mode of an open solve, from a second solve.

    An open (PML) solve returns genuine resonances of the cavity mixed with modes of
    the beam pipe and of the absorbing block, and they look alike in the results.
    Re-solving with a different ``beampipe_length`` (or ``pml_length``) separates
    them: a cavity mode does not care how long the pipe is, a pipe mode moves. This
    is the best discriminant found so far, and it is not perfect. On pillbox and
    TESLA test cases it miscounts a few per cent of the modes, in both directions.

    Each mode of *modes* is judged against *other*, per azimuthal order ``m``:

    1. Frequency, the primary test. Its partner must be its mutual nearest neighbour
       (each is the other's closest mode), within ``max(rtol_f, linewidth_frac / Q)``
       relative. The tolerance scales with the linewidth ``f/Q``: a broad radiating
       mode moves by a larger fraction than a narrow one for the same physical
       change. A partner is ambiguous, and rejected, when a runner-up sits within
       the tolerance and less than ``runner_up`` times further away.
    2. Q, a tie-breaker only. A mode whose Q moved by more than ``rtol_q`` is still
       kept, since the Q of a mode near the cutoff converges slowly with pipe length
       (the layer keeps reaching its evanescent tail). It is dropped only when it
       also has a near-degenerate partner of the same ``m``, closer than
       ``pair_frac`` of the typical spacing. At fixed ``m`` an axisymmetric cavity
       has no degeneracy, so such a pair is the left and right pipe, not one mode.

    Parameters
    ----------
    modes, other : pandas.DataFrame
        Two ``cav.eigenmode.qois_df`` frames of the same cavity, solved with a
        different pipe (or PML) length. Needs ``'freq [MHz]'`` and ``'Q []'``;
        ``'m'`` is used when present.

    Returns
    -------
    pandas.DataFrame
        Indexed like *modes*: ``'cavity mode'`` (the verdict), and the evidence behind
        it, ``'shift [ppm]'``, ``'tolerance [ppm]'``, ``'mutual'``, ``'f ok'``,
        ``'Q ok'``, ``'paired'`` and ``'reached'``.

    Notes
    -----
    Compare only where both runs reach. Each solve returns its lowest ``n_modes``,
    and a mode above *other*'s highest has nothing to be matched against: it comes
    back ``'reached' == False`` and is not called a cavity mode (unverified, not
    disproved). Raise ``n_modes`` until both runs cover the band of interest.
    """
    out = pd.DataFrame(index=modes.index)
    for col, default in (('shift [ppm]', np.nan), ('tolerance [ppm]', np.nan),
                         ('mutual', False), ('f ok', False), ('Q ok', False),
                         ('paired', False), ('reached', False)):
        out[col] = default

    def groups(df):
        if 'm' in df.columns:
            return {m: g for m, g in df.groupby('m')}
        return {0: df}

    other_groups = groups(other)
    for m, a in groups(modes).items():
        b = other_groups.get(m)
        f_a = a['freq [MHz]'].to_numpy(dtype=float)
        q_a = a['Q []'].to_numpy(dtype=float)
        if b is None or len(b) == 0:
            continue
        f_b = b['freq [MHz]'].to_numpy(dtype=float)
        q_b = b['Q []'].to_numpy(dtype=float)

        rel = np.abs(f_a[:, None] - f_b[None, :]) / f_a[:, None]
        j = rel.argmin(axis=1)
        d1 = rel[np.arange(len(f_a)), j]
        d2 = (np.partition(rel, 1, axis=1)[:, 1] if rel.shape[1] > 1
              else np.full(len(f_a), np.inf))
        back = (np.abs(f_b[:, None] - f_a[None, :]) / f_b[:, None]).argmin(axis=1)
        mutual = back[j] == np.arange(len(f_a))
        tol = np.maximum(rtol_f, linewidth_frac / np.maximum(np.abs(q_a), 1e-30))
        ambiguous = (d2 <= tol) & (d2 < runner_up * d1)
        f_ok = mutual & (d1 <= tol) & ~ambiguous
        q_ok = np.abs(q_b[j] - q_a) <= rtol_q * np.maximum(np.abs(q_a), 1e-30)

        # near-degenerate partner of the same m, relative to the typical spacing
        if len(f_a) > 1:
            gaps = np.abs(f_a[:, None] - f_a[None, :]) / f_a[:, None]
            np.fill_diagonal(gaps, np.inf)
            nearest = gaps.min(axis=1)
            paired = nearest < pair_frac * np.median(nearest)
        else:
            paired = np.zeros(len(f_a), dtype=bool)

        idx = a.index
        out.loc[idx, 'shift [ppm]'] = d1 * 1e6
        out.loc[idx, 'tolerance [ppm]'] = tol * 1e6
        out.loc[idx, 'mutual'] = mutual
        out.loc[idx, 'f ok'] = f_ok
        out.loc[idx, 'Q ok'] = q_ok
        out.loc[idx, 'paired'] = paired
        out.loc[idx, 'reached'] = f_a <= f_b.max() * (1 + rtol_f)

    out['cavity mode'] = (out['f ok'] & ~(~out['Q ok'] & out['paired'])
                          & out['reached']).astype(bool)
    return out


def pml_stable_modes(modes, other, **kwargs):
    """Boolean mask over *modes*: the cavity modes, judged against a second solve
    with a different pipe or PML length. See :func:`classify_modes`, which returns
    the same verdict with the evidence behind it.

    >>> keep = pml_stable_modes(cav_a.eigenmode.qois_df, cav_b.eigenmode.qois_df)
    >>> trusted = cav_a.eigenmode.qois_df[keep]
    """
    return classify_modes(modes, other, **kwargs)['cavity mode'].to_numpy(dtype=bool)
