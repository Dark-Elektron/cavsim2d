"""Impedance reconstructed from eigenmode results (equivalent-circuit model)."""
import numpy as np
import pytest

from cavsim2d.analysis.impedance import (reconstruct_impedance,
                                         reconstruct_impedance_qnm, C0)


def test_peak_is_the_shunt_impedance():
    """On resonance the resonator term is purely real and equals R = 1/2 Q (R/Q)."""
    f0, roq, q = 1.4e9, 120.0, 1e4
    z = reconstruct_impedance([f0], [roq], [q], np.array([f0]))
    R = 0.5 * q * roq
    assert z[0].real == pytest.approx(R)
    assert z[0].imag == pytest.approx(0.0, abs=1e-6 * R)


def test_resonance_has_the_right_width():
    """|Z| falls to R/sqrt(2) at the half-power points f0 (1 +/- 1/2Q)."""
    f0, roq, q = 1e9, 100.0, 1e3
    R = 0.5 * q * roq
    f_half = f0 * (1 + 1 / (2 * q))
    z = reconstruct_impedance([f0], [roq], [q], np.array([f_half]))
    assert abs(z[0]) == pytest.approx(R / np.sqrt(2), rel=1e-3)


def test_modes_superpose():
    """Two well-separated modes each keep their own peak."""
    fs, roqs, qs = [1.0e9, 2.0e9], [120.0, 60.0], [1e4, 1e4]
    z = reconstruct_impedance(fs, roqs, qs, np.array(fs))
    for i, (f, roq, q) in enumerate(zip(fs, roqs, qs)):
        assert abs(z[i]) == pytest.approx(0.5 * q * roq, rel=1e-3)


def test_transverse_carries_the_omega_over_c_factor():
    """The transverse R/Q [Ohm] becomes a shunt impedance [Ohm/m] via omega0/c."""
    f0, roq, q = 1.4e9, 15.0, 1e4
    z = reconstruct_impedance([f0], [roq], [q], np.array([f0]), transverse=True)
    expected = 0.5 * q * roq * (2 * np.pi * f0 / C0)      # on resonance f0/f == 1
    assert abs(z[0]) == pytest.approx(expected, rel=1e-9)


def test_dc_is_finite():
    """f = 0 is singular in the resonator term but both limits are finite:
    the longitudinal impedance vanishes, the transverse one does not."""
    fs, roqs, qs = [1.4e9, 2.0e9], [120.0, 1.0], [1e4, 5e1]
    f = np.array([0.0, 1e6])

    zz = reconstruct_impedance(fs, roqs, qs, f)
    assert zz[0] == 0
    assert np.isfinite(zz).all()

    zt = reconstruct_impedance(fs, roqs, qs, f, transverse=True)
    r_shunt = 0.5 * np.array(qs) * np.array(roqs) * (2 * np.pi * np.array(fs) / C0)
    assert zt[0] == pytest.approx(np.sum(1j * r_shunt / np.array(qs)))
    assert np.isfinite(zt).all()


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError):
        reconstruct_impedance([1e9, 2e9], [100.0], [1e4, 1e4], np.array([1e9]))
    with pytest.raises(ValueError, match='positive'):
        reconstruct_impedance([0.0], [100.0], [1e4], np.array([1e9]))


# --- through the public API -------------------------------------------------

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL                        # noqa: E402
from cavsim2d import Study, EllipticalCavity     # noqa: E402


def _solved(project_dir):
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs = Study(project_dir)
    cavs.add_cavity([cav], ['Z'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 5,
                        'mesh_config': {'h': 25, 'p': 3}})
    return cav


def _own_Q(cav):
    """The solve's own Q values, in the frequency order impedance() sorts into.

    These tests exercise the reconstruction ARITHMETIC — schema, peak heights, unit
    prefixes, spans — on a cheap closed solve. Reconstruction from a closed solve is
    refused (the Q above cutoff is the wall's, not the mode's), and the documented way
    to say "I know what Q I want" is to pass it. Handing back the solve's own Q makes
    these tests exactly what they were before the refusal existed, without turning them
    into slow PML runs. The refusal itself is tested in
    test_closed_boundary_reconstruction_is_refused.
    """
    return cav.eigenmode.qois_df.sort_values('freq [MHz]')['Q []'].to_numpy()


def test_impedance_matches_the_wakefield_schema(project_dir):
    """The frame uses the same columns as cav.wakefield.wake_z, so the two
    overlay without renaming; the transverse one is per metre."""
    cav = _solved(project_dir)
    Q = _own_Q(cav)

    # both solvers default to kOhm, so the two spectra land on the same scale
    z = cav.eigenmode.impedance(Q=Q)
    assert list(z.columns) == ['f [MHz]', '|Z| [kOhm]', 'Re(Z) [kOhm]', 'Im(Z) [kOhm]']

    zt = cav.eigenmode.impedance('transverse', Q=Q)
    assert list(zt.columns) == ['f [MHz]', '|Z| [kOhm/m]',
                                'Re(Z) [kOhm/m]', 'Im(Z) [kOhm/m]']
    assert not z.isna().any().any() and not zt.isna().any().any()


def test_impedance_peaks_hit_the_shunt_impedance(project_dir):
    """A uniform grid steps over a high-Q resonance and understates it; the mode
    linewidths are sampled explicitly, so every peak reaches R = 1/2 Q (R/Q)."""
    cav = _solved(project_dir)
    z = cav.eigenmode.impedance(unit='', Q=_own_Q(cav))   # R = 1/2 Q (R/Q) is in Ohm
    modes = cav.eigenmode.qois_df.query('m == 0')

    for _, mode in modes.iterrows():
        R = 0.5 * mode['Q []'] * mode['R/Q [Ohm]']
        if R < 1e3:                       # skip the TE modes (R/Q = 0)
            continue
        i = (z['f [MHz]'] - mode['freq [MHz]']).abs().idxmin()
        assert z.loc[i, '|Z| [Ohm]'] == pytest.approx(R, rel=0.02)


def test_impedance_span_and_Q_override(project_dir):
    cav = _solved(project_dir)
    modes = cav.eigenmode.qois_df.query('m == 0')
    Q = _own_Q(cav)

    # default span is 0 .. the highest computed mode
    z = cav.eigenmode.impedance(Q=Q)
    assert z['f [MHz]'].min() == pytest.approx(0.0)
    assert z['f [MHz]'].max() == pytest.approx(modes['freq [MHz]'].max())

    # an explicit span is honoured
    zs = cav.eigenmode.impedance(span=(500, 900), Q=Q)
    assert zs['f [MHz]'].min() == pytest.approx(500)
    assert zs['f [MHz]'].max() == pytest.approx(900)

    # peak height scales with Q: the shunt impedance is 1/2 Q (R/Q)
    f0 = cav.eigenmode.qois['freq [MHz]']
    roq = cav.eigenmode.qois['R/Q [Ohm]']
    zq = cav.eigenmode.impedance(Q=1e4, unit='')
    i = (zq['f [MHz]'] - f0).abs().idxmin()
    assert zq.loc[i, '|Z| [Ohm]'] == pytest.approx(0.5 * 1e4 * roq, rel=0.02)


def test_impedance_unit_prefix(project_dir):
    """Both solvers report kOhm by default and take the same `unit` argument, so
    an eigenmode spectrum and a wakefield spectrum land on the same scale.

    The wakefield frames used to be *labelled* [Ohm] while holding ABCI's kOhm
    numbers, which put a reconstructed spectrum a factor of 1000 above a
    simulated one plotted beside it.
    """
    cav = _solved(project_dir)
    Q = _own_Q(cav)

    ohm = cav.eigenmode.impedance(unit='', Q=Q)
    kohm = cav.eigenmode.impedance(unit='k', Q=Q)
    mohm = cav.eigenmode.impedance(unit='M', Q=Q)

    assert '|Z| [Ohm]' in ohm and '|Z| [kOhm]' in kohm and '|Z| [MOhm]' in mohm
    # the values really are rescaled, not just relabelled
    assert ohm['|Z| [Ohm]'].max() == pytest.approx(1e3 * kohm['|Z| [kOhm]'].max())
    assert kohm['|Z| [kOhm]'].max() == pytest.approx(1e3 * mohm['|Z| [MOhm]'].max())
    # ... and so are the real/imaginary parts
    assert ohm['Re(Z) [Ohm]'].abs().max() == pytest.approx(
        1e3 * kohm['Re(Z) [kOhm]'].abs().max())

    # transverse carries the /m
    assert '|Z| [MOhm/m]' in cav.eigenmode.impedance('transverse', unit='M', Q=Q)

    with pytest.raises(ValueError, match='unit prefix'):
        cav.eigenmode.impedance(unit='x', Q=Q)


def test_study_eigenmode_plot_impedance_overlays_every_cavity(project_dir):
    """study.eigenmode.plot_impedance() draws one warm-coloured curve per cavity
    on a shared frequency axis, so the cavities are actually comparable."""
    import matplotlib
    matplotlib.use('Agg')

    cavs = Study(project_dir)
    a = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    b = EllipticalCavity(2, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([a, b], ['A', 'B'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 5,
                        'mesh_config': {'h': 25, 'p': 3}})

    ax = cavs.eigenmode.plot_impedance(Q=1e4)
    lines = ax.get_lines()
    assert [ln.get_label() for ln in lines] == ['A', 'B']
    assert len({ln.get_color() for ln in lines}) == 2      # distinct colours
    assert ax.get_yscale() == 'log'

    # the span is shared across cavities, not per-cavity
    spans = [(ln.get_xdata().min(), ln.get_xdata().max()) for ln in lines]
    assert spans[0] == pytest.approx(spans[1])

    axt = cavs.eigenmode.plot_impedance('transverse', Q=1e4)
    assert r'\Omega' in axt.get_ylabel()
    assert len(axt.get_lines()) == 2

    # the study-level frame is the per-cavity one plus a 'cavity' column
    df = cavs.eigenmode.qois_df
    assert 'cavity' in df.columns
    assert set(df['cavity']) == {'A', 'B'}
    assert len(df) == len(a.eigenmode.qois_df) + len(b.eigenmode.qois_df)


def test_solver_namespaces_are_symmetric():
    """A cavity and a study reach the same plot by the same name, so there is no
    guessing whether impedance lives on the model, the solver or the study.

    The wakefield impedance used to be reachable only as cav.plot('ZL'), which
    said nothing about what it plotted and collided conceptually with the
    eigenmode-reconstructed impedance.
    """
    from cavsim2d.solvers.solver_objects import (EigenmodeSolver, WakefieldSolver,
                                                 StudyEigenmode, StudyWakefield)

    # the same name means the same thing at both levels
    for cavity_cls, study_cls, names in (
            (EigenmodeSolver, StudyEigenmode, ['plot_impedance', 'qois_df']),
            (WakefieldSolver, StudyWakefield,
             ['plot_impedance', 'plot_wake', 'plot_k_loss', 'plot_k_kick']),
    ):
        for name in names:
            assert hasattr(cavity_cls, name), f'{cavity_cls.__name__}.{name}'
            assert hasattr(study_cls, name), f'{study_cls.__name__}.{name}'

    # both namespaces can kick off their own run
    assert hasattr(StudyEigenmode, 'run') and hasattr(StudyWakefield, 'run')
    # the old, uninformative names (plot_z/plot_t; cav.plot('ZL')) were removed
    assert not hasattr(WakefieldSolver, 'plot_z')
    assert not hasattr(WakefieldSolver, 'plot_t')


def test_impedance_without_that_polarisation_is_reported(project_dir):
    """Asking for a transverse impedance with no dipole solve returns an empty
    frame and says why, rather than raising a KeyError from deep in the maths."""
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs = Study(project_dir)
    cavs.add_cavity([cav], ['NoDip'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': 'monopole', 'n_modes': 3})
    assert cav.eigenmode.impedance('transverse', Q=1e4).empty
    assert not cav.eigenmode.impedance(Q=1e4).empty


def test_closed_boundary_reconstruction_is_refused(project_dir):
    """A closed (PMC) solve cannot produce an impedance: above the beam-pipe cutoff the
    ends reflect, so each mode keeps the wall's ohmic Q0 instead of its radiation Q and
    |Z| lands orders of magnitude high. That is refused outright rather than warned
    about -- but an explicit Q= is the caller taking responsibility, and still works."""
    cav = _solved(project_dir)

    assert cav.eigenmode.impedance().empty
    assert cav.eigenmode.impedance('transverse').empty
    assert not cav.eigenmode.impedance(Q=1e4).empty          # explicit override

    import matplotlib
    matplotlib.use('Agg')
    assert cav.eigenmode.plot_impedance(show=False) is None  # nothing to draw


def test_open_boundary_reconstruction_is_allowed(project_dir):
    """With an open (PML) end the Q already contains the radiation, so the same call
    goes through untouched and uses each mode's own Q."""
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs = Study(project_dir)
    cavs.add_cavity([cav], ['Open'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'oo',
                        'polarisation': 'monopole', 'n_modes': 4,
                        'mesh_config': {'h': 25, 'p': 3}})

    z = cav.eigenmode.impedance()
    assert not z.empty
    assert list(z.columns) == ['f [MHz]', '|Z| [kOhm]', 'Re(Z) [kOhm]', 'Im(Z) [kOhm]']
    assert not z.isna().any().any()


def test_reconstruct_impedance_dc_limit():
    """Edge case f = 0 (allowed): the longitudinal impedance vanishes at DC while
    the transverse one tends to a finite i*R/Q — the resonator term's two limits.
    Guards the special-casing that a naive 1/(f/f0 - f0/f) would divide by zero."""
    f0, roq, q = np.array([1.0e9]), np.array([100.0]), np.array([1e4])
    grid = np.array([0.0, f0[0]])

    zl = reconstruct_impedance(f0, roq, q, grid, transverse=False)
    assert zl[0] == 0.0                                   # longitudinal -> 0 at DC

    zt = reconstruct_impedance(f0, roq, q, grid, transverse=True)
    assert zt[0].real == pytest.approx(0.0, abs=1e-9)
    # DC transverse limit i*R/Q: r_shunt/q cancels the q (r_shunt = 0.5 q roq w0/c).
    assert zt[0].imag == pytest.approx(0.5 * roq[0] * (2 * np.pi * f0[0] / C0))


def test_reconstruct_impedance_rejects_bad_input():
    """Edge cases: mismatched array lengths and a non-positive mode frequency must
    raise clearly rather than produce silent garbage."""
    with pytest.raises(ValueError, match="same length"):
        reconstruct_impedance([1e9, 2e9], [100.0], [1e4], np.array([1e9]))
    with pytest.raises(ValueError, match="positive"):
        reconstruct_impedance([0.0], [100.0], [1e4], np.array([1e9]))


def test_reconstruct_impedance_sums_modes():
    """Two well-separated modes: each peak reaches its own shunt impedance,
    independent of the other (a linear sum of resonators)."""
    f0 = np.array([1.0e9, 2.0e9])
    roq = np.array([100.0, 50.0])
    q = np.array([1e4, 1e4])
    z = reconstruct_impedance(f0, roq, q, f0)
    assert z[0].real == pytest.approx(0.5 * q[0] * roq[0], rel=1e-3)
    assert z[1].real == pytest.approx(0.5 * q[1] * roq[1], rel=1e-3)


# --- QNM (complex-residue) reconstruction ---------------------------------

def test_qnm_reduces_to_rlc_for_a_trapped_mode():
    """A real (R/Q)~ at high Q is a trapped mode, and there the pole sum must
    reproduce the RLC spectrum everywhere — that is what makes 'qnm' safe to use
    on a mode set that spans the cutoff."""
    f0 = 1.3e9
    f = np.unique(np.concatenate([[f0], np.linspace(0.05e9, 3e9, 20001)]))
    a = reconstruct_impedance([f0], [300.0], [1e4], f)
    b = reconstruct_impedance_qnm([f0], [300.0 + 0j], [1e4], f)
    big = np.abs(a) > 1e-3 * np.abs(a).max()
    assert np.allclose(np.abs(b[big]), np.abs(a[big]), rtol=1e-3)


def test_qnm_peak_is_the_shunt_impedance_at_any_q():
    """Off resonance the two forms differ by O(1/Q^2) — the RLC form is the
    approximation — but the peak stays 1/2 Q (R/Q) however broad the mode."""
    f0 = 1.3e9
    for q in (1e4, 100.0, 10.0, 3.0):
        # a broad pole peaks a little BELOW f0, so scan rather than sample at f0
        f = np.linspace(f0 * (1 - 3 / q), f0 * (1 + 3 / q), 20001)
        z = reconstruct_impedance_qnm([f0], [300.0 + 0j], [q], f)
        assert np.abs(z).max() == pytest.approx(0.5 * q * 300.0, rel=2e-3)


def test_qnm_wake_is_real():
    """Z(-w) = conj(Z(w)) — the conjugate-pole partner is what guarantees a real
    wake, and it is the term an ad-hoc complex residue would leave out."""
    f0 = np.array([1.3e9, 3.4e9])
    roq = np.array([300 + 120j, -80 + 40j])
    q = np.array([50.0, 4.0])
    fp = np.array([0.4e9, 2.1e9, 5.0e9])
    zp = reconstruct_impedance_qnm(f0, roq, q, fp)
    zm = reconstruct_impedance_qnm(f0, roq, q, -fp)
    assert np.allclose(zp, np.conj(zm))


def test_qnm_lets_overlapping_modes_interfere():
    """The point of the model. Two overlapping low-Q poles with opposite-sign
    residues must partly cancel; the RLC sum, which forces both residues
    positive-real, can only add them."""
    f0 = np.array([3.2e9, 3.35e9])
    q = np.array([5.0, 5.0])
    f = np.array([3.275e9])                       # midway between the two
    rlc = abs(reconstruct_impedance(f0, np.array([100.0, 100.0]), q, f)[0])
    same = abs(reconstruct_impedance_qnm(f0, np.array([100 + 0j, 100 + 0j]), q, f)[0])
    opp = abs(reconstruct_impedance_qnm(f0, np.array([100 + 0j, -100 + 0j]), q, f)[0])
    assert same == pytest.approx(rlc, rel=0.05)   # in phase -> same as RLC
    assert opp < 0.4 * rlc                        # out of phase -> cancels


def test_qnm_model_is_rejected_without_complex_residues(project_dir):
    """Asking for model='qnm' on a run that has no complex eigenvalue reports
    rather than silently falling back to a different physical model."""
    cav = _solved(project_dir)
    assert cav.eigenmode.impedance('longitudinal', model='qnm').empty


def test_impedance_rejects_unknown_model(project_dir):
    cav = _solved(project_dir)
    with pytest.raises(ValueError, match="model must be"):
        cav.eigenmode.impedance('longitudinal', model='banana')
