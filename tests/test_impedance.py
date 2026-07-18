"""Impedance reconstructed from eigenmode results (equivalent-circuit model)."""
import numpy as np
import pytest

from cavsim2d.analysis.impedance import reconstruct_impedance, C0


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
from cavsim2d import Cavities, EllipticalCavity     # noqa: E402


def _solved(project_dir):
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs = Cavities(project_dir)
    cavs.add_cavity([cav], ['Z'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 5,
                        'mesh_config': {'h': 25, 'p': 3}})
    return cav


def test_impedance_matches_the_wakefield_schema(project_dir):
    """The frame uses the same columns as cav.wakefield.wake_z, so the two
    overlay without renaming; the transverse one is per metre."""
    cav = _solved(project_dir)

    # both solvers default to kOhm, so the two spectra land on the same scale
    z = cav.eigenmode.impedance()
    assert list(z.columns) == ['f [MHz]', '|Z| [kOhm]', 'Re(Z) [kOhm]', 'Im(Z) [kOhm]']

    zt = cav.eigenmode.impedance('transverse')
    assert list(zt.columns) == ['f [MHz]', '|Z| [kOhm/m]',
                                'Re(Z) [kOhm/m]', 'Im(Z) [kOhm/m]']
    assert not z.isna().any().any() and not zt.isna().any().any()


def test_impedance_peaks_hit_the_shunt_impedance(project_dir):
    """A uniform grid steps over a high-Q resonance and understates it; the mode
    linewidths are sampled explicitly, so every peak reaches R = 1/2 Q (R/Q)."""
    cav = _solved(project_dir)
    z = cav.eigenmode.impedance(unit='')          # R = 1/2 Q (R/Q) is in Ohm
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

    # default span is 0 .. the highest computed mode
    z = cav.eigenmode.impedance()
    assert z['f [MHz]'].min() == pytest.approx(0.0)
    assert z['f [MHz]'].max() == pytest.approx(modes['freq [MHz]'].max())

    # an explicit span is honoured
    zs = cav.eigenmode.impedance(span=(500, 900))
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

    ohm = cav.eigenmode.impedance(unit='')
    kohm = cav.eigenmode.impedance(unit='k')
    mohm = cav.eigenmode.impedance(unit='M')

    assert '|Z| [Ohm]' in ohm and '|Z| [kOhm]' in kohm and '|Z| [MOhm]' in mohm
    # the values really are rescaled, not just relabelled
    assert ohm['|Z| [Ohm]'].max() == pytest.approx(1e3 * kohm['|Z| [kOhm]'].max())
    assert kohm['|Z| [kOhm]'].max() == pytest.approx(1e3 * mohm['|Z| [MOhm]'].max())
    # ... and so are the real/imaginary parts
    assert ohm['Re(Z) [Ohm]'].abs().max() == pytest.approx(
        1e3 * kohm['Re(Z) [kOhm]'].abs().max())

    # transverse carries the /m
    assert '|Z| [MOhm/m]' in cav.eigenmode.impedance('transverse', unit='M')

    with pytest.raises(ValueError, match='unit prefix'):
        cav.eigenmode.impedance(unit='x')


def test_study_eigenmode_plot_impedance_overlays_every_cavity(project_dir):
    """study.eigenmode.plot_impedance() draws one warm-coloured curve per cavity
    on a shared frequency axis, so the cavities are actually comparable."""
    import matplotlib
    matplotlib.use('Agg')

    cavs = Cavities(project_dir)
    a = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    b = EllipticalCavity(2, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([a, b], ['A', 'B'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 5,
                        'mesh_config': {'h': 25, 'p': 3}})

    ax = cavs.eigenmode.plot_impedance()
    lines = ax.get_lines()
    assert [ln.get_label() for ln in lines] == ['A', 'B']
    assert len({ln.get_color() for ln in lines}) == 2      # distinct colours
    assert ax.get_yscale() == 'log'

    # the span is shared across cavities, not per-cavity
    spans = [(ln.get_xdata().min(), ln.get_xdata().max()) for ln in lines]
    assert spans[0] == pytest.approx(spans[1])

    axt = cavs.eigenmode.plot_impedance('transverse')
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
    cavs = Cavities(project_dir)
    cavs.add_cavity([cav], ['NoDip'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': 'monopole', 'n_modes': 3})
    assert cav.eigenmode.impedance('transverse').empty
    assert not cav.eigenmode.impedance().empty
