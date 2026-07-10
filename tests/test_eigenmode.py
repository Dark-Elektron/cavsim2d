"""Eigenmode regression tests: monopole baseline, m-pole solver vs analytic,
per-polarisation result folders and rerun semantics, legacy-layout fallback."""
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d.cavity import Cavities, EllipticalCavity


def _run(project_dir, name='CAV', n_cells=1, config=None):
    cavs = Cavities(project_dir)
    cav = EllipticalCavity(n_cells, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], [name])
    cfg = {'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'}
    if config:
        cfg.update(config)
    cavs.run_eigenmode(cfg)
    return cav


def test_mode_count_resolution_defaults_and_aliases():
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP

    class DummyCavity:
        n_cells = 3

    solver = NGSolveMEVP()
    assert solver.requested_n_modes() == 10
    assert solver.requested_n_modes(DummyCavity()) == 5
    assert solver.requested_n_modes(DummyCavity(), {'n_modes': 7}) == 7
    assert solver.requested_n_modes(DummyCavity(), {'nmodes': 8}) == 8
    assert solver.pinvit_n_modes(8) == 10
    with pytest.raises(ValueError):
        solver.requested_n_modes(n_modes=0)


def test_monopole_writes_monopole_folder(project_dir):
    cav = _run(project_dir)
    eig = os.path.join(cav.self_dir, 'eigenmode')
    assert os.path.exists(os.path.join(eig, 'monopole', 'qois.json'))
    assert os.path.exists(os.path.join(eig, 'monopole', 'mesh.pkl'))
    # flat layout must NOT be written
    assert not os.path.exists(os.path.join(eig, 'qois.json'))


def test_monopole_qois_physical(project_dir):
    cav = _run(project_dir)
    cav.get_eigenmode_qois()
    q = cav.eigenmode_qois
    # ~800 MHz TESLA-like fundamental
    assert 750 < q['freq [MHz]'] < 850
    # copper geometric factor is mesh-independent and ~270 Ohm (P0-1 fix)
    assert 200 < q['G [Ohm]'] < 320
    assert q['R/Q [Ohm]'] > 0
    assert q['Q []'] > 0


def test_ploss_mesh_independent(project_dir):
    """Regression for P0-1: G must not drift with mesh density."""
    gvals = []
    for hdiv in (15, 30):
        pdir = os.path.join(project_dir, f'h{hdiv}')
        cav = _run(pdir, name=f'H{hdiv}', config={'mesh_config': {'h': hdiv}})
        cav.get_eigenmode_qois()
        gvals.append(cav.eigenmode_qois['G [Ohm]'])
    # < 5% spread across a 2x mesh change (was >10x before the fix)
    assert abs(gvals[0] - gvals[1]) / gvals[0] < 0.05, gvals


def test_mpole_pillbox_matches_analytic():
    """Dipole (m=1) eigenfrequencies of a PEC pillbox vs the analytic TM/TE
    spectrum, exercising _solve_eigenproblem_mpole directly."""
    from ngsolve import Mesh
    from netgen.geom2d import SplineGeometry
    from scipy.special import jn_zeros, jnp_zeros
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP

    c0 = 299792458.0
    R, Lz, m = 0.10, 0.15, 1

    geo = SplineGeometry()
    p = [geo.AppendPoint(*pt) for pt in [(0, 0), (Lz, 0), (Lz, R), (0, R)]]
    geo.Append(['line', p[0], p[1]], bc='AXI')
    geo.Append(['line', p[1], p[2]], bc='PEC')
    geo.Append(['line', p[2], p[3]], bc='PEC')
    geo.Append(['line', p[3], p[0]], bc='PEC')
    mesh = Mesh(geo.GenerateMesh(maxh=0.008))

    freqs, _, _ = NGSolveMEVP()._solve_eigenproblem_mpole(mesh, 2, m, 6)
    freqs = np.sort([f for f in freqs if f > 1.0])   # guard against any residual ~0 mode
    assert len(freqs) >= 3, freqs

    # analytic dipole modes
    ana = []
    for xmn in jn_zeros(m, 3):       # TM
        for pp in range(3):
            ana.append(c0 / (2*np.pi) * np.sqrt((xmn/R)**2 + (pp*np.pi/Lz)**2) * 1e-6)
    for xpmn in jnp_zeros(m, 3):     # TE (p>=1)
        for pp in range(1, 3):
            ana.append(c0 / (2*np.pi) * np.sqrt((xpmn/R)**2 + (pp*np.pi/Lz)**2) * 1e-6)
    ana = np.sort(np.array(ana))

    # the lowest analytic dipole modes must each appear among the computed
    # physical modes to < 1%
    for a in ana[:3]:
        assert np.min(np.abs(freqs - a) / a) < 0.01, (a, freqs)


def test_surface_resistance_helper():
    """Rs from conductivity vs a fixed override; the SRF path bypasses w."""
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import surface_resistance, SIGMA_COPPER
    w = 2 * np.pi * 1.3e9
    rs_cu = surface_resistance(w, SIGMA_COPPER)
    assert 1e-3 < rs_cu < 1e-1                     # ~9 mOhm for copper at 1.3 GHz
    assert surface_resistance(w, SIGMA_COPPER, rs=1e-8) == 1e-8   # fixed Rs wins
    # lower conductivity -> higher Rs
    assert surface_resistance(w, SIGMA_COPPER / 4) > rs_cu


def test_material_affects_q_not_g(project_dir):
    """A fixed SRF surface resistance raises Q by orders of magnitude but
    leaves the (material-independent) geometric factor G unchanged."""
    cu = _run(project_dir, name='CU')
    cu.get_eigenmode_qois()
    srf = _run(os.path.join(project_dir, 'srf'), name='SRF',
               config={'surface_resistance': 1e-8})
    srf.get_eigenmode_qois()
    assert srf.eigenmode_qois['Q []'] > 100 * cu.eigenmode_qois['Q []']
    assert abs(srf.eigenmode_qois['G [Ohm]'] - cu.eigenmode_qois['G [Ohm]']) < 1.0


def test_per_polarisation_rerun_isolation(project_dir):
    """A dipole rerun must not touch monopole results."""
    cav = _run(project_dir, config={'polarisation': ['monopole', 'dipole'], 'n_modes': 3})
    eig = os.path.join(cav.self_dir, 'eigenmode')
    mono = os.path.join(eig, 'monopole', 'qois.json')
    dip = os.path.join(eig, 'dipole', 'qois.json')
    assert os.path.exists(mono) and os.path.exists(dip)
    mono_mtime = os.path.getmtime(mono)

    cavs = Cavities(project_dir)
    cavs.add_cavity([cav], ['CAV2'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': 'dipole', 'n_modes': 3})
    assert os.path.exists(mono), "dipole rerun deleted monopole results"
    assert os.path.getmtime(mono) == mono_mtime, "dipole rerun rewrote monopole results"


def test_plot_fields_unsolved_polarisation_is_actionable(project_dir, capsys):
    """Plotting a polarisation that wasn't solved reports what IS available
    instead of a bare 'file does not exist'."""
    cav = _run(project_dir)                      # monopole only (fast)
    cav.plot_fields(mode=0, which='E', plotter='matplotlib', pol='dipole')
    captured = capsys.readouterr()
    out = captured.out + captured.err
    # the message names the missing pol and lists what is available
    assert 'dipole' in out
    assert 'monopole' in out


def test_mpole_only_results_do_not_crash_qoi_read(project_dir):
    """get_eigenmode_qois must not crash when only higher-order-mode results
    exist (no monopole) — it should return gracefully. Crafts the state from a
    monopole run so the assertion doesn't depend on m-pole QOI specifics."""
    cav = _run(project_dir)
    eig = os.path.join(cav.self_dir, 'eigenmode')
    shutil.move(os.path.join(eig, 'monopole'), os.path.join(eig, 'dipole'))
    assert cav._available_polarisations() == ['dipole']
    cav.get_eigenmode_qois()          # must not raise

    # with no eigenmode results at all, it must raise clearly
    shutil.rmtree(os.path.join(eig, 'dipole'))
    with pytest.raises(FileNotFoundError):
        cav.get_eigenmode_qois()


def test_mpole_dispersion_plot(project_dir):
    """plot_dispersion(pol='dipole') plots the dipole passband of a 2-cell
    cavity (distinct from, and above, the monopole passband)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cavs = Cavities(project_dir)
    cav = EllipticalCavity(2, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['D'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 4})
    cav.get_eigenmode_qois()

    plt.close('all')
    ax = cav.plot_dispersion(pol='monopole')
    ax = cav.plot_dispersion(pol='dipole', ax=ax)
    # both passbands drawn (2 lines), dipole above monopole
    assert len(ax.lines) == 2
    mono_y = ax.lines[0].get_ydata()
    dip_y = ax.lines[1].get_ydata()
    assert min(dip_y) > min(mono_y)
    plt.close('all')


def test_field_and_mesh_render_non_blank(project_dir):
    """The matplotlib renderers produce a non-empty figure (regression for the
    hardcoded edge mask that deleted every triangle, and for plot_fields
    ignoring plotter='matplotlib')."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cav = _run(project_dir)
    plt.close('all')
    cav.plot_fields(mode=1, which='E', plotter='matplotlib')
    fig = plt.gcf()
    # a real contour plot adds collections/artists; a blank one has none
    assert fig.axes and fig.axes[0].collections, "field plot is blank"
    plt.close('all')

    cav.plot_mesh(plotter='matplotlib')
    fig = plt.gcf()
    assert fig.axes and fig.axes[0].lines, "mesh plot is blank"
    plt.close('all')


def test_legacy_flat_layout_fallback(project_dir):
    """Monopole results written flat in eigenmode/ (refactor-era layout) must
    still be readable via the monopole_dir fallback."""
    cav = _run(project_dir)
    eig = os.path.join(cav.self_dir, 'eigenmode')
    mono = os.path.join(eig, 'monopole')
    for f in os.listdir(mono):
        shutil.move(os.path.join(mono, f), os.path.join(eig, f))
    os.rmdir(mono)

    cavs = Cavities(project_dir)
    cav2 = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav2], ['CAV'])
    cav2.get_eigenmode_qois()      # reads via fallback, must not raise
    assert cav2.freq > 0


# --- mode_of_interest: which eigenmode becomes the headline qois.json ---------

def test_modes_of_interest_resolution():
    """1-based in config, 0-based internally, and a polarisation may have several.
    Defaults: monopole -> n_cells (the pi-mode), m-pole -> 1."""
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP

    class Cav:
        n_cells = 9

    moi = NGSolveMEVP.modes_of_interest
    assert moi(Cav(), 0) == [8]                     # default monopole = n_cells = 9
    assert moi(Cav(), 1) == [0]                     # default dipole = 1
    assert moi(Cav(), 0, {'mode_of_interest': 1}) == [0]
    # a plain int applies to every polarisation
    assert moi(Cav(), 2, {'mode_of_interest': 4}) == [3]
    # any number of modes of interest for one polarisation; the first is primary,
    # order is preserved and duplicates collapse. Nothing is capped at two.
    assert moi(Cav(), 1, {'mode_of_interest': [1, 3]}) == [0, 2]
    assert moi(Cav(), 1, {'mode_of_interest': [3, 1]}) == [2, 0]
    assert moi(Cav(), 1, {'mode_of_interest': [2, 2]}) == [1]        # deduplicated
    assert moi(Cav(), 1, {'mode_of_interest': [1, 2, 3, 4, 5]}) == [0, 1, 2, 3, 4]
    assert moi(Cav(), 1, {'mode_of_interest': [5, 1, 9, 3]}) == [4, 0, 8, 2]
    assert moi(Cav(), 0, {'mode_of_interest': list(range(1, 8))}) == list(range(7))
    # polarisations may each carry a different number of them
    cfg_n = {'mode_of_interest': {'monopole': [1, 2, 3, 4], 'dipole': 4, 'quadrupole': [2, 3, 4]}}
    assert moi(Cav(), 0, cfg_n) == [0, 1, 2, 3]
    assert moi(Cav(), 1, cfg_n) == [3]
    assert moi(Cav(), 2, cfg_n) == [1, 2, 3]
    # dict keyed by name or by azimuthal number
    cfg = {'mode_of_interest': {'monopole': 9, 'dipole': [1, 2]}}
    assert moi(Cav(), 0, cfg) == [8]
    assert moi(Cav(), 1, cfg) == [0, 1]
    assert moi(Cav(), 2, cfg) == [0]                # absent -> m-pole default
    assert moi(Cav(), 1, {'mode_of_interest': {1: 3}}) == [2]

    class One:
        n_cells = 1
    assert moi(One(), 0) == [0]                     # 1-cell cavity/gun/pillbox

    with pytest.raises(ValueError, match='1-based'):
        moi(Cav(), 0, {'mode_of_interest': 0})
    with pytest.raises(ValueError, match='must be a positive integer'):
        moi(Cav(), 0, {'mode_of_interest': 2.5})
    with pytest.raises(ValueError, match='empty'):
        moi(Cav(), 0, {'mode_of_interest': []})
    with pytest.raises(ValueError, match='exceeds'):
        moi(Cav(), 0, {'mode_of_interest': 12}, n_modes=5)


def test_mode_of_interest_selects_the_headline_mode(project_dir):
    """Default picks the pi-mode; an override picks the requested passband mode."""
    import json
    from cavsim2d.cavity import Cavities, EllipticalCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]

    def run(tag, extra):
        cav = EllipticalCavity(3, tesla, tesla, tesla, beampipe='both')
        cavs = Cavities(os.path.join(project_dir, tag))
        cavs.add_cavity([cav], [tag])
        cfg = {'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'}
        cfg.update(extra)
        cavs.run_eigenmode(cfg)
        cav.get_eigenmode_qois()
        with open(os.path.join(cav.self_dir, 'eigenmode', 'monopole',
                               'qois_all_modes.json')) as fh:
            all_modes = json.load(fh)
        return cav.eigenmode_qois, [all_modes[str(i)]['freq [MHz]'] for i in range(3)]

    qois, freqs = run('default', {})
    assert freqs[0] < freqs[1] < freqs[2]                 # a rising passband
    assert qois['mode_of_interest'] == '3'                # n_cells, the pi-mode
    assert qois['freq [MHz]'] == pytest.approx(freqs[2], abs=1e-6)

    qois1, freqs1 = run('mode1', {'mode_of_interest': 1})
    assert qois1['mode_of_interest'] == '1'
    assert qois1['freq [MHz]'] == pytest.approx(freqs1[0], abs=1e-6)

    qois2, freqs2 = run('dict', {'mode_of_interest': {'monopole': 2}})
    assert qois2['freq [MHz]'] == pytest.approx(freqs2[1], abs=1e-6)


def test_mode_of_interest_is_metadata_not_a_qoi(project_dir):
    """Stored as a string so UQ's numeric coercion drops it, like 'polarisation'."""
    from cavsim2d.cavity import Cavities, EllipticalCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='both')
    cavs = Cavities(project_dir)
    cavs.add_cavity([cav], ['C'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    cav.get_eigenmode_qois()
    assert isinstance(cav.eigenmode_qois['mode_of_interest'], str)


def test_direct_solver_probe_and_default():
    """The default backend is whatever this NGSolve build actually provides:
    pardiso on Windows, umfpack on mac/linux, sparsecholesky as the fallback."""
    import platform
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import (default_direct_solver,
                                                        direct_solver_available)
    assert direct_solver_available('sparsecholesky')      # always built in
    assert direct_solver_available('a-backend-that-does-not-exist') is False

    chosen = default_direct_solver()
    assert direct_solver_available(chosen)
    preferred = ('pardiso', 'umfpack') if platform.system() == 'Windows' else ('umfpack', 'pardiso')
    for name in preferred:
        if direct_solver_available(name):
            assert chosen == name
            break
    else:
        assert chosen == 'sparsecholesky'
