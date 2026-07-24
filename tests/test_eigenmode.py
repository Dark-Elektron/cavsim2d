"""Eigenmode regression tests: monopole baseline, m-pole solver vs analytic,
per-polarisation result folders and rerun semantics, legacy-layout fallback."""
import gc
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d import Study, EllipticalCavity


def _run(project_dir, name='CAV', n_cells=1, config=None):
    cavs = Study(project_dir)
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


def test_qois_df_spans_every_polarisation(project_dir):
    """qois_df flattens the dict-of-dicts results into one filterable table:
    a row per (polarisation, mode), keyed '<m>-<mode>' like the convergence df."""
    cav = _run(project_dir, name='DF',
               config={'polarisation': ['monopole', 'dipole'], 'n_modes': 4})
    df = cav.eigenmode.qois_df

    assert not df.empty
    assert {'m', 'polarisation', 'mode', 'mode_index',
            'freq [MHz]', 'R/Q [Ohm]', 'Q []'}.issubset(df.columns)
    assert set(df['polarisation']) == {'monopole', 'dipole'}
    assert set(df['m']) == {0, 1}
    # polarisation-first composite key, and it is unique per row
    assert {'0-0', '1-0'}.issubset(set(df['mode_index']))
    assert df['mode_index'].is_unique
    # rows really are that polarisation's modes, in order
    mono = df[df.polarisation == 'monopole']
    assert mono['mode'].tolist() == sorted(mono['mode'].tolist())
    assert (mono['freq [MHz]'] > 0).all()

    # a cavity with no eigenmode results yields an empty frame, not an error
    fresh = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    Study(os.path.join(project_dir, 'empty')).add_cavity([fresh], ['E'])
    assert fresh.eigenmode.qois_df.empty


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


def _pillbox_mesh(R, Lz, maxh=0.008):
    """PEC cylinder with the axis at r = 0 (where the 1/r weights are singular)."""
    from ngsolve import Mesh
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    p = [geo.AppendPoint(*pt) for pt in [(0, 0), (Lz, 0), (Lz, R), (0, R)]]
    geo.Append(['line', p[0], p[1]], bc='AXI')
    geo.Append(['line', p[1], p[2]], bc='PEC')
    geo.Append(['line', p[2], p[3]], bc='PEC')
    geo.Append(['line', p[3], p[0]], bc='PEC')
    return Mesh(geo.GenerateMesh(maxh=maxh))


def _analytic_pillbox(m, R, Lz, n_each=3):
    """Analytic TM_mnp (p>=0) and TE_mnp (p>=1) pillbox frequencies [MHz]."""
    from scipy.special import jn_zeros, jnp_zeros
    c0 = 299792458.0
    tm, te = [], []
    for xmn in jn_zeros(m, n_each):
        for pp in range(3):
            tm.append(c0 / (2*np.pi) * np.sqrt((xmn/R)**2 + (pp*np.pi/Lz)**2) * 1e-6)
    for xpmn in jnp_zeros(m, n_each):
        for pp in range(1, 3):
            te.append(c0 / (2*np.pi) * np.sqrt((xpmn/R)**2 + (pp*np.pi/Lz)**2) * 1e-6)
    return np.sort(np.array(tm)), np.sort(np.array(te))


@pytest.mark.parametrize('m', [0, 1, 2])
def test_pillbox_matches_analytic_for_every_m(m):
    """One formulation solves every azimuthal order against the analytic pillbox
    spectrum — TM *and* TE. The old HCurl-only monopole path could not represent
    the m=0 TE modes at all (see test_monopole_te_modes_are_found)."""
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
    R, Lz = 0.10, 0.15
    mesh = _pillbox_mesh(R, Lz)

    freqs, _, _ = NGSolveMEVP()._solve_modes(mesh, 2, m, 8)
    freqs = np.sort([f for f in freqs if f > 1.0])   # guard against any residual ~0 mode
    assert len(freqs) >= 3, freqs

    tm, te = _analytic_pillbox(m, R, Lz)
    ana = np.sort(np.concatenate([tm, te]))
    for a in ana[:3]:
        assert np.min(np.abs(freqs - a) / a) < 0.01, (m, a, freqs)


def test_monopole_te_modes_are_found():
    """The monopole spectrum must contain the TE_0np modes.

    An HCurl-only (E_r, E_z) formulation has no azimuthal unknown, so a mode
    whose E is purely azimuthal is structurally invisible to it: the monopole
    solve silently returned TM modes only. The unified product space carries
    u_phi = r*E_phi, so the TE modes appear — and they carry no acceleration.
    """
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
    R, Lz = 0.10, 0.15
    mesh = _pillbox_mesh(R, Lz)

    freqs, gfu_E, _ = NGSolveMEVP()._solve_modes(mesh, 2, 0, 10)
    freqs = np.array(sorted(f for f in freqs if f > 1.0))

    tm, te = _analytic_pillbox(0, R, Lz)
    # every low TM mode is still there (no regression) ...
    for a in tm[:3]:
        assert np.min(np.abs(freqs - a) / a) < 0.01, ('TM', a, freqs)
    # ... and the TE modes, which the HCurl-only formulation could never find
    for a in te[:2]:
        assert np.min(np.abs(freqs - a) / a) < 0.01, ('TE', a, freqs)

    # a TE mode is purely azimuthal: its in-plane block is zero, so it cannot
    # accelerate. Identify it by matching the lowest analytic TE frequency.
    from ngsolve import Integrate, InnerProduct, Conj, y
    i_te = int(np.argmin(np.abs(freqs - te[0])))
    u_gf, uphi_gf = gfu_E[i_te].components
    e_inplane = abs(Integrate(y * InnerProduct(u_gf, Conj(u_gf)), mesh))
    e_phi = abs(Integrate(1 / y * uphi_gf * Conj(uphi_gf), mesh))
    assert e_inplane < 1e-6 * e_phi, (e_inplane, e_phi)


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

    cavs = Study(project_dir)
    cavs.add_cavity([cav], ['CAV2'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': 'dipole', 'n_modes': 3})
    assert os.path.exists(mono), "dipole rerun deleted monopole results"
    assert os.path.getmtime(mono) == mono_mtime, "dipole rerun rewrote monopole results"


def test_plot_fields_unsolved_polarisation_is_actionable(project_dir, capsys):
    """Plotting a polarisation that wasn't solved reports what IS available
    instead of a bare 'file does not exist'."""
    cav = _run(project_dir)                      # monopole only (fast)
    cav.show_fields(mode=0, which='E', plotter='matplotlib', pol='dipole')
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
    """The dispersion diagram overlays every polarisation and shows *all*
    computed modes (grouped into passbands of n_cells), not just the
    fundamental. The dipole fundamental sits above the monopole fundamental."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cavs = Study(project_dir)
    cav = EllipticalCavity(2, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['D'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 4})
    cav.get_eigenmode_qois()

    n_mono = len(cav.eigenmode.mpole_qois('monopole'))
    n_dip = len(cav.eigenmode.mpole_qois('dipole'))

    plt.close('all')
    # break_axis=False keeps every series on one axis; light_line=False so the
    # only lines are the data series we want to count
    ax = cav.plot_dispersion(break_axis=False, light_line=False)
    # one line per passband; passbands = ceil(modes / n_cells) per polarisation
    import math
    expected = math.ceil(n_mono / 2) + math.ceil(n_dip / 2)
    assert len(ax.lines) == expected
    # every computed mode is plotted (not just the fundamental two)
    assert sum(len(ln.get_ydata()) for ln in ax.lines) == n_mono + n_dip

    mono_fund = next(ln for ln in ax.lines if 'monopole' in ln.get_label())
    dip_fund = next(ln for ln in ax.lines if 'dipole' in ln.get_label())
    assert min(dip_fund.get_ydata()) > min(mono_fund.get_ydata())
    # monopole and dipole are drawn in different (warm) hues
    assert mono_fund.get_color() != dip_fund.get_color()
    # last x tick is pi, not the unreduced (n/n)*pi
    assert ax.get_xticklabels()[-1].get_text() == r'$\pi$'
    plt.close('all')


def test_dispersion_single_polarisation_still_works(project_dir):
    """A named polarisation plots just that one (backward compatible)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cavs = Study(project_dir)
    cav = EllipticalCavity(2, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['D'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 4})
    cav.get_eigenmode_qois()

    plt.close('all')
    ax = cav.plot_dispersion(pol='dipole', break_axis=False, light_line=False)
    assert ax.lines and all('dipole' in ln.get_label() for ln in ax.lines)
    plt.close('all')


def test_dispersion_light_line_is_the_folded_speed_of_light(project_dir):
    """The light line f = c*mu/(2*pi*d) is folded into the reduced zone, so it is
    a triangle wave spanning every band; for TESLA it meets the pi-mode near the
    design 1300 MHz."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from cavsim2d.constants import c0

    cavs = Study(project_dir)
    cav = EllipticalCavity(9, MIDCELL, MIDCELL, MIDCELL, beampipe='none')
    cavs.add_cavity([cav], ['T'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole'], 'n_modes': 12})
    cav.get_eigenmode_qois()

    d = cav._cell_length_m()
    assert d == pytest.approx(2 * MIDCELL[5] * 1e-3)          # 2 * L
    # the zone-boundary value is c/(2d) by construction, and passes near the
    # accelerating pi-mode (exact synchronism is design-dependent, so allow 5%)
    mono = cav.eigenmode.mpole_qois('monopole')
    fundamental = sorted(q['freq [MHz]'] for q in mono.values())[:cav.n_cells]
    pi_mode = fundamental[-1]
    assert c0 / (2 * d) / 1e6 == pytest.approx(pi_mode, rel=0.05)

    plt.close('all')
    ax = cav.plot_dispersion(pol='monopole')                  # light_line default True
    fig = ax.figure
    light = [ln for a in fig.axes for ln in a.lines if ln.get_linestyle() == '--']
    assert light                                              # the light line is drawn
    # it is folded: at least one segment falls (even band) and one rises (odd band)
    slopes = [ln.get_ydata()[-1] - ln.get_ydata()[0] for ln in light]
    assert any(s > 0 for s in slopes) and any(s < 0 for s in slopes)
    # exactly one legend entry for it
    labels = [ln.get_label() for a in fig.axes for ln in a.lines]
    assert labels.count('light line') == 1

    # a geometry with no defined cell period skips it without error
    from cavsim2d import Pillbox
    assert Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')._cell_length_m() is None
    plt.close('all')


def test_dispersion_break_axis_splits_far_apart_passbands(project_dir):
    """With break_axis=True (default) the y-axis splits into stacked sub-axes
    where passbands are far apart, and each series is drawn on every sub-axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cavs = Study(project_dir)
    cav = EllipticalCavity(2, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['D'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 4})
    cav.get_eigenmode_qois()

    series_labels = {s['label'] for s in cav._dispersion_series()[0]}

    plt.close('all')
    ax = cav.plot_dispersion()                      # break_axis defaults True
    fig_axes = ax.figure.axes
    assert len(fig_axes) > 1                         # broken into stacked sub-axes
    # every sub-axis carries the full set of data series (the extra Line2D
    # artists are the diagonal break marks, whose labels start with '_')
    for a in fig_axes:
        drawn = {ln.get_label() for ln in a.lines}
        assert series_labels <= drawn
    # the sub-axes cover disjoint frequency bands (broken, not overlapping)
    ylims = sorted(a.get_ylim() for a in fig_axes)
    for (lo1, hi1), (lo2, hi2) in zip(ylims, ylims[1:]):
        assert hi1 <= lo2 + 1e-6
    # turning it off gives a single axis
    plt.close('all')
    ax = cav.plot_dispersion(break_axis=False)
    assert len(ax.figure.axes) == 1
    plt.close('all')


def test_dispersion_breaks_a_passed_subplot_axis_in_place(project_dir):
    """Passing an ax (e.g. a mosaic slot) must still break: the slot is
    subdivided into stacked panels rather than drawn as one squashed axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cavs = Study(project_dir)
    cav = EllipticalCavity(2, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['D'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 4})
    cav.get_eigenmode_qois()

    plt.close('all')
    fig, ax = plt.subplot_mosaic([[1]], figsize=(5, 10))
    top = cav.plot_dispersion(ax=ax[1])            # break_axis defaults True
    assert len(fig.axes) > 1                        # the slot was subdivided
    assert top.figure is fig                        # broke in place, same figure
    plt.close('all')


def test_dispersion_uses_the_house_font(project_dir):
    """Legend, tick labels and the mathtext fraction ticks share the house font
    (STIX): the tick labels are mathtext, so the mathtext set must be 'stix'."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cavsim2d.utils.style import house_style

    cavs = Study(project_dir)
    cav = EllipticalCavity(2, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['D'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole'], 'n_modes': 2})
    cav.get_eigenmode_qois()

    plt.close('all')
    with house_style():
        assert matplotlib.rcParams['mathtext.fontset'] == 'stix'
        assert 'STIXGeneral' in matplotlib.rcParams['font.serif']
    plt.close('all')


def test_field_and_mesh_render_non_blank(project_dir):
    """The matplotlib renderers produce a non-empty figure (regression for the
    hardcoded edge mask that deleted every triangle, and for show_fields
    ignoring plotter='matplotlib')."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cav = _run(project_dir)
    plt.close('all')
    cav.show_fields(mode=1, which='E', plotter='matplotlib')
    fig = plt.gcf()
    # a real contour plot adds collections/artists; a blank one has none
    assert fig.axes and fig.axes[0].collections, "field plot is blank"
    plt.close('all')

    cav.show_mesh(plotter='matplotlib')
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

    cavs = Study(project_dir)
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
    from cavsim2d import Study, EllipticalCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]

    def run(tag, extra):
        cav = EllipticalCavity(3, tesla, tesla, tesla, beampipe='both')
        cavs = Study(os.path.join(project_dir, tag))
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
    from cavsim2d import Study, EllipticalCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='both')
    cavs = Study(project_dir)
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


def test_compare_scatter_marker_colours_match_the_legend(project_dir):
    """plot_compare drew the whole column once per cavity, so every marker took
    the last cavity's colour while the legend kept per-cavity colours. Each
    cavity must plot only its own point, in a warm house-palette colour."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from cavsim2d.utils.style import WARM

    cavs = Study(project_dir)
    a = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both'); a.plot_label = 'A'
    b = EllipticalCavity(1, [x * 1.02 for x in MIDCELL], [x * 1.02 for x in MIDCELL],
                         [x * 1.02 for x in MIDCELL], beampipe='both'); b.plot_label = 'B'
    cavs.add_cavity([a, b], ['A', 'B'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})

    plt.close('all')
    cavs.eigenmode.plot_compare()
    fig = plt.gcf()
    leg = fig.legends[0]
    leg_colors = [mcolors.to_hex(h.get_facecolor()[0]) for h in leg.legend_handles]
    assert len(set(leg_colors)) == 2                       # two distinct cavities
    assert all(c in [w.lower() for w in WARM] for c in leg_colors)   # warm palette

    # the two markers in the first subplot carry those same two colours
    marker_colors = sorted(mcolors.to_hex(coll.get_facecolor()[0])
                           for coll in fig.axes[0].collections)
    assert marker_colors == sorted(leg_colors)
    plt.close('all')


# --- adaptive mesh refinement ------------------------------------------------

def test_parse_adaptive_normalises():
    """mesh_config['adaptive'] enables refinement (True or a settings dict) and
    is off by default; the stop tolerance defaults to 1e-12."""
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP

    assert NGSolveMEVP._parse_adaptive({}) is None
    assert NGSolveMEVP._parse_adaptive({'adaptive': False}) is None

    cfg = NGSolveMEVP._parse_adaptive({'adaptive': True})
    assert cfg['tol'] == 1e-12
    assert cfg['theta'] == 0.25 and cfg['max_ndof'] == 100000 and cfg['max_refinements'] == 8
    assert 'n_modes' not in cfg                    # the modes-to-check knob is gone

    cfg = NGSolveMEVP._parse_adaptive({'adaptive': {'tol': 1e-8, 'max_ndof': 100}})
    assert cfg['tol'] == 1e-8 and cfg['max_ndof'] == 100
    assert cfg['max_refinements'] == 8            # unspecified keys keep defaults


def test_adaptive_refines_the_mesh_and_records_history(project_dir):
    """Turning on adaptive refinement grows the mesh over several error-driven
    levels, persists a per-level history, and keeps the frequency physical."""
    import json

    cav = _run(project_dir, name='AMR',
               config={'mesh_config': {'h': 40, 'adaptive': {'max_refinements': 3}}})
    hist_path = os.path.join(cav.self_dir, 'eigenmode', 'monopole', 'adaptive_history.json')
    assert os.path.exists(hist_path)
    with open(hist_path) as f:
        history = json.load(f)

    # at least one refinement actually happened (tol=1e-12 won't be met, so the
    # max_refinements cap governs)
    assert len(history) >= 2
    dofs = [lvl['No of DOFs'] for lvl in history]
    elems = [lvl['No of Mesh Elements'] for lvl in history]
    assert dofs == sorted(dofs) and dofs[-1] > dofs[0]      # DOFs grow monotonically
    assert elems[-1] > elems[0]                             # so does the element count
    # each level records a per-mode max_err list; the fundamental (mode 0, which
    # the refinement resolves) improves with refinement
    assert all(isinstance(lvl['max_err'], list) for lvl in history)
    assert history[-1]['max_err'][0] < history[0]['max_err'][0]

    # the reported result is the finest level; frequency is still physical
    cav.get_eigenmode_qois()
    assert 750 < cav.eigenmode_qois['freq [MHz]'] < 850
    # QOIs are evaluated on the finest mesh, matching the last history level
    assert cav.eigenmode_qois['No of Mesh Elements'] == elems[-1]


def test_study_mesh_convergence_is_adaptive_in_h(project_dir):
    """study_mesh_convergence drives h adaptively (no h_passes) while keeping the
    full per-mode table: one row per (order, refinement level, pol, mode) with
    all QOIs; DOFs recorded and growing per level; p still stepped."""
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs = Study(project_dir)
    cavs.add_cavity([cav], ['CONV'])

    cav.study_mesh_convergence(h=40, p=3, p_passes=2, p_step=1, n_modes=4,
                               tol=1e-12, max_refinements=2)
    df = cav.convergence_df_data
    # the rich per-mode schema is preserved (not reduced to freq columns)
    assert {'p', 'h_pass', 'No of DOFs', 'No of Mesh Elements', 'freq [MHz]',
            'Q []', 'R/Q [Ohm]', 'polarisation', 'mode_index', 'max_err'}.issubset(df.columns)
    assert set(df['p'].unique()) == {3, 4}                 # p was stepped, not h_passes
    # default polarisation is monopole + dipole
    assert set(df['polarisation']) == {'monopole', 'dipole'}
    # every mode is kept ('<m>-<mode>' keys, polarisation first), not just one
    assert {'0-0', '0-1', '1-0'}.issubset(set(df['mode_index']))

    # follow the fundamental ('0-0') through the adaptive path: DOFs only grow
    sub = df[(df['p'] == 3) & (df['mode_index'] == '0-0')].sort_values('h_pass')
    dofs = sub['No of DOFs'].tolist()
    assert dofs == sorted(dofs) and dofs[-1] > dofs[0]
    assert sub['No of Mesh Elements'].iloc[-1] > sub['No of Mesh Elements'].iloc[0]
    # each mode carries its own max_err (distinct per mode), and the fundamental
    # improves as the mesh refines
    assert df[(df['p'] == 3) & (df['h_pass'] == 0)]['max_err'].nunique() > 1
    assert sub['max_err'].iloc[-1] < sub['max_err'].iloc[0]


def test_no_mode_stalls_when_many_modes_share_a_mesh():
    """A low mode must not freeze while the mesh is refined for the loud ones.

    Summing the RAW per-mode error fields into the marking field let the high
    modes -- whose errors are orders of magnitude larger -- dominate it. The
    fundamental's worst element then never cleared theta*max, was never refined,
    and its max_err stayed at *exactly* the same value level after level (a flat
    step in an error-vs-DOF plot). _refinement_driver normalises each mode by its
    own peak, so every mode's worst element scores 1.0 and is always marked.
    """
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
    solver = NGSolveMEVP()

    # the driver must give every mode's peak the same weight, however much
    # bigger another mode's error is
    quiet = np.array([1e-9, 1e-12, 1e-12])     # peak in element 0
    loud = np.array([1e-3, 1.0, 1e-3])         # 1e12x bigger, peaks elsewhere
    driver = solver._refinement_driver([quiet, loud])
    assert driver[0] == pytest.approx(1.0)     # the quiet mode's peak still marks
    assert driver[1] == pytest.approx(1.0)
    assert (driver > 0.25).tolist() == [True, True, False]

    # end to end: the fundamental keeps improving with 10 modes on one mesh
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    mesh_p, n_modes = 5, 10
    mesh = solver._build_mesh(cav, 40e-3, mesh_p)
    system = solver._build_system(mesh, mesh_p, 0)

    m0_errs = []
    for _ in range(4):
        _, gfu_E, _ = solver._solve_system(system, n_modes, 20)
        fields = solver._error_fields(mesh, system['fes_rz'], gfu_E)
        m0_errs.append(float(fields[0].max()))
        driver = solver._refinement_driver(fields[:n_modes])
        mesh.ngmesh.Elements2D().NumPy()["refine"] = driver > 0.25
        del gfu_E, fields, driver
        gc.collect()
        mesh.Refine()
        mesh.Curve(mesh_p)

    # strictly decreasing — no flat step (the bug froze it at one exact value)
    assert all(b < a for a, b in zip(m0_errs, m0_errs[1:])), m0_errs
    assert m0_errs[-1] < m0_errs[0] / 100


def test_eigenmode_order_below_two_raises_clear_error():
    """p < 2 is unsupported (the HCurl(p) x H1(p+1) space is rank-deficient at p=1
    and the PINVIT solve returns NaN -> a cryptic scipy 'array must not contain
    infs or NaNs'). _build_system must fail early with a clear, actionable message
    instead. Guards the low-order crash the mesh-convergence example hit."""
    from cavsim2d import CircularWaveguide
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
    import tempfile

    cav = CircularWaveguide(230.0, 200.0)
    cav.set_workspace(os.path.join(tempfile.mkdtemp(), 'cyl'))
    cav.create()
    mevp = NGSolveMEVP()
    mesh = mevp._build_mesh(cav, 60, 2)          # a coarse mesh is enough
    with pytest.raises(ValueError, match=r"p\s*<\s*2|p>=2|unsupported"):
        mevp._build_system(mesh, mesh_p=1, m_pol=0)
