"""Multipacting integration tests.

The physics (RK4 tracking, secondary emission) is ported verbatim from the user's
validated PyMultipact; these tests cover the *cavsim2d plumbing*: the SEY table,
the eigenmode-field adapter (in-plane E + reconstructed 1j-phase H), and the
``cav.multipacting`` run/plot surface end-to-end.
"""
import os
import pickle

import numpy as np
import pytest


def test_sey_default_table_loads_and_is_picklable():
    """The bundled default SEY table loads and its interpolator survives pickling
    (multiprocessing passes it through process arguments)."""
    from cavsim2d.analysis.multipacting.sey import SEY, DEFAULT_SEY
    assert os.path.exists(DEFAULT_SEY)
    sey = SEY()                       # default table
    assert len(sey.data) > 5
    assert sey.Emin < 300 < sey.Emax
    # copper-like: yield exceeds 1 somewhere (multipacting is possible)
    assert (np.asarray(sey.data['sey']) > 1).any()
    # linear interpolation, picklable, deterministic
    round_trip = pickle.loads(pickle.dumps(sey))
    assert round_trip.sey(300.0) == pytest.approx(sey.sey(300.0))


# --- everything below needs ngsolve/gmsh -------------------------------------

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL                        # noqa: E402
from cavsim2d import EllipticalCavity                # noqa: E402


def _solved(tmp_path, monkeypatch, name='MP'):
    monkeypatch.chdir(tmp_path)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='none', name=name)
    cav.eigenmode.run({'polarisation': 'monopole', 'boundary_conditions': 'mm',
                       'mesh_config': {'h': 8, 'p': 3}})
    return cav


def test_field_adapter_inplane_E_and_1j_H(tmp_path, monkeypatch):
    """The adapter takes the in-plane E (a 2-vector) from the product-space mode
    and RECONSTRUCTS H = 1j/(mu0 w) curl(E) -- the 1j phase (which cavsim2d drops
    for its magnitude-only QOIs) is what the Lorentz integration needs."""
    from cavsim2d.solvers.eigenmode_result import monopole_dir
    from cavsim2d.analysis.multipacting.driver import (load_eigenmode_fields,
                                                       _surface_points)
    from cavsim2d.analysis.multipacting.fields import build_emfield

    cav = _solved(tmp_path, monkeypatch)
    fd = monopole_dir(cav.eigenmode.folder)
    mesh, gfu_E, _ = load_eigenmode_fields(fd)
    freqs = [m.frequency for m in cav.eigenmode.modes]
    assert freqs[0] > 100                  # physical monopole fundamental (MHz)
    assert freqs == sorted(freqs)          # ascending: [0] is the fundamental

    em = build_emfield(gfu_E, 0, freqs[0])
    xsurf = _surface_points(mesh)
    assert len(xsurf) > 5
    zc, rc = xsurf[len(xsurf) // 2]
    E = np.asarray(em.e(mesh(float(zc), float(rc))), dtype=complex).ravel()
    H = np.asarray(em.h(mesh(float(zc), float(rc))), dtype=complex).ravel()
    assert E.shape == (2,)                 # in-plane (E_z, E_r)
    assert H.shape == (1,)                 # azimuthal, scalar
    assert abs(H.imag).max() > 0           # the physical 1j phase is present


def test_multipacting_run_end_to_end(tmp_path, monkeypatch):
    """cav.multipacting.run(...) auto-reads the eigenmode field, sweeps the peak
    field and produces a finite counter function of the right length; the plots
    render. A short track keeps it fast -- the physics is validated separately.
    proc_count=2 exercises the parallel path: workers are plain subprocesses
    (`python -m ...driver`), so no __main__ guard is needed here (or in
    notebooks) even on Windows."""
    import matplotlib
    matplotlib.use('Agg')

    monkeypatch.chdir(tmp_path)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='none', name='MPE2E')
    epks = list(1e6 * np.linspace(5, 45, 4))
    cav.multipacting.run({'proc_count': 2, 'epks': epks,
                          'xrange': [-0.005, 0.005],
                          'phis': list(np.linspace(0, 2 * np.pi, 6)),
                          't_max': 50e-10,
                          'eigenmode_config': {'mesh_config': {'h': 8, 'p': 3}}})

    mp = cav.multipacting
    assert os.path.exists(os.path.join(mp.folder, 'mresults.pkl'))
    assert len(mp.counter) == len(epks)
    assert np.all(np.isfinite(mp.counter))
    assert (mp.counter >= 0).all()
    # peak-field axis is in MV/m and matches the requested sweep
    assert mp.epk.max() == pytest.approx(45.0, rel=0.05)
    assert len(mp.final_energy) == len(epks)
    # the sweep records its wall-clock time
    assert mp.results['sweep_time [s]'] > 0

    ax = mp.plot_counter()
    assert ax is not None and 'c_{20}/c_0' in ax.get_ylabel()
    assert mp.plot_final_energy() is not None
    assert mp.plot_sey() is not None

    # show=True is the default (a plt.show() fires); show=False keeps the axes
    # live so a second cavity can overlay onto the same one
    import matplotlib.pyplot as plt
    calls = {'n': 0}
    orig = plt.show
    plt.show = lambda *a, **k: calls.__setitem__('n', calls['n'] + 1)
    try:
        mp.plot_counter()                     # default show -> +1
        assert calls['n'] == 1
        axc = mp.plot_counter(show=False)     # compose -> no show
        mp.plot_counter(ax=axc, show=False)
        assert calls['n'] == 1
    finally:
        plt.show = orig


def test_multipacting_default_sey_and_folder(tmp_path, monkeypatch):
    """A bare cavity exposes cav.multipacting with the default SEY; set_sey swaps
    it; the run folder is <cav>/multipacting/."""
    from cavsim2d.analysis.multipacting.sey import DEFAULT_SEY
    monkeypatch.chdir(tmp_path)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='none', name='MPsey')
    assert cav.multipacting.sey.filepath == DEFAULT_SEY
    cav.multipacting.set_sey(DEFAULT_SEY)            # explicit, same table
    assert cav.multipacting.sey is not None


def test_profile_mesh_edge_maxh_refines_only_that_boundary():
    """Profile.mesh(edge_maxh={'PEC': h}) refines the wall to ~h while the axis
    keeps the global size. (Per-edge OCC maxh hints are silently dropped by the
    shape-healing, so this is done with RestrictH size points -- the test pins
    the behaviour, not the mechanism.)"""
    from ngsolve import BND
    prof = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='none').profile()

    def wall(mesh, name):
        names = mesh.GetBoundaries()
        lens = []
        for el in mesh.Elements(BND):
            if names[el.index] != name:
                continue
            vs = [mesh[v].point for v in el.vertices]
            lens.append(float(np.hypot(vs[1][0] - vs[0][0], vs[1][1] - vs[0][1])))
        return len(lens), float(np.mean(lens))

    coarse = prof.mesh(maxh=8e-3, order=1)
    fine = prof.mesh(maxh=8e-3, order=1, edge_maxh={'PEC': 1e-3})

    n0, mean0 = wall(coarse, 'PEC')
    n1, mean1 = wall(fine, 'PEC')
    assert n1 > 3 * n0                       # wall really refined
    assert mean1 < 1.6e-3                    # ~the requested 1 mm
    # the axis keeps the global size (refinement is local to the wall)
    _, axi0 = wall(coarse, 'AXI')
    _, axi1 = wall(fine, 'AXI')
    assert axi1 == pytest.approx(axi0, rel=0.3)

    with pytest.raises(ValueError, match='edge_maxh'):
        prof.mesh(maxh=8e-3, edge_maxh={'WALL': 1e-3})


def test_pec_maxh_gives_multipacting_its_own_field(tmp_path, monkeypatch):
    """pec_maxh solves the monopole field on a surface-refined mesh into
    multipacting/field/ and LEAVES cav.eigenmode's results byte-identical --
    the user's eigenmode QOIs (and everything downstream) must never be
    recomputed or replaced as a side effect of a multipacting setting."""
    import hashlib
    import json as json_mod
    import matplotlib
    matplotlib.use('Agg')

    cav = _solved(tmp_path, monkeypatch, name='MPown')     # normal eigenmode run
    eig_dir = os.path.join(cav.self_dir, 'eigenmode', 'monopole')

    def digest(p):
        with open(p, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    qois_before = digest(os.path.join(eig_dir, 'qois.json'))
    mesh_before = digest(os.path.join(eig_dir, 'mesh.pkl'))
    freq_before = cav.eigenmode.qois['freq [MHz]']

    cav.multipacting.run({'proc_count': 1, 'pec_maxh': 1.0,
                          'xrange': [-0.005, 0.005],
                          'epks': list(1e6 * np.linspace(5, 45, 3)),
                          'phis': list(np.linspace(0, 2 * np.pi, 6)),
                          't_max': 40e-10})
    mp = cav.multipacting

    # own field written, and the sweep used it
    fdir = os.path.join(mp.folder, 'field')
    for fn in ('mesh.pkl', 'gfu_EH.pkl', 'freqs.json'):
        assert os.path.exists(os.path.join(fdir, fn))
    assert mp.results['fields_dir'].endswith('field')
    assert len(mp.counter) == 3 and np.all(np.isfinite(mp.counter))

    # the shared eigenmode results are byte-identical -- untouched
    assert digest(os.path.join(eig_dir, 'qois.json')) == qois_before
    assert digest(os.path.join(eig_dir, 'mesh.pkl')) == mesh_before

    # same physics on the refined mesh: the fundamental agrees
    with open(os.path.join(fdir, 'freqs.json')) as f:
        f_own = json_mod.load(f)[0]
    assert f_own == pytest.approx(freq_before, rel=1e-3)


def test_stepwise_api_stage_preview_and_overlap_warning(tmp_path, monkeypatch):
    """The stepwise workflow: stage the mesh/emission band/sweep with set_*(),
    preview the emission points BEFORE running, then run. Staged values are
    used; a run()-config key that collides with a staged one raises a
    UserWarning and takes precedence. The saved config.json is complete (every
    DEFAULT_MULTIPACTING_CONFIG key present, merged result recorded)."""
    import json
    import warnings as warnings_mod
    import matplotlib
    matplotlib.use('Agg')

    from cavsim2d.solvers.solver_objects import DEFAULT_MULTIPACTING_CONFIG

    cav = _solved(tmp_path, monkeypatch, name='MPstep')
    mp = cav.multipacting
    mp.set_emission_points([-0.005, 0.005]).set_epks([2e6, 4e6])
    mp.set_phis(list(np.linspace(0, 2 * np.pi, 4)))

    # preview before running: emission sites of the staged xrange
    ax = mp.show_emission_points()
    assert ax is not None

    with warnings_mod.catch_warnings(record=True) as w:
        warnings_mod.simplefilter('always')
        mp.run({'xrange': [-0.004, 0.004]}, proc_count=1, t_max=20e-10)
    msgs = [str(x.message) for x in w if 'previously staged' in str(x.message)]
    assert msgs and "['xrange']" in msgs[0]

    cfg = json.load(open(os.path.join(mp.folder, 'config.json')))
    assert not set(DEFAULT_MULTIPACTING_CONFIG) - set(cfg)   # complete
    assert cfg['xrange'] == [-0.004, 0.004]                  # run() config wins
    assert cfg['epks'] == [2e6, 4e6]                         # staged survives
    assert cfg['proc_count'] == 1                            # kwarg recorded
    assert len(mp.counter) == 2


def test_eigenmode_kwargs_and_complete_saved_config(tmp_path, monkeypatch):
    """Eigenmode config keys can be passed as kwargs; the saved config.json is
    the complete merged dict (explicit None for disabled options), and a
    complete config with uq_config=None must NOT enter the UQ path."""
    import json

    from cavsim2d.solvers.solver_objects import DEFAULT_EIGENMODE_CONFIG

    monkeypatch.chdir(tmp_path)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='none', name='EKW')
    cav.eigenmode.run(mesh_config={'h': 10})                 # kwargs only

    saved = json.load(open(os.path.join(cav.eigenmode.folder, 'config.json')))
    assert not set(DEFAULT_EIGENMODE_CONFIG) - set(saved)    # complete
    assert saved['mesh_config'] == {'h': 10, 'p': 3, 'adaptive': None}  # nested merge
    assert saved['uq_config'] is None
    assert cav.eigenmode.qois['freq [MHz]'] > 100            # it actually solved
    assert not os.path.exists(os.path.join(cav.self_dir, 'uq'))  # no UQ side path


def test_trajectory_color_values_and_selection():
    """The trajectory-plot helpers: per-segment velocity/energy values, the log
    colour norm (real dynamic range: ~1e2 eV wall orbits vs ~1e5 eV
    gap-crossers), and the epk/phase/traj selection used by the animation."""
    from cavsim2d.analysis.multipacting.plots import (_traj_values, _traj_norm,
                                                      _select_trajectories)

    # straight path, constant speed: [z, r, phase] rows like Particles.paths
    path = np.column_stack([np.linspace(0, 1e-3, 50),
                            np.full(50, 0.1), np.zeros(50)])
    phis_v = np.linspace(0, 2 * np.pi, 4)

    class P:
        def __init__(self, paths, phis):
            self.bright_set = paths
            self.bright_init_phi = phis
            self.phis_v = phis_v

    class S:
        particles = [P([path], [0.0]), P([path, path], [0.0, phis_v[2]])]

    assert len(_select_trajectories(S)) == 3                       # all
    assert len(_select_trajectories(S, epk_i=1)) == 2              # one level
    assert len(_select_trajectories(S, epk_i=1, traj=0)) == 1      # one traj
    assert len(_select_trajectories(S, phi_i=2)) == 1              # one phase

    # traj and phi_i are UNIONed within a level (not intersected): at level 1,
    # traj 0 (phase index 0) OR phase index 2 -> both trajectories. The old AND
    # semantics returned nothing here (traj 0 is not at phase 2).
    both = _select_trajectories(S, epk_i=1, traj=[0], phi_i=[2])
    assert {k for _, k, _ in both} == {0, 1}
    # out-of-range field level is skipped with a warning, not an IndexError
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert len(_select_trajectories(S, epk_i=[0, 99])) == 1
    assert any('out of range' in str(x.message) for x in w)

    dt = 1e-11
    v, vlab = _traj_values(path, dt, 'velocity')
    e, elab = _traj_values(path, dt, 'energy')
    assert v == pytest.approx((1e-3 / 49) / dt)
    assert (e > 0).all() and 'eV' in elab and 'm/s' in vlab
    norm = _traj_norm(e)
    assert norm.vmin > 0 and norm.vmax >= norm.vmin                # log-safe
    with pytest.raises(ValueError, match='color_by'):
        _traj_values(path, dt, 'speediness')


def test_trajectory_animation_returns_funcanimation_and_saves(tmp_path):
    """animate_trajectories builds a FuncAnimation (so `anim.save(...)` works)
    and, given `save`, writes the file. A synthetic bright trajectory keeps this
    off the heavy solver — the plumbing (selection, colour values, writer) is
    what's under test."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.animation import FuncAnimation
    from cavsim2d.analysis.multipacting.plots import trajectory_animation

    # one synthetic 20-hit path (z, r, phase), bouncing near the equator
    t = np.linspace(0, 1, 60)
    path = np.column_stack([1e-3 * np.sin(2 * np.pi * t),
                            0.1 + 1e-4 * np.abs(np.sin(np.pi * t)), t])

    class _P:
        bright_set = [path]
        phis_v = np.linspace(0, 2 * np.pi, 4)

    class _Prof:
        def contour_points(self, ds, skip=()):
            z = np.linspace(-0.05, 0.05, 40)
            return np.column_stack([z, 0.1 - 0.5 * z ** 2 / 0.05])

    class _Cav:
        def profile(self):
            return _Prof()

    class _S:
        particles = [_P()]
        epk = np.array([30.0])
        results = {'freq [MHz]': 1300.0}
        cavity = _Cav()

    out = tmp_path / 'anim.gif'
    anim = trajectory_animation(_S, step=2, trail=5, fps=10,
                                save=str(out), progress=False, embed=False)
    assert isinstance(anim, FuncAnimation)
    assert out.exists() and out.stat().st_size > 0


def test_draw_trajectory_uses_sequences_and_equal_aspect():
    """_draw_trajectory plots the start marker with sequences (the scalar
    set_data crash inherited from PyMultipact) and keeps equal aspect."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cavsim2d.analysis.multipacting.plots import _draw_trajectory

    boundary = np.column_stack([np.linspace(-0.05, 0.05, 30), np.full(30, 0.1)])
    path = np.column_stack([np.linspace(0, 1e-3, 40), np.full(40, 0.1),
                            np.zeros(40)])
    fig, ax = plt.subplots()
    lc = _draw_trajectory(ax, boundary, path, 1e-11, 'energy',
                          plt.get_cmap('inferno'), 'label')
    assert lc is not None                              # colour-coded -> LineCollection
    assert ax.get_aspect() == 1.0                      # equal aspect
    lc2 = _draw_trajectory(ax, boundary, path, 1e-11, None,
                           plt.get_cmap('inferno'), 'label')
    assert lc2 is None                                 # plain line, no collection
    plt.close(fig)


def test_concave_corner_geometry_does_not_crash_the_sweep(tmp_path):
    """Regression: a sharp CONCAVE (re-entrant) corner used to kill the sweep.

    The wall-collision point is found on the straight collision polyline, so at a
    re-entrant corner it can land a hair outside the meshed domain and the field
    evaluation raised ``NgException: Meshpoint not in mesh!`` — surfacing only as
    "multipacting worker N produced no result -- it crashed". `RFGun(beampipe=...)`
    introduces exactly such a corner where the cathode pipe meets the funnel.
    `Integrators._wall_mip` now nudges the point back inside (no-op when it is
    already valid), so the sweep completes.
    """
    pytest.importorskip("ngsolve")
    from cavsim2d import RFGun

    gun = {'geometry': {'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
                        'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
                        'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
                        'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}
    cav = RFGun(gun, beampipe='both')        # cathode pipe -> concave corner
    cav.name = 'gun_corner'
    cav.set_workspace(os.path.join(tmp_path, 'gun_corner'))
    cav.multipacting.run({'proc_count': 1,                 # in-process: real traceback
                          'xrange': [-0.30, 0.0],           # cathode pipe + funnel
                          'epks': list(1e6 * np.linspace(1, 20, 2)),
                          'phis': list(np.linspace(0, 2 * np.pi, 8))})
    assert len(cav.multipacting.counter) == 2             # both field levels survived


def test_densify_wall_adds_sites_without_moving_the_wall():
    """More launch sites from the SAME mesh: emission sites are wall-polyline
    vertices, so a coarse mesh gives few in a narrow band. densify_wall
    interpolates along the existing straight segments — so the count goes up while
    the polyline (and hence the collision geometry) is unchanged."""
    from cavsim2d.analysis.multipacting.driver import densify_wall

    # a straight wall stretch, 5 mm apart: only 3 vertices inside [0, 0.01]
    wall = np.column_stack([np.linspace(0.0, 0.05, 11), np.full(11, 0.1)])
    band = [0.0, 0.01]
    assert ((wall[:, 0] >= band[0]) & (wall[:, 0] <= band[1])).sum() == 3

    dense = densify_wall(wall, band, n_points=25)
    inb = (dense[:, 0] >= band[0]) & (dense[:, 0] <= band[1])
    assert inb.sum() == 25                       # band resampled to 25 sites
    # geometry untouched: same extent, and every densified point lies ON the wall
    assert dense[:, 0].min() == pytest.approx(wall[:, 0].min())
    assert dense[:, 0].max() == pytest.approx(wall[:, 0].max())
    assert np.allclose(dense[inb][:, 1], 0.1)    # still on the r = 0.1 wall
    # out-of-band vertices survive untouched
    assert ((dense[:, 0] > band[1])).sum() == (wall[:, 0] > band[1]).sum()
    # asking for fewer than already present is a no-op
    assert np.array_equal(densify_wall(wall, band, n_points=2), wall)
    assert np.array_equal(densify_wall(wall, band, n_points=None), wall)


def test_compute_field_takes_eigenmode_config_polarisation_and_modes(tmp_path, monkeypatch):
    """compute_field() accepts the same keys as an eigenmode config, so
    multipacting can be driven by a chosen mode of a chosen polarisation (e.g. the
    2nd monopole mode, or a dipole mode) rather than only the fundamental."""
    monkeypatch.chdir(tmp_path)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both', name='MPpol')
    mp = cav.multipacting
    mp.compute_field(polarisation='monopole', n_modes=3,
                     mesh_config={'h': 8, 'p': 2}, pec_maxh=2)
    mono = mp.field_frequencies
    assert mp.has_field and len(mono) >= 3
    assert mono == sorted(mono) and mono[0] > 100          # ascending, physical

    cav2 = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both', name='MPdip')
    dip = cav2.multipacting.compute_field(polarisation='dipole', n_modes=3,
                                          mesh_config={'h': 8, 'p': 2},
                                          pec_maxh=2).field_frequencies
    assert len(dip) >= 3
    # a dipole band is a different spectrum from the monopole one
    assert abs(dip[0] - mono[0]) > 1.0

    # the chosen mode is what the sweep tracks, and is recorded
    mp.set_emission_points([-0.005, 0.005], n_points=12)
    mp.run({'proc_count': 1, 'mode': 1, 'epks': [2e6],
            'phis': list(np.linspace(0, 2 * np.pi, 4)), 't_max': 20e-10})
    assert mp.config['mode'] == 1


def test_animation_follow_zoom_tracks_the_live_particles(tmp_path, monkeypatch):
    """zoom='follow' gives a moving camera: the view re-frames each frame onto the
    trajectories still alive, so an opening burst across the domain closes in as
    the multipacting localises. zoom='auto' stays put for the whole animation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='none', name='MPanim')
    mp = cav.multipacting
    mp.set_mesh_parameters(h=6, p=3, pec_maxh=1)
    mp.set_emission_points([-0.005, 0.0], step=0.001)
    mp.run({'proc_count': 1, 'epks': list(1e6 * np.linspace(35, 45, 2)),
            'phis': list(np.linspace(0, 2 * np.pi, 24))})
    assert any(len(p.bright_set) for p in mp.particles), 'need bright trajectories'

    def spans(zoom):
        anim = mp.animate_trajectories(zoom=zoom, progress=False, embed=False)
        fig = plt.gcf()
        ax = fig.axes[0]
        out = []
        for f in (0, 20, 60, 150):
            anim._func(f)                       # render that frame
            out.append(ax.get_xlim()[1] - ax.get_xlim()[0])
        plt.close(fig)
        return out

    fixed = spans('auto')
    assert max(fixed) - min(fixed) < 1e-9       # one box for the whole animation

    moving = spans('follow')
    assert max(moving) - min(moving) > 1e-3     # the camera actually moves
    assert moving[-1] < moving[0]               # and ends closer in than it began


def test_densify_wall_handles_bands_with_under_two_vertices():
    """A band holding 0 or 1 wall vertices has no segment of its own to walk
    along — densify_wall brackets it with the neighbouring vertices so a coarse
    mesh can still be given launch sites (the case that matters most)."""
    from cavsim2d.analysis.multipacting.driver import densify_wall

    wall = np.column_stack([np.linspace(0.0, 0.05, 11), np.full(11, 0.1)])  # 5 mm
    for band in ([0.011, 0.014],          # 0 vertices inside
                 [0.009, 0.012]):         # 1 vertex inside
        dense = densify_wall(wall, band, n_points=25)
        inb = (dense[:, 0] >= band[0]) & (dense[:, 0] <= band[1])
        assert inb.sum() == 25
        assert np.allclose(dense[inb][:, 1], 0.1)      # still exactly on the wall
    # wall extent is never altered
    d = densify_wall(wall, [0.0, 0.01], n_points=25)
    assert d[:, 0].min() == pytest.approx(0.0) and d[:, 0].max() == pytest.approx(0.05)
