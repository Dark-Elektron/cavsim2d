"""Unit tests for the unified geometry Profile blueprint + native netgen mesh."""
import os
import numpy as np
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("netgen")

from cavsim2d.geometry import Profile
from cavsim2d.solvers.NGSolve.eigen_ngsolve import get_boundary_nodes


def test_profile_meshes_with_tagged_boundaries():
    """A simple box profile meshes natively with its edge tags preserved."""
    p = (Profile()
         .start(0, 0)
         .line_to(0, 0.1, 'PMC')
         .line_to(0.2, 0.1, 'PEC')
         .line_to(0.2, 0, 'PMC')
         .close('AXI'))
    mesh = p.mesh(maxh=0.03, order=1)
    assert set(mesh.GetBoundaries()) == {'AXI', 'PEC', 'PMC'}
    for name in ('AXI', 'PEC', 'PMC'):
        assert len(get_boundary_nodes(mesh, name)) > 0


def test_profile_collinear_split_boundaries():
    """Collinear adjacent segments with *different* tags (e.g. a beam aperture
    and the metal end plate share a plane) must keep distinct boundaries — OCC
    healing strips edge names, so naming is done on the mesh by geometry."""
    p = (Profile()
         .start(0, 0)
         .line_to(0, 0.02, 'PMC')     # aperture (collinear with next)
         .line_to(0, 0.1, 'PEC')      # end plate (same z as aperture)
         .line_to(0.2, 0.1, 'PEC')
         .line_to(0.2, 0, 'PMC')
         .close('AXI'))
    mesh = p.mesh(maxh=0.03, order=1)
    assert set(mesh.GetBoundaries()) == {'AXI', 'PEC', 'PMC'}
    # both the aperture (PMC) and plate (PEC) regions must be present
    assert len(get_boundary_nodes(mesh, 'PMC')) > 0
    assert len(get_boundary_nodes(mesh, 'PEC')) > 0


def assert_meshes_natively(cav, maxh=0.02, order=2, boundaries=('AXI', 'PEC', 'PMC')):
    """Prove the solver takes the native Profile path, not the gmsh fallback.

    Point ``geo_filepath`` at a file that does not exist: had ``_build_mesh``
    fallen back to the gmsh importer the call would fail, so a mesh coming back
    is proof the Profile branch ran.
    """
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
    assert cav.profile() is not None, f"{type(cav).__name__} exposes no profile()"
    cav.geo_filepath = 'does-not-exist.geo'
    mesh = NGSolveMEVP()._build_mesh(cav, maxh, order)
    assert mesh.ne > 0
    assert set(mesh.GetBoundaries()) == set(boundaries)
    return mesh


def test_pillbox_meshes_natively():
    from cavsim2d.cavity import Pillbox
    assert_meshes_natively(Pillbox(1, [100, 100, 20, 0, 0], beampipe='none'))
    assert_meshes_natively(Pillbox(2, [100, 100, 20, 20, 30], beampipe='both'))


def test_pillbox_profile_native_matches_gmsh(project_dir):
    """The native-profile pillbox eigenmode matches the gmsh-path result."""
    from cavsim2d.cavity import Cavities, Pillbox
    cavs = Cavities(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    assert pb.profile() is not None          # else this silently tests gmsh
    cavs.add_cavity([pb], ['PB'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    pb.get_eigenmode_qois()
    # gmsh path historically gave ~1159.6 MHz (analytic TM010 for R=100mm+aperture)
    assert abs(pb.eigenmode_qois['freq [MHz]'] - 1159.6) < 2.0


TESLA = [42, 42, 12, 19, 35, 57.7, 103.353]


def test_elliptical_profile_exact_ellipse_arcs():
    """A symmetric single-cell elliptical cavity builds a native profile whose
    iris/equator arcs are exact conics, and meshes with its tags intact."""
    from cavsim2d.cavity import EllipticalCavity
    cav = EllipticalCavity(1, TESLA, TESLA, TESLA, beampipe='none')
    prof = cav.profile()
    assert prof is not None
    assert sum(s['kind'] == 'ellipse' for s in prof._segs) == 4  # 2 iris + 2 equator
    mesh = prof.mesh(maxh=0.02, order=3)
    assert set(mesh.GetBoundaries()) == {'AXI', 'PEC', 'PMC'}


def test_elliptical_profile_covers_multicell_and_asymmetric():
    """Multicell, asymmetric end cells and one-sided beampipes are all native now."""
    from cavsim2d.cavity import EllipticalCavity
    end = list(TESLA)
    end[5] = 60.0                                    # different half-cell length
    for n, mid, el, er, bp in [(2, TESLA, TESLA, TESLA, 'both'),
                               (3, TESLA, TESLA, TESLA, 'none'),
                               (1, TESLA, end, TESLA, 'both'),
                               (1, TESLA, TESLA, TESLA, 'left')]:
        cav = EllipticalCavity(n, mid, el, er, beampipe=bp)
        prof = cav.profile()
        assert prof is not None, (n, bp)
        # every cell contributes 4 exact ellipse arcs (2 iris + 2 equator)
        assert sum(s['kind'] == 'ellipse' for s in prof._segs) == 4 * n


def test_elliptical_profile_returns_none_when_degenerate():
    """No tangent solution -> profile() returns None so the gmsh writer reports it.

    Note this covers only sets the tangent solver *fails* on (fsolve does not
    converge). Physically silly but solvable sets — e.g. Req < Ri — still return a
    contour; the profile layer does not validate them.
    """
    from cavsim2d.cavity import EllipticalCavity
    bad = [200, 42, 12, 19, 35, 57.7, 103.353]       # equator semi-axis A far too large
    assert EllipticalCavity(1, bad, bad, bad, beampipe='none').profile() is None


def test_elliptical_meshes_natively():
    from cavsim2d.cavity import EllipticalCavity
    assert_meshes_natively(EllipticalCavity(1, TESLA, TESLA, TESLA, beampipe='none'))
    assert_meshes_natively(EllipticalCavity(1, TESLA, TESLA, TESLA, beampipe='both'))


def test_elliptical_profile_native_matches_gmsh(project_dir):
    """Native exact-ellipse TESLA 1-cell reproduces the gmsh-path eigenmode."""
    from cavsim2d.cavity import Cavities, EllipticalCavity
    cavs = Cavities(project_dir)
    cav = EllipticalCavity(1, TESLA, TESLA, TESLA, beampipe='none')
    assert cav.profile() is not None          # else this silently tests gmsh
    cavs.add_cavity([cav], ['TESLA'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    cav.get_eigenmode_qois()
    qois = cav.eigenmode_qois
    assert abs(qois['freq [MHz]'] - 1300.17) < 0.5
    assert abs(qois['R/Q [Ohm]'] - 113.47) < 0.5
    assert abs(qois['G [Ohm]'] - 271.0) < 1.0


def test_multipac_export_writes_contour(tmp_path):
    """The Multipac exporter is reachable from the public API and emits a contour."""
    from cavsim2d.cavity import EllipticalCavity
    cav = EllipticalCavity(1, TESLA, TESLA, TESLA, beampipe='both')
    out = tmp_path / 'geodata.n'
    cav.export_multipac(str(out), plot=False)
    rows = [ln.split() for ln in out.read_text().strip().split('\n')]
    assert len(rows) > 50
    assert all(len(r) >= 2 for r in rows)          # (z, r) pairs at minimum
    assert max(float(r[1]) for r in rows) > 0.0    # non-degenerate radius


def test_geometry_package_is_the_single_home():
    """Geometry lives under cavsim2d.geometry; the old scattered modules are gone."""
    import importlib
    for mod in ('cavsim2d.geometry.profile', 'cavsim2d.geometry.tangency',
                'cavsim2d.geometry.primitives', 'cavsim2d.geometry.plotting',
                'cavsim2d.geometry.writers.gmsh', 'cavsim2d.geometry.writers.multipac',
                'cavsim2d.geometry.writers.cst', 'cavsim2d.geometry.writers.abci'):
        assert importlib.import_module(mod) is not None
    for gone in ('cavsim2d.utils.geometry', 'cavsim2d.analysis.wakefield.geometry',
                 'cavsim2d.analysis.wakefield.abci_geometry'):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)


def test_shared_functions_still_reexports():
    """The compatibility shim keeps working for existing user scripts."""
    import cavsim2d.utils.shared_functions as sf
    for name in ('tangent_coords', 'lineTo', 'arcTo', 'write_cavity_geometry_cli',
                 'writeCavityForMultipac', 'write_cst_paramters', 'perturb_geometry'):
        assert callable(getattr(sf, name)), name


FLATTOP = [62.22, 66.13, 30.22, 23.11, 80, 93.5, 171.20, 20]
GUN_SHAPE = {'geometry': {
    'y1': 1.5e-2, 'R2': 3e-2, 'T2': 0.7853981633974483, 'L3': 24e-2,
    'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
    'T9': 0.13962634015954636, 'R10': 3e-2, 'T10': 0.6981317007977318,
    'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}


def test_flattop_meshes_natively():
    from cavsim2d.cavity import EllipticalCavityFlatTop
    for n, bp in ((1, 'both'), (1, 'none'), (2, 'both'), (3, 'left')):
        assert_meshes_natively(EllipticalCavityFlatTop(n, FLATTOP, FLATTOP, FLATTOP, beampipe=bp))


def test_flattop_with_zero_flat_equals_elliptical(project_dir):
    """The flat-top builder degenerates exactly to the plain elliptical contour.

    This is the flat top's only ground truth: its `.geo` writer emits no file at
    all, so there is no gmsh path to compare against.
    """
    from cavsim2d.cavity import Cavities, EllipticalCavity, EllipticalCavityFlatTop
    base = FLATTOP[:7]

    def freq(cav, tag):
        cavs = Cavities(os.path.join(project_dir, tag))
        cavs.add_cavity([cav], [tag])
        cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
        cav.get_eigenmode_qois()
        return cav.eigenmode_qois['freq [MHz]']

    f_ft = freq(EllipticalCavityFlatTop(1, base + [0], base + [0], base + [0], beampipe='both'), 'ft')
    f_el = freq(EllipticalCavity(1, base, base, base, beampipe='both'), 'el')
    assert abs(f_ft - f_el) < 1e-3


def test_flattop_frequency_falls_as_flat_lengthens(project_dir):
    from cavsim2d.cavity import Cavities, EllipticalCavityFlatTop
    base = FLATTOP[:7]
    freqs = []
    for l in (0, 20, 40):
        cav = EllipticalCavityFlatTop(1, base + [l], base + [l], base + [l], beampipe='both')
        cavs = Cavities(os.path.join(project_dir, f'l{l}'))
        cavs.add_cavity([cav], [f'l{l}'])
        cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
        cav.get_eigenmode_qois()
        freqs.append(cav.eigenmode_qois['freq [MHz]'])
    assert freqs[0] > freqs[1] > freqs[2]


def test_rfgun_meshes_natively_with_exact_circular_arcs():
    from cavsim2d.cavity import RFGun
    gun = RFGun(GUN_SHAPE)
    prof = gun.profile()
    assert prof is not None
    assert sum(s['kind'] == 'arc' for s in prof._segs) == 8      # 8 circular arcs
    assert prof.points[-1][1] == pytest.approx(0.0, abs=1e-12)   # closes on the axis
    assert_meshes_natively(gun)


def test_circular_waveguide_meshes_natively():
    """The waveguide has no PMC region — all three walls are PEC, per its .geo."""
    from cavsim2d.cavity import CircularWaveguide
    assert_meshes_natively(CircularWaveguide(100.0, 150.0), boundaries=('AXI', 'PEC'))


def test_circular_waveguide_matches_analytic_tm010(project_dir):
    from scipy.special import jn_zeros
    from cavsim2d.cavity import Cavities, CircularWaveguide
    R = 100.0
    cw = CircularWaveguide(R, 150.0)
    assert cw.profile() is not None          # else this silently tests gmsh
    cavs = Cavities(project_dir)
    cavs.add_cavity([cw], ['CW'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    cw.get_eigenmode_qois()
    analytic = jn_zeros(0, 1)[0] * 299792458.0 / (2 * np.pi * R * 1e-3) / 1e6
    assert abs(cw.eigenmode_qois['freq [MHz]'] - analytic) / analytic < 2e-3


SPLINE_GEOM = {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
               'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}


def test_spline_to_bezier_and_bspline_mesh_natively():
    from cavsim2d.cavity import SplineCavity
    for kind in ('BSpline', 'Bezier'):
        cav = SplineCavity({'geometry': dict(SPLINE_GEOM)}, kind=kind)
        assert_meshes_natively(cav)


def test_bspline_bezier_decomposition_is_exact():
    """netgen's BSplineCurve is unclamped, so we decompose into Bezier arcs.

    The decomposition must reproduce the clamped B-spline exactly, otherwise the
    native wall would silently differ from the gmsh one.
    """
    from scipy.special import comb
    poles = [(0.0, 0.035), (0.0, 0.070), (0.030, 0.103),
             (0.085, 0.103), (0.115, 0.070), (0.115, 0.035)]
    p = Profile().start(*poles[0])
    p.spline_to(poles[1:], 'PEC', kind='bspline', degree=3)
    seg = p._segs[0]

    segments = Profile._bspline_to_bezier(poles, 3)
    assert np.allclose(segments[0][0], poles[0])     # clamped: starts on the contour
    assert np.allclose(segments[-1][-1], poles[-1])  # and ends on it

    def bezier(pl, t):
        pl = np.asarray(pl)
        m = len(pl) - 1
        i = np.arange(m + 1)
        return (comb(m, i) * (t ** i) * ((1 - t) ** (m - i))) @ pl

    ref = np.array(Profile._spline_points(seg, p._pts, n=401))
    n = len(segments)
    got = np.array([bezier(segments[min(int(u * n), n - 1)], u * n - min(int(u * n), n - 1))
                    for u in np.linspace(0, 1, 401)])
    assert np.abs(got - ref).max() < 1e-12


def test_spline_kind_alias_and_unknown():
    """'Berzier' was the old default and matched no branch -> a wall-less .geo."""
    from cavsim2d.cavity import SplineCavity
    assert SplineCavity({'geometry': dict(SPLINE_GEOM)}, kind='Berzier').spline_kind() == 'bezier'
    assert SplineCavity({'geometry': dict(SPLINE_GEOM)}, kind='Bezier').spline_kind() == 'bezier'
    assert SplineCavity({'geometry': dict(SPLINE_GEOM)}, kind='BSpline').spline_kind() == 'bspline'
    bad = SplineCavity({'geometry': dict(SPLINE_GEOM)}, kind='nonsense')
    assert bad.spline_kind() is None
    assert bad.profile() is None                      # falls back; writer then raises


def test_spline_wall_length_matches_the_true_curve():
    """The native wall reproduces the analytic arc length of the spline.

    Covers the 2-cell B-spline, where repeating the control polygon puts a genuine
    stationary corner at the iris. gmsh rounds that corner off (its wall comes out
    ~0.5% short); the native path resolves it.
    """
    from ngsolve import Integrate, CF, BND
    from cavsim2d.cavity import SplineCavity
    cav = SplineCavity({'geometry': dict(SPLINE_GEOM), 'n_cells': 2}, kind='BSpline')
    prof = cav.profile()
    seg = [s for s in prof._segs if s['kind'] == 'spline'][0]
    pts = np.array(Profile._spline_points(seg, prof._pts, n=200001))
    true_length = np.sum(np.hypot(*np.diff(pts, axis=0).T))

    mesh = prof.mesh(maxh=0.01, order=3)
    wall = Integrate(CF(1), mesh, BND, definedon=mesh.Boundaries('PEC'))
    assert abs(wall - true_length) / true_length < 1e-5


def test_spline_cavity_eigenmode(project_dir):
    from cavsim2d.cavity import Cavities, SplineCavity
    cavs = Cavities(project_dir)
    cav = SplineCavity({'geometry': dict(SPLINE_GEOM)}, kind='Bezier')
    assert cav.profile() is not None          # else this silently tests gmsh
    cavs.add_cavity(cav, 'SC')
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    cav.get_eigenmode_qois()
    # gmsh path gives 1534.0399 MHz for this Bezier wall; native matches to 1e-4 %
    assert abs(cav.eigenmode_qois['freq [MHz]'] - 1534.04) < 0.5


def test_stationary_corner_raises_actionable_error():
    """netgen's high-order curving fails at a spline's stationary corner.

    The raw OCC exception is `Standard_OutOfRange`, which says nothing. Profile
    detects the corner and explains it instead.
    """
    from cavsim2d.cavity import SplineCavity
    cav = SplineCavity({'geometry': dict(SPLINE_GEOM), 'n_cells': 2}, kind='BSpline')
    prof = cav.profile()
    corners = prof._stationary_corners()
    assert len(corners) == 1
    assert corners[0] == pytest.approx((0.115, 0.0466667), abs=1e-6)

    with pytest.raises(RuntimeError, match='stationary corner'):
        prof.mesh(maxh=0.004, order=3)

    # smooth contours report no corners and still mesh at a fine order
    for kind, n in (('Bezier', 2), ('BSpline', 1)):
        smooth = SplineCavity({'geometry': dict(SPLINE_GEOM), 'n_cells': n}, kind=kind).profile()
        assert smooth._stationary_corners() == []
        assert smooth.mesh(maxh=0.004, order=4).ne > 0


def test_non_spline_profiles_have_no_stationary_corners():
    from cavsim2d.cavity import EllipticalCavity, Pillbox
    assert EllipticalCavity(2, TESLA, TESLA, TESLA, beampipe='both').profile()._stationary_corners() == []
    assert Pillbox(1, [100, 100, 20, 0, 0], beampipe='none').profile()._stationary_corners() == []


# --- per-half-cell (multicell) representation --------------------------------

def test_half_cells_uniform_reproduces_the_wrapper_contour():
    """The half-cell builder is behaviour-preserving for uniform mid-cells."""
    from cavsim2d.cavity import EllipticalCavity
    from cavsim2d.geometry.contours import elliptical_profile
    end = [40, 40, 10, 21, 37, 56.0, 103.353]
    for n, el, er, bp in ((1, TESLA, TESLA, 'both'), (2, TESLA, TESLA, 'both'),
                          (3, TESLA, TESLA, 'none'), (2, end, TESLA, 'left')):
        cav = EllipticalCavity(n, TESLA, el, er, beampipe=bp)
        assert cav.half_cells().shape == (2 * n, 7)
        wrapper = elliptical_profile([v * 1e-3 for v in TESLA], [v * 1e-3 for v in el],
                                     [v * 1e-3 for v in er], n, bp)
        assert np.allclose(np.array(wrapper.points), np.array(cav.profile().points), atol=1e-15)


def test_half_cells_enforces_equator_and_iris_continuity():
    """Req is shared within a cell, Ri across the iris between adjacent cells."""
    from cavsim2d.cavity import EllipticalCavity
    from cavsim2d.geometry.contours import continuity_violations
    cav = EllipticalCavity(3, TESLA, TESLA, TESLA, beampipe='both')
    assert continuity_violations(cav.half_cells()) == []

    bad_req = cav.half_cells()
    bad_req[1, 6] = 99.0
    with pytest.raises(ValueError, match='Req mismatch'):
        cav.set_half_cells(bad_req)

    bad_ri = cav.half_cells()
    bad_ri[2, 4] = 33.0                       # one side of an iris only
    with pytest.raises(ValueError, match='Ri mismatch'):
        cav.set_half_cells(bad_ri)

    with pytest.raises(ValueError, match='shape'):
        cav.set_half_cells(np.zeros((3, 7)))


def test_half_cells_let_each_cell_vary_independently():
    from cavsim2d.cavity import EllipticalCavity
    cav = EllipticalCavity(3, TESLA, TESLA, TESLA, beampipe='both')
    hc = cav.half_cells()
    hc[1, 4] = hc[2, 4] = 33.0                # iris cell1|cell2
    hc[3, 4] = hc[4, 4] = 33.0                # iris cell2|cell3
    hc[2, 6] = hc[3, 6] = 104.5               # cell2's own equator
    cav.set_half_cells(hc)

    prof = cav.profile()
    assert prof is not None
    assert sum(s['kind'] == 'ellipse' for s in prof._segs) == 12   # 4 per cell
    mesh = prof.mesh(maxh=0.02, order=3)
    assert set(mesh.GetBoundaries()) == {'AXI', 'PEC', 'PMC'}

    cav.set_half_cells(None)                  # revert to the parameter-derived geometry
    assert np.allclose(cav.half_cells()[:, 4], 35.0)


def test_non_elliptical_models_have_no_half_cells():
    from cavsim2d.cavity import Pillbox
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    with pytest.raises(NotImplementedError, match='half-cell'):
        pb.half_cells()


def test_independent_cells_change_the_eigenmode(project_dir):
    """A narrower middle iris must lower the frequency and raise R/Q."""
    from cavsim2d.cavity import Cavities, EllipticalCavity

    def freq_rq(tag, mutate=None):
        cav = EllipticalCavity(3, TESLA, TESLA, TESLA, beampipe='both')
        cavs = Cavities(os.path.join(project_dir, tag))
        cavs.add_cavity([cav], [tag])
        if mutate:
            hc = cav.half_cells()
            mutate(hc)
            cav.set_half_cells(hc)
        cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
        cav.get_eigenmode_qois()
        return cav.eigenmode_qois['freq [MHz]'], cav.eigenmode_qois['R/Q [Ohm]']

    def narrow_middle_irises(hc):
        hc[1, 4] = hc[2, 4] = 33.0
        hc[3, 4] = hc[4, 4] = 33.0

    f0, rq0 = freq_rq('uniform')
    f1, rq1 = freq_rq('percell', narrow_middle_irises)
    assert f1 < f0 - 1.0
    assert rq1 > rq0 + 1.0
