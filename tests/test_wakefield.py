"""Wakefield smoke test (Windows + bundled ABCI.exe only): ABCI runs and
produces longitudinal + transversal output that the reader can load."""
import os

import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL, requires_abci
from cavsim2d.cavity import Cavities, EllipticalCavity


@requires_abci
def test_wakefield_runs_and_writes_output(project_dir):
    cavs = Cavities(project_dir)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['WF'])

    cavs.run_wakefield({'processes': 1, 'rerun': True,
                        'wakelength': 5, 'bunch_length': 25})

    wf = os.path.join(cav.self_dir, 'wakefield')
    assert os.path.exists(os.path.join(wf, 'longitudinal', 'cavity.top'))
    assert os.path.exists(os.path.join(wf, 'transversal', 'cavity.top'))

    # impedance data loads and plots without error
    cav.get_abci_data()
    ax = cav.plot('zl')
    assert ax is not None


@requires_abci
def test_pillbox_wakefield_end_to_end(project_dir):
    """Non-elliptical geometries run through ABCI too (P2-4)."""
    from cavsim2d.cavity import Cavities, Pillbox
    cavs = Cavities(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 50], beampipe='both')   # L_bp = 50 mm
    cavs.add_cavity([pb], ['PB'])
    cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 10})
    top = os.path.join(pb.self_dir, 'wakefield', 'longitudinal', 'cavity.top')
    assert os.path.getsize(top) > 0
    pb.get_abci_data()
    assert 'Long' in pb.abci_data


def test_abci_abort_raises_instead_of_silently_producing_nothing(tmp_path):
    """ABCI exits 0 even when it refuses to run, leaving empty .pot/.top files.

    Beam pipes are now added automatically, so the old trigger (a pillbox with
    L_bp = 0) no longer aborts. The guard itself is what matters, so drive it
    directly with a log ABCI would have written.
    """
    from cavsim2d.solvers.ABCI.abci import _raise_if_abci_aborted

    (tmp_path / 'cavity.out').write_text(
        ' NUMBER OF MESH LINES IN R      : NR   =         84\n'
        '0*** STOP *** THE BEAM PIPES AT BOTH ENDS ARE TOO SHORT.\n'
        ' THEY MUST HAVE AT LEAST 5 MESH LENGTH.\n')
    with pytest.raises(RuntimeError, match='ABCI refused to run'):
        _raise_if_abci_aborted(str(tmp_path))

    # a clean log passes through
    (tmp_path / 'cavity.out').write_text(' NORMAL COMPLETION\n')
    _raise_if_abci_aborted(str(tmp_path))


@requires_abci
def test_pillbox_without_beampipe_now_runs(project_dir):
    """L_bp = 0 used to make ABCI refuse; the pipes are added for it now."""
    from cavsim2d.cavity import Cavities, Pillbox
    cavs = Cavities(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    cavs.add_cavity([pb], ['PB'])
    cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 3})
    top = os.path.join(pb.self_dir, 'wakefield', 'longitudinal', 'cavity.top')
    assert os.path.getsize(top) > 0


# --- seam 2: the wakefield contour comes from Profile, not the .geo text -----

def test_contour_points_are_accurate():
    """Curved segments are sampled to the requested spacing."""
    import numpy as np
    from scipy.special import ellipe
    from cavsim2d.geometry import Profile

    a, b = 0.05, 0.03
    p = Profile().start(a, 0.0)
    p.ellipse_arc_to(0.0, b, center=(0.0, 0.0), semi_z=a, semi_r=b, boundary='PEC')
    p.line_to(0.0, 0.0, 'AXI')
    p.close('PMC')
    pts = np.asarray(p.contour_points(1e-4, skip=('AXI', 'PMC')))
    sampled = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    exact = a * ellipe(1 - (b / a) ** 2)          # quarter-ellipse perimeter
    assert abs(sampled - exact) / exact < 1e-5


def test_beampipes_are_added_only_where_missing():
    """ABCI needs a pipe at each end; cavities that have one are left alone."""
    import numpy as np
    from cavsim2d.cavity import EllipticalCavity, Pillbox, RFGun
    from cavsim2d.geometry.beampipes import abci_contour, beampipe_lengths

    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    ddz = 1.25e-3
    min_pipe = 6 * ddz

    prof = EllipticalCavity(1, tesla, tesla, tesla, beampipe='both').profile()
    _, al, ar = abci_contour(prof, ddz, min_pipe)
    assert (al, ar) == (0.0, 0.0)                 # already has pipes -> untouched

    cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='none')
    prof = cav.profile()
    raw = prof.contour_points(ddz, skip=('AXI',))
    axial = max(z for z, _ in raw) - min(z for z, _ in raw)
    pts, al, ar = abci_contour(prof, ddz, min_pipe)
    assert al == pytest.approx(3 * axial)         # default: 3x the axial length
    assert ar == pytest.approx(3 * axial)
    left, right = beampipe_lengths(pts)
    assert left >= min_pipe and right >= min_pipe
    assert abs(pts[0][1]) < 1e-12 and abs(pts[-1][1]) < 1e-12   # still ends on the axis

    # a gun already has a downstream drift tube: only the cathode side gets a pipe
    gun = RFGun({'geometry': {'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
                              'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
                              'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
                              'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}})
    _, al, ar = abci_contour(gun.profile(), ddz, min_pipe)
    assert al > 0 and ar == 0.0

    _, al, ar = abci_contour(Pillbox(1, [100, 100, 20, 0, 0], beampipe='none').profile(),
                             ddz, min_pipe)
    assert al > 0 and ar > 0


def test_abci_shape_keeps_arcs():
    """ABCI emits NaN wake potentials when an arc meeting a beam pipe tangentially
    (an elliptical iris) is handed to it as a dense polyline."""
    from cavsim2d.cavity import EllipticalCavity, Pillbox
    from cavsim2d.geometry.beampipes import abci_shape

    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    items, _, _ = abci_shape(EllipticalCavity(1, tesla, tesla, tesla, beampipe='both').profile(),
                             1.25e-3, 7.5e-3)
    assert [it[0] for it in items].count('arc') == 4     # 2 iris + 2 equator

    items, _, _ = abci_shape(Pillbox(1, [100, 100, 20, 0, 50], beampipe='both').profile(),
                             1.25e-3, 7.5e-3)
    assert all(it[0] == 'point' for it in items)         # all straight lines


@requires_abci
def test_wakefield_matches_the_legacy_geo_path(project_dir):
    """The Profile-based deck reproduces the old .geo-parsed deck exactly."""
    from cavsim2d.cavity import Cavities, EllipticalCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]

    def k_loss(legacy):
        cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='both')
        cavs = Cavities(os.path.join(project_dir, 'legacy' if legacy else 'profile'))
        cavs.add_cavity([cav], ['C'])
        original = EllipticalCavity.profile
        if legacy:
            EllipticalCavity.profile = lambda self: None
        try:
            cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 3})
            cav.get_abci_data()
            top = os.path.join(cav.self_dir, 'wakefield', 'longitudinal', 'cavity.top')
            with open(top, errors='replace') as fh:
                assert 'NaN' not in fh.read()
            return cav.abci_data['Long'].loss_factor['Longitudinal']
        finally:
            EllipticalCavity.profile = original

    assert k_loss(False) == pytest.approx(k_loss(True), abs=1e-9)


@requires_abci
def test_wakefield_runs_for_every_cavity_type(project_dir):
    """Flat-top, spline and gun had no working wakefield: the flat top writes no
    .geo at all, and the spline wall is a curve the old regex could not see."""
    import numpy as np
    from cavsim2d.cavity import (Cavities, EllipticalCavity, Pillbox, RFGun,
                                 SplineCavity, EllipticalCavityFlatTop)
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    ft = tesla + [20]
    gun = {'geometry': {'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
                        'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
                        'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
                        'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}
    geom = {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
            'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}

    cases = [('PB', Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')),
             ('ELN', EllipticalCavity(1, tesla, tesla, tesla, beampipe='none')),
             ('FT', EllipticalCavityFlatTop(1, ft, ft, ft, beampipe='both')),
             ('SC', SplineCavity({'geometry': dict(geom)}, kind='Bezier')),
             ('GUN', RFGun(gun))]
    for name, cav in cases:
        cavs = Cavities(os.path.join(project_dir, name))
        cavs.add_cavity([cav], [name])
        cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 3})
        top = os.path.join(cav.self_dir, 'wakefield', 'longitudinal', 'cavity.top')
        assert os.path.getsize(top) > 0
        with open(top, errors='replace') as fh:
            assert 'NaN' not in fh.read(), name + ': ABCI produced NaN wake potentials'
        cav.get_abci_data()
        assert cav.abci_data['Long'].loss_factor['Longitudinal'] != 0


@requires_abci
def test_added_beampipes_do_not_change_the_cavity_wake(project_dir):
    """The same elliptical cavity with beampipe='both' and with beampipe='none'
    (pipes added automatically) must give the same loss factor."""
    from cavsim2d.cavity import Cavities, EllipticalCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]

    def k(bp, tag):
        cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe=bp)
        cavs = Cavities(os.path.join(project_dir, tag))
        cavs.add_cavity([cav], [tag])
        cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 3})
        cav.get_abci_data()
        return cav.abci_data['Long'].loss_factor['Longitudinal']

    assert k('both', 'both') == pytest.approx(k('none', 'none'), rel=2e-3)
