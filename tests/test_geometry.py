"""Unit tests for the unified geometry Profile blueprint + native netgen mesh."""
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


def test_pillbox_profile_native_matches_gmsh(project_dir):
    """The native-profile pillbox eigenmode matches the gmsh-path result."""
    from cavsim2d.cavity import Cavities, Pillbox
    cavs = Cavities(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
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


def test_elliptical_profile_falls_back_to_gmsh_when_unsupported():
    """Multicell / asymmetric geometries return None so the solver uses gmsh."""
    from cavsim2d.cavity import EllipticalCavity
    end = list(TESLA)
    end[5] = 60.0  # different half-cell length -> asymmetric
    assert EllipticalCavity(2, TESLA, TESLA, TESLA, beampipe='both').profile() is None
    assert EllipticalCavity(1, TESLA, end, TESLA, beampipe='both').profile() is None
    assert EllipticalCavity(1, TESLA, TESLA, TESLA, beampipe='left').profile() is None


def test_elliptical_profile_native_matches_gmsh(project_dir):
    """Native exact-ellipse TESLA 1-cell reproduces the gmsh-path eigenmode."""
    from cavsim2d.cavity import Cavities, EllipticalCavity
    cavs = Cavities(project_dir)
    cav = EllipticalCavity(1, TESLA, TESLA, TESLA, beampipe='none')
    cavs.add_cavity([cav], ['TESLA'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    cav.get_eigenmode_qois()
    qois = cav.eigenmode_qois
    assert abs(qois['freq [MHz]'] - 1300.17) < 0.5
    assert abs(qois['R/Q [Ohm]'] - 113.47) < 0.5
    assert abs(qois['G [Ohm]'] - 271.0) < 1.0
