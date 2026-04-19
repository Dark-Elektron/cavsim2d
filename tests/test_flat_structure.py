"""Test: Verify flat directory structure — no Cavities/ subfolder.

This test does NOT require ABCI.exe or NGSolve.  It mocks the ngsolve
import and only tests directory structure creation.
"""
import os
import shutil
import sys
import types

# ---- Mock ngsolve before any cavsim2d import ----
class _MockModule(types.ModuleType):
    """A module mock that returns a dummy for any attribute."""
    __all__ = []  # supports 'from X import *'
    def __getattr__(self, name):
        import math
        if name == 'pi':
            return math.pi
        return lambda *a, **kw: None

_ng = _MockModule('ngsolve')
sys.modules['ngsolve'] = _ng

for submod in ['ngsolve.comp', 'ngsolve.fem', 'ngsolve.la', 'ngsolve.solve',
               'ngsolve.meshes', 'ngsolve.webgui', 'ngsolve.internal',
               'ngsolve.krylovsolver']:
    sys.modules[submod] = _MockModule(submod)

# Also mock netgen
sys.modules['netgen'] = _MockModule('netgen')
for submod in ['netgen.occ', 'netgen.meshing', 'netgen.geom2d',
               'netgen.csg', 'netgen.stl', 'netgen.webgui']:
    sys.modules[submod] = _MockModule(submod)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cavsim2d.cavity import Cavities, EllipticalCavity


# ----- helpers -----
def _make_tmp():
    d = os.path.join(os.path.dirname(__file__), '_tmp_flat_test')
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def _cleanup(d):
    if os.path.exists(d):
        shutil.rmtree(d)


# ----- geometry constants -----
MIDCELL   = [62.22, 66.13, 30.22, 23.11, 72, 93.5, 171.20]
ENDCELL_L = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]
ENDCELL_R = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]


def test_project_init_flat():
    """Cavities project init creates flat folder (no Cavities/ subdir)"""
    tmp = _make_tmp()
    try:
        cavs = Cavities(tmp, name='myproj')
        project_dir = cavs.projectDir

        assert os.path.isdir(project_dir), f"Project dir not created: {project_dir}"
        contents = os.listdir(project_dir)
        for bad in ['Cavities', 'OperatingPoints', 'PostData', 'Reference']:
            assert bad not in contents, f"'{bad}' subfolder should NOT exist, got: {contents}"
        print("PASS  test_project_init_flat")
    finally:
        _cleanup(tmp)


def test_cavity_create_flat():
    """EllipticalCavity.create() puts geometry directly in project/<name>/"""
    tmp = _make_tmp()
    try:
        cavs = Cavities(tmp, name='proj2')
        cav = EllipticalCavity(2, MIDCELL, ENDCELL_L, ENDCELL_R, beampipe='both')
        cavs.add_cavity([cav], ['TESLA'])

        assert cav.self_dir is not None, "self_dir was not set"
        assert cav.self_dir.endswith('TESLA'), \
            f"self_dir should end with cavity name, got: {cav.self_dir}"
        assert 'Cavities' not in cav.self_dir, \
            f"self_dir should NOT contain 'Cavities', got: {cav.self_dir}"

        geo_dir = os.path.join(cav.self_dir, 'geometry')
        assert os.path.isdir(geo_dir), f"geometry dir not created: {geo_dir}"

        geo_file = os.path.join(geo_dir, 'geodata.geo')
        assert os.path.isfile(geo_file), f"geodata.geo not found: {geo_file}"

        print("PASS  test_cavity_create_flat")
    finally:
        _cleanup(tmp)


def test_solver_dir_properties():
    """eigenmode_dir and wakefield_dir use flat paths"""
    tmp = _make_tmp()
    try:
        cavs = Cavities(tmp, name='proj3')
        cav = EllipticalCavity(2, MIDCELL, ENDCELL_L, ENDCELL_R, beampipe='both')
        cavs.add_cavity([cav], ['ICHIRO'])

        eigen_dir = cav.eigenmode_dir
        wake_dir = cav.wakefield_dir

        assert eigen_dir is not None
        assert wake_dir is not None
        assert eigen_dir.endswith(os.path.join('ICHIRO', 'eigenmode')), \
            f"eigenmode_dir wrong: {eigen_dir}"
        assert wake_dir.endswith(os.path.join('ICHIRO', 'wakefield')), \
            f"wakefield_dir wrong: {wake_dir}"
        assert 'Cavities' not in eigen_dir, \
            f"eigenmode_dir contains 'Cavities': {eigen_dir}"
        assert 'Cavities' not in wake_dir, \
            f"wakefield_dir contains 'Cavities': {wake_dir}"

        print("PASS  test_solver_dir_properties")
    finally:
        _cleanup(tmp)


def test_multiple_cavities_flat():
    """Multiple cavities go into the same flat project folder"""
    tmp = _make_tmp()
    try:
        cavs = Cavities(tmp, name='proj4')
        cav1 = EllipticalCavity(2, MIDCELL, ENDCELL_L, ENDCELL_R, beampipe='both')
        cav2 = EllipticalCavity(1, MIDCELL, ENDCELL_L, ENDCELL_R, beampipe='both')
        cavs.add_cavity([cav1, cav2], ['CAV_A', 'CAV_B'])

        project_contents = os.listdir(cavs.projectDir)
        assert 'CAV_A' in project_contents, f"CAV_A missing: {project_contents}"
        assert 'CAV_B' in project_contents, f"CAV_B missing: {project_contents}"
        assert 'Cavities' not in project_contents, \
            f"'Cavities' should NOT exist: {project_contents}"

        for name in ['CAV_A', 'CAV_B']:
            geo = os.path.join(cavs.projectDir, name, 'geometry', 'geodata.geo')
            assert os.path.isfile(geo), f"Missing geometry for {name}: {geo}"

        print("PASS  test_multiple_cavities_flat")
    finally:
        _cleanup(tmp)


def test_no_cavities_folder_anywhere():
    """No Cavities/ folder anywhere in the tree"""
    tmp = _make_tmp()
    try:
        cavs = Cavities(tmp, name='proj5')
        cav = EllipticalCavity(2, MIDCELL, ENDCELL_L, ENDCELL_R, beampipe='both')
        cavs.add_cavity([cav], ['DEEP_CHECK'])

        for root, dirs, files in os.walk(cavs.projectDir):
            assert 'Cavities' not in dirs, f"Found 'Cavities' subfolder at {root}"

        print("PASS  test_no_cavities_folder_anywhere")
    finally:
        _cleanup(tmp)


if __name__ == '__main__':
    test_project_init_flat()
    test_cavity_create_flat()
    test_solver_dir_properties()
    test_multiple_cavities_flat()
    test_no_cavities_folder_anywhere()
    print("\n=== ALL TESTS PASSED ===")
