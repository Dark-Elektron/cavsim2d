"""API-surface guards: removed legacy methods stay removed, and methods that
were only guarded (not ported) raise a clear NotImplementedError."""
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d.cavity import Cavities, EllipticalCavity
from cavsim2d.cavity.base import Cavity


@pytest.mark.parametrize("name", ["run_abci", "run_multipacting", "check_uq_config"])
def test_dead_cavities_wrappers_removed(name):
    assert not hasattr(Cavities, name), f"{name} should have been deleted"


def test_sweep_raises_clear_error(project_dir):
    cavs = Cavities(project_dir)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['S'])
    with pytest.raises(NotImplementedError):
        cav.sweep({'A': [40, 45, 3]})
