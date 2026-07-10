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


@requires_abci
def test_abci_abort_raises_instead_of_silently_producing_nothing(project_dir):
    """ABCI exits 0 even when it refuses to run, leaving empty .pot/.top files.

    Here beampipe='both' with L_bp=0 gives zero-length pipes, which ABCI rejects.
    The run used to report success and yield no wake data.
    """
    from cavsim2d.cavity import Cavities, Pillbox
    cavs = Cavities(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='both')    # L_bp = 0
    cavs.add_cavity([pb], ['PB'])
    with pytest.raises(RuntimeError, match='ABCI refused to run'):
        cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 10})
