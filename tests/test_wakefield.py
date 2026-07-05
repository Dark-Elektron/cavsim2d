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
