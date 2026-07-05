"""Tuning regression: an elliptical mid-cell tunes its equator radius to hit a
target frequency, and the tuned cavity is reachable via ``cav.tuned``."""
import numpy as np
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d.cavity import Cavities, EllipticalCavity


def test_tune_hits_target_frequency(project_dir):
    target = 801.58
    midcell = np.array(MIDCELL + [0])  # trailing alpha slot
    cav = EllipticalCavity(1, midcell, midcell, midcell, beampipe='none')
    cavs = Cavities(project_dir)
    cavs.add_cavity(cav, 'tune')

    tune_config = {
        'freqs': target,
        'cell_type': {'mid-cell': 'Req'},
        'processes': 1,
        'rerun': True,
        'eigenmode_config': {'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'},
    }
    cavs.run_tune(tune_config)

    tuned = cav.tuned
    assert tuned is not None, "cav.tuned should be populated after tuning"
    res = cav.tune.qois['mid-cell']
    assert abs(res['FREQ'] - target) < 0.5, res['FREQ']
    # tuning moved the equator radius away from its starting value
    assert res['parameters']['Req_m'] != MIDCELL[6]
