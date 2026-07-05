"""Eigenmode smoke tests for the non-elliptical cavity types."""
import os

import numpy as np
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import requires_abci
from cavsim2d.cavity import Cavities, Pillbox, RFGun


def test_pillbox_eigenmode(project_dir):
    """A pillbox resonates near the analytic TM010 frequency."""
    cavs = Cavities(project_dir)
    # dims = [L, Req, Ri, S, L_bp] (mm); R=100 mm -> TM010 ~ 1.147 GHz
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    cavs.add_cavity([pb], ['PB'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    pb.get_eigenmode_qois()
    assert 1050 < pb.eigenmode_qois['freq [MHz]'] < 1250


@requires_abci
def test_pillbox_wakefield(project_dir):
    """Pillbox wakefield runs and writes both ABCI polarisations."""
    cavs = Cavities(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    cavs.add_cavity([pb], ['PB'])
    cavs.run_wakefield({'processes': 1, 'rerun': True,
                        'wakelength': 3, 'bunch_length': 25})
    wf = os.path.join(pb.self_dir, 'wakefield')
    assert os.path.exists(os.path.join(wf, 'longitudinal', 'cavity.top'))
    assert os.path.exists(os.path.join(wf, 'transversal', 'cavity.top'))


def test_pillbox_inherits_base_run_eigenmode():
    """Pillbox no longer carries its own legacy run_* overrides."""
    for name in ('run_eigenmode', 'run_wakefield', 'run_tune'):
        assert name not in Pillbox.__dict__, f"Pillbox should inherit {name}"


def test_rfgun_eigenmode(project_dir):
    """A VHF-gun geometry solves to a sensible (~200 MHz) frequency."""
    shape = {'geometry': {
        'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
        'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
        'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
        'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}
    cavs = Cavities(project_dir)
    gun = RFGun(shape)
    cavs.add_cavity(gun, 'gun')
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    gun.get_eigenmode_qois()
    assert 150 < gun.eigenmode_qois['freq [MHz]'] < 300


def test_rfgun_qois_normalised_sensibly(project_dir):
    """RF-gun Eacc-normalised QOIs use the on-axis field extent (no 'L_m'),
    giving an O(1) Epk/Eacc rather than the bogus ~0.09 from L=1 mm."""
    shape = {'geometry': {
        'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
        'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
        'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
        'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}
    cavs = Cavities(project_dir)
    gun = RFGun(shape)
    cavs.add_cavity(gun, 'gun')
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    gun.get_eigenmode_qois()
    assert 1 < gun.eigenmode_qois['Epk/Eacc []'] < 50


def test_per_cavity_run_eigenmode_delegates(project_dir):
    """cav.run_eigenmode() (single-cavity) works via the modern pipeline."""
    cavs = Cavities(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    cavs.add_cavity([pb], ['PB'])
    assert pb.run_eigenmode() is True
    assert pb.freq > 0
