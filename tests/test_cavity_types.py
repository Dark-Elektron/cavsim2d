"""Eigenmode smoke tests for the non-elliptical cavity types."""
import os

import numpy as np
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import requires_abci
from cavsim2d import Study, Pillbox, RFGun, EllipticalCavity


def test_pillbox_eigenmode(project_dir):
    """A pillbox resonates near the analytic TM010 frequency."""
    cavs = Study(project_dir)
    # dims = [L, Req, Ri, S, L_bp] (mm); R=100 mm -> TM010 ~ 1.147 GHz
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    cavs.add_cavity([pb], ['PB'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    pb.get_eigenmode_qois()
    assert 1050 < pb.eigenmode_qois['freq [MHz]'] < 1250


@requires_abci
def test_pillbox_wakefield(project_dir):
    """Pillbox wakefield runs and writes both ABCI polarisations.

    ABCI needs beam pipes at both ends (>= 5 mesh lengths), so the pillbox needs a
    non-zero L_bp. This test used to run with beampipe='none' and L_bp=0, which
    ABCI silently refused: it left 0-byte cavity.top files, and the old
    `os.path.exists` assertions passed on them regardless.
    """
    cavs = Study(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 50], beampipe='both')
    cavs.add_cavity([pb], ['PB'])
    cavs.run_wakefield({'processes': 1, 'rerun': True,
                        'wakelength': 3, 'bunch_length': 25})
    wf = os.path.join(pb.self_dir, 'wakefield')
    for pol in ('longitudinal', 'transversal'):
        top = os.path.join(wf, pol, 'cavity.top')
        assert os.path.getsize(top) > 0, f'{pol}: ABCI produced no wake data'


def test_pillbox_tuning(project_dir):
    """Tune a pillbox's equator radius to a target frequency; the tuned
    cavity is reachable via cav.tuned and re-solves to the target."""
    cavs = Study(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    cavs.add_cavity([pb], ['PB'])
    cavs.run_tune({'freqs': 1100, 'cell_type': {'mid-cell': 'Req'},
                   'processes': 1, 'rerun': True,
                   'eigenmode_config': {'processes': 1, 'boundary_conditions': 'mm'}})
    res = pb.tune.qois['mid-cell']
    assert abs(res['FREQ'] - 1100) < 1.0
    assert res['parameters']['Req'] != 100     # Req moved
    tuned = pb.tuned
    assert tuned is not None and isinstance(tuned, Pillbox)


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
    cavs = Study(project_dir)
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
    cavs = Study(project_dir)
    gun = RFGun(shape)
    cavs.add_cavity(gun, 'gun')
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    gun.get_eigenmode_qois()
    assert 1 < gun.eigenmode_qois['Epk/Eacc []'] < 50


def test_parallel_eigenmode_processes_gt_1(project_dir):
    """processes>1 spawns worker processes (Windows uses 'spawn'); Cavity
    objects must pickle and both results must be written (P2-8)."""
    import os
    cavs = Study(project_dir)
    mid = [62.22, 66.13, 30.22, 23.11, 80, 93.5, 171.20]
    mid2 = [62.22, 66.13, 30.22, 23.11, 78, 93.5, 171.20]
    c1 = EllipticalCavity(1, mid, mid, mid, beampipe='both')
    c2 = EllipticalCavity(1, mid2, mid2, mid2, beampipe='both')
    cavs.add_cavity([c1, c2], ['A', 'B'])
    cavs.run_eigenmode({'processes': 2, 'rerun': True, 'boundary_conditions': 'mm'})
    for name in ('A', 'B'):
        qp = os.path.join(project_dir, 'cavities', name,
                          'eigenmode', 'monopole', 'qois.json')
        assert os.path.exists(qp), f"{name} results missing from processes=2 run"


def test_per_cavity_run_eigenmode_delegates(project_dir):
    """cav.run_eigenmode() (single-cavity) works via the modern pipeline."""
    cavs = Study(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    cavs.add_cavity([pb], ['PB'])
    assert pb.run_eigenmode() is True
    assert pb.freq > 0
