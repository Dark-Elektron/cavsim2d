"""A single cavity can be analysed on its own, without a Study/project; the run
provisions a workspace in the CWD, and save()/load() relocate or reopen it."""
import os

import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d.cavity import EllipticalCavity


def test_standalone_eigenmode_and_save_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                  # default workspace lands here
    # no Cavities / add_cavity — a bare cavity is analysed directly
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both', name='solo')
    cav.eigenmode.run({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})

    assert os.path.basename(cav.self_dir) == 'solo'      # provisioned ./solo/
    assert os.path.exists(os.path.join(cav.self_dir, 'eigenmode', 'monopole', 'qois.json'))
    freq = cav.eigenmode.qois['freq [MHz]']
    assert 700 < freq < 900

    # save() copies the whole workspace to a chosen folder and continues there
    out = tmp_path / 'saved' / 'solo'
    cav.save(str(out))
    assert os.path.abspath(cav.self_dir) == os.path.abspath(out)
    assert (out / 'eigenmode' / 'monopole' / 'qois.json').exists()

    # load() reopens a saved workspace in a fresh object
    reopened = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both').load(str(out))
    assert reopened.eigenmode.qois['freq [MHz]'] == pytest.approx(freq)


def test_save_without_overwrite_is_guarded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both', name='c1')
    cav.set_workspace(str(tmp_path / 'c1'))       # provision without a full run
    (tmp_path / 'dest').mkdir()
    cav.save(str(tmp_path / 'dest'))              # exists, no overwrite -> no-op
    assert os.path.basename(cav.self_dir) == 'c1'  # workspace unchanged
    cav.save(str(tmp_path / 'dest'), overwrite=True)
    assert os.path.abspath(cav.self_dir) == os.path.abspath(tmp_path / 'dest')


def test_load_missing_folder_raises():
    with pytest.raises(FileNotFoundError):
        EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL).load('does/not/exist')
