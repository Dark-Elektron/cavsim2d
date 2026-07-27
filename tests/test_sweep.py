"""Parameter sweep + persistence.

``Study.sweep`` builds one cavity per parameter combination (tensor) or per
zipped tuple (hadamard), each with its own folder and simulations; ``Study.load``
reconstructs a saved project so cached results reload without re-solving.
"""
import os

import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from cavsim2d import Study, Pillbox
from cavsim2d.models.base import Cavity
from cavsim2d.geometry import Profile


def _pillbox():
    return Pillbox(1, [100, 100, 20, 0, 50], beampipe='both')


# --- sweep structure (no solve) ---------------------------------------------

def test_sweep_tensor_is_the_full_grid(project_dir):
    sw = Study(project_dir).sweep(_pillbox(), {'Req': [95, 100, 105], 'Ri': [18, 20]},
                                  mode='tensor')
    assert len(sw.cavities_list) == 6
    assert list(sw.sweep_table.columns) == ['Req', 'Ri']
    assert len(sw.sweep_table) == 6
    # every (Req, Ri) combination is present exactly once
    combos = {tuple(r) for r in sw.sweep_table[['Req', 'Ri']].to_numpy()}
    assert combos == {(95, 18), (95, 20), (100, 18), (100, 20), (105, 18), (105, 20)}


def test_sweep_hadamard_is_elementwise(project_dir):
    sw = Study(project_dir).sweep(_pillbox(), {'Req': [98, 102], 'Ri': [19, 21]},
                                  mode='hadamard')
    assert len(sw.cavities_list) == 2
    combos = {tuple(r) for r in sw.sweep_table[['Req', 'Ri']].to_numpy()}
    assert combos == {(98, 19), (102, 21)}


def test_sweep_hadamard_unequal_lengths_raise(project_dir):
    with pytest.raises(ValueError, match='same length'):
        Study(project_dir).sweep(_pillbox(), {'Req': [98, 102, 106], 'Ri': [19, 21]},
                                 mode='hadamard')


def test_sweep_rejects_unknown_variable(project_dir):
    with pytest.raises(ValueError):
        Study(project_dir).sweep(_pillbox(), {'nope': [1, 2]}, mode='tensor')


def test_sweep_bad_mode_raises(project_dir):
    with pytest.raises(ValueError, match='tensor'):
        Study(project_dir).sweep(_pillbox(), {'Req': [95, 100]}, mode='outer')


# --- sweep results + persistence (one solve) --------------------------------

def test_sweep_results_and_reload(project_dir):
    """The swept cavities solve, ``results()`` joins the swept values with the
    QOIs, and ``Study.load`` reloads every cavity's cached result with no
    re-simulation (and reconstructs the right model type)."""
    sw = Study(project_dir).sweep(_pillbox(), {'Req': [98, 102]}, mode='tensor')
    sw.run_eigenmode({'polarisation': 'monopole', 'mesh_config': {'h': 14, 'p': 2}})

    res = sw.results('eigenmode')
    assert 'Req' in res.columns and 'freq [MHz]' in res.columns
    assert res['freq [MHz]'].notna().all()
    # larger equator radius -> lower frequency
    lo = res.loc[res['Req'] == 98, 'freq [MHz]'].iloc[0]
    hi = res.loc[res['Req'] == 102, 'freq [MHz]'].iloc[0]
    assert lo > hi

    reloaded = Study.load(sw.projectDir)
    assert len(reloaded.cavities_list) == len(sw.cavities_list)
    assert all(type(c).__name__ == 'Pillbox' for c in reloaded.cavities_list)
    for c in reloaded.cavities_list:
        orig = sw.cavities_dict[c.name].eigenmode.qois['freq [MHz]']
        assert c.eigenmode.qois['freq [MHz]'] == pytest.approx(orig)


def test_sweep_works_on_a_native_only_custom_model(project_dir):
    """A custom model with only ``profile()`` + ``rebuild()`` (no
    ``write_geometry``) sweeps — ``spawn`` writes a ``.geo`` only when a writer
    exists, else relies on ``profile()``."""
    class _Cone(Cavity):
        def __init__(self, dims, name='cone'):
            super().__init__(n_cells=1, beampipe='none', name=name)
            L, Req, Ri = dims
            self.n_cells, self.kind, self.beampipe = 1, 'cone', 'none'
            self.parameters = {'L': float(L), 'Req': float(Req), 'Ri': float(Ri)}
            self.shape = {'IC': [L, Req, Ri], 'BP': 'none'}

        def profile(self):
            L, Req, Ri = (self.parameters[k] * 1e-3 for k in ('L', 'Req', 'Ri'))
            p = Profile('cone')
            p.start(-L / 2, 0.0)
            p.line_to(-L / 2, Ri, 'PMC')
            p.line_to(0.0, Req, 'PEC')
            p.line_to(L / 2, Ri, 'PEC')
            p.line_to(L / 2, 0.0, 'PMC')
            p.close('AXI')
            return p

        def create(self, n_cells=None, beampipe=None, mode=None):
            if self.projectDir:
                self.self_dir = os.path.join(self.projectDir, self.name)
                os.makedirs(os.path.join(self.self_dir, 'geometry'), exist_ok=True)
                self.uq_dir = os.path.join(self.self_dir, 'uq')
                self.geo_filepath = None
                self._write_geometry_snapshot()

        def rebuild(self, parameters, beampipe=None):
            return _Cone([parameters['L'], parameters['Req'], parameters['Ri']],
                         name=self.name)

    assert not hasattr(_Cone, 'write_geometry')
    sw = Study(project_dir).sweep(_Cone([100, 100, 20]), {'Req': [90, 100]}, mode='tensor')
    assert len(sw.cavities_list) == 2
    assert all(c.geo_filepath is None and c.profile() is not None
               for c in sw.cavities_list)
