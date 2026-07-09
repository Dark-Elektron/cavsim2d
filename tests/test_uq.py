"""UQ tests: simplecell eigenmode UQ produces statistics; the unsupported
multicell UQ path raises a clear error rather than crashing in a worker."""
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d.cavity import Cavities, EllipticalCavity


def _cavs(project_dir):
    cavs = Cavities(project_dir)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['UQ'])
    return cavs, cav


def test_simplecell_uq_produces_statistics(project_dir):
    cavs, cav = _cavs(project_dir)
    cavs.run_eigenmode({
        'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
        'uq_config': {
            'variables': ['A', 'B'],
            'objectives': ['R/Q [Ohm]', 'freq [MHz]'],
            'delta': [0.05, 0.05],
            'processes': 1,
            'distribution': 'gaussian',
            'method': ['Quadrature', 'Stroud3'],
            'cell_type': 'mid-cell',
            'cell_complexity': 'simplecell',
        },
    })
    res = cavs.uq_fm_results['UQ']
    assert res['freq [MHz]']['expe'][0] > 0
    assert 'stdDev' in res['R/Q [Ohm]']


def test_rerun_clears_stale_uq_artefacts(project_dir):
    """A monopole rerun must drop the stale UQ perturbation folder so its
    results can't outlive the eigenmode results they were derived from (P2-6)."""
    import os
    cavs, cav = _cavs(project_dir)
    cavs.run_eigenmode({
        'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
        'uq_config': {'variables': ['A'], 'objectives': ['R/Q [Ohm]'],
                      'delta': [0.05], 'processes': 1,
                      'method': ['Quadrature', 'Stroud3'],
                      'cell_type': 'mid-cell', 'cell_complexity': 'simplecell'},
    })
    uq_dir = os.path.join(cav.self_dir, 'uq')
    assert os.path.exists(uq_dir) and os.listdir(uq_dir)   # UQ produced artefacts

    # rerun the monopole without UQ — the stale uq/ folder must be gone
    cavs2 = Cavities(project_dir)
    cav2 = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs2.add_cavity([cav2], ['UQ'])
    cavs2.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    assert not os.path.exists(uq_dir)


def test_multicell_uq_raises_clear_error(project_dir):
    cavs, cav = _cavs(project_dir)
    with pytest.raises(NotImplementedError, match="Multicell UQ"):
        cavs.run_eigenmode({
            'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
            'uq_config': {
                'variables': ['A'],
                'objectives': ['R/Q [Ohm]'],
                'delta': [0.05],
                'processes': 1,
                'method': ['Quadrature', 'Stroud3'],
                'cell_type': 'mid-cell',
                'cell_complexity': 'multicell',
            },
        })
