"""UQ tests: simplecell UQ perturbs the mid/end-cell parameter groups; multicell UQ
makes every half-cell an independent random variable, honouring the equator/iris
continuity constraints. Multicell wakefield UQ is still unimplemented."""
import os
import json
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d import Cavities, EllipticalCavity


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
            'objectives': ['monopole:R/Q [Ohm]', 'monopole:freq [MHz]'],
            'delta': [0.05, 0.05],
            'processes': 1,
            'distribution': 'gaussian',
            'method': ['Quadrature', 'Stroud3'],
            'cell_type': 'mid-cell',
            'cell_complexity': 'simplecell',
        },
    })
    res = cavs.uq_fm_results['UQ']
    assert res['monopole:freq [MHz]']['expe'][0] > 0
    assert 'stdDev' in res['monopole:R/Q [Ohm]']


def test_rerun_clears_stale_uq_artefacts(project_dir):
    """A monopole rerun must drop the stale UQ perturbation folder so its
    results can't outlive the eigenmode results they were derived from (P2-6)."""
    import os
    cavs, cav = _cavs(project_dir)
    cavs.run_eigenmode({
        'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
        'uq_config': {'variables': ['A'], 'objectives': ['monopole:R/Q [Ohm]'],
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


def test_multicell_wakefield_uq_still_raises_clear_error(project_dir):
    """Multicell UQ is implemented for eigenmode only; wakefield must say so."""
    from cavsim2d.processes.wakefield import run_wakefield_s
    cavs, cav = _cavs(project_dir)
    with pytest.raises(NotImplementedError, match="Multicell UQ"):
        run_wakefield_s({cav.name: cav}, {
            'processes': 1, 'rerun': True,
            'uq_config': {
                'variables': ['A'],
                'objectives': ['monopole:R/Q [Ohm]'],
                'delta': [0.05],
                'processes': 1,
                'method': ['Quadrature', 'Stroud3'],
                'cell_type': 'mid-cell',
                'cell_complexity': 'multicell',
            },
        }, '')


# --- multicell UQ: every half-cell an independent random variable -------------

MC_TESLA = [42, 42, 12, 19, 35, 57.7, 103.353]


def test_half_cell_free_variables_respect_continuity():
    """Shared entries collapse to one random variable each.

    Req is shared by a cell's two halves (n_cells vars); Ri is shared across an
    iris plane (n_cells + 1 vars: two apertures and n_cells - 1 irises); the rest
    are per half-cell (2 * n_cells vars).
    """
    from cavsim2d.utils.shapes import half_cell_free_variables, HALF_CELL_COLS
    for n in (1, 2, 3):
        spec = half_cell_free_variables(n, list(HALF_CELL_COLS))
        counts = {}
        for _, col, _ in spec:
            counts[HALF_CELL_COLS[col]] = counts.get(HALF_CELL_COLS[col], 0) + 1
        for v in ('A', 'B', 'a', 'b', 'L'):
            assert counts[v] == 2 * n, (v, n)
        assert counts['Ri'] == n + 1
        assert counts['Req'] == n

    with pytest.raises(ValueError, match='unknown half-cell variable'):
        half_cell_free_variables(2, ['alpha'])


def test_perturb_half_cells_preserves_continuity(project_dir):
    from cavsim2d import Cavities, EllipticalCavity
    from cavsim2d.utils.shapes import perturb_half_cells, half_cells_to_dataframe
    from cavsim2d.geometry.contours import continuity_violations

    cavs = Cavities(project_dir)
    cav = EllipticalCavity(2, MC_TESLA, MC_TESLA, MC_TESLA, beampipe='both')
    cavs.add_cavity([cav], ['C'])
    cfg = {'uq_config': {'variables': ['Ri', 'Req'], 'method': ['stroud3'],
                         'delta': [0.5, 0.5], 'cell_complexity': 'multicell'}}
    perturbed, weights = perturb_half_cells(cav, cfg)

    assert len(perturbed) == weights.shape[0] > 0
    for hc in perturbed.values():
        assert continuity_violations(hc) == []      # holds by construction
        assert hc[1, 4] == hc[2, 4]                 # the middle iris is one variable
        assert hc[0, 6] == hc[1, 6]                 # cell 1's equator
        assert hc[2, 6] == hc[3, 6]                 # cell 2's equator
    # cells really do vary independently of one another
    assert any(hc[0, 6] != hc[2, 6] for hc in perturbed.values())

    df = half_cells_to_dataframe(perturbed)
    assert df.shape == (len(perturbed), 4 * 7)
    assert list(df.columns[:7]) == ['A1', 'B1', 'a1', 'b1', 'Ri1', 'L1', 'Req1']


def test_half_cell_cavity_never_falls_back_to_gmsh(project_dir):
    """Installing half-cells drops the .geo: a fallback would solve a *different*
    cavity, because write_geometry can only express uniform mid-cells."""
    from cavsim2d import Cavities, EllipticalCavity
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP

    cavs = Cavities(project_dir)
    cav = EllipticalCavity(2, MC_TESLA, MC_TESLA, MC_TESLA, beampipe='both')
    cavs.add_cavity([cav], ['C'])
    hc = cav.half_cells()
    hc[1, 4] = hc[2, 4] = 33.0
    cav.set_half_cells(hc)
    assert cav.geo_filepath is None
    cav.create()
    assert cav.geo_filepath is None
    assert NGSolveMEVP()._build_mesh(cav, 0.02, 3).ne > 0

    original = EllipticalCavity.profile
    EllipticalCavity.profile = lambda self: None
    try:
        with pytest.raises(RuntimeError, match='no .geo file'):
            NGSolveMEVP()._build_mesh(cav, 0.02, 3)
    finally:
        EllipticalCavity.profile = original

    cav.set_half_cells(None)                        # revert restores the gmsh path
    cav.create()
    assert cav.geo_filepath and os.path.exists(cav.geo_filepath)


def test_multicell_uq_runs_end_to_end(project_dir):
    from cavsim2d import Cavities, EllipticalCavity

    cavs = Cavities(project_dir)
    cav = EllipticalCavity(2, MC_TESLA, MC_TESLA, MC_TESLA, beampipe='both')
    cavs.add_cavity([cav], ['C'])
    cavs.run_eigenmode({
        'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
        'uq_config': {'variables': ['Ri', 'Req'],
                      'objectives': ['monopole:freq [MHz]', 'monopole:R/Q [Ohm]'],
                      'method': ['stroud3'], 'delta': [0.5, 0.5],
                      'cell_complexity': 'multicell', 'processes': 1}})

    uq_path = os.path.join(cav.self_dir, 'uq', 'uq.json')
    assert os.path.exists(uq_path)
    with open(uq_path) as fh:
        uq = json.load(fh)
    for qoi in ('monopole:freq [MHz]', 'monopole:R/Q [Ohm]'):
        assert uq[qoi]['stdDev'][0] > 0

    # nodes.csv is the half-cell table: 2*n_cells rows of 7 parameters
    nodes = os.path.join(cav.self_dir, 'uq', 'nodes.csv')
    header = open(nodes).readline().strip().split('\t')
    assert len(header) == 4 * 7


def test_uq_works_for_every_cavity_type(project_dir):
    """Generic spawn() is built on rebuild(), so UQ is no longer elliptical-only.

    It used to die with
    `TypeError: Pillbox.__init__() got an unexpected keyword argument 'name'`
    because base.spawn called `type(self)(name=..., geo_filepath=...)`.
    """
    import numpy as np
    from cavsim2d import Cavities, Pillbox, EllipticalCavityFlatTop, RFGun

    ft = [62.22, 66.13, 30.22, 23.11, 80, 93.5, 171.20, 20]
    gun = {'geometry': {'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
                        'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
                        'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
                        'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}
    cases = [('PB', Pillbox(1, [100, 100, 20, 0, 50], beampipe='both'), 'Req'),
             ('FT', EllipticalCavityFlatTop(1, ft, ft, ft, beampipe='both'), 'A'),
             ('GUN', RFGun(gun), 'R6')]

    for name, cav, var in cases:
        cavs = Cavities(os.path.join(project_dir, name))
        cavs.add_cavity([cav], [name])
        cavs.run_eigenmode({
            'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
            'uq_config': {'variables': [var], 'objectives': ['monopole:freq [MHz]'],
                          'method': ['stroud3'], 'delta': [0.02], 'processes': 1}})
        with open(os.path.join(cav.self_dir, 'uq', 'uq.json')) as fh:
            uq = json.load(fh)
        stats = uq['monopole:freq [MHz]']
        assert stats['expe'][0] > 0
        assert stats['stdDev'][0] > 0, f'{name}: perturbing {var} moved nothing'
