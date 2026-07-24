"""UQ tests: simplecell UQ perturbs the mid/end-cell parameter groups; multicell UQ
makes every half-cell an independent random variable, honouring the equator/iris
continuity constraints. Multicell wakefield UQ is still unimplemented."""
import os
import json
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d import Study, EllipticalCavity


def _cavs(project_dir):
    cavs = Study(project_dir)
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
    cavs2 = Study(project_dir)
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
    from cavsim2d import Study, EllipticalCavity
    from cavsim2d.utils.shapes import perturb_half_cells, half_cells_to_dataframe
    from cavsim2d.geometry.contours import continuity_violations

    cavs = Study(project_dir)
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
    from cavsim2d import Study, EllipticalCavity
    from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP

    cavs = Study(project_dir)
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
    from cavsim2d import Study, EllipticalCavity

    cavs = Study(project_dir)
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
    from cavsim2d import Study, Pillbox, EllipticalCavityFlatTop, RFGun

    ft = [62.22, 66.13, 30.22, 23.11, 80, 93.5, 171.20, 20]
    gun = {'geometry': {'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
                        'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
                        'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
                        'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}
    cases = [('PB', Pillbox(1, [100, 100, 20, 0, 50], beampipe='both'), 'Req'),
             ('FT', EllipticalCavityFlatTop(1, ft, ft, ft, beampipe='both'), 'A'),
             ('GUN', RFGun(gun), 'R6')]

    for name, cav, var in cases:
        cavs = Study(os.path.join(project_dir, name))
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


# --- Paper-style (WEPB015) multicell perturbation + Sobol' sensitivity ---------

def test_perturb_half_cells_independent_welds_seams(project_dir):
    """Independent-then-welded multicell perturbation: every half-cell is
    perturbed on its own (continuity violated), then the shared iris/equator DOFs
    are averaged (continuity restored). Averaging two independent draws narrows
    the shared-DOF spread — the WEPB015 Fig. 4 effect."""
    import numpy as np
    from cavsim2d.utils.shapes import (perturb_half_cells_independent,
                                       half_cells_to_dataframe)
    from cavsim2d.geometry.contours import continuity_violations

    cav = EllipticalCavity(2, MC_TESLA, MC_TESLA, MC_TESLA, beampipe='both')
    Study(project_dir).add_cavity([cav], ['C'])
    cfg = {'uq_config': {'variables': ['Ri', 'Req'], 'method': ['normal', 200],
                         'perturbation_mode': ['add', 0.3],
                         'cell_complexity': 'multicell',
                         'independent_half_cells': True}}
    after, before, weights = perturb_half_cells_independent(cav, cfg)

    assert len(after) == len(before) == weights.shape[0] == 200
    # before violates continuity somewhere; after restores it everywhere
    assert any(continuity_violations(hc) for hc in before.values())
    assert all(continuity_violations(hc) == [] for hc in after.values())
    # welding narrows the spread of a shared equator DOF
    b = half_cells_to_dataframe(before)
    a = half_cells_to_dataframe(after)
    assert a['Req1'].std() < b['Req1'].std()


def test_sobol_indices_finds_dominant_variable():
    """The clean Sobol' path (surrogate + Saltelli + SALib): when the output
    depends almost entirely on one input, that input gets the dominant main index
    and the total index is close to it (little interaction)."""
    pytest.importorskip("SALib")
    pytest.importorskip("sklearn")
    import numpy as np
    import pandas as pd
    from cavsim2d.analysis.uq.sobol_sa import sobol_indices

    rng = np.random.default_rng(0)
    X = pd.DataFrame({'a': rng.normal(0, 1, 400),
                      'b': rng.normal(0, 1, 400),
                      'c': rng.normal(0, 1, 400)})
    Y = pd.DataFrame({'obj': 5.0 * X['a'] + 0.05 * X['b'] + 0.05 * X['c']})
    res = sobol_indices(X, Y, N=256)['obj']

    assert max(res, key=lambda v: res[v]['S1']) == 'a'
    assert 0.85 < res['a']['S1'] <= 1.05
    assert abs(res['a']['ST'] - res['a']['S1']) < 0.1   # linear -> S1 ~ ST


def test_surrogate_quality_reported():
    """analyse() reports the surrogate's goodness-of-fit per FM; a clean quadratic
    is represented near-perfectly by the degree-2 surrogate, and the cross-
    validated prediction array lines up with the samples (for the parity plot)."""
    pytest.importorskip("SALib")
    pytest.importorskip("sklearn")
    import numpy as np
    import pandas as pd
    from cavsim2d.analysis.uq.sobol_sa import analyse

    rng = np.random.default_rng(1)
    X = pd.DataFrame({'a': rng.normal(0, 1, 200), 'b': rng.normal(0, 1, 200)})
    Y = pd.DataFrame({'obj': 3.0 * X['a'] + 0.5 * X['b'] ** 2})
    _sobol, surro = analyse(X, Y, N=128)
    s = surro['obj']

    assert s['r2'] > 0.98 and s['cv_r2'] > 0.9        # degree-2 nails a quadratic
    assert len(s['predicted']) == s['n_samples'] == 200
    assert s['max_abs_err'] >= s['rmse'] >= 0.0


def test_surrogate_in_sample_r2_not_below_cv_r2():
    """Regression: the `n_terms` line must NOT re-fit any pipeline step (it once
    called fit_transform on the pipeline's own StandardScaler with a single row,
    corrupting the fitted model so the reported in-sample R^2 (0.47) came out BELOW
    the cross-validated R^2 (0.9999) — impossible). With large-magnitude inputs
    (like Ri~35, Req~103 mm) the scaler matters, so this guards the mutation bug."""
    pytest.importorskip("SALib")
    pytest.importorskip("sklearn")
    import numpy as np
    import pandas as pd
    from cavsim2d.analysis.uq.sobol_sa import analyse

    rng = np.random.default_rng(3)
    n = 80
    Ri = 35 + rng.normal(0, 0.3, n)
    Req = 103 + rng.normal(0, 0.3, n)
    X = pd.DataFrame({'Ri': Ri, 'Req': Req})
    Y = pd.DataFrame({'obj': 5.0 * (Ri - 35) + 2.0 * (Req - 103)
                      + 0.1 * (Ri - 35) ** 2 + rng.normal(0, 1e-3, n)})
    s = analyse(X, Y, N=64)[1]['obj']
    assert s['r2'] >= s['cv_r2'] - 1e-9        # in-sample can't be beaten by CV
    assert s['r2'] > 0.99                       # a clean quadratic fits near-perfectly


def test_surrogate_flags_near_constant_output():
    """A figure of merit with (near-)zero spread — e.g. a TUNED frequency — carries
    no signal, so it is flagged `near_constant` and its low/negative R^2 must not be
    read as a bad surrogate. A genuinely varying FM is not flagged."""
    pytest.importorskip("SALib")
    pytest.importorskip("sklearn")
    import numpy as np
    import pandas as pd
    from cavsim2d.analysis.uq.sobol_sa import analyse

    rng = np.random.default_rng(4)
    n = 60
    a = rng.normal(0, 1, n)
    X = pd.DataFrame({'a': a, 'b': rng.normal(0, 1, n)})
    Y = pd.DataFrame({'freq [MHz]': 1300.0 + rng.normal(0, 0.004, n),   # tuned ~const
                      'R/Q [Ohm]': 200.0 + 5.0 * a})                    # real signal
    surro = analyse(X, Y, N=64)[1]
    assert surro['freq [MHz]']['near_constant'] is True
    assert surro['R/Q [Ohm]']['near_constant'] is False
    assert surro['freq [MHz]']['y_std'] < surro['R/Q [Ohm]']['y_std']


def test_analyse_raises_when_samples_below_surrogate_terms():
    """Too few UQ samples for the surrogate must raise a clear error naming the
    counts, NOT silently write empty sobol.json/surrogate.json (the real WEPB015
    bug: 12 samples, 19 inputs -> 210 degree-2 terms -> every objective skipped)."""
    pytest.importorskip("SALib")
    pytest.importorskip("sklearn")
    import numpy as np
    import pandas as pd
    from cavsim2d.analysis.uq.sobol_sa import analyse

    rng = np.random.default_rng(5)
    n = 12
    X = pd.DataFrame({f'x{i}': rng.normal(0, 1, n) for i in range(19)})
    Y = pd.DataFrame({'obj': rng.normal(0, 1, n)})
    with pytest.raises(ValueError, match=r"too few"):
        analyse(X, Y, N=64)


def test_sa_plots_return_axes_not_figure(tmp_path):
    """Regression: `plot_surrogate_quality`/`plot_sobol_indices` must return an
    Axes/array-of-Axes, never a bare matplotlib Figure. A cell whose last expression
    is a Figure renders NOTHING under the ipympl (`%matplotlib widget`) backend, so
    these two silently showed nothing while every other plot (returning axes) worked."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure

    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cav.set_workspace(os.path.join(tmp_path, 'c'))
    os.makedirs(cav.uq_dir, exist_ok=True)
    fm = 'monopole:freq [MHz]'
    with open(os.path.join(cav.uq_dir, 'surrogate.json'), 'w') as f:
        json.dump({fm: {'r2': 0.99, 'cv_r2': 0.98, 'rmse': 0.1, 'max_abs_err': 0.2,
                        'y_mean': 1300.0, 'y_std': 0.5, 'near_constant': False,
                        'n_samples': 30, 'n_terms': 6,
                        'actual': [1.0, 2.0, 3.0], 'predicted': [1.1, 2.0, 2.9]}}, f)
    with open(os.path.join(cav.uq_dir, 'sobol.json'), 'w') as f:
        json.dump({fm: {'Ri': {'S1': 0.7, 'ST': 0.8}, 'Req': {'S1': 0.2, 'ST': 0.3}}}, f)

    for ret in (cav.eigenmode.plot_surrogate_quality(show=False),
                cav.eigenmode.plot_sobol_indices(show=False)):
        assert ret is not None
        assert not isinstance(ret, Figure)
        # array/list of Axes whose parent figure carries the drawn panels
        first = list(ret)[0]
        assert first.figure is not None and len(first.figure.axes) >= 1


def test_independent_inputs_drops_constant_and_welded_duplicates():
    """Edge case: independent_inputs must drop (a) constant columns — not random
    inputs — and (b) near-duplicate columns that welding produces when two
    half-cells share an iris/equator DOF. Only genuinely independent inputs survive
    (the Saltelli design needs them independent)."""
    pytest.importorskip("SALib")
    import numpy as np
    import pandas as pd
    from cavsim2d.analysis.uq.sobol_sa import independent_inputs

    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 50)
    X = pd.DataFrame({
        'a': a,
        'a_weld': a + 1e-12,          # welded duplicate of 'a'
        'const': np.full(50, 3.14),   # not a random input
        'b': rng.normal(0, 1, 50),
    })
    keep = independent_inputs(X)
    assert 'a' in keep and 'b' in keep
    assert 'a_weld' not in keep and 'const' not in keep
    assert len(keep) == 2


def test_analyse_raises_when_no_varying_inputs():
    """Edge case: if every input column is constant there is nothing to do SA on —
    analyse must raise a clear error rather than build a degenerate design."""
    pytest.importorskip("SALib")
    pytest.importorskip("sklearn")
    import numpy as np
    import pandas as pd
    from cavsim2d.analysis.uq.sobol_sa import analyse

    X = pd.DataFrame({'a': np.full(30, 1.0), 'b': np.full(30, 2.0)})
    Y = pd.DataFrame({'obj': np.arange(30.0)})
    with pytest.raises(ValueError, match=r"[Nn]o varying"):
        analyse(X, Y, N=64)
