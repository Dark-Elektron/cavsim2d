"""Optimisation smoke test: a tiny 1-generation run produces history and
Pareto-front artefacts via the solver-object API."""
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from cavsim2d import Study


def test_optimisation_small_run(project_dir):
    cavs = Study(project_dir)
    # Bounds bracket the known-good TESLA-like mid-cell so the inner tune
    # converges; mutation/crossover factors are integer offspring counts.
    config = {
        'initial_points': 2,
        'no_of_generation': 1,
        'method': {'LHS': {'seed': 5}},
        'bounds': {
            'A': [58, 66], 'B': [62, 70], 'a': [26, 34], 'b': [20, 27],
            'Ri': [76, 84], 'L': [90, 97], 'Req': [165, 178],
        },
        'objectives': [['min', 'monopole:freq [MHz]'], ['min', 'monopole:Epk/Eacc []']],
        'tune_config': {
            'freqs': 801.58,
            'cell_type': {'mid-cell': 'Req'},
            'processes': 1,
            'eigenmode_config': {'n_cells': 1, 'processes': 1,
                                 'boundary_conditions': 'mm'},
        },
        'mutation_factor': 2,
        'crossover_factor': 2,
        'elites_for_crossover': 2,
        'chaos_factor': 2,
        'weights': [1, 1],
    }
    cavs.run_optimisation(config)

    assert cavs.optimisation.folder.exists()
    # a run must record at least the evaluated candidates
    assert not cavs.optimisation.history.empty


def test_optimisation_generalises_to_pillbox(project_dir):
    """A pillbox optimises via its own tune variables (Ri/L swept, Req tuned) —
    the template is the added cavity, not a fabricated elliptical."""
    from cavsim2d import Pillbox
    cavs = Study(project_dir)
    cavs.add_cavity([Pillbox(1, [100, 100, 22, 0, 0], beampipe='none')], ['PB'])
    config = {
        'initial_points': 2,
        'no_of_generation': 1,
        'method': {'LHS': {'seed': 5}},
        'bounds': {'Ri': [18, 26], 'L': [90, 110]},   # pillbox params, not A/B/a/b
        'objectives': [['min', 'monopole:freq [MHz]'], ['min', 'monopole:R/Q [Ohm]']],
        'tune_config': {
            'freqs': 1300.0,
            'cell_type': {'mid-cell': 'Req'},          # pillbox has Req (no suffix)
            'processes': 1,
            'eigenmode_config': {'n_cells': 1, 'processes': 1, 'boundary_conditions': 'mm'},
        },
        'mutation_factor': 2, 'crossover_factor': 2, 'elites_for_crossover': 2,
        'chaos_factor': 2, 'weights': [1, 1],
    }
    cavs.run_optimisation(config)
    assert not cavs.optimisation.history.empty


def test_optimisation_generalises_to_spline(project_dir):
    """The decisive non-elliptical case: a spline is optimised over control-point
    coordinates ('p2_r'/'p4_r' swept, 'p3_r' tuned) that DO NOT exist on an
    elliptical cavity. If the optimiser fabricated an elliptical template (the
    old bug) instead of using the added cavity, `_resolve_suffixed_var('p3_r',
    ...)` would raise 'Unknown tune variable' — so this run passing proves the
    template is the added model. Also exercises the spline's coordinate-aware
    `get_tune_value` ('p3_r' -> control point 'p3'[r], not a flat parameters key)."""
    from cavsim2d import SplineCavity
    geom = {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
            'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}
    cavs = Study(project_dir)
    cavs.add_cavity([SplineCavity({'geometry': dict(geom)}, kind='Bezier')], ['SP'])
    config = {
        'initial_points': 2,
        'no_of_generation': 1,
        'method': {'LHS': {'seed': 3}},
        'bounds': {'p2_r': [98, 108], 'p4_r': [66, 74]},   # control coords, not on elliptical
        'objectives': [['min', 'monopole:freq [MHz]'], ['max', 'monopole:R/Q [Ohm]']],
        'tune_config': {
            'freqs': 1300.0,
            'cell_type': {'mid-cell': 'p3_r'},             # tune a control-point coordinate
            'processes': 1,
            'eigenmode_config': {'n_cells': 1, 'processes': 1, 'boundary_conditions': 'mm'},
        },
        'mutation_factor': 2, 'crossover_factor': 2, 'elites_for_crossover': 2,
        'chaos_factor': 2, 'weights': [1, 1],
    }
    cavs.run_optimisation(config)
    hist = cavs.optimisation.history
    assert not hist.empty
    assert 'p3_r' in hist.columns          # the tuned control-point coordinate was recorded


def test_end_cell_alias_ties_both_ends(tmp_path):
    """The '_e' end-cell alias drives BOTH end cells from a single variable, kept
    tied. As a bounds variable 'A_e' fans out to A_el AND A_er (same value); as a
    tune variable 'L_e' resolves to L_el (the tuner mirrors it into L_er). Fast:
    resolution + spawn only, no solve."""
    import numpy as np
    import pandas as pd
    from cavsim2d import EllipticalCavity
    from cavsim2d.processes.tune import _resolve_suffixed_var
    from cavsim2d.analysis.optimisation import _resolve_opt_columns

    mid = [42, 42, 12, 19, 35, 57.7, 103.3]
    cav = EllipticalCavity(3, mid, mid, mid, beampipe='both')

    # bounds resolution: alias fans out to both ends; bare/suffixed unchanged
    assert _resolve_opt_columns('A_e', 'end-cell', cav) == ['A_el', 'A_er']
    assert _resolve_opt_columns('A', 'mid-cell', cav) == ['A_m']
    assert _resolve_opt_columns('A_el', 'end-cell', cav) == ['A_el']
    # tune-variable resolution: L_e -> L_el (right end tied by the post-stage mirror)
    assert _resolve_suffixed_var('L_e', 'end-cell', cav) == 'L_el'
    # an unknown bare name is still rejected
    with pytest.raises(ValueError):
        _resolve_opt_columns('Z_e', 'end-cell', cav)

    # a spawned candidate driven by 'A_e' has both ends equal, mid untouched
    df = pd.DataFrame({'A_e': [45.0]}, index=['C0'])
    df_spawn = df.copy()
    for t in _resolve_opt_columns('A_e', 'end-cell', cav):
        df_spawn[t] = df['A_e']
    df_spawn.drop(columns=['A_e'], inplace=True)
    scav = cav.spawn(df_spawn, str(tmp_path)).cavities_dict['C0']
    assert np.isclose(scav.parameters['A_el'], scav.parameters['A_er'])
    assert np.isclose(scav.parameters['A_el'], 45.0)
    assert np.isclose(scav.parameters['A_m'], 42.0)      # mid cell left alone


def test_whole_cavity_two_stage_tune(project_dir):
    """A whole-cavity optimisation varies mid AND end shape and, with no cell_type
    given, tunes the mid cell (Req) and the end cells (L) to the frequency per
    candidate — the standard field-flat recipe. Guards the lifted single-pair
    restriction: both a mid-cell and an end-cell tune stage must run."""
    import json
    from pathlib import Path
    from cavsim2d import EllipticalCavity
    cavs = Study(project_dir)
    mid = [42, 42, 12, 19, 35, 57.7, 103.3]
    cavs.add_cavity([EllipticalCavity(3, mid, mid, mid, beampipe='both')], ['C'])
    config = {
        'initial_points': 2, 'no_of_generation': 1, 'method': {'LHS': {'seed': 5}},
        'bounds': {'A_m': [40, 44], 'A_e': [40, 44]},   # mid AND end shape, both ends tied
        'objectives': [['equal', 'monopole:R/Q [Ohm]', 330.0]],
        # NOTE: no 'cell_type' -> defaults to {'mid-cell':'Req','end-cell':'L'}
        'tune_config': {'freqs': 1300.0, 'processes': 1, 'tol': 2e-2,
                        'eigenmode_config': {'n_cells': 3, 'processes': 1,
                                             'boundary_conditions': 'mm'}},
        'mutation_factor': 2, 'crossover_factor': 2, 'elites_for_crossover': 2,
        'chaos_factor': 2, 'weights': [1],
    }
    cavs.run_optimisation(config)
    assert not cavs.optimisation.history.empty

    # a candidate was tuned in BOTH a mid-cell and an end-cell stage
    res = list(Path(cavs.optimisation.candidates_folder).glob('*/tuned/tune_info/tune_res.json'))
    assert res, 'no tuned candidate produced'
    stages = json.load(open(res[0]))
    assert 'mid-cell' in stages and 'end-cell' in stages


def test_robust_optimisation_ranks_by_uq_objective(project_dir):
    """Robust optimisation: a uq_config nested in the eigenmode_config makes every
    candidate a UQ sweep, and the history carries the mean/std/robust columns it
    is ranked by. Guards three fixed bugs in the UQ block (key was the index not a
    column; uq.json is under the candidate's own tuned/uq/ dir; objectives are
    polarisation-prefixed but EIGENMODE_QOIS holds bare names)."""
    from cavsim2d import EllipticalCavity
    cavs = Study(project_dir)
    mid = [42, 42, 12, 19, 35, 57.7, 103.353]
    cavs.add_cavity([EllipticalCavity(1, mid, mid, mid, beampipe='none')], ['TESLA'])
    config = {
        'initial_points': 2, 'no_of_generation': 1, 'method': {'LHS': {'seed': 5}},
        'bounds': {'A': [40, 46], 'B': [40, 46]},
        'objectives': [['min', 'monopole:Epk/Eacc []'],
                       ['min', 'monopole:Bpk/Eacc [mT/MV/m]']],
        'tune_config': {
            'freqs': 1300.0, 'cell_type': {'mid-cell': 'Req'}, 'processes': 1,
            'eigenmode_config': {
                'n_cells': 1, 'processes': 1, 'boundary_conditions': 'mm',
                'uq_config': {
                    'variables': ['A', 'B'],
                    'objectives': ['monopole:Epk/Eacc []', 'monopole:Bpk/Eacc [mT/MV/m]'],
                    'method': ['stroud3'], 'delta': [0.02, 0.02], 'processes': 1,
                    'cell_type': 'mid-cell', 'cell_complexity': 'simplecell'}}},
        'mutation_factor': 2, 'crossover_factor': 2, 'elites_for_crossover': 2,
        'chaos_factor': 2, 'weights': [1, 1]}
    cavs.run_optimisation(config)
    hist = cavs.optimisation.history
    assert not hist.empty
    # the UQ mean/std columns are present (empty -> the UQ block silently produced
    # nothing, the exact failure the fixes address)
    assert any(c.startswith('E[') for c in hist.columns)
    assert any(c.startswith('std[') for c in hist.columns)
