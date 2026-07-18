"""Optimisation smoke test: a tiny 1-generation run produces history and
Pareto-front artefacts via the solver-object API."""
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from cavsim2d import Cavities


def test_optimisation_small_run(project_dir):
    cavs = Cavities(project_dir)
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
    cavs = Cavities(project_dir)
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
    cavs = Cavities(project_dir)
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


def test_robust_optimisation_ranks_by_uq_objective(project_dir):
    """Robust optimisation: a uq_config nested in the eigenmode_config makes every
    candidate a UQ sweep, and the history carries the mean/std/robust columns it
    is ranked by. Guards three fixed bugs in the UQ block (key was the index not a
    column; uq.json is under the candidate's own tuned/uq/ dir; objectives are
    polarisation-prefixed but EIGENMODE_QOIS holds bare names)."""
    from cavsim2d import EllipticalCavity
    cavs = Cavities(project_dir)
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
