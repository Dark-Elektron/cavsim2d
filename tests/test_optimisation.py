"""Optimisation smoke test: a tiny 1-generation run produces history and
Pareto-front artefacts via the solver-object API."""
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from cavsim2d.cavity import Cavities


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
